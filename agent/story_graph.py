"""Story graph via graphify (vendor/graphify): *how things connect* across bible + chapters.

graphify runs headless (`graphify extract`) against the book folder using the active **graph** provider from
the admin menu (any OpenAI-compatible endpoint, local or cloud). It is slow and costs LLM calls, so it never
runs on the voice path: saves schedule a debounced background rebuild, and the agent reads whatever
`graphify-out/graph.json` exists. Edges are tagged EXTRACTED or INFERRED by graphify; the prompt shows
that tag so inferred links are never presented as canon.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from book_store import BookStore
from providers import Registry


class StoryGraph:
    def __init__(self, store: BookStore, reg: Registry, debounce: float = 20.0, timeout: float = 1800.0, command: list[str] | None = None):
        self.store, self.reg, self.debounce, self.timeout = store, reg, debounce, timeout
        self.command = command or [sys.executable, "-m", "graphify", "extract", ".", "--backend", "openai", "--no-gitignore"]
        self.status: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, tuple[float, dict]] = {}

    # --- building --------------------------------------------------------
    def schedule(self, book: str) -> None:
        """Debounced background rebuild; call after every saved change."""
        old = self._tasks.get(book)
        if old and not old.done() and self.status.get(book, {}).get("state") != "running":
            old.cancel()
        if old and not old.done() and self.status.get(book, {}).get("state") == "running":
            self.status[book]["dirty"] = True  # rebuild again once the running one finishes
            return
        self._tasks[book] = asyncio.create_task(self._later(book))

    def schedule_now(self, book: str) -> None:
        if self.status.get(book, {}).get("state") == "running":
            return
        old = self._tasks.get(book)
        if old and not old.done():
            old.cancel()
        self._tasks[book] = asyncio.create_task(self.rebuild(book))
        self.status[book] = {"state": "running", "started": time.time(), "dirty": False}

    async def _later(self, book: str) -> None:
        await asyncio.sleep(self.debounce)
        await self.rebuild(book)

    async def rebuild(self, book: str) -> dict:
        lock = self._locks.setdefault(book, asyncio.Lock())
        if lock.locked():
            return self.status[book]
        async with lock:
            p = self.reg.get("graph")
            if not p:
                self.status[book] = {"state": "disabled", "detail": "No graph provider configured."}
                return self.status[book]
            self.status[book] = st = {"state": "running", "started": time.time(), "dirty": False}
            env = {**os.environ, "OPENAI_BASE_URL": f"{p.base_url}/v1", "OPENAI_API_KEY": p.api_key or "none",
                   "GRAPHIFY_OPENAI_MODEL": p.model or "default", "GRAPHIFY_BACKEND": "openai"}
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self.command, cwd=self.store.book_dir(book), env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await asyncio.wait_for(proc.communicate(), self.timeout)
                tail = out.decode(errors="replace")[-600:]
                ok = proc.returncode == 0 and self._path(book).exists()
                st.update(state="ok" if ok else "error", detail=tail, finished=time.time())
                if ok:
                    g = self.load(book)
                    st.update(nodes=len(g["nodes"]), edges=len(g["links"]))
            except Exception as e:  # missing graphify, timeout, ...
                st.update(state="error", detail=f"{e.__class__.__name__}: {e}", finished=time.time())
            if st.pop("dirty", False):
                self.schedule(book)
            return st

    # --- reading ----------------------------------------------------------
    def _path(self, book: str) -> Path:
        return self.store.book_dir(book) / "graphify-out" / "graph.json"

    def load(self, book: str) -> dict:
        p = self._path(book)
        if not p.exists():
            return {"nodes": [], "links": []}
        mtime = p.stat().st_mtime
        hit = self._cache.get(book)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            g = json.loads(p.read_text())
        except (OSError, ValueError):
            return {"nodes": [], "links": []}
        g.setdefault("nodes", []); g.setdefault("links", g.pop("edges", []))
        self._cache[book] = (mtime, g)
        return g

    def context_for(self, book: str, text: str, limit: int = 12) -> str:
        """Relationship lines for nodes whose label appears in `text`."""
        g = self.load(book)
        label = {n["id"]: n.get("label", n["id"]) for n in g["nodes"]}
        named = {i for i, l in label.items() if len(l) > 2 and re.search(rf"\b{re.escape(l)}\b", text, re.I)}
        lines = []
        for e in g["links"]:
            if e["source"] in named or e["target"] in named:
                tag = e.get("confidence", "INFERRED")
                lines.append(f"{label.get(e['source'], e['source'])} —{e.get('relation', 'related to')}→ {label.get(e['target'], e['target'])} [{tag}]")
        return "\n".join(lines[:limit])

    def info(self, book: str) -> dict:
        g = self.load(book)
        return {**self.status.get(book, {"state": "idle"}), "nodes": len(g["nodes"]), "edges": len(g["links"]), "built": self._path(book).exists()}
