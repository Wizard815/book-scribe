"""Paragraph embedding index: meaning-based search over the manuscript.

The locator (kg.py) matches *words*, so "the part where the father is dismissive" finds nothing when the text never says
"dismissive". This index stores one vector per paragraph, made by the **embedding provider** from the admin menu, and
answers "which paragraphs mean something like this?".

* Only paragraphs that are new or changed are embedded (a hash of the text is stored with each vector).
* The index remembers which provider/model made it. Vectors from different models are not comparable, so after a change
  the index counts as empty until it is rebuilt (admin panel, "Rebuild").
* Searching costs one embedding call (the query) and a dot product; if the provider is down the agent simply falls back
  to word matching.
* Embeddings never fail over between tiers (see router.py): one model per index.

Files: <state>/embeddings/<book>.json
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from array import array
from pathlib import Path

try:  # numpy ships with the vendored graphify; without it search still works, just slower
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from book_store import BookStore
from providers import Profile, Registry
from router import NotConfigured, Router

BATCH = 24
MAX_CHARS = 2000


def _h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _pack(vec: list[float]) -> str:
    return base64.b64encode(array("f", vec).tobytes()).decode()


def _unpack(s: str):
    raw = base64.b64decode(s)
    if np is not None:
        return np.frombuffer(raw, dtype=np.float32)
    a = array("f"); a.frombytes(raw); return list(a)


class ParagraphIndex:
    def __init__(self, store: BookStore, reg: Registry, router: Router, http, state_dir: Path, debounce: float = 15.0):
        self.store, self.reg, self.router, self.http, self.debounce = store, reg, router, http, debounce
        self.dir = state_dir / "embeddings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cache: dict[str, tuple] = {}

    # --- provider ---------------------------------------------------------------------------------------------
    def _identity(self) -> dict | None:
        p = self.reg.get("embed")
        return {"provider": p.id, "base_url": p.base_url, "model": p.model or "embed"} if p else None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async def go(p: Profile) -> list[list[float]]:
            from clients import headers  # local: clients imports router, keep import order simple
            r = await self.http.post(f"{p.base_url}/v1/embeddings", headers=headers(p), timeout=60,
                                     json={"model": p.model or "embed", "input": [t[:MAX_CHARS] for t in texts]})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            if len(data) != len(texts):
                raise ValueError("embedding server returned the wrong number of vectors")
            return [d["embedding"] for d in data]

        return await self.router.call("embed", go)

    # --- storage ----------------------------------------------------------------------------------------------
    def _path(self, book: str) -> Path:
        self.store.book_dir(book)  # validates the name
        return self.dir / f"{book}.json"

    def _load(self, book: str) -> dict:
        p = self._path(book)
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except (OSError, ValueError):
            return {}

    def _save(self, book: str, data: dict) -> None:
        p = self._path(book)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
        self._cache.pop(book, None)

    def _matches_provider(self, data: dict) -> bool:
        ident = self._identity()
        return bool(ident and data.get("identity") == ident)

    # --- status ------------------------------------------------------------------------------------------------
    def status(self, book: str) -> dict:
        data = self._load(book)
        paras = self.store.all_paragraphs(book)
        items = data.get("items", {})
        same = self._matches_provider(data)
        fresh = sum(1 for p in paras if items.get(p.id, {}).get("h") == _h(p.text)) if same else 0
        prog = self.progress.get(book, {})
        ident = self._identity()
        return {
            "configured": ident is not None, "provider": ident, "same_provider": same, "paragraphs": len(paras), "indexed": fresh,
            "stale": len(paras) - fresh, "dims": data.get("dims"), "built": data.get("built"),
            "state": prog.get("state", "idle"), "done": prog.get("done", 0), "total": prog.get("total", 0), "error": prog.get("error", ""),
            "ready": bool(ident and same and fresh > 0),
        }

    # --- building ----------------------------------------------------------------------------------------------
    def schedule(self, book: str) -> None:
        """Debounced background update after the manuscript changed."""
        old = self._tasks.get(book)
        if old and not old.done() and self.progress.get(book, {}).get("state") != "running":
            old.cancel()
        if old and not old.done() and self.progress.get(book, {}).get("state") == "running":
            self.progress[book]["again"] = True
            return
        self._tasks[book] = asyncio.create_task(self._later(book))

    async def _later(self, book: str) -> None:
        await asyncio.sleep(self.debounce)
        await self.reindex(book)

    def start(self, book: str, full: bool = False) -> None:
        """Begin now in the background (the admin's Rebuild button)."""
        if self.progress.get(book, {}).get("state") == "running":
            return
        self.progress[book] = {"state": "running", "done": 0, "total": 0}
        self._tasks[book] = asyncio.create_task(self.reindex(book, full))

    async def reindex(self, book: str, full: bool = False) -> dict:
        lock = self._locks.setdefault(book, asyncio.Lock())
        if lock.locked():
            return self.status(book)
        async with lock:
            ident = self._identity()
            if not ident:
                self.progress[book] = {"state": "disabled", "error": "No embeddings provider is in any tier."}
                return self.status(book)
            data = self._load(book)
            if full or not self._matches_provider(data):
                data = {"identity": ident, "items": {}, "dims": None}
            items = data["items"]
            paras = self.store.all_paragraphs(book)
            todo = [p for p in paras if items.get(p.id, {}).get("h") != _h(p.text)]
            self.progress[book] = {"state": "running", "done": 0, "total": len(todo)}
            try:
                for i in range(0, len(todo), BATCH):
                    chunk = todo[i : i + BATCH]
                    vecs = await self.embed([p.text for p in chunk])
                    for p, v in zip(chunk, vecs):
                        if data["dims"] and len(v) != data["dims"]:
                            raise ValueError("the embedding model changed size mid-build; rebuild from scratch")
                        data["dims"] = len(v)
                        items[p.id] = {"h": _h(p.text), "v": _pack(v)}
                    self.progress[book]["done"] = i + len(chunk)
                keep = {p.id for p in paras}
                data["items"] = {k: v for k, v in items.items() if k in keep}
                data["built"] = int(time.time())
                self._save(book, data)
                self.progress[book].update(state="ok", error="")
            except Exception as e:
                self.progress[book].update(state="error", error=f"{e.__class__.__name__}: {str(e)[:200]}")
            again = self.progress[book].pop("again", False)
        if again:
            self.schedule(book)
        return self.status(book)

    # --- searching ---------------------------------------------------------------------------------------------
    def _matrix(self, book: str):
        data = self._load(book)
        if not self._matches_provider(data) or not data.get("items"):
            return None
        mt = self._path(book).stat().st_mtime
        hit = self._cache.get(book)
        if hit and hit[0] == mt:
            return hit[1], hit[2]
        ids = list(data["items"])
        vecs = [_unpack(data["items"][i]["v"]) for i in ids]
        if np is not None:
            m = np.vstack(vecs).astype(np.float32)
            m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        else:  # pragma: no cover
            m = [[x / (sum(y * y for y in v) ** 0.5 or 1) for x in v] for v in vecs]
        self._cache[book] = (mt, ids, m)
        return ids, m

    async def search(self, book: str, query: str, k: int = 5, chapter_of=None) -> list[tuple[str, float]]:
        """Paragraph ids most similar in meaning to `query`, best first. [] when there is no usable index or the provider fails."""
        mat = self._matrix(book)
        if not mat or not query.strip():
            return []
        ids, m = mat
        try:
            (q,) = await self.embed([query])
        except (NotConfigured, Exception):
            return []
        if np is not None:
            qv = np.asarray(q, dtype=np.float32)
            if qv.shape[0] != m.shape[1]:
                return []
            qv /= max(float(np.linalg.norm(qv)), 1e-9)
            scores = m @ qv
            order = np.argsort(-scores)
            out = [(ids[i], float(scores[i])) for i in order if chapter_of is None or chapter_of(ids[i])]
        else:  # pragma: no cover
            n = sum(x * x for x in q) ** 0.5 or 1
            sc = [sum(a * b / n for a, b in zip(row, q)) for row in m]
            out = sorted(((ids[i], sc[i]) for i in range(len(ids)) if chapter_of is None or chapter_of(ids[i])), key=lambda t: -t[1])
        return out[:k]
