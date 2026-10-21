"""Writer-preference memory on the vendored mem0 (vendor/mem0). JSON-file fallback so the agent always runs.

mem0's LLM and embedder follow the highest-priority chat/embed providers in the admin menu's tiers (embeddings never fail over:
different models give incompatible vectors); the memory object is
rebuilt automatically when those change. Memories live in a Qdrant collection named after the embedder, so
switching to an embedding model with different dimensions never corrupts an existing collection.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import httpx

from clients import headers
from providers import Registry
from settings import Settings

os.environ.setdefault("MEM0_TELEMETRY", "False")  # a private writing tool should not phone home


class FileMemory:
    """Keyword-overlap recall over a JSON list. Deliberately simple; mem0 is the real backend."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def add(self, text: str, kind: str = "note") -> None:
        with self._lock:
            items = self._load()
            if any(i["text"] == text for i in items):
                return
            items.append({"text": text, "kind": kind})
            self.path.write_text(json.dumps(items, indent=1))

    def search(self, query: str, limit: int = 5) -> list[str]:
        words = set(re.findall(r"\w{4,}", query.lower()))
        scored = []
        for it in self._load():
            overlap = len(words & set(re.findall(r"\w{4,}", it["text"].lower())))
            score = overlap + (1 if it["kind"] == "preference" else 0)  # preferences are always mildly relevant
            if score:
                scored.append((score, it["text"]))
        return [t for _, t in sorted(scored, reverse=True)[:limit]]


class Mem0Memory:
    def __init__(self, s: Settings, reg: Registry):
        from mem0 import Memory  # vendored; imported lazily, it is heavy

        chat, emb = reg.get("chat"), reg.get("embed")
        if not (chat and emb):
            raise RuntimeError("chat and embed providers are required")
        # Probe the embedder once for its vector size (Qdrant needs it up front).
        r = httpx.post(f"{emb.base_url}/v1/embeddings", headers=headers(emb), json={"model": emb.model or "embed", "input": "dim"}, timeout=30)
        r.raise_for_status()
        dims = len(r.json()["data"][0]["embedding"])
        tag = hashlib.sha1(f"{emb.base_url}|{emb.model}|{dims}".encode()).hexdigest()[:8]
        cfg = {
            "llm": {"provider": "openai", "config": {"model": chat.model or "default", "openai_base_url": f"{chat.base_url}/v1", "api_key": chat.api_key or "none"}},
            "embedder": {"provider": "openai", "config": {"model": emb.model or "embed", "openai_base_url": f"{emb.base_url}/v1", "api_key": emb.api_key or "none", "embedding_dims": dims}},
            "vector_store": {"provider": "qdrant", "config": {"collection_name": f"bookscribe_{tag}", "path": str(s.state_dir / "mem0_qdrant"), "on_disk": True, "embedding_model_dims": dims}},
            "history_db_path": str(s.state_dir / "mem0_history.db"),
        }
        self.m = Memory.from_config(cfg)
        self.uid = s.user_id

    def add(self, text: str, kind: str = "note") -> None:
        # infer=False stores the writer's words verbatim instead of paying for an LLM extraction per message
        self.m.add([{"role": "user", "content": text}], user_id=self.uid, metadata={"kind": kind}, infer=False)

    def search(self, query: str, limit: int = 5) -> list[str]:
        res = self.m.search(query, top_k=limit, filters={"user_id": self.uid})
        rows = res.get("results", []) if isinstance(res, dict) else res
        return [r["memory"] for r in rows]


class MemoryManager:
    """What the agent holds. Delegates to mem0 when possible, rebuilding when providers change."""

    def __init__(self, s: Settings, reg: Registry, force_file: bool = False):
        self.s, self.reg, self.force_file = s, reg, force_file
        self.file = FileMemory(s.state_dir / "memory.json")
        self._impl, self._built_for, self.backend, self.error = None, -1, "file", ""

    def _current(self):
        if self.force_file or self.s.mem_backend != "mem0":
            return self.file
        if self._built_for != self.reg.version:
            self._built_for = self.reg.version
            try:
                self._impl, self.backend, self.error = Mem0Memory(self.s, self.reg), "mem0", ""
            except Exception as e:  # embedder unreachable, bad key, ...
                self._impl, self.backend, self.error = None, "file", f"{e.__class__.__name__}: {str(e)[:200]}"
        return self._impl or self.file

    def add(self, text: str, kind: str = "note") -> None:
        impl = self._current()
        try:
            impl.add(text, kind)
        except Exception:
            self.file.add(text, kind)

    def search(self, query: str, limit: int = 5) -> list[str]:
        impl = self._current()
        try:
            return impl.search(query, limit)
        except Exception:
            return self.file.search(query, limit)
