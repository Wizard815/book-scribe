"""Writer-preference memory. mem0 when available, JSON-file fallback so the agent always runs."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from settings import Settings


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
            # preferences are always relevant; give them a floor
            score = overlap + (1 if it["kind"] == "preference" else 0)
            if score:
                scored.append((score, it["text"]))
        return [t for _, t in sorted(scored, reverse=True)[:limit]]


class Mem0Memory:
    def __init__(self, s: Settings):
        from mem0 import Memory  # imported lazily: heavy, optional

        cfg = {
            "llm": {"provider": "openai", "config": {"model": s.llm_model, "openai_base_url": f"{s.llama_swap_url}/v1", "api_key": "none"}},
            "embedder": {"provider": "openai", "config": {"model": s.embed_model, "openai_base_url": f"{s.llama_swap_url}/v1", "api_key": "none"}},
            "vector_store": {"provider": "qdrant", "config": {"path": str(s.state_dir / "mem0_qdrant"), "on_disk": True}},
        }
        self.m = Memory.from_config(cfg)
        self.uid = s.user_id

    def add(self, text: str, kind: str = "note") -> None:
        self.m.add([{"role": "user", "content": text}], user_id=self.uid, metadata={"kind": kind})

    def search(self, query: str, limit: int = 5) -> list[str]:
        res = self.m.search(query, user_id=self.uid, limit=limit)
        rows = res.get("results", []) if isinstance(res, dict) else res
        return [r["memory"] for r in rows]


def make_memory(s: Settings):
    """Returns (memory, backend_name)."""
    if s.mem_backend == "mem0":
        try:
            return Mem0Memory(s), "mem0"
        except Exception as e:  # missing package, unreachable embedder, ...
            print(f"[memory] mem0 unavailable ({e!r}); using file fallback")
    return FileMemory(s.state_dir / "memory.json"), "file"
