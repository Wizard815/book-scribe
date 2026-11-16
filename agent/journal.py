"""The journal: everything said in the one long conversation, kept for good.

The writer talks to a single agent that never needs a "new chat". To make that work the agent's working memory is kept small and tidy
(a rolling summary plus the latest turns), while *every* turn is written here, one line of JSON each, one file per day:

    <book>/journal/2026-11-16.jsonl      {"at": 1794762418, "role": "user"|"assistant"|"note", "text": "...", "via": "voice"|"typed"}
    <book>/journal/summary.json          the rolling summary of what has been folded out of working memory
    <book>/journal/screen.json           when the writer last cleared the screen (only the chat window starts fresh; memory does not)

Voice is stored as the text the speech model heard, not as audio: a transcript is a few kilobytes an hour, audio is hundreds of
megabytes, and only the words can be searched. The journal is cut into chunks that are embedded (embeddings.py), so an idea or a
description said months ago can be found again by meaning ("that thing I said about the old lighthouse").

Only the writer's own endpoints read it; the admin page cannot.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK_CHARS = 800


class Journal:
    def __init__(self, store):
        self.store = store

    def dir(self, book: str) -> Path:
        d = self.store.book_dir(book) / "journal"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- writing -------------------------------------------------------------------------------------------------
    def append(self, book: str, role: str, text: str, via: str = "", at: float | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        at = at or time.time()
        day = datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%d")
        rec = {"at": int(at), "role": role, "text": text[:6000]}
        if via:
            rec["via"] = via
        with open(self.dir(book) / f"{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- reading -------------------------------------------------------------------------------------------------
    def _files(self, book: str) -> list[Path]:
        return sorted(self.dir(book).glob("????-??-??.jsonl"))

    def _read(self, path: Path) -> list[dict]:
        out = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if isinstance(r, dict) and isinstance(r.get("text"), str) and "at" in r:
                        out.append(r)
                except ValueError:
                    continue  # a half-written line after a crash: skip it, keep the rest
        except OSError:
            pass
        return out

    def recent(self, book: str, n: int = 40, since_screen: bool = True) -> list[dict]:
        """The last n turns (newest last). With since_screen, only turns after the writer last cleared the screen."""
        floor = self.screen_from(book) if since_screen else 0
        out: list[dict] = []
        for path in reversed(self._files(book)):
            rows = [r for r in self._read(path) if r["at"] >= floor]
            out = rows + out
            if len(out) >= n:
                break
        return out[-n:]

    def count(self, book: str) -> int:
        return sum(len(self._read(p)) for p in self._files(book))

    def days(self, book: str) -> int:
        return len(self._files(book))

    # --- the screen and the summary ----------------------------------------------------------------------------------
    def _state(self, book: str, name: str) -> dict:
        try:
            return json.loads((self.dir(book) / name).read_text())
        except (OSError, ValueError):
            return {}

    def _put(self, book: str, name: str, data: dict) -> None:
        p = self.dir(book) / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(p)

    def screen_from(self, book: str) -> int:
        return int(self._state(book, "screen.json").get("from", 0))

    def clear_screen(self, book: str) -> None:
        self._put(book, "screen.json", {"from": int(time.time()) + 1})

    def summary(self, book: str) -> str:
        return str(self._state(book, "summary.json").get("text", ""))

    def set_summary(self, book: str, text: str) -> None:
        self._put(book, "summary.json", {"text": text.strip()[:3000], "at": int(time.time())})

    # --- for the embedding index and for keyword search ---------------------------------------------------------------
    def chunks(self, book: str) -> dict[str, str]:
        """key -> text. A chunk is a stretch of consecutive turns of up to ~800 characters, labelled with its date, so a search hit tells
        the model when it was said. Keys are stable for finished days (mem:<date>:<n>)."""
        docs: dict[str, str] = {}
        for path in self._files(book):
            day = path.stem
            cur, n = "", 0
            for r in self._read(path):
                who = {"user": "The writer said", "assistant": "The co-writer said", "note": "Note"}.get(r["role"], "Note")
                line = f"{who}: {r['text'][:CHUNK_CHARS]}"
                if cur and len(cur) + len(line) > CHUNK_CHARS:
                    docs[f"mem:{day}:{n}"] = f"On {day}. {cur.strip()}"
                    cur, n = "", n + 1
                cur += line + "\n"
            if cur.strip():
                docs[f"mem:{day}:{n}"] = f"On {day}. {cur.strip()}"
        return docs

    def search_words(self, book: str, query: str, k: int = 5) -> list[tuple[str, str]]:
        """Keyword fallback when there is no embedding index: [(key, text)] best first."""
        q = {w for w in re.findall(r"[a-z0-9']{3,}", query.lower())}
        if not q:
            return []
        scored = []
        for key, text in self.chunks(book).items():
            words = set(re.findall(r"[a-z0-9']{3,}", text.lower()))
            s = len(q & words)
            if s:
                scored.append((s / (len(q) ** 0.5), key, text))
        scored.sort(reverse=True)
        return [(key, text) for _, key, text in scored[:k]]
