"""Which passages of the book is a piece of text talking about?

Used to light those paragraphs up on the writer's screen while the agent speaks about them, so nobody has to hunt for "the part where…".
Three signals, all cheap and local (no model call):
  1. explicit references, "chapter 3, paragraph 2"
  2. text quoted from the book (a run of 6+ words that appears in a paragraph)
  3. a long verbatim run (7+ words) even without quote marks
"""
from __future__ import annotations

import re

from intent import norm, parse_ref

K = 6          # words in a matching run
MAX_HITS = 5   # never light up more than this many paragraphs at once


class RefFinder:
    def __init__(self, store, kg):
        self.store, self.kg = store, kg
        self._sig = None
        self._shingles: dict[str, set[str]] = {}
        self._words: dict[str, str] = {}

    def _index(self) -> None:
        sig = (self.kg._sig, len(self.kg._paras))
        if sig == self._sig:
            return
        self._sig = sig
        self._shingles, self._words = {}, {}
        for p in self.kg._paras:
            w = norm(p.text).split()
            self._words[p.id] = " " + " ".join(w) + " "
            for i in range(len(w) - K + 1):
                self._shingles.setdefault(" ".join(w[i : i + K]), set()).add(p.id)

    def find(self, book: str, text: str) -> list[str]:
        """Paragraph ids the text refers to, in the order they are mentioned."""
        if not text.strip():
            return []
        self._index()
        found: dict[str, int] = {}  # id -> position of first mention
        # 1. "chapter N ... paragraph M" (a sentence may hold several: split at each "chapter")
        starts = [m.start() for m in re.finditer(r"\bchapter\b", text, re.I)]
        for a, b in zip(starts, starts[1:] + [len(text)]):
            seg = re.split(r"[.!?\n]", text[a:b], maxsplit=1)[0]
            ref = parse_ref(seg)
            if ref.explicit and ref.chapter:
                p = self.store.nth_paragraph(book, ref.chapter, ref.para, ref.scene)
                if p:
                    found.setdefault(p.id, a)
        # 2. quoted spans
        for m in re.finditer(r"[“\"]([^”\"]{20,400})[”\"]", text):
            span = " " + " ".join(norm(m.group(1)).split()) + " "
            if len(span.split()) >= 5:
                for pid, body in self._words.items():
                    if span in body:
                        found.setdefault(pid, m.start())
        # 3. long verbatim runs
        words = norm(text).split()
        hits: dict[str, int] = {}
        for i in range(len(words) - K + 1):
            for pid in self._shingles.get(" ".join(words[i : i + K]), ()):
                hits[pid] = hits.get(pid, 0) + 1
                found.setdefault(pid, 10_000 + i) if hits[pid] >= 2 else None
        return [pid for pid, _ in sorted(found.items(), key=lambda kv: kv[1])][:MAX_HITS]
