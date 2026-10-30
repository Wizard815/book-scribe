"""Fast, LLM-free paragraph locator.

Resolves "the café scene" or "where Mr Bennet teases his wife" to paragraph_ids instantly (it runs on the voice
path). Canon facts come from bible.py and relationships from story_graph.py (graphify); this module only answers
*where in the text*.

Scoring is rarity-weighted word overlap (rare words count more), with light stemming so "quicker" matches
"quickness", plus a boost for paragraphs that mention a character named in the request. It also reports a
*confidence* so the agent can read a guess back instead of silently rewriting the wrong paragraph.
Meaning-level paraphrase ("the father is dismissive") is beyond lexical matching; that needs the embedding provider.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from book_store import BookStore, Paragraph
from intent import norm

STOP = set("""the a an and or but of to in on at for with from that this was were are is it he she they his her their then when where what who
which have had has not into out over under about up down as be been by so if than them him you your i me my we our us do does did can
could would should will just very more less much some any each all one part bit scene paragraph line section chapter sentence says say said
tell tells told make makes made change changes rewrite fix tighten soften there here also too like really
call calls called ask asks asked reply replies replied answer answers speak speaks spoke talk talks mention mentions get gets got goes went come comes came see sees saw look looks looked want wants wanted think thinks thought know knows knew""".split())


def stem(w: str) -> str:
    for suf in ("ingly", "edly", "ness", "ing", "ed", "ly", "er", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def tokens(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z']{3,}", norm(text)) if w not in STOP}


@dataclass
class Hit:
    para: Paragraph
    score: float


class StoryKG:
    def __init__(self, store: BookStore):
        self.store = store
        self.characters: dict[str, str] = {}  # name -> description block
        self._names: dict[str, list[str]] = {}  # name -> normalised spoken forms (name + aliases)
        self.mentions: dict[str, set[str]] = defaultdict(set)  # character -> paragraph_ids
        self.edges: dict[tuple[str, str], int] = defaultdict(int)
        self._paras: list[Paragraph] = []
        self._tok: dict[str, set[str]] = {}
        self._idf: dict[str, float] = {}
        self._sig = None

    # --- indexing ---------------------------------------------------------
    def _signature(self, book: str):
        d = self.store.book_dir(book)
        files = list((d / "chapters").glob("chapter_*.md")) + list((d / "bible" / "characters").glob("*.md"))
        return tuple(sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files))

    def refresh(self, book: str) -> None:
        """Rebuild only when a chapter or character file changed (the agent calls this on every turn)."""
        sig = (book, self._signature(book))
        if sig == self._sig:
            return
        self._sig = sig
        chars = self.store.bible(book).characters()
        self.characters = {n: e.summary() for n, e in chars.items()}
        self._names = {n: [norm(x) for x in [n, *map(str, e.state.get("aliases") or [])] if norm(x)] for n, e in chars.items()}
        self._paras = self.store.all_paragraphs(book)
        self.mentions.clear()
        self.edges.clear()
        self._tok = {p.id: tokens(p.text) for p in self._paras}
        df = Counter(t for ts in self._tok.values() for t in ts)
        n = max(len(self._paras), 1)
        self._idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
        for p in self._paras:
            padded = f" {norm(p.text)} "
            present = [c for c, forms in self._names.items() if any(f" {f} " in padded for f in forms)]
            for c in present:
                self.mentions[c].add(p.id)
            for i, a in enumerate(present):
                for b in present[i + 1 :]:
                    self.edges[tuple(sorted((a, b)))] += 1

    def named_in(self, text: str) -> list[str]:
        padded = f" {norm(text)} "
        return [c for c, forms in self._names.items() if any(f" {f} " in padded for f in forms)]

    # --- search -----------------------------------------------------------
    def search(self, query: str, limit: int = 3, chapter: int | None = None) -> list[Hit]:
        q = tokens(query)
        named = self.named_in(query)
        hits = []
        for p in self._paras:
            if chapter is not None and p.chapter != chapter:
                continue
            overlap = q & self._tok[p.id]
            score = sum(self._idf.get(t, 1.0) for t in overlap)
            if score and named:
                score += sum(1.5 for c in named if p.id in self.mentions[c])
            if score:
                hits.append(Hit(p, score))
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

    def find_paragraphs(self, query: str, limit: int = 3, chapter: int | None = None) -> list[Paragraph]:
        return [h.para for h in self.search(query, limit, chapter)]

    @staticmethod
    def confident(hits: list[Hit], query_tokens: int) -> bool:
        """True when the top hit is clearly the one: enough rare-word evidence AND a clear lead over the runner-up."""
        if not hits:
            return False
        top = hits[0].score
        lead = top / hits[1].score if len(hits) > 1 and hits[1].score else 9.9
        return top >= 7.0 and lead >= 1.3 and query_tokens >= 2

    def context_for(self, text: str) -> str:
        """Character sheets + relationships for whoever is named in `text`."""
        named = self.named_in(text)
        lines = [f"{c}: {self.characters[c]}" for c in named if self.characters[c]]
        for (a, b), n in self.edges.items():
            if a in named or b in named:
                lines.append(f"{a} and {b} share {n} paragraph(s).")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {"characters": len(self.characters), "paragraphs": len(self._paras), "edges": len(self.edges)}
