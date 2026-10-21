"""Story knowledge graph.

Lightweight in-process KG over characters.md / bible.md / chapters, built so it can be swapped for
graphify-novel (or Neo4j) later behind the same three methods: `refresh`, `find_paragraphs`, `context_for`.
Nodes are characters; edges are co-occurrence in a paragraph. Good enough to resolve "the café scene" or
"where Mira meets Jonas" to paragraph_ids and to feed relevant facts into the prompt.
"""
from __future__ import annotations

import re
from collections import defaultdict

from book_store import BookStore

STOP = set("the a an and or but of to in on at for with from that this was were are is it he she they his her their then when where what who which have had has not into out over under".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']{3,}", s.lower()) if w not in STOP}


class StoryKG:
    def __init__(self, store: BookStore):
        self.store = store
        self.characters: dict[str, str] = {}  # name -> description block
        self.mentions: dict[str, set[str]] = defaultdict(set)  # character -> paragraph_ids
        self.edges: dict[tuple[str, str], int] = defaultdict(int)
        self._paras = []

    def refresh(self, book: str) -> None:
        self.characters = self._parse_characters(self.store.read_doc(book, "characters"))
        self._paras = self.store.all_paragraphs(book)
        self.mentions.clear()
        self.edges.clear()
        for p in self._paras:
            present = [c for c in self.characters if re.search(rf"\b{re.escape(c)}\b", p.text, re.I)]
            for c in present:
                self.mentions[c].add(p.id)
            for i, a in enumerate(present):
                for b in present[i + 1 :]:
                    self.edges[tuple(sorted((a, b)))] += 1

    @staticmethod
    def _parse_characters(md: str) -> dict[str, str]:
        out: dict[str, str] = {}
        name, buf = None, []
        for line in md.splitlines():
            m = re.match(r"^#{2,3}\s+(.+)", line)
            if m:
                if name:
                    out[name] = "\n".join(buf).strip()
                name, buf = m.group(1).strip(), []
            elif name:
                buf.append(line)
        if name:
            out[name] = "\n".join(buf).strip()
        return out

    def find_paragraphs(self, query: str, limit: int = 3) -> list:
        """Rank paragraphs by token overlap, boosted when a named character in the query appears."""
        q = _tokens(query)
        named = [c for c in self.characters if re.search(rf"\b{re.escape(c)}\b", query, re.I)]
        scored = []
        for p in self._paras:
            score = len(q & _tokens(p.text)) + sum(2 for c in named if p.id in self.mentions[c])
            if score:
                scored.append((score, p))
        scored.sort(key=lambda t: -t[0])
        return [p for _, p in scored[:limit]]

    def context_for(self, text: str) -> str:
        """Character sheets + relationships for whoever is named in `text`."""
        named = [c for c in self.characters if re.search(rf"\b{re.escape(c)}\b", text, re.I)]
        lines = [f"{c}: {self.characters[c]}" for c in named if self.characters[c]]
        for (a, b), n in self.edges.items():
            if a in named or b in named:
                lines.append(f"{a} and {b} share {n} paragraph(s).")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {"characters": len(self.characters), "paragraphs": len(self._paras), "edges": len(self.edges)}
