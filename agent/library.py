"""The library: files the writer has added (notes, research, scans, photos, family trees, old drafts) kept as readable text.

    <book>/library/<slug>/source.<ext>   the original file, untouched
    <book>/library/<slug>/text.md        what could be read from it (for pictures: what a vision model saw)
    <book>/library/<slug>/meta.json      title, kind, when added, status, notes
    <book>/library/<slug>/tree.json      for family trees: the people and families

The text is cut into chunks and embedded (embeddings.py) so the agent finds things by meaning. Only the writer can see or change it.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import genealogy
from bible import slugify

CHUNK_CHARS = 900
MAX_CHUNKS_PER_ITEM = 700


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > size * 1.5:  # a very long paragraph: cut at a sentence end if possible
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = ""
        cur = (cur + "\n\n" + p).strip()
    if cur:
        out.append(cur)
    return out[:MAX_CHUNKS_PER_ITEM]


class Library:
    def __init__(self, store):
        self.store = store

    def dir(self, book: str) -> Path:
        d = self.store.book_dir(book) / "library"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _item_dir(self, book: str, slug: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,80}", slug):
            raise KeyError(slug)
        d = self.dir(book) / slug
        if not d.is_dir():
            raise KeyError(slug)
        return d

    # --- adding ----------------------------------------------------------------------------------------------------
    def add(self, book: str, filename: str, data: bytes, title: str, kind: str, text: str, status: str = "read", notes: list[str] | None = None,
            pages: int = 0, tree: genealogy.Tree | None = None) -> dict:
        base = slugify(title) if re.search(r"[A-Za-z0-9]", title) else "file"
        slug, n = base[:60], 2
        while (self.dir(book) / slug).exists():
            slug, n = f"{base[:56]}-{n}", n + 1
        d = self.dir(book) / slug
        d.mkdir()
        ext = re.sub(r"[^a-z0-9.]", "", Path(filename).suffix.lower())[:8]
        (d / f"source{ext}").write_bytes(data)
        (d / "text.md").write_text(text, encoding="utf-8")
        meta = {"slug": slug, "title": title, "filename": Path(filename).name[:120], "kind": kind, "added": int(time.time()), "status": status,
                "chars": len(text), "pages": pages, "bytes": len(data), "notes": (notes or [])[:6]}
        if tree:
            meta["people"] = len(tree.people)
            (d / "tree.json").write_text(json.dumps({"people": {k: asdict(v) for k, v in tree.people.items()}, "families": {k: asdict(v) for k, v in tree.families.items()}}), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    def update_text(self, book: str, slug: str, text: str, status: str, notes: list[str] | None = None) -> None:
        d = self._item_dir(book, slug)
        (d / "text.md").write_text(text, encoding="utf-8")
        meta = self.meta(book, slug)
        meta.update(status=status, chars=len(text), notes=(notes or meta.get("notes", []))[:6])
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # --- reading ---------------------------------------------------------------------------------------------------
    def meta(self, book: str, slug: str) -> dict:
        try:
            return json.loads((self._item_dir(book, slug) / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise KeyError(slug)

    def items(self, book: str) -> list[dict]:
        out = []
        for d in self.dir(book).iterdir():
            try:
                out.append(self.meta(book, d.name))
            except KeyError:
                continue
        return sorted(out, key=lambda m: m.get("added", 0), reverse=True)

    def text(self, book: str, slug: str) -> str:
        return (self._item_dir(book, slug) / "text.md").read_text(encoding="utf-8")

    def source(self, book: str, slug: str) -> Path | None:
        return next(iter(sorted(self._item_dir(book, slug).glob("source*"))), None)

    def tree(self, book: str, slug: str) -> genealogy.Tree | None:
        p = self._item_dir(book, slug) / "tree.json"
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        t = genealogy.Tree(source=slug)
        for k, v in raw["people"].items():
            v["birth"], v["death"] = tuple(v["birth"]), tuple(v["death"])
            t.people[k] = genealogy.Person(**v)
        for k, v in raw["families"].items():
            v["marriage"] = tuple(v["marriage"])
            t.families[k] = genealogy.Family(**v)
        return t

    def trees(self, book: str) -> list[tuple[str, genealogy.Tree]]:
        out = []
        for m in self.items(book):
            if m.get("kind") == "family_tree":
                t = self.tree(book, m["slug"])
                if t:
                    out.append((m["slug"], t))
        return out

    def save_aliases(self, book: str, slug: str, mapping: dict) -> None:
        """Real name -> made-up name for living relatives. Kept here (private to the writer), never in the story bible."""
        p = self._item_dir(book, slug) / "aliases.json"
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cur = {}
        cur.update(mapping)
        p.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")

    def aliases(self, book: str, slug: str) -> dict:
        try:
            return json.loads((self._item_dir(book, slug) / "aliases.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def delete(self, book: str, slug: str) -> None:
        shutil.rmtree(self._item_dir(book, slug))

    # --- for the embedding index and keyword search ----------------------------------------------------------------------
    def chunks(self, book: str) -> dict[str, str]:
        docs: dict[str, str] = {}
        for m in self.items(book):
            try:
                text = self.text(book, m["slug"])
            except OSError:
                continue
            for i, c in enumerate(chunk_text(text)):
                docs[f"lib:{m['slug']}:{i}"] = f"From '{m['title']}': {c}"
        return docs

    def search_words(self, book: str, query: str, k: int = 5) -> list[tuple[str, str]]:
        q = {w for w in re.findall(r"[a-z0-9']{3,}", query.lower())}
        if not q:
            return []
        scored = []
        for key, text in self.chunks(book).items():
            s = len(q & set(re.findall(r"[a-z0-9']{3,}", text.lower())))
            if s:
                scored.append((s / (len(q) ** 0.5), key, text))
        scored.sort(reverse=True)
        return [(key, text) for _, key, text in scored[:k]]
