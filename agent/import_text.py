"""Import an existing plain-text manuscript into a book folder.

    python import_text.py --book mybook --file draft.txt [--chapters 5] [--heading REGEX]

Splits on chapter headings, unwraps hard-wrapped lines into one paragraph each, and assigns stable paragraph_ids.
Run it with PROJECT_DIR pointing at your books folder (same as the agent).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from book_store import BookStore

DEFAULT_HEADING = r"(?i)^\s*chapter\s+(?:\d+|[ivxlc]+)\b[.:]?\s*(.*)$"


def _plain(paragraph: str) -> str:
    """Drop Project Gutenberg's _emphasis_ markers: the book text is plain, and raw underscores would be read aloud/shown."""
    return re.sub(r"(?<![\w])_([^_\n]+)_(?![\w])", r"\1", paragraph)


def split_chapters(text: str, heading: str = DEFAULT_HEADING) -> list[tuple[str, list[str]]]:
    """-> [(title, [paragraph, ...])]. Text before the first heading is ignored (front matter, licences)."""
    rx = re.compile(heading)
    chapters: list[tuple[str, list[list[str]]]] = []
    for line in text.replace("\r\n", "\n").replace("﻿", "").split("\n"):
        m = rx.match(line)
        if m:
            chapters.append((m.group(1).strip() if m.groups() else "", [[]]))
        elif chapters:
            paras = chapters[-1][1]
            if line.strip():
                paras[-1].append(line.strip())
            elif paras[-1]:
                paras.append([])
    return [(t, [_plain(" ".join(p)) for p in ps if p]) for t, ps in chapters]


def import_text(store: BookStore, book: str, text: str, max_chapters: int | None = None, heading: str = DEFAULT_HEADING) -> tuple[int, int]:
    store.ensure_book(book)
    n_ch = n_par = 0
    for i, (title, paras) in enumerate(split_chapters(text, heading), 1):
        if max_chapters and i > max_chapters:
            break
        store.create_chapter(book, i, title)
        for p in paras:
            store.append_paragraph(book, i, p)
        n_ch, n_par = n_ch + 1, n_par + len(paras)
    return n_ch, n_par


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--chapters", type=int, help="import only the first N chapters")
    ap.add_argument("--heading", default=DEFAULT_HEADING, help="regex for a chapter heading line; group 1 = title")
    a = ap.parse_args()
    root = Path(os.getenv("PROJECT_DIR", "/data/books"))
    ch, par = import_text(BookStore(root), a.book, a.file.read_text(encoding="utf-8-sig", errors="replace"), a.chapters, a.heading)
    print(f"imported {ch} chapters, {par} paragraphs into {root / a.book}")
