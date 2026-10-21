"""Manuscript store: chapters/*.md with stable paragraph_ids (the bible lives in bible.py).

Chapter format (see BOOK_WRITER.md):

    ---
    chapter: 1
    title: "The Departure"
    ---
    # Chapter 1: The Departure
    ## Scene 1
    <!-- paragraph_id: ch1_s1_p1 -->
    Text of the paragraph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PID_RE = re.compile(r"^<!--\s*paragraph_id:\s*([A-Za-z0-9_\-]+)\s*-->$")
SCENE_RE = re.compile(r"^##\s+Scene\s+(\d+)", re.I)
CHAPTER_FILE_RE = re.compile(r"chapter_(\d+)\.md$")


class BookError(Exception):
    pass


@dataclass
class Paragraph:
    id: str
    text: str
    chapter: int
    scene: int


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise BookError(f"invalid name: {name!r}")
    return name


def parse_chapter(raw: str, chapter: int) -> list[Paragraph]:
    paras: list[Paragraph] = []
    scene = 1
    pending: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal pending, buf
        if pending and buf:
            paras.append(Paragraph(pending, "\n".join(buf).strip(), chapter, scene))
        pending, buf = None, []

    in_front = False
    for i, line in enumerate(raw.splitlines()):
        if i == 0 and line.strip() == "---":
            in_front = True
            continue
        if in_front:
            if line.strip() == "---":
                in_front = False
            continue
        m = PID_RE.match(line.strip())
        if m:
            flush()
            pending = m.group(1)
            continue
        sm = SCENE_RE.match(line.strip())
        if sm or line.startswith("#"):
            flush()
            if sm:
                scene = int(sm.group(1))
            continue
        if pending is not None:
            if line.strip():
                buf.append(line)
            elif buf:
                flush()
    flush()
    return paras


class BookStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # --- books -------------------------------------------------------
    def book_dir(self, book: str) -> Path:
        return self.root / _safe_name(book)

    def bible(self, book: str):
        from bible import Bible  # local import: bible.py needs yaml, book_store stays dependency-free

        return Bible(self.book_dir(book))

    def ensure_book(self, book: str) -> Path:
        d = self.book_dir(book)
        (d / "chapters").mkdir(parents=True, exist_ok=True)
        (d / "memory").mkdir(exist_ok=True)
        bible = self.bible(book)
        bible.migrate_legacy()
        bible.ensure()
        ignore = d / ".graphifyignore"  # graphify reads only what is canon: bible/ + chapters/
        if not ignore.exists():
            ignore.write_text("memory/\ndraft/\nstatic/\ngraphify-out/\n")
        return d

    # --- chapters ----------------------------------------------------
    def chapter_numbers(self, book: str) -> list[int]:
        d = self.book_dir(book) / "chapters"
        if not d.is_dir():
            return []
        nums = [int(m.group(1)) for f in d.iterdir() if (m := CHAPTER_FILE_RE.search(f.name))]
        return sorted(nums)

    def chapter_path(self, book: str, n: int) -> Path:
        return self.book_dir(book) / "chapters" / f"chapter_{n:02d}.md"

    def read_chapter_raw(self, book: str, n: int) -> str:
        p = self.chapter_path(book, n)
        if not p.exists():
            raise BookError(f"chapter {n} not found")
        return p.read_text()

    def paragraphs(self, book: str, n: int) -> list[Paragraph]:
        return parse_chapter(self.read_chapter_raw(book, n), n)

    def all_paragraphs(self, book: str) -> list[Paragraph]:
        out: list[Paragraph] = []
        for n in self.chapter_numbers(book):
            out.extend(self.paragraphs(book, n))
        return out

    def create_chapter(self, book: str, n: int | None = None, title: str = "") -> int:
        self.ensure_book(book)
        if n is None:
            nums = self.chapter_numbers(book)
            n = (nums[-1] + 1) if nums else 1
        p = self.chapter_path(book, n)
        if p.exists():
            raise BookError(f"chapter {n} already exists")
        title = title.replace('"', "'") or f"Chapter {n}"
        p.write_text(f'---\nchapter: {n}\ntitle: "{title}"\n---\n\n# Chapter {n}: {title}\n\n## Scene 1\n\n')
        return n

    def find(self, book: str, paragraph_id: str) -> Paragraph:
        for para in self.all_paragraphs(book):
            if para.id == paragraph_id:
                return para
        raise BookError(f"paragraph {paragraph_id} not found")

    def replace_paragraph(self, book: str, paragraph_id: str, new_text: str) -> Paragraph:
        """Rewrite only the named paragraph; every other byte of the chapter is untouched."""
        para = self.find(book, paragraph_id)
        path = self.chapter_path(book, para.chapter)
        lines = path.read_text().split("\n")
        marker = next((i for i, l in enumerate(lines) if (m := PID_RE.match(l.strip())) and m.group(1) == paragraph_id), None)
        if marker is None:
            raise BookError(f"paragraph {paragraph_id} not found")
        end = marker + 1
        while end < len(lines) and lines[end].strip() and not PID_RE.match(lines[end].strip()) and not lines[end].startswith("#"):
            end += 1
        lines[marker + 1 : end] = new_text.strip().split("\n")
        path.write_text("\n".join(lines))
        return self.find(book, paragraph_id)

    def append_paragraph(self, book: str, chapter: int, text: str, scene: int | None = None) -> Paragraph:
        paras = self.paragraphs(book, chapter)
        scene = scene or (paras[-1].scene if paras else 1)
        k = 1 + sum(1 for p in paras if p.scene == scene)
        pid = f"ch{chapter}_s{scene}_p{k}"
        path = self.chapter_path(book, chapter)
        raw = path.read_text()
        if not any(SCENE_RE.match(l.strip()) and int(SCENE_RE.match(l.strip()).group(1)) == scene for l in raw.splitlines()):
            raw = raw.rstrip("\n") + f"\n\n## Scene {scene}\n"
        path.write_text(raw.rstrip("\n") + f"\n\n<!-- paragraph_id: {pid} -->\n{text.strip()}\n")
        return Paragraph(pid, text.strip(), chapter, scene)

    def resolve_ref(self, book: str, chapter: int, scene: int, para: int) -> Paragraph | None:
        pid = f"ch{chapter}_s{scene}_p{para}"
        try:
            return self.find(book, pid)
        except BookError:
            return None
