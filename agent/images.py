"""Pictures in the book.

A picture is a paragraph of its own in the chapter file:

    <!-- paragraph_id: ch3_s1_p4 -->
    ![The hand rising from the lake](image:lake-hand-1a2b3c.jpg)

so it can be moved, deleted, undone and highlighted like any paragraph, and a caption (which is also what is read aloud and what the search index
sees) always travels with it. The files live in <book>/images/. A new picture is first kept as a *candidate* (images/.candidates/) while the writer
is asked yes or no, and only a yes puts it in the book.

Every picture, whatever its source (a generation server, the writer's library, a web search), is opened, checked, shrunk and re-saved here, so a
file that is not really a picture, an enormous one, or one carrying hidden data never reaches the book.
"""
from __future__ import annotations

import hashlib
import json
import io
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse decompression bombs

IMAGE_MD = re.compile(r"^!\[([^\]\n]*)\]\(image:([A-Za-z0-9][A-Za-z0-9._-]{0,100})\)$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")
MAX_INPUT = 15 * 1024 * 1024
MAX_SIDE = 1600
MIN_SIDE = 24
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class PictureError(Exception):
    """A picture that cannot be used. The message is fit to read to the writer."""


def parse(text: str) -> tuple[str, str] | None:
    """(caption, file name) when a paragraph is a picture, else None."""
    m = IMAGE_MD.match((text or "").strip())
    return (m.group(1), m.group(2)) if m else None


def clean_caption(caption: str) -> str:
    c = re.sub(r"<!--.*?-->", " ", caption or "", flags=re.S)
    c = re.sub(r"[\[\]\n\r]+", " ", c)
    return re.sub(r"\s{2,}", " ", c).strip()[:240] or "A picture"


def markdown(caption: str, filename: str) -> str:
    return f"![{clean_caption(caption)}](image:{filename})"


def plain(text: str) -> str:
    """What a paragraph says in words: a picture is "Picture: <caption>". Used for reading aloud, for search and for the outline."""
    p = parse(text)
    return f"Picture: {p[0]}" if p else text


def normalise(data: bytes) -> tuple[bytes, str]:
    """-> (bytes to save, extension). Opens it as a picture, fixes its rotation, shrinks it, drops metadata."""
    if not data or len(data) > MAX_INPUT:
        raise PictureError("That picture is too big or empty.")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im)
    except Exception:
        raise PictureError("That file is not a picture I can open.")
    if min(im.size) < MIN_SIDE:
        raise PictureError("That picture is too small to use.")
    transparent = im.mode in ("RGBA", "LA", "P") and ("A" in im.getbands() or "transparency" in im.info)
    im = im.convert("RGBA" if transparent else "RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    if transparent:
        im.save(out, "PNG", optimize=True)
        return out.getvalue(), ".png"
    im.save(out, "JPEG", quality=86, optimize=True)
    return out.getvalue(), ".jpg"


class ImageStore:
    def __init__(self, store):
        self.store = store

    def dir(self, book: str) -> Path:
        d = self.store.book_dir(book) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cand_dir(self, book: str) -> Path:
        d = self.dir(book) / ".candidates"
        d.mkdir(exist_ok=True)
        return d

    def path(self, book: str, name: str, candidate: bool = False) -> Path:
        if not NAME_RE.match(name or ""):
            raise FileNotFoundError(name)
        p = (self.cand_dir(book) if candidate else self.dir(book)) / name
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(name)
        return p

    @staticmethod
    def _name(hint: str, data: bytes, ext: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", (hint or "picture").lower()).strip("-")[:40] or "picture"
        return f"{base}-{hashlib.sha1(data).hexdigest()[:8]}{ext}"

    def stage(self, book: str, data: bytes, hint: str = "") -> str:
        """Check and keep a picture as a candidate. -> its name. Old candidates are swept."""
        clean, ext = normalise(data)
        name = self._name(hint, clean, ext)
        (self.cand_dir(book) / name).write_bytes(clean)
        cutoff = time.time() - 2 * 86400
        for f in self.cand_dir(book).iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        return name

    def accept(self, book: str, name: str) -> str:
        src = self.path(book, name, candidate=True)
        dest = self.dir(book) / name
        src.replace(dest)
        return name

    def discard(self, book: str, name: str) -> None:
        try:
            self.path(book, name, candidate=True).unlink()
        except FileNotFoundError:
            pass

    def save(self, book: str, data: bytes, hint: str = "") -> str:
        return self.accept(book, self.stage(book, data, hint))

    def used(self, book: str) -> set[str]:
        out = set()
        for p in self.store.all_paragraphs(book):
            if (m := parse(p.text)):
                out.add(m[1])
        return out

    def _notes_path(self, book: str) -> Path:
        return self.dir(book) / ".descriptions.json"

    def note(self, book: str, name: str, text: str | None = None) -> str:
        """What the app said a picture shows (kept so "describe the picture in chapter 2" can say it again later). Reads when text is None."""
        try:
            notes = json.loads(self._notes_path(book).read_text())
        except (OSError, ValueError):
            notes = {}
        if text is None:
            return str(notes.get(name, ""))
        text = " ".join(s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if not re.search(r"\b(?:chapter \d+|paragraph|should i|shall i|put it in|yes or no|say yes)\b", s, re.I))
        if text.strip():
            notes[name] = text.strip()[:600]
            self._notes_path(book).write_text(json.dumps(notes))
        return text

    def info(self, book: str, name: str, caption: str = "", candidate: bool = False) -> dict:
        base = f"/api/books/{book}/images/" + ("candidate/" if candidate else "")
        return {"url": base + name, "caption": caption, "file": name}
