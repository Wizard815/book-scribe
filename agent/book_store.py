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



def _read(path) -> str:
    """Read a chapter file the way a person's editor may have saved it: UTF-8 (with or without a BOM), UTF-16, or old Windows text.
    The file is always written back as UTF-8, so a manuscript saved by Notepad or Word is repaired the first time it is edited."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")

class BookError(Exception):
    pass


def clean_paragraph(text: str) -> str:
    """One paragraph, one line, nothing that could be mistaken for file structure.

    Model output sometimes contains blank lines (extra paragraphs), a leading '#' or an HTML comment; written raw, those
    would silently split or corrupt the chapter file and orphan later paragraph_ids.
    """
    s = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    s = re.sub(r"\s*\n\s*", " ", s.strip())
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        raise BookError("empty paragraph")
    return s


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
        return _read(p)

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
        title = title.replace('"', "'").strip()
        head = f"Chapter {n}: {title}" if title else f"Chapter {n}"
        p.write_text(f'---\nchapter: {n}\ntitle: "{title or f"Chapter {n}"}"\n---\n\n# {head}\n\n## Scene 1\n\n')
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
        lines = _read(path).split("\n")
        marker = next((i for i, l in enumerate(lines) if (m := PID_RE.match(l.strip())) and m.group(1) == paragraph_id), None)
        if marker is None:
            raise BookError(f"paragraph {paragraph_id} not found")
        end = marker + 1
        while end < len(lines) and lines[end].strip() and not PID_RE.match(lines[end].strip()) and not lines[end].startswith("#"):
            end += 1
        lines[marker + 1 : end] = [clean_paragraph(new_text)]
        self._snapshot(book, para.chapter, "edit", paragraph_id)
        path.write_text("\n".join(lines))
        self._stamp(book)
        return self.find(book, paragraph_id)

    def append_paragraph(self, book: str, chapter: int, text: str, scene: int | None = None) -> Paragraph:
        paras = self.paragraphs(book, chapter)
        scene = scene or (paras[-1].scene if paras else 1)
        nums = [int(m.group(1)) for p in paras if p.scene == scene and (m := re.search(r"_p(\d+)$", p.id))]
        k = 1 + max(nums, default=0)  # one past the highest, so ids stay unique after a paragraph was deleted
        pid = f"ch{chapter}_s{scene}_p{k}"
        path = self.chapter_path(book, chapter)
        raw = _read(path)
        if not any(SCENE_RE.match(l.strip()) and int(SCENE_RE.match(l.strip()).group(1)) == scene for l in raw.splitlines()):
            raw = raw.rstrip("\n") + f"\n\n## Scene {scene}\n"
        text = clean_paragraph(text)
        self._snapshot(book, chapter, "append", pid)
        path.write_text(raw.rstrip("\n") + f"\n\n<!-- paragraph_id: {pid} -->\n{text}\n")
        self._stamp(book)
        return Paragraph(pid, text, chapter, scene)

    def delete_paragraph(self, book: str, paragraph_id: str) -> Paragraph:
        """Remove one paragraph (undoable). Returns what was removed."""
        para = self.find(book, paragraph_id)
        path = self.chapter_path(book, para.chapter)
        lines = _read(path).split("\n")
        marker = next((i for i, l in enumerate(lines) if (m := PID_RE.match(l.strip())) and m.group(1) == paragraph_id), None)
        if marker is None:
            raise BookError(f"paragraph {paragraph_id} not found")
        end = marker + 1
        while end < len(lines) and lines[end].strip() and not PID_RE.match(lines[end].strip()) and not lines[end].startswith("#"):
            end += 1
        while end < len(lines) and not lines[end].strip():
            end += 1
        self._snapshot(book, para.chapter, "delete", paragraph_id)
        path.write_text("\n".join(lines[:marker] + lines[end:]))
        self._stamp(book)
        return para

    def insert_paragraph(self, book: str, chapter: int, text: str, before_id: str | None = None) -> Paragraph:
        """Add a paragraph just before another one (or at the end when before_id is None). Undoable."""
        if before_id is None:
            return self.append_paragraph(book, chapter, text)
        before = self.find(book, before_id)
        if before.chapter != chapter:
            raise BookError("That paragraph is in a different chapter.")
        path = self.chapter_path(book, chapter)
        lines = _read(path).split("\n")
        i = next((k for k, l in enumerate(lines) if (m := PID_RE.match(l.strip())) and m.group(1) == before_id), None)
        if i is None:
            raise BookError(f"paragraph {before_id} not found")
        paras = self.paragraphs(book, chapter)
        nums = [int(m.group(1)) for p in paras if p.scene == before.scene and (m := re.search(r"_p(\d+)$", p.id))]
        pid = f"ch{chapter}_s{before.scene}_p{1 + max(nums, default=0)}"
        text = clean_paragraph(text)
        self._snapshot(book, chapter, "append", pid)
        lines[i:i] = [f"<!-- paragraph_id: {pid} -->", text, ""]
        path.write_text("\n".join(lines))
        self._stamp(book)
        return Paragraph(pid, text, chapter, before.scene)

    def move_paragraph(self, book: str, paragraph_id: str, to_chapter: int, where="end") -> Paragraph:
        """Move a paragraph: to the end of a chapter (the default), to its start, or just before / after another paragraph. A new chapter is made if it
        is the next number. The whole move is one undo."""
        import uuid
        para = self.find(book, paragraph_id)
        same = to_chapter == para.chapter
        if same and where == "end" and self.paragraphs(book, to_chapter)[-1].id == paragraph_id:
            raise BookError("It is already there.")
        nums = self.chapter_numbers(book)
        if to_chapter not in nums:
            if to_chapter != (max(nums) + 1 if nums else 1):
                raise BookError(f"There is no chapter {to_chapter}, and it is not the next one.")
            self.create_chapter(book, to_chapter)
        before_id: str | None = None
        existing = [p for p in self.paragraphs(book, to_chapter) if p.id != paragraph_id] if to_chapter in nums or same else []
        if where == "start":
            before_id = existing[0].id if existing else None
        elif isinstance(where, tuple):
            k = where[1]
            if not 1 <= k <= len(self.paragraphs(book, to_chapter)):
                raise BookError(f"There is no paragraph {k} in chapter {to_chapter}.")
            anchor = self.paragraphs(book, to_chapter)[k - 1]
            if anchor.id == paragraph_id:
                raise BookError("That is the paragraph you are moving.")
            if where[0] == "before":
                before_id = anchor.id
            else:
                later = [p for p in self.paragraphs(book, to_chapter) if p.id != paragraph_id]
                at = next(i for i, p in enumerate(later) if p.id == anchor.id)
                before_id = later[at + 1].id if at + 1 < len(later) else None
        if same and before_id is None and where == "end":
            pass
        self._group = uuid.uuid4().hex
        try:
            new = self.insert_paragraph(book, to_chapter, para.text, before_id)
            self.delete_paragraph(book, paragraph_id)
        finally:
            self._group = None
        return new

    # --- undo: every saved change first copies the chapter as it was, so the writer can always take it back ---------------------------
    HISTORY_KEEP = 200

    def _hist(self, book: str) -> Path:
        d = self.book_dir(book) / "history"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _snapshot(self, book: str, chapter: int, what: str, paragraph_id: str) -> None:
        import hashlib, json, time
        path = self.chapter_path(book, chapter)
        before = _read(path) if path.exists() else ""
        d = self._hist(book)
        import uuid
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}_ch{chapter}.md"  # unique even for two saves in one millisecond
        (d / name).write_text(before, encoding="utf-8")
        with open(d / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": name, "chapter": chapter, "what": what, "paragraph": paragraph_id, "at": int(time.time()), "after": None,
                                "group": getattr(self, "_group", None)}) + "\n")
        self._pending_hash = (book, hashlib.sha1(before.encode()).hexdigest())

    @staticmethod
    def _rows(log) -> list[dict]:
        import json
        out = []
        try:
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if isinstance(r, dict) and "file" in r and "chapter" in r:
                        out.append(r)
                except ValueError:
                    continue  # one damaged line must not disable undo
        except OSError:
            pass
        return out

    def _stamp(self, book: str) -> None:
        """Remember what the chapter looked like right after the change, so an undo can tell if someone edited it from outside meanwhile."""
        import hashlib, json
        log = self._hist(book) / "log.jsonl"
        try:
            rows = self._rows(log)
            last = rows[-1]
            last["after"] = hashlib.sha1(_read(self.chapter_path(book, last["chapter"])).encode()).hexdigest()
            rows[-1] = last
            log.write_text("".join(json.dumps(r) + "\n" for r in rows[-self.HISTORY_KEEP:]), encoding="utf-8")
        except (OSError, ValueError, IndexError, KeyError):
            pass

    def undo_last(self, book: str) -> dict | None:
        """Put the book back as it was before the last saved change. -> {"chapter", "what", "paragraph"} or None when there is nothing to undo.
        Raises BookError if the chapter was changed outside the app since (undoing would throw that work away)."""
        import hashlib, json
        log = self._hist(book) / "log.jsonl"
        rows = self._rows(log)
        while rows:
            last = rows[-1]
            snap = self._hist(book) / str(last.get("file", ""))
            if snap.is_file() and snap.parent == self._hist(book):
                break
            rows.pop()  # a log line whose copy is gone cannot be undone
        else:
            return None
        batch = [last]  # a move is two saved changes; undoing it takes back both
        if last.get("group"):
            batch = [r for r in rows if r.get("group") == last["group"]][::-1]
        checked: set[int] = set()
        for r in batch:  # newest first. Check everything before touching anything, so a refusal leaves the book exactly as it is.
            if r["chapter"] in checked:
                continue  # an older save of the same chapter inside this move: only the newest one can match the file as it is now
            checked.add(r["chapter"])
            p = self.chapter_path(book, r["chapter"])
            if r.get("after") and hashlib.sha1((_read(p) if p.exists() else "").encode()).hexdigest() != r["after"]:
                raise BookError("That chapter was changed outside Book Scribe since, so I will not undo over it.")
        for r in batch:
            s = self._hist(book) / str(r["file"])
            self.chapter_path(book, r["chapter"]).write_text(s.read_text(encoding="utf-8"), encoding="utf-8")
            s.unlink()
            rows.remove(r)
        log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return {"chapter": last["chapter"], "what": "move" if last.get("group") else last["what"], "paragraph": last.get("paragraph", "")}

    def nth_paragraph(self, book: str, chapter: int, n: int, scene: int | None = None) -> Paragraph | None:
        """n-th paragraph of a chapter (1-based, -1 = last), optionally within one scene. None if out of range."""
        try:
            paras = [p for p in self.paragraphs(book, chapter) if scene is None or p.scene == scene]
        except BookError:
            return None
        if not paras or n == 0 or abs(n) > len(paras):
            return None
        return paras[n - 1] if n > 0 else paras[n]

    def resolve_ref(self, book: str, chapter: int, scene: int, para: int) -> Paragraph | None:
        pid = f"ch{chapter}_s{scene}_p{para}"
        try:
            return self.find(book, pid)
        except BookError:
            return None
