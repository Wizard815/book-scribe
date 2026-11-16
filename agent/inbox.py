"""The inbox folder and the file finder: how files get to Book Scribe without the browser's upload button.

Inbox     <book>/inbox/   a plain folder. Drop files into it from the computer (it is inside the books folder that is already shared with the
          container). The agent notices new files, asks "I see a new file ... add it?", and moves it into the library when the writer says yes.
Finder    Looks for a file the writer describes ("the PDF I just downloaded") in the folders the *administrator* shared with Book Scribe
          (FILE_SEARCH_DIRS, for example a read-only mount of Downloads). It only lists names, sizes and dates. Nothing is read or copied until
          the writer confirms which file it is, and nothing outside those folders is ever visible.

Safety: only regular files under a shared root are reachable (symlinks are not followed and real paths are checked), hidden and temporary
files are skipped, and the model only ever sees short ids, never paths.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

PARTIAL = (".crdownload", ".part", ".partial", ".tmp", ".download", ".opdownload")
SETTLE_SECONDS = 3  # a file still being copied in is left alone until it stops changing
STOP = set("""a an the my i me we you it its they them this that these those one ones just new newly downloaded download downloads file files can could would should
will shall may might must please open find look looking search for from on in of at to by with about into over after before again and or but so then if is are was were
be been am have has had do did does get got put show up out recent recently latest last now want need wanted needed know see use using used thing things stuff something
there here what which where who whom whose also any some all each both few more most other such no nor not only own same than too very s t""".split())
KINDS = {
    "pdf": (".pdf",), "word": (".docx", ".doc", ".odt", ".rtf"), "document": (".docx", ".doc", ".odt", ".rtf", ".pdf", ".txt", ".md"),
    "spreadsheet": (".xlsx", ".xls", ".csv", ".ods"), "excel": (".xlsx", ".xls", ".csv"),
    "picture": (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tif", ".tiff"), "photo": (".jpg", ".jpeg", ".png", ".heic", ".webp"),
    "image": (".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"), "screenshot": (".png", ".jpg", ".jpeg"),
    "recording": (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"), "audio": (".mp3", ".wav", ".m4a", ".ogg", ".flac"),
    "tree": (".ged", ".gdz", ".ftm", ".ftmb", ".gramps"), "gedcom": (".ged", ".gdz"), "ftm": (".ftm", ".ftmb"), "genealogy": (".ged", ".gdz", ".ftm", ".ftmb", ".gramps"),
    "zip": (".zip",), "ebook": (".epub",), "slides": (".pptx", ".ppt", ".odp"), "text": (".txt", ".md"),
}


def human_size(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB" if n >= 1_000_000 else f"{max(1, n // 1000)} KB"


def ago(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{s // 60} minutes ago"
    if s < 86400 * 2:
        return f"{s // 3600} hours ago"
    return f"{s // 86400} days ago"


@dataclass
class Found:
    id: str  # "1", "2"... shown to the model
    root: int
    rel: str
    name: str
    size: int
    mtime: float
    where: str  # "Downloads"


class Inbox:
    def __init__(self, store):
        self.store = store

    def dir(self, book: str) -> Path:
        d = self.store.book_dir(book) / "inbox"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_path(self, book: str) -> Path:
        return self.dir(book) / ".seen.json"

    def _state(self, book: str) -> dict:
        try:
            return json.loads(self._state_path(book).read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, book: str, st: dict) -> None:
        self._state_path(book).write_text(json.dumps(st))

    @staticmethod
    def _key(path: Path) -> str:
        st = path.stat()
        return f"{path.name}|{st.st_size}|{int(st.st_mtime)}"

    def files(self, book: str, settled_only: bool = True) -> list[dict]:
        out = []
        now = time.time()
        for p in sorted(self.dir(book).iterdir()):
            if p.name.startswith(".") or p.is_symlink() or not p.is_file() or p.name.lower().endswith(PARTIAL):
                continue
            st = p.stat()
            if settled_only and now - st.st_mtime < SETTLE_SECONDS:
                continue
            out.append({"name": p.name, "size": st.st_size, "modified": int(st.st_mtime), "key": self._key(p)})
        return out

    def fresh(self, book: str) -> list[dict]:
        """Files that have not been announced to the writer yet. Marks them announced, so each file is asked about once."""
        st = self._state(book)
        seen = set(st.get("seen", []))
        new = [f for f in self.files(book) if f["key"] not in seen]
        if new:
            st["seen"] = sorted(seen | {f["key"] for f in new})[-500:]
            self._save(book, st)
        return new

    def path(self, book: str, name: str) -> Path:
        base = Path(name).name
        p = self.dir(book) / base
        if base != name or base.startswith(".") or p.is_symlink() or not p.is_file():
            raise FileNotFoundError(name)
        return p

    def done(self, book: str, name: str) -> None:
        """After a file is in the library: move it into inbox/added/ (the writer's own copy is never deleted)."""
        p = self.path(book, name)
        dest = self.dir(book) / "added"
        dest.mkdir(exist_ok=True)
        target = dest / p.name
        n = 2
        while target.exists():
            target = dest / f"{p.stem}-{n}{p.suffix}"
            n += 1
        p.replace(target)


class Finder:
    """Searches the administrator's shared folders by description."""

    def __init__(self, roots: list[str], max_bytes: int):
        self.roots = [Path(r).expanduser().resolve() for r in roots if r.strip()]
        self.roots = [r for r in self.roots if r.is_dir()]
        self.max_bytes = max_bytes

    @property
    def enabled(self) -> bool:
        return bool(self.roots)

    def label(self, i: int) -> str:
        return self.roots[i].name or str(self.roots[i])

    def search(self, query: str, within_days: float | None = None, limit: int = 5, max_depth: int = 3, max_files: int = 30000) -> list[Found]:
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOP]
        exts: set[str] = set()
        for w in list(words):
            if w in KINDS:
                exts |= set(KINDS[w])
                words.remove(w)
        now, scored, seen = time.time(), [], 0
        for ri, root in enumerate(self.roots):
            for dirpath, dirs, files in os.walk(root, followlinks=False):
                depth = len(Path(dirpath).relative_to(root).parts)
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")] if depth < max_depth else []
                for fn in files:
                    seen += 1
                    if seen > max_files:
                        break
                    if fn.startswith(".") or fn.lower().endswith(PARTIAL):
                        continue
                    full = Path(dirpath) / fn
                    try:
                        if full.is_symlink():
                            continue
                        st = full.stat()
                    except OSError:
                        continue
                    if st.st_size == 0 or st.st_size > self.max_bytes:
                        continue
                    age = now - st.st_mtime
                    if within_days and age > within_days * 86400:
                        continue
                    ext = full.suffix.lower()
                    if exts and ext not in exts:
                        continue
                    toks = set(re.findall(r"[a-z0-9]+", full.stem.lower()))
                    hit = sum(1 for w in words if w in toks or any(w in t for t in toks if len(w) >= 3))
                    if words and not hit:
                        continue
                    score = hit * 3 + (2 if exts else 0) + (4 if age < 600 else 3 if age < 3600 else 2 if age < 86400 else 1 if age < 7 * 86400 else 0)
                    scored.append((score, st.st_mtime, Found("", ri, str(full.relative_to(root)), fn, st.st_size, st.st_mtime, self.label(ri))))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out = []
        for i, (_, _, f) in enumerate(scored[:limit], 1):
            f.id = str(i)
            out.append(f)
        return out

    def resolve(self, root: int, rel: str) -> Path:
        """The real file for a Found entry, re-checked: it must still be a regular file inside the shared root."""
        base = self.roots[root]
        full = (base / rel).resolve()
        if base not in full.parents or not full.is_file() or full.is_symlink() or (base / rel).is_symlink():
            raise FileNotFoundError(rel)
        return full
