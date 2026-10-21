"""The story bible: canonical, exact facts about the book (what is true *now*).

Layout follows graphify-novel (vendor/graphify-novel/SKILL.md):

    <book>/bible/premise.md
    <book>/bible/timeline.md
    <book>/bible/characters/<slug>.md   YAML front-matter = current state; body = arc log
    <book>/bible/threads/<slug>.md      front-matter `status: open|resolved`
    <book>/bible/world/<topic>.md

Canon is read directly from these files (never from similarity search), and the agent only changes them
through explicit, confirmed edits. graphify builds the *relationship* graph from bible + chapters;
mem0 holds everything fuzzy (writer preferences, drift notes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
FRONT_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)


class BibleError(Exception):
    pass


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise BibleError(f"cannot make a slug from {name!r}")
    return s


@dataclass
class Entry:
    slug: str
    name: str
    state: dict = field(default_factory=dict)
    body: str = ""

    def render(self) -> str:
        fm = yaml.safe_dump({"name": self.name, **{k: v for k, v in self.state.items() if k != "name"}}, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{fm}\n---\n{self.body.strip()}\n"

    def summary(self) -> str:
        facts = "; ".join(f"{k}: {v}" for k, v in self.state.items() if k != "name" and v not in (None, "", []))
        body = self.body.strip()
        return "\n".join(x for x in (facts, body) if x)


def parse_entry(slug: str, raw: str) -> Entry:
    m = FRONT_RE.match(raw)
    state, body = {}, raw
    if m:
        try:
            state = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        body = m.group(2)
    return Entry(slug, str(state.get("name") or slug.replace("-", " ").title()), state, body)


class Bible:
    def __init__(self, book_dir: Path):
        self.root = Path(book_dir) / "bible"

    # --- layout ---------------------------------------------------------
    def ensure(self) -> None:
        for d in ("characters", "threads", "world"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        for f, title in (("premise.md", "Premise"), ("timeline.md", "Timeline")):
            p = self.root / f
            if not p.exists():
                p.write_text(f"# {title}\n\n")

    def _dir(self, kind: str) -> Path:
        if kind not in ("characters", "threads", "world"):
            raise BibleError("unknown bible section")
        return self.root / kind

    def _path(self, kind: str, slug: str) -> Path:
        if not SLUG_RE.match(slug):
            raise BibleError(f"invalid slug {slug!r}")
        return self._dir(kind) / f"{slug}.md"

    # --- reads ----------------------------------------------------------
    def entries(self, kind: str) -> dict[str, Entry]:
        d = self._dir(kind)
        if not d.is_dir():
            return {}
        out = {}
        for f in sorted(d.glob("*.md")):
            if f.stem.startswith("_"):  # _index.md is a roster, not an entry
                continue
            e = parse_entry(f.stem, f.read_text())
            out[e.name] = e
        return out

    def characters(self) -> dict[str, Entry]:
        return self.entries("characters")

    def threads(self) -> dict[str, Entry]:
        return self.entries("threads")

    def read(self, name: str) -> str:
        p = self.root / f"{name}.md"
        return p.read_text() if name in ("premise", "timeline") and p.exists() else ""

    def mentioned(self, kind: str, text: str) -> list[Entry]:
        return [e for n, e in self.entries(kind).items() if re.search(rf"\b{re.escape(n)}\b", text, re.I)]

    def context_for(self, text: str, budget: int = 4000) -> str:
        """Everything the model must treat as fact for a request about `text`."""
        parts = []
        premise = self.read("premise").strip()
        if premise:
            parts.append("PREMISE:\n" + premise)
        for e in self.mentioned("characters", text):
            parts.append(f"CHARACTER {e.name}:\n{e.summary()}")
        open_threads = [e for e in self.threads().values() if str(e.state.get("status", "open")) != "resolved"]
        if open_threads:
            parts.append("OPEN THREADS: " + "; ".join(f"{t.name}" for t in open_threads))
        for e in self.mentioned("world", text):
            parts.append(f"WORLD {e.name}:\n{e.summary()}")
        return "\n\n".join(parts)[:budget]

    # --- writes (callers confirm with the writer first) ----------------------
    def upsert(self, kind: str, name: str, state: dict | None = None, body: str | None = None) -> Entry:
        self.ensure()
        slug = slugify(name)
        p = self._path(kind, slug)
        e = parse_entry(slug, p.read_text()) if p.exists() else Entry(slug, name)
        if state:
            e.state.update(state)
        if body is not None:
            e.body = body
        e.name = name
        p.write_text(e.render())
        return e

    def append_arc(self, name: str, note: str) -> Entry:
        slug = slugify(name)
        p = self._path("characters", slug)
        if not p.exists():
            raise BibleError(f"no character {name!r}")
        e = parse_entry(slug, p.read_text())
        if "## Arc log" not in e.body:
            e.body = e.body.rstrip() + "\n\n## Arc log\n"
        e.body = e.body.rstrip() + f"\n- {note.strip()}\n"
        p.write_text(e.render())
        return e

    # --- migration from the old single-file layout ----------------------------
    def migrate_legacy(self) -> bool:
        book = self.root.parent
        chars, bible = book / "characters.md", book / "bible.md"
        if not (chars.exists() or bible.exists()):
            return False
        self.ensure()
        if bible.exists():
            text = bible.read_text().strip()
            if text and text != "# Story Bible":
                (self.root / "premise.md").write_text(re.sub(r"^# Story Bible\s*", "# Premise\n\n", text) + "\n")
            bible.unlink()
        if chars.exists():
            name, buf = None, []
            blocks = []
            for line in chars.read_text().splitlines():
                m = re.match(r"^#{2,3}\s+(.+)", line)
                if m:
                    if name:
                        blocks.append((name, "\n".join(buf).strip()))
                    name, buf = m.group(1).strip(), []
                elif name:
                    buf.append(line)
            if name:
                blocks.append((name, "\n".join(buf).strip()))
            for n, b in blocks:
                self.upsert("characters", n, body=b)
            chars.unlink()
        return True
