"""Skills: short playbooks the model can load when a job needs a method ("interview the writer about a character").

A skill is a markdown file with a front-matter header:

    ---
    name: character-interview
    description: one line saying when to use it
    ---
    numbered steps...

Built-in skills live in agent/skills/. The administrator can add more in ADDONS_DIR/skills/. The model sees only the names and one-line
descriptions until it calls use_skill, so skills cost nothing when unused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str


def load(*dirs: Path | None) -> dict[str, Skill]:
    out: dict[str, Skill] = {}
    for d in dirs:
        if not d or not Path(d).is_dir():
            continue
        for f in sorted(Path(d).glob("*.md")):
            m = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", f.read_text(encoding="utf-8"), re.S)
            if not m:
                continue
            head = dict(re.findall(r"^(\w+):\s*(.+)$", m.group(1), re.M))
            name = re.sub(r"[^a-z0-9\-]", "", head.get("name", f.stem).lower())
            if name:
                out[name] = Skill(name, head.get("description", "").strip(), m.group(2).strip())
    return out
