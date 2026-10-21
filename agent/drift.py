"""Drift check: after a paragraph is saved, ask the model whether it contradicts the bible.

Contradictions are *noted*, never auto-fixed: they go to mem0 (kind="drift") so the agent keeps them in mind, and
are read back to the writer on the next turn. The bible stays the only source of canon.
"""
from __future__ import annotations

import json
import re

PROMPT = """You check a new paragraph of a novel against the story bible.
Report ONLY direct contradictions of facts stated in the bible or story graph (names, states, rules, timeline).
Do not report style, and do not report things the bible simply doesn't mention.
Answer with a JSON list of short strings, e.g. ["Mira is described as blind but the bible says she is sighted"]. Answer [] if there are none."""


def parse_issues(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    return [str(i).strip() for i in items if isinstance(i, str) and i.strip()][:5]


async def review(llm, bible_ctx: str, graph_ctx: str, paragraph_id: str, text: str) -> list[str]:
    if not bible_ctx and not graph_ctx:
        return []
    user = f"BIBLE:\n{bible_ctx or '(none)'}\n\nSTORY GRAPH:\n{graph_ctx or '(none)'}\n\nNEW PARAGRAPH ({paragraph_id}):\n{text}"
    raw = await llm.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": user}], temperature=0.0, max_tokens=400)
    return [f"{paragraph_id}: {i}" for i in parse_issues(raw)]
