"""Quality gate: cheap, deterministic checks that keep prose in plain English.

`check()` returns human-readable issues; the agent feeds them back for one rewrite pass.
"""
from __future__ import annotations

import re

AI_ISMS = [
    "tapestry", "testament to", "a symphony of", "dance of", "shiver down", "shivers down", "little did",
    "couldn't help but", "barely above a whisper", "eyes sparkled", "palpable", "kaleidoscope",
    "delve", "unbeknownst", "in that moment", "the weight of", "orbs",
]
TELLING = re.compile(r"\b(she|he|they|i)\s+(felt|was feeling|realized|knew that|couldn't believe)\b", re.I)
PASSIVE = re.compile(r"\b(was|were|been|being|is|are)\s+\w+ed\s+by\b", re.I)


def check(text: str) -> list[str]:
    issues: list[str] = []
    low = text.lower()
    hits = [p for p in AI_ISMS if p in low]
    if hits:
        issues.append("Avoid cliché phrases: " + ", ".join(sorted(hits)))
    words = re.findall(r"[a-z']+", low)
    if len(words) >= 30:
        for i in range(len(words) - 8):
            w = words[i]
            if len(w) > 4 and words[i : i + 8].count(w) >= 3:
                issues.append(f"The word '{w}' repeats too often nearby.")
                break
    if len(PASSIVE.findall(text)) >= 2:
        issues.append("Too much passive voice; prefer active verbs.")
    if len(TELLING.findall(text)) >= 2:
        issues.append("Show the feeling through action or detail instead of naming it.")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences and max(len(s.split()) for s in sentences) > 55:
        issues.append("One sentence is far too long; split it.")
    if re.search(r"[*_#`]{2,}|^\s*[-*]\s", text, re.M):
        issues.append("Plain prose only: no markdown, bullets, or headings.")
    return issues
