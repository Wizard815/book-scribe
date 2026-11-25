"""Quality gate: cheap, deterministic checks that keep prose in plain English.

`check()` returns human-readable issues; the agent feeds them back for one rewrite pass.
"""
from __future__ import annotations

import re

AI_ISMS = [
    "tapestry", "testament to", "a symphony of", "dance of", "shiver down", "shivers down", "little did",
    "couldn't help but", "barely above a whisper", "eyes sparkled", "palpable", "kaleidoscope",
    "delve", "unbeknownst", "in that moment", "the weight of", "orbs",
    "searing pain", "crushing weight", "cut a wide swath", "in their wake", "hues of", "a new beginning", "man of the earth", "a lifeline",
    "distant roar", "fading dream", "harsh reality", "would never be the same", "forever changed", "a stark reminder", "bittersweet",
    "etched with", "dust motes", "voice thick with", "a constant companion", "a constant reminder", "practiced ease", "a wave of", "truly home",
    "knot in his stomach", "settled in his gut", "eyes closed for the last time", "breath plumed", "the silence stretched", "words hung heavy",
    "lump in his throat", "lump forming in his throat", "tracing a path down", "understanding passed between", "understanding passing between",
]
SUMMING_UP = re.compile(r"\b(?:no more, but|a reminder of|a symbol of|forever (?:changed|marked)|would never be the same|shaped by both|a new chapter|the end of an era|nothing would ever|unknown future|into the future|(?:a|the) future (?:that|which)|still demanded|would always (?:remember|carry)|never (?:forget|forgot)|began at last|had only just begun|at last,? (?:home|free|at peace))\b", re.I)
TELLING = re.compile(r"\b(she|he|they|i)\s+(felt|was feeling|realized|knew that|couldn't believe)\b", re.I)
PASSIVE = re.compile(r"\b(was|were|been|being|is|are)\s+\w+ed\s+by\b", re.I)


STRICTNESS = ("off", "gentle", "strict")
BASE_MAX_SENTENCE = 55


def baseline_max_sentence(paragraphs) -> int:
    """How long a sentence this book's own author allows themselves: 25% above their 98th percentile, never below the default.
    A writer in the long-sentence tradition (Austen, Dickens) must not be told to "split it" by a rule meant for someone else."""
    lens = sorted(len(s.split()) for p in paragraphs for s in re.split(r"(?<=[.!?])\s+", p.text) if s.strip())
    if len(lens) < 40:
        return BASE_MAX_SENTENCE
    return max(BASE_MAX_SENTENCE, int(lens[int(0.98 * (len(lens) - 1))] * 1.25))


def check(text: str, strictness: str = "gentle", max_sentence: int = BASE_MAX_SENTENCE) -> list[str]:
    """Problems worth a rewrite. gentle (default): AI-sounding clichés, markdown, a word repeated nearby. strict adds sentence length (against
    the book's own baseline), passive voice and "telling". off: nothing. Advice, not law: a model is never forced to flatten a voice."""
    if strictness == "off":
        return []
    issues: list[str] = []
    low = text.lower()
    hits = [p for p in AI_ISMS if p in low]
    if hits:
        issues.append("Avoid cliché phrases: " + ", ".join(sorted(hits)))
    last = (re.split(r"(?<=[.!?])\s+", text.strip()) or [""])[-1]
    if SUMMING_UP.search(last):
        issues.append("Do not end on a line that sums up what the moment meant; end on a concrete action, image or spoken line.")
    words = re.findall(r"[a-z']+", low)
    if len(words) >= 30:
        for i in range(len(words) - 8):
            w = words[i]
            if len(w) > 4 and words[i : i + 8].count(w) >= 3:
                issues.append(f"The word '{w}' repeats too often nearby.")
                break
    if re.search(r"[*_#`]{2,}|^\s*[-*]\s", text, re.M):
        issues.append("Plain prose only: no markdown, bullets, or headings.")
    if strictness == "strict":
        if len(PASSIVE.findall(text)) >= 2:
            issues.append("Too much passive voice; prefer active verbs.")
        if len(TELLING.findall(text)) >= 2:
            issues.append("Show the feeling through action or detail instead of naming it.")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if sentences and max(len(s.split()) for s in sentences) > max_sentence:
            issues.append("One sentence is far too long; split it.")
    return issues


_PREAMBLE = re.compile(r"^(?:sure|certainly|of course|here(?:'s| is| are)|okay|ok)\b[^\n]*:\s*$", re.I)


def clean_output(text: str) -> str:
    """Strip what chat models wrap around requested prose: code fences, a 'Here is the rewritten paragraph:' lead-in,
    surrounding quotes. Applied before the quality gate and before anything is shown or saved."""
    s = re.sub(r"^```\w*\n|\n```$", "", text.strip())
    lines = s.split("\n")
    if len(lines) > 1 and _PREAMBLE.match(lines[0].strip()):
        s = "\n".join(lines[1:]).strip()
    if len(s) > 1 and s[0] == s[-1] == '"' and s.count('"') == 2:
        s = s[1:-1].strip()
    return s
