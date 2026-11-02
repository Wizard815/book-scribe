"""Turning a paragraph into who-says-what, so each character can be voiced.

`segments()` splits prose into narration and quoted dialogue. `attribute()` decides who speaks each quote:

  1. a dialogue tag naming a character next to the quote  ("…," said Mr. Bennet / Elizabeth replied, "…")
  2. otherwise two-person alternation: the speaker before the last one (A, B, then the next untagged quote is A)
  3. otherwise the narrator, so a guess is never an invented speaker

Pronoun-only tags ("he said") are resolved by alternation. A model can resolve what's left (`llm_attribute`), cached.
Names come from the bible: a character's name plus any `aliases:` (e.g. "his wife"), so relational tags can be taught.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from intent import norm

NARRATOR = "narrator"
_QUOTE = re.compile(r"[“\"]([^”\"]+)[”\"]")
_VERBS = ("said replied cried asked returned answered exclaimed whispered shouted added continued observed rejoined inquired "
          "murmured muttered declared called remarked persisted interposed sighed laughed snapped insisted suggested protested "
          "retorted begged urged resumed interrupted stammered growled demanded").split()
_VERB = r"(?:%s)" % "|".join(_VERBS)


@dataclass
class Seg:
    text: str
    speaker: str  # "narrator" or a character name
    quoted: bool = False
    resolved: bool = True  # False = a quote nobody could be assigned to by a tag (alternation/narrator fallback was used)


def split_quotes(text: str) -> list[tuple[str, bool]]:
    out, pos = [], 0
    for m in _QUOTE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _find_name(fragment: str, forms: dict[str, list[str]]) -> str | None:
    """A character named in a dialogue tag: 'said Mr. Bennet', 'Elizabeth replied', 'cried his wife'."""
    n = f" {norm(fragment)} "
    best = None
    for name, fs in forms.items():
        for f in fs:
            m = re.search(rf"(?:\b{_VERB}\b\s+(?:\w+\s+)?{re.escape(f)}\b|\b{re.escape(f)}\s+(?:\w+\s+)?{_VERB}\b)", n)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), name)
    return best[1] if best else None


def _vocatives(quote: str, forms: dict[str, list[str]]) -> list[str]:
    """Characters *spoken to* in a quote: "My dear Mr. Bennet, ..." / "Come here, Mr. Bennet." A name merely mentioned
    ("I dare say Mr. Bingley will be glad to see you") is a third party, not the next speaker."""
    out = []
    for name, fs in forms.items():
        for f in fs:
            tok = r"\.?\s+".join(re.escape(w) for w in f.split()) + r"\.?"
            if re.search(rf"(?:^[\W_]*|\bmy dear\s+|\bdear\s+|\bsir,?\s+){tok}\s*[,!?]|[,]\s*{tok}\s*[.!?]?[\W_]*$", quote, re.I):
                out.append(name)
                break
    return out


def _indirect(text: str, forms: dict[str, list[str]]) -> str | None:
    """"Mr. Bennet replied that he had not." – no quote, but he has just spoken, so the next untagged quote is the other person's."""
    return _find_name(text, forms)


def _pronoun_speaker(fragment: str, recent: list[str], genders: dict[str, str]) -> str | None:
    """"returned she" / "he said": the most recent speaker of that gender."""
    n = f" {norm(fragment)} "
    want = "female" if re.search(rf"\b{_VERB}\s+(?:she|her)\b|\b(?:she|her)\s+{_VERB}\b", n) else "male" if re.search(rf"\b{_VERB}\s+(?:he|him)\b|\b(?:he|him)\s+{_VERB}\b", n) else None
    return next((s for s in reversed(recent) if genders.get(s) == want), None) if want else None


def attribute(text: str, forms: dict[str, list[str]], recent: list[str] | None = None, genders: dict[str, str] | None = None) -> tuple[list[Seg], list[str]]:
    """-> (segments, updated list of recent speakers, newest last). `forms`: name -> normalised spoken forms."""
    recent, genders = list(recent or []), genders or {}
    parts = split_quotes(text)
    has_quote = any(q for _, q in parts)
    segs: list[Seg] = []
    for i, (t, quoted) in enumerate(parts):
        if not quoted:
            segs.append(Seg(t, NARRATOR))
            if not has_quote and (who := _indirect(t, forms)):
                recent.append(who)
            continue
        after = parts[i + 1][0][:90] if i + 1 < len(parts) and not parts[i + 1][1] else ""
        before = parts[i - 1][0][-90:] if i > 0 and not parts[i - 1][1] else ""
        who = _find_name(after, forms) or _find_name(before, forms) or _pronoun_speaker(after, recent, genders) or _pronoun_speaker(before, recent, genders)
        resolved = who is not None
        if who is None:
            others = [s for s in reversed(recent[:-1]) if s != (recent[-1] if recent else None)]
            who = others[0] if others else NARRATOR
        segs.append(Seg(t, who, True, resolved))
        if who != NARRATOR:
            # Someone addressed by name ("My dear Mr. Bennet, ...") is the likely next speaker: slot them in as "the other person".
            recent += [n for n in _vocatives(t, forms) if n != who][:1]
            recent.append(who)
    return merge_tiny(segs), recent[-4:]


def merge_tiny(segs: list[Seg]) -> list[Seg]:
    """Don't send ', ' or '.' to a TTS engine on its own: fold fragments with no letters into the previous segment."""
    out: list[Seg] = []
    for s in segs:
        if out and not re.search(r"[A-Za-z0-9]", s.text):
            out[-1].text += s.text
        elif out and out[-1].speaker == s.speaker and out[-1].quoted == s.quoted:
            out[-1].text += s.text
        else:
            out.append(Seg(s.text, s.speaker, s.quoted, s.resolved))
    return [s for s in out if s.text.strip()]


def forms_for(characters: dict) -> dict[str, list[str]]:
    """name -> spoken forms, from the bible entries (name + aliases), normalised the way transcripts are."""
    return {n: [f for f in {norm(n), *(norm(str(a)) for a in e.state.get("aliases") or [])} if f] for n, e in characters.items()}


def narration_only(text: str) -> list[Seg]:
    return [Seg(text, NARRATOR)] if text.strip() else []


# --- model assist for quotes the rules couldn't place ---------------------------------------------------
ATTRIBUTE_PROMPT = """You assign speakers to quoted dialogue in a novel. Reply with ONLY a JSON list with one entry per quote,
in order: a character name from the list, or null if nobody in the list is speaking it.
Characters: {names}"""


def cache_key(prev: str, text: str) -> str:
    return hashlib.sha1((prev + "\n" + text).encode()).hexdigest()[:16]


async def llm_attribute(llm, names: list[str], prev: str, segs: list[Seg], text: str) -> list[str | None] | None:
    quotes = [s.text for s in segs if s.quoted]
    if not quotes:
        return None
    user = f"PREVIOUS PARAGRAPH:\n{prev or '(none)'}\n\nPARAGRAPH:\n{text}\n\nQuotes in order:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(quotes))
    raw = await llm.chat([{"role": "system", "content": ATTRIBUTE_PROMPT.format(names=", ".join(names))}, {"role": "user", "content": user}], temperature=0.0, max_tokens=300)
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(out, list) or len(out) != len(quotes):
        return None
    return [x if isinstance(x, str) and x in names else None for x in out]
