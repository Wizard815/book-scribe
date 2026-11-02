"""Understanding *spoken* commands. Pure functions, no I/O, so every rule is unit-testable.

Speech-to-text output is not typed text: numbers arrive as words ("chapter two"), punctuation is unreliable, silence is
sometimes transcribed as "Thank you.", and people say "yes, but make it shorter". Routing therefore works on a normalised
copy of the utterance (`norm`) while the original wording is still what the model sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
ORDINALS = {w: i for i, w in enumerate("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_WORD = re.compile(r"[a-z']+|\d+|[^a-z'\d]+")


def norm(text: str) -> str:
    """Lowercase, spoken numbers -> digits ("twenty one" -> 21), titles unified, punctuation dropped."""
    toks = _WORD.findall(text.lower().replace("’", "'"))
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 2] if i + 2 < len(toks) and toks[i + 1].strip(" -") == "" else None  # skip one space/hyphen
        if t in _TENS and nxt in _UNITS and 0 < _UNITS[nxt] < 10:
            out.append(str(_TENS[t] + _UNITS[nxt])); i += 3; continue
        if t in _TENS:
            out.append(str(_TENS[t]))
        elif t in _UNITS and t != "zero":
            out.append(str(_UNITS[t]))
        elif t in ("mister",):
            out.append("mr")
        elif t in ("missus", "misses"):
            out.append("mrs")
        else:
            out.append(t)
        i += 1
    s = re.sub(r"(?<=\w)\.(?=\s|$)", "", "".join(out))  # "mr." -> "mr"
    s = re.sub(r"\b(?!(?:let|it|he|she|that|there|here|what|who|how|where|when|why|one)'s\b)(\w+)'s\b", r"\1", s)  # "Jonas's voice" -> "jonas voice"
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s)).strip()


# --- whisper artefacts --------------------------------------------------------------------------
_NOISE = {
    "you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "bye bye", "goodbye",
    "okay", "ok", "uh", "um", "hmm", "mm", "mhm", "so", "oh", "ah", "blank audio", "silence", "music", "applause",
    "subtitles by the amaraorg community", "please subscribe", "like and subscribe",
}


def is_noise(text: str) -> bool:
    """Empty/near-empty audio that a speech model turned into a stock phrase. Never worth a model call."""
    n = norm(re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", text))  # "[BLANK_AUDIO]", "(silence)"
    return n == "" or n in _NOISE or (len(n) <= 2 and not n.isdigit())


# --- confirmations ---------------------------------------------------------------------------------
_YES_CORE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "save", "confirm", "ahead", "good", "perfect", "great", "fine", "alright", "correct", "right", "approve", "accept", "keep"}
_YES_FILL = {"please", "do", "it", "that", "go", "sounds", "looks", "thanks", "thank", "you", "now", "then", "and", "the", "is", "its", "it's", "one", "this", "all", "i", "like", "love", "that's", "thats", "lets", "let's", "sure", "just"}
_NO_CORE = {"no", "nope", "nah", "cancel", "discard", "scrap", "never", "dont", "don't", "stop", "reject", "undo"}
_NO_FILL = {"please", "mind", "that", "it", "save", "use", "thanks", "thank", "you", "this", "one", "do", "not", "now", "i", "dont", "don't", "want"}
HOLD = re.compile(r"^(not yet|wait|hold on|hang on|one (moment|second|sec)|give me a (moment|second|sec)|let me think)\b")


def _words(text: str) -> list[str]:
    return norm(text).split()


def confirmation(text: str) -> tuple[str | None, str]:
    """-> ("yes"|"no"|"hold"|None, remainder).

    ("yes", "")                  "yes please save it"      -> save
    ("yes", "make it shorter")   "yes but make it shorter" -> NOT a clean yes: caller must treat the remainder as a revision
    ("no", "make it longer")     "no, make it longer"      -> revise, don't just drop
    (None, text)                 "don't change the ending" -> a new instruction, not an answer
    """
    w = _words(text)
    if not w:
        return None, ""
    if HOLD.match(norm(text)):
        return "hold", ""
    # a yes/no word must lead the utterance; everything after the leading run of filler is the remainder
    if w[0] in _YES_CORE or w[0] in ("do", "go", "sounds", "looks", "that's", "thats") and any(x in _YES_CORE for x in w[:3]):
        kind, core, fill = "yes", _YES_CORE, _YES_FILL
    elif w[0] in _NO_CORE:
        kind, core, fill = "no", _NO_CORE, _NO_FILL
    else:
        return None, text
    i = 0
    while i < len(w) and (w[i] in core or w[i] in fill):
        i += 1
    rest = w[i:]
    if kind == "no" and w[0] in ("dont", "don't") and rest:
        return None, text  # "don't change the ending of chapter 2" is a new instruction, not a rejection
    if rest and rest[0] in ("but", "except", "only", "although", "and", "though"):
        rest = rest[1:]
    return kind, " ".join(rest)


# --- modes ------------------------------------------------------------------------------------------
_THERAPIST = re.compile(r"\b(therapist|feedback|critique|critic|coach|reviewer|editor)\b.*\bmode\b|\bmode\b.*\b(therapist|feedback|critique|coach)\b|\bquick feedback\b|\bgive me feedback\b|\bswitch to (therapist|feedback)\b")
_NARRATOR = re.compile(r"\b(narrator|narration|writing|drafting|story|co ?writer|writer)\b.*\bmode\b|\bmode\b.*\b(narrator|writing|drafting)\b|\bback to (writing|narrator|the story|drafting)\b")


def detect_mode(text: str) -> str | None:
    n = norm(text)
    if _NARRATOR.search(n):
        return "narrator"
    if _THERAPIST.search(n):
        return "therapist"
    return None


# --- references ---------------------------------------------------------------------------------------
@dataclass
class Ref:
    chapter: int | None = None
    scene: int | None = None
    para: int | None = None      # 1-based; -1 = last
    whole_book: bool = False     # "the opening line of the book"

    @property
    def explicit(self) -> bool:
        return self.para is not None


_PARA_WORD = r"(?:paragraph|para|line|sentence|part|section|bit)"
_ORD = "|".join(sorted(ORDINALS, key=len, reverse=True) + ["last", "final", "opening", "closing", "very first", "very last", "beginning", "start", "end"])


def parse_ref(text: str) -> Ref:
    n = norm(text)
    r = Ref()
    if m := re.search(r"\bchapter (\d+)\b", n):
        r.chapter = int(m.group(1))
    if m := re.search(r"\bscene (\d+)\b", n):
        r.scene = int(m.group(1))
    if m := re.search(rf"\b{_PARA_WORD} (\d+)\b", n):
        r.para = int(m.group(1))
    elif m := re.search(rf"\b({_ORD}) (?:{_PARA_WORD})\b", n):
        w = m.group(1).replace("very ", "")
        r.para = -1 if w in ("last", "final", "closing", "end") else ORDINALS.get(w, 1)
    elif re.search(r"\b(opening|beginning|first) (line|sentence|words)\b", n) or re.search(r"\bthe (very )?(start|beginning) of\b", n):
        r.para = 1
    if r.para is not None and r.chapter is None and re.search(r"\b(of|in) the (book|novel|story)\b|\bopening\b|\bvery first\b", n):
        r.chapter, r.whole_book = 1, True
    return r


# --- command shapes -----------------------------------------------------------------------------------------
_POLITE = r"(?:(?:please|kindly|hey|ok|okay|so|now|and|then)\s+)*(?:(?:can|could|would|will) you (?:please )?|i want you to |i'd like you to |i would like you to |let's |lets |go ahead and |you should |just )?"
_EDIT_VERBS = r"(?:change|make|rewrite|rephrase|reword|fix|edit|tweak|soften|tighten|revise|replace|shorten|lengthen|expand|trim|polish|improve|punch up|sharpen|simplify|modernize|modernise)"
_EDIT_LEAD = re.compile(rf"^{_POLITE}{_EDIT_VERBS}\b")
_EDIT_CLAUSE = re.compile(rf"[,;]\s*(?:and |then |please )*{_EDIT_VERBS}\b|\b(?:and|then) {_EDIT_VERBS} (?:him|her|them|it|that|this|the)\b")
_DRAFT_LEAD = re.compile(rf"^{_POLITE}(?:write|draft|continue|carry on|keep going|go on|next scene|add (?:a|another|one more) (?:scene|paragraph)|compose|describe)\b")
_QUESTION = re.compile(r"^(?:so |and |but )?(how|what|why|should|do you|does|is|are|which|would|am i|did|will it|shall)\b")
_OPINION = re.compile(r"^(?:i |i'm |im |i've |my )?(?:hate|like|love|dislike|prefer|think|feel|want|wish|need|always|never|don't like|dont like|can't|cant)\b|^i (?:hate|like|love|dislike|prefer|think|feel|wish|really|just|always|never|don't|dont|can't|cant)\b")


def is_question(text: str) -> bool:
    return bool(_QUESTION.match(norm(text))) or text.strip().endswith("?") and not _EDIT_LEAD.match(norm(text))


_NEGATIVE = re.compile(r"^(?:please )?(?:don't|dont|do not|never|stop|avoid)\b")  # a constraint on future edits, not an edit
_VERB_ANYWHERE = re.compile(rf"\b{_EDIT_VERBS}\b")
_LOCATOR_OPEN = re.compile(r"^(?:in|at|when|where|during|after|before|on|for|the|that|this|chapter|paragraph|scene)\b")


def is_edit(text: str) -> bool:
    """An imperative aimed at existing text. Transcripts rarely have commas, so "in chapter 3 when they meet make him
    nervous" counts: an edit verb after a location phrase. Questions and opinions ("how do I make...", "I hate it when
    you make...") do not."""
    n = norm(text)
    if is_question(text) or _OPINION.match(n) or _NEGATIVE.match(n):
        return False
    if _EDIT_LEAD.match(n) or _EDIT_CLAUSE.search(text.lower()):
        return True
    return bool(_VERB_ANYWHERE.search(n) and (_LOCATOR_OPEN.match(n) or parse_ref(text).chapter is not None))


def is_draft(text: str) -> bool:
    n = norm(text)
    return not (is_question(text) or _OPINION.match(n)) and bool(_DRAFT_LEAD.match(n))


_GOTO = re.compile(r"^(?:(?:please|ok|okay|now|and)\s+)*(?:(?:go|jump|skip|turn|flip|take me|back|switch|move)(?: back| ahead| forward)?(?: to| over to)?|open|show(?: me)?|read(?: me)?|start(?: at| from)?|let's (?:go to|read)) (?:the )?(?:start of )?chapter (\d+)(?: again)?$")
_STEP = re.compile(r"^(?:(?:please|ok|okay|now|and)\s+)*(?:(?:go|read|show|move|skip)(?: me)?(?: to| on to)? )?(?:the )?(next|previous|prior|following|one before) (?:paragraph|section|part|one)$|^(next|previous|continue reading|read on|go back|back|go on|keep reading)$")


def goto_chapter(text: str) -> int | None:
    """Only a bare navigation command: 'go to chapter two', never 'rewrite chapter 3 paragraph 2'."""
    m = _GOTO.match(norm(text)) or re.match(r"^chapter (\d+)$", norm(text))
    return int(m.group(1)) if m else None


def step_direction(text: str) -> int:
    m = _STEP.match(norm(text))
    if not m:
        return 0
    w = m.group(1) or m.group(2)
    return +1 if w in ("next", "following", "continue reading", "read on", "go on", "keep reading") else -1


def is_read_aloud(text: str) -> bool:
    """'read chapter 3 paragraph 2' / 'read me the opening line' – navigate to a paragraph and speak it."""
    n = norm(text)
    return bool(re.match(r"^(?:(?:please|ok|okay|now|and)\s+)*(?:read|show|go to|take me to|jump to|play)(?: me)?(?: back)?(?: to)? ", n)) and parse_ref(text).para is not None


_MARKER = re.compile(r"\b(?:where|when|during|after|before|while|the part|the bit|the scene|the moment|the line|the section)\b")


def locator_query(text: str) -> str:
    """The part of an edit request that *describes where*, without the instruction words.

    "make Mr Bennet more sarcastic where he teases his wife" -> "where he teases his wife": searching the whole sentence
    would match paragraphs containing the word "sarcastic", i.e. the style the writer wants, not the scene they mean.
    """
    n = norm(text)
    m = _MARKER.search(n)
    if not m:
        return text
    tail = n[m.start():]
    v = _VERB_ANYWHERE.search(tail)
    if v and v.start() > 0:  # "...when they meet at the cafe make him nervous": the clause ends where the instruction starts
        tail = tail[: v.start()]
    return tail.strip() or text


# --- voices ---------------------------------------------------------------------------------------------------
_V_LIST = re.compile(r"\b(?:what|which|list|show|tell me)\b.*\bvoices?\b|\bavailable voices\b|\bvoice options\b")
_V_RESET = re.compile(r"\b(?:reset|forget|clear|remove|undo|start over)\b.*\bvoice\b")
_V_ANOTHER = re.compile(r"\b(?:another|different|other|new|next)\b.*\bvoice\b|\bvoice\b.*\b(?:another|different)\b|\btry another\b|\bnot (?:this|that) voice\b")
_V_AUDITION = re.compile(r"^(?:(?:please|ok|okay|now|and|so)\s+)*(?:let me hear|lets hear|let's hear|can i hear|could i hear|i want to hear|i'd like to hear|play|play me|audition|test|try out|say something (?:as|in)|read (?:me )?a line (?:as|in)|what does .+ sound like|how does .+ sound)\b|\bsound(?:s)? like\b")


def voice_command(text: str) -> dict | None:
    """A request about how someone *sounds*. Returns {"action": ...} or None.

    The caller still has to find out *whose* voice (a character, the narrator, "your voice"): "play the next paragraph"
    matches the audition shape but names nobody, so it is not a voice command.
    """
    n = norm(text)
    has_voice = bool(re.search(r"\bvoices?\b", n))
    if _V_LIST.search(n) and has_voice:
        return {"action": "list"}
    if _V_RESET.search(n):
        return {"action": "reset"}
    if _V_ANOTHER.search(n):
        return {"action": "another"}
    if _V_AUDITION.search(n) and not re.search(r"\b(?:more|less|too|deeper|slower|faster|higher|lower)\b", n):
        return {"action": "audition"}
    if has_voice:
        return {"action": "adjust", "feedback": text}
    return None
