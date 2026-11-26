"""Understanding *spoken* commands. Pure functions, no I/O, so every rule is unit-testable.

Speech-to-text output is not typed text: numbers arrive as words ("chapter two"), punctuation is unreliable, silence is
sometimes transcribed as "Thank you.", and people say "yes, but make it shorter". Routing therefore works on a normalised
copy of the utterance (`norm`) while the original wording is still what the model sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
ORDINALS = {w: i for i, w in enumerate("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_WORD = re.compile(r"[a-z']+|\d+|[^a-z'\d]+")


def norm(text: str) -> str:
    """Lowercase, spoken numbers -> digits ("twenty one" -> 21), titles unified, punctuation dropped."""
    text = re.sub(r"(?<=\d)\s*[-\u2013\u2014]\s*(?=\d)", " to ", text)  # "paragraphs 2-4" means 2 to 4
    toks = _WORD.findall(text.lower().replace("’", "'"))
    out: list[str] = []
    i = 0
    def small(j: int) -> tuple[int, int] | None:  # a number below 100 starting at token j -> (value, next index)
        if j >= len(toks):
            return None
        w = toks[j]
        n2 = toks[j + 2] if j + 2 < len(toks) and toks[j + 1].strip(" -") == "" else None
        if w in _TENS and n2 in _UNITS and 0 < _UNITS[n2] < 10:
            return _TENS[w] + _UNITS[n2], j + 3
        if w in _TENS:
            return _TENS[w], j + 1
        if w in _UNITS and w != "zero":
            return _UNITS[w], j + 1
        return None

    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 2] if i + 2 < len(toks) and toks[i + 1].strip(" -") == "" else None  # skip one space/hyphen
        if nxt == "hundred" and (t == "a" or (t in _UNITS and 0 < _UNITS[t] < 10)):  # "one hundred and five" -> 105
            val, j = (1 if t == "a" else _UNITS[t]) * 100, i + 3
            k = j + 2 if j + 1 < len(toks) and toks[j].strip() == "" and toks[j + 1] == "and" else j
            if k < len(toks) and toks[k].strip() == "" and (rest := small(k + 1)):
                val, j = val + rest[0], rest[1]
            out.append(str(val)); i = j; continue
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
    s = re.sub(r"(?<!paragraph )(?<!paragraphs )(?<!chapter )(?<!chapters )(?<!scene )(?<!and )(?<!to )\b(1[1-9]) ([1-9]\d|0[1-9])\b", lambda m: m.group(1) + m.group(2), s)  # "eighteen sixty two" -> 18 62 -> 1862: a spoken year
    s = re.sub(r"^(?:\W*(?:um+|uh+|uhm|erm|er|hmm+|mm+|ah)\b[\s,.]*)+(?=\w)", "", s)  # "um, undo that": a spoken filler is not part of the command
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
    return n == "" or n in _NOISE or (len(n) <= 2 and not n.isdigit() and n not in ("no", "ok", "hi", "go", "so"[:0] or "no"))


# --- confirmations ---------------------------------------------------------------------------------
_YES_CORE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "save", "confirm", "ahead", "good", "perfect", "great", "fine", "alright", "correct", "right", "approve", "accept", "keep"}
_YES_FILL = {"them", "those", "these", "beautiful", "fantastic", "marvelous", "marvellous", "grand", "splendid", "terrific", "super", "fine", "good", "great", "right", "very", "really", "so", "oh", "well", "please", "do", "it", "that", "go", "sounds", "looks", "better", "nice", "nicer", "lovely", "wonderful", "excellent", "much", "thanks", "thank", "you", "now", "then", "and", "the", "is", "its", "it's", "one", "this", "all", "i", "like", "love", "that's", "thats", "lets", "let's", "sure", "just"}
_NO_CORE = {"no", "nope", "nah", "cancel", "discard", "scrap", "never", "dont", "don't", "stop", "reject", "undo"}
_NO_FILL = {"please", "mind", "that", "it", "save", "use", "thanks", "thank", "you", "this", "one", "do", "not", "now", "i", "dont", "don't", "want"}
HOLD = re.compile(r"^(not yet|wait|hold on|hang on|one (moment|second|sec)|give me a (moment|second|sec)|let me think)\b")


def _words(text: str) -> list[str]:
    return norm(text).split()


_NO_DROP = re.compile(r"^(?:just )?(?:leave|keep) (?:it|that|this|them)?(?: as it was| as it is| as is| the same| alone| how it was| like it was)?(?: please)?$|^(?:forget|drop|cancel|skip|scrap) (?:it|that|this)(?: please)?$|^never ?mind(?: that| it)?$|^(?:as it was|the same|leave it)$|^(?:thanks|thank you|thank|please)$|^not now$|^(?:that'?s|thats|it'?s|its|that is|it is|all) (?:fine|ok|okay|good|alright|all right)(?: thanks| thank you)?$|^(?:i )?(?:changed|change) my mind$")
_CHANGE = re.compile(r"\b(?:leave (?:\w+ ){0,3}out|skip|exclude|ignore|drop|make|change|different|instead|another|darker|lighter|bigger|smaller|more|less|without|but|not|no|don'?t|empty|remove|rename|call it|shorter|longer|except|only|wrong|other)\b")
_SAVE_WORDS = re.compile(r"^(?:(?:please )?(?:go ahead(?: and)?|add|save|put|use|keep|do|write|store|file)(?: it| that| this| them| those| these| all| all of them| (?:that|this|the) (?:one|1)| the (?:paragraph|picture))?(?: in| in there| down| to the book| into the book| away| in the library| to the library| in my library| in the wiki| to the wiki)?(?: please| now| for me)?|that one|that'?s? (?:the (?:one|1)|fine|good|right|great|perfect|lovely|nice|mine|it)|(?:it'?s|this is) (?:mine|good|great|fine|right|perfect|lovely|nice|the (?:one|1))|do it|mine|(?:the )?(?:one|1)|that|this|it|them|those|these|all|thanks|thank you|great|perfect|lovely|good|nice|fine|right|exactly)(?:,? (?:put it in|use it|add it|save it|keep it|go ahead|please|thank you|thanks))*$")


def confirmation(text: str) -> tuple[str | None, str]:
    """-> ("yes"|"no"|"hold"|None, remainder).

    ("yes", "")                  "yes please save it"      -> save
    ("yes", "make it shorter")   "yes but make it shorter" -> NOT a clean yes: caller must treat the remainder as a revision
    ("no", "make it longer")     "no, make it longer"      -> revise, don't just drop
    (None, text)                 "don't change the ending" -> a new instruction, not an answer
    """
    w = _words(text)
    if len(w) > 1 and w[0] in ("so", "well", "oh") and (w[1] in _YES_CORE or w[1] in _NO_CORE):
        w = w[1:]  # "well yes", "so no"
    if not w:
        return None, ""
    if HOLD.match(norm(text)) and (len(w) <= 4 or not re.search(r"\b(?:write|change|make|call|put|start|go|read|delete|move|add|save|what|where|who|how|it should)\b", norm(text))):
        return "hold", ""  # "Wait a minute, what's the book called?" is a question, not "hold on"
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
    rest = [x for x in w[i:] if x not in ("um", "umm", "uh", "uhh", "er", "erm", "uhm", "hmm")]  # fillers are not words of a change
    if kind == "no" and w[0] in ("dont", "don't") and rest:
        return None, text  # "don't change the ending of chapter 2" is a new instruction, not a rejection
    if rest and rest[0] in ("but", "except", "only", "although", "and", "though"):
        if len(rest) == 1 and rest[0] != "and":
            return "hold", ""  # "yes, but" and then the microphone closed: the writer has more to say, so nothing is saved
        rest = rest[1:]
    if kind == "no" and _NO_DROP.match(" ".join(rest)):
        rest = []  # "no, leave it as it was": a plain no
    if kind == "yes" and re.fullmatch(r"(?:that'?s |thats |now )?(?:more like it|much better|a lot better|better|just right|right)(?: now)?", " ".join(rest)):
        rest = []  # "Yes, that's more like it."
    if kind == "yes" and (_SAVE_WORDS.match(" ".join(rest)) or re.fullmatch(r"(?:in|to|into) (?:the|my) (?:library|wiki|book|story wiki)(?: please| too)?", " ".join(rest))):
        rest = []  # "yes, add it" / "yes save that" / "yes go ahead and put it in": more yes, not a change request
    return kind, " ".join(rest)


# --- modes ------------------------------------------------------------------------------------------
_THERAPIST = re.compile(r"\b(therapist|feedback|critique|critic|coach|reviewer|editor)\b.*\bmode\b|\bmode\b.*\b(therapist|feedback|critique|coach)\b|\bquick feedback\b|\bgive me feedback\b|\bswitch to (therapist|feedback)\b")
_NARRATOR = re.compile(r"\b(narrator|narration|writing|drafting|story|co ?writer|writer)\b.*\bmode\b|\bmode\b.*\b(narrator|writing|drafting)\b|\bback to (writing|narrator|the story|drafting)\b")


def detect_mode(text: str) -> str | None:
    n = norm(text)
    if len(n.split()) > 8:
        return None  # "okay, back to the story. Everett boards the train...": the story goes on, it is not a mode switch
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
    picture: bool = False        # "the picture in chapter 3": the writer means the picture, not a paragraph of writing

    @property
    def explicit(self) -> bool:
        return self.para is not None


_PARA_WORD = r"(?:paragraph|para|line|sentence|part|section|bit)"
_ORD = "|".join(sorted(ORDINALS, key=len, reverse=True) + ["last", "final", "opening", "closing", "very first", "very last", "beginning", "start", "end"])


def parse_ref(text: str) -> Ref:
    n = norm(text)
    r = Ref()
    # "as in chapter two", "same lady as chapter six", "like chapter 2": a chapter the writer is pointing back to, not where the work goes
    n_dest = re.sub(r"\b(?:as in|as|like in|like|same as(?: in)?|just as in|back in|earlier in|mentioned in|remember(?: in)?|unlike(?: in)?|similar to|than in|compared (?:to|with)|the way (?:it was )?in|from earlier in|that we (?:wrote|had) in|we had in)\s+chapter\s+\d+", " ", n)
    fix = re.search(r"\b(?:no wait|wait no|sorry|i mean|i meant|no i mean|or rather|actually)\b", n_dest)  # "chapter one, no wait, chapter two": the last one counts
    if ms := re.findall(r"\bchapter (\d+)\b", n_dest):
        r.chapter = int(ms[-1] if fix else ms[0])
    if m := re.search(r"\bscene (\d+)\b", n):
        r.scene = int(m.group(1))
    if m := re.search(rf"\b(?:(second|next|third) (?:to|from) (?:the )?last|(\d+|second|third) from the end) (?:{_PARA_WORD}|one)\b|\b(?:{_PARA_WORD}|one) (?:(second|next|third) (?:to|from) (?:the )?last)\b", n):
        w = m.group(1) or m.group(2) or m.group(3)
        r.para = -({"next": 2, "second": 2, "third": 3}.get(w) or int(w))  # "the second to last paragraph" is one before the last
    elif fix and (ps := re.findall(rf"\b{_PARA_WORD} (\d+)\b", n)):
        r.para = int(ps[-1])
    elif m := re.search(rf"\b{_PARA_WORD} (\d+)\b", n):
        r.para = int(m.group(1))
    elif m := (re.search(rf"\b({_ORD}) (?:paragraph|para|section|part|bit)\b", n) or re.search(rf"\b({_ORD}) (?:line|sentence)\b", n)):
        # "the last sentence of the first paragraph" is paragraph 1: a paragraph word wins over a sentence or line word
        w = m.group(1).replace("very ", "")
        r.para = -1 if w in ("last", "final", "closing", "end") else ORDINALS.get(w, 1)
    elif re.search(r"\b(opening|beginning|first) (line|sentence|words)\b", n) or re.search(r"\bthe (very )?(start|beginning) of\b", n):
        r.para = 1
    if r.para is not None and r.chapter is None and re.search(r"\b(of|in) the (book|novel|story)\b|\bopening\b|\bvery first\b", n):
        r.chapter, r.whole_book = 1, True
    pic = r"(?:picture|photo|photograph|image|illustration|drawing)"
    if re.search(rf"\b{_PARA_WORD}s? (?:right )?(?:before|after|above|below|next to|beside|under|over|around) (?:the |that |this )?{pic}\b", n):
        r.para, r.picture = None, False  # "the paragraph before the picture": a paragraph, which one the writer has not said
    elif re.search(rf"\b(?:the |that |this |my |its |last |first )?{pic}\b", n):
        if not re.search(rf"\b{_PARA_WORD} \d+\b", n) or re.search(rf"\b{pic} (?:after|before|in|at|following|near|beside|next to|under|over)\b", n):
            r.picture = True  # "delete the picture in chapter 3" and "the picture after paragraph 2" (the picture, whatever number follows)
            r.para = -1 if re.search(rf"\blast {pic}\b", n) else 1
    return r


# --- command shapes -----------------------------------------------------------------------------------------
_POLITE = r"(?:(?:please|kindly|hey|ok|okay|so|now|and|then)\s+)*(?:(?:can|could|would|will) you (?:please )?|i want you to |i'd like you to |i would like you to |let's |lets |go ahead and |you should |just )?"
_EDIT_VERBS = r"(?:take (?:that|this|it|the)(?: [a-z0-9]+){0,9}? out|cut (?:that|this|it|the)(?: [a-z0-9]+){0,9}? out|cut out|take out|leave out|change|make|rewrite|rephrase|reword|fix|edit|tweak|soften|tighten|revise|replace|shorten|lengthen|expand|trim|polish|improve|punch up|sharpen|simplify|modernize|modernise)"
_EDIT_LEAD = re.compile(rf"^{_POLITE}{_EDIT_VERBS}\b")
_EDIT_CLAUSE = re.compile(rf"[,;]\s*(?:and |then |please )*{_EDIT_VERBS}\b|\b(?:and|then) {_EDIT_VERBS} (?:him|her|them|it|that|this|the)\b")
_DRAFT_LEAD = re.compile(rf"^{_POLITE}(?:write|draft|continue|carry on|keep going|go on|next scene|next paragraph|(?:1|one) more (?:paragraph|bit|part)|(?:1|one) more(?= (?:the|a|of|about|with|where|when|\w+ing)\b)|(?:and )?(?:then |now )?the (?:part|scene|bit|moment) (?:where|when|in which|with)|tell (?:me )?(?:the |about the )?(?:part|bit|scene|story|moment) (?:where|when|of|about|in which)|now a paragraph (?:about|on|where|for)|a paragraph (?:about|where) |another (?:paragraph|bit)|more of the story|the next (?:bit|part)|(?:the )?(?:last|final|closing|opening|first|next) paragraph (?:for|of|in) chapter \d+(?= \w)|add (?:a|another|one more) (?:scene|paragraph)|compose|describe)\b")
_QUESTION = re.compile(r"^(?:so |and |but )?(how|what|why|who|whom|whose|where|should|do you|does|is|are|which|would|am i|did|will it|shall)\b")
_OPINION = re.compile(r"^(?:i |i'm |im |i've |my )?(?:hate|like|love|dislike|prefer|think|feel|want|wish|need|always|never|don't like|dont like|can't|cant)\b|^i (?:hate|like|love|dislike|prefer|think|feel|wish|really|just|always|never|don't|dont|can't|cant)\b")


def is_question(text: str) -> bool:
    return bool(_QUESTION.match(norm(text))) or text.strip().endswith("?") and not _EDIT_LEAD.match(norm(text))


_NEGATIVE = re.compile(r"^(?:please )?(?:don't|dont|do not|never|stop|avoid)\b")  # a constraint on future edits, not an edit
_VERB_ANYWHERE = re.compile(rf"\b{_EDIT_VERBS}\b")
_LOCATOR_OPEN = re.compile(r"^(?:in|at|when|where|during|after|before|on|for|the|that|this|chapter|paragraph|scene)\b")


_WIKI_TALK = re.compile(r"\b(?:wiki|story notes?|story bible|bible|character pages?|notes? (?:for|on|about)|pages? for|save (?:notes?|them|these)|story pages?)\b")


_SUBJECT = re.compile(r"\b(?:they|he|she|we|everyone|nobody|both|the family|his \w+|her \w+|their \w+)\b")


def _narration(text: str, n: str) -> bool:
    """A person in the story is the one who changes or makes something: a subject (a pronoun or a name) comes before the first edit verb,
    and the writer named no paragraph or chapter number."""
    r = parse_ref(text)
    if r.para is not None or r.chapter is not None:
        return False
    v = _VERB_ANYWHERE.search(n)
    if not v:
        return False
    if re.search(rf"(?<!\bto )(?<!\bhad )\b{_EDIT_VERBS} (?:it|that|this|him|her|them) (?:\w+er|more|less|sound|feel|seem|read|look)\b", n):
        return False  # "when Everett leaves make it sadder": an instruction about the text
    before = n[: v.start()]
    if _SUBJECT.search(before):
        return True
    raw_words = re.findall(r"[A-Za-z']+", text)
    first_verb = v.group().split()[0]
    for i, w in enumerate(raw_words[1:], 1):  # a name (a capital letter, not the first word) before the verb: "After the funeral Mira had to make dinner"
        if w.lower() == first_verb:
            break
        if w[0].isupper() and w not in ("I",) and raw_words[i - 1][-1:] != "." :
            return True
    return False


def is_edit(text: str) -> bool:
    """An imperative aimed at existing text. Transcripts rarely have commas, so "in chapter 3 when they meet make him
    nervous" counts: an edit verb after a location phrase. Questions and opinions ("how do I make...", "I hate it when
    you make...") do not."""
    n = norm(text)
    if is_question(text) or _OPINION.match(n) or _NEGATIVE.match(n):
        return False
    if re.match(rf"^{_POLITE}(?:yes,? |no,? )?(?:please )?(?:make|do|write) (?:one|some|it|that|them)(?: please| now| for me| again)?$", n) and parse_ref(text).para is None:
        return False  # "yes, make one": an answer to my question, not an edit of any paragraph
    if _WIKI_TALK.search(n) and parse_ref(text).para is None:
        return False  # "make wiki pages for the father, the farm and the camp" is about the story wiki, not an edit of a paragraph
    if _EDIT_LEAD.match(n):
        return True
    if _narration(text, n):
        return False  # "In this chapter they fix up the farmhouse and make it a home": someone in the story does it, the writer is telling me the story
    if _EDIT_CLAUSE.search(text.lower()):
        return True
    return bool(_VERB_ANYWHERE.search(n) and (_LOCATOR_OPEN.match(n) or parse_ref(text).chapter is not None))


def is_draft(text: str) -> bool:
    n = norm(text)
    if _WIKI_TALK.search(n) and parse_ref(text).para is None:
        return False  # "write story notes for three characters" is about the wiki, not a new paragraph
    return not (is_question(text) or _OPINION.match(n)) and bool(_DRAFT_LEAD.match(n))


def draft_first(text: str) -> bool:
    """A request for NEW text ("next paragraph, the lords try three times", "one more paragraph...") must not be taken for an edit of an old paragraph just because
    it mentions a chapter or a verb like "make". It stays an edit when it names a paragraph or says again / instead / rewrite."""
    n = norm(text)
    return is_draft(text) and parse_ref(text).para is None and not re.search(r"\b(?:again|instead|rewrite|redo|change|fix|replace|shorten|lengthen|expand|trim|polish)\b", n)


_GOTO = re.compile(r"^(?:(?:please|ok|okay|now|and|then|so|um|uh)\s+)*(?:(?:go|jump|skip|turn|flip|take me|back|switch|move)(?: back| ahead| forward)?(?: to| over to)?|open|show(?: me)?|read(?: me)?|start(?: at| from)?|let's (?:go to|read)) (?:the )?(?:start of )?chapter (\d+)(?: again)?$")
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
    if _V_ANOTHER.search(n) and (has_voice or re.search(r"\b(?:sound|sounds|speak|speaks|talks|reads?|read aloud|accent)\b", n)):
        return {"action": "another"}  # "try another one for the march" is about writing, not a voice
    if _V_AUDITION.search(n) and not re.search(r"\b(?:more|less|too|deeper|slower|faster|higher|lower)\b", n):
        return {"action": "audition"}
    if has_voice:
        return {"action": "adjust", "feedback": text}
    return None


# --- random idea requests ("give me a name for the porter") ----------------------------------------------------------------------------
_IDEA_KINDS = [
    (r"\bnames?\b|\bcall (?:him|her|them|it|the \w+)\b", "name"),
    (r"\bplaces?\b|\blocations?\b|\bsettings?\b|\btowns?\b|\bvillages?\b|\binns?\b|\btaverns?\b|\bpubs?\b|\bstreets?\b", "place"),
    (r"\bcharacters?\b|\bsomeone interesting\b", "character"),
    (r"\bproblems?\b|\bcomplications?\b|\bconflicts?\b|\bobstacles?\b", "problem"),
    (r"\btwists?\b", "twist"), (r"\btitles?\b", "title"), (r"\bobjects?\b|\bprops?\b", "object"),
    (r"\bsensory\b|\bsmells?\b|\bsounds?\b|\bdetails?\b", "sense"), (r"\bweather\b", "weather"),
    (r"\bjobs?\b|\boccupations?\b|\bprofessions?\b", "occupation"), (r"\bquirks?\b|\bhabits?\b", "quirk"),
    (r"\bchallenges?\b|\bprompts?\b|\bexercises?\b|\bconstraints?\b|\bwriting ideas?\b", "constraint"),
]
_IDEA_VERB = re.compile(r"\b(?:give me|gimme|suggest|generate|i need|i want|need|want|think of|come up with|got any|have any|any|show me|pick|roll|random|surprise me|describe|what should i call|what should we call)\b")
_CULTURES = [
    ("irish", "irish"), ("scottish", "scottish"), ("scots", "scottish"), ("french", "french"), ("german", "german"), ("italian", "italian"), ("spanish", "spanish"),
    ("russian", "russian"), ("polish", "polish"), ("greek", "greek"), ("japanese", "japanese"), ("chinese", "chinese"), ("korean", "korean"), ("indian", "indian"),
    ("arabic", "arabic"), ("nigerian", "west_african"), ("ghanaian", "west_african"), ("african", "west_african"), ("norse", "norse"), ("viking", "norse"),
    ("swedish", "swedish"), ("scandinavian", "swedish"), ("welsh", "welsh"), ("english", "english"), ("british", "english"), ("elvish", "fantasy_elvish"), ("elf", "fantasy_elvish"), ("dwarf", "fantasy_dwarvish"),
    ("dwarvish", "fantasy_dwarvish"), ("orc", "fantasy_harsh"), ("goblin", "fantasy_harsh"),
]


_FILE_NOUN = r"(?:file|files|pdf|pdfs|document|documents|doc|docx|spreadsheet|picture|pictures|photo|photos|image|screenshot|scan|recording|zip|gedcom|ged|family tree|tree|ftm|download|downloads)"
_FILE_REQ = [
    re.compile(rf"\b(?:just|recently|newly|now)\s+(?:downloaded|saved|got|received|scanned|exported|copied)\b"),
    re.compile(rf"\b(?:find|look for|search for|locate|pull up|fetch|import|get|open|add)\b.*\b(?:my |the |a |that |this |new |latest |last )*(?:{_FILE_NOUN})\b.*\b(?:download|downloads|downloaded|computer|desktop|documents folder|folder)\b"),
    re.compile(rf"\b(?:find|look for|search for|locate|pull up|fetch|import)\b\s+(?:me\s+)?(?:my |the |a |that |this |new |newest |latest |last )+(?:{_FILE_NOUN})\b"),
    re.compile(rf"\bdownloaded\b.*\b{_FILE_NOUN}\b|\b{_FILE_NOUN}\b.*\bdownloaded\b"),
]


_NEW_CHAPTER = re.compile(rf"^{_POLITE}(?:start|begin|open|make|create|add|write|give me)(?: me)?(?: a| the)?(?: new| next| fresh)? chapter(?: (?:number )?(\d+))?(?: please| now)?[.!?]?$|^(?:a )?new chapter(?: please)?[.!?]?$")


def new_chapter(text: str) -> int | None:
    """"start chapter 2", "begin a new chapter": -> the chapter number (0 = the next one), else None."""
    m = _NEW_CHAPTER.match(norm(text))
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 0


_NEW_CHAPTER_TITLED = re.compile(r"(?i)^" + _POLITE + r"(?:start|begin|open|make|create|add|write)(?: me)?(?: a| the)?(?: new| next| fresh)? chapter(?: (?:number )?(\d+))?(\s*[,:\u2014-]|\s+and)?\s+(?:and )?((?:call(?:ed)?|title[d]?|name[d]?|label(?:l?ed)?)(?: it| the chapter| this chapter)?(?: to| as)?\s+)?[\"'“]?(.{2,70}?)[\"'”]?[.!?]?$")
_NUMBER_WORDS = "|".join(sorted(list(_UNITS) + list(_TENS), key=len, reverse=True))


def new_chapter_titled(text: str) -> tuple[int, str] | None:
    """"start chapter 3 and call it The Lake", "begin chapter 4, The Round Table" -> (chapter number, 0 = the next one; the title)."""
    raw = re.sub(r"(?i)^(?:\W*(?:um+|uh+|uhm|erm|er)\b[\s,.]*)+", "", text.strip())
    num = None
    if mw := re.search(rf"(?i)\bchapter (?:number )?((?:{_NUMBER_WORDS})(?:[\s-]+(?:{_NUMBER_WORDS}))?)\b", raw):  # "start chapter two, The Lake"
        if (d := re.match(r"\d+$", norm(mw.group(1)))):
            num, raw = int(d.group()), raw[:mw.start(1)] + d.group() + raw[mw.end(1):]
    m = _NEW_CHAPTER_TITLED.match(raw)
    if not m or not (m.group(4) or "").strip():
        return None
    title = m.group(4).strip()
    sep = (m.group(2) or "").strip()
    n0 = int(m.group(1)) if m.group(1) else int(title) if title.isdigit() else (num or 0)
    if not m.group(3) and sep not in (",", ":", "\u2014", "-") or len(title.split()) > 9 \
            or re.fullmatch(rf"(?i)(?:please|now|for me|again|{_NUMBER_WORDS}|\d+)(?: please| now)?", title) \
            or re.match(r"(?i)(?:where|when|about|and|write|it'?s|now|please|then|with|in which|so)\b", title):
        return (n0, "")  # "start chapter two please", "start a new chapter about the voyage": a new chapter, but no name was given
    return (n0, title)


def chapter_story(text: str) -> str:
    """What follows "start chapter 2" when it tells the story or asks for writing: "...where they arrive in Kansas", "...and write the first paragraph"."""
    m = re.search(r"(?i)\bchapter(?: (?:number )?\w+)?[\s,]+(?:and |then )?((?:where|when|about|in which|with|write|draft)\b.*)$", text.strip())
    return m.group(1).strip() if m and len(m.group(1).split()) >= 2 else ""


def _chapter_digits(raw: str) -> str:
    """"chapter four" -> "chapter 4" in the writer's own text (the title keeps its capitals)."""
    def sub(m):
        d = re.match(r"\d+$", norm(m.group(2)))
        return m.group(1) + (d.group() if d else m.group(2))
    return re.sub(rf"(?i)(\bchapter (?:number )?)((?:{_NUMBER_WORDS})(?:[\s-]+(?:{_NUMBER_WORDS}))?)\b", sub, raw)


def book_title(text: str) -> str | None:
    """"Call the book Samuel's War", "the title of the book is …", "I want the book to be called …" -> the title, else None."""
    raw = re.sub(r"(?i)^(?:\W*(?:um+|uh+|er|erm|so|okay|ok|now|and)\b[\s,.]*)+", "", text.strip())
    m = re.match(r"(?i)^(?:(?:please|can you|could you|let's|lets|i want to|i'd like to)\s+)*(?:call|name|title)\s+(?:the|my|this|our)\s+(?:whole\s+)?(?:book|story)\s+(?:to\s+|as\s+)?[\"'\u201c]?(.{2,80}?)[\"'\u201d]?[.!]?$", raw) \
        or re.match(r"(?i)^(?:i want |i'd like |let )?(?:the|my|this) (?:whole )?(?:book|story)(?:'s)? (?:title |name )?(?:is going to be called|to be called|is called|be called|should be called|should be|will be|to be|is)\s+[\"'\u201c]?(.{2,80}?)[\"'\u201d]?[.!]?$", raw) \
        or re.match(r"(?i)^(?:please )?(?:make sure |be sure )?(?:i want |i'd like )?the (?:title|name) of (?:the|my|this) (?:book|story)(?: itself)? (?:is|should be|will be|to be)\s+[\"'\u201c]?(.{2,60}?)[\"'\u201d]?(?:,\s+so\b.*)?[.!]?$", raw)
    if not m:
        m = re.search(r"(?i)\b(?:the book|the title|the front|it)\b[^.?!]{0,40}\?\s+(?:it|the title|the book) should (?:be|say)(?: called)?\s+[\"'\u201c]?([^\"\u201d.!?]{2,60})", raw)  # "What's the book called on the front? It should be Samuel's War."
    if not m:
        return None
    t = m.group(1).strip(" ,.")
    return None if re.match(r"(?i)(?:a |an |about |for |what|something|longer|shorter)", t) else t


def title_chapter(text: str) -> tuple[int | None, str] | None:
    """"call chapter 3 The Lake", "title this chapter 'Dozmary Pool'", "change the name of chapter four to Home Again",
    "I want the chapter title to be Home Again" -> (chapter number or None for the one on screen, the title)."""
    raw = _chapter_digits(re.sub(r"(?i)^(?:\W*(?:um+|uh+|uhm|erm|er)\b[\s,.]*)+", "", text.strip()))
    q = r"[\"'“]?(.{2,70}?)[\"'”]?[.!?]?$"
    m2 = re.match(rf"(?i)^{_POLITE}(?:(?:change|set|make|switch)\s+)?the (?:name|title) (?:of|for) (?:(?:the|this) chapter|chapter (\d+))\s+(?:to(?: be)?|into|as)\s+{q}", raw) \
        or re.match(rf"(?i)^(?:i want|i'd like|i would like|let's have|let's make|make)?\s*(?:the )?(?:(?:this )?chapter(?: (\d+))?(?:'s)? (?:title|name)|(?:title|name) of (?:the |this )?chapter(?: (\d+))?)\s+(?:to be|should be|will be|is|be|to)\s+{q}", raw)
    m2 = m2 or re.match(rf"(?i)^{_POLITE}(?:fix|change|correct)\s+the (?:name|title) of (?:(?:the|this) chapter|chapter (\d+))[,:]?\s+(?:it|that) (?:should|must|has to|ought to|will|is to)(?: just| only)? be\s+{q}", raw)
    if m2:
        g = [x for x in m2.groups()]
        num = next((x for x in g[:-1] if x), None)
        return (int(num) if num else None, g[-1].strip())
    raw = re.sub(r"(?i)(\bchapter \d+[,:]?\s+)(?:(?:um+|uh+|er|erm)\b[,\s]*)+", r"\1", raw)  # "name chapter two, um, Called Up"
    m = re.match(rf"(?i)^{_POLITE}(?:call|title|name|rename|label)(?: the)?( (?:this|current|new|next))? chapter(?: (\d+))?[,:]?(?: to| as)?\s+[\"'“]?(.{{2,70}}?)[\"'”]?[.!?]?$", raw)
    if not m or (not m.group(1) and not m.group(2)):  # "call the chapter house" is not naming a chapter
        return None
    return (int(m.group(2)) if m.group(2) else None, m.group(3).strip())


_RANGE = re.compile(r"\bparagraphs? (\d+) (?:to|through|thru|until|till) (\d+)\b")
_LIST = re.compile(r"\bparagraphs? (\d{1,3}(?:,? (?:and )?\d{1,3}){0,12})(?: (?:of|in) chapter (\d+))?$")
_LASTN = re.compile(r"\bthe (?:last|final) (\d+) paragraphs\b")


def delete_many(text: str) -> tuple[int | None, list[int]] | None:
    """"delete paragraphs 2 to 4", "remove paragraphs 2, 3 and 5 of chapter 3", "delete the last 3 paragraphs" -> (chapter or None, numbers; negative = from the end).
    A remark first is fine: "Those are the same as one and two. Delete paragraphs 3 and 4." """
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    if len(sents) > 1 and (got := _delete_many(sents[-1])) is not None:
        return got
    return _delete_many(text)


def _delete_many(text: str) -> tuple[int | None, list[int]] | None:
    n = norm(text)
    m0 = re.match(rf"^{_POLITE}(?:delete|remove|cut|scrap|erase|get rid of|take out)\b(.+)$", n)
    if not m0 or len(n.split()) > 18:
        return None
    body = m0.group(1)
    ch = parse_ref(body).chapter
    if m := _RANGE.search(body):
        a, b = int(m.group(1)), int(m.group(2))
        return (ch, list(range(a, b + 1))) if a < b and b - a < 30 else None
    if m := _LASTN.search(body):
        k = int(m.group(1))
        return (ch, [-i for i in range(k, 0, -1)]) if 2 <= k <= 20 else None
    if m := _LIST.search(body):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if m.group(2):
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        return (int(m.group(2)) if m.group(2) else ch, nums) if len(nums) >= 2 else None
    return None


_DELETE = re.compile(rf"^{_POLITE}(?:delete|remove|cut|scrap|erase|get rid of|take out)\b(.+)$")
_TAKE_OUT = re.compile(rf"^{_POLITE}(?:take|cut|leave)\b(.+?)\b(?:out|away)(?:,? (?:please|now|for me))?$")
_MOVE_LEAD = re.compile(rf"^{_POLITE}(?:move|put|shift|send)\b(.+?)(?:,? (?:please|now|for me))?$")
_TOPWORD = r"(?:start|beginning|top|front)"


class Move(NamedTuple):
    ref: "Ref"
    chapter: int | None  # None: the chapter the paragraph is already in
    where: object  # "end" | "start" | ("before"|"after", paragraph number)


def delete_request(text: str) -> Ref | None:
    """"delete paragraph 3", "remove the last paragraph of chapter 2" -> which paragraph. Only an explicit paragraph counts."""
    whole = norm(text)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    for cand in ([text] + ([sents[-1]] if len(sents) > 1 else [])):  # "The second paragraph is the same as the first. Take the second one out."
        n = norm(cand)
        if len(n.split()) > 14:
            continue
        m = _DELETE.match(n) or _TAKE_OUT.match(n)
        if not m:
            continue
        what = m.group(1)
        if re.search(r"\bparagraphs?\b", whole) and not re.search(r"\b(?:paragraph|picture|photo|chapter)\b", what):
            what = re.sub(r"\b(?:one|1)\s*$", "paragraph", what.strip())  # "the second one": the paragraph the writer was just talking about
        ref = parse_ref(what)
        if ref.para is not None:
            return ref
    return None


def move_request(text: str) -> "Move | None":
    """"move paragraph 5 to chapter 2", "move the last paragraph to the start of chapter 2", "move paragraph 4 before paragraph 1", "move it to the end"."""
    n = norm(text)
    if re.search(r"\b(?:a|an|another|new|some) (?:picture|photo|photograph|image|illustration|map|drawing)\b", n):
        return None  # "put a picture of the barn in chapter 2 after paragraph 1" asks for a NEW picture; moving needs "the picture"
    m0 = _MOVE_LEAD.match(n)
    if not m0 or len(n.split()) > 18:
        return None
    body = m0.group(1).strip()
    if m := re.search(r"^(.+?)\b(before|after|ahead of|behind)\b (?:the )?(?:paragraph|section|part) (\d+)(?: (?:of|in) chapter (\d+))?$", body):
        ref = parse_ref(m.group(1))
        return Move(ref, int(m.group(4)) if m.group(4) else None, ("before" if m.group(2) in ("before", "ahead of") else "after", int(m.group(3)))) if ref.para is not None else None
    if m := re.search(rf"^(.+?)\bto (?:the )?{_TOPWORD}(?: of)?(?: (?:the |its |this |that )?chapter(?: (\d+))?)?(?: of the chapter)?$", body):
        ref = parse_ref(m.group(1))
        return Move(ref, int(m.group(2)) if m.group(2) else None, "start") if ref.para is not None else None
    if m := re.search(r"^(.+?)\bto (?:the )?end(?: of)?(?: (?:the |its |this |that )?chapter(?: (\d+))?)?(?: of the chapter)?$", body):
        ref = parse_ref(m.group(1))
        return Move(ref, int(m.group(2)) if m.group(2) else None, "end") if ref.para is not None else None
    if m := re.search(r"^(.+?)\b(?:to|into|in) (?:the )?(?:end of |start of |beginning of )?chapter (\d+)(?:,? at the (?:end|start|beginning))?$", body):
        ref = parse_ref(m.group(1))
        return Move(ref, int(m.group(2)), "end") if ref.para is not None else None
    return None


_PICK_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_PICK = re.compile(r"^(?:(?:i |we |let's |lets |i'll |ill )?(?:like|take|want|choose|pick|use|prefer|go with|will take)\s+)?(?:the |number |option |name |idea |no )*(\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?: one| 1| please| is good| sounds good)?$")


def pick_number(text: str) -> int | None:
    """"number 2", "the second one", "I like 3", "use the third": which of the numbered ideas the writer chose."""
    m = _PICK.match(norm(text))
    if not m:
        return None
    w = m.group(1)
    return int(w) if w.isdigit() else _PICK_WORDS[w]


_UNDO = re.compile(r"^(?:please |ok |okay |no |oh |hey )*(?:can you |could you )?(?:undo|revert|put (?:it|that|them) back|take (?:it|that) back|go back to (?:how|what) it was|change it back|bring (?:it|that) back)(?: that| it| the last (?:change|edit|one)| my last (?:change|edit)| please)*[.!?]?$")


def undo_request(text: str) -> bool:
    """"undo that", "put it back", "take that back": the writer wants the last saved change reversed."""
    n = norm(text)
    if not _UNDO.match(n):  # "Hmm. Undo that, I liked cold better.": the first real clause
        clauses = [norm(c) for c in re.split(r"[,.;!?]\s*|\s+(?:because|i liked|i like|i prefer)\b", text.strip())]
        clauses = [c for c in clauses if c and c not in ("oh", "well", "ok", "okay", "no", "so", "hmm", "ah", "oh no", "oh dear", "wait")]
        hit = next((c for c in clauses if _UNDO.match(c) and len(c.split()) <= 6), None)  # "That's not the paragraph at all. Undo that."
        n = hit if hit and len(norm(text).split()) <= 24 else (clauses[0] if clauses else n)
    return len(n.split()) <= 9 and bool(_UNDO.match(n))


_DESCRIBE = re.compile(r"^{p}(?:describe|tell me about|tell me what|what(?:'s| is| does)(?: in)?|what do you see in)(?: (?:it|that|this|the))?(?: (?:picture|image|photo|photograph|illustration|drawing|portrait|painting|snapshot|map))?(?: again| to me| for me| please)*(?: look like| looks like| shows| is| in it| on it)?(?: in (?:more )?detail| more)?(?: again| to me| for me| please)*[.?!]?$".replace("{p}", _POLITE))

_EXPORT_NOUN = re.compile(r"\b(?:book|story|novel|manuscript)\b")


def export_request(text: str) -> str | None:
    """"save my book as a file", "export the book as an e-book", "give me the whole story to print" -> "html" | "epub" | "md", else None."""
    n = norm(text)
    if len(n.split()) <= 16 and re.search(r"\bso (?:i|we) can print\b|\bto print (?:it|out)\b|\bprint (?:it|the book|my book) (?:out|on paper)\b", n) \
            and re.search(r"^(?:\w+ ){0,4}(?:save|make|give|export|send)\b", n) and not re.search(r"\b(?:chapter|paragraph|picture)\b", n):
        return "html"  # "Save it so I can print it": a page that prints, not an e-book
    fmt_word = re.search(r"\b(?:epub|e book|ebook|kindle|e reader|web page|webpage|html|pdf)\b", n)  # "save it as an e-book" needs no "book" noun
    if len(n.split()) > 16 or not (_EXPORT_NOUN.search(n) or fmt_word) or re.search(r"\b(?:chapter|paragraph|wiki|picture|file i|downloaded)\b", n):
        return None
    strong = re.search(r"\b(?:export|save|download|print)\b", " ".join(n.split()[:6]))
    soft = re.match(rf"^{_POLITE}(?:give me|make me|send me|get me|i want|i would like|i'd like|can i have|let me have)\b", n) and \
        re.search(r"\b(?:file|epub|e ?book|pdf|html|web ?page|zip|markdown|copy|whole|entire|complete|finished|to print|to keep)\b", n)
    if not (strong or soft):
        return None
    if re.search(r"\b(?:epub|e book|ebook|kindle|e reader)\b", n):
        return "epub"
    if re.search(r"\b(?:markdown|zip|plain text|text folder)\b", n):
        return "md"
    return "html"


_AUTO_ON = re.compile(r"^{p}(?:just )?(?:save (?:it |them |everything |each (?:one|paragraph) )?(?:as (?:you|we) go|straight away|right away|automatically|without asking)|stop asking(?: me)?(?: to save| to confirm| every time| all the time)?|don'?t (?:keep )?ask(?:ing)?(?: me)?(?: to save| every time| all the time)?|(?:keep )?saving(?: it| them| everything)?(?: as you go)?|auto ?save(?: on)?|no need to ask)(?: please)?[.!?]?$".replace("{p}", _POLITE))
_FROM_NOW = r"(?:,? (?:from now on|every time|each time|always|again|please|for now|for the rest of the book))*"
_AUTO_OFF = re.compile((r"^{p}(?:(?:from now on,? )?(?:ask me (?:first|before (?:saving|you save|you do (?:that|anything))(?: anything| it| them)?|again|every time)|ask before (?:you )?sav(?:e|ing)(?: anything)?|stop saving (?:everything|automatically|by yourself|straight away)|auto ?save off|check with me (?:first|before saving)|wait for my yes|(?:don'?t|do not) save (?:anything |it |them |paragraphs )?(?:until|unless|without) (?:i (?:say|tell you|say yes|say so|ok it|okay it|agree)|my (?:yes|ok|okay|ok it)|asking(?: me)?)|(?:don'?t|do not) save (?:anything |it |them |paragraphs )?by yourself|go back to asking(?: me)?))" + _FROM_NOW + r"[.!?]?$").replace("{p}", _POLITE))


def autosave_mode(text: str) -> str | None:
    """"save as you go" -> "on"; "ask me first" -> "off"; else None."""
    n = norm(text)
    if len(n.split()) > 10:
        return None
    return "on" if _AUTO_ON.match(n) else "off" if _AUTO_OFF.match(n) else None




def describe_request(text: str) -> bool:
    """"describe it to me", "what does it look like": the writer wants the picture just shown put into words (not a new paragraph written)."""
    first = re.split(r"[,;.:!?]| because | as | since | I can| I cannot| I can't| I cant", text.strip(), maxsplit=1, flags=re.I)[0]  # "describe it to me again, I can't see it well": the first clause is the request
    n = norm(first)
    if len(n.split()) <= 9 and _DESCRIBE.match(n) and re.search(r"\b(?:it|that|this|picture|image|photo|photograph|illustration|drawing|portrait|painting|snapshot|map|look like)\b", n):
        return True
    # "describe the railway station picture", "describe the picture in chapter 4 to me": a named picture, never a request for a new paragraph
    return len(n.split()) <= 12 and bool(re.match(rf"^{_POLITE}(?:describe|tell me about|what(?:'s| is) in|what does)\b", n)) and bool(
        re.search(r"\b(?:picture|image|photo|photograph|illustration|portrait|snapshot)\b", n)) and not re.search(r"\b(?:make|write|draw|find|add|put|new|another)\b", n)


_PICTURE_NOUN = r"(?:pictures?|images?|illustrations?|drawings?|paintings?|photos?|photographs?|maps?|sketch(?:es)?|portraits?|cover)"
_PICTURE_VERB = r"(?:make|draw|paint|create|generate|find|get|add|put|use|show|search(?: for)?|look(?: up| for)?|fetch|want|need|like|give)"
_ON_COMPUTER = re.compile(r"\b(?:downloaded|download|downloads|on my computer|my computer|desktop|my folder|inbox|i just (?:saved|got|scanned|took)|from my (?:phone|camera|files))\b")


def picture_request(text: str) -> bool:
    """"make me a picture of the lake", "find a map of Camelot", "I would like an illustration here": a picture for the book, to be made or found on the web.
    Not "I just downloaded a photo" (that is a file on the computer) and not an edit of the words."""
    n = norm(text)
    if _ON_COMPUTER.search(n) or len(n.split()) > 40:
        return False
    return bool(re.search(rf"\b{_PICTURE_VERB}\b.*\b{_PICTURE_NOUN}\b", n) or re.search(rf"\b{_PICTURE_NOUN}\b.*\b(?:of|for|showing|with)\b", n) and re.search(rf"\b(?:chapter|here|book|story|scene)\b|^(?:a|an|the|one|another)\b", n))


def file_request(text: str) -> bool:
    """"I just downloaded a new PDF, can you open it": the writer wants a file found on the computer."""
    n = norm(text)
    return len(n.split()) <= 40 and any(r.search(n) for r in _FILE_REQ)


def open_request(text: str) -> str | None:
    """"open the harbour notes" / "show me my family tree file" -> what to open (to match against the library), else None."""
    m = re.match(r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?(?:open|show me|show|pull up|bring up|go to)\s+(?:up\s+)?(?:the\s+|my\s+)?(.{3,60}?)\s*(?:please)?[.?!]?\s*$", norm(text))
    if not m or re.search(r"\b(chapter|paragraph|scene|page|settings|ideas|library)\b", m.group(1)):
        return None
    return m.group(1)


def ideas_request(text: str) -> dict | None:
    """{"kind", "n", "opts"} when the writer asks for random ideas (names, places, twists…), else None. Questions about names that
    already exist ("what is the name of the inn in chapter 2") are not requests for new ones."""
    n = norm(text)
    if len(n.split()) > 22:  # a long message that merely mentions a title or a name is the writer telling a story, not asking for random ideas
        return None
    if re.match(r"^(?:give me |show me |can i have |let me have )?(?:more|another(?: (?:one|1))?|again|different ones|some more|(?:one|1) more|another please)(?: please)?$", n):
        return {"repeat": True}
    if re.match(r"^(?:what|who|which|where|when|why|how) (?!should (?:i|we) call)", n) or re.search(r"\bchapter \d+", n) or title_chapter(text):
        return None  # "I want the chapter title to be Home Again" names it; it does not ask for ideas
    if re.search(r"\bthe (?:real |actual |right )?(?:name|names|details?|problems?|title|dates?) (?:of|from|in|on|for|with)\b|\b(?:problems?|issues?|mistakes?) (?:with|in) (?:the|my|his|her)\b|\bdetails? (?:from|in) (?:the|his|her|my)\b", n):
        return None  # "the name of Everett's father", "problems with the timeline": facts the book or the documents hold, not random new ideas
    short_form = len(n.split()) <= 5 and bool(re.match(r"^(?:some|a few|a couple of|\d+|an?|another|new|more)\b", n))  # "some names", "ten names"
    if not _IDEA_VERB.search(n) and not short_form:
        return None
    if re.search(r"\b(?:write|wrote|writing|put|type|paint|carve|sign|spell|read)\b", n):  # "write a name on the wall": about the story's text, not a request for ideas
        return None
    if re.search(r"\b(?:dog|dogs|horse|horses|cat|cats|cow|cows|mule|mules|ox|oxen|pony|pet|pets|puppy|calf|goat|pig|hen|rooster)\b", n):
        return None  # a name for the farm dog: the model can do that; the name lists are for people and places
    kind = next((k for rx, k in _IDEA_KINDS if re.search(rx, n)), None)
    if kind is None or is_edit(text):
        return None
    opts: dict = {}
    if kind in ("name", "character"):
        if re.search(r"\b(?:girl|woman|female|lady|she|her|mother|daughter|sister|queen|wife|aunt|heroine|princess|nun|maiden)\b", n):
            opts["gender"] = "female"
        elif re.search(r"\b(?:boy|man|male|gentleman|he|him|father|son|brother|king|husband|uncle|hero|knight|sir|lord|prince|squire|monk|priest)\b", n):
            opts["gender"] = "male"
        for word, culture in _CULTURES:
            if re.search(rf"\b{word}\b", n):
                opts["culture"] = culture
                break
        else:
            if re.search(r"\bfantasy\b", n):
                opts["culture"] = "fantasy_elvish"
    if kind == "place":
        for rx, typ in ((r"\binns?\b|\btaverns?\b|\bpubs?\b", "inn"), (r"\bship\b|\bboat\b", "ship"), (r"\bstreets?\b|\blane\b", "street"),
                        (r"\bforest\b|\bwoods?\b|\bmoor\b|\bmarsh\b|\bwild\b|\bswamp\b", "wild"), (r"\btowns?\b|\bcity\b|\bport\b", "town")):
            if re.search(rx, n):
                opts["type"] = typ
                break
    m = re.search(r"\b(\d{1,2}) (?:\w+ ){0,3}(?:names?|places?|characters?|problems?|twists?|titles?|objects?|ideas?|details?|jobs?|quirks?|challenges?)\b", n)
    count = int(m.group(1)) if m else 3 if re.search(r"\b(?:a few|couple|some)\b", n) else 1 if kind in ("character", "title") and re.search(r"\b(?:a|one)\b", n) else 5
    return {"kind": kind, "n": max(1, min(count, 12)), "opts": opts}


_CREATIVE = re.compile(r"\b(?:ideas?|suggest|suggestions?|what if|stuck|brainstorm|surprise me|something (?:new|different|fresh)|twist|come up with|think of (?:a|an|some|something|another|new|any)|options|alternatives|boring|stale|predictable|cliche|inspire|inspiration)\b")


def is_creative_ask(text: str) -> bool:
    """The writer wants *new material*, not an edit: worth giving the model random sparks."""
    return bool(_CREATIVE.search(norm(text)))


_POS_START = re.compile(r"\b(?:(?:at|to|in) the (?:very )?(?:start|beginning|top|front|opening)(?: of (?:the |this )?(?:chapter|page|story)(?: \d+)?)?|before (?:the )?(?:first|1) (?:paragraph|one)|first thing(?: in the chapter)?|on top|(?:at|as) the first|(?:put|place|move|make|have) (?:it|that|this|them) (?:go )?(?:first|up top|at the top))\b")
_POS_END = re.compile(r"\b(?:(?:at|to|in) the (?:very )?(?:end|bottom|back)(?: of (?:the |this )?(?:chapter|page|story)(?: \d+)?)?|after (?:the )?last (?:paragraph|one)|(?:put|place|move|make|have) (?:it|that|this|them) (?:go )?(?:last|at the bottom))\b")
_POS_AFTER = re.compile(r"\bafter (?:the )?(?:paragraph|para|part) (\d+)\b|\bafter (?:the )?(\d+)(?:st|nd|rd|th)? (?:paragraph|para)\b")
_POS_BEFORE = re.compile(r"\bbefore (?:the )?(?:paragraph|para|part) (\d+)\b|\bbefore (?:the )?(\d+)(?:st|nd|rd|th)? (?:paragraph|para)\b")
_POS_BETWEEN = re.compile(r"\bbetween (?:paragraphs? )?(?:the )?(\d+)(?:st|nd|rd|th)?(?: paragraph)? and (?:paragraph )?(?:the )?(\d+)\b")


_STRICT_START = re.compile(r"\b(?:(?:at|to|in) the (?:very )?(?:start|beginning|top|front|opening) of (?:the |this )?(?:chapter|page|story)(?: \d+)?|(?:at|to|in) the very (?:start|beginning|top|front)\b(?! of the \w+ of)|before (?:the )?(?:first|1) (?:paragraph|one))\b")
_STRICT_END = re.compile(r"\b(?:(?:at|to|in) the (?:very )?(?:end|bottom|back) of (?:the |this )?(?:chapter|page|story)(?: \d+)?|(?:at|to|in) the very (?:end|bottom)\b|after (?:the )?last (?:paragraph|one))\b")
_PLACE_CMD = re.compile(r"^(?:(?:no|yes|ok|okay|yeah|nope)\s+)*(?:please\s+)?(?:(?:put|place|move|make|have|let|add|stick|set|send|i want|i'd like|i would like)\b|(?:at|in|to) the (?:very )?(?:start|beginning|top|front|end|bottom|back)(?: of (?:the |this )?(?:chapter|page|story)(?: \d+)?)?(?: please)?$|(?:first|last)(?: please)?$|after (?:the )?(?:paragraph|para|part)\b|before (?:the )?(?:paragraph|para|part)\b|between\b)")


def placement_command(text: str) -> bool:
    """Is this short utterance an instruction about WHERE something goes ("put it first", "at the end", "after paragraph 2"), rather than story talk?"""
    n = norm(text)
    return len(n.split()) <= 12 and bool(_PLACE_CMD.match(n))


def picture_position(text: str, strict: bool = False) -> int | None:
    """Where in the chapter a picture should go, as the number of paragraphs that come BEFORE it: 0 = the very start, -1 = the end, 3 = after paragraph 3.
    None when the writer said nothing about it (the end of the chapter is the default). "put it at the start of chapter 1", "after paragraph 2", "before paragraph 3"."""
    n = norm(text)
    n = re.sub(rf"\b({'|'.join(sorted((w for w in ORDINALS if w != 'zeroth'), key=len, reverse=True))}) (paragraph|para|part)\b", lambda m: f"{ORDINALS[m.group(1)]} {m.group(2)}", n)  # "after the first paragraph"
    if strict:  # a request for a NEW picture: "of the church at the top of the hill" is the subject, not the place
        if (m := _POS_BETWEEN.search(n)):
            return min(int(m.group(1)), int(m.group(2)))
        if (m := _POS_AFTER.search(n)):
            return int(m.group(1) or m.group(2))
        if (m := _POS_BEFORE.search(n)):
            return max(0, int(m.group(1)) - 1 if m.group(1) else int(m.group(2)) - 1)
        if _STRICT_START.search(n):
            return 0
        if _STRICT_END.search(n):
            return -1
        return None
    if (m := _POS_BETWEEN.search(n)):
        return min(int(m.group(1)), int(m.group(2)))
    if (m := _POS_AFTER.search(n)):
        return int(m.group(1) or m.group(2))
    if (m := _POS_BEFORE.search(n)):
        return max(0, int(m.group(1) or m.group(2)) - 1)
    if _POS_START.search(n):
        return 0
    if _POS_END.search(n):
        return -1
    return None


_SENT_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "last": -1, "final": -1, "closing": -1, "opening": 1}
_SENT_RM = re.compile(r"^{p}(?:take|cut|remove|delete|drop|leave|get rid of)(?: out| away| off)? (?:just |only )?(?:that|this|the|my) (?:(first|1st|second|2nd|third|3rd|fourth|4th|last|final|closing|opening) )?sentence(?: (?:of|from|in|at the end of|at the start of) (.+?))?(?: out| away| off)?(?: for me| please)*$".replace("{p}", _POLITE))


def sentence_removal(text: str) -> tuple["Ref | None", int] | None:
    """"take the last sentence of paragraph 2 out" / "take that sentence out" -> (which paragraph, which sentence: 1-based, -1 = last). The paragraph is None for "that one"."""
    n = norm(text)
    while (m0 := re.match(r"^(?:no|nope|yes|yeah|okay|ok|well|just|so|and|now)\s+", n)):
        n = n[m0.end():]  # "no, just take that sentence out for me"
    m = _SENT_RM.match(n)
    if not m:
        return None
    which = _SENT_ORD.get(m.group(1) or "", -1 if not m.group(1) else 1)
    ref = parse_ref(m.group(2)) if m.group(2) else None
    if m.group(2) and not re.fullmatch(r"(?:that|this|it|the|that paragraph|this paragraph|the paragraph|the draft|the one (?:on the screen|above))", m.group(2).strip()) \
            and ref is not None and ref.para is None and ref.chapter is None:
        return None  # "take the last sentence of the letter and ...": story talk, not a paragraph reference
    if ref is not None and ref.para is None:
        # "of the paragraph in chapter 2" / "of the first paragraph": parse_ref found a chapter or an ordinal; a bare "paragraph" with only a chapter means paragraph 1 is unknown
        ref = ref if (ref.chapter is not None or ref.para is not None) else None
    return ref, which
