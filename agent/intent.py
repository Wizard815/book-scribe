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


_CHANGE = re.compile(r"\b(?:make|change|different|instead|another|darker|lighter|bigger|smaller|more|less|without|but|not|no|don'?t|empty|remove|rename|call it|shorter|longer|except|only|wrong|other)\b")
_SAVE_WORDS = re.compile(r"^(?:(?:please )?(?:go ahead(?: and)?|add|save|put|use|keep|do|write|store|file)(?: it| that| this| them| those| these| all| all of them| (?:that|this|the) (?:one|1)| the (?:paragraph|picture))?(?: in| in there| down| to the book| to chapter \d+| into the book| away)?(?: please| now| for me)?|that one|that'?s? (?:the (?:one|1)|fine|good|right|great|perfect|lovely|nice|mine|it)|(?:it'?s|this is) (?:mine|good|great|fine|right|perfect|lovely|nice|the (?:one|1))|do it|mine|(?:the )?(?:one|1)|thanks|thank you|great|perfect|lovely|good|nice|fine|right|exactly)(?:,? (?:put it in|use it|add it|save it|keep it|go ahead|please|thank you|thanks))*$")


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
    if kind == "yes" and _SAVE_WORDS.match(" ".join(rest)):
        rest = []  # "yes, add it" / "yes save that" / "yes go ahead and put it in": more yes, not a change request
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
    # "as in chapter two", "same lady as chapter six", "like chapter 2": a chapter the writer is pointing back to, not where the work goes
    n_dest = re.sub(r"\b(?:as in|as|like in|like|same as(?: in)?|just as in|back in|earlier in|mentioned in|remember(?: in)?|unlike(?: in)?|similar to|than in|compared (?:to|with)|the way (?:it was )?in|from earlier in|that we (?:wrote|had) in|we had in)\s+chapter\s+\d+", " ", n)
    if m := re.search(r"\bchapter (\d+)\b", n_dest):
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
_DRAFT_LEAD = re.compile(rf"^{_POLITE}(?:write|draft|continue|carry on|keep going|go on|next scene|next paragraph|(?:1|one) more (?:paragraph|bit|part)|another (?:paragraph|bit)|more of the story|the next (?:bit|part)|(?:the )?(?:last|final|closing|opening|first|next) paragraph (?:for|of|in) chapter \d+(?= \w)|add (?:a|another|one more) (?:scene|paragraph)|compose|describe)\b")
_QUESTION = re.compile(r"^(?:so |and |but )?(how|what|why|who|whom|whose|where|should|do you|does|is|are|which|would|am i|did|will it|shall)\b")
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


def draft_first(text: str) -> bool:
    """A request for NEW text ("next paragraph, the lords try three times", "one more paragraph...") must not be taken for an edit of an old paragraph just because
    it mentions a chapter or a verb like "make". It stays an edit when it names a paragraph or says again / instead / rewrite."""
    n = norm(text)
    return is_draft(text) and parse_ref(text).para is None and not re.search(r"\b(?:again|instead|rewrite|redo|change|fix|replace|shorten|lengthen|expand|trim|polish)\b", n)


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
    ("welsh", "welsh"), ("english", "english"), ("british", "english"), ("elvish", "fantasy_elvish"), ("elf", "fantasy_elvish"), ("dwarf", "fantasy_dwarvish"),
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


_NEW_CHAPTER_TITLED = re.compile(r"(?i)^" + _POLITE + r"(?:start|begin|open|make|create|add|write)(?: me)?(?: a| the)?(?: new| next| fresh)? chapter(?: (?:number )?(\d+))?(?:,|:| and)?\s+(?:and )?(?:(?:call(?:ed)?|title[d]?|name[d]?|label(?:l?ed)?)(?: it| the chapter| this chapter)?(?: to| as)?\s+)?[\"'“]?(.{2,70}?)[\"'”]?[.!?]?$")


def new_chapter_titled(text: str) -> tuple[int, str] | None:
    """"start chapter 3 and call it The Lake", "begin chapter 4, The Round Table" -> (chapter number, 0 = the next one; the title)."""
    raw = text.strip()
    m = _NEW_CHAPTER_TITLED.match(raw)
    if not m or not (m.group(2) or "").strip():
        return None
    title = m.group(2).strip()
    if re.fullmatch(r"(?i)(?:please|now|for me|again)", title) or len(title.split()) > 9:
        return None
    return (int(m.group(1)) if m.group(1) else 0, title)


def title_chapter(text: str) -> tuple[int | None, str] | None:
    """"call chapter 3 The Lake", "title this chapter 'Dozmary Pool'" -> (chapter number or None for the one on screen, the title)."""
    m = re.match(rf"(?i)^{_POLITE}(?:call|title|name|rename|label)(?: the)?( (?:this|current|new|next))? chapter(?: (\d+))?[,:]?(?: to| as)?\s+[\"'“]?(.{{2,70}}?)[\"'”]?[.!?]?$", text.strip())
    if not m or (not m.group(1) and not m.group(2)):  # "call the chapter house" is not naming a chapter
        return None
    return (int(m.group(2)) if m.group(2) else None, m.group(3).strip())


_RANGE = re.compile(r"\bparagraphs? (\d+) (?:to|through|thru|until|till) (\d+)\b")
_LIST = re.compile(r"\bparagraphs? ((?:\d+(?:,? (?:and )?)?)+)(?: (?:of|in) chapter (\d+))?$")
_LASTN = re.compile(r"\bthe (?:last|final) (\d+) paragraphs\b")


def delete_many(text: str) -> tuple[int | None, list[int]] | None:
    """"delete paragraphs 2 to 4", "remove paragraphs 2, 3 and 5 of chapter 3", "delete the last 3 paragraphs" -> (chapter or None, numbers; negative = from the end)."""
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
_MOVE_LEAD = re.compile(rf"^{_POLITE}(?:move|put|shift|send)\b(.+?)(?:,? (?:please|now|for me))?$")
_TOPWORD = r"(?:start|beginning|top|front)"


class Move(NamedTuple):
    ref: "Ref"
    chapter: int | None  # None: the chapter the paragraph is already in
    where: object  # "end" | "start" | ("before"|"after", paragraph number)


def delete_request(text: str) -> Ref | None:
    """"delete paragraph 3", "remove the last paragraph of chapter 2" -> which paragraph. Only an explicit paragraph counts."""
    m = _DELETE.match(norm(text))
    if not m or len(norm(text).split()) > 14:
        return None
    ref = parse_ref(m.group(1))
    return ref if ref.para is not None else None


def move_request(text: str) -> "Move | None":
    """"move paragraph 5 to chapter 2", "move the last paragraph to the start of chapter 2", "move paragraph 4 before paragraph 1", "move it to the end"."""
    n = norm(text)
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
    return len(n.split()) <= 9 and bool(_UNDO.match(n))


_DESCRIBE = re.compile(r"^{p}(?:describe|tell me about|tell me what|what(?:'s| is| does)(?: in)?|what do you see in)(?: (?:it|that|this|the))?(?: (?:picture|image|photo|illustration|drawing))?(?: again| to me| for me| please)*(?: look like| looks like| shows| is| in it| on it)?(?: in (?:more )?detail| more)?(?: again| to me| for me| please)*[.?!]?$".replace("{p}", _POLITE))

_EXPORT_NOUN = re.compile(r"\b(?:book|story|novel|manuscript)\b")


def export_request(text: str) -> str | None:
    """"save my book as a file", "export the book as an e-book", "give me the whole story to print" -> "html" | "epub" | "md", else None."""
    n = norm(text)
    if len(n.split()) > 16 or not _EXPORT_NOUN.search(n) or re.search(r"\b(?:chapter|paragraph|wiki|picture|file i|downloaded)\b", n):
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
_AUTO_OFF = re.compile(r"^{p}(?:ask me (?:first|before saving|again|every time)|ask before (?:you )?sav(?:e|ing)|stop saving (?:everything|automatically|by yourself)|auto ?save off|check with me (?:first|before saving))(?: please)?[.!?]?$".replace("{p}", _POLITE))


def autosave_mode(text: str) -> str | None:
    """"save as you go" -> "on"; "ask me first" -> "off"; else None."""
    n = norm(text)
    if len(n.split()) > 10:
        return None
    return "on" if _AUTO_ON.match(n) else "off" if _AUTO_OFF.match(n) else None




def describe_request(text: str) -> bool:
    """"describe it to me", "what does it look like": the writer wants the picture just shown put into words (not a new paragraph written)."""
    n = norm(text)
    return len(n.split()) <= 9 and bool(_DESCRIBE.match(n)) and bool(re.search(r"\b(?:it|that|this|picture|image|photo|illustration|drawing|look like)\b", n))


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
    if re.match(r"^(?:what|who|which|where|when|why|how) (?!should (?:i|we) call)", n) or re.search(r"\bchapter \d+", n):
        return None
    short_form = len(n.split()) <= 5 and bool(re.match(r"^(?:some|a few|a couple of|\d+|an?|another|new|more)\b", n))  # "some names", "ten names"
    if not _IDEA_VERB.search(n) and not short_form:
        return None
    if re.search(r"\b(?:write|wrote|writing|put|type|paint|carve|sign|spell|read)\b", n):  # "write a name on the wall": about the story's text, not a request for ideas
        return None
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
