"""Character voices: who sounds like what, remembered per book.

A TTS backend is just `/v1/audio/speech` with a voice *name* (plus optionally a speed and a style instruction). So a
character's voice is stored as backend-independent **traits** (gender, pitch), a **speed** and a **style**, and each
TTS provider gets its own concrete voice name chosen from that provider's voice list. That matters with fallback tiers:
T1 (local) and T3 (cloud) don't share voice names, but Mr. Bennet should still sound like a low, dry man on both.

    <book>/voices.json   {"profiles": {"narrator": {...}, "Mr. Bennet": {traits, speed, style, pinned, assigned, auto}}}

* `assigned`  provider id -> voice name chosen automatically, then *remembered* so a voice never changes by surprise.
* `pinned`    provider id -> voice name the writer asked for by name; beats `assigned`.
* `auto`      True until the writer has approved the voice in an audition.

Speakers: "assistant" (the agent talking to the writer), "narrator" (prose), or a character from the bible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from intent import norm
from providers import Profile, Registry

ASSISTANT, NARRATOR = "assistant", "narrator"
SPEED_MIN, SPEED_MAX = 0.6, 1.6
GENDERS = ("male", "female", "neutral")
PITCHES = ("low", "mid", "high")


# --- the voices a provider offers ---------------------------------------------------------------------
@dataclass(frozen=True)
class PoolVoice:
    name: str
    tags: frozenset

    @property
    def gender(self) -> str:
        return next((g for g in GENDERS if g in self.tags), "neutral")

    @property
    def pitch(self) -> str:
        return next((p for p in ("low", "high", "mid") if p in self.tags), "mid")


def default_tags(name: str) -> set[str]:
    """Kokoro-style names carry their own tags: af_/am_ (American f/m), bf_/bm_ (British f/m)."""
    m = re.match(r"^[a-z]([fm])_", name)
    return {"female" if m.group(1) == "f" else "male"} if m else set()


def parse_pool(p: Profile) -> list[PoolVoice]:
    """`Profile.voices` is "alloy, onyx:male/low, nova:female" (voices separated by commas, tags by "/") -> voices with tags.
    Falls back to the single default `voice`."""
    out, seen = [], set()
    for item in re.split(r"[,;\n]", p.voices or ""):
        name, _, tags = item.strip().partition(":")
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(PoolVoice(name, frozenset({t.strip().lower() for t in re.split(r"[/|+\s]+", tags) if t.strip()} | default_tags(name))))
    if not out and p.voice:
        out = [PoolVoice(p.voice, frozenset(default_tags(p.voice)))]
    return out


# --- guessing a voice from the bible ---------------------------------------------------------------------
_MALE = set("father husband man boy son brother king prince lord sir mr gentleman uncle grandfather nephew duke baron he his him himself mister".split())
_FEMALE = set("mother wife woman girl daughter sister queen princess lady mrs miss ms she her hers herself aunt grandmother niece duchess madam missus".split())
_YOUNG = set("child kid boy girl young teen teenager youthful little".split())
_OLD = set("old elderly aged grandfather grandmother ancient veteran".split())
_LOW = set("deep gravelly booming baritone bass gruff husky growling".split())
_HIGH = set("shrill squeaky bright piping high-pitched".split())
_STYLE = [  # word in the character's description -> how they speak
    ({"sarcastic", "dry", "wry", "deadpan", "detached", "ironic"}, "dry, deadpan, understated"),
    ({"anxious", "nervous", "fretful", "flustered", "talkative", "chatty"}, "quick, breathy, a little anxious"),
    ({"gentle", "kind", "sweet", "warm", "caring"}, "warm and soft"),
    ({"proud", "cold", "aloof", "formal", "stern", "haughty"}, "cool, clipped, formal"),
    ({"lively", "witty", "playful", "cheerful", "bright", "spirited"}, "bright, lively, playful"),
    ({"angry", "fierce", "furious", "tempestuous"}, "intense and sharp"),
    ({"sad", "weary", "tired", "melancholy", "gloomy"}, "tired and low-energy"),
    ({"pompous", "pedantic", "obsequious"}, "pompous, over-precise"),
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z\-]+", norm(text))


def guess_traits(name: str, state: dict | None = None, body: str = "") -> dict:
    """gender/pitch from whatever the bible says. An explicit `gender:` or `voice_gender:` always wins."""
    state = state or {}
    for k in ("gender", "voice_gender", "sex"):
        v = str(state.get(k, "")).lower()
        if v:
            g = "male" if v[0] == "m" or v in ("man", "he") else "female" if v[0] == "f" or v in ("woman", "she") else "neutral"
            break
    else:
        w = _words(" ".join([name, *map(str, state.values()), body]))
        m = sum(x in _MALE for x in w) + (3 if re.match(r"^\s*(mr|mister|sir|lord)\b", norm(name)) else 0)
        f = sum(x in _FEMALE for x in w) + (3 if re.match(r"^\s*(mrs|miss|ms|lady|madam)\b", norm(name)) else 0)
        g = "male" if m > f else "female" if f > m else "neutral"
    w = set(_words(" ".join([*map(str, state.values()), body])))
    pitch = "low" if (w & _LOW) or (g == "male" and (w & _OLD)) else "high" if (w & _HIGH) or (w & _YOUNG and not w & _OLD and g != "male") else "mid"
    return {"gender": g, "pitch": pitch}


def guess_style(state: dict | None = None, body: str = "") -> str:
    w = set(_words(" ".join([*map(str, (state or {}).values()), body])))
    return ", ".join([s for words, s in _STYLE if words & w][:2])


# --- profile ------------------------------------------------------------------------------------------------
def new_profile(traits: dict, style: str = "", speed: float = 1.0) -> dict:
    return {"traits": dict(traits), "speed": speed, "style": style, "pinned": {}, "assigned": {}, "auto": True}


def _score(v: PoolVoice, traits: dict, used: set[str], speaker: str) -> tuple:
    s = 0.0
    g = traits.get("gender", "neutral")
    s += 3 if v.gender == g else (1 if "neutral" in (v.gender, g) else -3)
    s += 2 if v.pitch == traits.get("pitch", "mid") else 0
    s += 0 if v.name in used else 5  # prefer a voice nobody else has
    tiebreak = int(hashlib.sha1(f"{speaker}|{v.name}".encode()).hexdigest()[:6], 16)  # stable, so re-picking is deterministic
    return (s, tiebreak)


@dataclass
class Resolved:
    voice: str
    speed: float = 1.0
    style: str = ""


class VoiceBook:
    """Loads/saves voices.json per book and resolves (speaker, provider) -> the exact voice to request."""

    def __init__(self, store, reg: Registry):
        self.store, self.reg = store, reg
        self._lock = threading.Lock()
        self._drafts: dict[str, tuple[str, dict]] = {}

    # --- persistence ---------------------------------------------------
    def _path(self, book: str) -> Path:
        return self.store.book_dir(book) / "voices.json"

    def _load(self, book: str) -> dict:
        p = self._path(book)
        try:
            return json.loads(p.read_text()) if p.exists() else {"version": 1, "profiles": {}}
        except (OSError, ValueError):
            return {"version": 1, "profiles": {}}

    def _save(self, book: str, data: dict) -> None:
        self.store.ensure_book(book)
        p = self._path(book)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        tmp.replace(p)

    # --- who is who --------------------------------------------------------
    def canonical(self, book: str, name: str) -> str:
        """Map a spoken/typed name to the bible's spelling ("mister bennet" -> "Mr. Bennet"); narrator/assistant pass through."""
        n = norm(name)
        if n in (NARRATOR, "the narrator"):
            return NARRATOR
        if n in (ASSISTANT, "you", "your", "yours", "assistant", "the assistant"):
            return ASSISTANT
        for c, e in self.store.bible(book).characters().items():
            if n in {norm(c), *(norm(str(a)) for a in e.state.get("aliases") or [])}:
                return c
        return name.strip()

    def _initial(self, book: str, speaker: str) -> dict:
        if speaker == NARRATOR:
            return new_profile({"gender": "neutral", "pitch": "mid"}, "calm, clear storyteller")
        if speaker == ASSISTANT:
            return new_profile({"gender": "neutral", "pitch": "mid"})
        e = self.store.bible(book).characters().get(speaker)
        if e:
            return new_profile(guess_traits(e.name, e.state, e.body), guess_style(e.state, e.body))
        return new_profile(guess_traits(speaker))

    def profile(self, book: str, speaker: str, create: bool = True) -> dict | None:
        """The remembered profile. Created (and saved) the first time a speaker needs a voice."""
        speaker = self.canonical(book, speaker)
        with self._lock:
            data = self._load(book)
            prof = data["profiles"].get(speaker)
            if prof is None and create:
                prof = data["profiles"][speaker] = self._initial(book, speaker)
                self._save(book, data)
            return copy.deepcopy(prof) if prof else None

    def save_profile(self, book: str, speaker: str, prof: dict) -> None:
        speaker = self.canonical(book, speaker)
        with self._lock:
            data = self._load(book)
            data["profiles"][speaker] = prof
            self._save(book, data)

    def forget(self, book: str, speaker: str) -> bool:
        speaker = self.canonical(book, speaker)
        with self._lock:
            data = self._load(book)
            gone = data["profiles"].pop(speaker, None) is not None
            if gone:
                self._save(book, data)
            return gone

    def all_profiles(self, book: str) -> dict:
        return self._load(book)["profiles"]

    # --- choosing a concrete voice -------------------------------------------------
    def _pick(self, book: str, speaker: str, prof: dict, p: Profile, taken: dict[str, str]) -> str:
        pool = parse_pool(p)
        if not pool:
            return p.voice or "alloy"
        used = {v for who, v in taken.items() if who != speaker}
        return max(pool, key=lambda v: _score(v, prof["traits"], used, speaker)).name

    def _voice_for(self, book: str, speaker: str, prof: dict, p: Profile) -> str:
        """The concrete voice for provider `p`, recorded in prof["assigned"] when it had to be chosen."""
        pool = {v.name for v in parse_pool(p)}
        voice = prof["pinned"].get(p.id)
        if voice and (not pool or voice in pool):
            return voice
        voice = prof["assigned"].get(p.id)
        if voice and (not pool or voice in pool):
            return voice
        taken = {who: pr["assigned"].get(p.id) or pr["pinned"].get(p.id) for who, pr in self.all_profiles(book).items()}
        if speaker != ASSISTANT:
            taken[ASSISTANT] = p.voice or ""  # nobody should sound like the assistant either
        prof["assigned"][p.id] = self._pick(book, speaker, prof, p, {k: v for k, v in taken.items() if v})
        return prof["assigned"][p.id]

    def resolve(self, book: str, speaker: str, p: Profile, draft: str | None = None) -> Resolved:
        """The voice/speed/style to send to provider `p`. Remembers an automatic choice so it stays put."""
        speaker = self.canonical(book, speaker)
        if draft and draft in self._drafts and self._drafts[draft][0] == speaker:
            prof = self._drafts[draft][1]  # an audition: choices are remembered for the draft only, so repeat plays sound identical
            return Resolved(self._voice_for(book, speaker, prof, p), float(prof.get("speed", 1.0)), prof.get("style", ""))
        if speaker == ASSISTANT and self.profile(book, ASSISTANT, create=False) is None:
            return Resolved(p.voice or (parse_pool(p)[0].name if parse_pool(p) else "alloy"))  # the default, until the writer changes it
        prof = self.profile(book, speaker)
        had = dict(prof["assigned"])
        voice = self._voice_for(book, speaker, prof, p)
        if prof["assigned"] != had:
            self.save_profile(book, speaker, prof)
        return Resolved(voice, float(prof.get("speed", 1.0)), prof.get("style", ""))

    def freeze(self, book: str, speaker: str, draft: str, **changes) -> dict:
        """Approve an audition: save the draft *with the concrete voice heard on every tier*, so that what the writer
        approved is exactly what is used later (not a fresh automatic pick that might differ)."""
        speaker = self.canonical(book, speaker)
        prof = copy.deepcopy(self._drafts[draft][1]) if draft in self._drafts else self.profile(book, speaker)
        for _, p in self.reg.chain("tts"):
            self._voice_for(book, speaker, prof, p)
        prof.update(changes)
        self.save_profile(book, speaker, prof)
        return prof

    # --- audition drafts ---------------------------------------------------------------
    def new_draft(self, speaker: str, prof: dict) -> str:
        did = uuid.uuid4().hex[:10]
        self._drafts[did] = (speaker, copy.deepcopy(prof))
        while len(self._drafts) > 24:
            self._drafts.pop(next(iter(self._drafts)))
        return did

    def primary(self) -> Profile | None:
        c = self.reg.chain("tts")
        return c[0][1] if c else None


# --- editing a voice with the writer ------------------------------------------------------------------------
_SPEED_RULES = [
    (r"\b(much|way|a lot) (slower|too fast)\b|\bmuch slower\b", 0.7), (r"\b(slower|slow (it )?down|too fast|more slowly|less rushed)\b", 0.85),
    (r"\b(much|way|a lot) (faster|too slow)\b|\bmuch faster\b", 1.3), (r"\b(faster|speed (it )?up|too slow|quicker|more quickly|more urgent)\b", 1.15),
]
_PITCH_RULES = [(r"\b(deeper|lower|bass|baritone|gravelly|huskier|gruffer|more gravel)\b", "low"), (r"\b(higher|brighter|squeakier|lighter|thinner|younger|girlish|boyish)\b", "high"),
                (r"\b(older|more mature|more weathered)\b", "low")]
_GENDER_RULES = [(r"\b(more masculine|manlier|male voice|a man|like a man)\b", "male"), (r"\b(more feminine|female voice|a woman|like a woman)\b", "female")]
_MOODS = "warm cold dry soft harsh gentle stern cheerful sad tired breathy nervous calm intense playful formal casual pompous whispery raspy smooth sarcastic bored angry kind smug haughty gruff friendly flat expressive monotone dramatic sleepy eager".split()


def heuristic_edit(prof: dict, feedback: str) -> tuple[dict, list[str]]:
    """Plain-language adjustments without a model: slower/faster, deeper/higher, and any mood word ("more sarcastic")."""
    n = norm(feedback)
    new, said = copy.deepcopy(prof), []
    for pat, mult in _SPEED_RULES:
        if re.search(pat, n):
            new["speed"] = round(min(SPEED_MAX, max(SPEED_MIN, new["speed"] * mult)), 2)
            said.append("slower" if mult < 1 else "faster")
            break
    for pat, pitch in _PITCH_RULES:
        if re.search(pat, n):
            new["traits"]["pitch"] = pitch
            new["pinned"], new["assigned"] = {}, {}  # a different pitch means a different voice: re-pick
            said.append("deeper" if pitch == "low" else "higher")
            if "younger" in n:
                new["style"] = _merge_style(new["style"], "youthful")
            break
    for pat, g in _GENDER_RULES:
        if re.search(pat, n):
            new["traits"]["gender"] = g
            new["pinned"], new["assigned"] = {}, {}
            said.append(g)
            break
    for w in _MOODS:
        if re.search(rf"\b{w}\b", n) and not re.search(rf"\b(less|not so|not as|too) {w}\b", n):
            new["style"] = _merge_style(new["style"], w)
            said.append(w)
    for m in re.finditer(r"\bless (\w+)", n):
        new["style"] = ", ".join(x for x in new["style"].split(", ") if m.group(1) not in x.lower())
        said.append(f"less {m.group(1)}")
    return new, said


_LEXICON = set("slower faster slow fast quicker deeper lower higher brighter lighter thinner younger older masculine feminine manlier male female "
               "bass baritone gravelly husky gruffer gruff girlish boyish mature weathered squeakier urgent rushed".split()) | set(_MOODS)
_FILLER = set("make makes it its his her their the a an bit little more less much way lot too so very and or but voice voices sound sounds sounding "
              "like be please just can you could would change give to for of with that this slightly somewhat him them he she they is are was i want id "
              "kind sort maybe now then also still really quite abit as not".split())


def residual(feedback: str, ignore=frozenset()) -> list[str]:
    """Content words the keyword rules know nothing about ("professor", "grandmother"): a sign the writer is describing a voice
    in their own words, which is a job for the model rather than the lexicon."""
    return [w for w in norm(feedback).split() if w not in _LEXICON and w not in _FILLER and w not in ignore and not w.isdigit()]


def _merge_style(style: str, add: str) -> str:
    parts = [x for x in style.split(", ") if x]
    if add.lower() not in " ".join(parts).lower():
        parts.append(add)
    return ", ".join(parts[-4:])[:200]


def next_voice(prof: dict, p: Profile, taken: dict[str, str], speaker: str) -> dict:
    """"Try another": the best-scoring voice for the same traits that this speaker isn't already using."""
    new = copy.deepcopy(prof)
    current = new["pinned"].get(p.id) or new["assigned"].get(p.id)
    pool = [v for v in parse_pool(p) if v.name != current]
    if not pool:
        return new
    used = {v for who, v in taken.items() if who != speaker and v}
    best = max(pool, key=lambda v: _score(v, new["traits"], used, speaker + str(len(current or ""))))
    new["pinned"][p.id] = best.name
    new["assigned"].pop(p.id, None)
    return new


def pin_voice(prof: dict, p: Profile, name: str) -> dict | None:
    """The writer named a voice ("use onyx"). Also adopt its gender/pitch so other tiers choose something similar."""
    v = next((x for x in parse_pool(p) if norm(x.name) == norm(name)), None)
    if not v:
        return None
    new = copy.deepcopy(prof)
    new["pinned"][p.id] = v.name
    new["traits"] = {"gender": v.gender, "pitch": v.pitch}
    new["assigned"] = {k: x for k, x in new["assigned"].items() if k == p.id}
    return new


EDIT_PROMPT = """You adjust how a character's voice sounds for a text-to-speech engine, based on the writer's feedback.
Reply with ONLY a JSON object: {"voice": <one of the available voice names, or null to keep the current one>,
"speed": <number 0.6-1.6>, "style": <short delivery instruction, max 15 words, e.g. "dry, deadpan, slightly weary">,
"gender": "male"|"female"|"neutral", "pitch": "low"|"mid"|"high", "say": <one short sentence telling the writer what you changed>}.
Change only what the feedback asks for; keep everything else as it is."""


def parse_llm_edit(raw: str, prof: dict, p: Profile) -> tuple[dict, str] | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    new = copy.deepcopy(prof)
    pool = {x.name: x for x in parse_pool(p)}
    if isinstance(d.get("voice"), str) and d["voice"] in pool:
        v = pool[d["voice"]]
        new["pinned"][p.id] = v.name
        new["assigned"].pop(p.id, None)
        new["traits"] = {"gender": v.gender, "pitch": v.pitch}
    if isinstance(d.get("speed"), (int, float)):
        new["speed"] = round(min(SPEED_MAX, max(SPEED_MIN, float(d["speed"]))), 2)
    if isinstance(d.get("style"), str):
        new["style"] = d["style"].strip()[:200]
    if d.get("gender") in GENDERS and not (isinstance(d.get("voice"), str) and d["voice"] in pool):
        new["traits"]["gender"] = d["gender"]
    if d.get("pitch") in PITCHES and not (isinstance(d.get("voice"), str) and d["voice"] in pool):
        new["traits"]["pitch"] = d["pitch"]
    if new["traits"] != prof["traits"] and not new["pinned"].get(p.id, "") == prof["pinned"].get(p.id, ""):
        pass
    elif new["traits"] != prof["traits"]:
        new["pinned"], new["assigned"] = {}, {}
    return new, str(d.get("say") or "").strip()[:200]


def describe(prof: dict, p: Profile | None, voice: str = "") -> str:
    v = voice or (p and (prof["pinned"].get(p.id) or prof["assigned"].get(p.id))) or "automatic"
    bits = [f"voice {v}", f"{prof['traits'].get('pitch', 'mid')} pitch"]
    if abs(prof.get("speed", 1) - 1) > 0.01:
        bits.append(f"speed {prof['speed']}")
    if prof.get("style"):
        bits.append(prof["style"])
    return ", ".join(bits)
