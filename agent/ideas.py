"""Random idea generators: the writer's (and the model's) way out of the first obvious idea.

Language models settle on the same few names ("Elara", "Thorne"), places and plot turns. Real randomness drawn from wide lists breaks that
habit, and costs no model call. Everything here is local, seedable (the same seed gives the same answer) and avoids repeating itself within
a session. The lists live in ideas_data.py: add to them freely.

    generate("name", n=5, culture="irish", gender="female")  -> ["Siobhan Gallagher", ...]
    sparks(3)                                                 -> a few unrelated seeds to push a draft somewhere new
"""
from __future__ import annotations

import random
import re
from collections import deque

import ideas_data as D

CULTURES = list(D.NAMES) + ["fantasy_elvish", "fantasy_dwarvish", "fantasy_harsh", "fantasy_soft"]
GENDERS = ("any", "female", "male", "neutral")
PLACE_TYPES = ("village", "town", "inn", "ship", "street", "wild")

# what the UI offers: key -> (label, options)
KINDS: dict[str, dict] = {
    "name": {"label": "Names", "one": "name", "options": {"culture": ["any"] + CULTURES, "gender": list(GENDERS)}},
    "place": {"label": "Places", "one": "place", "options": {"type": list(PLACE_TYPES)}},
    "character": {"label": "Character sketches", "one": "character", "options": {"culture": ["any"] + CULTURES, "gender": list(GENDERS)}},
    "problem": {"label": "Problems and complications", "one": "problem", "options": {}},
    "twist": {"label": "Twists", "one": "twist", "options": {}},
    "title": {"label": "Book titles", "one": "title", "options": {}},
    "object": {"label": "Objects", "one": "object", "options": {}},
    "sense": {"label": "Sights, sounds and smells", "one": "detail", "options": {}},
    "weather": {"label": "Weather", "one": "weather", "options": {}},
    "occupation": {"label": "Jobs", "one": "job", "options": {}},
    "quirk": {"label": "Habits and quirks", "one": "quirk", "options": {}},
    "constraint": {"label": "Writing challenges", "one": "challenge", "options": {}},
}


class Ideas:
    """Generators with a short memory so a session never hears the same answer twice in a row."""

    def __init__(self, seed: int | None = None, memory: int = 80):
        self.rng = random.Random(seed)
        self.recent: deque[str] = deque(maxlen=memory)

    # --- helpers --------------------------------------------------------------------------------------------------------
    def _fresh(self, make, tries: int = 12) -> str:
        out = make()
        for _ in range(tries):
            if out not in self.recent:
                break
            out = make()
        self.recent.append(out)
        return out

    def _pick(self, seq):
        return self.rng.choice(seq)

    # --- individual kinds -------------------------------------------------------------------------------------------------
    def name(self, culture: str = "any", gender: str = "any") -> str:
        if culture not in CULTURES:
            culture = self._pick(list(D.NAMES))  # "any": a real-world culture, chosen at random
        if culture.startswith("fantasy_"):
            first, last = D.FANTASY[culture.split("_", 1)[1]]
            return self._pick(first) + self._pick(last)
        pool = D.NAMES[culture]
        g = gender if gender in ("female", "male") else self._pick(["female", "male"]) if gender != "neutral" else "neutral"
        given = self._pick(D.NEUTRAL_GIVEN) if g == "neutral" else self._pick(pool[g])
        family = self._pick(pool["family"])
        if g == "female" and culture == "russian" and re.search(r"(ov|ev|in)$", family):
            family += "a"  # Sokolov -> Sokolova
        elif g == "female" and culture == "polish" and family.endswith(("ski", "cki")):
            family = family[:-1] + "a"  # Kowalski -> Kowalska
        return f"{given} {family}"

    def place(self, type: str = "village") -> str:
        if type in D.PLACE_PARTS:
            a, b = D.PLACE_PARTS[type]
            return self._pick(a) + self._pick(b)
        tpl, a, b = D.PLACE_LISTS.get(type, D.PLACE_LISTS["wild"])
        return tpl.format(a=self._pick(a), b=self._pick(b))

    def character(self, culture: str = "any", gender: str = "any") -> str:
        return (f"{self.name(culture, gender)}, {self._pick(D.OCCUPATIONS)}. Quirk: {self._pick(D.QUIRKS)}. Flaw: {self._pick(D.FLAWS)}. "
                f"Wants {self._pick(D.WANTS)}. Secret: {self._pick(D.SECRETS)}.")

    def title(self) -> str:
        pat = self._pick(D.TITLE_PATTERNS)
        return re.sub(r"\{(\w)\}", lambda m: self._pick(D.TITLE_WORDS[m.group(1)]), pat)

    def simple(self, kind: str) -> str:
        return self._pick({"problem": D.PROBLEMS, "twist": D.TWISTS, "object": D.OBJECTS, "sense": D.SENSES, "weather": D.WEATHER,
                           "occupation": D.OCCUPATIONS, "quirk": D.QUIRKS, "constraint": D.CONSTRAINTS}[kind])

    # --- public ----------------------------------------------------------------------------------------------------------------
    def generate(self, kind: str, n: int = 5, **opts) -> list[str]:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose from {', '.join(KINDS)}")
        n = max(1, min(int(n), 12))
        allowed = KINDS[kind]["options"]
        opts = {k: v for k, v in opts.items() if k in allowed and v in allowed[k]}
        make = {
            "name": lambda: self.name(**opts), "place": lambda: self.place(**opts), "character": lambda: self.character(**opts),
            "title": self.title,
        }.get(kind, lambda: self.simple(kind))
        return [self._fresh(make) for _ in range(n)]

    def sparks(self, n: int = 3) -> list[str]:
        """A few unrelated seeds ("a character named…", "a sound…") to nudge a draft away from the obvious."""
        kinds = self.rng.sample(D.SPARK_KINDS, k=min(n, len(D.SPARK_KINDS)))
        label = {"name": "a character called", "object": "an object:", "sense": "a detail:", "problem": "a complication:", "constraint": "a way to write it:",
                 "weather": "the weather:", "quirk": "a quirk for someone:"}
        return [f"{label[k]} {self.generate(k, 1)[0]}" for k in kinds]


def describe_kinds() -> list[dict]:
    return [{"kind": k, "label": v["label"], "one": v["one"], "options": v["options"]} for k, v in KINDS.items()]
