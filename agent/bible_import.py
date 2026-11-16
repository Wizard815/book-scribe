"""Family tree -> story bible. A preview first, then (after the writer says yes) real wiki pages.

A genealogy file holds thousands of people; a story bible wants a handful of well-described characters. So the import is selective:

  who      the person the story is about (the root) plus ancestors / descendants for N generations, siblings and spouses, nearest first,
           capped (default 150) so the bible stays quick to search. Or "everyone", oldest first, up to the same cap.
  how      each person becomes a character page: facts in the front matter (gender, born, died, birthplace, occupation, family links to the
           others who were chosen) and a plain-words life story in the body, ready for the model to read.
  places   every place that comes up becomes a world page listing who was born, lived or died there.
  time     a family timeline (births, marriages, deaths) is added to timeline.md. Re-importing replaces that section, never duplicates it.
  privacy  living people (no death recorded and probably born in the last 105 years) get a made-up name by default, exact dates are cut to the
           year, and the real name is kept only in the library (aliases.json), never in the bible. "real" keeps true names (the writer's choice).
  safety   an existing page is never overwritten: new facts are appended under a marked section, existing front-matter keys are kept.

Nothing is written by plan(); apply() writes, and the callers only run it after the writer confirms.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from bible import slugify
from genealogy import Tree, _year
from ideas import Ideas

DEFAULT_MAX = 150
HARD_MAX = 400
LIVING_YEARS = 105


@dataclass
class Plan:
    title: str
    slug: str
    privacy: str
    characters: list[dict] = field(default_factory=list)  # {id, name, status, state, body}
    places: list[dict] = field(default_factory=list)  # {name, status, body}
    timeline: list[tuple[int, str]] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # real name -> made-up name (living people only)
    truncated: bool = False

    def summary(self) -> dict:
        new = sum(1 for c in self.characters if c["status"] == "new")
        return {"people": len(self.characters), "new": new, "existing": len(self.characters) - new, "places": len(self.places), "timeline_entries": len(self.timeline),
                "aliased": len(self.aliases), "truncated": self.truncated, "privacy": self.privacy,
                "names": [{"name": c["name"], "status": c["status"]} for c in self.characters[:60]],
                "sample": self.characters[0]["body"][:600] if self.characters else ""}


# ---------------------------------------------------------------------------------------------------- who to include
def _est_year(tree: Tree, pid: str) -> int | None:
    p = tree.people[pid]
    y = _year(p.birth[0])
    if y:
        return int(y)
    kids = [int(x) for x in (_year(tree.people[c].birth[0]) for c in tree.children(pid)) if x]
    if kids:
        return min(kids) - 25
    par = [int(x) for x in (_year(tree.people[c].birth[0]) for c in tree.parents(pid)) if x]
    return max(par) + 28 if par else None


def is_living(tree: Tree, pid: str, now_year: int | None = None) -> bool:
    now_year = now_year or time.gmtime().tm_year
    p = tree.people[pid]
    if p.death[0] or p.death[1] or any(f.startswith(("Buried", "Cause of death")) for f in p.facts):
        return False
    est = _est_year(tree, pid)
    return est is not None and est >= now_year - LIVING_YEARS


def choose(tree: Tree, root: str | None, ancestors: int = 2, descendants: int = 2, siblings: bool = True, max_people: int = DEFAULT_MAX) -> tuple[list[str], bool]:
    """-> (person ids, nearest first; whether the cap cut the list)."""
    max_people = max(1, min(max_people, HARD_MAX))
    if not root or root not in tree.people:
        order = sorted(tree.people, key=lambda i: (_year(tree.people[i].birth[0]) or "9999", tree.people[i].name))
        return order[:max_people], len(order) > max_people
    seen: dict[str, int] = {root: 0}
    for step, limit in ((tree.parents, ancestors), (tree.children, descendants)):
        level = [root]
        for d in range(1, max(0, limit) + 1):
            nxt = []
            for pid in level:
                for x in step(pid):
                    if x not in seen:
                        seen[x] = d
                        nxt.append(x)
            level = nxt
    core = list(seen)
    extra: list[str] = []
    for pid in core:  # spouses (and siblings of the root) come with the people they belong to
        for x in tree.spouses(pid) + (tree.siblings(pid) if siblings and pid == root else []):
            if x not in seen and x not in extra:
                extra.append(x)
    order = sorted(core, key=lambda i: seen[i]) + extra
    return order[:max_people], len(order) > max_people


# ---------------------------------------------------------------------------------------------------- building the pages
def _place_name(place: str) -> str:
    return re.sub(r"\s+", " ", place).strip(" ,")[:60]


def plan(tree: Tree, title: str, slug: str, root: str | None = None, ancestors: int = 2, descendants: int = 2, siblings: bool = True, privacy: str = "alias",
         existing: set[str] | None = None, max_people: int = DEFAULT_MAX, existing_places: set[str] | None = None, ideas: Ideas | None = None) -> Plan:
    existing = {e.lower() for e in (existing or set())}
    existing_places = {e.lower() for e in (existing_places or set())}
    ideas = ideas or Ideas()
    ids, cut = choose(tree, root, ancestors, descendants, siblings, max_people)
    out = Plan(title=title, slug=slug, privacy=privacy, truncated=cut)
    chosen = set(ids)
    taken = set(existing) | {tree.people[i].name.lower() for i in ids}
    alias: dict[str, str] = {}

    def name_of(pid: str) -> str:
        p = tree.people[pid]
        real = p.name or "Unknown relative"
        if privacy == "alias" and is_living(tree, pid):
            if pid not in alias:
                for _ in range(40):
                    new = ideas.name("any", p.sex if p.sex in ("female", "male") else "any")
                    if new.lower() not in taken:
                        break
                taken.add(new.lower())
                alias[pid] = new
                out.aliases[real] = new
            return alias[pid]
        return real

    place_lines: dict[str, list[str]] = {}

    def note_place(place: str, line: str) -> None:
        if place.strip():
            place_lines.setdefault(_place_name(place), []).append(line)

    for pid in ids:
        p = tree.people[pid]
        nm = name_of(pid)
        live = pid in alias
        y_b, y_d = _year(p.birth[0]), _year(p.death[0])
        state: dict = {}
        if p.sex:
            state["gender"] = p.sex
        if live:
            state["based_on"] = "a living relative (name changed)"
            if y_b:
                state["born"] = f"about {y_b[:3]}0s"
        else:
            if p.birth[0] or p.birth[1]:
                state["born"] = p.birth[0] or y_b or ""
            if p.death[0]:
                state["died"] = p.death[0]
        if p.birth[1]:
            state["birthplace"] = _place_name(p.birth[1])
        if p.death[1] and not live:
            state["deathplace"] = _place_name(p.death[1])
        occ = [f.split(": ", 1)[1] for f in p.facts if f.startswith("Occupation:")]
        if occ:
            state["occupation"] = occ[0]
        rel = {"parents": tree.parents(pid), "spouse": tree.spouses(pid), "children": tree.children(pid), "siblings": tree.siblings(pid)}
        for k, v in rel.items():
            names = [name_of(x) for x in v if x in chosen]
            if names:
                state[k] = names
        state["from_family_tree"] = title

        bits = [f"{nm}" + (f" ({y_b or '?'}–{y_d})" if (y_b or y_d) and not live else "") + (" was born" if y_b or p.birth[1] else "")
                + (f" in {_place_name(p.birth[1])}" if p.birth[1] else "") + (f" around {y_b}" if live and y_b else (f" on {p.birth[0]}" if p.birth[0] and not live else "")) + "."]
        if p.death[0] or p.death[1]:
            bits.append(f"Died{(' ' + p.death[0]) if p.death[0] else ''}{(' in ' + _place_name(p.death[1])) if p.death[1] else ''}.")
        for k, label in (("parents", "Parents"), ("spouse", "Married to"), ("children", "Children"), ("siblings", "Brothers and sisters")):
            if k in state:
                bits.append(f"{label}: {', '.join(state[k])}.")
        for f in p.facts[:10]:
            if not f.startswith("Also known as") or not live:
                bits.append(f if f.endswith(".") else f + ".")
        if not live:  # a living person's notes can hold private details: they are not copied
            bits += [f"Note from the family record: {n[:300]}" for n in p.notes[:2]]
        body = " ".join(bits) + f"\n\n*(from the family tree '{title}', id {slug}:{pid.strip('@')})*"
        out.characters.append({"id": pid, "name": nm, "status": "exists" if nm.lower() in existing else ("aliased" if live else "new"), "state": state, "body": body})
        if p.birth[1]:
            note_place(p.birth[1], f"{nm} was born here" + (f" ({y_b})" if y_b and not live else "") + ".")
        if p.death[1] and not live:
            note_place(p.death[1], f"{nm} died here" + (f" ({y_d})" if y_d else "") + ".")
        for f in p.facts:
            if f.startswith(("Lived:", "Immigrated:", "Emigrated:")) and not live:
                kind, _, rest = f.partition(": ")
                place = rest.split(",")[0].strip() if "," not in rest else ", ".join(x.strip() for x in rest.split(",")[:-1]) or rest
                note_place(place, f"{nm}: {f}.")
        if y_b and y_b.isdigit():
            out.timeline.append((int(y_b), f"{nm} was born" + (f" in {_place_name(p.birth[1])}" if p.birth[1] else "") + "."))
        if y_d and y_d.isdigit() and not live:
            out.timeline.append((int(y_d), f"{nm} died" + (f" in {_place_name(p.death[1])}" if p.death[1] else "") + "."))
    for fam in tree.families.values():
        a, b = fam.husb, fam.wife
        if a in chosen and b in chosen and _year(fam.marriage[0]):
            out.timeline.append((int(_year(fam.marriage[0])), f"{name_of(a)} and {name_of(b)} married" + (f" in {_place_name(fam.marriage[1])}" if fam.marriage[1] else "") + "."))
            note_place(fam.marriage[1], f"{name_of(a)} and {name_of(b)} married here ({_year(fam.marriage[0])}).")
    out.timeline.sort()
    for place, lines in sorted(place_lines.items(), key=lambda kv: -len(kv[1]))[:40]:
        out.places.append({"name": place, "status": "exists" if place.lower() in existing_places else "new", "body": f"A place in the family history. " + " ".join(dict.fromkeys(lines))[:1500]})
    return out


# ---------------------------------------------------------------------------------------------------- writing
MARK = "<!-- family-tree:{slug} -->"


def _append_section(old: str, slug: str, text: str) -> str:
    """Add `text` under a marker; if this tree was imported before, that section is replaced (never duplicated)."""
    start, end = MARK.format(slug=slug), MARK.format(slug=slug).replace("<!--", "<!-- /")
    old = re.sub(re.escape(start) + r".*?" + re.escape(end), "", old, flags=re.S).strip()
    return (old + "\n\n" if old else "") + f"{start}\n{text.strip()}\n{end}\n"


def apply(bible, pl: Plan) -> dict:
    """Write the plan into the bible. Existing pages keep everything they had."""
    bible.ensure()
    made = merged = 0
    have = {e.name.lower(): e for e in bible.entries("characters").values()}
    for c in pl.characters:
        old = have.get(c["name"].lower())
        if old:
            state = {k: v for k, v in c["state"].items() if k not in old.state}  # never overwrite what the writer already decided
            body = _append_section(old.body, pl.slug, "From the family tree: " + c["body"])
            bible.upsert("characters", old.name, state, body)
            merged += 1
        else:
            bible.upsert("characters", c["name"], c["state"], c["body"])
            made += 1
    places_made = 0
    wh = {e.name.lower(): e for e in bible.entries("world").values()}
    for pl_ in pl.places:
        old = wh.get(pl_["name"].lower())
        if old:
            bible.upsert("world", old.name, None, _append_section(old.body, pl.slug, pl_["body"]))
        else:
            try:
                bible.upsert("world", pl_["name"], {"from_family_tree": pl.title}, pl_["body"])
                places_made += 1
            except Exception:  # a place name that cannot be a page name is skipped
                continue
    if pl.timeline:
        lines = "\n".join(f"- {y}: {t}" for y, t in pl.timeline)
        old_t = bible.read("timeline") or "# Timeline\n"
        bible.write_doc("timeline", _append_section(old_t, pl.slug, f"## Family timeline ({pl.title})\n{lines}"))
    return {"characters_added": made, "characters_extended": merged, "places_added": places_made, "timeline_entries": len(pl.timeline), "aliased": len(pl.aliases)}


# ---------------------------------------------------------------------------------------------------- shared by the page and the agent
def prepare(store, library, book: str, slug: str, root: str | None = None, ancestors: int = 2, descendants: int = 2, siblings: bool = True,
            privacy: str = "alias", max_people: int = DEFAULT_MAX) -> Plan:
    """Build the plan for a library family tree. Made-up names are seeded from the tree and choices, so the preview and the final write agree."""
    import hashlib

    tree, meta = library.tree(book, slug), library.meta(book, slug)
    if not tree:
        raise KeyError(slug)
    b = store.bible(book)
    seed = int(hashlib.sha1(f"{slug}|{root}|{ancestors}|{descendants}|{privacy}".encode()).hexdigest()[:8], 16)
    return plan(tree, meta["title"], slug, root, max(0, min(ancestors, 8)), max(0, min(descendants, 8)), siblings, "real" if privacy == "real" else "alias",
                set(b.entries("characters")), max_people, set(b.entries("world")), Ideas(seed=seed))


def commit(store, library, book: str, pl: Plan) -> dict:
    res = apply(store.bible(book), pl)
    if pl.aliases:
        library.save_aliases(book, pl.slug, pl.aliases)
    return res
