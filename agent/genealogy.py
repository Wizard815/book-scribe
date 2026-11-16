"""Family trees: read a tree from the formats genealogy programs export, so the agent can answer "who was Anna's mother?".

Supported
  * GEDCOM (.ged, 5.5 / 5.5.1 / 7, also zipped .gdz): the universal exchange format. Family Tree Maker (2017-2026), Ancestry, MyHeritage,
    FamilySearch, RootsMagic, Legacy, Gramps and Reunion can all export it, so this is the dependable route.
  * Gramps XML (.gramps, gzipped or plain).
  * A Family Tree Maker database (.ftm) or backup (.ftmb) is read as a SQLite file. Its internal tables are not a documented format, so
    this is best effort: the rows are turned into readable text (see raw_database_text). Exporting GEDCOM from FTM gives the full structure.

Everything becomes a `Tree` of people and families that can be described in plain words, searched by name, and rendered to text for the
embedding index.
"""
from __future__ import annotations

import gzip
import io
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

MAX_PEOPLE = 20000


@dataclass
class Person:
    id: str
    name: str = ""
    sex: str = ""
    birth: tuple[str, str] = ("", "")  # (date, place)
    death: tuple[str, str] = ("", "")
    famc: list[str] = field(default_factory=list)  # families where this person is a child
    fams: list[str] = field(default_factory=list)  # families where this person is a spouse
    facts: list[str] = field(default_factory=list)  # "Occupation: miller", "Lived: Cork, 1901" ...
    notes: list[str] = field(default_factory=list)


@dataclass
class Family:
    id: str
    husb: str = ""
    wife: str = ""
    children: list[str] = field(default_factory=list)
    marriage: tuple[str, str] = ("", "")


@dataclass
class Tree:
    people: dict[str, Person] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    source: str = ""

    # --- relationships ------------------------------------------------------------------------------------
    def parents(self, pid: str) -> list[str]:
        p = self.people.get(pid)
        out: list[str] = []
        for f in (self.families.get(x) for x in (p.famc if p else [])):
            if f:
                out += [x for x in (f.husb, f.wife) if x and x in self.people and x not in out]
        return out

    def spouses(self, pid: str) -> list[str]:
        p = self.people.get(pid)
        out: list[str] = []
        for f in (self.families.get(x) for x in (p.fams if p else [])):
            if f:
                out += [x for x in (f.husb, f.wife) if x and x != pid and x in self.people and x not in out]
        return out

    def children(self, pid: str) -> list[str]:
        p = self.people.get(pid)
        out: list[str] = []
        for f in (self.families.get(x) for x in (p.fams if p else [])):
            if f:
                out += [c for c in f.children if c in self.people and c not in out]
        return out

    def siblings(self, pid: str) -> list[str]:
        p = self.people.get(pid)
        out: list[str] = []
        for f in (self.families.get(x) for x in (p.famc if p else [])):
            if f:
                out += [c for c in f.children if c != pid and c in self.people and c not in out]
        return out

    # --- finding and describing ----------------------------------------------------------------------------
    def find(self, query: str, limit: int = 5) -> list[Person]:
        q = re.sub(r"[^\w ]", " ", query.lower()).split()
        if not q:
            return []
        scored = []
        for p in self.people.values():
            words = re.sub(r"[^\w ]", " ", p.name.lower()).split()
            hit = sum(1 for w in q if w in words)
            if hit:
                scored.append((hit == len(q), hit, -len(words), p))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [t[3] for t in scored[:limit]]

    def label(self, pid: str) -> str:
        p = self.people.get(pid)
        if not p:
            return ""
        yrs = " – ".join(x for x in (_year(p.birth[0]), _year(p.death[0])) if x)
        return f"{p.name or 'Unknown'}" + (f" ({yrs})" if yrs else "")

    def describe(self, pid: str) -> str:
        p = self.people.get(pid)
        if not p:
            return ""
        lines = [self.label(pid) + (f", {p.sex}" if p.sex else "")]
        if p.birth[0] or p.birth[1]:
            lines.append("Born " + " in ".join(x for x in (p.birth[0], p.birth[1]) if x))
        if p.death[0] or p.death[1]:
            lines.append("Died " + " in ".join(x for x in (p.death[0], p.death[1]) if x))
        rel = [("Parents", self.parents(pid)), ("Spouse", self.spouses(pid)), ("Children", self.children(pid)), ("Siblings", self.siblings(pid))]
        lines += [f"{k}: " + "; ".join(self.label(x) for x in v) for k, v in rel if v]
        for f in p.facts[:12]:
            lines.append(f)
        for n in p.notes[:3]:
            lines.append("Note: " + n[:400])
        return ". ".join(lines)

    def to_text(self) -> str:
        """One paragraph per person, so the text can be split and embedded like any document."""
        order = sorted(self.people.values(), key=lambda p: (_year(p.birth[0]) or "9999", p.name))
        head = f"Family tree with {len(self.people)} people and {len(self.families)} families."
        return head + "\n\n" + "\n\n".join(self.describe(p.id) for p in order)

    def summary(self) -> str:
        years = [int(y) for y in (_year(p.birth[0]) for p in self.people.values()) if y.isdigit()]
        span = f", born between {min(years)} and {max(years)}" if years else ""
        return f"{len(self.people)} people in {len(self.families)} families{span}"


def _year(date: str) -> str:
    m = re.search(r"\b(\d{4})\b", date or "")
    return m.group(1) if m else ""


# ------------------------------------------------------------------------------------------------------------- GEDCOM
_LINE = re.compile(r"^\s*(\d+)\s+(?:(@[^@]+@)\s+)?([A-Za-z_0-9]+)(?:\s(.*))?$")


def decode_text(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # old GEDCOM (ANSI/ANSEL) files: close enough for names and places


class _Node:
    __slots__ = ("tag", "value", "xref", "kids")

    def __init__(self, tag: str, value: str, xref: str = ""):
        self.tag, self.value, self.xref, self.kids = tag, value, xref, []

    def get(self, tag: str) -> "_Node | None":
        return next((k for k in self.kids if k.tag == tag), None)

    def val(self, tag: str) -> str:
        k = self.get(tag)
        return k.value.strip() if k else ""

    def all(self, tag: str) -> list["_Node"]:
        return [k for k in self.kids if k.tag == tag]


def _nodes(text: str) -> list[_Node]:
    roots: list[_Node] = []
    stack: list[tuple[int, _Node]] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _LINE.match(line)
        if not m:
            continue
        level, xref, tag, value = int(m.group(1)), m.group(2) or "", m.group(3).upper(), (m.group(4) or "")
        if tag in ("CONC", "CONT") and stack:
            node = stack[-1][1]
            node.value += ("\n" if tag == "CONT" else "") + value
            continue
        node = _Node(tag, value, xref)
        while stack and stack[-1][0] >= level:
            stack.pop()
        (stack[-1][1].kids if stack else roots).append(node)
        stack.append((level, node))
    return roots


def _clean_name(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.replace("/", " ")).strip()


def _place_date(node: "_Node | None") -> tuple[str, str]:
    return (node.val("DATE"), node.val("PLAC")) if node else ("", "")


FACT_TAGS = {"OCCU": "Occupation", "RESI": "Lived", "EDUC": "Education", "RELI": "Religion", "IMMI": "Immigrated", "EMIG": "Emigrated",
             "NATU": "Naturalised", "CENS": "Census", "BURI": "Buried", "CHR": "Christened", "BAPM": "Baptised", "TITL": "Title",
             "NATI": "Nationality", "PROP": "Property", "MILI": "Military service", "CAUS": "Cause of death", "FACT": "Fact"}


def parse_gedcom(text: str) -> Tree:
    roots = _nodes(text)
    notes = {n.xref: n.value.strip() for n in roots if n.tag in ("NOTE", "SNOTE") and n.xref}
    tree = Tree(source="GEDCOM")
    for n in roots:
        if n.tag == "INDI" and len(tree.people) < MAX_PEOPLE:
            p = Person(id=n.xref)
            names = n.all("NAME")
            if names:
                nm = names[0]
                given, sur = nm.val("GIVN"), nm.val("SURN")
                p.name = _clean_name(f"{given} {sur}") if (given or sur) and not nm.value.strip() else _clean_name(nm.value)
                if not p.name:
                    p.name = _clean_name(f"{given} {sur}")
                for alt in names[1:3]:
                    a = _clean_name(alt.value)
                    if a and a != p.name:
                        p.facts.append(f"Also known as: {a}")
            sex = n.val("SEX")
            p.sex = {"M": "male", "F": "female"}.get(sex.upper(), "")
            p.birth, p.death = _place_date(n.get("BIRT")), _place_date(n.get("DEAT"))
            p.famc = [k.value.strip() for k in n.all("FAMC")]
            p.fams = [k.value.strip() for k in n.all("FAMS")]
            for k in n.kids:
                if k.tag in FACT_TAGS:
                    d, pl = _place_date(k)
                    bits = ", ".join(x for x in (k.value.strip(), pl, d) if x)
                    if bits:
                        p.facts.append(f"{FACT_TAGS[k.tag]}: {bits}")
                elif k.tag in ("NOTE", "SNOTE"):
                    v = k.value.strip()
                    v = notes.get(v, v) if v.startswith("@") else v
                    inner = "\n".join(x.value for x in k.kids if x.tag in ("TEXT",))
                    if v or inner:
                        p.notes.append((v or inner).replace("\n", " "))
            tree.people[p.id] = p
        elif n.tag == "FAM":
            f = Family(id=n.xref, husb=n.val("HUSB"), wife=n.val("WIFE"), children=[k.value.strip() for k in n.all("CHIL")], marriage=_place_date(n.get("MARR")))
            tree.families[f.id] = f
    if not tree.people:
        raise ValueError("no people found in that GEDCOM file")
    return tree


# --------------------------------------------------------------------------------------------------------------- Gramps
def _ns(root: ET.Element) -> str:
    return root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""


def parse_gramps(raw: bytes) -> Tree:
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    root = ET.fromstring(raw)
    ns = _ns(root)
    q = lambda e, t: e.find(f"{ns}{t}")  # noqa: E731
    places = {}
    for pl in root.iter(f"{ns}placeobj"):
        nm = q(pl, "pname")
        places[pl.get("handle")] = nm.get("value") if nm is not None else ""
    events = {}
    for ev in root.iter(f"{ns}event"):
        t, d, pl = q(ev, "type"), q(ev, "dateval"), q(ev, "place")
        date = (d.get("val") if d is not None else "") or ""
        events[ev.get("handle")] = ((t.text or "") if t is not None else "", date, places.get(pl.get("hlink"), "") if pl is not None else "")
    tree = Tree(source="Gramps")
    handle_to_id = {}
    for pe in root.iter(f"{ns}person"):
        pid = pe.get("id") or pe.get("handle")
        handle_to_id[pe.get("handle")] = pid
        p = Person(id=pid)
        nm = q(pe, "name")
        if nm is not None:
            first, sur = q(nm, "first"), q(nm, "surname")
            p.name = " ".join(x.text.strip() for x in (first, sur) if x is not None and x.text)
        g = q(pe, "gender")
        p.sex = {"M": "male", "F": "female"}.get((g.text or "").upper(), "") if g is not None else ""
        for er in pe.findall(f"{ns}eventref"):
            kind, date, place = events.get(er.get("hlink"), ("", "", ""))
            if kind == "Birth":
                p.birth = (date, place)
            elif kind == "Death":
                p.death = (date, place)
            elif kind:
                bits = ", ".join(x for x in (place, date) if x)
                if bits:
                    p.facts.append(f"{kind}: {bits}")
        p.famc = [c.get("hlink") for c in pe.findall(f"{ns}childof")]
        p.fams = [c.get("hlink") for c in pe.findall(f"{ns}parentin")]
        tree.people[p.id] = p
    h2p = handle_to_id
    fam_ids = {}
    for fe in root.iter(f"{ns}family"):
        fam_ids[fe.get("handle")] = fe.get("id") or fe.get("handle")
    for fe in root.iter(f"{ns}family"):
        fid = fam_ids[fe.get("handle")]
        fa, mo = q(fe, "father"), q(fe, "mother")
        f = Family(id=fid, husb=h2p.get(fa.get("hlink"), "") if fa is not None else "", wife=h2p.get(mo.get("hlink"), "") if mo is not None else "",
                   children=[h2p.get(c.get("hlink"), "") for c in fe.findall(f"{ns}childref")])
        for er in fe.findall(f"{ns}eventref"):
            kind, date, place = events.get(er.get("hlink"), ("", "", ""))
            if kind == "Marriage":
                f.marriage = (date, place)
        tree.families[fid] = f
    for p in tree.people.values():  # Gramps links by handle; our Tree links by family id
        p.famc = [fam_ids.get(h, h) for h in p.famc]
        p.fams = [fam_ids.get(h, h) for h in p.fams]
    if not tree.people:
        raise ValueError("no people found in that Gramps file")
    return tree


# ------------------------------------------------------------------------------------------- SQLite family-tree databases
def raw_database_text(raw: bytes, max_rows: int = 4000) -> str:
    """Readable dump of a SQLite file (such as a Family Tree Maker .ftm). Only tables that look like people, places or events are
    worth much, so those are listed first, but every table gets its name and a few rows so the writer can see what is inside."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tree.sqlite"
        path.write_bytes(raw)
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            names = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
            names.sort(key=lambda n: (0 if re.search(r"person|name|fact|event|place|relation|child|marri|famil", n, re.I) else 1, n))
            out = [f"Database with {len(names)} tables: " + ", ".join(names)]
            budget = max_rows
            for n in names:
                cols = [r[1] for r in con.execute(f'pragma table_info("{n}")')]
                total = con.execute(f'select count(*) from "{n}"').fetchone()[0]
                if not total:
                    continue
                out.append(f"\nTable {n} ({total} rows; columns: {', '.join(cols)})")
                for row in con.execute(f'select * from "{n}" limit {min(budget, 500)}'):
                    cells = [f"{c}={str(v)[:80]}" for c, v in zip(cols, row) if v not in (None, "", 0) and not isinstance(v, (bytes, bytearray))]
                    if cells:
                        out.append("; ".join(cells))
                        budget -= 1
                if budget <= 0:
                    out.append("(more rows not shown)")
                    break
            return "\n".join(out)
        finally:
            con.close()


def inner_file(raw: bytes) -> tuple[str, bytes] | None:
    """Family Tree Maker backups (.ftmb) and GEDCOM 7 packages (.gdz) are zip files: find the tree inside."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return None
    names = z.namelist()
    for want in (".ged", ".ftm", ".gramps", ".sqlite", ".db"):
        for n in names:
            if n.lower().endswith(want) and z.getinfo(n).file_size < 400_000_000:
                return n, z.read(n)
    return None


def read_tree(raw: bytes, filename: str) -> tuple[Tree | None, str]:
    """-> (Tree or None, readable text). A Tree when the structure is known (GEDCOM, Gramps); text only for raw databases."""
    low = filename.lower()
    if low.endswith((".ftmb", ".gdz", ".zip")):
        inner = inner_file(raw)
        if not inner:
            raise ValueError("I could not find a family tree inside that package.")
        return read_tree(inner[1], inner[0])
    if low.endswith(".gramps") or raw[:2] == b"\x1f\x8b":
        t = parse_gramps(raw)
        return t, t.to_text()
    if raw[:15] == b"SQLite format 3" or low.endswith((".ftm", ".sqlite", ".db")):
        return None, raw_database_text(raw)
    t = parse_gedcom(decode_text(raw))
    return t, t.to_text()


def looks_like_gedcom(raw: bytes) -> bool:
    head = decode_text(raw[:600]).lstrip()
    return bool(re.match(r"0\s+HEAD\b", head))
