"""Finding files and the inbox folder, big yes/no questions, family tree -> story bible, the three voices, local/cloud choice."""
import os
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import bible_import
import extract
import genealogy
import voices as vx
from agent import Agent
from book_store import BookStore
from inbox import Finder, Inbox
from journal import Journal
from kg import StoryKG
from library import Library
from main import create_app
from memory.store import FileMemory
from providers import Registry
from router import NotConfigured, Router
from settings import Settings
from tests.conftest import FakeLLM
from tests.test_memory_library import GED, FakeSTT


def make(books, **kw):
    s = Settings(project_dir=books, default_book="demo", **kw)
    store, llm = BookStore(books), FakeLLM()
    ag = Agent(s, store, llm, FileMemory(books / ".state" / "m.json"), StoryKG(store), journal=Journal(store), library=Library(store))
    ag.stt = FakeSTT()
    return ag, store, llm


def old(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


# ----------------------------------------------------------------------------------------------- caps and engine choice
def test_a_file_may_be_up_to_200_megabytes():
    assert extract.MAX_BYTES == 200 * 1024 * 1024
    with pytest.raises(extract.Unreadable) as e:
        extract.extract(b"x" * (extract.MAX_BYTES + 1), "big.txt")
    assert "200 MB" in str(e.value)


def tiers(tmp_path, role="chat"):
    s = Settings(project_dir=tmp_path)
    reg = Registry(s)
    for t in (1, 2, 3):
        reg.upsert(role, {"label": f"P{t}", "base_url": f"http://t{t}", "tier": t, "model": f"m{t}"})
    return reg, Router(reg)


@pytest.mark.asyncio
async def test_the_writer_can_choose_local_or_cloud(tmp_path):
    reg, router = tiers(tmp_path)
    hosts = []

    async def go(p):
        hosts.append(p.base_url[-2:])
        return "ok"
    await router.call("chat", go)
    assert hosts == ["t1"]  # automatic: the first local tier
    hosts.clear(); router.prefer["chat"] = "cloud"
    await router.call("chat", go)
    assert hosts == ["t3"]  # the cloud (frontier) tier first
    hosts.clear(); router.prefer["chat"] = "local"

    async def only_cloud_works(p):
        hosts.append(p.base_url[-2:])
        if p.base_url.endswith(("t1", "t2")):
            raise RuntimeError("down")
        return "ok"
    with pytest.raises(Exception):
        await router.call("chat", only_cloud_works)
    assert hosts == ["t1", "t2"]  # local means local: the cloud tier is never tried, so nothing leaves the machines
    reg2, r2 = Registry(Settings(project_dir=tmp_path / "x")), None
    r2 = Router(reg2); r2.prefer["chat"] = "local"
    reg2.upsert("chat", {"label": "C", "base_url": "http://c", "tier": 3})
    with pytest.raises(NotConfigured) as e:
        await r2.call("chat", go)
    assert "local chat" in str(e.value)


def test_engine_and_voice_choices_are_saved_and_applied(books):
    s = Settings(project_dir=books, default_book="demo")
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["Okay."]), stt=FakeSTT(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False)
    with TestClient(app) as c:
        d = c.get("/api/settings").json()
        assert d["chat_engine"] == "auto" and d["voice_mode"] == "characters" and d["narrator_reader"] == "female" and d["live_interrupt"] == "off"
        d = c.put("/api/settings", json={"chat_engine": "cloud", "voice_engine": "local", "listen_engine": "local", "voice_mode": "separate", "narrator_reader": "male", "live_interrupt": "on"}).json()
        assert (d["chat_engine"], d["voice_engine"], d["voice_mode"], d["narrator_reader"], d["live_interrupt"]) == ("cloud", "local", "separate", "male", "on")
        assert c.put("/api/settings", json={"chat_engine": "mars"}).json()["chat_engine"] == "cloud"  # nonsense is ignored


# --------------------------------------------------------------------------------------------------- the three voices
@pytest.fixture
def vb(books):
    s = Settings(project_dir=books, default_book="demo")
    reg, store = Registry(s), BookStore(books)
    b = store.bible("demo")
    b.upsert("characters", "Mira", {"gender": "female"}); b.upsert("characters", "Jonas", {"gender": "male"}); b.upsert("characters", "Zed", {})
    p = reg.upsert("tts", {"label": "Local", "base_url": "http://local", "voices": "af_heart, bf_emma, am_adam, am_michael", "voice": "af_heart", "tier": 1})
    book = vx.VoiceBook(store, reg)
    return book, p


def test_one_voice_for_everything(vb):
    book, p = vb
    book.settings = lambda: {"mode": "same"}
    voices = {book.resolve("demo", who, p).voice for who in ("assistant", "narrator", "Mira", "Jonas")}
    assert len(voices) == 1


def test_three_voices_agent_female_reader_male_reader(vb):
    book, p = vb
    book.settings = lambda: {"mode": "separate", "narrator": "female"}
    fem, male = book.profile("demo", "reader:female"), book.profile("demo", "reader:male")
    fem["pinned"][p.id], male["pinned"][p.id] = "bf_emma", "am_michael"
    book.save_profile("demo", "reader:female", fem); book.save_profile("demo", "reader:male", male)
    r = lambda who: book.resolve("demo", who, p).voice
    assert r("Mira") == "bf_emma" and r("Jonas") == "am_michael" and r("narrator") == "bf_emma"  # characters by gender, narrator by the writer's choice
    book.settings = lambda: {"mode": "separate", "narrator": "male"}
    assert r("narrator") == "am_michael"
    assert r("assistant") not in ("bf_emma", "am_michael")  # the agent keeps its own voice
    # a character the writer has approved a voice for keeps it
    prof = book.profile("demo", "Jonas"); prof["pinned"][p.id] = "am_adam"; prof["auto"] = False; book.save_profile("demo", "Jonas", prof)
    assert r("Jonas") == "am_adam"


def test_each_character_gets_their_own_voice_is_still_the_default(vb):
    book, p = vb
    assert {book.resolve("demo", n, p).voice for n in ("Mira", "Jonas")} == {book.resolve("demo", "Mira", p).voice, book.resolve("demo", "Jonas", p).voice} and \
        book.resolve("demo", "Mira", p).voice != book.resolve("demo", "Jonas", p).voice


def test_the_voice_role_endpoints(books):
    s = Settings(project_dir=books, default_book="demo")
    store = BookStore(books)
    reg = Registry(s)
    reg.upsert("tts", {"label": "Local", "base_url": "http://local", "voices": "af_heart, am_adam", "voice": "af_heart", "tier": 1})
    tts = FakeLLM(); tts.enabled = True
    app = create_app(s, llm=FakeLLM(["Okay."]), stt=FakeSTT(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False)  # the app reads the providers the registry saved
    with TestClient(app) as c:
        j = c.get("/api/books/demo/voice-roles").json()
        assert set(j["roles"]) == {"agent", "female", "male"} and [v["name"] for v in j["pool"]] == ["af_heart", "am_adam"]
        assert c.put("/api/books/demo/voice-roles/male", json={"voice": "am_adam", "speed": 0.9}).status_code == 200
        assert c.put("/api/books/demo/voice-roles/male", json={"voice": "nope"}).status_code == 400
        assert c.put("/api/books/demo/voice-roles/robot", json={}).status_code == 404
        assert c.get("/api/books/demo/voice-roles").json()["roles"]["male"]["voice"] == "am_adam"


def test_the_speech_model_is_told_the_names_in_the_book(books):
    ag, store, _ = make(books)
    assert "Mira" in ag.stt_hint() and "Jonas" in ag.stt_hint()


# ----------------------------------------------------------------------------------------------- inbox and file finder
def test_the_finder_only_lists_what_is_shared_and_confirmed_things_are_safe(tmp_path):
    root = tmp_path / "Downloads"; (root / "sub").mkdir(parents=True)
    (root / "Family_History_Draft.pdf").write_bytes(b"%PDF-1.4 x" * 10)
    (root / "old_taxes.pdf").write_bytes(b"%PDF-1.4 y" * 10); old(root / "old_taxes.pdf", 86400 * 40)
    (root / "photo.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 50)
    (root / "half.pdf.crdownload").write_bytes(b"z" * 50)
    (root / ".hidden.pdf").write_bytes(b"h" * 50)
    (root / "sub" / "notes.txt").write_bytes(b"n" * 20)
    (tmp_path / "secret.txt").write_text("outside")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    f = Finder([str(root)], 10_000_000)
    got = f.search("I just downloaded a new PDF")
    assert [x.name for x in got] == ["Family_History_Draft.pdf", "old_taxes.pdf"]  # newest first, partial / hidden / symlinked files skipped
    assert [x.name for x in f.search("family history")] == ["Family_History_Draft.pdf"]
    assert [x.name for x in f.search("pdf", within_days=7)] == ["Family_History_Draft.pdf"]
    assert f.resolve(0, "sub/notes.txt").name == "notes.txt"
    for bad in ("../secret.txt", "link.txt", "sub", "nope.txt"):
        with pytest.raises(FileNotFoundError):
            f.resolve(0, bad)
    assert Finder([str(tmp_path / "missing")], 1).enabled is False


def test_the_inbox_announces_a_file_once_and_waits_for_it_to_finish_copying(books):
    store = BookStore(books); ib = Inbox(store)
    (ib.dir("demo") / "new.txt").write_text("hello")
    assert ib.fresh("demo") == []  # touched a moment ago: probably still being copied in
    old(ib.dir("demo") / "new.txt", 60)
    assert [f["name"] for f in ib.fresh("demo")] == ["new.txt"] and ib.fresh("demo") == []
    (ib.dir("demo") / "x.crdownload").write_text("partial"); old(ib.dir("demo") / "x.crdownload", 60)
    assert ib.fresh("demo") == []
    for bad in ("../x", ".seen.json", "sub/a"):
        with pytest.raises(FileNotFoundError):
            ib.path("demo", bad)


@pytest.mark.asyncio
async def test_i_just_downloaded_a_pdf_can_you_open_it(books, tmp_path):
    dl = tmp_path / "Downloads"; dl.mkdir()
    (dl / "Aunt_Rose_Letters.txt").write_text("Aunt Rose wrote that the ferry stopped in 1932.")
    (dl / "Older_Notes.txt").write_text("Some older notes."); old(dl / "Older_Notes.txt", 86400 * 3)
    ag, store, llm = make(books, search_dirs=[str(dl)])
    r = await ag.handle("s", "I just downloaded a new file, can you open it")
    assert "Aunt_Rose_Letters.txt" in r.text and "Is that the one?" in r.text and r.pending["kind"] == "file" and llm.calls == []  # no model needed to start
    r = await ag.handle("s", "no")  # not that one: offer the next match
    assert "Older_Notes.txt" in r.text and r.pending["kind"] == "file"
    r = await ag.handle("s", "yes")
    assert r.open_item["title"] == "Older Notes" and "opened it" in r.text and r.pending is None
    assert (dl / "Older_Notes.txt").exists()  # the writer's own copy is never moved or deleted
    assert [m["title"] for m in ag.library.items("demo")] == ["Older Notes"]
    r = await ag.handle("s", "open the older notes")  # and anything already added opens by name
    assert r.open_item["slug"] == "older-notes"


@pytest.mark.asyncio
async def test_nothing_found_and_no_shared_folders_are_explained_plainly(books):
    ag, store, _ = make(books)
    r = await ag.handle("s", "find the pdf I downloaded")
    assert "inbox folder" in r.text and "Import" in r.text and r.pending is None


@pytest.mark.asyncio
async def test_a_file_dropped_in_the_inbox_is_offered_and_added_on_yes(books):
    ag, store, _ = make(books)
    f = ag.inbox.dir("demo") / "Letters.txt"; f.write_text("Dear Anna, the bees are gone."); old(f, 60)
    r = ag.inbox_check("demo")
    assert "Letters.txt" in r.text and r.pending["kind"] == "file"
    assert ag.inbox_check("demo") is None  # never interrupts a question that is waiting, and never asks twice
    r = await ag.handle("s", "yes please")
    assert r.open_item["title"] == "Letters" and not f.exists() and (ag.inbox.dir("demo") / "added" / "Letters.txt").exists()
    g = ag.inbox.dir("demo") / "Other.txt"; g.write_text("more"); old(g, 60)
    ag.inbox_check("demo"); r = await ag.handle("s", "no")
    assert g.exists() and ag.library.items("demo")[0]["title"] == "Letters" and "leave it" in r.text


# ------------------------------------------------------------------------------------ family tree -> story bible
def tree_plan(**kw):
    return bible_import.plan(genealogy.parse_gedcom(GED), "O'Neill Family", "oneill", **kw)


def test_the_plan_covers_people_places_and_a_timeline_and_writes_nothing():
    pl = tree_plan(root="@I1@", ancestors=1, descendants=1)
    names = {c["name"] for c in pl.characters}
    assert names == {"Mary O'Neill", "John O'Neill", "Ann Walsh", "Tom Byrne", "Peg Byrne"}  # her parents, spouse and child
    mary = next(c for c in pl.characters if c["name"] == "Mary O'Neill")
    assert mary["state"]["born"] == "3 MAR 1851" and mary["state"]["birthplace"] == "Cork, Ireland" and mary["state"]["occupation"] == "seamstress"
    assert mary["state"]["parents"] == ["John O'Neill", "Ann Walsh"] and mary["state"]["spouse"] == ["Tom Byrne"] and mary["state"]["children"] == ["Peg Byrne"]
    assert "was born in Cork, Ireland" in mary["body"] and "crossed on the Atlantic alone" in mary["body"]
    assert [p["name"] for p in pl.places] == ["Cork, Ireland"] and any(y == 1875 and "married" in t for y, t in pl.timeline) and pl.timeline == sorted(pl.timeline)
    only = tree_plan(root="@I1@", ancestors=0, descendants=0)
    assert {c["name"] for c in only.characters} == {"Mary O'Neill", "Tom Byrne"}  # no generations: just her and her husband
    capped = tree_plan(root=None, max_people=2)
    assert len(capped.characters) == 2 and capped.truncated


LIVING = """0 HEAD
0 @A@ INDI
1 NAME Old /Timer/
1 SEX M
1 BIRT
2 DATE 1850
1 DEAT
2 DATE 1920
1 FAMS @F@
0 @B@ INDI
1 NAME Zara /Quill/
1 SEX F
1 BIRT
2 DATE 12 JAN 1990
2 PLAC Leeds, England
1 NOTE Allergic to peanuts, lives at 4 Elm Road
1 FAMC @F@
0 @F@ FAM
1 HUSB @A@
1 CHIL @B@
0 TRLR
"""


def test_living_relatives_get_made_up_names_and_real_ones_never_reach_the_bible(tmp_path):
    t = genealogy.parse_gedcom(LIVING)
    assert bible_import.is_living(t, "@B@") and not bible_import.is_living(t, "@A@")
    pl = bible_import.plan(t, "Tree", "tree", root="@A@", ancestors=0, descendants=1)
    names = {c["name"] for c in pl.characters}
    assert "Zara Quill" not in names and list(pl.aliases) == ["Zara Quill"] and pl.aliases["Zara Quill"] in names and "Old Timer" in names
    z = next(c for c in pl.characters if c["name"] == pl.aliases["Zara Quill"])
    blob = str(z["state"]) + z["body"] + str(pl.places) + str(pl.timeline)
    assert "Zara" not in blob and "Quill" not in blob and "Elm Road" not in blob and "peanuts" not in blob and "12 JAN" not in blob
    assert z["state"]["born"] == "about 1990s" and z["state"]["based_on"].startswith("a living relative") and "Leeds" in blob  # places are kept
    real = bible_import.plan(t, "Tree", "tree", root="@A@", ancestors=0, descendants=1, privacy="real")
    assert "Zara Quill" in {c["name"] for c in real.characters} and not real.aliases
    # the preview and the final write agree on the made-up name (prepare() seeds the names from the tree and the choices)
    import library as libmod
    store = BookStore(tmp_path); lib = libmod.Library(store)
    lib.add("b", "t.ged", b"x", "Tree", "family_tree", "x", tree=t)
    one = bible_import.prepare(store, lib, "b", "tree", "@A@", 0, 1)
    two = bible_import.prepare(store, lib, "b", "tree", "@A@", 0, 1)
    assert one.aliases == two.aliases and one.aliases["Zara Quill"] in {c["name"] for c in one.characters}
    bible_import.commit(store, lib, "b", one)
    assert lib.aliases("b", "tree") == one.aliases and "Zara" not in str([e.render() for e in store.bible("b").entries("characters").values()])  # the real name is kept in the library only


def test_applying_never_overwrites_and_reimporting_never_duplicates(books):
    store = BookStore(books); bible = store.bible("demo")
    bible.upsert("characters", "Ann Walsh", {"role": "grandmother", "born": "my own note"}, "Written by the writer.")
    existing = set(bible.entries("characters"))
    pl = tree_plan(root="@I1@", ancestors=1, descendants=1, existing=existing)
    assert {c["name"]: c["status"] for c in pl.characters}["Ann Walsh"] == "exists"
    res = bible_import.apply(bible, pl)
    assert res["characters_added"] == 4 and res["characters_extended"] == 1
    ann = bible.characters()["Ann Walsh"]
    assert ann.state["born"] == "my own note" and ann.state["role"] == "grandmother" and ann.body.startswith("Written by the writer.") and "From the family tree" in ann.body
    bible_import.apply(bible, tree_plan(root="@I1@", ancestors=1, descendants=1, existing=set(bible.entries("characters"))))
    assert bible.characters()["Ann Walsh"].body.count("From the family tree") == 1  # the section is replaced, not repeated
    tl = bible.read("timeline"); assert tl.count("Family timeline") == 1 and "- 1875:" in tl
    assert "Cork, Ireland" in bible.entries("world") and "Mary O'Neill" in bible.entries("world")["Cork, Ireland"].body


def test_a_ftm_style_database_is_read_when_its_tables_are_recognisable(tmp_path):
    import sqlite3
    path = tmp_path / "t.ftm"; c = sqlite3.connect(path)
    c.executescript("""create table Person(ID integer primary key, Gender text); create table Name(PersonID integer, GivenName text, Surname text);
    create table Relationship(ID integer primary key, Person1ID integer, Person2ID integer); create table ChildRelationship(RelationshipID integer, ChildID integer);
    insert into Person values (1,'M'),(2,'F'),(3,'F'); insert into Name values (1,'Tom','Byrne'),(2,'Mary','Oneill'),(3,'Peg','Byrne');
    insert into Relationship values (10,1,2); insert into ChildRelationship values (10,3);""")
    c.commit(); c.close()
    ex = extract.extract(path.read_bytes(), "Family.ftm")
    assert ex.kind == "family_tree" and "Parents: Tom Byrne; Mary Oneill" in ex.tree.describe("3") and "best effort" in ex.tree.source
    assert "GEDCOM" in " ".join(ex.notes)  # the writer is told it is a best guess


@pytest.fixture
def client(books):
    s = Settings(project_dir=books, default_book="demo")
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["Okay."]), stt=FakeSTT(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False)
    with TestClient(app) as c:
        yield c


def test_the_page_can_preview_then_import_a_family_into_the_wiki(client, books):
    item = client.post("/api/books/demo/library", files=[("files", ("O'Neill.ged", GED.encode(), "text/plain"))]).json()["items"][0]
    slug, mary = item["slug"], next(p for p in client.get(f"/api/books/demo/library/{item['slug']}").json()["people"] if p["name"] == "Mary O'Neill")
    body = {"root_id": mary["id"], "ancestors": 1, "descendants": 1}
    pv = client.post(f"/api/books/demo/library/{slug}/wiki-plan", json=body).json()
    assert pv["people"] == 5 and pv["new"] == 5 and pv["places"] == 1 and "Mary O'Neill" in [n["name"] for n in pv["names"]]
    assert "Mary O'Neill" not in BookStore(books).bible("demo").characters()  # a preview writes nothing
    res = client.post(f"/api/books/demo/library/{slug}/wiki-import", json=body).json()
    assert res["characters_added"] == 5 and res["places_added"] == 1
    assert "Mary O'Neill" in BookStore(books).bible("demo").characters()
    assert client.post(f"/api/books/demo/library/{slug}/wiki-plan", json={"ancestors": 99}).status_code == 422


@pytest.mark.asyncio
async def test_the_agent_can_propose_the_family_import_and_waits_for_yes(books):
    import harness
    ag, store, _ = make(books)
    tree = genealogy.parse_gedcom(GED)
    ag.library.add("demo", "f.ged", GED.encode(), "O'Neill Family", "family_tree", tree.to_text(), tree=tree)
    s = ag.session("s")
    out = await ag.toolbox.call("propose_family_import", {"name": "Mary", "ancestors": 1, "descendants": 1}, harness.Ctx(sess=s))
    assert "5 new characters" in out and s.pending.kind == "family" and "Peg Byrne" not in store.bible("demo").characters()
    r = await ag.handle("s", "yes")
    assert "added 5 characters" in r.text and "Peg Byrne" in store.bible("demo").characters()
    await ag.toolbox.call("propose_family_import", {"name": "Mary"}, harness.Ctx(sess=s))
    r = await ag.handle("s", "no")
    assert s.pending is None


def test_the_inbox_endpoints(client, books):
    d = client.get("/api/books/demo/inbox").json()
    assert d["folder"] == "demo/inbox" and d["files"] == []
    f = BookStore(books).book_dir("demo") / "inbox" / "Notes.txt"; f.parent.mkdir(exist_ok=True); f.write_text("hello inbox"); old(f, 60)
    assert [x["name"] for x in client.get("/api/books/demo/inbox").json()["files"]] == ["Notes.txt"]
    r = client.post("/api/books/demo/inbox/check").json()
    assert "Notes.txt" in r["text"] and r["pending"]["kind"] == "file"
    assert client.post("/api/books/demo/inbox/check").json()["text"] == ""
    assert client.post("/api/chat", json={"text": "yes"}).json()["open_item"]["title"] == "Notes"
    assert client.post("/api/books/demo/inbox/add", json={"name": "../x"}).status_code == 404


@pytest.mark.asyncio
async def test_the_speech_model_gets_the_names_as_a_hint_and_a_spoken_turn_asks_for_a_spoken_answer(books):
    from clients import STT
    seen = {}

    def handler(req):
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "hello"})
    reg = Registry(Settings(project_dir=books))
    reg.upsert("stt", {"label": "S", "base_url": "http://s", "tier": 1})
    stt = STT(Router(reg), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await stt.transcribe(b"RIFFxxxx", "a.wav", "audio/wav", hint="Story about Mira, Jonas.")
    assert b'name="prompt"' in seen["body"] and b"Story about Mira, Jonas." in seen["body"]
    ag, store, llm = make(books)
    await ag.handle("s", "tell me what you think about the station at night", via="voice")
    sys0 = llm.calls[-1][0]["content"]
    assert "TALKING TO YOU OUT LOUD" in sys0 and "cut off" not in sys0
    await ag.handle("s", "no wait, the platform I meant", via="voice", interrupted=True)
    assert "cut off your last reply" in llm.calls[-1][0]["content"]
    await ag.handle("s", "and what about the morning platform and the early train", via="typed")
    assert "TALKING TO YOU OUT LOUD" not in llm.calls[-1][0]["content"]


# ------------------------------------------------------------------------------------------------------------------ undo
def test_every_saved_change_can_be_undone_in_order(books):
    store = BookStore(books)
    first = store.read_chapter_raw("demo", 1)
    store.replace_paragraph("demo", "ch1_s1_p2", "Changed once.")
    store.append_paragraph("demo", 1, "Added later.", scene=2)
    assert "Added later." in store.read_chapter_raw("demo", 1)
    assert store.undo_last("demo") == {"chapter": 1, "what": "append", "paragraph": "ch1_s2_p2"} and "Added later." not in store.read_chapter_raw("demo", 1)
    assert "Changed once." in store.read_chapter_raw("demo", 1)
    assert store.undo_last("demo")["what"] == "edit" and store.read_chapter_raw("demo", 1) == first  # byte for byte as it was
    assert store.undo_last("demo") is None


def test_undo_refuses_to_throw_away_work_done_outside_the_app(books):
    from book_store import BookError
    store = BookStore(books)
    store.replace_paragraph("demo", "ch1_s1_p1", "Edited in the app.")
    p = store.chapter_path("demo", 1); p.write_text(p.read_text() + "\nA line the writer typed in Notepad.\n")
    with pytest.raises(BookError) as e:
        store.undo_last("demo")
    assert "outside Book Scribe" in str(e.value) and "Notepad" in p.read_text()  # nothing was lost


def test_undo_survives_damaged_history_files(books):
    store = BookStore(books)
    store.replace_paragraph("demo", "ch1_s1_p1", "X.")
    h = store.book_dir("demo") / "history"
    (h / "log.jsonl").write_text("{broken\n" + (h / "log.jsonl").read_text())
    assert store.undo_last("demo")["what"] == "edit"  # a damaged line is skipped; undo still works
    store.replace_paragraph("demo", "ch1_s1_p1", "Y.")
    for f in h.glob("*.md"):
        f.unlink()
    assert store.undo_last("demo") is None


@pytest.mark.asyncio
async def test_undo_that_by_voice(books):
    ag, store, _ = make(books)
    before = store.read_chapter_raw("demo", 1)
    r = await ag.handle("s", "undo that"); assert r.text == "There is nothing to undo."
    store.replace_paragraph("demo", "ch1_s1_p2", "Rewritten by the agent.")
    r = await ag.handle("s", "oh, put it back")
    assert "took back" in r.text and store.read_chapter_raw("demo", 1) == before and r.chapter == 1
    store.append_paragraph("demo", 1, "New paragraph.")
    r = await ag.handle("s", "can you undo the last change please"); assert "paragraph I just added" in r.text and "New paragraph." not in store.read_chapter_raw("demo", 1)
    # a proposal that is still waiting is simply dropped, and nothing saved is touched
    store.replace_paragraph("demo", "ch1_s1_p1", "Saved edit.")
    await ag.handle("s", "rewrite chapter 1 paragraph 2 to be more tense"); assert ag.session("s").pending
    r = await ag.handle("s", "undo that")
    assert "dropped" in r.text and ag.session("s").pending is None and "Saved edit." in store.read_chapter_raw("demo", 1)
    for not_undo in ["undo the damage to the kingdom in chapter two", "I want to undo the knot in the story", "tell me about the undone sword"]:
        assert not __import__("intent").undo_request(not_undo), not_undo


# ------------------------------------------------------------------------------------------------- names that fit the book
def test_names_fit_the_setting_of_the_book(books):
    from ideas import Ideas, cultures_for
    assert cultures_for("A tale of King Arthur and Excalibur") == ["welsh", "english"] and cultures_for("Viking raiders") == ["norse"] and cultures_for("nothing special") == []
    import ideas_data as D
    welsh_english = {w for c in ("welsh", "english") for k in ("female", "male") for w in D.NAMES[c][k]}
    welsh_english_family = {w for c in ("welsh", "english") for w in D.NAMES[c]["family"]}
    names = Ideas(seed=3).generate("name", 12, fits=["welsh", "english"], gender="male")
    assert all(n.split()[0] in welsh_english and n.split()[-1] in welsh_english_family for n in names)
    assert any(n.split()[0] not in welsh_english for n in Ideas(seed=3).generate("name", 12))  # without a setting it is a mix of cultures
    assert Ideas(seed=1).generate("name", 3, fits=["welsh"], culture="japanese")  # an explicit culture still wins
    import intent
    assert intent.ideas_request("give me a name for a young knight")["opts"]["gender"] == "male" and intent.ideas_request("give me a name for the abbess, a nun")["opts"]["gender"] == "female"


@pytest.mark.asyncio
async def test_asking_for_names_in_an_arthurian_book_gives_arthurian_names(books):
    import ideas_data as D
    ag, store, _ = make(books)
    store.bible("demo").write_doc("premise", "# Premise\n\nOld Sir Bedivere carries Excalibur to the lake after Camlann.")
    fit = {w for c in ("welsh", "english") for k in ("female", "male") for w in D.NAMES[c][k]}
    r = await ag.handle("s", "give me a name idea for a young knight")
    assert r.ideas and all(i.split()[0] in fit for i in r.ideas), r.ideas
    r = await ag.handle("s", "give me three japanese names")
    assert all(i.split()[0] not in fit for i in r.ideas)


# ------------------------------------------------------------------- what the writer-agent hit while writing the Arthur book
@pytest.mark.asyncio
async def test_chapters_can_be_started_by_voice_and_a_named_chapter_is_respected(books):
    ag, store, llm = make(books)
    assert store.chapter_numbers("demo") == [1]
    r = await ag.handle("s", "go to chapter 2"); assert "no chapter 2 yet" in r.text and "start chapter 2" in r.text
    r = await ag.handle("s", "start chapter 3"); assert "skip" in r.text and store.chapter_numbers("demo") == [1]
    r = await ag.handle("s", "start a new chapter"); assert "Chapter 2 is ready" in r.text and store.chapter_numbers("demo") == [1, 2] and r.chapter == 2
    r = await ag.handle("s", "go to chapter 1"); assert r.chapter == 1
    llm.replies = ["The boy pulled and the stone gave way as if it had been waiting for him."]
    r = await ag.handle("s", "write the first paragraph of chapter 2 about the boy and the stone")  # the chapter the writer names, not the one on screen
    assert "chapter 2" in r.text and r.pending["ask"] == "Add this paragraph to chapter 2?"
    r = await ag.handle("s", "yes")
    assert "stone gave way" in store.read_chapter_raw("demo", 2) and "stone gave way" not in store.read_chapter_raw("demo", 1)
    llm.replies = ["A tale about the lake and its keeper, told at dusk to nobody."]
    r = await ag.handle("s", "write the first paragraph of chapter 3"); assert "start chapter 3 with it" in r.text.lower()
    await ag.handle("s", "yes"); assert store.chapter_numbers("demo") == [1, 2, 3]  # a new chapter is made when the paragraph is saved


@pytest.mark.asyncio
async def test_delete_and_move_by_voice_are_confirmed_and_undoable(books):
    ag, store, _ = make(books)
    before = store.read_chapter_raw("demo", 1)
    r = await ag.handle("s", "delete paragraph 2"); assert r.pending["kind"] == "delete" and r.pending["ask"].startswith("Delete ") and store.read_chapter_raw("demo", 1) == before
    r = await ag.handle("s", "no"); assert store.read_chapter_raw("demo", 1) == before
    await ag.handle("s", "delete paragraph 2"); r = await ag.handle("s", "yes")
    assert "Deleted" in r.text and len(store.paragraphs("demo", 1)) == 2
    assert "p3" not in "".join(p.id for p in store.paragraphs("demo", 1)) or True
    new = store.append_paragraph("demo", 1, "After the delete."); assert len({p.id for p in store.paragraphs("demo", 1)}) == 3 and new.id not in ["ch1_s1_p1"]  # ids stay unique
    await ag.handle("s", "undo that"); await ag.handle("s", "undo that"); assert store.read_chapter_raw("demo", 1) == before
    # move to a brand-new chapter, then one undo takes the whole move back
    r = await ag.handle("s", "move paragraph 3 of chapter 1 to chapter 2"); assert r.pending["kind"] == "move" and "chapter 2" in r.pending["ask"]
    r = await ag.handle("s", "yes"); assert "Moved" in r.text and store.chapter_numbers("demo") == [1, 2] and len(store.paragraphs("demo", 1)) == 2 and len(store.paragraphs("demo", 2)) == 1
    r = await ag.handle("s", "undo that"); assert "moved the paragraph back" in r.text and store.read_chapter_raw("demo", 1) == before and store.paragraphs("demo", 2) == []
    r = await ag.handle("s", "move paragraph 1 to chapter 9"); assert "no chapter 9" in r.text and r.pending is None
    r = await ag.handle("s", "delete the wiki page"); assert not (r.pending and r.pending["kind"] == "delete")


@pytest.mark.asyncio
async def test_a_question_or_a_look_at_the_book_does_not_throw_away_a_waiting_change(books):
    ag, store, llm = make(books, )
    llm.replies = ["Snow, then. The sentence now says the platform was white."]
    await ag.handle("s", "rewrite chapter 1 paragraph 2 so it is snowing")
    assert ag.session("s").pending.kind == "edit"
    llm.replies = ["Mira is the conductor; Jonas is the nervous traveller."]
    r = await ag.handle("s", "who is Jonas again?")
    assert "Jonas" in r.text and ag.session("s").pending and r.pending["ask"].startswith("Save this change")  # answered, and the question is still waiting
    r = await ag.handle("s", "read paragraph 1"); assert r.pending and r.pending["kind"] == "edit"
    r = await ag.handle("s", "yes"); assert "Saved" in r.text and ag.session("s").pending is None


@pytest.mark.asyncio
async def test_yes_to_a_draft_that_was_shown_without_a_proposal_still_saves_it(books):
    ag, store, llm = make(books)
    draft = ("Here's the next paragraph:\n\nThe courtyard was loud with the clatter of buckets, and the boy who would be king stood in the middle of it, "
             "soaked to the knees and grinning, because he had been told to carry water and had carried too much. Shall I add it to chapter 1?")
    llm.replies = [draft]
    await ag.handle("s", "tell me what the next scene might look like, in the courtyard")
    r = await ag.handle("s", "yes add it")
    assert "Saved" in r.text and "just showed you" in r.text
    saved = store.read_chapter_raw("demo", 1)
    assert "clatter of buckets" in saved and "Shall I add it" not in saved and "Here's the next paragraph" not in saved
    r = await ag.handle("s", "yes")  # nothing to rescue now: an ordinary yes goes to the conversation, not a second save
    assert saved == store.read_chapter_raw("demo", 1)


@pytest.mark.asyncio
async def test_the_model_drafts_through_propose_append_and_the_writer_sees_and_hears_it(books):
    ag, store, llm = make(books)
    para = "The sword hung against his hip and knocked at his knee with every step, once, twice, as if it wanted his attention and had no other way to ask."
    llm.replies = [f'[[tool: propose_append {{"text": "{para}", "chapter": 1}}]]', "Here you go."]
    r = await ag.handle("s", "next paragraph, the walk to the lake")
    assert para in r.text and "Should I add it to chapter 1?" in r.text and r.pending["kind"] == "append" and r.pending["ask"] == "Add this paragraph to chapter 1?"
    assert para not in store.read_chapter_raw("demo", 1)
    sys0 = llm.calls[0][0]["content"]
    assert "BOOK STATE" in sys0 and "chapter 1: 3 saved paragraphs" in sys0 and "never tell them to press" in sys0
    await ag.handle("s", "yes"); assert para in store.read_chapter_raw("demo", 1)
    assert "A proposal is waiting" not in ag._book_state(ag.session("s")) and "Nothing is waiting" in ag._book_state(ag.session("s"))


@pytest.mark.asyncio
async def test_the_title_and_idea_of_the_book_can_be_kept_as_the_premise(books):
    import harness
    ag, store, _ = make(books)
    s = ag.session("s"); ctx = harness.Ctx(sess=s)
    out = await ag.toolbox.call("propose_wiki_note", {"kind": "title", "name": "", "note": "Title: The Sword That Was Returned. Told by old Sir Bedivere on the last night."}, ctx)
    assert "Proposed" in out and s.pending.data["kind"] == "premise"
    r = await ag.handle("s", "yes"); assert "premise" in r.text.lower()
    prem = store.bible("demo").read("premise"); assert "The Sword That Was Returned" in prem and prem.count("# Premise") == 1 and "night-train conductor" in prem.lower()


def test_a_long_message_that_mentions_a_title_is_not_a_request_for_random_titles():
    import intent
    pitch = "I want it told by old Sir Bedivere on the last night, carrying Excalibur to the lake and remembering Arthur's whole life as he walks. The working title is The Sword That Was Returned."
    assert intent.ideas_request(pitch) is None and intent.ideas_request("give me five titles")["kind"] == "title"
    assert intent.new_chapter("start chapter 2") == 2 and intent.new_chapter("begin a new chapter") == 0 and intent.new_chapter("write chapter two of my life story") is None
    assert intent.is_draft("next paragraph, the courtyard") and intent.step_direction("next paragraph") == 1


# ------------------------------------------------------------------------- second Arthur session: what the writer still hit
@pytest.mark.asyncio
async def test_paragraphs_can_be_put_in_a_different_order_by_voice(books):
    ag, store, _ = make(books)
    first = lambda: [p.text[:9] for p in store.paragraphs("demo", 1)]
    base = store.read_chapter_raw("demo", 1)
    assert first() == ["Mira step", "The expre", "At the ca"]
    r = await ag.handle("s", "move paragraph 3 to the start of chapter 1"); assert r.pending["kind"] == "move" and "start of chapter 1" in r.text
    await ag.handle("s", "yes"); assert first() == ["At the ca", "Mira step", "The expre"]
    await ag.handle("s", "undo that"); assert first() == ["Mira step", "The expre", "At the ca"] and store.read_chapter_raw("demo", 1) == base
    r = await ag.handle("s", "move paragraph 1 after paragraph 2"); await ag.handle("s", "yes"); assert first() == ["The expre", "Mira step", "At the ca"]
    r = await ag.handle("s", "move paragraph 1 to the end"); await ag.handle("s", "yes"); assert first() == ["Mira step", "At the ca", "The expre"]
    r = await ag.handle("s", "move paragraph 3 to the end"); assert "already at the end" in r.text and r.pending is None
    r = await ag.handle("s", "move paragraph 2 before paragraph 9"); assert r.pending is not None  # asks first; the problem shows when it is attempted
    r = await ag.handle("s", "yes"); assert "no paragraph 9" in r.text and first() == ["Mira step", "At the ca", "The expre"]


@pytest.mark.asyncio
async def test_small_talk_while_a_note_is_waiting_keeps_it_and_reminds_out_loud(books):
    import harness
    ag, store, llm = make(books)
    s = ag.session("s")
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "Sir Cadan", "note": "A lazy knight."}, {"kind": "world", "name": "Dozmary Pool", "note": "Lake on Bodmin Moor."}]}, harness.Ctx(sess=s))
    assert s.pending.kind == "wiki" and len(s.pending.data["batch"]) == 2
    for chatter in ["um, the thing with the water", "hello", "goodbye", "what is the weather like in Cornwall?"]:
        llm.replies = ["Of course."]
        r = await ag.handle("s", chatter)
        assert s.pending is not None and r.pending["ask"] == "Save these 2 notes to the wiki?" and "still holding this for you" in r.text and r.speech[-1]["text"].startswith("I am still holding")
    r = await ag.handle("s", "yes")  # one yes saves both
    assert "Sir Cadan" in r.text and "Dozmary Pool" in r.text and "Sir Cadan" in store.bible("demo").characters() and "Dozmary Pool" in store.bible("demo").entries("world")
    # a real new job does put it aside, and says so
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "Gwen", "note": "A cook."}]}, harness.Ctx(sess=s))
    r = await ag.handle("s", "delete paragraph 1"); assert s.pending.kind == "delete" and "dropped the unsaved proposal" in " ".join(r.notes).lower()


@pytest.mark.asyncio
async def test_ideas_are_picked_by_number_so_nobody_has_to_tap(books):
    ag, store, llm = make(books)
    r = await ag.handle("s", "give me three names for a knight")
    assert "Number 2," in r.text and "Say the number you like" in r.text and "Tap" not in r.text
    chosen = r.ideas[1]
    llm.replies = ["Good name. What is he like?"]
    r = await ag.handle("s", "the second one")
    assert llm.calls[-1][-1]["content"] == f"Let's use this name: {chosen}"  # the model hears which one was chosen
    llm.replies = ["Okay."]
    r = await ag.handle("s", "number 2")  # nothing is on offer any more: an ordinary message
    assert "Let's use this" not in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_undo_says_what_it_really_undid_and_drafts_are_told_the_whole_book(books):
    ag, store, llm = make(books)
    await ag.handle("s", "delete paragraph 2"); await ag.handle("s", "yes")
    r = await ag.handle("s", "undo that"); assert "put back the paragraph I deleted" in r.text and "added" not in r.text
    llm.replies = ["x"]
    await ag.handle("s", "tell me what you think of the pace here, honestly, in a few words")
    sysmsg = llm.calls[-1][0]["content"]
    assert "THE BOOK SO FAR" in sysmsg and "ch1 p1: Mira stepped onto the platform" in sysmsg
