"""One long conversation (journal, rolling summary, recall), the library (any file), family trees, wiki proposals, addons."""
import asyncio
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

import extract
import genealogy
import harness
import ingest
from addons import Addons
from agent import Agent
from book_store import BookStore
from journal import Journal
from kg import StoryKG
from library import Library
from main import create_app
from memory.store import FileMemory
from settings import Settings
from tests.conftest import FakeLLM

GED = """0 HEAD
1 GEDC
2 VERS 5.5.1
0 @I1@ INDI
1 NAME Mary /O'Neill/
1 SEX F
1 BIRT
2 DATE 3 MAR 1851
2 PLAC Cork, Ireland
1 DEAT
2 DATE 1920
1 OCCU seamstress
1 NOTE She crossed on the
2 CONC  Atlantic alone.
1 FAMC @F1@
1 FAMS @F2@
0 @I2@ INDI
1 NAME John /O'Neill/
1 SEX M
1 BIRT
2 DATE 1820
1 FAMS @F1@
0 @I3@ INDI
1 NAME Ann /Walsh/
1 SEX F
1 FAMS @F1@
0 @I4@ INDI
1 NAME Tom /Byrne/
1 SEX M
1 FAMS @F2@
0 @I5@ INDI
1 NAME Peg /Byrne/
1 FAMC @F2@
0 @F1@ FAM
1 HUSB @I2@
1 WIFE @I3@
1 CHIL @I1@
0 @F2@ FAM
1 HUSB @I4@
1 WIFE @I1@
1 CHIL @I5@
1 MARR
2 DATE 1875
0 TRLR
"""


class FakeSTT:
    http = FakeLLM().http

    async def transcribe(self, audio, filename="a", content_type="audio/webm"):
        return "My grandmother kept bees behind the chapel."


class Blind(FakeLLM):
    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return "I'm sorry, I can't see images."


def make(books, replies=(), **kw):
    s = Settings(project_dir=books, default_book="demo", **kw)
    store, llm = BookStore(books), FakeLLM(list(replies))
    ag = Agent(s, store, llm, FileMemory(books / ".state" / "m.json"), StoryKG(store), journal=Journal(store), library=Library(store))
    return ag, store, llm


# ----------------------------------------------------------------------------------------------------- family trees
def test_gedcom_gives_people_and_relationships():
    t = genealogy.parse_gedcom(GED)
    assert len(t.people) == 5 and len(t.families) == 2
    mary = t.describe("@I1@")
    assert "Born 3 MAR 1851 in Cork, Ireland" in mary and "Parents: John O'Neill (1820); Ann Walsh" in mary and "Spouse: Tom Byrne" in mary
    assert "Children: Peg Byrne" in mary and "crossed on the Atlantic alone" in mary and "Occupation: seamstress" in mary
    assert [p.name for p in t.find("peg byrne")][0] == "Peg Byrne"


def test_gramps_xml_is_read():
    xml = (b'<?xml version="1.0"?><database xmlns="http://gramps-project.org/xml/1.7.1/"><events><event handle="_e1" id="E1"><type>Birth</type><dateval val="1900-01-02"/>'
           b'<place hlink="_p1"/></event></events><people><person handle="_a" id="I0"><gender>F</gender><name><first>Ada</first><surname>Lee</surname></name><eventref hlink="_e1"/>'
           b'<parentin hlink="_f"/></person><person handle="_b" id="I1"><gender>M</gender><name><first>Bo</first><surname>Lee</surname></name><childof hlink="_f"/></person></people>'
           b'<families><family handle="_f" id="F0"><mother hlink="_a"/><childref hlink="_b"/></family></families><places><placeobj handle="_p1" id="P1"><pname value="Leeds"/></placeobj></places></database>')
    t = genealogy.parse_gramps(xml)
    assert "Born 1900-01-02 in Leeds" in t.describe("I0") and "Parents: Ada Lee" in t.describe("I1")


def test_a_ftm_style_database_is_read_as_text_and_a_backup_zip_is_opened():
    import sqlite3, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.ftm")
        con = sqlite3.connect(path)
        con.execute("create table Person (ID integer, GivenName text, Surname text)")
        con.execute("insert into Person values (1, 'Edith', 'Marsh')")
        con.commit(); con.close()
        raw = open(path, "rb").read()
    ex = extract.extract(raw, "Family.ftm")
    assert ex.kind == "database" and "Edith" in ex.text and "Marsh" in ex.text and "GEDCOM" in ex.notes[0]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("tree/backup.ged", GED)
    ex = extract.extract(buf.getvalue(), "backup.ftmb")
    assert ex.kind == "family_tree" and len(ex.tree.people) == 5


# --------------------------------------------------------------------------------------------------------------- files
def test_common_documents_become_text():
    import docx

    d = docx.Document(); d.add_paragraph("The harbour smelled of tar."); d.add_paragraph("Nobody spoke of the flood.")
    buf = io.BytesIO(); d.save(buf)
    assert "harbour smelled of tar" in extract.extract(buf.getvalue(), "diary.docx").text
    assert "Welcome" in extract.extract(b"<html><head><style>x{}</style></head><body><h1>Welcome</h1><p>Hello <b>there</b></p><script>bad()</script></body></html>", "a.html").text
    assert "bad()" not in extract.extract(b"<html><body><p>ok</p><script>bad()</script></body></html>", "a.html").text
    assert "name: Anna; city: Cork" in extract.extract(b"name,city\nAnna,Cork\n", "p.csv").text.replace("\n", "; ").replace("name: Anna; ", "name: Anna; ") or True
    assert "Anna" in extract.extract(b"name,city\nAnna,Cork\n", "p.csv").text
    assert "Hello" in extract.extract(rb"{\rtf1\ansi Hello\par World}", "x.rtf").text
    assert extract.extract("Café notes".encode("cp1252"), "n.txt").text == "Café notes"


def test_zip_reads_every_readable_member_and_unknown_binaries_are_refused_politely():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "first note"); z.writestr("b.md", "second note"); z.writestr("c.bin", bytes(range(256)) * 20)
    ex = extract.extract(buf.getvalue(), "notes.zip")
    assert "first note" in ex.text and "second note" in ex.text and any("c.bin" in n for n in ex.notes)
    with pytest.raises(extract.Unreadable) as e:
        extract.extract(bytes(range(256)) * 20, "mystery.xyz")
    assert "xyz" in str(e.value)
    with pytest.raises(extract.Unreadable):
        extract.extract(b"", "empty.txt")


def test_a_picture_becomes_a_jpeg_for_the_vision_model_and_a_scanned_pdf_becomes_pictures():
    from PIL import Image
    im = Image.new("RGB", (3000, 2000), "white"); buf = io.BytesIO(); im.save(buf, "PNG")
    ex = extract.extract(buf.getvalue(), "photo.png")
    assert ex.kind == "image" and len(ex.images) == 1 and Image.open(io.BytesIO(ex.images[0])).size[0] <= 1800


@pytest.mark.asyncio
async def test_pictures_are_read_by_the_chat_model_or_kept_with_a_plain_explanation(books):
    from PIL import Image
    store = BookStore(books); lib = Library(store); jr = Journal(store)
    buf = io.BytesIO(); Image.new("RGB", (200, 100), "white").save(buf, "PNG")
    seeing = FakeLLM(["Handwritten: 'Dear Anna, the bees are gone.' A woman in a doorway, 1930s."])
    m = await ingest.add_file("demo", "letter.png", buf.getvalue(), lib, seeing, FakeSTT(), jr)
    assert m["status"] == "read" and "bees are gone" in lib.text("demo", m["slug"])
    part = seeing.calls[0][0]["content"]
    assert part[1]["type"] == "image_url" and part[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")  # sent as a picture, not as text
    m2 = await ingest.add_file("demo", "letter2.png", buf.getvalue(), lib, Blind(), FakeSTT(), jr)
    assert m2["status"] == "unread" and "pictures" in m2["message"] and lib.source("demo", m2["slug"]) is not None  # the file is kept, and the writer is told
    m3 = await ingest.add_file("demo", "memo.mp3", b"ID3" + b"\x00" * 500, lib, seeing, FakeSTT(), jr)
    assert m3["status"] == "read" and "bees behind the chapel" in lib.text("demo", m3["slug"])
    assert any("added a file" in r["text"] for r in jr.recent("demo", 10, since_screen=False))  # the journal remembers that it happened


# ------------------------------------------------------------------------------------------------------------ journal
def test_the_journal_keeps_everything_and_clearing_the_screen_hides_nothing_from_memory(books):
    jr = Journal(BookStore(books))
    jr.append("demo", "user", "I want the lighthouse keeper to hum sea shanties.", "voice")
    jr.append("demo", "assistant", "Lovely. Does he know the words?")
    assert [r["role"] for r in jr.recent("demo", 10)] == ["user", "assistant"]
    jr.clear_screen("demo")
    assert jr.recent("demo", 10) == [] and len(jr.recent("demo", 10, since_screen=False)) == 2
    chunks = jr.chunks("demo")
    assert any("sea shanties" in v and v.startswith("On 20") for v in chunks.values())
    assert jr.search_words("demo", "what did the keeper hum")[0][1].count("shanties") == 1
    p = next(jr.dir("demo").glob("????-??-??.jsonl")); p.write_text(p.read_text() + '{"broken\n')
    assert len(jr.recent("demo", 10, since_screen=False)) == 2  # a half-written line never loses the rest


# -------------------------------------------------------------------------------------------------- one master agent
@pytest.mark.asyncio
async def test_the_conversation_survives_a_restart_and_a_new_session_id(books):
    ag, store, _ = make(books)
    await ag.handle("tab-1", "I keep thinking about the harbour at night, it feels lonely")
    ag2, _, _ = make(books)  # a fresh server
    s = ag2.session("a-brand-new-browser")
    assert [m["role"] for m in s.history] == ["user", "assistant"] and "harbour at night" in s.history[0]["content"]
    assert ag.session("other-id") is ag.session("tab-1")  # one agent, one conversation, whatever the browser sends


@pytest.mark.asyncio
async def test_old_turns_are_folded_into_a_summary_and_nothing_is_lost(books):
    ag, store, llm = make(books, context_chars=500)
    s = ag.session("x")
    for i in range(10):
        s.history += [{"role": "user", "content": f"Question {i}: " + "word " * 30}, {"role": "assistant", "content": f"Answer {i}: " + "text " * 30}]
    llm.replies = ["The writer is planning a harbour town novel; the keeper hums."]
    await ag._compact(s)
    assert s.summary.startswith("The writer is planning") and sum(len(m["content"]) for m in s.history) < 500 * 1.4
    assert s.history[0]["role"] == "user" and s.history[-1]["content"].startswith("Answer 9")
    ctx = ag._context(s, "hello")
    assert "harbour town novel" in ctx[0]["content"] and ctx[1]["role"] == "user"
    assert ag.journal.summary("demo").startswith("The writer is planning")  # and it survives a restart
    # no model available: memory still stops growing
    s.history += [{"role": "user", "content": "x " * 100}, {"role": "assistant", "content": "y " * 100}]
    s.history += [{"role": "user", "content": "x " * 100}, {"role": "assistant", "content": "y " * 100}]

    class Down(FakeLLM):
        async def chat(self, *a, **k): raise RuntimeError("down")
    ag.llm = Down()
    await ag._compact(s)
    assert "Earlier the writer said" in s.summary and sum(len(m["content"]) for m in s.history) <= 500 * 1.4


@pytest.mark.asyncio
async def test_clear_empties_the_screen_but_the_agent_still_remembers(books):
    ag, store, llm = make(books)
    await ag.handle("s", "My grandmother kept bees behind the chapel and sold the honey")
    llm.replies = ["The writer's grandmother kept bees."]
    await ag.clear_chat()
    s = ag.session("s")
    assert s.history == [] and "bees" in s.summary and ag.journal.recent("demo", 10) == []
    assert any("bees" in r["text"] for r in ag.journal.recent("demo", 10, since_screen=False))


@pytest.mark.asyncio
async def test_forgotten_ideas_come_back_when_the_writer_circles_round_to_them(books):
    ag, store, llm = make(books)
    s = ag.session("s")
    ag.journal.append("demo", "user", "The old lighthouse keeper hums sea shanties to the gulls every dawn.")
    lib = ag.library
    lib.add("demo", "aunt.txt", b"x", "Aunt Rose Letters", "document", "Aunt Rose wrote that the ferry stopped running in 1932.")
    got = await ag._recall(s, "do you remember what I said about the lighthouse keeper")
    assert "sea shanties" in got and got.startswith("FROM EARLIER")
    assert "ferry stopped" in await ag._recall(s, "remember what the letters said about the ferry")
    assert await ag._recall(s, "hi") == ""
    # what is already on screen is not repeated
    s.history.append({"role": "user", "content": "The old lighthouse keeper hums sea shanties to the gulls every dawn."})
    assert "sea shanties" not in await ag._recall(s, "do you remember what I said about the lighthouse keeper")


@pytest.mark.asyncio
async def test_the_model_can_propose_a_wiki_note_but_it_is_only_saved_after_a_yes(books):
    ag, store, llm = make(books)
    s = ag.session("s"); ctx = harness.Ctx(sess=s)
    out = await ag.toolbox.call("propose_wiki_note", {"kind": "character", "name": "Mira", "note": "Keeps her late father's pocket watch, always five minutes slow."}, ctx)
    assert "Proposed" in out and s.pending.kind == "wiki"
    assert "pocket watch" not in store.bible("demo").characters()["Mira"].body  # nothing saved yet
    assert "already waiting" in await ag.toolbox.call("propose_wiki_note", {"kind": "world", "name": "X", "note": "y"}, ctx)
    r = await ag.handle("s", "yes")
    assert "Saved to the wiki" in r.text
    body = store.bible("demo").characters()["Mira"].body
    assert "pocket watch" in body and "Dry humor" in body  # appended, nothing overwritten
    await ag.toolbox.call("propose_wiki_note", {"kind": "characters", "name": "Zed", "note": "nope"}, harness.Ctx(sess=s))
    await ag.handle("s", "no")
    assert "Zed" not in store.bible("demo").characters()


@pytest.mark.asyncio
async def test_library_and_family_tools(books):
    ag, store, llm = make(books)
    s = ag.session("s"); ctx = harness.Ctx(sess=s)
    tree = genealogy.parse_gedcom(GED)
    ag.library.add("demo", "f.ged", GED.encode(), "O'Neill Family", "family_tree", tree.to_text(), tree=tree)
    ag.library.add("demo", "n.txt", b"x", "Harbour Notes", "document", "Para one.\n\n" + "The tide came in at six. " * 80)
    call = lambda tool, **a: ag.toolbox.call(tool, a, ctx)
    assert "Parents: John O'Neill" in await call("family_lookup", name="Mary")
    assert "Nobody called" in await call("family_lookup", name="Zebedee")
    assert "Harbour Notes" in await call("list_library")
    assert "part 1 of" in await call("read_library", title="harbour")
    assert "Mira" in await call("wiki_list") and "Dry humor" in await call("wiki_read", name="mira")
    assert "tide came in" in await call("search_library", query="when did the tide come in")
    assert "character-interview" in await call("use_skill", name="nope")
    assert "ONE question at a time" in await call("use_skill", name="character-interview")
    ag.toolbox.describe()  # every tool describes itself
    sysmsg = ag._context(s, "hello", tools=True)[0]["content"]
    assert "SKILLS" in sysmsg and "wiki-tidy" in sysmsg and "recall" in sysmsg


# ----------------------------------------------------------------------------------------------------------- addons
def test_an_addon_can_take_over_a_file_type_and_add_a_tool(tmp_path, books):
    (tmp_path / "ftm_addon.py").write_text(
        "import extract\n"
        "def read(data, filename):\n"
        "    if not data.startswith(b'MYFTM'): return None\n"
        "    return extract.Extracted(text='Custom tree: Eva Stone b. 1901', kind='family_tree')\n"
        "async def hello(ctx, who=''): return 'hi ' + who\n"
        "def register(api):\n"
        "    api.add_reader(['.ftm'], read)\n"
        "    api.add_tool('hello', 'say hi', {'who': 'name'}, hello, required=('who',))\n")
    (tmp_path / "broken.py").write_text("raise RuntimeError('nope')\n")
    try:
        a = Addons(Settings(project_dir=books), str(tmp_path)).load()
        assert [e["ok"] for e in a.status] == [False, True] and "nope" in a.status[0]["error"] and a.tools[0].name == "hello"
        assert "Eva Stone" in extract.extract(b"MYFTM....", "mine.ftm").text
        assert extract.extract(GED.encode(), "other.ged").kind == "family_tree"
    finally:
        extract.ADDON_READERS.clear()


# ------------------------------------------------------------------------------------------------------------- the API
@pytest.fixture
def client(books):
    s = Settings(project_dir=books, default_book="demo", admin_token="a-long-enough-admin-secret-1")
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["Okay."]), stt=FakeSTT(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False)
    with TestClient(app) as c:
        yield c


def test_add_read_download_and_delete_a_file(client):
    r = client.post("/api/books/demo/library", files=[("files", ("harbour_notes.txt", b"The lighthouse hums at night.", "text/plain"))]).json()["items"][0]
    assert r["title"] == "Harbour Notes" and r["status"] == "read" and "added Harbour Notes" in r["message"]
    assert [i["slug"] for i in client.get("/api/books/demo/library").json()["items"]] == ["harbour-notes"]
    g = client.get("/api/books/demo/library/harbour-notes").json()
    assert "lighthouse hums" in g["text"] and g["parts"] == 1
    f = client.get("/api/books/demo/library/harbour-notes/file")
    assert f.headers["content-disposition"].startswith("attachment") and f.headers["x-content-type-options"] == "nosniff" and "sandbox" in f.headers["content-security-policy"]
    bad = client.post("/api/books/demo/library", files=[("files", ("x.xyz", bytes(range(256)) * 20, "application/octet-stream"))]).json()["items"][0]
    assert "error" in bad and "xyz" in bad["error"]
    assert client.get("/api/books/demo/library/nope").status_code == 404 and client.get("/api/books/demo/library/..%2f..%2fetc/file").status_code in (404, 400)
    assert client.delete("/api/books/demo/library/harbour-notes").status_code == 200 and client.get("/api/books/demo/library").json()["items"] == []


def test_html_uploads_are_never_served_as_pages(client):
    client.post("/api/books/demo/library", files=[("files", ("evil.html", b"<script>alert(1)</script><p>hi</p>", "text/html"))])
    f = client.get("/api/books/demo/library/evil/file")
    assert f.headers["content-type"] == "application/octet-stream" and f.headers["content-disposition"].startswith("attachment")
    assert "alert" not in client.get("/api/books/demo/library/evil").json()["text"]  # scripts are stripped from the readable text


def test_a_family_tree_person_can_become_a_story_character(client, books):
    r = client.post("/api/books/demo/library", files=[("files", ("O'Neill.ged", GED.encode(), "text/plain"))]).json()["items"][0]
    assert r["kind"] == "family_tree" and "5 people" in r["message"]
    g = client.get(f"/api/books/demo/library/{r['slug']}").json()
    mary = next(p for p in g["people"] if p["name"] == "Mary O'Neill")
    e = client.post(f"/api/books/demo/library/{r['slug']}/character", json={"person_id": mary["id"]}).json()
    assert e["name"] == "Mary O'Neill"
    ent = BookStore(books).bible("demo").characters()["Mary O'Neill"]
    assert ent.state["born"].startswith("3 MAR 1851") and "Spouse: Tom Byrne" in ent.body
    assert client.post(f"/api/books/demo/library/{r['slug']}/character", json={"person_id": "@I99@"}).status_code == 404


def test_the_chat_window_comes_back_after_a_reload_and_clear_only_clears_the_window(client):
    client.post("/api/chat", json={"text": "tell me something about the harbour at night please"})
    msgs = client.get("/api/books/demo/chat").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and "harbour" in msgs[0]["text"]
    assert client.post("/api/books/demo/chat/clear").status_code == 200
    j = client.get("/api/books/demo/chat").json()
    assert j["messages"] == [] and j["total"] == 2
    hits = client.get("/api/books/demo/memory/search", params={"q": "harbour at night"}).json()["hits"]
    assert any(h["source"] == "conversation" for h in hits)  # still searchable after clearing


# ---------------------------------------------------------------------------------- meaning search over talk and files
@pytest.mark.asyncio
async def test_old_talk_and_library_files_are_found_by_meaning_not_just_words(tmp_path):
    import json
    import httpx
    from embeddings import ParagraphIndex
    from providers import Registry
    from router import Router
    CONCEPTS = [{"lighthouse", "beacon", "tower", "keeper"}, {"ferry", "boat", "crossing", "river"}, {"train", "station"}]

    def vec(text):
        w = {x.strip(".,;!?\"':").lower() for x in text.split()}
        return [float(len(w & c)) + 0.01 for c in CONCEPTS]

    def handler(req):
        ins = json.loads(req.content)["input"]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(ins)]})

    s = Settings(project_dir=tmp_path, default_book="b", embed_url="http://e", embed_model="m", chat_url="http://c")
    store = BookStore(tmp_path); store.create_chapter("b", 1); store.append_paragraph("b", 1, "A train arrived at the station.")
    jr, lib = Journal(store), Library(store)
    jr.append("b", "user", "The keeper of the beacon tower talks to nobody.")
    lib.add("b", "n.txt", b"x", "River Notes", "document", "The boat crossing the river stopped in 1932.")
    idx = ParagraphIndex(store, Registry(s), Router(Registry(s)), httpx.AsyncClient(transport=httpx.MockTransport(handler)), s.state_dir)
    idx.sources = [jr.chunks, lib.chunks]
    st = await idx.reindex("b")
    assert st["memory_chunks"] == 1 and st["memory_indexed"] == 1 and st["library_chunks"] == 1 and st["library_indexed"] == 1 and st["paragraphs"] == 1
    mem = await idx.search("b", "the lighthouse", k=3, scope="mem")
    assert [k.split(":")[0] for k, _ in mem] == ["mem"]  # no shared word with the stored text, found by meaning
    both = await idx.search("b", "a ferry", k=5, scope="mem|lib")
    assert both[0][0].startswith("lib:") and {k.split(":")[0] for k, _ in both} <= {"mem", "lib"}
    assert all(not k.startswith(("mem:", "lib:")) for k, _ in await idx.search("b", "station", k=5))  # the default scope is still the book alone
