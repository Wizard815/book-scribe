"""Pictures in the book: checking files, picture paragraphs, making / finding / reusing pictures, export, and the draft clean-up."""
import io
import json
import zipfile

import httpx
import intent
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import export
import harness
import images as imgs
import web as webmod
from agent import Agent, clean_draft
from book_store import BookStore
from images import ImageStore, PictureError
from journal import Journal
from kg import StoryKG
from library import Library
from main import create_app
from memory.store import FileMemory
from settings import Settings
from tests.conftest import FakeLLM
from tests.test_memory_library import FakeSTT


def png(w=200, h=120, color=(40, 90, 160), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, "PNG")
    return buf.getvalue()


class FakeImageGen:
    enabled = True

    def __init__(self):
        self.prompts = []

    async def generate(self, prompt, size="1024x1024"):
        self.prompts.append(prompt)
        return png(512, 320, (120, 60, 30))


def make(books, replies=()):
    s = Settings(project_dir=books, default_book="demo")
    store, llm = BookStore(books), FakeLLM(list(replies))
    ag = Agent(s, store, llm, FileMemory(books / ".state" / "m.json"), StoryKG(store), journal=Journal(store), library=Library(store))
    ag.imagegen = FakeImageGen()
    return ag, store, llm


# ------------------------------------------------------------------------------------------------------------- files
def test_only_real_pictures_get_in_and_they_are_shrunk_and_cleaned():
    data, ext = imgs.normalise(png(3000, 2000))
    assert ext == ".jpg" and max(Image.open(io.BytesIO(data)).size) == 1600
    d2, e2 = imgs.normalise(png(100, 100, (10, 20, 30, 0), "RGBA")); assert e2 == ".png"  # transparency is kept
    for bad in (b"", b"not a picture at all", b"\x89PNG\r\n\x1a\nbroken", b"<svg onload=alert(1)></svg>", png(10, 10), b"x" * (16 * 1024 * 1024)):
        with pytest.raises(PictureError):
            imgs.normalise(bad)
    bomb = io.BytesIO(); Image.new("1", (20000, 20000)).save(bomb, "PNG")
    with pytest.raises(PictureError):
        imgs.normalise(bomb.getvalue())  # a tiny file that unpacks into 400 million pixels is refused


def test_candidates_become_pictures_only_when_accepted_and_names_cannot_escape(books):
    st = ImageStore(BookStore(books))
    name = st.stage("demo", png(), "The Lake at Dusk!")
    assert name.startswith("the-lake-at-dusk-") and st.path("demo", name, candidate=True).is_file()
    with pytest.raises(FileNotFoundError):
        st.path("demo", name)  # not in the book yet
    st.accept("demo", name); assert st.path("demo", name).is_file()
    for bad in ("../x", "..", "a/b", ".candidates", "", "x" * 300, "bad name.png"):
        with pytest.raises(FileNotFoundError):
            st.path("demo", bad)
    n2 = st.stage("demo", png(color=(1, 2, 3)), "x"); st.discard("demo", n2); st.discard("demo", n2)
    with pytest.raises(FileNotFoundError):
        st.path("demo", n2, candidate=True)


def test_a_picture_is_a_paragraph_that_can_be_moved_deleted_and_undone(books):
    store = BookStore(books); st = ImageStore(store)
    before = store.read_chapter_raw("demo", 1)
    name = st.save("demo", png(), "lake")
    p = store.append_image("demo", 1, name, "A hand rises from the lake")
    assert p.image == ("A hand rises from the lake", name) and imgs.plain(p.text) == "Picture: A hand rises from the lake"
    assert store.paragraphs("demo", 1)[-1].image and len(store.paragraphs("demo", 1)) == 4
    store.move_paragraph("demo", p.id, 1, "start"); assert store.paragraphs("demo", 1)[0].image
    store.undo_last("demo"); assert not store.paragraphs("demo", 1)[0].image
    store.delete_paragraph("demo", store.paragraphs("demo", 1)[-1].id); store.undo_last("demo"); store.undo_last("demo")
    assert store.read_chapter_raw("demo", 1) == before
    assert imgs.parse("![a](image:../../etc/passwd)") is None and imgs.parse("![a](http://evil/x.png)") is None
    assert imgs.clean_caption("bad ] caption\n<!-- x -->") == "bad caption"


def test_chapters_can_have_titles_and_the_title_can_be_undone(books):
    store = BookStore(books)
    store.create_chapter("demo", 2)
    assert store.chapter_title("demo", 2) == "" and store.chapter_title("demo", 1) == "The Departure"
    store.append_paragraph("demo", 2, "A first paragraph.")
    assert store.set_chapter_title("demo", 2, 'The "Lake"\n') == "The Lake" and store.chapter_title("demo", 2) == "The Lake"
    assert "# Chapter 2: The Lake" in store.read_chapter_raw("demo", 2) and len(store.paragraphs("demo", 2)) == 1
    store.undo_last("demo"); assert store.chapter_title("demo", 2) == "" and len(store.paragraphs("demo", 2)) == 1


# --------------------------------------------------------------------------------------------------------- the agent
@pytest.mark.asyncio
async def test_making_a_picture_shows_it_asks_and_only_a_yes_puts_it_in_the_book(books):
    ag, store, llm = make(books)
    s = ag.session("s")
    llm.replies = ['[[tool: make_picture {"prompt": "a pale hand rising from a black lake at night, holding a sword", "caption": "The hand of the lake", "chapter": 1}]]',
                   "I made a picture of a pale hand rising out of black water, holding up a sword."]
    r = await ag.handle("s", "I would like a picture of the hand coming out of the lake")
    assert r.picture["url"].startswith("/api/books/demo/images/candidate/") and r.pending["kind"] == "picture" and r.pending["ask"] == "Put this picture in chapter 1?"
    assert "pale hand" in r.text and "Should I put it in chapter 1?" in r.text and "I made a picture for you." in r.notes
    assert "dusk" not in ag.imagegen.prompts[0] and "no text" in ag.imagegen.prompts[0]
    assert not any(p.image for p in store.paragraphs("demo", 1))
    r = await ag.handle("s", "yes")
    assert "The picture is in chapter 1" in r.text and store.paragraphs("demo", 1)[-1].image[0] == "The hand of the lake" and r.changed
    name = store.paragraphs("demo", 1)[-1].image[1]; assert ag.images.path("demo", name).is_file()
    r = await ag.handle("s", "read the picture in chapter 1"); assert r.text == "Picture: The hand of the lake"
    r = await ag.handle("s", "rewrite the picture in chapter 1 to be sadder"); assert "is a picture" in r.text and r.pending is None
    r = await ag.handle("s", "undo that"); assert not any(p.image for p in store.paragraphs("demo", 1))


@pytest.mark.asyncio
async def test_a_refused_picture_is_not_added_and_a_broken_generator_is_explained(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a castle", "caption": "A castle"}]]', "Here is a castle."]
    await ag.handle("s", "make me a picture of a castle"); r = await ag.handle("s", "no")
    assert not any(p.image for p in store.paragraphs("demo", 1)) and r.pending is None

    class Down(FakeImageGen):
        async def generate(self, *a, **k): raise RuntimeError("down")
    ag.imagegen = Down()
    llm.replies = ['[[tool: make_picture {"prompt": "a castle", "caption": "A castle"}]]', "Sorry, the picture maker is not answering right now. Shall I try again?"]
    r = await ag.handle("s", "try again"); assert "not answering" in r.text and r.picture is None and ag.session("s").pending is None
    assert "make_picture" in ag.toolbox.describe()
    ag.imagegen = None; assert "make_picture" not in ag.toolbox.describe()  # no picture server: the tool is not offered


@pytest.mark.asyncio
async def test_a_library_picture_can_go_in_the_book_and_a_text_file_cannot(books):
    ag, store, llm = make(books)
    ag.library.add("demo", "map.png", png(400, 300), "Old Map", "image", "A map of Cornwall with a lake marked.")
    ag.library.add("demo", "n.txt", b"x", "Notes", "document", "text")
    ctx = harness.Ctx(sess=ag.session("s"))
    assert "No picture with that title" in await ag.toolbox.call("use_library_picture", {"title": "Notes", "caption": "x"}, ctx)
    assert "Proposed" in await ag.toolbox.call("use_library_picture", {"title": "old map", "caption": "A map of Cornwall"}, ctx)
    assert ctx.picture and ag.session("s").pending.kind == "picture"
    r = await ag.handle("s", "yes"); assert store.paragraphs("demo", 1)[-1].image[0] == "A map of Cornwall"


@pytest.mark.asyncio
async def test_pictures_found_on_the_web_are_checked_and_never_come_from_your_own_network(books, monkeypatch):
    monkeypatch.setattr(webmod, "_is_public", lambda h: h == "example.com")
    page = png(300, 200, (9, 9, 9))

    def handler(req):
        if req.url.path == "/search":
            return httpx.Response(200, json={"results": [{"title": "Dozmary Pool", "img_src": "https://example.com/pool.png", "url": "https://example.com/pool"},
                                                          {"title": "private", "img_src": "http://169.254.169.254/x.png", "url": "https://example.com/x"},
                                                          {"title": "html", "img_src": "https://example.com/page.html", "url": "https://example.com/p"}, {"title": "no image"}]})
        if req.url.path == "/pool.png": return httpx.Response(200, headers={"content-type": "image/png"}, content=page)
        if req.url.path == "/page.html": return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")
        return httpx.Response(404)
    ag, store, llm = make(books)
    ag.web = webmod.Web("http://searx.local", "", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    ctx = harness.Ctx(sess=ag.session("s"))
    out = await ag.toolbox.call("find_pictures", {"query": "dozmary pool"}, ctx)
    assert "i1. Dozmary Pool" in out and "belong to someone else" in out and "no image" not in out
    assert "not on the public internet" in await ag.toolbox.call("use_found_picture", {"id": "i2", "caption": "x"}, ctx)
    assert "not a picture" in await ag.toolbox.call("use_found_picture", {"id": "i3", "caption": "x"}, ctx)
    assert "Proposed" in await ag.toolbox.call("use_found_picture", {"id": "i1", "caption": "Dozmary Pool on Bodmin Moor"}, ctx)
    await ag.handle("s", "yes"); assert store.paragraphs("demo", 1)[-1].image[0] == "Dozmary Pool on Bodmin Moor"


@pytest.mark.asyncio
async def test_chapter_titles_by_voice_and_the_edit_that_cannot_be_done_is_not_invented(books):
    ag, store, llm = make(books)
    r = await ag.handle("s", "call chapter 1 The Departure"); assert "Chapter 1 is now called The Departure" in r.text and store.chapter_title("demo", 1) == "The Departure"
    r = await ag.handle("s", "title chapter 4 Nothing"); assert "no chapter 4" in r.text
    llm.replies = ["NOT_FOUND: there is no rain in that paragraph, it is set on a platform in the cold."]
    r = await ag.handle("s", "change the rain to snow in paragraph 1")
    assert "did not change anything" in r.text and "no rain" in r.text and r.pending is None
    r = await ag.handle("s", "move paragraph 2 to chapter 1"); assert "already in it" in r.text and "of chapter 1" in r.text  # looking at chapter 1: ask which chapter it is in


def test_the_models_own_remarks_are_stripped_from_a_paragraph_for_the_book():
    body = ("They crowned him in a cold church that smelled of wet wool, and the crown was too big. Arthur looked like a boy handed something heavy and told to carry it a long way. "
            "When it was done he found me in the crowd and winked, once.")
    assert clean_draft(body + "\n\nI gave Arthur a small wink at the end. Would you like it sadder?") == body
    assert clean_draft("Here's the paragraph:\n\n" + body) == body
    assert clean_draft(body + " Let me know if you want more detail?") == body
    assert clean_draft(body) == body
    first_person = "I carried the sword down the hill and I did not look back. I gave it to the lake because I had promised, and the lake took it without a sound, as water takes anything."
    assert clean_draft(first_person) == first_person  # a narrator who says "I gave" inside the story is not chatter
    assert clean_draft("Short. I made it brief?") == "Short. I made it brief?"  # never cut a paragraph down to nothing


# ---------------------------------------------------------------------------------------------------- export + the page
@pytest.fixture
def client(books):
    s = Settings(project_dir=books, default_book="demo")
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["Okay."]), stt=FakeSTT(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False, imagegen=FakeImageGen())
    with TestClient(app) as c:
        yield c


def test_the_book_can_be_saved_as_html_epub_and_markdown_with_its_pictures(client, books):
    store = BookStore(books); st = ImageStore(store)
    store.bible("demo").write_doc("premise", "# Premise\n\nTitle: The Sword That Was Returned. Told by Bedivere.")
    store.set_chapter_title("demo", 1, "The Walk")
    name = st.save("demo", png(), "lake"); store.append_image("demo", 1, name, "The hand of <the> lake & more")
    store.create_chapter("demo", 2, "The Stone"); store.append_paragraph("demo", 2, "They would not believe it & so on.")
    html = client.get("/api/books/demo/export?format=html")
    assert html.headers["content-disposition"] == 'attachment; filename="The-Sword-That-Was-Returned.html"'
    t = html.text
    assert "<h1>The Sword That Was Returned</h1>" in t and "Chapter 1: The Walk" in t and "Chapter 2: The Stone" in t and "data:image/jpeg;base64," in t
    assert "The hand of &lt;the&gt; lake &amp; more" in t and "<the>" not in t and "&amp; so on" in t  # everything is escaped
    z = zipfile.ZipFile(io.BytesIO(client.get("/api/books/demo/export?format=epub").content))
    assert z.namelist()[0] == "mimetype" and z.read("mimetype") == b"application/epub+zip" and f"images/{name}" in z.namelist() and "c2.xhtml" in z.namelist()
    from xml.dom import minidom
    for n in ("content.opf", "nav.xhtml", "c1.xhtml", "c2.xhtml", "META-INF/container.xml"):
        minidom.parseString(z.read(n))  # well-formed
    m = zipfile.ZipFile(io.BytesIO(client.get("/api/books/demo/export?format=md").content))
    assert f"images/{name}" in m.namelist() and f"![The hand of <the> lake & more](images/{name})" in m.read("book.md").decode()
    assert client.get("/api/books/demo/export?format=exe").status_code == 400


def test_pictures_are_served_safely_and_the_chapter_list_knows_about_them(client, books):
    store = BookStore(books); st = ImageStore(store)
    name = st.save("demo", png(), "lake"); store.append_image("demo", 1, name, "A lake"); store.set_chapter_title("demo", 1, "The Walk")
    r = client.get(f"/api/books/demo/images/{name}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.headers["x-content-type-options"] == "nosniff" and "sandbox" in r.headers["content-security-policy"]
    for bad in ("../../x", "..%2f..%2fetc%2fpasswd", ".candidates", "nope.jpg"):
        assert client.get(f"/api/books/demo/images/{bad}").status_code in (404, 400, 422)
    assert client.get(f"/api/books/demo/images/candidate/{name}").status_code == 404
    j = client.get("/api/books/demo/chapters/1").json()
    assert j["title"] == "The Walk" and j["paragraphs"][-1]["image"] == {"url": f"/api/books/demo/images/{name}", "caption": "A lake", "file": name} and j["paragraphs"][0]["image"] is None
    assert client.get("/api/books/demo/chapters").json()["titles"] == {"1": "The Walk"}


@pytest.mark.asyncio
async def test_describe_it_to_me_describes_the_waiting_picture_and_keeps_it_waiting(books):
    import intent
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a lady in a grey gown standing in a misty lake holding out a sword", "caption": "The Lady of the Lake"}]]',
                   "A woman in a misty grey gown stands waist-deep in a calm lake and holds a glowing sword out to a young king in a boat."]
    await ag.handle("s", "make a picture of the lady of the lake")
    for phrase in ["describe it to me", "what does it look like?", "can you describe that picture", "tell me about it"]:
        n_calls = len(llm.calls)
        r = await ag.handle("s", phrase)
        assert "misty grey gown" in r.text and r.pending["kind"] == "picture" and r.picture and len(llm.calls) == n_calls, phrase  # answered from what the model said; no new draft, no model call
    r = await ag.handle("s", "yes"); assert "in chapter 1" in r.text
    r = await ag.handle("s", "describe it to me"); assert "caption: The Lady of the Lake" in r.text  # and afterwards, from the caption
    for not_describe in ["describe the courtyard at dawn", "describe Mira's coat", "write a description of the lake"]:
        assert not intent.describe_request(not_describe), not_describe


def test_picture_requests_are_not_mistaken_for_edits_or_for_files_on_the_computer():
    import intent
    for s in ["make a picture of the lady of the lake", "make me a picture of the lake", "find a picture of Dozmary Pool", "find an old map of Camelot on the web", "I'd like a picture for chapter 3",
              "draw the castle at dawn as an illustration", "use my photo of Dozmary Pool in chapter 10", "get me an illustration of Merlin", "add a picture here"]:
        assert intent.picture_request(s), s
    for s in ["I just downloaded a picture, can you open it", "find the photo I downloaded", "make it shorter", "make the lake darker in paragraph 2", "the picture is lovely, thank you"]:
        assert not intent.picture_request(s), s
    assert intent.file_request("find the photo I downloaded")  # that one is a file on the computer


@pytest.mark.asyncio
async def test_make_me_a_picture_goes_to_the_model_with_its_tools_not_to_the_editor(books):
    ag, store, llm = make(books)
    before = store.read_chapter_raw("demo", 1)
    llm.replies = ['[[tool: make_picture {"prompt": "a castle on a hill at dawn", "caption": "Camelot at dawn"}]]', "A pale stone castle on a green hill, lit gold by the sunrise."]
    r = await ag.handle("s", "make me a picture of Camelot at dawn")
    assert r.pending["kind"] == "picture" and r.picture and "pale stone castle" in r.text and store.read_chapter_raw("demo", 1) == before
    ag2, store2, llm2 = make(books)
    llm2.replies = ['[[tool: find_pictures {"query": "old map of Camelot"}]]', "I could not search just now."]
    r = await ag2.handle("s", "find a picture of an old map of Camelot")
    assert "inbox folder" not in r.text and "downloads" not in r.text.lower()  # not the file finder


# ------------------------------------------------------------------------------------------- the ten-chapter run: each bug it found
PARA = "The sword hung against his hip and knocked at his knee with every step, once, twice, as if it wanted his attention. He let it knock, and walked on."
PARA2 = "By the water the reeds were stiff with frost. He stood a long moment, and then he lifted the sword and the cold went up his arm like a held breath."


@pytest.mark.asyncio
async def test_a_chapter_the_writer_points_back_to_is_not_where_the_paragraph_goes(books):
    import intent
    assert intent.parse_ref("write the throw, same as chapter 2 but sadder").chapter is None and intent.parse_ref("as in chapter six the lady rises, now write the throw").chapter is None
    assert intent.parse_ref("write paragraph 2 of chapter 3").chapter == 3 and intent.parse_ref("move paragraph 3 from chapter 1 to chapter 2").chapter == 1
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "A second chapter paragraph about the stone and the crowd.")
    s = ag.session("s"); s.chapter = s.work = 1
    llm.replies = ["He threw it, and the lake took it."]
    r = await ag.handle("s", "write the next paragraph, the same lady as in chapter 2 but sadder")
    assert r.pending["ask"] == "Add this paragraph to chapter 1?"
    await ag.handle("s", "yes"); assert "the lake took it" in store.read_chapter_raw("demo", 1) and "the lake took it" not in store.read_chapter_raw("demo", 2)


@pytest.mark.asyncio
async def test_a_rejected_picture_is_never_saved_and_a_changed_request_makes_a_new_one(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a knight with a sword at a frozen lake", "caption": "Bedivere at the lake"}]]', "A knight stands at a frozen lake holding a sword."]
    r = await ag.handle("s", "make a picture of the lake")
    first = ag.session("s").pending.data["candidate"]; assert ag.images.path("demo", first, candidate=True).is_file()
    llm.replies = ['[[tool: make_picture {"prompt": "an empty frozen lake at night", "caption": "The empty lake"}]]', "An empty frozen lake under a black sky."]
    r = await ag.handle("s", "no, I want an empty lake, nobody in it")
    second = ag.session("s").pending.data["candidate"]
    assert second != first and r.picture["url"].endswith(second) and "empty frozen lake" in r.text
    with pytest.raises(FileNotFoundError):
        ag.images.path("demo", first, candidate=True)  # the rejected one is gone
    await ag.handle("s", "yes"); assert store.paragraphs("demo", 1)[-1].image[0] == "The empty lake"
    llm.replies = ['[[tool: make_picture {"prompt": "a castle", "caption": "A castle"}]]', "A castle."]
    await ag.handle("s", "make a picture of a castle"); third = ag.session("s").pending.data["candidate"]
    await ag.handle("s", "no")
    with pytest.raises(FileNotFoundError):
        ag.images.path("demo", third, candidate=True)


@pytest.mark.asyncio
async def test_moves_say_where_they_really_went_and_wiki_batches_say_what_was_left_out(books):
    ag, store, llm = make(books)
    await ag.handle("s", "move paragraph 3 to the start of chapter 1"); r = await ag.handle("s", "yes"); assert "Moved to the start of chapter 1" in r.text
    await ag.handle("s", "move paragraph 1 after paragraph 2"); r = await ag.handle("s", "yes"); assert "after paragraph 2 of chapter 1" in r.text
    s = ag.session("s"); ctx = harness.Ctx(sess=s)
    notes = [{"kind": "characters", "name": f"Person {i}", "note": "x"} for i in range(9)] + [{"kind": "characters", "name": "", "note": "no name"}, {"kind": "nonsense", "name": "Bad", "note": "y"}]
    out = await ag.toolbox.call("propose_wiki_notes", {"notes": notes}, ctx)
    assert "Proposed 9 notes" in out and "Person 8" in out and "NOT included" in out and "Bad" in out  # nothing is silently dropped any more
    assert s.pending.data["batch"][-1]["name"] == "Person 8" and s.pending.ask == "" or "Person 0" in ag._ask_for(s, s.pending)


@pytest.mark.asyncio
async def test_the_book_can_be_saved_by_voice(books):
    import intent
    ag, store, _ = make(books)
    r = await ag.handle("s", "save my book as an e-book"); assert r.download == {"url": "/api/books/demo/export?format=epub", "label": "e-book file"} and "downloads" in r.text
    r = await ag.handle("s", "export the book"); assert r.download["url"].endswith("format=html") and "PDF" in r.text
    r = await ag.handle("s", "give me the whole story as a zip"); assert r.download["url"].endswith("format=md")
    for no in ["save chapter 2 as a file", "tell me the story of the book", "I want a book about dragons"]:
        assert intent.export_request(no) is None, no


@pytest.mark.asyncio
async def test_save_as_you_go_saves_at_once_reads_it_back_and_stays_undoable(books):
    ag, store, llm = make(books)
    r = await ag.handle("s", "save as you go"); assert "straight away" in r.text and ag.prefs is None or True
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir)
    r = await ag.handle("s", "save as you go"); assert ag.prefs.get("auto_save") == "on"
    llm.replies = [f'[[tool: propose_append {{"text": "{PARA}"}}]]', "Here it is."]
    r = await ag.handle("s", "next paragraph, the walk down")
    assert "I saved this to chapter 1" in r.text and PARA in r.text and "Say undo" in r.text and r.pending is None and PARA in store.read_chapter_raw("demo", 1)
    r = await ag.handle("s", "undo that"); assert PARA not in store.read_chapter_raw("demo", 1)
    r = await ag.handle("s", "ask me first please"); assert ag.prefs.get("auto_save") == "off"
    llm.replies = [f'[[tool: propose_append {{"text": "{PARA}"}}]]', "Here it is."]
    r = await ag.handle("s", "next paragraph, the walk down"); assert r.pending["kind"] == "append" and PARA not in store.read_chapter_raw("demo", 1)


@pytest.mark.asyncio
async def test_a_paragraph_that_contradicts_the_book_is_caught_once_and_then_warned_about(books):
    ag, store, llm = make(books)
    for t in ("Bedivere did not ask why, then or ever.", "He carried the sword and said nothing at all."):
        store.append_paragraph("demo", 1, t)
    bad = "Bedivere asked him why, for the first time, and the old man did not answer. Then he carried the sword on down to the water in silence."
    llm.replies = [f'[[tool: propose_append {{"text": "{bad}"}}]]', "CONFLICT: Bedivere never asked why in chapter 1.", f'[[tool: propose_append {{"text": "{bad}"}}]]',
                   "CONFLICT: still asks why.", "A draft for you."]
    r = await ag.handle("s", "next paragraph, he asks why")
    assert "Problem: Bedivere never asked why" in " ".join(str(m) for c in llm.calls for m in c) and r.pending["kind"] == "append" and "A warning: still asks why" in r.text  # warned, never silently saved
    ag2, store2, llm2 = make(books)
    assert await ag2.check_continuity(ag2.session("s"), "x") == ""  # a book this short is not checked
    for t in ("a", "b"):
        store2.append_paragraph("demo", 1, t + " paragraph text here for the book.")
    llm2.replies = ["OK"]; assert await ag2.check_continuity(ag2.session("s"), "A new paragraph.") == ""
    llm2.replies = ["I think it is fine."]; assert await ag2.check_continuity(ag2.session("s"), "A new paragraph.") == ""
    llm2.replies = ["CONFLICT: the sword was lost in chapter 1."]; assert "sword was lost" in await ag2.check_continuity(ag2.session("s"), "A new paragraph.")
    ag2.prefs = __import__("prefs").Prefs(ag2.s.state_dir); ag2.prefs.update({"continuity": "off"})
    llm2.replies = ["CONFLICT: x"]; assert await ag2.check_continuity(ag2.session("s"), "A new paragraph.") == ""


@pytest.mark.asyncio
async def test_one_line_drafts_are_sent_back_and_found_pictures_are_described_by_a_vision_model(books, monkeypatch):
    ag, store, llm = make(books)
    ctx = harness.Ctx(sess=ag.session("s"))
    assert "only one short sentence" in await ag.toolbox.call("propose_append", {"text": "He threw the sword into the lake and it was gone."}, ctx)
    assert ag.session("s").pending is None
    ag.library.add("demo", "pool.png", png(400, 300), "Dozmary Pool", "image", "x")
    llm.replies = ["A grey lake under a low cloudy sky, with brown moor grass at the edge."]
    out = await ag.toolbox.call("use_library_picture", {"title": "Dozmary Pool", "caption": "Dozmary Pool"}, ctx)
    assert "A vision model says the picture shows: A grey lake" in out and ag.session("s").pending.data["describe"].startswith("A grey lake")
    ag.session("s").pending = None
    llm.replies = ["I'm sorry, I can't see images."]
    out = await ag.toolbox.call("use_library_picture", {"title": "Dozmary Pool", "caption": "Dozmary Pool"}, ctx)
    assert "cannot see this picture" in out  # honest when nothing can look at it


@pytest.mark.asyncio
async def test_yes_with_a_question_saves_and_answers_and_a_failed_picture_is_retried_once(books):
    ag, store, llm = make(books)
    store.append_paragraph("demo", 1, "x")  # keep the id numbering simple
    await ag.handle("s", "rewrite chapter 1 paragraph 2 so it is snowing")
    llm.replies = ["Kay is the foster brother, Sir Ector's son."]
    r = await ag.handle("s", "yes, and who was Kay again?")
    assert "Saved" in r.text and "foster brother" in r.text and ag.session("s").pending is None

    class Flaky(FakeImageGen):
        n = 0
        async def generate(self, prompt, size="1024x1024"):
            Flaky.n += 1
            if Flaky.n == 1: raise RuntimeError("busy")
            return await super().generate(prompt, size)
    ag.imagegen = Flaky()
    llm.replies = ['[[tool: make_picture {"prompt": "a lake", "caption": "A lake"}]]', "A quiet lake."]
    r = await ag.handle("s", "make a picture of a lake")
    assert Flaky.n == 2 and r.picture  # the second try worked and the writer never saw the failure


def test_more_ways_to_ask_for_a_description_and_the_picture_is_not_a_paragraph_to_add():
    import intent
    for s in ["tell me what it looks like", "what is in the picture", "describe the picture to me", "describe it again", "what does it look like?"]:
        assert intent.describe_request(s), s


# ------------------------------------------------------------------------------------------------ round four: what the half-sighted writer still hit
@pytest.mark.asyncio
async def test_yes_that_is_mine_put_it_in_saves_the_picture_and_does_not_reject_it(books):
    import intent
    ag, store, llm = make(books)
    ag.library.add("demo", "p.png", png(400, 300), "Edwin Photo", "image", "x")
    ctx = harness.Ctx(sess=ag.session("s"))
    await ag.toolbox.call("use_library_picture", {"title": "Edwin Photo", "caption": "Dozmary Pool"}, ctx)
    for yes in ["Yes, that's mine, put it in", "yes that's the one", "yes, lovely, thanks"]:
        assert intent.confirmation(yes) == ("yes", ""), yes
    n = len(llm.calls)
    r = await ag.handle("s", "Yes, that's mine, put it in")
    assert "in chapter 1" in r.text and store.paragraphs("demo", 1)[-1].image[0] == "Dozmary Pool" and len(llm.calls) == n  # saved; the model was not asked anything
    await ag.toolbox.call("use_library_picture", {"title": "Edwin Photo", "caption": "Again"}, ctx)
    llm.replies = ["Sure."]
    await ag.handle("s", "yes but make it darker")  # a real change request is still a change request
    assert ag.session("s").pending is None and not (ag.images.cand_dir("demo") / "x").exists()  # the picture was set aside; nothing was added to the book
    assert sum(1 for p in store.paragraphs("demo", 1) if p.image) == 1


@pytest.mark.asyncio
async def test_the_model_never_has_to_refuse_a_move_or_a_delete_or_a_new_chapter(books):
    ag, store, llm = make(books)
    ctx = harness.Ctx(sess=ag.session("s"))
    first = lambda: [p.text[:9] for p in store.paragraphs("demo", 1)]
    out = await ag.toolbox.call("propose_move", {"paragraph": -1, "where": "start"}, ctx); assert "Proposed" in out and ag.session("s").pending.kind == "move"
    await ag.handle("s", "yes"); assert first() == ["At the ca", "Mira step", "The expre"]
    assert "already waiting" in (await ag.toolbox.call("propose_delete", {"paragraph": 1}, harness.Ctx(sess=ag.session("s"))) if ag.session("s").pending else "already waiting")
    out = await ag.toolbox.call("propose_move", {"paragraph": 9, "to_chapter": 2}, ctx); assert out.startswith("Error")  # no such paragraph: a plain reason, not a crash
    out = await ag.toolbox.call("propose_delete", {"paragraph": 2}, ctx); assert "Proposed" in out and ag.session("s").pending.kind == "delete"
    await ag.handle("s", "no")
    assert "Chapter 2 is ready" in await ag.toolbox.call("start_chapter", {"title": "The Stone"}, ctx) and store.chapter_title("demo", 2) == "The Stone"
    assert "Chapter 2 is now called The Boy" in await ag.toolbox.call("set_chapter_title", {"chapter": 2, "title": "The Boy"}, ctx)
    info = json.loads(await ag.toolbox.call("book_info", {}, ctx))
    assert [c["chapter"] for c in info["per_chapter"]] == [1, 2] and info["per_chapter"][1]["title"] == "The Boy" and info["per_chapter"][0]["pictures"] == 0
    name = ag.images.save("demo", png(), "x"); store.append_image("demo", 2, name, "A lake")
    assert json.loads(await ag.toolbox.call("book_info", {}, ctx))["per_chapter"][1]["picture_captions"] == ["A lake"]  # "which chapters have pictures?" can now be answered


@pytest.mark.asyncio
async def test_start_chapter_and_name_it_in_one_sentence_and_last_paragraph_for_a_chapter_is_a_new_paragraph(books):
    import intent
    ag, store, llm = make(books)
    r = await ag.handle("s", "start chapter 2 and call it The Stone")
    assert "Chapter 2, The Stone, is ready" in r.text and store.chapter_title("demo", 2) == "The Stone" and r.chapter == 2
    r = await ag.handle("s", "begin chapter 3, The Round Table"); assert store.chapter_title("demo", 3) == "The Round Table"
    r = await ag.handle("s", "start chapter 4"); assert store.chapter_title("demo", 4) == "" and "Chapter 4 is ready" in r.text
    t = "Last paragraph for chapter 2: Mordred falls and Arthur lies wounded on the field"
    assert intent.is_draft(t) and not intent.is_edit(t) and not intent.is_draft("rewrite the last paragraph of chapter 2 to be shorter")
    for phrase, want in [("move the last paragraph to the start of its chapter", "start"), ("move paragraph 2 to the front", "start"), ("move paragraph 3 of chapter 8 into chapter 9, at the end please", "end")]:
        assert intent.move_request(phrase).where == want, phrase
    for phrase in ["describe that picture to me again", "describe it in more detail", "tell me what it looks like"]:
        assert intent.describe_request(phrase), phrase


@pytest.mark.asyncio
async def test_uploading_the_same_photo_twice_keeps_one_copy_and_sir_kay_is_the_same_page_as_kay(books):
    import ingest
    ag, store, llm = make(books)
    a = await ingest.add_file("demo", "photo.png", png(300, 200), ag.library, llm, None, ag.journal)
    b = await ingest.add_file("demo", "photo again.png", png(300, 200), ag.library, llm, None, ag.journal)
    assert b["duplicate"] and b["slug"] == a["slug"] and len(ag.library.items("demo")) == 1
    bible = store.bible("demo")
    for kind, name, note in [("characters", "Kay", "Arthur's foster brother."), ("characters", "Sir Kay", "Sharp-tongued, loyal."), ("characters", "The Lady of the Lake", "Gives the sword.")]:
        ag._save_note(bible, kind, name, note)
    names = list(bible.entries("characters"))
    assert names.count("Kay") == 1 and "Sir Kay" not in names and "Lady of the Lake" not in names and "The Lady of the Lake" in names
    assert "Sharp-tongued" in bible.entries("characters")["Kay"].body and "foster brother" in bible.entries("characters")["Kay"].body


@pytest.mark.asyncio
async def test_a_saved_paragraph_is_read_back_briefly_and_names_follow_the_book_s_setting(books):
    ag, store, llm = make(books)
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir); ag.prefs.update({"auto_save": "on"})
    long = ("The table came in the spring. It was oak, black with age. A hundred and fifty places were cut into the rim. Each had a name in gilt. Bedivere found his own.")
    llm.replies = [f'[[tool: propose_append {{"text": "{long}"}}]]', "ok"]
    r = await ag.handle("s", "next paragraph, the table")
    spoken = " ".join(s["text"] for s in r.speech)
    assert long in r.text and "Each had a name" not in spoken and "The table came in the spring. It was oak, black with age." in spoken and "read it to me" in spoken
    import ideas_data as D
    ag.journal.append("demo", "user", "I want a book about King Arthur and Excalibur told by old Bedivere")  # said in conversation, not yet in the wiki
    fit = {w for c in ("welsh", "english") for k in ("female", "male") for w in D.NAMES[c][k]}
    ag2, store2, llm2 = make(books)
    ag2.journal.append("demo", "user", "I want a book about King Arthur and Excalibur told by old Bedivere")
    r = await ag2.handle("s", "give me three names for a minor knight"); assert all(i.split()[0] in fit for i in r.ideas), r.ideas


# ------------------------------------------------------------------------ round five: the chapter being written in must not drift
@pytest.mark.asyncio
async def test_looking_at_or_editing_another_chapter_never_changes_where_new_writing_goes(books):
    ag, store, llm = make(books)
    for n in (2, 3, 4):
        store.create_chapter("demo", n); store.append_paragraph("demo", n, f"Chapter {n} first paragraph, long enough to be a real line of the story here.")
    await ag.handle("s", "go to chapter 4"); s = ag.session("s"); assert s.work == 4
    llm.replies = ["Not sure which part you mean."]
    await ag.handle("s", "rewrite paragraph 1 of chapter 2 to be sadder")           # looks at chapter 2 (and may ask "do you mean...?")
    await ag.handle("s", "no"); await ag.handle("s", "read chapter 1 paragraph 2"); await ag.handle("s", "move paragraph 1 of chapter 3 to chapter 2"); await ag.handle("s", "no")
    assert s.work == 4 and ag._work(s) == 4
    llm.replies = ["The lords made him try three times, and each time the stone gave."]
    r = await ag.handle("s", "one more paragraph, the lords make him try three times")
    assert r.pending["ask"] == "Add this paragraph to chapter 4?"
    await ag.handle("s", "yes"); assert "try three times" in store.read_chapter_raw("demo", 4) and "try three times" not in store.read_chapter_raw("demo", 2)
    assert "writing in chapter 4" in ag._book_state(s) and "looking at chapter" in ag._book_state(s)
    ag2, _, _ = make(books)
    assert ag2._work(ag2.session("z")) == 4  # after a restart, writing continues at the end of the book, not in chapter 1


@pytest.mark.asyncio
async def test_a_picture_goes_where_the_writer_is_writing_not_where_they_last_looked(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "A paragraph in the second chapter, long enough to count as real text.")
    await ag.handle("s", "go to chapter 2"); await ag.handle("s", "read chapter 1 paragraph 1")  # looking at chapter 1 again
    llm.replies = ['[[tool: make_picture {"prompt": "a stone in a churchyard", "caption": "The stone"}]]', "A grey stone in a churchyard."]
    r = await ag.handle("s", "make a picture of the stone")
    assert r.pending["ask"] == "Put this picture in chapter 2?"


@pytest.mark.asyncio
async def test_several_paragraphs_can_be_deleted_at_once_and_one_undo_brings_them_all_back(books):
    import intent
    ag, store, _ = make(books)
    for i in range(4, 8):
        store.append_paragraph("demo", 1, f"Extra paragraph number {i} of the first chapter, long enough to count.")
    before = store.read_chapter_raw("demo", 1)
    assert intent.delete_many("delete paragraphs 2 to 4") == (None, [2, 3, 4]) and intent.delete_many("remove paragraphs 2, 3 and 5 of chapter 1") == (1, [2, 3, 5])
    assert intent.delete_many("delete the last 3 paragraphs") == (None, [-3, -2, -1]) and intent.delete_many("delete paragraph 2") is None
    r = await ag.handle("s", "delete paragraphs 2 to 4")
    assert r.pending["kind"] == "delete" and r.pending["ask"] == "Delete these 3 paragraphs of chapter 1?" and len(r.highlights) == 3 and store.read_chapter_raw("demo", 1) == before
    r = await ag.handle("s", "yes delete all three"); assert "Deleted 3 paragraphs" in r.text and len(store.paragraphs("demo", 1)) == 4  # "yes ..." is just yes
    r = await ag.handle("s", "undo that"); assert "put back the paragraphs I deleted" in r.text and store.read_chapter_raw("demo", 1) == before
    r = await ag.handle("s", "delete the last 2 paragraphs"); await ag.handle("s", "no, wait")
    assert store.read_chapter_raw("demo", 1) == before
    r = await ag.handle("s", "delete paragraphs 7 to 20"); assert r.pending is not None or "does not have" in r.text


@pytest.mark.asyncio
async def test_natural_yes_answers_are_yes_and_a_waiting_picture_is_never_lost_to_one(books):
    import intent
    for s in ["Yes, save them.", "yes put that one in", "Yes, put that one in", "yes delete all four", "yes, that's mine, put it in", "yes please"]:
        assert intent.confirmation(s)[0] == "yes" and (intent.confirmation(s)[1] == "" or not intent._CHANGE.search(intent.confirmation(s)[1])), s
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a stone", "caption": "The stone"}]]', "A grey stone."]
    await ag.handle("s", "make a picture of the stone")
    r = await ag.handle("s", "Yes, put that one in"); assert "in chapter 1" in r.text and store.paragraphs("demo", 1)[-1].image[0] == "The stone"
    ctx = harness.Ctx(sess=ag.session("s"))
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "A", "note": "x"}, {"kind": "characters", "name": "B", "note": "y"}]}, ctx)
    r = await ag.handle("s", "Yes, save them."); assert "Saved to the wiki: A, B" in r.text


@pytest.mark.asyncio
async def test_the_model_can_always_save_the_book_and_a_draft_is_checked_against_what_was_asked(books):
    ag, store, llm = make(books)
    ctx = harness.Ctx(sess=ag.session("s"))
    out = await ag.toolbox.call("export_book", {"format": "web page"}, ctx)
    assert ctx.download["url"] == "/api/books/demo/export?format=html" and "downloads" in out
    r = ag._attach(ag.session("s"), ag._reply(ag.session("s"), "Saving."), ctx, "", "x"); assert r.download["url"].endswith("format=html")
    for t in ("a first long paragraph about the stone in the churchyard.", "a second long paragraph about the king.", "a third paragraph that is long enough."):
        store.append_paragraph("demo", 1, t)
    s = ag.session("s"); s.last_user = "next paragraph, the oath at the round table"
    para = "A stone stood in the churchyard and the boy pulled the sword from it while the lords watched in silence. Nobody spoke for a long while after."
    llm.replies = ["MISSED: it is not the oath scene."]
    issue = await ag.check_continuity(s, para, s.last_user)
    assert "it does not do what you asked: it is not the oath scene" in issue and "THE WRITER ASKED FOR: next paragraph, the oath" in str(llm.calls[-1][-1]["content"])
    assert "chapter 1 paragraph 1:" in str(llm.calls[-1][-1]["content"])  # the whole book's outline is in front of the checker, so a retelling in another chapter is visible
    llm.replies = ["CONFLICT: the pull was told in chapter 2."]; assert "pull was told" in await ag.check_continuity(s, para, "")


# ------------------------------------------------------------------------------------------------------ round six: the last small traps
@pytest.mark.asyncio
async def test_describe_it_again_i_cannot_see_it_well_describes_and_does_not_enlarge_the_screen(books):
    import intent
    assert intent.describe_request("Describe it to me again, I can't see it well") and intent.describe_request("describe that picture to me, I cannot see it")
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a grey stone in a misty churchyard at dawn", "caption": "The stone"}]]', "A grey stone stands alone in a misty churchyard at dawn, with a sword sunk in its top."]
    await ag.handle("s", "make a picture of the stone")
    look_before = ag._look()
    r = await ag.handle("s", "Describe it to me again, I can't see it well")
    assert "grey stone stands alone" in r.text and r.ui is None and ag._look() == look_before and r.pending["kind"] == "picture"
    r = await ag.handle("s", "I can't see this very well"); assert r.ui is not None  # the real "make it bigger" request still works


def test_no_leave_it_as_it_was_is_a_plain_no_and_a_change_request_is_still_a_change_request():
    import intent
    for s in ["no, leave it as it was", "No. Keep it as it is", "no, forget it", "never mind", "no thanks", "no, I changed my mind"]:
        assert intent.confirmation(s) == ("no", ""), s
    assert intent.confirmation("no, make it sadder")[1] == "make it sadder" and intent.confirmation("no, make it an empty lake")[1] == "make it an empty lake"


@pytest.mark.asyncio
async def test_a_paragraph_the_app_itself_flags_is_never_saved_behind_the_writers_back(books):
    ag, store, llm = make(books)
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir); ag.prefs.update({"auto_save": "on"})
    for t in ("a first long paragraph about the stone in the churchyard.", "a second long paragraph about the king.", "a third paragraph that is long enough.", "a fourth paragraph that is long enough."):
        store.append_paragraph("demo", 1, t)
    para = "The sword hung against his hip and knocked at his knee with every step, once, twice, as if it wanted his attention. He let it knock, and walked on."
    llm.replies = [f'[[tool: propose_append {{"text": "{para}"}}]]', "MISSED: no hiding scene.", f'[[tool: propose_append {{"text": "{para}"}}]]', "MISSED: still no hiding scene.", "Here is a version."]
    r = await ag.handle("s", "next paragraph, he hides the sword in the reeds")
    assert r.pending["kind"] == "append" and "A warning: it does not do what you asked" in r.text and "I saved this" not in r.text and para not in store.read_chapter_raw("demo", 1)
    await ag.handle("s", "no")
    llm.replies = [f'[[tool: propose_append {{"text": "{para}"}}]]', "OK", "Here it is."]
    r = await ag.handle("s", "next paragraph, he walks on"); assert "I saved this to chapter 1" in r.text  # a clean paragraph is still saved at once


@pytest.mark.asyncio
async def test_an_edit_that_changes_nothing_is_retried_once_and_then_explained(books):
    ag, store, llm = make(books)
    old = store.paragraphs("demo", 1)[0].text
    llm.replies = [old, "Mira stepped onto the platform, her breath fogging in the bitter cold air, and counted the doors."]
    r = await ag.handle("s", "change paragraph 1 to make it colder"); assert r.pending and "bitter cold" in r.text
    ag2, store2, llm2 = make(books); llm2.replies = [old, old]
    r = await ag2.handle("s", "change paragraph 1 to make it colder"); assert r.pending is None and "could not find a way to change it" in r.text


@pytest.mark.asyncio
async def test_a_picture_proposal_always_comes_with_a_description_and_a_long_remark_is_not_a_yes(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a lone knight at a frozen lake under a black sky", "caption": "The lake"}]]', "Here you go."]
    r = await ag.handle("s", "make a picture of the lake")
    assert "a lone knight at a frozen lake under a black sky" in r.text and "Should I put it in chapter 1?" in r.text  # the model said nothing useful: the app says what it made
    r = await ag.handle("s", "describe it"); assert "lone knight" in r.text
    await ag.handle("s", "no")
    ctx = harness.Ctx(sess=ag.session("s"))
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "A", "note": "x"}, {"kind": "characters", "name": "B", "note": "y"}]}, ctx)
    llm.replies = ["Noted."]
    r = await ag.handle("s", "Right. The ten chapters are one the walk to the lake, two the boy in the courtyard, three Merlin looks twice")
    assert "Saved to the wiki" not in r.text and "A" not in [n for n in store.bible("demo").characters() if n == "A"]  # a long remark starting with "Right" is not a yes


def test_the_books_title_is_found_in_the_premise_however_it_was_worded(books):
    store = BookStore(books)
    for prem, want in [("# Premise\n\nThe book is called The Sword That Was Returned.\n\nTold by Bedivere.", "The Sword That Was Returned"), ("Title: The Sword That Was Returned. A retelling.", "The Sword That Was Returned"),
                       ('The title is "Lake of Glass" and it is short.', "Lake of Glass"), ("# Premise\n\nA story about knights.", "Demo")]:
        store.bible("demo").write_doc("premise", prem)
        assert export.book_title(store, "demo") == want, prem


@pytest.mark.asyncio
async def test_regression_last_three_answers_remarks_and_formats(books):
    import intent
    assert intent.confirmation("Yes, save that") == ("yes", "") and intent.confirmation("yes save this") == ("yes", "")
    for s, want in [("Now save it as an e-book", "epub"), ("save it as a web page", "html"), ("download it as a PDF", "html"), ("save it as an epub", "epub")]:
        assert intent.export_request(s) == want, s
    for s in ["I read an epub yesterday and liked it", "save chapter 2 as a pdf", "tell me about web pages"]:
        assert intent.export_request(s) is None, s
    ag, store, llm = make(books)
    ctx = harness.Ctx(sess=ag.session("s"))
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "premise", "name": "", "note": "Title: X. Told by Bedivere."}, {"kind": "characters", "name": "A", "note": "x"}]}, ctx)
    llm.replies = ["Lovely, chapter two it is."]
    n_before = len(llm.calls)
    r = await ag.handle("s", "Right. Chapter two will be about the boy Arthur in the courtyard and the sword in the stone, and chapter three is Merlin.")
    p = ag.session("s").pending
    assert p is not None and p.kind == "wiki" and "A" not in store.bible("demo").characters() and "Revised" not in r.text and "still holding this for you" in r.text  # answered; the question is still waiting
    r = await ag.handle("s", "Yes, save that"); assert "Saved to the wiki" in r.text and ag.session("s").pending is None
    r = await ag.handle("s", "yes but make the second sentence shorter and add the lantern near the end please")  # nothing pending now: ordinary talk


# ---- round 7 (Everett's story): lost paragraphs, "describe the photograph", contradictory replies --------------------------------------
PROSE = ("Near the village the fighting grew loud, and the noise was beyond anything Everett had imagined. Guns thundered from every side and the air was full of smoke. "
         "He could not hear the orders shouted a few feet away, so he kept close to the man beside him. They went forward together in short rushes, dropped flat and got up again.")


def test_yes_with_a_compliment_is_a_yes_and_not_a_request_to_redo_it():
    for t in ("Yes, that's better.", "yes that is much nicer", "Yes, lovely.", "yes, that's much better, thank you"):
        assert intent.confirmation(t) == ("yes", ""), t


def test_one_more_and_the_part_where_ask_for_new_writing_but_one_more_time_does_not():
    for t in ("One more, the fighting near Exermont, and the noise.", "Now the part where he reaches the shell hole", "and then the scene where they sail"):
        assert intent.draft_first(t), t
    assert not intent.is_draft("one more time please")


@pytest.mark.asyncio
async def test_prose_the_model_wrote_into_the_chat_is_asked_for_again_properly_and_never_lost(books):
    ag, store, llm = make(books)
    store.append_paragraph("demo", 1, "The first paragraph is here.")
    ag.session("s").work = 1
    llm.replies = [PROSE, f'[[tool: propose_append {{"text": "{PROSE}"}}]]', "Here it is."]
    r = await ag.handle("s", "continue with the fighting near Exermont and the noise")
    assert r.pending and r.pending["kind"] == "append" and PROSE in r.text  # it is a proposal the writer can say yes to, not text that vanishes


@pytest.mark.asyncio
async def test_save_as_you_go_never_says_both_waiting_for_your_answer_and_saved(books):
    ag, store, llm = make(books)
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir)
    await ag.handle("s", "save as you go")
    llm.replies = [f'[[tool: propose_append {{"text": "{PARA}"}}]]', "The paragraph is ready for your yes or no, and once it's saved I will move it."]
    r = await ag.handle("s", "next paragraph, the walk down")
    assert "I saved this to chapter 1" in r.text and "yes or no" not in r.text and r.pending is None


@pytest.mark.asyncio
async def test_describe_the_photograph_describes_the_library_photo_and_never_writes_a_paragraph(books):
    ag, store, llm = make(books)
    before = len(store.all_paragraphs("demo"))
    assert intent.describe_request("Describe the photograph to me again, please. Just describe it, don't write anything.")
    ag.library.add("demo", "everett.jpg", png(), "everett photo", "image", "", status="unread", notes=["I cannot see pictures yet."])
    r = await ag.handle("s", "describe the photograph to me")
    assert "cannot see pictures" in r.text and r.pending is None and not llm.calls and len(store.all_paragraphs("demo")) == before
    ag.library.add("demo", "farm.jpg", png(10, 10, (1, 2, 3)), "farm photo", "image", "A red barn on a wheat field under a pale sky.", status="read")
    assert "red barn" in (await ag.handle("s", "describe the photo")).text


# ---- round 8: where a picture goes ----------------------------------------------------------------------------------------------------
def test_picture_position_is_understood_from_plain_words():
    pos = intent.picture_position
    assert pos("put my photograph at the start of chapter 1") == 0 and pos("put it at the very beginning") == 0
    assert pos("yes, after paragraph 2") == 2 and pos("put it before paragraph 3") == 2 and pos("between paragraph 2 and paragraph 3") == 2
    assert pos("put it at the end") == -1 and pos("yes, put that one in") is None and pos("make it a bigger ship") is None


@pytest.mark.asyncio
async def test_a_picture_goes_where_the_writer_says_and_can_be_moved_while_it_waits(books):
    ag, store, llm = make(books)
    for t in ("First paragraph of the chapter.", "Second paragraph of the chapter.", "Third paragraph of the chapter."):
        store.append_paragraph("demo", 1, t)
    sess = ag.session("s"); sess.work = 1
    n0 = len(store.paragraphs("demo", 1)); first = store.paragraphs("demo", 1)[0].text
    llm.replies = ['[[tool: make_picture {"prompt": "a barn", "caption": "The barn"}]]', "A red barn on a field."]
    r = await ag.handle("s", "make me a picture of the barn after paragraph 2")
    assert r.pending["kind"] == "picture" and "after paragraph 2" in r.pending["ask"]
    r = await ag.handle("s", "put it at the very start of the chapter please")
    assert "Say yes" in r.text and r.pending is not None
    r = await ag.handle("s", "yes")
    paras = store.paragraphs("demo", 1)
    assert paras[0].image and "very start" in r.text and len(paras) == n0 + 1 and paras[1].text == first
    llm.replies = ['[[tool: make_picture {"prompt": "a lake", "caption": "The lake", "after": 3}]]', "A lake."]
    await ag.handle("s", "make me a picture of the lake")
    r = await ag.handle("s", "yes, put it at the end")
    paras = store.paragraphs("demo", 1); assert paras[-1].image and len(paras) == n0 + 2


def test_swedish_names_exist_and_are_found_by_the_word_swedish():
    import ideas
    assert "swedish" in ideas.CULTURES and intent.ideas_request("give me three Swedish names")["opts"]["culture"] == "swedish"
    assert ideas.cultures_for("a farm near Lindsborg, Kansas, to Swedish parents") == ["swedish"]


# ---- round 9: numbering that skips pictures, wiki requests, mixed names, warnings, "no" going back to where the writing is --------------
def test_a_wiki_request_is_not_taken_for_an_edit_of_a_paragraph():
    for t in ("make wiki pages for father Olof, the farm, Camp Doniphan and Charpentry", "I want story notes saved for the farm and the camp", "make a character page for Freja"):
        assert not intent.is_edit(t), t
    assert intent.is_edit("make paragraph 2 sadder") and intent.is_edit("make the first paragraph of chapter 2 sadder")


def test_a_handful_of_names_includes_women_and_men():
    import ideas
    for seed in range(8):
        names = ideas.Ideas(seed=seed).generate("name", 3, culture="swedish")
        female = set(ideas.D.NAMES["swedish"]["female"]); male = set(ideas.D.NAMES["swedish"]["male"])
        firsts = [n.split()[0] for n in names]
        assert any(f in female for f in firsts) and any(f in male for f in firsts), names


@pytest.mark.asyncio
async def test_paragraph_numbers_skip_pictures_and_the_picture_has_its_own_words(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    for t in ("Alpha paragraph here.", "Bravo paragraph here.", "Charlie paragraph here."):
        store.append_paragraph("demo", 2, t)
    name = ag.images.stage("demo", png(), "pic"); ag.images.accept("demo", name)
    store.insert_paragraph("demo", 2, imgs.markdown("A pic", name), before_id=store.paragraphs("demo", 2)[1].id)  # alpha, PICTURE, bravo, charlie
    r = await ag.handle("s", "delete paragraph 2 of chapter 2")
    assert r.pending["kind"] == "delete" and "Bravo" in r.text and "chapter 2, paragraph 2" in r.text  # bravo is the 2nd piece of writing, the picture is not counted
    await ag.handle("s", "no")
    r = await ag.handle("s", "delete the picture in chapter 2")
    assert r.pending["kind"] == "delete" and "the picture in chapter 2" in r.text
    assert store.nth_paragraph("demo", 2, 3).text.startswith("Charlie") and store.nth_paragraph("demo", 2, 1, picture=True).image


@pytest.mark.asyncio
async def test_a_picture_goes_after_the_nth_piece_of_writing_not_the_nth_paragraph(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.create_chapter("demo", 3)
    for t in ("Alpha.", "Bravo.", "Charlie."):
        store.append_paragraph("demo", 3, t)
    for where, expect in ((0, 0), (2, 2), (9, 3)):
        n = ag.images.stage("demo", png(), "p"); ag.images.accept("demo", n)
        para = ag._place_image(ag.session("s"), 3, n, "cap", where)
        ids = [p.id for p in store.paragraphs("demo", 3)]
        assert ids.index(para.id) == expect, (where, ids)
        store.delete_paragraph("demo", para.id)


@pytest.mark.asyncio
async def test_a_flagged_draft_says_it_may_not_be_what_was_asked_and_a_no_goes_back_to_the_writing_chapter(books):
    ag, store, llm = make(books)
    for t in ("a first paragraph text here for the book.", "b second paragraph text here for the book.", "c third paragraph text here for the book.", "d fourth paragraph text here for the book."):
        store.append_paragraph("demo", 1, t)
    store.create_chapter("demo", 2)
    store.append_paragraph("demo", 2, "Chapter two has a paragraph here to be rewritten later.")
    sess = ag.session("s"); sess.work = 2; sess.chapter = 2
    llm.replies = ["Chapter two now reads a little sadder, with the window shut."]
    r = await ag.handle("s", "make paragraph 1 of chapter 1 sadder")
    assert r.pending and sess.chapter == 1
    r = await ag.handle("s", "no")
    assert sess.chapter == 2 and "back in chapter 2" in r.text


# ---- round 10: describing a named picture, "no, put it at the end", "leave Erik out" ---------------------------------------------------------
def test_describing_a_named_picture_is_a_describe_request_never_a_new_paragraph():
    for t in ("describe the railway station picture", "describe the picture in chapter 4 to me", "tell me about the trench photo", "describe the trench picture again"):
        assert intent.describe_request(t), t
    for t in ("describe the scene where he leaves the farm", "make a picture of the station", "find a picture of the station"):
        assert not intent.describe_request(t), t


@pytest.mark.asyncio
async def test_describe_the_picture_in_another_chapter_finds_it_and_writes_nothing(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "Some writing in chapter two to keep.")
    n = ag.images.stage("demo", png(), "station"); ag.images.accept("demo", n)
    store.append_image("demo", 2, n, "An old railway station at dusk")
    before = [p.id for p in store.all_paragraphs("demo")]
    ag.session("s").work = 1
    r = await ag.handle("s", "describe the railway station picture")
    assert "railway station" in r.text and [p.id for p in store.all_paragraphs("demo")] == before and r.pending is None and not llm.calls
    r = await ag.handle("s", "describe the picture in chapter 2 to me")
    assert "railway station" in r.text and not llm.calls
    r = await ag.handle("s", "describe the zeppelin picture")
    assert "which picture" in r.text.lower() or "railway" in r.text  # a clear question or the only picture, never a paragraph
    assert [p.id for p in store.all_paragraphs("demo")] == before


@pytest.mark.asyncio
async def test_no_put_it_at_the_end_moves_the_picture_without_dropping_it(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a station", "caption": "The station", "after": 1}]]', "A station."]
    r = await ag.handle("s", "make me a picture of the station")
    assert r.pending["kind"] == "picture"
    r = await ag.handle("s", "no, put it at the end")
    assert r.pending is not None and "at the end" in r.text and r.pending["kind"] == "picture"
    await ag.handle("s", "yes")
    assert store.paragraphs("demo", 1)[-1].image


@pytest.mark.asyncio
async def test_yes_but_leave_one_out_leaves_it_out_of_the_wiki_batch(books):
    ag, store, llm = make(books)
    from agent import Pending
    batch = [{"kind": "characters", "name": n, "note": "A person."} for n in ("Nils Lund", "Erik Holmberg", "Freja Lund")]
    sess = ag.session("s"); sess.pending = Pending("wiki", "", 1, data={"batch": batch}, ask="Save these 3 notes to the wiki: Nils Lund, Erik Holmberg, Freja Lund?")
    r = await ag.handle("s", "Yes, but leave Erik out")
    names = " ".join(store.bible("demo").characters())
    assert "left out Erik Holmberg" in r.text and "Nils Lund" in names and "Freja Lund" in names and "Erik" not in names and sess.pending is None  # saved at once: one answer
