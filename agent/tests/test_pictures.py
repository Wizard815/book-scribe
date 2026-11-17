"""Pictures in the book: checking files, picture paragraphs, making / finding / reusing pictures, export, and the draft clean-up."""
import io
import json
import zipfile

import httpx
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
    r = await ag.handle("s", "read paragraph 4"); assert r.text == "Picture: The hand of the lake"
    r = await ag.handle("s", "rewrite paragraph 4 to be sadder"); assert "is a picture" in r.text and r.pending is None
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
