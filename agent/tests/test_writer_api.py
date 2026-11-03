import pytest
from fastapi.testclient import TestClient

from bible import Bible
from main import create_app
from memory.store import FileMemory
from settings import Settings
from tests.conftest import FakeLLM


@pytest.fixture
def client(books):
    s = Settings(project_dir=books, default_book="demo", admin_token="a-long-enough-admin-secret-1")
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["ok"]), stt=FakeLLM(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False, require_auth=False)
    with TestClient(app) as c:
        yield c


def test_bible_lists_everything_in_the_book(client):
    b = client.get("/api/books/demo/bible").json()
    assert {c["name"] for c in b["characters"]} == {"Mira", "Jonas"} and "night-train conductor" in b["premise"].lower()
    mira = next(c for c in b["characters"] if c["name"] == "Mira")
    assert mira["slug"] == "mira" and mira["state"]["role"] == "protagonist" and "Dry humor" in mira["body"]
    assert [t["name"] for t in b["threads"]] == ["The unmapped station"] and [w["name"] for w in b["world"]] == ["The station"]


def test_add_edit_rename_and_delete_a_character(client, books):
    r = client.put("/api/books/demo/bible/characters", json={"name": "Old Tom", "state": {"gender": "male", "aliases": ["Tom", "the porter"], "temperament": "gruff"}, "body": "Station porter."})
    assert r.json() == {"slug": "old-tom", "name": "Old Tom"}
    tom = next(c for c in client.get("/api/books/demo/bible").json()["characters"] if c["slug"] == "old-tom")
    assert tom["state"] == {"gender": "male", "aliases": ["Tom", "the porter"], "temperament": "gruff"} and tom["body"] == "Station porter."
    # the editor saves exactly what it shows: removing a field removes it (no silent merge)
    client.put("/api/books/demo/bible/characters", json={"slug": "old-tom", "name": "Old Tom", "state": {"gender": "male"}, "body": "Station porter."})
    assert next(c for c in client.get("/api/books/demo/bible").json()["characters"] if c["slug"] == "old-tom")["state"] == {"gender": "male"}
    # renaming moves the file instead of leaving a duplicate
    client.put("/api/books/demo/bible/characters", json={"slug": "old-tom", "name": "Tom Hale", "state": {}, "body": ""})
    slugs = {c["slug"] for c in client.get("/api/books/demo/bible").json()["characters"]}
    assert "tom-hale" in slugs and "old-tom" not in slugs
    assert client.delete("/api/books/demo/bible/characters/tom-hale").status_code == 200
    assert client.delete("/api/books/demo/bible/characters/tom-hale").status_code == 404


def test_the_agent_and_the_voices_use_what_the_editor_saved(client, books):
    client.put("/api/books/demo/bible/characters", json={"name": "Lizzy", "state": {"gender": "female", "aliases": ["Eliza"]}, "body": "Quick-witted."})
    from book_store import BookStore
    from voices import guess_traits
    e = BookStore(books).bible("demo").characters()["Lizzy"]
    assert guess_traits(e.name, e.state, e.body)["gender"] == "female" and e.state["aliases"] == ["Eliza"]


def test_state_is_cleaned_so_a_pasted_blob_cannot_break_the_file(client, books):
    r = client.put("/api/books/demo/bible/characters", json={"name": "Odd", "state": {"Bad Key!": "x", "name": "hijack", "ok key": "y" * 1000, "list": ["a"] * 100, "nested": {"a": 1}, "": "z"}, "body": ""})
    assert r.status_code == 200
    st = next(c for c in client.get("/api/books/demo/bible").json()["characters"] if c["slug"] == "odd")["state"]
    assert set(st) == {"ok_key", "list", "nested"} and len(st["ok_key"]) == 300 and len(st["list"]) == 30
    assert "name" not in st


def test_bible_input_limits_and_path_safety(client):
    assert client.put("/api/books/demo/bible/characters", json={"name": "", "state": {}}).status_code == 422
    assert client.put("/api/books/demo/bible/characters", json={"name": "x" * 200, "state": {}}).status_code == 422
    assert client.put("/api/books/demo/bible/characters", json={"name": "ok", "state": {}, "body": "x" * 30000}).status_code == 422
    assert client.put("/api/books/demo/bible/secrets", json={"name": "ok"}).status_code == 400  # unknown section
    assert client.get("/api/books/..%2Fetc/bible").status_code in (400, 404)
    assert client.put("/api/books/demo/bible/characters", json={"slug": "../../x", "name": "Z"}).status_code == 400  # a path in the slug is refused


def test_premise_and_timeline_documents(client):
    assert client.put("/api/books/demo/bible/doc/premise", json={"text": "# Premise\n\nA new premise."}).status_code == 200
    assert "A new premise." in client.get("/api/books/demo/bible").json()["premise"]
    assert client.put("/api/books/demo/bible/doc/passwords", json={"text": "x"}).status_code == 400


def test_preferences_are_saved_and_reach_the_prompt(client, books):
    assert client.get("/api/settings").json() == {"writer_age": "", "display_name": "", "default_book": "demo"}
    client.put("/api/settings", json={"writer_age": "71", "display_name": "Ann"})
    assert client.get("/api/settings").json()["writer_age"] == "71"
    assert client.put("/api/settings", json={"writer_age": "x" * 50}).status_code == 422
    client.post("/api/chat", json={"text": "tell me about sentences", "session_id": "s"})
    # the agent was created inside the app; check the stored file the agent reads
    assert '"writer_age": "71"' in (books / ".state" / "prefs.json").read_text()


@pytest.mark.asyncio
async def test_the_writer_age_preference_is_in_the_system_prompt(books):
    from agent import Agent
    from book_store import BookStore
    from kg import StoryKG
    from prefs import Prefs
    s = Settings(project_dir=books, default_book="demo")
    prefs = Prefs(s.state_dir); prefs.update({"writer_age": "71"})
    llm = FakeLLM(["hi"])
    a = Agent(s, BookStore(books), llm, FileMemory(books / "m.json"), StoryKG(BookStore(books)), prefs=prefs)
    await a.handle("s", "tell me about sentences")
    assert "The writer is 71" in llm.calls[-1][0]["content"]


def test_create_a_book_and_import_a_draft(client, books):
    assert client.post("/api/books", json={"name": "My Novel"}).json() == {"name": "My-Novel"}
    assert client.post("/api/books", json={"name": "My Novel"}).status_code == 409
    assert client.post("/api/books", json={"name": "../bad"}).status_code == 400
    assert "My-Novel" in client.get("/api/books").json()["books"]
    draft = "Chapter 1\n\nIt was dark\nand stormy.\n\n“Hello,” she said.\n\nChapter 2\n\nMorning came.\n"
    assert client.post("/api/books/My-Novel/import", json={"text": draft}).json() == {"chapters": 2, "paragraphs": 3}
    assert [p["text"] for p in client.get("/api/books/My-Novel/chapters/1").json()["paragraphs"]] == ["It was dark and stormy.", "“Hello,” she said."]
    assert client.post("/api/books/My-Novel/import", json={"text": draft}).status_code == 409  # never overwrite a book that has chapters
    assert client.post("/api/books", json={"name": "Empty"}).status_code == 200
    assert client.post("/api/books/Empty/import", json={"text": "no headings here"}).status_code == 400


def test_forgetting_a_voice(client, books):
    from book_store import BookStore
    import voices as vx
    from providers import Registry
    vb = vx.VoiceBook(BookStore(books), Registry(Settings(project_dir=books)))
    vb.profile("demo", "Mira")
    assert client.delete("/api/books/demo/voices/Mira").json() == {"forgotten": True}
    assert client.delete("/api/books/demo/voices/Mira").json() == {"forgotten": False}
