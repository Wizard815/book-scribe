import pytest
from fastapi.testclient import TestClient

from agent import Agent
from book_store import BookStore, parse_chapter
from kg import StoryKG
from main import create_app
from memory.store import FileMemory
from quality import check
from settings import Settings
from tests.conftest import FakeLLM


def make(books, replies):
    s = Settings(project_dir=books, default_book="demo")
    store, llm = BookStore(books), FakeLLM(replies)
    mem = FileMemory(books / ".state" / "m.json")
    return Agent(s, store, llm, mem, StoryKG(store)), store, llm, mem


def test_parse_paragraph_ids(books):
    store = BookStore(books)
    ids = [p.id for p in store.paragraphs("demo", 1)]
    assert ids == ["ch1_s1_p1", "ch1_s1_p2", "ch1_s2_p1"]


def test_replace_only_touches_target(books):
    store = BookStore(books)
    before = store.read_chapter_raw("demo", 1)
    store.replace_paragraph("demo", "ch1_s1_p2", "The express was late again.")
    after = store.read_chapter_raw("demo", 1)
    assert "The express was late again." in after
    assert after.replace("The express was late again.", "X") .count("X") == 1
    assert [p.text for p in store.paragraphs("demo", 1)][0] == "Mira stepped onto the platform, her breath fogging in the cold air."
    assert before.split("<!-- paragraph_id: ch1_s1_p2 -->")[0] == after.split("<!-- paragraph_id: ch1_s1_p2 -->")[0]
    assert "ch1_s2_p1" in after


def test_append_generates_next_id(books):
    store = BookStore(books)
    p = store.append_paragraph("demo", 1, "New line.", scene=2)
    assert p.id == "ch1_s2_p2"
    assert store.find("demo", "ch1_s2_p2").text == "New line."


def test_book_name_traversal_rejected(books):
    with pytest.raises(Exception):
        BookStore(books).book_dir("../etc")


def test_kg_resolves_cafe_scene(books):
    store = BookStore(books)
    kg = StoryKG(store)
    kg.refresh("demo")
    assert kg.find_paragraphs("the cafe where Jonas orders tea", 1)[0].id == "ch1_s2_p1"


def test_quality_flags_cliches():
    assert check("It was a testament to her will, a symphony of rain.")
    assert not check("She shut the door and listened to the rain.")


@pytest.mark.asyncio
async def test_edit_requires_confirmation_then_saves(books):
    agent, store, llm, _ = make(books, ["At the café, Jonas fidgeted with his cup."])
    r = await agent.handle("s", "In chapter 1 scene 2 paragraph 1 make Jonas more nervous")
    assert r.pending and r.pending["paragraph_id"] == "ch1_s2_p1"
    assert "tea he did not drink" in store.find("demo", "ch1_s2_p1").text  # not yet written
    r = await agent.handle("s", "yes save it")
    assert r.changed == ["ch1_s2_p1"]
    assert store.find("demo", "ch1_s2_p1").text == "At the café, Jonas fidgeted with his cup."


@pytest.mark.asyncio
async def test_rejection_is_remembered_and_nothing_written(books):
    agent, store, _, mem = make(books, ["Something else entirely."])
    await agent.handle("s", "rewrite the first paragraph of chapter 1")
    r = await agent.handle("s", "no")
    assert r.pending is None
    assert "Mira stepped" in store.find("demo", "ch1_s1_p1").text
    assert any("rejected" in m for m in mem.search("rejected proposed edit"))


@pytest.mark.asyncio
async def test_quality_gate_triggers_rewrite(books):
    agent, _, llm, _ = make(books, ["A testament to the rain.", "Rain hit the roof."])
    r = await agent.handle("s", "write the next scene")
    assert r.pending["text"] == "Rain hit the roof."
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_preference_learning_and_recall(books):
    agent, _, llm, mem = make(books, ["Sure."])
    await agent.handle("s", "I prefer short sentences and no adverbs")
    await agent.handle("s", "tell me about sentences")
    assert any("short sentences" in m["content"] for m in llm.calls[-1] if m["role"] == "system")


@pytest.mark.asyncio
async def test_therapist_mode_never_drafts(books):
    agent, store, llm, _ = make(books, ["What does Mira want?"])
    await agent.handle("s", "switch to therapist mode")
    r = await agent.handle("s", "write the next scene")
    assert r.pending is None and r.mode == "therapist"


def test_api_chat_and_voice(books):
    class STT:
        http = FakeLLM().http
        async def transcribe(self, *a, **k): return "go to chapter 1"
    app = create_app(Settings(project_dir=books, admin_token="tok"), llm=FakeLLM(), stt=STT(), memory=FileMemory(books / "m.json"))
    with TestClient(app) as c:
        assert c.get("/api/health").json()["ok"]
        r = c.post("/api/voice", files={"audio": ("a.webm", b"x", "audio/webm")}).json()
        assert r["user_text"] == "go to chapter 1" and r["chapter"] == 1
        assert len(c.get("/api/books/demo/chapters/1").json()["paragraphs"]) == 3
        assert c.post("/api/tts", json={"text": "hi"}).status_code == 501


def admin_client(books, token="tok"):
    s = Settings(project_dir=books, admin_token=token, llama_swap_url="http://llm:8080")
    app = create_app(s, llm=FakeLLM(), stt=FakeLLM(), tts=FakeLLM(), memory=FileMemory(books / "m.json"))
    return TestClient(app)


def test_admin_requires_token(books):
    with admin_client(books) as c:
        assert c.get("/api/admin/providers").status_code == 401
        assert c.get("/api/admin/providers", headers={"x-admin-token": "bad"}).status_code == 401
        assert c.get("/api/admin/providers", headers={"x-admin-token": "tok"}).status_code == 200
    with admin_client(books, token="") as c:
        assert c.get("/api/admin/providers", headers={"x-admin-token": ""}).status_code == 503


def test_admin_provider_lifecycle_hides_keys(books):
    H = {"x-admin-token": "tok"}
    with admin_client(books) as c:
        snap = c.get("/api/admin/providers", headers=H).json()
        assert snap["roles"]["chat"]["active"] == "local-chat"
        assert snap["roles"]["chat"]["profiles"][0]["base_url"] == "http://llm:8080"
        p = c.put("/api/admin/providers/chat", headers=H, json={"label": "GPT", "kind": "cloud", "base_url": "https://api.openai.com/v1/", "api_key": "sk-secret-1234", "model": "gpt-x"}).json()
        assert p["base_url"] == "https://api.openai.com" and p["has_key"] and "secret" not in str(p)
        assert "sk-secret" not in c.get("/api/admin/providers", headers=H).text
        c.post(f"/api/admin/providers/chat/{p['id']}/activate", headers=H)
        c.put("/api/admin/providers/chat", headers=H, json={"id": p["id"], "label": "GPT", "kind": "cloud", "base_url": "https://api.openai.com", "model": "gpt-y"})  # blank key keeps old
        f = books / ".state" / "providers.json"
        assert "sk-secret-1234" in f.read_text() and (f.stat().st_mode & 0o077) == 0
        assert c.get("/api/admin/providers", headers=H).json()["roles"]["chat"]["active"] == p["id"]
        assert c.put("/api/admin/providers/chat", headers=H, json={"label": "x", "base_url": "ftp://x"}).status_code == 400
        assert c.put("/api/admin/providers/nope", headers=H, json={"label": "x"}).status_code == 400
        assert c.delete("/api/admin/providers/chat/local-chat", headers=H).status_code == 200
        assert c.delete(f"/api/admin/providers/chat/{p['id']}", headers=H).status_code == 400
