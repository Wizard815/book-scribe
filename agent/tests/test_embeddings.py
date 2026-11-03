import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import Agent
from book_store import BookStore
from embeddings import ParagraphIndex
from kg import StoryKG
from main import create_app
from memory.store import FileMemory
from providers import Registry
from router import Router
from settings import Settings
from tests.conftest import FakeLLM

# A tiny "meaning" model: words belong to concepts, a text is the count of each concept. It stands in for a real embedder so we
# can test the *pipeline* (what gets embedded, when, how results are combined) with exact, predictable answers.
CONCEPTS = [
    {"dismissive", "dismisses", "waved", "silly", "ignorant", "scorn"},
    {"father", "daughters", "children", "girls", "family"},
    {"train", "platform", "goodbye", "noon", "station"},
    {"coins", "ferryman", "pay", "money"},
]


def vec(text: str):
    words = set(w.strip(".,;!?\"'").lower() for w in text.split())
    return [float(len(words & c)) + 0.01 for c in CONCEPTS]


class Server:
    def __init__(self):
        self.calls, self.fail, self.model_dims = [], False, None

    def handler(self, req: httpx.Request):
        if self.fail:
            raise httpx.ConnectError("embedder down")
        body = json.loads(req.content)
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        self.calls.append(inputs)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(inputs)]})


@pytest.fixture
def env(tmp_path):
    s = Settings(project_dir=tmp_path, default_book="b", embed_url="http://embed", embed_model="m1", chat_url="http://chat")
    reg = Registry(s)
    store = BookStore(tmp_path)
    store.create_chapter("b", 1)
    A = store.append_paragraph("b", 1, "He waved away every one of his daughters' accomplishments; they were all silly and ignorant like other girls.").id
    B = store.append_paragraph("b", 1, "The train left at noon and nobody on the platform waved goodbye at all.").id
    C = store.append_paragraph("b", 1, "She counted the coins twice before paying the ferryman at the river.").id
    srv = Server()
    http = httpx.AsyncClient(transport=httpx.MockTransport(srv.handler))
    idx = ParagraphIndex(store, reg, Router(reg), http, s.state_dir)
    return s, reg, store, srv, http, idx, (A, B, C)


@pytest.mark.asyncio
async def test_building_embeds_every_paragraph_once_in_batches(env):
    s, reg, store, srv, _, idx, _ = env
    st = await idx.reindex("b")
    assert st["indexed"] == 3 and st["stale"] == 0 and st["ready"] and st["dims"] == 4 and st["state"] == "ok"
    assert sum(len(c) for c in srv.calls) == 3
    await idx.reindex("b")
    assert sum(len(c) for c in srv.calls) == 3  # nothing changed: nothing re-embedded


@pytest.mark.asyncio
async def test_only_changed_paragraphs_are_re_embedded(env):
    s, reg, store, srv, _, idx, (A, B, C) = env
    await idx.reindex("b")
    srv.calls.clear()
    store.replace_paragraph("b", B, "The ferry left at noon.")
    store.append_paragraph("b", 1, "A new final paragraph about the station.")
    st = await idx.reindex("b")
    assert sum(len(c) for c in srv.calls) == 2 and st["indexed"] == 4 and st["stale"] == 0
    assert (idx._load("b")["items"][A]["h"]) is not None


@pytest.mark.asyncio
async def test_deleted_paragraphs_leave_the_index(env):
    s, reg, store, srv, _, idx, (A, B, C) = env
    await idx.reindex("b")
    (store.chapter_path("b", 1)).write_text(store.read_chapter_raw("b", 1).split("<!-- paragraph_id: ch1_s1_p3 -->")[0])
    await idx.reindex("b")
    assert set(idx._load("b")["items"]) == {A, B}


@pytest.mark.asyncio
async def test_search_finds_the_paragraph_by_meaning_not_words(env):
    s, reg, store, srv, _, idx, (A, B, C) = env
    await idx.reindex("b")
    hits = await idx.search("b", "the part where the father is dismissive about his children")
    assert hits[0][0] == A and hits[0][1] > hits[1][1]
    assert (await idx.search("b", "someone paying the ferryman with money"))[0][0] == C
    assert (await idx.search("b", "waiting at the station"))[0][0] == B


@pytest.mark.asyncio
async def test_a_different_embedding_model_makes_the_old_index_unusable_until_rebuilt(env):
    s, reg, store, srv, _, idx, _ = env
    await idx.reindex("b")
    reg.upsert("embed", {"id": "env-embed", "model": "other-model"})  # vectors from another model are not comparable
    st = idx.status("b")
    assert st["ready"] is False and st["same_provider"] is False and st["indexed"] == 0
    assert await idx.search("b", "the father") == []
    srv.calls.clear()
    st = await idx.reindex("b")
    assert st["ready"] and sum(len(c) for c in srv.calls) == 3  # rebuilt from scratch with the new model


@pytest.mark.asyncio
async def test_provider_failure_gives_no_results_not_an_exception(env):
    s, reg, store, srv, _, idx, _ = env
    await idx.reindex("b")
    srv.fail = True
    assert await idx.search("b", "the father is dismissive") == []
    st = await idx.reindex("b", full=True)
    assert st["state"] == "error" and "ConnectError" in st["error"]


@pytest.mark.asyncio
async def test_no_embedding_provider_means_disabled_not_broken(tmp_path):
    s = Settings(project_dir=tmp_path)
    reg = Registry(s)
    store = BookStore(tmp_path); store.create_chapter("b", 1); store.append_paragraph("b", 1, "Hello there.")
    idx = ParagraphIndex(store, reg, Router(reg), httpx.AsyncClient(), s.state_dir)
    st = await idx.reindex("b")
    assert st["state"] == "disabled" and st["configured"] is False and not st["ready"]


# --- how the agent uses it ---------------------------------------------------------------------------------------
def make_agent(env):
    s, reg, store, srv, http, idx, ids = env
    return Agent(s, store, FakeLLM(["Rewritten."]), FileMemory(s.project_dir / "m.json"), StoryKG(store), index=idx), srv, idx


@pytest.mark.asyncio
async def test_when_words_find_nothing_meaning_proposes_the_paragraph_but_the_writer_still_confirms(env):
    agent, srv, idx = make_agent(env)
    A = env[6][0]
    await idx.reindex("b")
    r = await agent.handle("s", "make it softer where the father is dismissive about his children", "b")
    assert r.pending["kind"] == "confirm_target" and r.pending["paragraph_id"] == A  # found by meaning, read back, not rewritten yet
    assert "Do you mean" in r.text


@pytest.mark.asyncio
async def test_agreement_between_words_and_meaning_is_trusted_without_asking(env):
    agent, srv, idx = make_agent(env)
    A = env[6][0]
    await idx.reindex("b")
    agent.kg.refresh("b")  # handle() always does this first
    para, sure = await agent._resolve(agent.session("s", "b"), "the silly girls")
    assert para.id == A and sure is True


@pytest.mark.asyncio
async def test_confident_word_matches_never_call_the_embedder(env):
    agent, srv, idx = make_agent(env)
    await idx.reindex("b")
    srv.calls.clear()
    para, sure = await agent._resolve(agent.session("s", "b"), "rewrite chapter 1 paragraph 2")  # explicit reference
    assert sure and srv.calls == []


@pytest.mark.asyncio
async def test_without_a_ready_index_the_agent_behaves_exactly_as_before(env):
    agent, srv, idx = make_agent(env)
    para, sure = await agent._resolve(agent.session("s", "b"), "the part where the father is dismissive about his children")
    assert srv.calls == [] and sure is False  # no index yet: no embedding call, word matching only


@pytest.mark.asyncio
async def test_saving_a_change_schedules_a_background_update(env):
    s, reg, store, srv, http, idx, (A, B, C) = env
    idx.debounce = 0.01
    agent = Agent(s, store, FakeLLM(["Fresh words about a station."]), FileMemory(s.project_dir / "m.json"), StoryKG(store), index=idx)
    await idx.reindex("b")
    srv.calls.clear()
    await agent.handle("s", "rewrite chapter 1 paragraph 2", "b")
    await agent.handle("s", "yes", "b")
    import asyncio
    for _ in range(50):
        await asyncio.sleep(0.05)
        if srv.calls:
            break
    assert srv.calls and any("station" in t for c in srv.calls for t in c)  # only the changed paragraph was re-embedded


# --- admin API --------------------------------------------------------------------------------------------------------
TOKEN = "a-long-enough-admin-secret-1"


def test_admin_embeddings_endpoints(env):
    s, reg, store, srv, http, idx, _ = env
    s.admin_token = TOKEN
    app = create_app(s, llm=FakeLLM(), stt=FakeLLM(), tts=type("T", (), {"enabled": False, "http": httpx.AsyncClient()})(), memory=FileMemory(s.project_dir / "mm.json"),
                     check_drift=False, require_auth=False, embed_http=http)
    H = {"x-admin-token": TOKEN}
    with TestClient(app) as c:
        st = c.get("/api/admin/embeddings", headers=H).json()
        assert st["index"]["configured"] and st["index"]["indexed"] == 0 and st["memory"]["backend"] in ("file", "mem0", "injected") and "b" in st["books"]
        assert c.post("/api/admin/embeddings/test", headers=H).json() == {"ok": True, "dims": 4}
        assert c.post("/api/admin/books/b/embeddings/rebuild", headers=H, json={"full": True}).status_code == 200
        for _ in range(60):
            cur = c.get("/api/admin/embeddings?book=b", headers=H).json()["index"]
            if cur["state"] == "ok":
                break
            time.sleep(0.05)
        assert cur["state"] == "ok" and cur["indexed"] == 3 and cur["ready"]
        assert c.get("/api/admin/embeddings", headers={}).status_code == 401
        srv.fail = True
        assert c.post("/api/admin/embeddings/test", headers=H).json()["ok"] is False
