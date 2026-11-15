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

CONCEPTS = [
    {"porter", "luggage", "bags", "carries", "suitcases", "trunks", "heavy"},
    {"lighthouse", "lamp", "keeper", "beacon", "tower", "ships"},
    {"train", "platform", "station", "noon", "ticket"},
    {"mystery", "secret", "hidden", "locked", "key"},
]


def vec(text):
    w = {x.strip(".,;!?\"':").lower() for x in text.split()}
    return [float(len(w & c)) + 0.01 for c in CONCEPTS]


class Server:
    def __init__(self): self.calls, self.fail = [], False
    def handler(self, req):
        if self.fail: raise httpx.ConnectError("down")
        ins = json.loads(req.content)["input"]
        self.calls.append(list(ins))
        return httpx.Response(200, json={"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(ins)]})


@pytest.fixture
def env(tmp_path):
    s = Settings(project_dir=tmp_path, default_book="b", embed_url="http://e", embed_model="m", chat_url="http://c")
    reg, store = Registry(s), BookStore(tmp_path)
    store.create_chapter("b", 1)
    store.append_paragraph("b", 1, "The train pulled into the station at noon.")
    store.append_paragraph("b", 1, "A man waited on the platform with a ticket.")
    bible = store.bible("b")
    bible.upsert("characters", "Old Tom", {"role": "porter"}, "Carries heavy luggage and suitcases at the station. Knows everyone's secrets.")
    bible.upsert("characters", "Mira", {"role": "protagonist"}, "A conductor who keeps a notebook.")
    bible.upsert("threads", "The locked room", {"status": "open"}, "A hidden room with a locked door and a missing key.")
    bible.upsert("world", "The lighthouse", {}, "A tower with a beacon lamp that warns ships.")
    bible.write_doc("premise", "# Premise\n\nA mystery on a railway line.")
    srv = Server()
    http = httpx.AsyncClient(transport=httpx.MockTransport(srv.handler))
    idx = ParagraphIndex(store, reg, Router(reg), http, s.state_dir)
    return s, store, srv, http, idx


@pytest.mark.asyncio
async def test_the_bible_is_indexed_alongside_the_paragraphs_and_counted_separately(env):
    s, store, srv, _, idx = env
    st = await idx.reindex("b")
    assert st["paragraphs"] == 2 and st["indexed"] == 2 and st["bible_entries"] == 5 and st["bible_indexed"] == 5 and st["ready"]
    assert sum(len(c) for c in srv.calls) == 7
    keys = set(idx._load("b")["items"])
    assert {"bible:premise", "bible:characters:old-tom", "bible:threads:the-locked-room", "bible:world:the-lighthouse"} <= keys


@pytest.mark.asyncio
async def test_paragraph_search_and_bible_search_never_mix(env):
    _, _, _, _, idx = env
    await idx.reindex("b")
    assert all(not k.startswith("bible:") for k, _ in await idx.search("b", "heavy luggage at the station", k=10))
    hits = await idx.search("b", "heavy luggage at the station", k=10, bible=True)
    assert hits and all(k.startswith("bible:") for k, _ in hits) and hits[0][0] == "bible:characters:old-tom"


@pytest.mark.asyncio
async def test_a_change_to_one_bible_entry_re_embeds_only_that_entry(env):
    s, store, srv, _, idx = env
    await idx.reindex("b")
    srv.calls.clear()
    store.bible("b").save_entry("characters", "Old Tom", {"role": "porter"}, "Now he carries bags AND secrets.", "old-tom")
    await idx.reindex("b")
    assert [len(c) for c in srv.calls] == [1] and "bags AND secrets" in srv.calls[0][0]


@pytest.mark.asyncio
async def test_canon_is_found_by_meaning_when_the_name_is_not_said(env):
    s, store, srv, _, idx = env
    await idx.reindex("b")
    agent = Agent(s, store, FakeLLM([]), FileMemory(s.project_dir / "m.json"), StoryKG(store), index=idx)
    sess = agent.session("s", "b")
    got = await agent._canon_by_meaning(sess, "the old man who carries the bags and trunks")
    assert "ALSO RELEVANT" in got and "Old Tom" in got and "porter" in got.lower()
    assert "Mira" not in got
    # named explicitly: already in the prompt by name, so not repeated as "also relevant"
    assert "Old Tom" not in await agent._canon_by_meaning(sess, "Old Tom carries the heavy bags")


@pytest.mark.asyncio
async def test_the_model_sees_that_canon_in_its_prompt(env):
    s, store, srv, _, idx = env
    await idx.reindex("b")
    llm = FakeLLM(["Ok."])
    agent = Agent(s, store, llm, FileMemory(s.project_dir / "m.json"), StoryKG(store), index=idx)
    await agent.handle("s", "what is the heavy luggage man up to with the trunks?", "b")
    system = llm.calls[0][0]["content"]
    assert "ALSO RELEVANT (matched by meaning)" in system and "Old Tom" in system and "secrets" in system


@pytest.mark.asyncio
async def test_without_an_index_or_with_the_embedder_down_nothing_breaks(env):
    s, store, srv, _, idx = env
    agent = Agent(s, store, FakeLLM([]), FileMemory(s.project_dir / "m.json"), StoryKG(store), index=idx)
    sess = agent.session("s", "b")
    assert await agent._canon_by_meaning(sess, "the luggage man") == ""  # no index yet
    await idx.reindex("b")
    srv.fail = True
    assert await agent._canon_by_meaning(sess, "the luggage man") == ""  # provider down
    assert await Agent(s, store, FakeLLM([]), FileMemory(s.project_dir / "m2.json"), StoryKG(store))._canon_by_meaning(sess, "x") == ""  # no index at all


# --- API ------------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def client(env):
    s, store, srv, http, idx = env
    s.admin_token = "a-long-enough-admin-secret-1"
    s.index_debounce = 0.05
    tts = FakeLLM(); tts.enabled = False
    app = create_app(s, llm=FakeLLM(["ok"]), stt=FakeLLM(), tts=tts, memory=FileMemory(s.project_dir / "mm.json"), check_drift=False, require_auth=False, embed_http=http)
    with TestClient(app) as c:
        yield c, srv


def test_saving_a_bible_entry_through_the_api_schedules_an_index_update(client):
    c, srv = client
    c.put("/api/books/b/bible/characters", json={"name": "Nell", "state": {}, "body": "A lamp keeper at the lighthouse."})
    for _ in range(80):
        time.sleep(0.1)
        if any("lighthouse" in t for call in srv.calls for t in call):
            break
    assert any("Nell" in t for call in srv.calls for t in call)


def test_ideas_endpoints(client):
    c, _ = client
    kinds = c.get("/api/ideas/kinds").json()["kinds"]
    assert {k["kind"] for k in kinds} >= {"name", "place", "twist", "constraint"} and next(k for k in kinds if k["kind"] == "name")["options"]["gender"]
    r = c.post("/api/ideas", json={"kind": "name", "n": 4, "opts": {"culture": "japanese", "gender": "male"}})
    assert r.status_code == 200 and len(r.json()["items"]) == 4
    assert c.post("/api/ideas", json={"kind": "recipe"}).status_code == 400
    assert c.post("/api/ideas", json={"kind": "name", "n": 99}).status_code == 422


def test_settings_accept_a_look_and_validate_everything(client):
    c, _ = client
    out = c.put("/api/settings", json={"look": {"size": 99, "contrast": "dark", "bold": True, "evil": 1}, "surprise": "high", "strictness": "strict"}).json()
    assert out["look"]["size"] == 7 and out["look"]["contrast"] == "dark" and out["look"]["bold"] is True and "evil" not in out["look"]
    assert out["surprise"] == "high" and out["strictness"] == "strict"
    out = c.put("/api/settings", json={"surprise": "chaos", "strictness": "maximum"}).json()
    assert out["surprise"] == "high" and out["strictness"] == "strict"  # unknown values are ignored, the old ones stay
    assert c.get("/api/settings").json()["look"]["size"] == 7


def test_the_look_set_by_voice_is_what_the_settings_page_reads_back(client):
    c, _ = client
    r = c.post("/api/chat", json={"text": "I can't see this", "session_id": "s", "book": "b"}).json()
    assert r["ui"]["size"] == 4 and c.get("/api/settings").json()["look"]["size"] == 4


def test_admin_can_see_which_tools_the_model_used(client):
    c, _ = client
    H = {"x-admin-token": "a-long-enough-admin-secret-1"}
    assert c.get("/api/admin/traces", headers=H).json() == {"traces": []}
    assert c.get("/api/admin/traces").status_code == 401
