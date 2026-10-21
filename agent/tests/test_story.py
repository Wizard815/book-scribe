import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent import Agent
from bible import Bible
from book_store import BookStore
from drift import parse_issues
from kg import StoryKG
from memory.store import FileMemory
from providers import Registry
from settings import Settings
from story_graph import StoryGraph
from tests.conftest import FakeLLM


def test_bible_context_uses_only_named_entries(books):
    b = Bible(books / "demo")
    ctx = b.context_for("make Jonas more nervous")
    assert "CHARACTER Jonas" in ctx and "CHARACTER Mira" not in ctx
    assert "PREMISE" in ctx and "The unmapped station" in ctx  # open thread listed
    assert "WORLD The station" in b.context_for("describe the station in the rain")


def test_bible_upsert_and_arc_log_round_trip(books):
    b = Bible(books / "demo")
    b.upsert("characters", "Mira", {"location": "the unmapped station"})
    b.append_arc("Mira", "Ch1: steps off the train")
    m = b.characters()["Mira"]
    assert m.state["location"] == "the unmapped station" and m.state["role"] == "protagonist"  # merged, not replaced
    assert "Arc log" in m.body and "steps off the train" in m.body


def test_bible_rejects_path_tricks(books):
    with pytest.raises(Exception):
        Bible(books / "demo").append_arc("../../x", "no")


def test_legacy_single_file_layout_is_migrated(tmp_path):
    book = tmp_path / "old"
    (book / "chapters").mkdir(parents=True)
    (book / "bible.md").write_text("# Story Bible\n\nA ghost story.\n")
    (book / "characters.md").write_text("# Characters\n\n## Ann\nBrave.\n")
    BookStore(tmp_path).ensure_book("old")
    assert not (book / "bible.md").exists() and not (book / "characters.md").exists()
    assert "Brave." in Bible(book).characters()["Ann"].body
    assert "ghost story" in Bible(book).read("premise")


def test_registry_adds_graph_role_to_old_provider_file(tmp_path):
    s = Settings(project_dir=tmp_path, llama_swap_url="http://llm:1")
    s.state_dir.mkdir(parents=True)
    (s.state_dir / "providers.json").write_text(json.dumps({"active": {"chat": "c"}, "profiles": {"chat": [{"id": "c", "label": "C", "base_url": "http://x"}]}}))
    r = Registry(s)
    assert r.get("chat").base_url == "http://x"
    assert r.get("graph").base_url == "http://llm:1"


def test_graph_context_shows_confidence_tags(books):
    store = BookStore(books)
    out = books / "demo" / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps({"nodes": [{"id": "m", "label": "Mira"}, {"id": "j", "label": "Jonas"}],
        "links": [{"source": "j", "target": "m", "relation": "works_with", "confidence": "EXTRACTED"}]}))
    g = StoryGraph(store, Registry(Settings(project_dir=books)))
    assert g.context_for("demo", "what does Mira think") == "Jonas —works_with→ Mira [EXTRACTED]"
    assert g.context_for("demo", "the weather") == ""


def test_parse_issues_is_forgiving():
    assert parse_issues('Sure: ["Mira is blind"]') == ["Mira is blind"]
    assert parse_issues("[]") == [] and parse_issues("no json") == [] and parse_issues('["a", 3, ""]') == ["a"]


@pytest.mark.asyncio
async def test_drift_is_remembered_and_delivered_next_turn(books):
    s = Settings(project_dir=books, default_book="demo")
    store = BookStore(books)
    llm = FakeLLM(["Jonas kept his hands in his pockets.", '["Jonas is said to be sighted but the bible says blind"]', "ok"])
    mem = FileMemory(books / ".state" / "m.json")
    agent = Agent(s, store, llm, mem, StoryKG(store), graph=None, check_drift=True)
    await agent.handle("s", "rewrite chapter 1 scene 2 paragraph 1")
    r = await agent.handle("s", "yes save it")
    assert r.changed == ["ch1_s2_p1"]
    await asyncio.gather(*agent._bg)
    nxt = await agent.handle("s", "go to chapter 1")
    assert any("may contradict" in n and "blind" in n for n in nxt.notes)
    assert any("Drift from canon" in m for m in mem.search("Drift from canon contradict"))
    assert (await agent.handle("s", "go to chapter 1")).notes == []  # delivered once


@pytest.mark.asyncio
async def test_prompt_includes_bible_and_graph(books):
    s = Settings(project_dir=books, default_book="demo")
    store = BookStore(books)
    out = books / "demo" / "graphify-out"; out.mkdir()
    (out / "graph.json").write_text(json.dumps({"nodes": [{"id": "j", "label": "Jonas"}, {"id": "m", "label": "Mira"}], "links": [{"source": "j", "target": "m", "relation": "lies_to", "confidence": "INFERRED"}]}))
    llm = FakeLLM(["Sure."])
    agent = Agent(s, store, llm, FileMemory(books / "m.json"), StoryKG(store), graph=StoryGraph(store, Registry(s)))
    await agent.handle("s", "tell me about Jonas")
    system = llm.calls[-1][0]["content"]
    assert "CHARACTER Jonas" in system and "lies_to→ Mira [INFERRED]" in system


# --- real graphify, stubbed model -------------------------------------------------------------
EXT = {"nodes": [{"id": "mira", "label": "Mira", "file_type": "document", "source_file": "chapters/chapter_01.md", "source_location": "L8"},
                 {"id": "jonas", "label": "Jonas", "file_type": "document", "source_file": "chapters/chapter_01.md", "source_location": "L11"}],
       "edges": [{"source": "jonas", "target": "mira", "relation": "works_with", "confidence": "EXTRACTED", "confidence_score": 0.9, "source_file": "chapters/chapter_01.md"}],
       "hyperedges": []}


class Stub(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        Stub.seen["auth"] = self.headers.get("Authorization")
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                          "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(EXT)}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


@pytest.mark.asyncio
async def test_real_graphify_builds_graph_through_active_graph_provider(books):
    pytest.importorskip("graphify")
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = Settings(project_dir=books, llama_swap_url=f"http://127.0.0.1:{srv.server_port}")
    reg = Registry(s)
    reg.upsert("graph", {"id": "local-graph", "label": "Local graph", "kind": "local", "base_url": s.llama_swap_url, "model": "m", "api_key": "k-123"})
    g = StoryGraph(BookStore(books), reg)
    st = await g.rebuild("demo")
    srv.shutdown()
    assert st["state"] == "ok", st
    assert Stub.seen["auth"] == "Bearer k-123"  # the admin-menu provider (and its key) was used
    assert g.info("demo")["nodes"] >= 2
    assert "Jonas —works_with→ Mira" in g.context_for("demo", "Mira")
