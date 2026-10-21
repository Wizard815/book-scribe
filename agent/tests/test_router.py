import httpx
import pytest

from clients import LLM, STT, TTS
from providers import Registry
from router import AllTiersFailed, Health, NotConfigured, Router
from settings import Settings


def setup(tmp_path, handler, tiers=(1, 2, 3), role="chat"):
    """Registry with one profile per tier (hosts t1/t2/t3) and a mock transport recording which hosts were hit."""
    s = Settings(project_dir=tmp_path)
    reg = Registry(s)
    ids = [reg.upsert(role, {"label": f"P{t}", "base_url": f"http://t{t}", "tier": t, "model": f"m{t}"}).id for t in tiers]
    hits = []

    def wrapped(req: httpx.Request):
        hits.append(req.url.host)
        return handler(req)

    return reg, Router(reg, now=lambda: clock[0]), httpx.AsyncClient(transport=httpx.MockTransport(wrapped)), hits, ids


clock = [1000.0]


def ok(req):
    return httpx.Response(200, json={"choices": [{"message": {"content": f"from {req.url.host}"}}], "text": f"from {req.url.host}"})


@pytest.mark.asyncio
async def test_t1_answers_and_lower_tiers_are_never_called(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, ok)
    assert await LLM(router, http).chat([]) == "from t1"
    assert hits == ["t1"]
    assert router.served("chat")["chat"] == {"tier": 1, "label": "T1 · Local ROCm", "profile": reg.chain("chat")[0][1].id, "fell_back": False}


@pytest.mark.asyncio
async def test_frontier_only_after_local_and_cpu_both_fail(tmp_path):
    def h(req):
        if req.url.host == "t1":
            raise httpx.ConnectError("rocm box down")
        if req.url.host == "t2":
            return httpx.Response(503)
        return ok(req)

    reg, router, http, hits, _ = setup(tmp_path, h)
    assert await LLM(router, http).chat([]) == "from t3"
    assert hits == ["t1", "t2", "t3"]
    assert router.served("chat")["chat"]["fell_back"] and router.served("chat")["chat"]["tier"] == 3


@pytest.mark.asyncio
async def test_cpu_tier_catches_before_frontier(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, lambda r: ok(r) if r.url.host != "t1" else httpx.Response(500))
    assert await LLM(router, http).chat([]) == "from t2"
    assert "t3" not in hits  # frontier (paid) never touched


@pytest.mark.asyncio
async def test_failed_provider_is_skipped_during_cooldown_then_retried(tmp_path):
    state = {"t1_up": False}

    def h(req):
        if req.url.host == "t1" and not state["t1_up"]:
            raise httpx.ConnectError("down")
        return ok(req)

    clock[0] = 1000.0
    reg, router, http, hits, _ = setup(tmp_path, h)
    llm = LLM(router, http)
    await llm.chat([]); hits.clear()
    await llm.chat([])  # within cooldown: straight to T2, no wasted T1 attempt
    assert hits == ["t2"]
    state["t1_up"] = True; hits.clear(); clock[0] += reg.cooldown + 1
    assert await llm.chat([]) == "from t1" and hits == ["t1"]  # cooldown over, T1 is back in front
    assert router.health_snapshot()[reg.chain("chat")[0][1].id]["cooling"] is False


@pytest.mark.asyncio
async def test_cooling_providers_are_still_the_last_resort(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, lambda r: httpx.Response(500) if r.url.host != "t1" else ok(r))
    router.health[reg.chain("chat")[0][1].id] = Health(failed=clock[0], fails_in_row=1)
    assert await LLM(router, http).chat([]) == "from t1"  # t2, t3 failed; cooling t1 tried last and answered
    assert hits == ["t2", "t3", "t1"]


@pytest.mark.asyncio
async def test_all_tiers_failing_reports_every_reason(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(AllTiersFailed) as e:
        await LLM(router, http).chat([])
    assert len(e.value.errors) == 3 and "T3 · Frontier" in e.value.errors[2]


@pytest.mark.asyncio
async def test_bad_json_from_a_tier_counts_as_failure(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, lambda r: httpx.Response(200, text="<html>proxy error</html>") if r.url.host == "t1" else ok(r))
    assert await LLM(router, http).chat([]) == "from t2"


@pytest.mark.asyncio
async def test_unassigned_profiles_are_never_called_and_empty_role_is_not_configured(tmp_path):
    s = Settings(project_dir=tmp_path)
    reg = Registry(s)
    reg.upsert("chat", {"label": "Primary", "base_url": "http://primary", "tier": 1})
    reg.upsert("chat", {"label": "Spare", "base_url": "http://spare"})  # saved, in no tier
    hits = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (hits.append(r.url.host), ok(r))[1]))
    await LLM(Router(reg), http).chat([])
    assert hits == ["primary"]  # the unassigned spare is never called
    reg.set_tiers("chat", [[], [], []])
    with pytest.raises(NotConfigured):
        await LLM(Router(reg), http).chat([])


@pytest.mark.asyncio
async def test_stt_falls_back_and_resends_the_audio(tmp_path):
    bodies = []

    def h(req):
        bodies.append(len(req.content))
        return httpx.Response(500) if req.url.host == "t1" else ok(req)

    reg, router, http, hits, _ = setup(tmp_path, h, role="stt")
    assert await STT(router, http).transcribe(b"x" * 5000) == "from t2"
    assert bodies[0] == bodies[1] > 5000  # the same clip went to both


@pytest.mark.asyncio
async def test_tts_enabled_follows_tiers(tmp_path):
    s = Settings(project_dir=tmp_path)
    reg = Registry(s)
    tts = TTS(Router(reg))
    assert not tts.enabled  # seeded local TTS has no URL, so it is not callable
    reg.upsert("tts", {"label": "Kokoro", "base_url": "http://kokoro:8880", "tier": 2})
    assert tts.enabled


@pytest.mark.asyncio
async def test_embeddings_never_fail_over(tmp_path):
    reg, router, http, hits, _ = setup(tmp_path, lambda r: httpx.Response(500), role="embed")
    calls = []

    async def fn(p):
        calls.append(p.base_url)
        raise RuntimeError("boom")

    with pytest.raises(AllTiersFailed):
        await router.call("embed", fn)
    assert calls == ["http://t1"]  # a different model's vectors must not be mixed into the same memory


def test_old_active_format_migrates_to_tier_1(tmp_path):
    import json
    s = Settings(project_dir=tmp_path)
    s.state_dir.mkdir(parents=True)
    (s.state_dir / "providers.json").write_text(json.dumps({"active": {"chat": "b"}, "profiles": {"chat": [
        {"id": "a", "label": "A", "base_url": "http://a"}, {"id": "b", "label": "B", "base_url": "http://b"}]}}))
    reg = Registry(s)
    assert reg.tiers["chat"] == [["b"], [], []] and reg.get("chat").base_url == "http://b"


def test_nothing_is_seeded_without_urls_and_env_urls_seed_t1(tmp_path):
    empty = Registry(Settings(project_dir=tmp_path / "a"))
    assert all(empty.chain(r) == [] for r in ("chat", "embed", "stt", "tts", "graph"))
    s = Settings(project_dir=tmp_path / "b", chat_url="http://c:1", stt_url="http://s:2/v1/audio/transcriptions", tts_url="")
    full = Registry(s)
    assert full.get("chat").base_url == "http://c:1" and full.get("stt").base_url == "http://s:2"
    assert full.get("embed").base_url == "http://c:1"  # embeddings default to the chat server but are their own provider
    assert full.get("tts") is None and full.tiers["chat"] == [["env-chat"], [], []]
    s.embed_url = "http://e:3"
    assert Registry(Settings(project_dir=tmp_path / "c", chat_url="http://c:1", embed_url="http://e:3")).get("embed").base_url == "http://e:3"


def test_removed_seed_is_not_resurrected_on_restart(tmp_path):
    s = Settings(project_dir=tmp_path, chat_url="http://c:1")
    reg = Registry(s)
    reg.delete("chat", "env-chat")
    assert Registry(s).chain("chat") == []  # the writer deleted it in the admin menu; .env must not bring it back
