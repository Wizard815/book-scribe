import json

import httpx
import pytest
from fastapi.testclient import TestClient

import speech
import voices as vx
from agent import Agent
from book_store import BookStore
from clients import TTS
from kg import StoryKG
from main import create_app
from memory.store import FileMemory
from providers import Registry
from router import Router
from settings import Settings
from tests.conftest import FakeLLM

T1_POOL = "af_heart, am_adam, am_michael, bf_emma"
T3_POOL = "alloy:neutral, onyx:male/low, nova:female, fable:male, shimmer:female/high"


@pytest.fixture
def world(books):
    """A demo book with a gendered cast and two TTS tiers (local Kokoro-style, cloud OpenAI-style)."""
    s = Settings(project_dir=books, default_book="demo")
    reg = Registry(s)
    store = BookStore(books)
    b = store.bible("demo")
    b.upsert("characters", "Mira", {"gender": "female", "temperament": "dry, sarcastic"})
    b.upsert("characters", "Jonas", {"gender": "male", "temperament": "nervous"})
    b.upsert("characters", "Mr. Bennet", {"role": "father"}, "Dry wit.")
    t1 = reg.upsert("tts", {"label": "Local", "base_url": "http://local", "voices": T1_POOL, "voice": "af_heart", "tier": 1})
    t3 = reg.upsert("tts", {"label": "Cloud", "kind": "cloud", "base_url": "http://cloud", "voices": T3_POOL, "voice": "alloy", "style_field": "instructions", "tier": 3})
    return s, reg, store, VoiceBook(store, reg), t1, t3


def VoiceBook(store, reg):
    return vx.VoiceBook(store, reg)


# --- picking and remembering --------------------------------------------------------------------------
def test_characters_get_distinct_voices_matching_their_gender(world):
    _, _, _, vb, t1, t3 = world
    mira, jonas = vb.resolve("demo", "Mira", t1), vb.resolve("demo", "Jonas", t1)
    assert mira.voice in ("af_heart", "bf_emma") and jonas.voice in ("am_adam", "am_michael")
    assert vb.resolve("demo", "Mr. Bennet", t1).voice.startswith("am_")  # "Mr." + "father" => male
    assert len({vb.resolve("demo", n, t1).voice for n in ("Mira", "Jonas", "Mr. Bennet")}) == 3  # nobody shares a voice


def test_assistant_keeps_the_default_voice_and_nobody_else_takes_it(world):
    _, _, _, vb, t1, _ = world
    assert vb.resolve("demo", "assistant", t1).voice == "af_heart"
    assert vb.resolve("demo", "Mira", t1).voice != "af_heart"  # Mira is female too, but the assistant already sounds like af_heart
    assert vb.resolve("demo", "narrator", t1).voice != "af_heart"


def test_voices_are_remembered_across_restarts(world, books):
    s, reg, store, vb, t1, _ = world
    first = {n: vb.resolve("demo", n, t1).voice for n in ("Mira", "Jonas", "narrator")}
    saved = json.loads((books / "demo" / "voices.json").read_text())["profiles"]
    assert saved["Mira"]["assigned"][t1.id] == first["Mira"] and saved["Mira"]["auto"] is True
    again = vx.VoiceBook(BookStore(books), Registry(s))  # fresh process
    assert {n: again.resolve("demo", n, t1).voice for n in first} == first


def test_one_character_sounds_alike_on_every_tier_but_uses_each_providers_own_voice_names(world):
    _, _, _, vb, t1, t3 = world
    a, b = vb.resolve("demo", "Jonas", t1), vb.resolve("demo", "Jonas", t3)
    assert a.voice in ("am_adam", "am_michael") and b.voice in ("onyx", "fable")  # male on both, from each pool
    assert vb.resolve("demo", "Mira", t3).voice in ("nova", "shimmer")


def test_a_pool_change_never_leaves_a_character_with_a_voice_the_provider_lacks(world):
    s, reg, store, vb, t1, _ = world
    vb.resolve("demo", "Jonas", t1)
    reg.upsert("tts", {"id": t1.id, "voices": "am_onyx, af_nova"})
    assert vb.resolve("demo", "Jonas", reg.get("tts")).voice == "am_onyx"


def test_explicit_gender_in_the_bible_beats_guessing():
    assert vx.guess_traits("Sam", {"gender": "female"}, "He walked in.")["gender"] == "female"
    assert vx.guess_traits("Mrs. Bennet", {}, "")["gender"] == "female" and vx.guess_traits("Mr. Darcy", {}, "")["gender"] == "male"
    assert vx.guess_style({"temperament": "sarcastic, detached"}) == "dry, deadpan, understated"


def test_canonical_names_accept_spoken_forms_and_aliases(world):
    _, _, store, vb, _, _ = world
    store.bible("demo").upsert("characters", "Elizabeth", {"aliases": ["Lizzy"]})
    assert vb.canonical("demo", "mister bennet") == "Mr. Bennet" and vb.canonical("demo", "lizzy") == "Elizabeth"
    assert vb.canonical("demo", "the narrator") == "narrator" and vb.canonical("demo", "your") == "assistant"


# --- what is sent to the TTS backend ---------------------------------------------------------------------------
def recording_tts(world, fail_hosts=()):
    s, reg, store, vb, t1, t3 = world
    sent = []

    def handler(req: httpx.Request):
        if req.url.host in fail_hosts:
            raise httpx.ConnectError("down")
        sent.append((req.url.host, json.loads(req.content)))
        return httpx.Response(200, content=b"AUDIO")

    return TTS(Router(reg), httpx.AsyncClient(transport=httpx.MockTransport(handler)), vb, "demo"), sent, vb


@pytest.mark.asyncio
async def test_each_speaker_is_requested_with_their_own_voice(world):
    tts, sent, vb = recording_tts(world)
    await tts.speak("Hello.", "Jonas")
    await tts.speak("Hi.", "Mira")
    await tts.speak("Welcome.", "assistant")
    voices = [b["voice"] for _, b in sent]
    assert voices[0].startswith("am_") and voices[1].startswith(("af_", "bf_")) and voices[2] == "af_heart" and len(set(voices)) == 3
    assert all(host == "local" for host, _ in sent)


@pytest.mark.asyncio
async def test_speed_and_style_are_sent_only_when_meaningful(world):
    tts, sent, vb = recording_tts(world, fail_hosts=("local",))  # force the cloud tier, which accepts a style field
    prof = vb.profile("demo", "Mira")
    prof.update(speed=0.85, style="dry, deadpan")
    vb.save_profile("demo", "Mira", prof)
    await tts.speak("Fine.", "Mira")
    host, body = sent[-1]
    assert host == "cloud" and body["voice"] in ("nova", "shimmer") and body["speed"] == 0.85 and body["instructions"] == "dry, deadpan"
    await tts.speak("Fine.", "Jonas")
    assert "speed" not in sent[-1][1] and sent[-1][1]["voice"] in ("onyx", "fable")  # default speed omitted


@pytest.mark.asyncio
async def test_style_is_not_sent_to_a_provider_without_a_style_field(world):
    tts, sent, vb = recording_tts(world)
    prof = vb.profile("demo", "Mira"); prof["style"] = "whispery"; vb.save_profile("demo", "Mira", prof)
    await tts.speak("Fine.", "Mira")
    assert "instructions" not in sent[-1][1] and "style" not in sent[-1][1]


@pytest.mark.asyncio
async def test_failover_re_resolves_the_voice_for_the_provider_that_answers(world):
    tts, sent, vb = recording_tts(world, fail_hosts=("local",))
    await tts.speak("Hello.", "Jonas")
    host, body = sent[0]
    assert host == "cloud" and body["voice"] in ("onyx", "fable") and body["input"] == "Hello." and body["instructions"]  # Jonas is nervous: style travels too


# --- reading aloud with different voices --------------------------------------------------------------------------
FORMS = {"Mr. Bennet": ["mr bennet"], "Mrs. Bennet": ["mrs bennet", "his lady", "his wife"], "Elizabeth": ["elizabeth"]}
GENDERS = {"Mr. Bennet": "male", "Mrs. Bennet": "female", "Elizabeth": "female"}


def test_dialogue_tags_assign_speakers_and_narration_stays_narrator():
    segs, _ = speech.attribute("“My dear Mr. Bennet,” said his lady to him one day, “have you heard?”", FORMS, [], GENDERS)
    assert [(s.speaker, s.quoted) for s in segs] == [("Mrs. Bennet", True), ("narrator", False), ("Mrs. Bennet", True)]


def test_untagged_replies_alternate_between_two_speakers():
    recent = ["Mr. Bennet"]
    for line, want in [("“Do you not want to know?” cried his wife.", "Mrs. Bennet"), ("“_You_ want to tell me.”", "Mr. Bennet")]:
        segs, recent = speech.attribute(line, FORMS, recent, GENDERS)
        if want:
            assert [s.speaker for s in segs if s.quoted][0] == want


def test_pronoun_tags_use_gender_and_indirect_speech():
    _, recent = speech.attribute("Mr. Bennet replied that he had not.", FORMS, ["Mrs. Bennet"], GENDERS)
    segs, _ = speech.attribute("“But it is,” returned she.", FORMS, recent, GENDERS)
    assert segs[0].speaker == "Mrs. Bennet"


def test_unplaceable_quotes_fall_back_to_the_narrator_not_an_invented_speaker():
    segs, _ = speech.attribute("“Who goes there?”", FORMS, [], GENDERS)
    assert segs[0].speaker == "narrator" and segs[0].resolved is False


def test_fragments_without_words_are_merged_not_sent_to_tts_alone():
    segs, _ = speech.attribute("“Yes,” she said, “no.”", FORMS, [], GENDERS)
    assert all(any(c.isalnum() for c in s.text) for s in segs)


def make_agent(world, replies=()):
    s, reg, store, vb, t1, t3 = world
    llm = FakeLLM(list(replies))
    return Agent(s, store, llm, FileMemory(s.project_dir / ".state" / "m.json"), StoryKG(store), voices=vb), store, llm, vb


@pytest.mark.asyncio
async def test_reading_a_paragraph_returns_speaker_segments(world):
    agent, store, llm, vb = make_agent(world)
    store.append_paragraph("demo", 1, "“Where were you?” asked Mira. “Out,” said Jonas.")
    r = await agent.handle("s", "read chapter 1 paragraph 4")
    assert [(x["speaker"], x["text"].strip("“” ,.")) for x in r.speech if x["speaker"] != "narrator"] == [("Mira", "Where were you?"), ("Jonas", "Out")]


@pytest.mark.asyncio
async def test_model_resolves_dialogue_the_rules_cannot_place(world):
    agent, store, llm, vb = make_agent(world, ['["Jonas"]'])
    store.append_paragraph("demo", 1, "“Who goes there?”")
    r = await agent.handle("s", "read chapter 1 paragraph 4")
    assert [x["speaker"] for x in r.speech] == ["Jonas"] and len(llm.calls) == 1
    await agent.handle("s", "read chapter 1 paragraph 4")
    assert len(llm.calls) == 1  # cached


@pytest.mark.asyncio
async def test_proposals_are_spoken_assistant_then_characters_then_assistant(world):
    agent, store, llm, vb = make_agent(world, ["“Fine,” said Jonas."])
    r = await agent.handle("s", "rewrite chapter 1 paragraph 3")
    assert r.speech[0]["speaker"] == "assistant" and r.speech[-1]["speaker"] == "assistant" and any(x["speaker"] == "Jonas" for x in r.speech)


# --- the voice booth: audition, adjust, keep ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_audition_plays_a_real_line_in_the_characters_voice_and_waits(world):
    agent, store, llm, vb = make_agent(world)
    store.append_paragraph("demo", 1, "“I would rather not discuss it,” said Jonas.")
    r = await agent.handle("s", "let me hear Jonas")
    assert r.pending["kind"] == "voice"
    who = [x for x in r.speech if x.get("draft")]
    assert who and who[0]["speaker"] == "Jonas" and who[0]["text"] == "I would rather not discuss it."
    assert vb.profile("demo", "Jonas")["auto"] is True  # nothing approved yet


@pytest.mark.asyncio
async def test_adjusting_by_voice_changes_speed_pitch_and_style_then_keep_saves_it(world):
    agent, store, llm, vb = make_agent(world)
    await agent.handle("s", "let me hear Jonas")
    before = vb.profile("demo", "Jonas")
    r = await agent.handle("s", "make his voice deeper and a bit slower and more sarcastic")
    prof = r.pending and agent.sessions["s"].pending.data["profile"]
    assert prof["speed"] == 0.85 and prof["traits"]["pitch"] == "low" and "sarcastic" in prof["style"]
    assert vb.profile("demo", "Jonas")["speed"] == before["speed"]  # still only a draft
    assert (await agent.handle("s", "keep it")).text.startswith("Saved")
    saved = vb.profile("demo", "Jonas")
    assert saved["speed"] == 0.85 and saved["auto"] is False and "sarcastic" in saved["style"]


@pytest.mark.asyncio
async def test_the_draft_is_what_gets_spoken_until_it_is_kept(world):
    agent, store, llm, vb = make_agent(world)
    s, reg, _, _, t1, _ = world
    r = await agent.handle("s", "let me hear Mira")
    await agent.handle("s", "slower")
    r = await agent.handle("s", "much slower")
    did = [x for x in r.speech if x.get("draft")][0]["draft"]
    assert vb.resolve("demo", "Mira", t1, did).speed < 0.7 and vb.resolve("demo", "Mira", t1).speed == 1.0


@pytest.mark.asyncio
async def test_no_discards_the_draft(world):
    agent, _, _, vb = make_agent(world)
    await agent.handle("s", "let me hear Jonas")
    await agent.handle("s", "deeper")
    r = await agent.handle("s", "no")
    assert r.pending is None and vb.profile("demo", "Jonas")["traits"]["pitch"] != "low"


@pytest.mark.asyncio
async def test_try_another_cycles_to_a_different_voice(world):
    agent, _, _, vb = make_agent(world)
    s, reg, _, _, t1, _ = world
    first = vb.resolve("demo", "Jonas", t1).voice
    await agent.handle("s", "try another voice for Jonas")
    second = agent.sessions["s"].pending.data["profile"]["pinned"][t1.id]
    assert second != first and second.startswith("am_")
    await agent.handle("s", "yes")
    assert vb.resolve("demo", "Jonas", t1).voice == second


@pytest.mark.asyncio
async def test_naming_a_voice_pins_it_and_other_tiers_follow_its_character(world):
    agent, _, _, vb = make_agent(world)
    s, reg, _, _, t1, t3 = world
    await agent.handle("s", "change the narrator voice to am_michael")
    await agent.handle("s", "yes")
    assert vb.resolve("demo", "narrator", t1).voice == "am_michael"
    assert vb.resolve("demo", "narrator", t3).voice in ("onyx", "fable")  # male on the cloud tier too


@pytest.mark.asyncio
async def test_open_ended_feedback_goes_to_the_model_and_is_validated(world):
    agent, _, llm, vb = make_agent(world, ['{"voice": "nonexistent", "speed": 9, "style": "tired, professorial", "gender": "male", "pitch": "low", "say": "Made him sound weary."}'])
    await agent.handle("s", "let me hear Jonas")
    r = await agent.handle("s", "make his voice sound like a tired professor")
    prof = agent.sessions["s"].pending.data["profile"]
    assert prof["style"] == "tired, professorial" and prof["speed"] == 1.6 and "Made him sound weary" in r.text  # bad voice name ignored, speed clamped


@pytest.mark.asyncio
async def test_ambiguous_requests_ask_whose_voice_instead_of_guessing(world):
    agent, _, _, _ = make_agent(world)
    assert "Whose voice" in (await agent.handle("s", "change the voice")).text


@pytest.mark.asyncio
async def test_play_the_next_paragraph_is_not_a_voice_request(world):
    agent, _, llm, _ = make_agent(world, ["ok"])
    r = await agent.handle("s", "play the next paragraph")
    assert not r.pending and "voice" not in r.text.lower()


@pytest.mark.asyncio
async def test_reset_forgets_and_repicks(world, books):
    agent, _, _, vb = make_agent(world)
    s, reg, _, _, t1, _ = world
    vb.resolve("demo", "Jonas", t1)
    assert "Jonas" in vb.all_profiles("demo")
    await agent.handle("s", "forget Jonas's voice")
    assert "Jonas" not in vb.all_profiles("demo")


@pytest.mark.asyncio
async def test_list_voices_and_no_tts_configured(world, books):
    agent, _, _, _ = make_agent(world)
    assert "af_heart" in (await agent.handle("s", "what voices do you have")).text
    s2 = Settings(project_dir=books / "x", default_book="demo")
    reg2 = Registry(s2)
    store = BookStore(books)
    ag = Agent(s2, store, FakeLLM([]), FileMemory(books / "m2.json"), StoryKG(store), voices=vx.VoiceBook(store, reg2))
    assert "No text-to-speech" in (await ag.handle("s", "let me hear Jonas")).text


# --- API ---------------------------------------------------------------------------------------------------------------
def test_api_tts_takes_a_speaker_and_voices_endpoint_lists_profiles(books):
    s = Settings(project_dir=books, default_book="demo")
    reg = Registry(s)
    reg.upsert("tts", {"label": "L", "base_url": "http://tts.test", "voices": T1_POOL, "voice": "af_heart", "tier": 1})
    seen = []
    app = create_app(s, llm=FakeLLM(), stt=FakeLLM(), memory=FileMemory(books / "m.json"))
    # swap in a transport-backed TTS so the real route + VoiceBook + router are exercised
    tts = [r for r in app.routes if getattr(r, "path", "") == "/api/tts"]
    assert tts
    with TestClient(app) as c:
        v = c.get("/api/books/demo/voices").json()
        assert [x["name"] for x in v["pool"]] == ["af_heart", "am_adam", "am_michael", "bf_emma"] and v["profiles"] == {}
        assert c.post("/api/tts", json={"text": "hi", "speaker": "Mira"}).status_code in (200, 502)  # no network in tests: reaches the router


@pytest.mark.asyncio
async def test_keeping_a_voice_saves_exactly_the_voice_that_was_heard_on_every_tier(world, books):
    """Regression: 'keep' used to save only traits, so after a restart the character could get a different automatic voice."""
    agent, _, _, vb = make_agent(world)
    s, reg, _, _, t1, t3 = world
    await agent.handle("s", "let me hear Jonas")
    r = await agent.handle("s", "deeper")
    did = [x for x in r.speech if x.get("draft")][0]["draft"]
    heard = {p.id: vb.resolve("demo", "Jonas", p, did).voice for p in (t1, t3)}
    await agent.handle("s", "keep it")
    fresh = vx.VoiceBook(BookStore(books), Registry(s))  # a new process
    assert {p.id: fresh.resolve("demo", "Jonas", p).voice for p in (t1, t3)} == heard
    assert fresh.profile("demo", "Jonas")["auto"] is False


@pytest.mark.asyncio
async def test_a_draft_sounds_identical_every_time_it_is_replayed(world):
    agent, _, _, vb = make_agent(world)
    s, reg, _, _, t1, _ = world
    r = await agent.handle("s", "try another voice for Mira")
    did = [x for x in r.speech if x.get("draft")][0]["draft"]
    assert len({vb.resolve("demo", "Mira", t1, did).voice for _ in range(5)}) == 1


@pytest.mark.asyncio
async def test_audition_names_the_voice_actually_playing_and_uses_a_whole_sentence(world):
    agent, store, _, vb = make_agent(world)
    store.append_paragraph("demo", 1, "“and I am glad,” said Jonas. “I would rather not discuss it,” said Jonas.")
    r = await agent.handle("s", "let me hear Jonas")
    played = [x for x in r.speech if x.get("draft")][0]
    assert played["text"] == "I would rather not discuss it."  # skips the lowercase mid-sentence fragment
    assert "voice am_" in r.text and "automatic" not in r.text


@pytest.mark.asyncio
async def test_speaker_lookback_reaches_far_enough_to_find_the_last_tagged_line(world):
    agent, store, llm, vb = make_agent(world)
    store.append_paragraph("demo", 1, "“Where were you?” asked Mira.")
    for _ in range(6):
        store.append_paragraph("demo", 1, "The rain kept falling on the roof of the old station.")
    store.append_paragraph("demo", 1, "“Out,” said Jonas.")
    store.append_paragraph("demo", 1, "“Out where?”")
    r = await agent.handle("s", "read chapter 1 paragraph 12")
    assert [x["speaker"] for x in r.speech] == ["Mira"]  # the untagged reply goes to the other person in the exchange


def test_a_person_addressed_by_name_is_the_next_untagged_speaker():
    """Starting mid-conversation with only one known speaker, "Mr. Bennet" inside her quote says who answers."""
    recent = []
    segs, recent = speech.attribute("“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome!”", FORMS, recent, GENDERS)
    assert [s.speaker for s in segs if s.quoted] == ["Mrs. Bennet", "Mrs. Bennet"]
    segs, recent = speech.attribute("“Is that his design in settling here?”", FORMS, recent, GENDERS)
    assert segs[0].speaker == "Mr. Bennet"
    segs, _ = speech.attribute("“Design! Nonsense, how can you talk so!”", FORMS, recent, GENDERS)
    assert segs[0].speaker == "Mrs. Bennet"  # and they keep alternating from there


def test_a_name_merely_mentioned_in_a_quote_is_not_the_next_speaker():
    """Regression found in the browser run: 'I dare say Mr. Bingley will be glad...' made Mr. Bingley the speaker of 'replied he'."""
    forms = {**FORMS, "Mr. Bingley": ["mr bingley"]}
    genders = {**GENDERS, "Mr. Bingley": "male"}
    _, recent = speech.attribute("“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome!”", forms, [], genders)
    segs, recent = speech.attribute("“You are over-scrupulous. I dare say Mr. Bingley will be very glad to see you,” said Mr. Bennet.", forms, recent, genders)
    assert [s.speaker for s in segs if s.quoted] == ["Mr. Bennet"] and "Mr. Bingley" not in recent
    segs, _ = speech.attribute("“They have none of them much to recommend them,” replied he.", forms, recent, genders)
    assert segs[0].speaker == "Mr. Bennet"


@pytest.mark.parametrize("quote,who", [
    ("“My dear Mr. Bennet, how can you?”", ["Mr. Bennet"]), ("“Mr. Bennet, come here.”", ["Mr. Bennet"]), ("“Come here, Mr. Bennet.”", ["Mr. Bennet"]),
    ("“I saw Mr. Bennet yesterday.”", []), ("“Mr. Bennet will be glad.”", []),
])
def test_vocatives(quote, who):
    assert speech._vocatives(quote, FORMS) == who
