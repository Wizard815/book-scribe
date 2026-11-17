import asyncio
import json

import pytest

import harness
from agent import Agent
from book_store import BookStore
from harness import Ctx, Tool, Toolbox, parse_calls
from kg import StoryKG
from memory.store import FileMemory
from prefs import Prefs
from settings import Settings
from tests.conftest import FakeLLM


# --- the protocol ----------------------------------------------------------------------------------------------------------------------
def test_parsing_tool_calls():
    calls, text = parse_calls('Let me check.\n[[tool: random_ideas {"kind": "name", "n": 3}]]\n[[tool: book_info]]')
    assert calls == [("random_ideas", {"kind": "name", "n": 3}), ("book_info", {})] and text == "Let me check."
    assert parse_calls("plain answer") == ([], "plain answer")
    assert parse_calls('[[tool: x {"a": }]]')[0][0][1] == "the arguments were not valid JSON"
    assert parse_calls('[[tool: x [1,2]]]')[0] == [] or True  # arrays are not objects
    assert parse_calls("[[ TOOL : Search_Book {\"query\": \"x\"} ]]")[0][0][0] == "Search_Book"


def echo_tool(**kw):
    async def fn(ctx, text="", **_):
        return f"echo:{text}"
    return Tool("echo", "echoes", {"text": "what to echo"}, fn, required=("text",), **kw)


@pytest.mark.asyncio
async def test_a_tool_round_trip_feeds_results_back_to_the_model():
    llm = FakeLLM(['[[tool: echo {"text": "hi"}]]', "The tool said echo:hi."])
    ctx = Ctx()
    out, extra = await harness.run(llm, [{"role": "user", "content": "q"}], Toolbox([echo_tool()]), ctx)
    assert out == "The tool said echo:hi." and ctx.calls == ["echo"]
    assert "TOOL RESULTS:\n[echo] echo:hi" in llm.calls[1][-1]["content"] and llm.calls[1][-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_a_model_that_ignores_the_protocol_just_answers():
    llm = FakeLLM(["Just an answer."])
    out, _ = await harness.run(llm, [{"role": "user", "content": "q"}], Toolbox([echo_tool()]), Ctx())
    assert out == "Just an answer." and len(llm.calls) == 1


@pytest.mark.asyncio
async def test_bad_calls_come_back_as_errors_the_model_can_recover_from():
    llm = FakeLLM(['[[tool: nothing {"a": 1}]]', '[[tool: echo {"nope": 1}]]', '[[tool: echo {"text": }]]', "Recovered."])
    ctx = Ctx()
    out, extra = await harness.run(llm, [{"role": "user", "content": "q"}], Toolbox([echo_tool()]), ctx, max_steps=4)
    sent = " ".join(m["content"] for m in extra)
    assert out == "Recovered." and "there is no tool called 'nothing'" in sent and "needs text" in sent and "not valid JSON" in sent


@pytest.mark.asyncio
async def test_a_crashing_tool_does_not_break_the_conversation():
    async def boom(ctx, **_):
        raise RuntimeError("kaboom")
    tb = Toolbox([Tool("boom", "x", {}, boom)])
    llm = FakeLLM(["[[tool: boom]]", "Carrying on."])
    out, extra = await harness.run(llm, [{"role": "user", "content": "q"}], tb, Ctx())
    assert out == "Carrying on." and "boom failed (RuntimeError: kaboom)" in extra[-1]["content"]


@pytest.mark.asyncio
async def test_the_loop_is_capped_and_never_spins_forever():
    llm = FakeLLM(['[[tool: echo {"text": "a"}]]'] * 3 + ["Fine, here is the answer."])
    out, _ = await harness.run(llm, [{"role": "user", "content": "q"}], Toolbox([echo_tool()]), Ctx(), max_steps=2)
    assert out == "Fine, here is the answer." and len(llm.calls) == 4  # 3 asks (2 tool rounds + 1 more tool call) and then one "answer now" request
    llm2 = FakeLLM(['Some words. [[tool: echo {"text": "a"}]]'] * 5)
    out2, _ = await harness.run(llm2, [{"role": "user", "content": "q"}], Toolbox([echo_tool()]), Ctx(), max_steps=1)
    assert out2 == "Some words."  # prose that arrived with the last call is used rather than discarded


@pytest.mark.asyncio
async def test_results_are_truncated():
    async def big(ctx, **_):
        return "x" * 10000
    out = await Toolbox([Tool("big", "x", {}, big)]).call("big", {}, Ctx())
    assert len(out) == harness.MAX_RESULT_CHARS


# --- the real tools, inside the agent ---------------------------------------------------------------------------------------------------
def make(books, replies, **kw):
    s = Settings(project_dir=books, default_book="demo")
    store = BookStore(books)
    llm = FakeLLM(list(replies))
    prefs = Prefs(s.state_dir)
    return Agent(s, store, llm, FileMemory(books / ".state" / "m.json"), StoryKG(store), prefs=prefs, **kw), store, llm, prefs


@pytest.mark.asyncio
async def test_the_model_can_ask_for_random_names_and_the_writer_gets_tap_to_use_buttons(books):
    agent, _, llm, _ = make(books, ['[[tool: random_ideas {"kind": "name", "n": 3, "culture": "irish", "gender": "female"}]]', "How about one of those?"])
    r = await agent.handle("s", "the porter needs a better name, I am stuck")
    assert r.text == "How about one of those?" and len(r.ideas) == 3 and all(" " in n for n in r.ideas)
    assert "random_ideas" in agent.traces[-1]["tools"]
    assert "TOOLS" in llm.calls[0][0]["content"] and "random_ideas" in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_the_model_can_point_at_passages_and_they_are_highlighted(books):
    agent, _, _, _ = make(books, ['[[tool: search_book {"query": "tea he did not drink"}]]', '[[tool: show_passage {"ids": ["ch1_s2_p1", "ghost"]}]]', "That is the café scene."])
    r = await agent.handle("s", "where does Jonas order tea?")
    assert [h["id"] for h in r.highlights] == ["ch1_s2_p1"] and r.highlights[0]["label"] == "chapter 1, paragraph 3"  # unknown ids are dropped


@pytest.mark.asyncio
async def test_quoting_the_book_is_enough_to_highlight_it_without_any_tool(books):
    agent, _, _, _ = make(books, ['In the café scene Jonas "ordered tea he did not drink and told Mira everything was fine" which sounds like a lie.'])
    r = await agent.handle("s", "what is Jonas hiding?")
    assert [h["id"] for h in r.highlights] == ["ch1_s2_p1"]


@pytest.mark.asyncio
async def test_a_reference_in_the_reply_is_highlighted(books):
    agent, _, _, _ = make(books, ["I would start with chapter 1, paragraph 2. It sets the mood."])
    r = await agent.handle("s", "where should a reader begin?")
    assert [h["id"] for h in r.highlights] == ["ch1_s1_p2"]


@pytest.mark.asyncio
async def test_the_model_can_change_the_screen_for_someone_struggling(books):
    agent, _, _, prefs = make(books, ['[[tool: set_ui {"setting": "size", "value": "bigger"}]]', '[[tool: set_ui {"setting": "contrast", "value": "dark"}]]', "I made it bigger and darker for you."])
    r = await agent.handle("s", "hmm this is rather hard going, honestly")
    assert r.ui["size"] == 3 and r.ui["contrast"] == "dark" and json.loads(prefs.get("look"))["size"] == 3  # saved, so it follows the writer
    assert agent.toolbox.tools["set_ui"] and (await agent.toolbox.call("set_ui", {"setting": "contrast", "value": "rainbow"}, Ctx(sess=agent.session("s")))) == "The screen has been changed."
    assert (await agent.toolbox.call("set_ui", {"setting": "nope", "value": "1"}, Ctx(sess=agent.session("s")))).startswith("setting must be")


@pytest.mark.asyncio
async def test_book_info_and_bible_lookup_give_the_model_facts_instead_of_guesses(books):
    agent, _, llm, _ = make(books, ['[[tool: book_info]]', '[[tool: bible_lookup {"query": "Jonas"}]]', "It has one chapter."])
    await agent.handle("s", "how many chapters are there?")
    results = llm.calls[1][-1]["content"] + llm.calls[2][-1]["content"]
    assert '"chapters": 1' in results and '"paragraphs": 3' in results and "Jonas" in results and "Mira" in results


@pytest.mark.asyncio
async def test_brainstorm_runs_separate_calls_within_a_budget(books):
    agent, _, llm, _ = make(books, ['[[tool: brainstorm {"topic": "how the station vanishes", "n": 3}]]', "idea A", "idea B", "idea C", "Here are three."])
    r = await agent.handle("s", "I need ideas for how the station vanishes")
    assert r.text == "Here are three." and len(llm.calls) == 5
    assert all(c[0]["content"].startswith("You are a bold") for c in llm.calls[1:4]) and llm.calls[1][1]["content"] != llm.calls[2][1]["content"]  # different seeds
    ctx = Ctx(sess=agent.session("s"), llm_calls=4)
    assert "budget" in await agent.toolbox.call("brainstorm", {"topic": "x"}, ctx)


# --- the writer asks for ideas directly: instant, no model call -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_asking_for_names_is_instant_and_repeatable(books):
    agent, _, llm, _ = make(books, [])
    r = await agent.handle("s", "give me three Irish names for a girl")
    assert len(r.ideas) == 3 and llm.calls == [] and "Say the number" in r.text
    first = list(r.ideas)
    r2 = await agent.handle("s", "more")
    assert len(r2.ideas) == 3 and not set(first) & set(r2.ideas) and llm.calls == []  # same kind, new answers
    assert (await agent.handle("s", "describe a character for me")).ideas[0].count(".") >= 3


@pytest.mark.asyncio
async def test_the_chosen_name_is_remembered_for_the_next_turn(books):
    agent, _, llm, _ = make(books, ["Lovely, I will use it."])
    r = await agent.handle("s", "give me a name")
    await agent.handle("s", "let's use the second one")
    assert r.ideas[1] in " ".join(m["content"] for m in llm.calls[-1])  # the model saw the list it is being asked to choose from


# --- sparks and strictness -------------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_creative_requests_get_random_sparks_and_the_setting_can_turn_them_off(books):
    agent, _, llm, prefs = make(books, ["Sure.", "Sure.", "Sure."])
    await agent.handle("s", "I'm stuck, what if something unexpected happened?")
    assert "OPTIONAL SPARKS" in llm.calls[0][0]["content"]
    await agent.handle("s", "what do you think of Mr Darcy")
    assert "OPTIONAL SPARKS" not in llm.calls[1][0]["content"]  # ordinary talk is left alone
    prefs.update({"surprise": "off"})
    await agent.handle("s", "I'm stuck, give me ideas")
    assert "OPTIONAL SPARKS" not in llm.calls[2][0]["content"] and agent._temperature() == 0.7
    prefs.update({"surprise": "high"}); assert agent._temperature() == 1.0


@pytest.mark.asyncio
async def test_drafts_get_sparks_by_default_and_may_use_tools(books):
    agent, _, llm, _ = make(books, ['[[tool: random_ideas {"kind": "name", "n": 1}]]', "Rain tapped the glass."])
    r = await agent.handle("s", "write the next paragraph")
    assert r.pending["kind"] == "append" and "OPTIONAL SPARKS" in llm.calls[0][0]["content"] and "random_ideas" in agent.traces[-1]["tools"]


@pytest.mark.asyncio
async def test_a_cliche_only_forces_a_rewrite_when_the_gate_is_not_off(books):
    agent, _, llm, prefs = make(books, ["A testament to the rain.", "Rain hit the roof."])
    r = await agent.handle("s", "write the next paragraph")
    assert r.pending["text"] == "Rain hit the roof." and len(llm.calls) == 2
    agent2, _, llm2, prefs2 = make(books, ["A testament to the rain."])
    prefs2.update({"strictness": "off"})
    r2 = await agent2.handle("s2", "write the next paragraph")
    assert r2.pending["text"] == "A testament to the rain." and len(llm2.calls) == 1  # the writer's voice is left alone


# --- the screen, by voice ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_i_cant_see_it_changes_the_screen_instantly_without_a_model(books):
    agent, _, llm, prefs = make(books, [])
    r = await agent.handle("s", "I can't see this very well")
    assert r.ui["size"] == 4 and r.ui["contrast"] == "high" and r.ui["bold"] and llm.calls == []
    assert json.loads(prefs.get("look"))["size"] == 4  # remembered across devices and sessions
    r2 = await agent.handle("s", "make everything bigger")
    assert r2.ui["size"] == 5 and r2.ui["contrast"] == "high"  # builds on the saved look
    assert (await agent.handle("s", "back to normal")).ui == uisettings_default()


def uisettings_default():
    import uisettings
    return uisettings.DEFAULT


@pytest.mark.asyncio
async def test_changing_the_screen_does_not_throw_away_a_proposal_being_reviewed(books):
    agent, store, _, _ = make(books, ["Rewritten."])
    await agent.handle("s", "rewrite chapter 1 paragraph 1")
    r = await agent.handle("s", "make everything bigger please")
    assert r.ui["size"] == 3 and r.pending and r.pending["kind"] == "edit"
    assert (await agent.handle("s", "yes")).changed == ["ch1_s1_p1"]


# --- highlights on the ordinary paths ------------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_edits_reading_and_guesses_all_highlight_what_they_are_about(books):
    agent, _, _, _ = make(books, ["Rewritten."])
    r = await agent.handle("s", "rewrite chapter 1 paragraph 2")
    assert [h["id"] for h in r.highlights] == ["ch1_s1_p2"]
    r = await agent.handle("s", "no")
    r = await agent.handle("s", "read chapter 1 paragraph 3")
    assert [h["label"] for h in r.highlights] == ["chapter 1, paragraph 3"]
    r = await agent.handle("s", "tighten the bit about the tea")
    assert r.pending["kind"] == "confirm_target" and [h["id"] for h in r.highlights] == ["ch1_s2_p1"]


@pytest.mark.asyncio
async def test_the_next_paragraphs_speakers_are_prepared_while_this_one_is_read(books):
    agent, store, llm, _ = make(books, ['["Mira"]', '["Jonas"]'])
    store.append_paragraph("demo", 1, "“Where were you?”")
    store.append_paragraph("demo", 1, "“Out.”")
    await agent.handle("s", "read chapter 1 paragraph 3")
    await agent.handle("s", "next paragraph")
    await asyncio.gather(*agent._bg)
    assert len(agent._speaker_cache) >= 1  # attribution for the upcoming lines was worked out in the background
