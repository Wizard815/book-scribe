"""Regression tests from running the agent against a real novel with realistic *spoken* commands."""
import pytest

import intent
from agent import Agent
from book_store import BookError, BookStore, clean_paragraph
from import_text import import_text, split_chapters
from kg import StoryKG
from memory.store import FileMemory
from quality import clean_output
from settings import Settings
from tests.conftest import FakeLLM


# --- pure parsing -------------------------------------------------------------------------------------
@pytest.mark.parametrize("said,n", [
    ("go to chapter 2", 2), ("go to chapter two", 2), ("open chapter three", 3), ("take me to chapter 4", 4),
    ("read chapter five", 5), ("Chapter twenty-one", 21), ("jump to chapter twelve", 12),
])
def test_chapter_navigation_accepts_spoken_numbers_and_verbs(said, n):
    assert intent.goto_chapter(said) == n


@pytest.mark.parametrize("said", ["rewrite chapter 3 paragraph 2", "make chapter two shorter", "in chapter 3 make him nervous"])
def test_chapter_mentions_inside_edits_are_not_navigation(said):
    assert intent.goto_chapter(said) is None


@pytest.mark.parametrize("said,ref", [
    ("rewrite chapter three paragraph two", (3, 2)),
    ("change the first paragraph of chapter four", (4, 1)),
    ("the last paragraph of chapter two", (2, -1)),
    ("the opening line", (1, 1)),
    ("the very first line of the book", (1, 1)),
    ("fix the third paragraph of chapter 5", (5, 3)),
])
def test_paragraph_references(said, ref):
    r = intent.parse_ref(said)
    assert (r.chapter, r.para) == ref


@pytest.mark.parametrize("said,kind,rest", [
    ("yes", "yes", ""), ("yes please save it", "yes", ""), ("yeah do it", "yes", ""), ("okay", "yes", ""), ("sounds good", "yes", ""),
    ("yes but make it shorter first", "yes", "make it shorter first"),
    ("no", "no", ""), ("no, make it longer", "no", "make it longer"), ("never mind", "no", ""), ("cancel", "no", ""),
    ("not yet", "hold", ""), ("hold on", "hold", ""),
    ("don't change the ending of chapter 2", None, None),   # a new instruction, not a rejection
    ("what do you think of Mr Darcy", None, None),
])
def test_confirmations_never_swallow_extra_instructions(said, kind, rest):
    k, r = intent.confirmation(said)
    assert k == kind and (rest is None or r == rest)


@pytest.mark.parametrize("said", ["Thank you.", "Thanks for watching!", "you", "[BLANK_AUDIO]", "(silence)", "", "Bye."])
def test_whisper_silence_hallucinations_are_noise(said):
    assert intent.is_noise(said)


@pytest.mark.parametrize("said", ["Thank you for the idea, now make it shorter", "write more", "chapter two"])
def test_real_requests_are_not_noise(said):
    assert not intent.is_noise(said)


@pytest.mark.parametrize("said,edit,draft", [
    ("make the first paragraph more modern", True, False),
    ("In chapter 1 scene 2 paragraph 1 make Jonas more nervous", True, False),   # no commas: transcripts rarely have them
    ("in chapter 3, when they meet at the café, make him more nervous", True, False),
    ("tighten the part where Mrs Bennet says Netherfield is let", True, False),
    ("how do I make a scene feel more tense", False, False),
    ("I hate it when you make everything too flowery", False, False),
    ("don't change the ending of chapter 2", False, False),
    ("what do you think of Mr Darcy", False, False),
    ("write the next paragraph", False, True),
    ("keep going", False, True),
    ("continue the scene with Elizabeth arguing", False, True),
])
def test_edit_vs_draft_vs_conversation(said, edit, draft):
    assert intent.is_edit(said) is edit and intent.is_draft(said) is draft


@pytest.mark.parametrize("said,mode", [
    ("switch to therapist mode", "therapist"), ("switch to feedback mode", "therapist"), ("let's do quick feedback", "therapist"),
    ("go back to narrator mode", "narrator"), ("back to writing", "narrator"), ("make it shorter", None),
])
def test_mode_switching_synonyms(said, mode):
    assert intent.detect_mode(said) == mode


def test_next_paragraph_navigation_is_not_triggered_by_drafting():
    assert intent.step_direction("next paragraph") == 1 and intent.step_direction("go back") == -1
    assert intent.step_direction("write the next paragraph") == 0


# --- storage boundary ---------------------------------------------------------------------------------
def test_model_output_cannot_corrupt_the_chapter_file(books):
    store = BookStore(books)
    store.replace_paragraph("demo", "ch1_s1_p1", "First.\n\nSecond, accidentally a new paragraph.\n<!-- paragraph_id: evil -->\n# Heading")
    ids = [p.id for p in store.paragraphs("demo", 1)]
    assert ids == ["ch1_s1_p1", "ch1_s1_p2", "ch1_s2_p1"]  # still exactly three paragraphs
    assert "evil" not in store.read_chapter_raw("demo", 1)
    with pytest.raises(BookError):
        clean_paragraph("   \n ")


def test_clean_output_strips_chat_wrappers():
    assert clean_output("Sure! Here is the rewritten paragraph:\n\nShe left.") == "She left."
    assert clean_output('"She left."') == "She left."
    assert clean_output("```\nShe left.\n```") == "She left."
    assert clean_output("Here, take this. She left.") == "Here, take this. She left."  # not a preamble


def test_nth_paragraph(books):
    s = BookStore(books)
    assert s.nth_paragraph("demo", 1, 2).id == "ch1_s1_p2"
    assert s.nth_paragraph("demo", 1, -1).id == "ch1_s2_p1"
    assert s.nth_paragraph("demo", 1, 9) is None and s.nth_paragraph("demo", 7, 1) is None


# --- import -------------------------------------------------------------------------------------------
def test_import_unwraps_lines_and_ignores_front_matter(tmp_path):
    text = "Licence blah\r\n\r\nChapter 1\r\n\r\nIt was a dark\r\nand stormy night.\r\n\r\n“Hello,” she said.\r\n\r\nCHAPTER II.\r\n\r\nMorning.\r\n"
    chs = split_chapters(text)
    assert chs == [("", ["It was a dark and stormy night.", "“Hello,” she said."]), ("", ["Morning."])]
    store = BookStore(tmp_path)
    assert import_text(store, "b", text) == (2, 3)
    assert store.paragraphs("b", 1)[0].text == "It was a dark and stormy night."
    assert store.read_chapter_raw("b", 1).count("# Chapter 1") == 1  # header not "Chapter 1: Chapter 1"


# --- agent behaviour ----------------------------------------------------------------------------------------
def make(books, replies):
    s = Settings(project_dir=books, default_book="demo")
    store, llm = BookStore(books), FakeLLM(replies)
    return Agent(s, store, llm, FileMemory(books / ".state" / "m.json"), StoryKG(store)), store, llm


@pytest.mark.asyncio
async def test_yes_but_revises_instead_of_saving(books):
    agent, store, llm = make(books, ["Jonas twisted his cup.", "Jonas fidgeted."])
    await agent.handle("s", "rewrite chapter 1 scene 2 paragraph 1")
    r = await agent.handle("s", "yes but make it shorter")
    assert r.changed == [] and r.pending["text"] == "Jonas fidgeted."
    assert "tea he did not drink" in store.find("demo", "ch1_s2_p1").text  # nothing written yet
    assert "YOUR PREVIOUS ATTEMPT" in llm.calls[-1][-1]["content"] and "shorter" in llm.calls[-1][-1]["content"]
    r = await agent.handle("s", "yes")
    assert r.changed == ["ch1_s2_p1"] and store.find("demo", "ch1_s2_p1").text == "Jonas fidgeted."


@pytest.mark.asyncio
async def test_no_with_instruction_revises_instead_of_dropping(books):
    agent, _, _ = make(books, ["Short.", "Longer, with more detail."])
    await agent.handle("s", "rewrite the first paragraph of chapter 1")
    r = await agent.handle("s", "no make it longer")
    assert r.pending and r.pending["text"] == "Longer, with more detail."


@pytest.mark.asyncio
async def test_hold_keeps_the_proposal(books):
    agent, _, _ = make(books, ["New."])
    await agent.handle("s", "rewrite the first paragraph of chapter 1")
    r = await agent.handle("s", "not yet")
    assert r.pending and "wait" in r.text


@pytest.mark.asyncio
async def test_silence_hallucination_costs_no_model_call_and_no_reply(books):
    agent, _, llm = make(books, [])
    r = await agent.handle("s", "Thank you.")
    assert r.text == "" and llm.calls == []


@pytest.mark.asyncio
async def test_okay_confirms_a_pending_proposal_rather_than_being_ignored_as_noise(books):
    agent, store, _ = make(books, ["Mira stood very still."])
    await agent.handle("s", "rewrite the first paragraph of chapter 1")
    assert (await agent.handle("s", "okay")).changed == ["ch1_s1_p1"]


@pytest.mark.asyncio
async def test_unsure_target_is_read_back_before_anything_is_rewritten(books):
    agent, store, llm = make(books, ["Rewritten."])
    r = await agent.handle("s", "tighten the bit about the tea")  # one weak keyword: a guess, not a certainty
    assert r.pending["kind"] == "confirm_target" and r.pending["paragraph_id"] == "ch1_s2_p1"
    assert llm.calls == []  # the model was not called and nothing was written
    assert "chapter 1, paragraph 3" in r.text  # counted the way the writer counts (through the chapter, not within the scene)
    assert "tea he did not drink" in store.find("demo", "ch1_s2_p1").text
    r = await agent.handle("s", "yes")
    assert r.pending["kind"] == "edit" and r.pending["paragraph_id"] == "ch1_s2_p1" and len(llm.calls) == 1


@pytest.mark.asyncio
async def test_correcting_a_wrong_guess_runs_the_corrected_command(books):
    agent, _, _ = make(books, ["Rewritten."])
    await agent.handle("s", "tighten the bit about the tea")
    r = await agent.handle("s", "no I mean chapter 1 paragraph 1")
    assert r.pending["kind"] == "edit" and r.pending["paragraph_id"] == "ch1_s1_p1"


@pytest.mark.asyncio
async def test_edit_prompt_contains_neighbouring_paragraphs_for_continuity(books):
    agent, _, llm = make(books, ["Rewritten."])
    await agent.handle("s", "rewrite chapter 1 paragraph 2")
    system = llm.calls[-1][0]["content"]
    assert "[PRECEDING] Mira stepped onto the platform" in system
    assert "[FOLLOWING]" in system and "tea he did not drink" in system


@pytest.mark.asyncio
async def test_chat_sees_the_manuscript_not_just_the_bible(books):
    agent, _, llm = make(books, ["It works."])
    await agent.handle("s", "read chapter 1 paragraph 2")
    await agent.handle("s", "do you think this works")
    system = llm.calls[-1][0]["content"]
    assert "MANUSCRIPT AROUND THE WORK" in system and "[PRECEDING] Mira stepped onto the platform" in system


@pytest.mark.asyncio
async def test_stepping_past_the_chapter_end_says_so(books):
    agent, _, _ = make(books, [])
    await agent.handle("s", "read chapter 1 paragraph 3")
    r = await agent.handle("s", "next paragraph")
    assert "end of the book" in r.text  # demo has one chapter; must not silently repeat the last paragraph


@pytest.mark.asyncio
async def test_unsaved_proposal_dropped_by_new_command_is_announced(books):
    agent, _, _ = make(books, ["New text."])
    await agent.handle("s", "rewrite the first paragraph of chapter 1")
    r = await agent.handle("s", "go to chapter 1")
    assert "dropped the unsaved proposal" in " ".join(r.notes).lower()


def test_import_strips_gutenberg_emphasis_markers():
    chs = split_chapters("Chapter 1\n\n“Mr. Bennet, how _can_ you abuse your children?” and a_b stays.\n")
    assert chs[0][1] == ["“Mr. Bennet, how can you abuse your children?” and a_b stays."]


@pytest.mark.parametrize("said,locator", [
    ("make Mr Bennet more sarcastic where he teases his wife", "where he teases his wife"),
    ("tighten the part where Mrs Bennet says Netherfield is let", "the part where mrs bennet says netherfield is let"),
    ("in chapter 3 when they meet at the cafe make him more nervous", "when they meet at the cafe"),
    ("make it better", "make it better"),  # no location phrase: nothing to separate
])
def test_locator_query_drops_the_instruction(said, locator):
    assert intent.locator_query(said) == locator


def test_instruction_words_do_not_steer_the_paragraph_search(tmp_path):
    store = BookStore(tmp_path)
    store.create_chapter("b", 1)
    store.append_paragraph("b", 1, "He was sarcastic about everything, even the weather and the soup.")
    store.append_paragraph("b", 1, "Jonas teased his wife about the soup and she laughed at him.")
    store.append_paragraph("b", 1, "The train left at noon, and nobody on the platform waved goodbye at all.")
    kg = StoryKG(store)
    kg.refresh("b")
    s = Settings(project_dir=tmp_path, default_book="b")
    agent = Agent(s, store, FakeLLM([]), FileMemory(tmp_path / "m.json"), kg)
    sess = agent.session("s", "b")
    para, _ = agent._resolve(sess, "make it more sarcastic where he teases his wife about the soup")
    assert para.id == "ch1_s1_p2"  # the scene described, not the paragraph that merely contains the word "sarcastic"


def test_name_forms_spoken_vs_written_match(books):
    store = BookStore(books)
    kg = StoryKG(store)
    store.bible("demo").upsert("characters", "Mr. Jonas Smith", {"aliases": ["Jonny"]})
    kg.refresh("demo")
    assert "Mr. Jonas Smith" in kg.named_in("make mister jonas smith nervous")  # "mister" == "Mr."
    assert kg.named_in("where Jonny sits") == ["Mr. Jonas Smith"]  # alias from the bible front-matter


def test_kg_index_is_reused_until_a_file_changes(books):
    store = BookStore(books)
    kg = StoryKG(store)
    kg.refresh("demo")
    first = kg._tok
    kg.refresh("demo")
    assert kg._tok is first  # no rebuild on every turn
    store.append_paragraph("demo", 1, "A brand new paragraph about lanterns.")
    kg.refresh("demo")
    assert kg._tok is not first and any("lantern" in " ".join(v) for v in kg._tok.values())
