"""The capability checklist (docs/CAPABILITIES.md) as scripted scenarios: a grandfather writing a five-chapter family-history book with pictures,
researched history, PDF family documents and a family tree. The model is a script; everything else is the real app."""
import json
from pathlib import Path

import pytest

import harness
import ingest
from tests.test_pictures import PARA, FakeImageGen, make, png

FILES = Path(__file__).resolve().parents[2] / "examples" / "grandpa-lindsborg" / "family-files"  # fictional sample PDFs and a GEDCOM


async def add(ag, name):
    return await ingest.add_file("demo", name, (FILES / name).read_bytes(), ag.library, ag.llm, None, ag.journal)


@pytest.mark.asyncio
async def test_pdf_family_documents_are_read_and_found_by_meaning(books):
    ag, store, llm = make(books)
    for n in ("draft_registration_card_1917.pdf", "letter_from_france_october_1918.pdf", "obituary_everett_lund_1971.pdf", "ship_passenger_list_1884.pdf"):
        m = await add(ag, n)
        assert m["status"] == "read", (n, m)
    out = await ag.toolbox.call("search_library", {"query": "Charpentry postcard letter to mother"}, harness.Ctx(sess=ag.session("s")))
    assert "Charpentry" in out
    again = await add(ag, "draft_registration_card_1917.pdf")
    assert again.get("duplicate") and "already added" in again["message"]
    text = ag.library.text("demo", next(m["slug"] for m in ag.library.items("demo") if "draft" in m["title"].lower() or "registration" in m["title"].lower()))
    assert "Lindsborg" in text and "1896" in text


@pytest.mark.asyncio
async def test_a_family_tree_becomes_characters_places_and_a_timeline_after_a_yes(books):
    ag, store, llm = make(books)
    m = await add(ag, "lund_family.ged")
    assert m["kind"] == "family_tree" and m["status"] == "read"
    s = ag.session("s")
    out = await ag.toolbox.call("family_lookup", {"name": "Everett"}, harness.Ctx(sess=s))
    assert "1896" in out and "Lindsborg" in out
    out = await ag.toolbox.call("propose_family_import", {"name": "Everett Lund", "ancestors": 2, "descendants": 2}, harness.Ctx(sess=s))
    assert s.pending and s.pending.kind == "family"
    r = await ag.handle("s", "yes, add them")
    names = " ".join(store.bible("demo").characters())
    assert "Nils" in names and "Everett" in names and "Sigrid" in names and "added" in r.text
    # the living are protected: Dorothy (born 1950) is made-up or absent, never listed under the real name
    assert "Dorothy" not in names


@pytest.mark.asyncio
async def test_researched_history_is_looked_up_announced_and_kept_only_on_a_yes(books):
    ag, store, llm = make(books)
    # (the web tools are covered in test_web; here: a lookup result becomes a wiki note only after a yes)
    ag.session("s")
    llm.replies = ['[[tool: propose_wiki_notes {"notes": [{"kind": "world", "name": "Meuse-Argonne Offensive", "note": "26 September to 11 November 1918; the largest American operation of the war."}]}]]', "I have a note about it."]
    r = await ag.handle("s", "save the facts about the Meuse-Argonne in the wiki")
    assert r.pending and r.pending["kind"] == "wiki" and "Meuse-Argonne Offensive" not in " ".join(store.bible("demo").entries("world"))
    r = await ag.handle("s", "yes")
    assert "Meuse-Argonne Offensive" in " ".join(store.bible("demo").entries("world"))


@pytest.mark.asyncio
async def test_the_model_can_move_and_delete_a_picture_by_name_not_number(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    for t in ("Alpha is here.", "Bravo is here.", "Charlie is here."):
        store.append_paragraph("demo", 2, t)
    n = ag.images.stage("demo", png(), "pic"); ag.images.accept("demo", n)
    store.append_image("demo", 2, n, "A barn")
    s = ag.session("s"); s.chapter = 2
    out = await ag.toolbox.call("propose_move", {"chapter": 2, "picture": "true", "where": "start"}, harness.Ctx(sess=s))
    assert s.pending and s.pending.kind == "move", out
    await ag.handle("s", "yes")
    paras = store.paragraphs("demo", 2); assert paras[0].image and len(paras) == 4
    out = await ag.toolbox.call("propose_delete", {"chapter": 2, "picture": "true"}, harness.Ctx(sess=s))
    assert s.pending and s.pending.kind == "delete" and "picture" in s.pending.ask.lower() + out.lower()
    await ag.handle("s", "no")
    assert any(p.image for p in store.paragraphs("demo", 2))


@pytest.mark.asyncio
async def test_a_picture_can_be_moved_by_voice(books):
    ag, store, llm = make(books)
    n = ag.images.stage("demo", png(), "pic"); ag.images.accept("demo", n)
    store.append_image("demo", 1, n, "A barn")
    r = await ag.handle("s", "move the picture in chapter 1 to the start of the chapter")
    assert r.pending and r.pending["kind"] == "move", r.text
    await ag.handle("s", "yes")
    assert store.paragraphs("demo", 1)[0].image


@pytest.mark.asyncio
async def test_ask_me_first_sticks_and_a_chat_apology_is_never_saved_as_a_paragraph(books):
    ag, store, llm = make(books)
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir)
    await ag.handle("s", "save as you go"); assert ag.prefs.get("auto_save") == "on"
    await ag.handle("s", "ask me first from now on"); assert ag.prefs.get("auto_save") == "off"
    llm.replies = [f'[[tool: propose_append {{"text": "{PARA}"}}]]', "Here you are."]
    n0 = len(store.paragraphs("demo", 1))
    r = await ag.handle("s", "next paragraph, the walk down")
    assert r.pending and len(store.paragraphs("demo", 1)) == n0
    await ag.handle("s", "no")
    apology = ("I'm sorry, I can't single out the middle picture on my own, and I did not want to guess which one you meant, so I have left everything exactly as it was. "
               "Tell me the chapter and I will try again, and if that does not work we can delete it together. Would you like me to try that now?")
    ag.session("s").history.append({"role": "assistant", "content": apology})
    r = await ag.handle("s", "yes")
    assert len(store.paragraphs("demo", 1)) == n0 and "I'm sorry" not in store.read_chapter_raw("demo", 1)


@pytest.mark.asyncio
async def test_an_unanswered_draft_that_is_dropped_is_said_out_loud_and_can_be_brought_back(books):
    ag, store, llm = make(books)
    llm.replies = [f'[[tool: propose_append {{"text": "{PARA}"}}]]', "Here."]
    r = await ag.handle("s", "next paragraph, the walk down")
    assert r.pending
    llm.replies = ["He climbed the hill slowly and the wind was in his face and the road was long before him, with fields on either side that he had known all his life."]
    r = await ag.handle("s", "what colour was the barn?")
    r = await ag.handle("s", "delete paragraph 1 of chapter 1")
    assert "I dropped the paragraph" in r.text and any("dropped" in n for n in r.notes) or "dropped" in r.text
    await ag.handle("s", "no")
    r = await ag.handle("s", "bring it back")
    assert r.pending and r.pending["kind"] == "append" and "Here it is again" in r.text


@pytest.mark.asyncio
async def test_no_make_it_a_bigger_ship_is_about_the_story_not_the_screen(books):
    ag, store, llm = make(books)
    llm.replies = ["A bigger ship it is. Shall I make a new picture?"]
    r = await ag.handle("s", "no, make it a bigger ship")
    assert r.ui is None
    llm.replies = ["Okay."]
    r = await ag.handle("s", "yes, make one")
    assert "chapter 1 picture" not in r.text


@pytest.mark.asyncio
async def test_a_new_draft_puts_the_old_one_aside_out_loud_and_bring_it_back_swaps_them(books):
    ag, store, llm = make(books)
    A = "The first draft is about the walk down to the lake. The cold wind came in with the evening, and the road ahead of him was quiet."
    B = "The second draft is about the supper table. His mother counted out the plates by the lamp for the whole family."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "Here."]
    r = await ag.handle("s", "next paragraph, the walk down")
    assert r.pending and A in r.text
    llm.replies = [f'[[tool: propose_append {{"text": "{B}"}}]]', "And here."]
    r = await ag.handle("s", "now the part where they eat supper")
    assert B in r.text and ("put aside the paragraph" in r.text or "dropped the paragraph" in r.text) and "say no" not in r.text.lower()
    assert ag.session("s").pending.text == B
    r = await ag.handle("s", "bring it back")
    assert A in r.text or ag.session("s").pending.text == A
    assert ag.session("s").pending.text == A and ag.session("s").last_dropped.text == B
    await ag.handle("s", "yes")
    assert A in store.read_chapter_raw("demo", 1) and B not in store.read_chapter_raw("demo", 1)


@pytest.mark.asyncio
async def test_put_a_picture_of_the_barn_in_chapter_2_after_paragraph_1_makes_a_picture(books):
    import intent
    assert intent.move_request("put a picture of the barn in chapter 2 after paragraph 1") is None
    assert intent.picture_request("put a picture of the barn in chapter 2 after paragraph 1")
    assert intent.move_request("move the picture in chapter 2 after paragraph 1") is not None
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    for t in ("Alpha is here.", "Bravo is here."):
        store.append_paragraph("demo", 2, t)
    llm.replies = ['[[tool: make_picture {"prompt": "a red barn", "caption": "The barn", "chapter": 2}]]', "A red barn."]
    r = await ag.handle("s", "put a picture of the barn in chapter 2 after paragraph 1")
    assert r.pending and r.pending["kind"] == "picture" and "after paragraph 1" in r.pending["ask"]


def test_take_that_part_out_is_an_edit_not_new_writing():
    import intent
    for t in ("take that part out", "take the clerk part out", "cut that out", "leave out the clerk"):
        assert intent.is_edit(t), t
    assert not intent.is_draft("take that part out")


@pytest.mark.asyncio
async def test_a_picture_keeps_what_was_said_about_it_so_it_can_be_described_again(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a red barn in a wheat field", "caption": "The barn"}]]', "A red barn stands alone in a golden wheat field under a pale sky."]
    await ag.handle("s", "make me a picture of the barn")
    await ag.handle("s", "yes")
    ag.session("s").work = 1
    r = await ag.handle("s", "describe the barn picture")
    assert "red barn" in r.text and "golden wheat" in r.text and not store.paragraphs("demo", 1)[-1].text.startswith("It is")


@pytest.mark.asyncio
async def test_goodbye_never_gets_an_empty_answer(books):
    ag, store, llm = make(books)
    llm.replies = ["   "]
    r = await ag.handle("s", "goodbye")
    assert "Goodbye" in r.text


@pytest.mark.asyncio
async def test_bring_it_back_reads_the_paragraph_again(books):
    ag, store, llm = make(books)
    A = "The first draft is about the walk down to the lake. The cold wind came in with the evening, and the road ahead of him was quiet."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "Here."]
    await ag.handle("s", "next paragraph, the walk down")
    llm.replies = ["Okay."]
    await ag.handle("s", "delete paragraph 1 of chapter 1")
    await ag.handle("s", "no")
    r = await ag.handle("s", "bring it back")
    assert A in r.text


def test_the_last_sentence_of_the_first_paragraph_is_paragraph_one():
    import intent
    assert intent.parse_ref("take the last sentence of the first paragraph out").para == 1
    assert intent.parse_ref("change the last paragraph").para == -1
    assert intent.parse_ref("read the first line of chapter 2").para == 1 and intent.parse_ref("make the first sentence of paragraph 3 shorter").para == 3


@pytest.mark.asyncio
async def test_hello_while_a_draft_waits_says_which_draft_it_is(books):
    ag, store, llm = make(books)
    A = "The first draft is about the walk down to the lake. The cold wind came in with the evening, and the road ahead of him was quiet."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "Here."]
    await ag.handle("s", "next paragraph, the walk down")
    llm.replies = ["Hello there."]
    r = await ag.handle("s", "hello")
    assert "It begins: The first draft is about the walk" in r.text


@pytest.mark.asyncio
async def test_a_wordy_wikipedia_query_is_retried_with_shorter_words():
    import httpx
    import web as webmod
    calls = []

    def handler(req):
        q = dict(req.url.params)
        calls.append(q.get("search") or q.get("titles"))
        if q.get("action") == "opensearch":
            return httpx.Response(200, json=[q["search"], ["Spanish flu"] if q["search"] == "1918 flu" else [], [], []])
        return httpx.Response(200, json={"query": {"pages": {"1": {"title": "Spanish flu", "extract": "The 1918 influenza pandemic killed millions."}}}})

    w = webmod.Web(wikipedia_url="https://wiki.test", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    got = await w.wikipedia("1918 flu in the army camps")
    assert got and got["title"] == "Spanish flu" and len(calls) >= 2


@pytest.mark.asyncio
async def test_a_paragraph_in_the_wrong_person_is_sent_back_once_and_long_drafts_are_read_from_the_start(books):
    ag, store, llm = make(books)
    store.bible("demo").write("premise", "A family history told in the third person by a storyteller who never appears.") if hasattr(store.bible("demo"), "write") else None
    first = "I remember my great-uncle standing in the field. I could see the wheat and I loved it."
    third = ("He stood in the field at the edge of the farm. The wheat ran to the road in gold waves. His father called him in for supper, twice. "
             "The lamp was lit in the kitchen before he came. He ate quickly and said nothing about the day.")
    llm.replies = [f'[[tool: propose_append {{"text": "{first} The sun went down behind the barn."}}]]', f'[[tool: propose_append {{"text": "{third}"}}]]', "Here."]
    r = await ag.handle("s", "next paragraph, the field")
    assert r.pending and "gold waves" in r.text and "I remember" not in r.text
    spoken = " ".join(x["text"] for x in r.speech)
    assert "only the start" in spoken and "ate quickly" not in spoken
    r = await ag.handle("s", "read it all")
    assert "ate quickly" in " ".join(x["text"] for x in r.speech) and r.pending


@pytest.mark.asyncio
async def test_take_the_last_sentence_out_edits_in_place_with_no_model_and_is_undoable(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    store.append_paragraph("demo", 2, "He registered on the fifth of June. He was twenty years old. The war had his name.")
    store.append_paragraph("demo", 2, "A lottery in Washington set the order. Farm work was a reason to ask for a deferment.")
    ag.session("s").chapter = 5 if False else 1
    n0 = len(store.paragraphs("demo", 2))
    r = await ag.handle("s", "take the last sentence of the first paragraph out of chapter 2")
    assert r.pending and r.pending["kind"] == "edit" and "The war had his name." in r.text and not llm.calls
    await ag.handle("s", "yes")
    paras = store.paragraphs("demo", 2)
    assert len(paras) == n0 and paras[0].text == "He registered on the fifth of June. He was twenty years old." and paras[1].text.startswith("A lottery")
    await ag.handle("s", "undo that")
    assert store.paragraphs("demo", 2)[0].text.endswith("The war had his name.")
    r = await ag.handle("s", "no, just take the first sentence of paragraph 2 out for me")
    assert r.pending and "A lottery in Washington set the order." in r.text
    await ag.handle("s", "no")
    assert len(store.paragraphs("demo", 2)) == n0


@pytest.mark.asyncio
async def test_hello_while_a_draft_waits_is_answered_without_the_model_and_no_is_not_noise(books):
    import intent
    assert not intent.is_noise("no") and intent.is_noise("you")
    ag, store, llm = make(books)
    A = "The first draft is about the walk down to the lake. The cold wind came in with the evening, and the road ahead of him was quiet."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "Here."]
    await ag.handle("s", "next paragraph, the walk down")
    calls = len(llm.calls)
    r = await ag.handle("s", "hello")
    assert "I'm here." in r.text and "The first draft is about the walk" in r.text and len(llm.calls) == calls


@pytest.mark.asyncio
async def test_no_make_it_a_bigger_ship_after_a_picture_is_about_the_picture(books):
    ag, store, llm = make(books)
    ag.session("s").history.append({"role": "assistant", "content": "I made a picture of a troopship. Would you like a different picture?"})
    llm.replies = ['[[tool: make_picture {"prompt": "a very large troopship", "caption": "A bigger troopship"}]]', "A much bigger ship, crowded with men."]
    r = await ag.handle("s", "no, make it a bigger ship")
    assert r.pending and r.pending["kind"] == "picture" and r.ui is None


@pytest.mark.asyncio
async def test_the_model_can_replace_a_saved_paragraph_in_place(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "The old paragraph is here and it will be replaced soon enough.")
    s = ag.session("s")
    out = await ag.toolbox.call("propose_edit_paragraph", {"chapter": 2, "paragraph": 1, "text": "The new paragraph is here, and it replaces the old one entirely."}, harness.Ctx(sess=s))
    assert s.pending and s.pending.kind == "edit", out
    await ag.handle("s", "yes")
    paras = store.paragraphs("demo", 2)
    assert len(paras) == 1 and paras[0].text.startswith("The new paragraph")


@pytest.mark.asyncio
async def test_a_pending_picture_can_be_sent_to_another_chapter_and_place_in_one_answer(books):
    ag, store, llm = make(books)
    for n in (2, 3):
        store.create_chapter("demo", n)
        for t in ("Alpha here for the book.", "Bravo here for the book."):
            store.append_paragraph("demo", n, t)
    llm.replies = ['[[tool: make_picture {"prompt": "a troopship", "caption": "A troopship", "chapter": 2}]]', "A big grey troopship at the dock."]
    r = await ag.handle("s", "make me a picture of a troopship")
    assert r.pending["kind"] == "picture" and "chapter 2" in r.pending["ask"]
    r = await ag.handle("s", "no, put it in chapter 3 after paragraph 1")
    assert "chapter 3" in r.text and "after paragraph 1" in r.text and r.pending["kind"] == "picture"
    await ag.handle("s", "yes")
    paras = store.paragraphs("demo", 3)
    assert paras[1].image and len(paras) == 3 and not any(p.image for p in store.paragraphs("demo", 2))


@pytest.mark.asyncio
async def test_a_new_picture_goes_to_the_chapter_the_writer_turned_to(books):
    ag, store, llm = make(books)
    for n in (2, 3):
        store.create_chapter("demo", n)
        store.append_paragraph("demo", n, "Alpha here for the book.")
    ag.session("s").work = 2
    llm.replies = ['[[tool: make_picture {"prompt": "a station", "caption": "A station"}]]', "A station."]
    r = await ag.handle("s", "make me a picture of a station", viewing=3)
    assert r.pending["kind"] == "picture" and "chapter 3" in r.pending["ask"]
    ag.session("s").pending = None
    ag.session("s").screen_wins = False
    ag.session("s").work = 2; ag.session("s").chapter = 2
    llm.replies = ['[[tool: make_picture {"prompt": "a station", "caption": "A station"}]]', "A station."]
    r = await ag.handle("s", "make me a picture of a station", viewing=3)  # the page still shows chapter 3: that is where the picture goes
    assert r.pending and "chapter 3" in r.pending["ask"]


def test_a_remembered_description_drops_old_placement_talk(books):
    import images as imgs
    st = imgs.ImageStore(__import__("book_store").BookStore(books))
    st.note("demo", "x.jpg", "A grey ship at a dock. I placed it in chapter 5 after the first paragraph. Should I put it in?")
    assert st.note("demo", "x.jpg") == "A grey ship at a dock."


@pytest.mark.asyncio
async def test_the_chapter_on_screen_beats_the_chapter_the_model_guessed_for_a_picture(books):
    ag, store, llm = make(books)
    for n in (2, 3):
        store.create_chapter("demo", n); store.append_paragraph("demo", n, "Alpha here for the book.")
    llm.replies = ['[[tool: make_picture {"prompt": "a station", "caption": "A station", "chapter": 2}]]', "A station."]
    r = await ag.handle("s", "make me a picture of a station", viewing=3)
    assert "chapter 3" in r.pending["ask"]


@pytest.mark.asyncio
async def test_ask_me_first_with_a_draft_waiting_changes_the_setting_and_keeps_the_draft(books):
    ag, store, llm = make(books)
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir)
    await ag.handle("s", "save as you go")
    A = "The first draft is about the walk down to the lake. The cold wind came in with the evening, and the road ahead of him was quiet."
    ag.session("s").pending = __import__("agent").Pending("append", A, 1)
    r = await ag.handle("s", "ask me first from now on")
    assert ag.prefs.get("auto_save") == "off" and ag.session("s").pending is not None and "still waiting" in r.text


@pytest.mark.asyncio
async def test_no_put_it_first_moves_a_waiting_paragraph_without_dropping_it(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    for t in ("Alpha is here for the book.", "Bravo is here for the book."):
        store.append_paragraph("demo", 2, t)
    NEW = "The new paragraph is about the supper table. His mother counted out the plates by the lamp for the whole family."
    llm.replies = [f'[[tool: propose_append {{"text": "{NEW}", "chapter": 2}}]]', "Here."]
    r = await ag.handle("s", "next paragraph for chapter 2, the supper table")
    assert r.pending
    r = await ag.handle("s", "no, put it first")
    assert "very start of chapter 2" in r.text and r.pending is not None
    await ag.handle("s", "yes")
    assert store.paragraphs("demo", 2)[0].text.startswith("The new paragraph") and len(store.paragraphs("demo", 2)) == 3


@pytest.mark.asyncio
async def test_describe_it_with_a_text_draft_waiting_does_not_describe_some_other_picture(books):
    ag, store, llm = make(books)
    n = ag.images.stage("demo", png(), "pic"); ag.images.accept("demo", n); store.append_image("demo", 1, n, "A barn")
    ag.session("s").pending = __import__("agent").Pending("append", "A draft paragraph about the supper table and the lamp, for the book.", 1)
    r = await ag.handle("s", "describe it")
    assert "paragraph waiting" in r.text and "barn" not in r.text.lower()


def test_story_notes_wording_is_about_the_wiki_not_a_new_paragraph():
    import intent
    for t in ("write story notes for three characters including Erik Holmberg", "make wiki notes for three new characters", "describe the three characters in my story notes"):
        assert not intent.is_draft(t) and not intent.is_edit(t), t
    assert intent.is_draft("write the part where he reaches the station")


@pytest.mark.asyncio
async def test_a_change_to_the_waiting_draft_revises_it_instead_of_guessing_an_old_paragraph(books):
    ag, store, llm = make(books)
    A = "She lived there all her life. The farm was quiet in the evenings, and the lamp was lit before the men came in from the field."
    ag.session("s").pending = __import__("agent").Pending("append", A, 1)
    llm.replies = ["The farm was quiet in the evenings, and the lamp was lit before the men came in from the field."]
    r = await ag.handle("s", "leave out the part about her living there all her life")
    assert r.pending and "She lived there" not in ag.session("s").pending.text and "Do you mean" not in r.text


@pytest.mark.asyncio
async def test_a_kept_page_is_not_offered_again(books):
    import web as webmod
    ag, store, llm = make(books)
    ag.web = webmod.Web(wikipedia_url="https://wiki.test")
    ag.library.add("demo", "m.txt", b"x", "Meuse-Argonne Offensive", "document", "Text about it.")
    s = ag.session("s"); s.web_hits["last"] = {"title": "Meuse-Argonne Offensive", "url": "https://w/x", "text": "Text about it."}
    out = await ag.toolbox.call("propose_save_page", {}, harness.Ctx(sess=s))
    assert "already kept" in out and s.pending is None


@pytest.mark.asyncio
async def test_a_paragraph_that_re_says_the_book_is_sent_back_once_by_plain_word_overlap(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    old = "He returned to the farm in the spring of 1919. He worked the same land near Lindsborg for more than fifty years, never talking about the war."
    store.append_paragraph("demo", 2, old)
    again = "In the spring of 1919 he returned to the farm. He worked the same land near Lindsborg for more than fifty years, never talking about the war to anyone."
    assert "repeats much of" in ag.check_repeat(ag.session("s"), again)
    assert ag.check_repeat(ag.session("s"), "A letter came from France in October, and his mother read it twice at the kitchen table before she folded it into her apron pocket.") == ""
    llm.replies = [f'[[tool: propose_append {{"text": "{again}", "chapter": 2}}]]', "Hmm.", f'[[tool: propose_append {{"text": "{again}", "chapter": 2}}]]', "Here."]
    r = await ag.handle("s", "next paragraph, he came home")
    assert "Problem: This repeats" in " ".join(str(m) for c in llm.calls for m in c)


@pytest.mark.asyncio
async def test_details_nobody_gave_are_flagged_and_never_saved_behind_the_writers_back(books):
    ag, store, llm = make(books)
    await add(ag, "draft_registration_card_1917.pdf")
    ag.prefs = __import__("prefs").Prefs(ag.s.state_dir)
    await ag.handle("s", "save as you go")
    ag.session("s").history.append({"role": "user", "content": "Everett was a farmer near Lindsborg in 1917."})
    grounded = "In June 1917 Everett registered for the draft at Lindsborg and gave his trade as a farmer, as the card says. He was twenty years old and single."
    assert ag.ungrounded(ag.session("s"), grounded) == ""
    inv = "In 1923 Everett bought a blue tractor from Mr Halvorsen in Wichita. He paid 400 dollars for it and drove it home through the rain."
    out = ag.ungrounded(ag.session("s"), inv)
    assert "1923" in out and "Halvorsen" in out
    llm.replies = [f'[[tool: propose_append {{"text": "{inv}"}}]]', f'[[tool: propose_append {{"text": "{inv}"}}]]', "Here."]  # sent back once, then shown with its warning
    r = await ag.handle("s", "next paragraph, the tractor")
    assert r.pending and "could not find" in r.text and "tractor" in r.pending["text"]  # not auto-saved


@pytest.mark.asyncio
async def test_a_picture_is_not_sent_to_a_chapter_that_does_not_exist_yet(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a troopship", "caption": "A troopship"}]]', "A grey ship."]
    await ag.handle("s", "make me a picture of a troopship")
    work = ag.session("s").work
    r = await ag.handle("s", "no, put it in chapter 9 after paragraph 1")
    assert "does not exist yet" in r.text and ag.session("s").pending is not None and ag.session("s").work == work
    assert not any(9 == n for n in store.chapter_numbers("demo"))


def test_the_same_story_in_other_words_is_called_a_repeat(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    store.append_paragraph("demo", 2, "Elsa Berglund was born in Lindsborg in September, and she grew up on a farm there before marrying Everett in the summer after the war ended.")
    again = "Born in Lindsborg in September, Elsa Berglund grew up on a farm there and married Everett in the summer after the war had ended, a few years later."
    assert ag.check_repeat(ag.session("s"), again)


@pytest.mark.asyncio
async def test_sentences_the_documents_never_mention_are_pointed_out(books):
    ag, store, llm = make(books)
    await add(ag, "draft_registration_card_1917.pdf"); await add(ag, "letter_from_france_october_1918.pdf")
    ag.session("s").history.append({"role": "user", "content": "Everett was a farmer near Lindsborg in 1917."})
    good = "Everett registered for the draft at Lindsborg and gave his trade as a farmer. He wrote to his mother about the mud and the flu."
    assert ag.ungrounded(ag.session("s"), good) == ""
    flowery = "The village shimmered beneath wandering clouds while merchants hurried between cobbled courtyards carrying baskets overflowing with ripe peaches. Every cathedral bell echoed gloriously across the sleepy valley."
    out = ag.ungrounded(ag.session("s"), flowery)
    assert "not in your documents" in out


@pytest.mark.asyncio
async def test_a_revised_draft_made_while_another_waits_is_what_yes_saves(books):
    ag, store, llm = make(books)
    A = "The first draft is about the old photograph. It is stiff and brown, and the young man in it is trying not to smile for the camera."
    B = "The second draft is about the photograph too. He stands in a studio in his uniform and hat, and he is trying hard not to smile."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "Here."]
    await ag.handle("s", "next paragraph, the photograph")
    llm.replies = [f'[[tool: propose_append {{"text": "{B}"}}]]', "A shorter one."]
    r = await ag.handle("s", "where does the brown colour come from? only use what is in my papers, make it short")
    assert ag.session("s").pending is not None and ag.session("s").pending.text == B
    await ag.handle("s", "yes")
    raw = store.read_chapter_raw("demo", 1)
    assert B in raw and A not in raw


@pytest.mark.asyncio
async def test_describe_the_picture_in_a_chapter_with_two_pictures_says_both(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "Alpha is here for the book.")
    for cap in ("A railway station", "A wheat field"):
        n = ag.images.stage("demo", png(color=(len(cap), 50, 90)), cap); ag.images.accept("demo", n); store.append_image("demo", 2, n, cap)
    r = await ag.handle("s", "describe the picture in chapter 2 to me")
    assert "2 pictures" in r.text and "railway station" in r.text and "wheat field" in r.text
