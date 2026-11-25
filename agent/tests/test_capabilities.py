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
    store.bible("demo").write_doc("premise", "A family history told in the third person by a storyteller who never appears.")
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


# ---- the audit: confirmed bugs, each with its repro ------------------------------------------------------------------------------------
def test_audit_ranges_with_hyphens_and_huge_numbers_are_safe():
    import time
    import intent
    assert intent.delete_many("delete paragraphs 2-4")[1] == [2, 3, 4] and intent.delete_many("delete paragraphs 2 - 4")[1] == [2, 3, 4]
    t = time.time(); intent.delete_many("delete paragraph " + "1" * 40 + "x"); assert time.time() - t < 0.5


def test_audit_sentences_keep_quotes_and_do_not_split_on_abbreviations():
    from agent import split_sentences
    s = split_sentences('He said "Go." Then she left. Mr. Bennet sat down. She was born on 1 Jan. in St. Paul. The end.')
    assert s[0] == 'He said "Go."' and s[1] == "Then she left." and s[2] == "Mr. Bennet sat down." and s[3].startswith("She was born on 1 Jan.") and s[-1] == "The end." and len(s) == 5


def test_audit_picture_words_and_story_talk():
    import intent
    assert intent.parse_ref("delete the picture after paragraph 2").picture and intent.parse_ref("remove the picture before paragraph 3").picture
    r = intent.parse_ref("delete the paragraph before the picture"); assert not r.picture and r.para is None
    assert intent.picture_position("make me a picture of the church at the top of the hill", strict=True) is None
    assert intent.picture_position("make a picture of the sailor at the start of his journey", strict=True) is None
    assert intent.picture_position("put a picture of a lake at the very start of chapter 1", strict=True) == 0
    assert not intent.placement_command("In the end she forgave him") and intent.placement_command("put it first")
    assert intent.sentence_removal("take the last sentence of the letter and make it the title") is None
    assert intent.sentence_removal("take the last sentence of chapter 1 out")[0].para is None


def test_audit_screen_words_in_story_talk_are_left_alone():
    import uisettings
    D = uisettings.DEFAULT
    for t in ("simplify paragraph 2", "make him speak slower", "he was bold", "rewrite paragraph 2, the words are too small", "the wall is too small", "he looked too busy", "make the letter bigger for her"):
        assert uisettings.parse(t, D) is None, t
    assert uisettings.parse("make the buttons bigger", D) is not None and uisettings.parse("simplify the screen", D) is not None


@pytest.mark.asyncio
async def test_audit_a_stale_dropped_draft_is_not_brought_back_over_an_undo(books):
    ag, store, llm = make(books)
    A = "The first draft is about the sea. The waves rolled in grey and slow, and the gulls wheeled over the empty harbour all morning."
    B = "The second draft is about the sky. The clouds piled up over the hills, and the light went yellow before the rain came down."
    llm.replies = [f'[[tool: propose_append {{"text": "{A}"}}]]', "A."]
    await ag.handle("s", "write a paragraph about the sea")
    llm.replies = [f'[[tool: propose_append {{"text": "{B}"}}]]', "B."]
    await ag.handle("s", "write another paragraph about the sky")
    await ag.handle("s", "yes")
    assert B in store.read_chapter_raw("demo", 1)
    r = await ag.handle("s", "bring it back")  # B was just saved: this is an undo, not a revival of the dropped A
    assert A not in store.read_chapter_raw("demo", 1) and (ag.session("s").pending is None or ag.session("s").pending.text != A)


@pytest.mark.asyncio
async def test_audit_hold_greeting_and_read_after_an_agreeing_word_do_not_rewrite_the_draft(books):
    ag, store, llm = make(books)
    A = "The first draft is about the sea. The waves rolled in grey and slow, and the gulls wheeled over the empty harbour all morning."
    ag.session("s").pending = __import__("agent").Pending("append", A, 1)
    n = len(llm.calls)
    for t in ("okay hold on", "good night", "good morning", "yes please read it", "sure, let me think about it", "okay read it to me"):
        r = await ag.handle("s", t)
        assert ag.session("s").pending is not None and ag.session("s").pending.text == A, t
    assert len(llm.calls) == n


@pytest.mark.asyncio
async def test_audit_yes_put_it_first_places_the_paragraph_and_a_remark_is_not_a_placement(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    for t in ("Alpha is here for the book.", "Bravo is here for the book."):
        store.append_paragraph("demo", 2, t)
    NEW = "The new paragraph is about the supper table. His mother counted out the plates by the lamp for the whole family."
    ag.session("s").pending = __import__("agent").Pending("append", NEW, 2)
    r = await ag.handle("s", "yes, put it first")
    assert store.paragraphs("demo", 2)[0].text == NEW and not llm.calls
    ag.session("s").pending = __import__("agent").Pending("append", NEW, 2)
    llm.replies = ["Noted."]
    await ag.handle("s", "In the end she forgave him")
    assert ag.session("s").pending is not None and (ag.session("s").pending.data or {}).get("where") is None


@pytest.mark.asyncio
async def test_audit_a_command_about_the_waiting_draft_shortens_that_draft(books):
    ag, store, llm = make(books)
    store.append_paragraph("demo", 1, "A saved paragraph with two sentences. The second one is here.")
    before = store.paragraphs("demo", 1)[-1].text
    D = "The draft starts here. It has a middle part. It ends with this last sentence."
    ag.session("s").pending = __import__("agent").Pending("append", D, 1)
    r = await ag.handle("s", "take the last sentence out")
    assert ag.session("s").pending.text == "The draft starts here. It has a middle part." and store.paragraphs("demo", 1)[-1].text == before


@pytest.mark.asyncio
async def test_audit_a_picture_request_does_not_move_the_writing_chapter(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "Alpha here for the book.")
    ag.session("s").work = 2
    llm.replies = ['[[tool: make_picture {"prompt": "a lake", "caption": "A lake"}]]', "A lake."]
    await ag.handle("s", "make me a picture of a lake", viewing=1)
    await ag.handle("s", "no")
    assert ag.session("s").work == 2


def test_audit_voice_repeat_and_grounding_false_positives(books):
    ag, store, llm = make(books)
    store.bible("demo").write_doc("premise", "A story told in the third person.")
    sess = ag.session("s")
    for t in ("Henry I ruled for many years and was feared.", "Part I begins with a storm over the sea.", "'I will go,' she said, and went out into the cold.",
              "She thought: I must go. The road was long and the night was cold around her."):
        assert ag.check_voice(sess, t) == "", t
    assert ag.check_voice(sess, "I walked down to the lake and I saw the old boat there that morning.") != ""
    store.create_chapter("demo", 2); store.append_paragraph("demo", 2, "It was raining again.")
    assert ag.check_repeat(sess, "At noon the carter came by the gate with the post, and he said it was raining again over the hills and the whole valley road.") == ""
    assert ag.ungrounded(sess, "On Tuesday in October a man called Bergman arrived in Lisbon with 12 trunks and a very long story to tell about the sea.") == ""  # no documents: a novel is not gated


# ------------------------------------------------------------------------------------------- second logic audit (regressions)
DRAFT = "The wind came off the prairie in long cold gusts. Everett pulled his collar up and walked the last mile to the station with his bag."


def _wait(ag, kind="append", text=DRAFT, chapter=1, **kw):
    ag.session("s").pending = __import__("agent").Pending(kind, text, chapter, **kw)


@pytest.mark.asyncio
async def test_audit2_keep_it_or_leave_it_refuses_a_delete_or_move(books):
    ag, store, llm = make(books)
    before = store.read_chapter_raw("demo", 1)
    for cmd, ans in (("delete paragraph 2", "keep it"), ("delete paragraph 2", "fine, leave it"), ("delete paragraph 2", "right, actually leave it"),
                     ("move paragraph 1 to the end", "keep it where it is"), ("move paragraph 1 to the end", "keep it there")):
        await ag.handle("s", cmd)
        assert ag.session("s").pending is not None, cmd
        await ag.handle("s", ans)
        assert store.read_chapter_raw("demo", 1) == before and ag.session("s").pending is None, ans


@pytest.mark.asyncio
async def test_audit2_a_yes_never_saves_a_list_of_ideas_or_a_picture_offer(books):
    ag, store, llm = make(books)
    n = len(store.paragraphs("demo", 1))
    s = ag.session("s")
    for offer in ("Here are some ideas for the next part. One, Everett finds a letter in the barn from his father. Two, a storm closes the road to town. "
                  "Three, his sister arrives from Kansas City with news about the farm and the bank. Shall I write the first one?",
                  "I could make a picture of the station at night, with the lamps lit and the snow coming down over the tracks and the empty platform, "
                  "and the old clock above the door showing ten past eleven. Shall I make that picture for chapter 1?"):
        s.history.append({"role": "assistant", "content": offer})
        llm.replies = ["Okay."]
        await ag.handle("s", "yes please")
        assert len(store.paragraphs("demo", 1)) == n


@pytest.mark.asyncio
async def test_audit2_a_question_after_okay_or_a_cut_off_yes_but_saves_nothing(books):
    ag, store, llm = make(books)
    n = len(store.paragraphs("demo", 1))
    for t in ("okay, who was waiting at the station?", "good, what happens next?", "alright, where were we?", "yes, but", "yes but um"):
        _wait(ag)
        llm.replies = ["Nobody was waiting."]
        await ag.handle("s", t)
        assert len(store.paragraphs("demo", 1)) == n and ag.session("s").pending is not None and ag.session("s").pending.text == DRAFT, t
    await ag.handle("s", "um yes")
    assert len(store.paragraphs("demo", 1)) == n + 1


@pytest.mark.asyncio
async def test_audit2_naming_another_chapter_in_the_answer_puts_the_draft_there(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    _wait(ag)
    await ag.handle("s", "yes, add it to chapter 2")
    assert DRAFT in store.read_chapter_raw("demo", 2) and DRAFT not in store.read_chapter_raw("demo", 1)
    _wait(ag, text=DRAFT.replace("Everett", "Nils"))
    r = await ag.handle("s", "no, put it in chapter 2")
    assert "chapter 2" in r.text and ag.session("s").pending.chapter == 2
    _wait(ag)
    r = await ag.handle("s", "put it in chapter 9")
    assert "no chapter 9" in r.text and ag.session("s").pending.chapter == 1


@pytest.mark.asyncio
async def test_audit2_correcting_which_paragraph_keeps_the_change_and_never_hits_the_rejected_one(books):
    ag, store, llm = make(books)
    p2 = store.paragraphs("demo", 1)[1].id
    for ans, want in (("no, I meant paragraph three", 2), ("no, the paragraph before that", 0), ("no, the one after that", 2), ("no, not that one, the last one", 2)):
        _wait(ag, "confirm_target", "make the bit about the apology sadder", paragraph_id=p2)
        llm.replies = ["A sadder paragraph than before, which is long enough to be a real paragraph of the story."]
        await ag.handle("s", ans)
        s = ag.session("s")
        assert s.pending and s.pending.kind == "edit" and s.pending.paragraph_id == store.paragraphs("demo", 1)[want].id, ans
    _wait(ag, "edit", "A rewritten paragraph two, long enough to look like a real paragraph of the story.", paragraph_id=p2, data={"instruction": "make it sadder"})
    llm.replies = ["A sadder paragraph three, long enough to look like a real paragraph of the story here."]
    await ag.handle("s", "no, I meant paragraph three")
    assert ag.session("s").pending.paragraph_id == store.paragraphs("demo", 1)[2].id and "make it sadder" in llm.calls[-1][-1]["content"]
    _wait(ag, "confirm_target", "make the bit about the apology sadder", paragraph_id=p2)
    llm.replies = ["A sadder and shorter paragraph two, long enough to look like a real paragraph here."]
    await ag.handle("s", "yes but make it shorter")
    assert ag.session("s").pending.kind == "edit" and ag.session("s").pending.paragraph_id == p2 and "shorter" in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_audit2_yes_and_write_the_next_one_saves_then_drafts(books):
    ag, store, llm = make(books)
    drafts = ["Snow had settled on the fence posts overnight, and the cows stood close together by the barn door waiting for hay.",
              "His mother kept the letters in a tin box under the bed, tied with blue string, and read them only on Sundays.",
              "The harvest that year was thin, so the boys went to work for the railroad laying ties west of Salina all summer.",
              "At the church supper the pastor spoke about the war in Europe, and the room grew quiet as the coffee went cold."]
    for t, d in zip(("yes save it and write the next one", "keep going", "great, carry on", "yes, and then the part where he buys a ticket"), drafts):
        n = len(store.paragraphs("demo", 1))
        _wait(ag, text=d + f" ({t})")
        llm.replies = ["Inside the station the air was warm and smelled of coal smoke, and the clerk nodded to him from behind the ledger."] * 4
        r = await ag.handle("s", t)
        assert len(store.paragraphs("demo", 1)) == n + 1 and f"({t})" in store.read_chapter_raw("demo", 1), t
        assert ag.session("s").pending and ag.session("s").pending.kind == "append" and "coal smoke" in ag.session("s").pending.text, t


def test_audit2_starting_a_chapter_does_not_take_words_for_a_title():
    import intent
    for t in ("Start chapter two.", "start chapter two please", "start a new chapter now please", "let's start chapter two where they arrive in Kansas"):
        assert intent.new_chapter_titled(t)[1] == "", t
    assert intent.new_chapter_titled("Start chapter two.")[0] == 2
    assert intent.new_chapter_titled("start chapter 3 and call it The Lake") == (3, "The Lake")
    assert intent.new_chapter_titled("begin chapter four, The Round Table") == (4, "The Round Table")
    assert intent.chapter_story("let's start chapter two where they arrive in Kansas") == "where they arrive in Kansas"


def test_audit2_story_told_with_edit_verbs_is_not_an_edit_and_facts_are_not_idea_requests():
    import intent
    for t in ("In this chapter they fix up the old farmhouse and make it a home.", "After the wedding they had to change their plans and move to Topeka.",
              "After the funeral Mira had to make dinner for twelve people."):
        assert not intent.is_edit(t), t
    for t in ("in chapter 3 when they meet make him nervous", "when Everett leaves make it sadder", "In the station scene, make the porter older"):
        assert intent.is_edit(t), t
    for t in ("I need the name of Everett's father", "are there any problems with the timeline", "can you give me the details from the obituary"):
        assert intent.ideas_request(t) is None, t
    assert intent.ideas_request("give me some names for a girl")["kind"] == "name"


def test_audit2_second_to_last_corrections_and_fillers():
    import intent
    assert intent.parse_ref("delete the second to last paragraph").para == -2
    assert intent.parse_ref("read me the next to last paragraph").para == -2
    r = intent.parse_ref("delete paragraph two in chapter one, no wait, chapter two")
    assert (r.chapter, r.para) == (2, 2)
    assert intent.goto_chapter("uh, go to chapter two") == 2 and intent.undo_request("um, undo that")
    assert intent.confirmation("well yes") == ("yes", "") and intent.confirmation("no, that's fine") == ("no", "")
    assert intent.detect_mode("okay, back to the story. Everett boards the train and finds a seat by the window") is None


@pytest.mark.asyncio
async def test_audit2_read_me_a_chapter_reads_it_and_keeps_the_writing_chapter(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    await ag.handle("s", "go to chapter 2")
    r = await ag.handle("s", "read me chapter one")
    assert store.paragraphs("demo", 1)[0].text[:20] in r.text and ag.session("s").work == 2


# ---------------------------------------------------------------------------------------------- grandpa round 9 (regressions)
@pytest.mark.asyncio
async def test_r9_yes_then_a_second_sentence_saves_and_then_does_the_rest(books):
    ag, store, llm = make(books)
    n = len(store.paragraphs("demo", 1))
    _wait(ag)
    llm.replies = ["Good, 1918 it is."]
    r = await ag.handle("s", "Yes, add it. We'll just say nineteen eighteen.")
    assert len(store.paragraphs("demo", 1)) == n + 1 and DRAFT in store.read_chapter_raw("demo", 1) and "nineteen eighteen" in llm.calls[-1][-1]["content"]
    _wait(ag, text=DRAFT.replace("Everett", "Nils"))
    llm.replies = ["A shorter version of the paragraph about Nils walking to the station with his bag in the cold wind."]
    await ag.handle("s", "Yes. But make it shorter.")  # one thing: a change to the waiting draft
    assert len(store.paragraphs("demo", 1)) == n + 1 and ag.session("s").pending is not None
    ag.session("s").pending = None
    await ag.handle("s", "delete paragraph 2")
    llm.replies = ["Changed."]
    r = await ag.handle("s", "No! Don't take it out. Name chapter one The Station.")
    assert len(store.paragraphs("demo", 1)) == n + 1 and store.chapter_title("demo", 1) == "The Station" and "Nothing was deleted" not in r.text or "left it" in r.text


@pytest.mark.asyncio
async def test_r9_no_i_meant_after_a_change_that_found_nothing_retargets_the_change_not_an_old_delete(books):
    ag, store, llm = make(books)
    await ag.handle("s", "delete paragraph 2")
    await ag.handle("s", "no, wait, keep it")
    llm.replies = ["NOT_FOUND: the coat is not in this paragraph."]
    r = await ag.handle("s", "Change paragraph two so that his father wears his Sunday coat.")
    assert "did not change" in r.text
    llm.replies = ["At the café across the street his father sat in his Sunday coat, and Jonas ordered tea he did not drink."]
    r = await ag.handle("s", "Oh. No, I meant paragraph three.")
    s = ag.session("s")
    assert s.pending and s.pending.kind == "edit" and s.pending.paragraph_id == store.paragraphs("demo", 1)[2].id and "Sunday coat" in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_r9_navigation_never_moves_a_waiting_draft_and_never_mind_changes_nothing(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    _wait(ag, chapter=2)
    await ag.handle("s", "Um, go to chapter one.")
    assert ag.session("s").pending.chapter == 2
    ag.session("s").pending = None
    before = store.read_chapter_raw("demo", 1)
    n = len(llm.calls)
    r = await ag.handle("s", "Never mind, cancel that.")
    assert len(llm.calls) == n and ag.session("s").pending is None and store.read_chapter_raw("demo", 1) == before and "undo" in r.text


@pytest.mark.asyncio
async def test_r9_titles_and_screen_size_are_not_paragraph_edits(books):
    ag, store, llm = make(books)
    n = len(llm.calls)
    for t, want in (("Change the name of chapter one to Home Again.", "Home Again"), ("I want the chapter title to be The Platform.", "The Platform"),
                    ("Call chapter one The Late Train.", "The Late Train")):
        await ag.handle("s", t)
        assert store.chapter_title("demo", 1) == want, t
    r = await ag.handle("s", "Make the words bigger, I can't see them so good.")
    assert r.ui and ag.session("s").pending is None and len(llm.calls) == n
    ag.session("s").pending = __import__("agent").Pending("confirm_target", "make it sadder", 1, paragraph_id=store.paragraphs("demo", 1)[0].id)
    await ag.handle("s", "No. Call chapter one Mira Arrives.")
    assert store.chapter_title("demo", 1) == "Mira Arrives" and ag.session("s").pending is None


@pytest.mark.asyncio
async def test_r9_put_it_in_anyway_at_the_end_places_the_picture_at_the_end(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "a station at night", "caption": "The station", "after": 1}]]', "Here it is, a station at night with lamps."]
    await ag.handle("s", "make a picture of the station at night")
    assert ag.session("s").pending and ag.session("s").pending.kind == "picture"
    await ag.handle("s", "Well, put it in anyway, at the end of the chapter. It's real at least.")
    paras = store.paragraphs("demo", 1)
    assert paras[-1].image and ag.session("s").pending is None


@pytest.mark.asyncio
async def test_r9_read_me_chapter_one_reads_the_whole_chapter(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    await ag.handle("s", "go to chapter 2")
    r = await ag.handle("s", "read me chapter one")
    assert all(p.text[:20] in r.text for p in store.paragraphs("demo", 1) if not p.image) and ag.session("s").work == 2


def test_r9_the_family_tree_gives_wedding_days():
    import genealogy
    t = genealogy.parse_gedcom(genealogy.decode_text((FILES / "lund_family.ged").read_bytes()))
    everett = next(p for p in t.people.values() if p.name.startswith("Everett"))
    assert "14 JUN 1919" in t.describe(everett.id)


@pytest.mark.asyncio
async def test_r9_a_page_the_writer_already_said_yes_to_is_kept_without_asking_again(books):
    ag, store, llm = make(books)
    import web as webmod
    ag.web = webmod.Web(wikipedia_url="https://wiki.test")
    s = ag.session("s")
    s.web_hits["last"] = s.research["Lindsborg"] = {"title": "Lindsborg", "url": "http://x/wiki/Lindsborg", "text": "Lindsborg was founded in 1869 by Swedish immigrants."}
    s.research["Swedish emigration"] = {"title": "Swedish emigration", "url": "http://x/wiki/Swe", "text": "About 1.3 million Swedes emigrated between 1850 and 1930."}
    s.last_user = "Yes, keep the Swedish emigration page."
    out = await ag.toolbox.call("propose_save_page", {"title": "Swedish emigration"}, harness.Ctx(sess=s))
    assert "Do NOT ask" in out and s.pending is None and any("emigration" in m["title"].lower() for m in ag.library.items("demo"))


# ---------------------------------------------------------------------------------------------- grandpa round 10 (regressions)
@pytest.mark.asyncio
async def test_r10_a_yes_with_remarks_around_it_is_still_a_yes(books):
    ag, store, llm = make(books)
    n = len(store.paragraphs("demo", 1))
    _wait(ag)
    llm.replies = ["Inside the station the air was warm and smelled of coal smoke, and the clerk nodded to him from behind the ledger."] * 4
    await ag.handle("s", "That's good. Yes, put the draft one in. Then write about him going to the station.")
    assert len(store.paragraphs("demo", 1)) == n + 1 and DRAFT in store.read_chapter_raw("demo", 1)
    assert ag.session("s").pending and "coal smoke" in ag.session("s").pending.text
    ag.session("s").pending = None
    for t in ("Oh that's nice. Yes, add it.", "Yes, that's him all the same. Put it in.", "Yes. Put the picture in."):
        d = f"A different paragraph for {t[:12]}: the harvest came in late that year and the barn smelled of dust."
        _wait(ag, text=d)
        await ag.handle("s", t)
        assert d in store.read_chapter_raw("demo", 1), t


@pytest.mark.asyncio
async def test_r10_two_commands_in_one_breath(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    await ag.handle("s", "Call chapter one Over There. Then go to chapter two.")
    assert store.chapter_title("demo", 1) == "Over There" and ag.session("s").chapter == 2
    await ag.handle("s", "Fix the name of chapter one, it should just be The Platform. And start chapter three.")
    assert store.chapter_title("demo", 1) == "The Platform" and 3 in store.chapter_numbers("demo")
    llm.replies = ["He came home in the spring of 1919, thinner, and walked the last mile from the station to the farm with his bag."] * 3
    r = await ag.handle("s", "Start chapter four. He comes home from France in nineteen nineteen.")
    s = ag.session("s")
    assert 4 in store.chapter_numbers("demo") and s.pending and s.pending.kind == "append" and s.pending.chapter == 4


@pytest.mark.asyncio
async def test_r10_you_cant_see_anything_about_a_picture_is_not_the_screen(books):
    ag, store, llm = make(books)
    llm.replies = ['[[tool: make_picture {"prompt": "soldiers in the rain", "caption": "Soldiers", "after": 1}]]', "Here it is: soldiers walking in the rain."]
    await ag.handle("s", "make a picture of soldiers in the rain")
    r = await ag.handle("s", "Hm. I don't like that one much, you can't see anything. Well, put it in anyway, at the end of the chapter.")
    assert not r.ui and store.paragraphs("demo", 1)[-1].image


@pytest.mark.asyncio
async def test_r10_bring_it_back_and_put_it_in_saves_the_dropped_draft(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2)
    A = "The draft card came in June. Everett filled it in at the courthouse in a careful hand and folded the copy into his coat."
    B = "At the training camp the winter came early, and the tents snapped in the wind all night along the long rows."
    s = ag.session("s")
    s.pending = __import__("agent").Pending("append", A, 2)
    assert ag._supersede(s)
    s.pending = __import__("agent").Pending("append", B, 2)
    await ag.handle("s", "No, no. I said yes to that one. Bring it back and put it in chapter two.")
    assert A in store.read_chapter_raw("demo", 2) and s.pending and s.pending.text == B


def test_r10_the_tree_finds_a_child_whose_father_lacks_the_family_link():
    import genealogy
    t = genealogy.parse_gedcom(genealogy.decode_text((FILES / "lund_family.ged").read_bytes()))
    karl = next(p for p in t.people.values() if p.name.startswith("Karl"))
    assert any(t.people[c].name.startswith("Dorothy") for c in t.children(karl.id))


def test_r10_undo_with_a_reason_and_plurals_of_known_words():
    import intent
    assert intent.undo_request("Undo that, I liked cold better.")


# ---------------------------------------------------------------------------------------------- grandpa round 11 (regressions)
async def _picture_waiting(ag, llm):
    llm.replies = ['[[tool: make_picture {"prompt": "a young soldier in uniform", "caption": "Everett, 1918", "after": 1}]]', "Here he is: a young soldier in uniform."]
    await ag.handle("s", "make a picture of Everett in his uniform")
    assert ag.session("s").pending and ag.session("s").pending.kind == "picture"


@pytest.mark.asyncio
async def test_r11_remarks_before_a_yes_on_a_picture_are_not_screen_requests(books):
    for t in ("I can't see his face so good, it's very faded. Yes, that's him all the same. Put it in.", "I can't really see it, but alright. Yes, put it in there."):
        ag, store, llm = make(books)
        await _picture_waiting(ag, llm)
        n = len(llm.calls)
        r = await ag.handle("s", t)
        assert not r.ui and ag.session("s").pending is None and any(p.image for p in store.paragraphs("demo", 1)) and len(llm.calls) == n, t


@pytest.mark.asyncio
async def test_r11_yes_with_praise_or_them_and_a_request_after(books):
    ag, store, llm = make(books)
    _wait(ag)
    await ag.handle("s", "Yes, that's beautiful.")
    assert DRAFT in store.read_chapter_raw("demo", 1)
    s = ag.session("s")
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "Tova Berg", "note": "A neighbour."}]}, harness.Ctx(sess=s))
    llm.replies = ["The Wieland came into New York harbour on the twenty-first of May, and Nils held the boy up to see the city across the water."] * 3
    await ag.handle("s", "Yes, save them, and write the next one, um, start the first chapter.")
    assert "Tova Berg" in store.bible("demo").characters() and s.pending and s.pending.kind == "append"
    s.pending = None
    d2 = "Snow lay on the fence posts and the cows stood close by the barn door waiting for the hay to come down."
    _wait(ag, text=d2)
    llm.replies = ["I can put the farm photo at the end of chapter one."]
    await ag.handle("s", "Yes, that's nice. Put it in. Now, I gave you an old picture of the farm. Can you put that one at the end of chapter one?")
    assert d2 in store.read_chapter_raw("demo", 1) and "farm" in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_r11_a_request_before_the_yes_is_done_after_saving(books):
    ag, store, llm = make(books)
    s = ag.session("s")
    await ag.toolbox.call("propose_wiki_notes", {"notes": [{"kind": "characters", "name": "Nils Lund", "note": "Everett's father."}]}, harness.Ctx(sess=s))
    llm.replies = ["I will add Dorothy and Karin as well."]
    await ag.handle("s", "Dorothy is my little sister, she's still living. Put her in too, and Karin. Yes, save them all.")
    assert "Nils Lund" in store.bible("demo").characters() and "Put her in too" in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_r11_bring_it_back_and_put_it_in_saves_even_after_another_save(books):
    ag, store, llm = make(books)
    A = "The draft card came in June. Everett filled it in at the courthouse in a careful hand and folded the copy into his coat."
    s = ag.session("s")
    s.pending = __import__("agent").Pending("append", A, 1)
    assert ag._supersede(s)
    s.dropped_stale = True  # something else was saved since
    await ag.handle("s", "No, no. I said yes to that one. Bring it back and put it in chapter one.")
    assert A in store.read_chapter_raw("demo", 1)


def test_r11_undo_after_a_hmm_and_smaller_screen_words_at_100_percent():
    import intent, uisettings
    assert intent.undo_request("Hmm. Undo that, I liked it better before.")
    look = dict(uisettings.DEFAULT); look["size"] = 5
    new, said = uisettings.parse("Make the words a bit smaller again, that's too big.", look)
    assert new["size"] == 4


# ------------------------------------------------------------------------------- real model (Gemini Flash) round 1 (regressions)
def test_rm1_every_tool_call_format_is_understood_and_never_read_aloud():
    for raw, name in (('[[tool: list_library {}]]\\nOkay.', "list_library"), ('[[propose_delete: {"paragraph": 1, "chapter": 4}]]', "propose_delete"),
                      ('[[bible_lookup {"name": "x"}]] hi', "bible_lookup"), ("[[list_library]]", "list_library")):
        calls, text = harness.parse_calls(raw.replace("\\\\n", "\\n"))
        assert calls and calls[0][0] == name and "[[" not in text


@pytest.mark.asyncio
async def test_rm1_a_note_instead_of_a_tool_call_is_asked_again_and_never_becomes_a_draft(books):
    ag, store, llm = make(books)
    P = ("Snow lay on the fence posts that winter. The cows stood close by the barn door, waiting for the hay to come down from the loft. "
         "Samuel broke the ice on the trough every morning with the back of an axe. His mother kept the kitchen stove going from dark to dark.")
    llm.replies = ["[proposed a new paragraph]", f'[[tool: propose_append {{"text": "{P}"}}]]', "Here it is."]
    r = await ag.handle("s", "Tell me about the winter on the farm, and put it in the book as a paragraph about the snow")
    s = ag.session("s")
    assert s.pending and s.pending.text == P and "[proposed" not in r.text
    s.pending = None
    llm.replies = ["[proposed a new paragraph]", "Still a note.", "Still a note."]
    r = await ag.handle("s", "write the next paragraph")
    assert s.pending is None and "could not write" in r.text


@pytest.mark.asyncio
async def test_rm1_a_claimed_save_that_did_not_happen_is_corrected(books):
    ag, store, llm = make(books)
    before = store.read_chapter_raw("demo", 1)
    llm.replies = ["Wonderful, I've saved that paragraph to chapter 1 for you."]
    r = await ag.handle("s", "Is the letter in the book now? Thank you so much for everything today.")
    assert store.read_chapter_raw("demo", 1) == before and "not changed anything" in r.text


def test_rm1_undo_in_a_later_sentence_and_a_picture_question_is_not_the_screen():
    import intent, uisettings
    assert intent.undo_request("That's not the paragraph at all. Undo that.")
    assert intent.undo_request("No, that's in the wrong place after the battle. Undo.")
    assert uisettings.parse("Now I have the old tintype picture of Samuel. Can you tell me what it looks like? I can't make it out.", None) is None
    assert intent.ideas_request("give me a name for the farm dog") is None
    from agent import clean_draft
    assert clean_draft("My apologies. I had proposed the paragraph, but it seems there was a misunderstanding. Here is the paragraph I drafted: The rain came.") == "The rain came."


@pytest.mark.asyncio
async def test_rm2_native_tool_calls_are_used_when_the_model_supports_them(books):
    ag, store, llm = make(books)
    P = ("Snow lay on the fence posts that winter. The cows stood close by the barn door, waiting for the hay to come down from the loft. "
         "Samuel broke the ice on the trough every morning with the back of an axe. His mother kept the kitchen stove going from dark to dark.")
    seen = []

    async def chat_tools(messages, tools, **kw):
        seen.append([t["function"]["name"] for t in tools])
        if not any(m.get("role") == "tool" for m in messages):
            return {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "propose_append", "arguments": json.dumps({"text": P})}}]}
        return {"content": "Here is a paragraph about the winter."}
    llm.chat_tools = chat_tools
    r = await ag.handle("s", "Tell me about the winter on the farm and give me a paragraph about it")
    assert seen and "propose_append" in seen[0] and ag.session("s").pending and ag.session("s").pending.text == P


@pytest.mark.asyncio
async def test_rm2_no_wait_keep_it_refuses_a_delete(books):
    ag, store, llm = make(books)
    before = store.read_chapter_raw("demo", 1)
    await ag.handle("s", "delete paragraph 2")
    await ag.handle("s", "No, wait. Keep it.")
    assert store.read_chapter_raw("demo", 1) == before and ag.session("s").pending is None


@pytest.mark.asyncio
async def test_rm2_a_paragraph_written_into_the_chat_becomes_a_real_draft(books):
    ag, store, llm = make(books)
    P = ("Samuel walked the last mile home in the rain with his left arm still stiff from the wound, and the fields along the road lay flat and brown under the low sky. "
         "At the gate he stopped and looked for a long time at the house, at the smoke from the kitchen chimney and the lamp already lit in the window. "
         "His mother came out onto the porch with her apron still on, and she did not say a word until she had both her hands on his face.")
    llm.replies = [f"Of course, Walter. Here is the paragraph:\n\n{P}\n\nHow does that sound?"]
    r = await ag.handle("s", "Now a paragraph about him coming home to the farm in the rain")
    s = ag.session("s")
    assert s.pending and s.pending.kind == "append" and s.pending.text.startswith("Samuel walked") and "How does" not in s.pending.text


def test_rm2_spoken_years_markdown_and_junk_around_drafts():
    import intent
    from agent import Agent, clean_draft
    assert intent.norm("the tintype eighteen sixty-two") == "the tintype 1862" and intent.norm("paragraphs 12 13") == "paragraphs 12 13"
    assert Agent._plain_talk("**Muster card:** he enlisted.\n- one\n* two") == "Muster card: he enlisted.\none\ntwo"
    assert clean_draft("propose_append: The rain came down on the camp all night.") == "The rain came down on the camp all night."
    assert clean_draft("The rain came down on the camp all night. How does this sound for your book?") == "The rain came down on the camp all night."


@pytest.mark.asyncio
async def test_rm2_start_the_next_chapter_while_an_earlier_one_is_on_screen(books):
    ag, store, llm = make(books)
    store.create_chapter("demo", 2); store.create_chapter("demo", 3)
    llm.replies = ["He came home in the spring, thinner, and walked the last mile from the station to the farm with his bag over his good shoulder."] * 3
    r = await ag.handle("s", "Start chapter four. He comes home from the war in eighteen sixty-five.", viewing=1)
    assert 4 in store.chapter_numbers("demo") and ag.session("s").pending and ag.session("s").pending.chapter == 4


@pytest.mark.asyncio
async def test_rm3_book_title_spoken_titles_and_junk_never_becomes_a_draft(books):
    ag, store, llm = make(books)
    await ag.handle("s", "Call the book Samuel's War.")
    import export
    assert export.book_title(store, "demo") == "Samuel's War"
    from agent import clean_draft
    assert clean_draft("I can put this in the book for you: The rain came down all night on the camp.") == "The rain came down all night on the camp."
    assert clean_draft('How about this: "The rain came down all night on the camp."') == "The rain came down all night on the camp."
    assert clean_draft("(Note for me: I called propose_append. The app now shows this draft.) The rain came down.") == "The rain came down."
    out = await ag.toolbox.call("propose_append", {"text": "I apologize, you're right. Daniel was not at Vicksburg, so I have changed the paragraph to leave him out of it."}, harness.Ctx(sess=ag.session("s")))
    assert out.startswith("Error") and ag.session("s").pending is None


@pytest.mark.asyncio
async def test_rm4_a_plain_word_swap_needs_no_model_and_a_cut_off_rewrite_is_refused(books):
    ag, store, llm = make(books)
    p2 = store.paragraphs("demo", 1)[1]
    n = len(llm.calls)
    await ag.handle("s", "In paragraph two, just change Jonas to Nils. Nothing else.")
    s = ag.session("s")
    assert len(llm.calls) == n and s.pending and s.pending.kind == "edit" and s.pending.text == p2.text.replace("Jonas", "Nils", 1)
    s.pending = None
    llm.replies = ['propose_change( paragraph_id="ch1_s1_p2", text="The express was late." )', "The express was late, and Jonas waited by the"]
    r = await ag.handle("s", "rewrite paragraph two so it is snowing")
    assert s.pending is None and "left the paragraph as it is" in r.text


@pytest.mark.asyncio
async def test_rm4_the_writers_own_picture_is_found_by_spoken_words_and_placed(books):
    ag, store, llm = make(books)
    m = await ingest.add_file("demo", "samuel_whitfield_tintype_1862.jpg", png(400, 300), ag.library, None, None, ag.journal)
    r = await ag.handle("s", "Put the old tintype from eighteen sixty-two after the first paragraph of chapter one.")
    s = ag.session("s")
    assert s.pending and s.pending.kind == "picture" and s.pending.data.get("where") == 1 and r.picture


@pytest.mark.asyncio
async def test_rm4_yes_inside_a_long_sentence_and_book_title_while_a_card_waits(books):
    ag, store, llm = make(books)
    _wait(ag)
    await ag.handle("s", "It reads a bit like a school book, but yes, save it.")
    assert DRAFT in store.read_chapter_raw("demo", 1)
    _wait(ag, text=DRAFT.replace("Everett", "Samuel"))
    await ag.handle("s", "Call the book Samuel's War.")
    import export
    assert export.book_title(store, "demo") == "Samuel's War"


def test_rm4_writing_and_print_requests():
    import intent
    assert intent.is_draft("Tell the part where he rides into Newark and signs up at Camp Sherman.")
    assert intent.export_request("Save it so I can print it") == "html"


@pytest.mark.asyncio
async def test_rm5_a_remark_then_the_book_title_and_junk_never_reaches_the_book(books):
    ag, store, llm = make(books)
    llm.replies = ["Buck is a fine name for a farm dog."]
    await ag.handle("s", "Buck. I like Buck. Now, call the book Samuel's War.")
    import export
    assert export.book_title(store, "demo") == "Samuel's War"
    for junk in ("Let me try again, using only what we know. How about this: \"The rain fell on the camp all night and nobody slept at all.\"",
                 "The rain fell on the camp all night and nobody slept at all. Should I save it?"):
        _wait(ag, text=junk)
        await ag.handle("s", "yes")
    raw = store.read_chapter_raw("demo", 1)
    assert "Let me try" not in raw and "Should I save" not in raw and "How about" not in raw


def test_rm5_more_answers_and_refusals():
    import intent
    assert intent.confirmation("Yes, that's more like it.") == ("yes", "")
    assert intent.book_title("Please make sure the title of the book itself is Samuel's War, so it's on the front page when I print it.") == "Samuel's War"
