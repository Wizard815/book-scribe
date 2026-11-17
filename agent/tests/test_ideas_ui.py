import pytest

import uisettings
from ideas import KINDS, Ideas
from intent import ideas_request, is_creative_ask
from quality import baseline_max_sentence, check


# --- generators ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", list(KINDS))
def test_every_kind_produces_distinct_non_empty_answers(kind):
    out = Ideas(seed=1).generate(kind, 6)
    assert len(out) == 6 and all(x.strip() for x in out) and len(set(out)) >= 5


def test_the_same_seed_gives_the_same_answers_and_different_seeds_differ():
    assert Ideas(seed=5).generate("name", 8) == Ideas(seed=5).generate("name", 8)
    assert Ideas(seed=5).generate("name", 8) != Ideas(seed=6).generate("name", 8)


def test_a_session_does_not_hear_the_same_answer_twice():
    g = Ideas(seed=2)
    seen = [x for _ in range(20) for x in g.generate("name", 5, culture="english", gender="female")]  # 400 possible, 100 drawn
    assert len(seen) == 100 and len(set(seen)) >= 98


def test_name_options_are_respected():
    from ideas_data import NAMES
    for n in Ideas(seed=3).generate("name", 12, culture="irish", gender="female"):
        first, last = n.split(" ", 1)
        assert first in NAMES["irish"]["female"] and last in NAMES["irish"]["family"]
    assert all(" " not in n for n in Ideas(seed=3).generate("name", 6, culture="fantasy_elvish"))
    assert Ideas(seed=1).generate("name", 3, culture="nonsense") and Ideas(seed=1).generate("place", 3, type="inn")[0].startswith("The ")


def test_surnames_agree_with_gender_where_the_language_does():
    names = Ideas(seed=1).generate("name", 12, culture="russian", gender="female")
    assert all(n.split()[1].endswith("a") for n in names if n.split()[1][-3:-1] in ("ov", "ev", "in"))
    assert any(n.endswith("a") for n in names)


def test_unknown_kind_is_refused_and_counts_are_capped():
    with pytest.raises(ValueError):
        Ideas().generate("recipe")
    assert len(Ideas().generate("name", 500)) == 12


def test_sparks_are_varied_and_labelled():
    s = Ideas(seed=4).sparks(3)
    assert len(s) == 3 and len({x.split(":")[0] for x in s}) == 3


# --- asking for ideas out loud ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("said,kind,opts", [
    ("give me a name for the porter", "name", {}),
    ("I need an Irish name for a girl", "name", {"culture": "irish", "gender": "female"}),
    ("give me three Japanese male names", "name", {"culture": "japanese", "gender": "male"}),
    ("suggest a place, maybe an inn", "place", {"type": "inn"}),
    ("come up with a twist", "twist", {}),
    ("describe a character for me", "character", {}),
    ("give me a fantasy name", "name", {"culture": "fantasy_elvish"}),
    ("what should I call him", "name", {"gender": "male"}),
])
def test_asking_for_ideas(said, kind, opts):
    r = ideas_request(said)
    assert r["kind"] == kind and r["opts"] == opts


@pytest.mark.parametrize("said", ["what is the name of the inn in chapter 2", "make the name more modern", "I like the name Mira",
                                  "I want to write a name on the wall", "go to chapter two", "how do I name a boat"])
def test_questions_and_edits_about_existing_names_are_not_idea_requests(said):
    assert ideas_request(said) is None


def test_counts_and_repeats():
    assert ideas_request("give me three names")["n"] == 3 and ideas_request("ten names")["n"] == 10
    assert ideas_request("more") == {"repeat": True} and ideas_request("another one") == {"repeat": True}


# --- the screen -------------------------------------------------------------------------------------------------------------------------
D = uisettings.DEFAULT


def look(said, cur=None):
    r = uisettings.parse(said, cur or D)
    return r[0] if r else None


@pytest.mark.parametrize("said", ["I can't see this", "I cannot see the words", "it's too small", "my eyes are tired", "I can't read it", "the buttons are too small"])
def test_cant_see_makes_everything_bigger_bolder_and_stronger_at_once(said):
    l = look(said)
    assert l["size"] == 4 and l["contrast"] == "high" and l["bold"] is True and l["spacing"] == 1


@pytest.mark.parametrize("said", ["I can't see where this story is going", "make the first paragraph bigger", "make Mr Bennet more sarcastic",
                                  "how big is the book", "go to chapter two", "I can't see her doing that", "what is the weather like in chapter 3"])
def test_story_talk_is_not_mistaken_for_a_screen_request(said):
    assert uisettings.parse(said, D) is None


def test_size_steps_and_limits():
    assert look("make everything bigger")["size"] == 3 and look("make everything much bigger")["size"] == 5 and look("make the screen a lot bigger")["size"] == 5
    assert look("biggest text")["size"] == uisettings.SIZE_MAX and look("make everything smaller", {**D, "size": 0})["size"] == 0
    assert look("make everything bigger", {**D, "size": 7}) is not None  # at the top it says so rather than erroring
    assert uisettings.parse("make everything bigger", {**D, "size": 7})[1] == "That is already as far as it goes."
    assert look("normal size", {**D, "size": 6})["size"] == 2


def test_colours_bold_spacing_speech_and_layout():
    assert look("high contrast")["contrast"] == "high" and look("yellow on black")["contrast"] == "dark" and look("normal colours", {**D, "contrast": "dark"})["contrast"] == "normal"
    assert look("dark mode")["theme"] == "dark" and look("bold letters")["bold"] is True and look("more space between the lines")["spacing"] == 1
    assert look("read slower")["rate"] == 0.85 and look("talk faster")["rate"] == 1.15
    assert look("just the chat")["layout"] == "chat" and look("just the book")["layout"] == "book" and look("too many buttons")["simple"] is True
    assert look("show everything", {**D, "simple": True, "layout": "chat"}) == {**D, "simple": False, "layout": ""}


def test_back_to_normal_resets_everything():
    assert look("back to normal", {**D, "size": 6, "contrast": "dark", "bold": True, "simple": True}) == D


def test_a_look_is_always_valid():
    l = uisettings.clean({"size": 99, "contrast": "rainbow", "bold": "yes", "spacing": -4, "rate": 9, "layout": "../x", "evil": 1, "theme": "dark"})
    assert l == {**D, "size": 7, "spacing": 0, "rate": 1.5, "theme": "dark"} and "evil" not in l
    assert uisettings.clean(None) == D and uisettings.clean({"size": True})["size"] == 2


# --- the style check ------------------------------------------------------------------------------------------------------------------------
LONG = "He " + "went down the long road and then across the river and over the hills " * 5 + "home."


def test_strictness_levels():
    cliche = "It was a testament to her will."
    assert check(cliche, "off") == [] and check(cliche, "gentle") and check(cliche, "strict")
    assert check(LONG, "gentle") == [] and any("too long" in i for i in check(LONG, "strict"))  # length is only enforced when asked
    assert check("They were all silly and ignorant.", "strict") == []


def test_the_limit_follows_the_books_own_long_sentences():
    from book_store import Paragraph
    austen = [Paragraph(f"p{i}", "She said that " + "a thing was so, and then another was also so, " * 6 + "indeed it was. A short one.", 1, 1) for i in range(60)]
    plain = [Paragraph(f"q{i}", "He ran. She fell. They left the house at dawn. It rained.", 1, 1) for i in range(60)]
    assert baseline_max_sentence(austen) > 70 and baseline_max_sentence(plain) == 55 and baseline_max_sentence(austen[:3]) == 55
    assert check(LONG, "strict", baseline_max_sentence(austen)) == [] and check(LONG, "strict", 55)


# ---- the book page's own zoom (100 to 500 percent): bigger words, more pages -------------------------------------------------------
def zoom(said, cur=100):
    r = uisettings.parse(said, {**D, "zoom": cur})
    return r and (r[0]["zoom"], r[0]["size"])


def test_book_zoom_by_voice_steps_through_the_levels_and_leaves_the_rest_of_the_screen_alone():
    assert zoom("make the book text bigger") == (125, 2)
    assert zoom("zoom in", 200) == (250, 2)
    assert zoom("zoom out", 200) == (150, 2)
    assert zoom("make the words on the page a lot bigger", 100) == (150, 2)
    assert zoom("make the story text smaller", 300) == (250, 2)
    assert zoom("biggest book text") == (500, 2)


def test_book_zoom_takes_a_percentage_and_snaps_to_a_level():
    assert zoom("set the book to 300 percent") == (300, 2)
    assert zoom("zoom to 500%") == (500, 2)
    assert zoom("text 180 percent please") == (200, 2)
    assert zoom("zoom back to normal", 400) == (100, 2)  # only the zoom, not the whole look
    assert zoom("I am 100 percent sure") is None
    assert zoom("join the zoom call") is None


def test_plain_bigger_words_means_the_book_page_and_only_screen_or_everything_means_the_whole_screen():
    assert zoom("make the words bigger") == (125, 2) and zoom("that's too big", 200) == (150, 2) and zoom("make it bigger") == (125, 2)
    assert zoom("make the buttons bigger") == (100, 3) and zoom("make everything bigger") == (100, 3) and zoom("make the screen smaller", 100) == (100, 1)


def test_zoom_is_validated_and_set_by_the_model_tool_too():
    assert uisettings.clean({"zoom": 9999})["zoom"] == 500 and uisettings.clean({"zoom": 3})["zoom"] == 100 and uisettings.clean({"zoom": "x"})["zoom"] == 100
    assert uisettings.step_zoom(500, 1) == 500 and uisettings.step_zoom(100, -1) == 100
    assert uisettings.DEFAULT["zoom"] == 100
