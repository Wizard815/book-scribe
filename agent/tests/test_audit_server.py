"""Regressions from the HTTP / data-handling audit."""
import io
import zipfile

import pytest

import genealogy
from book_store import BookStore
from import_text import import_text


def test_a_manuscript_with_hash_scene_breaks_imports_whole(books):
    store = BookStore(books)
    text = "Chapter 1: Arrival\n\nShe came by the late train.\n\n#\n\nMorning came grey.\n\n* * *\n\nNoon.\n\nChapter 2: Leaving\n\nThey left."
    assert import_text(store, "fresh", text) == (2, 4)
    ps = store.paragraphs("fresh", 1)
    assert [p.scene for p in ps] == [1, 2, 3] and ps[1].text == "Morning came grey."
    assert not (store.book_dir("fresh") / "history").exists() or not any((store.book_dir("fresh") / "history").glob("*.md"))  # no undo copy per paragraph


def test_a_backslash_in_a_chapter_title_is_just_a_character(books):
    store = BookStore(books)
    assert store.set_chapter_title("demo", 1, r"Love\Hate \2") == r"Love\Hate \2"
    assert store.chapter_title("demo", 1) == r"Love\Hate \2"


def test_a_hand_named_chapter_file_is_found(books):
    store = BookStore(books)
    d = store.book_dir("demo") / "chapters"
    (d / "chapter_02.md").rename(d / "chapter_2.md") if (d / "chapter_02.md").exists() else None
    raw = store.read_chapter_raw("demo", 1).replace("chapter: 1", "chapter: 7")
    (d / "chapter_7.md").write_text(raw)
    assert 7 in store.chapter_numbers("demo") and store.paragraphs("demo", 7)


def test_control_characters_never_reach_the_chapter_file(books):
    store = BookStore(books)
    p = store.append_paragraph("demo", 1, "Bell\x07 and esc\x1b and \x0bvertical tab")
    assert "\x07" not in p.text and "\x0b" not in store.read_chapter_raw("demo", 1)


def test_old_undo_copies_are_deleted_when_they_fall_off_the_end(books):
    store = BookStore(books)
    store.HISTORY_KEEP = 3
    for i in range(8):
        store.append_paragraph("demo", 1, f"Paragraph number {i} of the test.")
    assert len(list((store.book_dir("demo") / "history").glob("*.md"))) == 3


def test_one_bad_byte_does_not_garble_a_utf8_file():
    assert genealogy.decode_text("Zoë Müller from Kraków".encode() + b"\x96") .startswith("Zoë Müller from Kraków")
    assert genealogy.decode_text("Zoë Müller".encode("cp1252")) == "Zoë Müller"


def test_a_zip_with_one_damaged_file_keeps_the_good_ones():
    import extract
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("good.txt", "A perfectly good letter from Kansas about the harvest and the rain.")
        z.writestr("broken.docx", b"PK\x03\x04 not really a docx")
    ex = extract.extract(buf.getvalue(), "letters.zip")
    assert "Kansas" in ex.text and any("broken.docx" in n for n in ex.notes)


def test_a_new_wiki_entry_never_replaces_an_existing_page(books):
    from bible import BibleError
    b = BookStore(books).bible("demo")
    b.save_entry("characters", "Tova", {"age": "31"}, "Three pages of notes about Tova.")
    with pytest.raises(BibleError):
        b.save_entry("characters", "Tóva", {}, "new")  # same page file, no slug: refused
    assert "Three pages" in {x.name: x for x in b.entries("characters").values()}["Tova"].body


def test_family_import_keeps_the_writers_page_and_both_people_with_one_name(books):
    import bible_import
    b = BookStore(books).bible("demo")
    b.upsert("characters", "Zoë Ward", {"born": "1826 (my version)"}, "MY OWN NOTES: she hides letters in the piano.")
    pl = bible_import.Plan(slug="family", title="Family", privacy="real")
    pl.characters = [{"name": "Zoe Ward", "state": {"born": "1825"}, "body": "Zoe from the tree."},
                     {"name": "John Smith", "state": {"born": "1820"}, "body": "Senior."},
                     {"name": "John Smith", "state": {"born": "1850"}, "body": "Junior."}]
    pl.places, pl.timeline, pl.aliases = [], [], {}
    out = bible_import.apply(b, pl)
    e = {x.slug: x for x in b.entries("characters").values()}
    assert "MY OWN NOTES" in e["zoe-ward"].body and e["zoe-ward"].state["born"] == "1826 (my version)"
    assert "Senior." in e["john-smith"].body and "Junior." in e["john-smith-1850"].body and out["characters_added"] == 2
