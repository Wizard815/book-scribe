# What Book Scribe must be able to do (the checklist)

Each line is something a half-sighted grandparent writing a family-history book by voice must be able to do. **U** = a scripted unit test
(`agent/tests/`, the model is a script), **E** = a step in the end-to-end "grandpa test" (a writer agent and a model agent drive the real app;
see `examples/grandpa-lindsborg/`). A line is only ticked when it has passed. When a step fails, the missing logic is added and the test is run again.

## 1. Getting family material in
- [U/E] Add PDF family documents (draft card, letter, obituary, passenger list); text is read; a duplicate is recognised ("already added").
- [U/E] Add a family tree (GEDCOM); look a person up by voice ("who was Everett's father?").
- [U/E] Add an old photograph; say honestly if the model cannot see it; keep the writer's own description with it.
- [U/E] Search the added documents by meaning ("the letter with the postcard").
- [U/E] Bring a person and relatives from the tree into the wiki (characters, places, timeline), only after a yes; the living are protected.

## 2. Researching history
- [U/E] Look facts up (Wikipedia/web), say it did so, mark the result as untrusted data.
- [U/E] Keep a page or note only after a yes; never let a page change the book by itself.
- [E] Use the researched facts and the family documents consistently in the story; never invent facts about real family members.

## 3. Writing
- [U/E] Draft a paragraph that is exactly what was asked; show it, read it, save only after a yes (or "save as you go"); undo.
- [U/E] "One more, ...", "now the part where ...", "continue with ..." all mean new writing; prose written in chat is never lost.
- [U/E] Start and title chapters; five chapters; no repeated scenes; text always goes into the chapter being written.
- [U/E] "Ask me first" and "save as you go" stick; a dropped unanswered draft is said out loud and can be brought back.
- [U/E] A flagged draft says it may not be what was asked.

## 4. Pictures
- [U/E] Make a picture, find one on the web, use the writer's own; always described in words; yes before it goes in.
- [U/E] Put a picture where asked ("very start", "after paragraph 2", "at the end"); "no, put it at the end" moves it.
- [U/E] Describe any picture by chapter or by what it shows; never writes a paragraph.
- [U/E] Move or delete a picture, by voice or by the model's tools.

## 5. Editing
- [U] Rewrite, move, delete paragraphs by number (pictures not counted), "the picture in chapter 3"; every change is undoable.
- [U/E] A "no" to a rewrite of an older chapter returns to the chapter being written.
- [U/E] "Yes, but leave X out" leaves it out of a batch.

## 6. The screen (half-sighted writer)
- [U/E] Pages with 100% to 500% zoom by voice ("make the words bigger", "zoom to 300 percent", "back to normal"); the whole screen separately.
- [U/E] "Make it a bigger ship" is the story, not the screen.
- [U/E] Nothing ever tells the writer to press or look at anything.

## 7. Memory and finishing
- [U/E] Remembers how the book is to be told; which chapters have pictures; what is saved.
- [U/E] Save the whole book as an e-book and a web page with its pictures.

## Status (grandpa test, round 8)
Round 8 again produced a complete five-chapter book (third person, pictures where asked, uploads, family tree, research, sentence removal, redirects, describe/move/delete,
zoom, exports). Items that passed: everything in sections 1, 2, 5 (except the redirect below), 6 and 7. Findings and what was done afterwards (unit-tested only, not re-run through the story):
* invented details and repeats between chapters slipped through: the app now flags years/numbers/names found nowhere, and sentences whose words the documents never mention,
  and sends back a paragraph that re-says the book (word-sequence overlap or the same content words);
* "no, put it in chapter 4" when chapter 4 did not exist created an untitled chapter and moved where the writing went: the picture now stays waiting and the app says the chapter does not exist yet.
Known and not fixed: a stand-in model's flowery filler cannot be told from fact by the app (only by comparing with the documents), web pictures are described only by their title,
a real vision/writing model is untested.
