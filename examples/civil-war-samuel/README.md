# Civil War test: "Samuel's War", written by voice with a real model

A writer agent playing Walter Mercer (79, half-sighted, voice only) made a five-chapter book about his great-grandfather Samuel Thomas Whitfield
(Company B, 76th Ohio Volunteer Infantry, 1862–1865) by talking to the real app. The language model behind the app was a real one,
`google/gemini-2.5-flash` through OpenRouter, capped at $2 for all six runs ($1.79 spent, 967 model calls). After each run the problems were fixed
(see the git log, "Real-model round 1" to "round 6"). This folder holds the book from the last run.

* `family-files/`: the fictional papers (muster-in card, a letter from Ringgold, the 1890 pension declaration, the obituary), the GEDCOM family
  tree, and two old-looking photographs made with an image model for the test. The obituary's birth year (1842) disagrees with the tree (1841) on purpose.
* `samuels_war.html`: the exported book (title, five named chapters, the tintype, the farm photo, a web photo, a made picture).
  `final_book.txt`: its text. `writer-notes-run6.md`: the tester's notes and PASS/FAIL list for the last run.
* `wiki-mock/`: the history articles the mock Wikipedia served (real Wikipedia limited the test machine's requests).
* `test-rig/`: the spending-capped OpenRouter bridge, the mock web, and the writer's command script.

About the pictures (read this before sharing the book): ALL FOUR are made by an image model, none is a real photograph. The "tintype" and the
"1885 farm photo" were generated for the test and given file names that sound real; the "archive photographs" the web search returned were also
generated for the test, and their captions ("archive photograph", "Ringgold Gap, Georgia, 1864") were written by the test rig, not taken from any
archive (Ringgold Gap was fought on 27 November 1863). The made picture in chapter 5 is captioned as if it were a photo; since then the app labels
made pictures "(an illustration)". The "tintype" also shows officer's shoulder straps on a private, and the farm photo shows a much older man than
Samuel would have been in 1885.

About the history: the mock Wikipedia in `wiki-mock/` was written for the test and leaves things out that a real researcher would find (for
example the regiment's veteran re-enlistment in early 1864); a reviewer also points out that discharge papers for Ohio regiments were often given
in Columbus after the muster-out in Louisville. Check such details against real sources.

About the book: the tester was asked for only two paragraphs per chapter, so this is a synopsis with pictures, not a finished book.

Honest notes: the 76th Ohio and the battles are real history; Samuel and his papers are fictional. A small, cheap model still makes small factual
slips (an invented detail, a wrong count of people in a photo) that the writer must catch; the app now refuses to save its talk, refusals or tool text
as story, and checks claimed saves, but it cannot make the model's facts right on its own.
