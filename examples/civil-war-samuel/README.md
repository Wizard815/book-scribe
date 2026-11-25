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

Honest notes: the 76th Ohio and the battles are real history; Samuel and his papers are fictional. A small, cheap model still makes small factual
slips (an invented detail, a wrong count of people in a photo) that the writer must catch; the app now refuses to save its talk, refusals or tool text
as story, and checks claimed saves, but it cannot make the model's facts right on its own.
