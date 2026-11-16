# How Book Scribe thinks: the agent and the chat together

There is one conversation and one brain per book. The screen's chat panel is just the window onto it; the "agent" is what answers. The
writer never starts a new chat or manages context: the server does, as below. (Any session id a browser sends is ignored.)

## Memory, in four layers

| Layer | What it holds | How it is used |
|---|---|---|
| Working memory | The newest turns, up to `CONTEXT_CHARS` (default 9000 characters) | Sent word for word with every request |
| Rolling summary | Everything older, folded into about 250 words by the model whenever working memory grows past 1.4x the budget (extractive fallback if no model) | Sent as "earlier in this long conversation" |
| Journal | **Every** turn, one JSON line each, one file per day (`<book>/journal/`). Voice is stored as the text the speech model heard, not as audio | Cut into chunks and embedded, so any idea or description said months ago can be found by meaning (`recall` tool and automatic recall) |
| Library | Files the writer added (`<book>/library/`): the original plus the text read from it | Embedded the same way (`search_library`, `read_library`, `family_lookup`) |

The story wiki (the bible) stays the exact, current truth; the embedding index covers paragraphs, wiki pages, journal chunks and library chunks
in one file per book, with a scope on each search (`para`, `bible`, `mem`, `lib`).

**Clear chat** empties the screen and working memory, but first folds the talk into the summary; the journal keeps every word. After a reload,
or on another device, the chat window is rebuilt from the journal.

**Audio.** The microphone goes to the speech-to-text backend and only the transcript is kept (a few kilobytes an hour, searchable). Audio
files added to the library are transcribed the same way. The original recording is kept beside the text if it is added as a file.

**Files.** Text and office documents, PDFs (scanned pages are read as pictures), spreadsheets, ebooks, zips, family trees, pictures (read by the
chat model if it can see pictures) and recordings. If a model that is needed is missing, the file is kept and marked "not read yet" with a
Try again button, never silently dropped.

## One turn, step by step

1. **Hear.** Voice goes to the speech-to-text backend; typed text arrives as-is.
2. **Route (no model, instant).** `intent.py` normalises the sentence and checks for things the app can do itself:
   go to a chapter, read this aloud, "I can't see this" (change the screen), "give me three Irish names" (random generator),
   voice changes, yes/no to something waiting for confirmation. These cost **zero model calls**.
3. **Gather context (cheap).** Relevant story-bible entries (by name *and* by meaning, via embeddings), passages found by the
   locator, the writer's preferences, mem0 memory.
4. **Think (model, with tools).** The model gets that context plus a small toolbox (`harness.py`, `tools.py`). It may answer
   straight away, or reply with a line such as `[[tool: random_ideas {"kind":"name","culture":"irish"}]]`. The harness runs it,
   returns the result, and lets the model continue (at most 3 rounds). A model that never uses the protocol just answers.
5. **Attach.** Whatever the tools and the reply point at (passages to light up, a new screen look, tap-to-use idea buttons)
   is attached to the reply. Plain references in the text ("chapter 2, the line about the carriage") are also found and lit.
6. **Confirm before writing.** Anything that would change the book is proposed in words and only saved after a yes.
   The model has **no** tool that writes to the manuscript.
7. **Speak.** The reply goes to text-to-speech with a voice per character.

## Where the freedom is, and where the limits are

* **Free:** content, opinions, suggestions, prose style (guidance only; the style check is calibrated to the book and defaults
  to "gentle"), how bold the ideas are (Settings: surprise off / low / high raises the model's temperature and nudges it to
  use the random generator).
* **Fixed on purpose:** the manuscript is never changed without a yes; story-bible facts are not contradicted; one writer
  account and a separate admin who cannot read the book.

## Tools the model can use

| Tool | What it does |
|---|---|
| `random_ideas`, `roll_sparks` | Real randomness for names (17 cultures + fantasy), places, characters, problems, twists, titles... |
| `search_book`, `read_passage`, `bible_lookup`, `book_info` | Look things up instead of guessing |
| `show_passage` | Light up a passage on the writer's screen |
| `set_ui` | Make words bigger, raise contrast, simplify the screen |
| `recall` | Search everything ever said in the long conversation, by meaning |
| `search_library`, `read_library`, `list_library` | Find and read the files the writer added |
| `family_lookup` | Look up a person in an imported family tree |
| `wiki_list`, `wiki_read` | Browse the story wiki |
| `propose_wiki_note` | Offer to add a note to a wiki page; saved only after the writer says yes |
| `use_skill` | Load a playbook (`agent/skills/*.md`): character interview, continuity check, scene from memory, family to fiction, research digest, wiki tidy, sensory scene, dialogue polish |
| `brainstorm` | Several different takes at once, each with its own random ingredients (costs extra time) |

Add a tool: write an async function in `agent/tools.py` and register it in the `Toolbox` list at the bottom.
Add ideas: extend the lists in `agent/ideas_data.py`.

## Known limits

* Replies arrive whole, not streamed word by word. Streaming is the next big speed win.
* Tool use depends on the model following the one-line text protocol; small local models may ignore it (nothing breaks, they just answer).
