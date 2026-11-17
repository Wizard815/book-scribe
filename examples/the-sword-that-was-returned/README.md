# The Sword That Was Returned (example output)

A complete ten-chapter book written through Book Scribe in a test: one Claude model played the writer (Edwin, 78, half-sighted, talking by "voice"),
another played the language model behind Book Scribe's chat, using the app's real prompts, tools and flows.

* `book.html`: the whole book with its pictures embedded (open in a browser; print to PDF). `book.epub`: the same for e-readers.
* `chapters/`, `images/`, `bible/`: the book exactly as Book Scribe stores it (Markdown chapters with picture paragraphs, the pictures, the story wiki).

**The pictures are placeholders.** The test had no image model, so a small script drew simple illustrations from the words of each prompt (the
caption bar says so), and the "web" pictures came from a mock search server. They show that pictures are made, described, confirmed, placed,
moved, saved and exported; real art needs a real `IMAGE_URL` server. The Wikipedia answers were also served from a mock written for the test.
The prose is whatever the model wrote under the app's rules; it has small continuity slips (the writer noted a few), which is the honest state of a
first draft made by talking.
