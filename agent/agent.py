"""Conversation orchestrator: intent routing, targeted edits with confirmation, memory, quality gate."""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from book_store import BookError, BookStore, Paragraph
import asyncio

from clients import LLM
import bible_import
import ingest
from extract import MAX_BYTES, Unreadable
from inbox import Finder, Inbox, ago, human_size
from drift import review as drift_review
import harness
import intent
import speech
import tools as tools_mod
import uisettings
import voices as vx
from kg import StoryKG, tokens
from modes import MODES
from ideas import KINDS, Ideas
from quality import baseline_max_sentence, check, clean_output
from references import RefFinder
from settings import Settings

SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text()

PREF = re.compile(r"\b(i (prefer|like|love|hate|don'?t like|dislike)|always|never|too (long|flowery|wordy|formal)|remember (that|this))\b|^\s*(please )?(don'?t|do not|stop|avoid)\b", re.I)
EXCERPT_WORDS = 10


def excerpt(text: str, n: int = EXCERPT_WORDS) -> str:
    w = text.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


@dataclass
class Pending:
    kind: str  # "edit" | "append" | "confirm_target"
    text: str  # proposed text (edit/append) or the writer's original instruction (confirm_target)
    chapter: int
    paragraph_id: str | None = None
    scene: int | None = None
    data: dict | None = None  # kind "voice": {speaker, profile, draft}
    ask: str = ""  # the short question shown in big letters ("Add this to chapter 2?")


@dataclass
class Session:
    book: str
    mode: str = "narrator"
    chapter: int = 1
    focus: str | None = None
    history: list[dict] = field(default_factory=list)
    pending: Pending | None = None
    notices: list[str] = field(default_factory=list)  # background findings, delivered on the next reply
    voice_focus: str | None = None  # whose voice "make it deeper" / "his voice" refers to
    recent_speakers: list[str] = field(default_factory=list)
    last_ideas: dict | None = None  # the last random-idea request, so "more" can repeat it
    web_hits: dict = field(default_factory=dict)  # last web search results by id
    last_items: list = field(default_factory=list)  # the numbered random ideas just offered, so "number 2" can pick one
    last_items_kind: str = "idea"
    via: str = "typed"  # how the last message arrived: "voice" means the reply will be spoken aloud
    interrupted: bool = False  # the writer cut my last reply off by speaking over it
    found: list = field(default_factory=list)  # files the finder offered, best first (inbox.Found)
    found_i: int = 0  # which of them is being asked about
    summary: str = ""  # rolling summary of everything folded out of working memory (the full record lives in the journal)
    compacting: bool = False


@dataclass
class Reply:
    text: str
    mode: str
    chapter: int
    focus: str | None
    changed: list[str] = field(default_factory=list)
    pending: dict | None = None
    notes: list[str] = field(default_factory=list)
    speech: list[dict] = field(default_factory=list)  # what to say aloud, in order: [{"speaker", "text", "draft"?}]
    highlights: list[dict] = field(default_factory=list)  # passages the reply is about: [{"id", "chapter", "label"}]; the page lights them up
    ui: dict | None = None  # a new look for the screen (size, contrast…), when the writer asked for one
    ideas: list[str] = field(default_factory=list)  # random options the page shows as tap-to-use buttons
    open_item: dict | None = None  # a library file for the page to open: {"slug", "title"}


class Agent:
    def __init__(self, s: Settings, store: BookStore, llm: LLM, memory, kg: StoryKG, graph=None, check_drift: bool = False, voices=None, prefs=None, index=None, journal=None, library=None, addons=None):
        self.s, self.store, self.llm, self.memory, self.kg = s, store, llm, memory, kg
        self.graph, self.check_drift, self.voices, self.prefs, self.index = graph, check_drift, voices, prefs, index
        self.journal, self.library, self.addons = journal, library, addons  # long-term record, imported files, administrator addons
        self.inbox, self.stt = Inbox(store), None  # the drop folder; speech-to-text for audio files (set by main)
        self.web = None  # web.Web when the administrator set up Wikipedia / SearXNG (set by main)
        self._speaker_cache: dict[str, list[str | None]] = {}
        self.sessions: dict[str, Session] = {}
        self._bg: set[asyncio.Task] = set()
        self.ideas = Ideas()  # real randomness for names, places and twists (see ideas.py)
        self.refs = RefFinder(store, kg)  # which passages a piece of text is about
        self.toolbox = tools_mod.build(self)  # what the model may use (harness.py)
        self.traces: deque = deque(maxlen=60)  # recent turns that used tools, for the admin page
        self._baseline: tuple = (None, 55)

    # --- preferences the agent reads live ---------------------------------------------------------
    def _pref(self, key: str, default: str = "") -> str:
        return (self.prefs.get(key) if self.prefs else "") or default

    def _look(self) -> dict:
        try:
            return uisettings.clean(json.loads(self._pref("look", "{}")))
        except ValueError:
            return uisettings.clean(None)

    def _save_look(self, look: dict) -> None:
        if self.prefs:
            self.prefs.update({"look": json.dumps(look)})

    def _max_sentence(self) -> int:
        sig = self.kg._sig
        if self._baseline[0] != sig:
            self._baseline = (sig, baseline_max_sentence(self.kg._paras))
        return self._baseline[1]

    def _hl(self, sess: "Session", ids: list[str]) -> list[dict]:
        out = []
        for i in ids:
            try:
                p = self.store.find(sess.book, i)
            except BookError:
                continue
            out.append({"id": p.id, "chapter": p.chapter, "label": self._where(sess, p)})
        return out

    def session(self, sid: str = "", book: str | None = None) -> Session:
        """There is ONE conversation per book: the writer never starts a "new chat", and `sid` (kept for older clients) is ignored.
        A fresh server picks the conversation up from the journal: the rolling summary plus the latest turns."""
        book = book or self.s.default_book
        sess = self.sessions.get(book)
        if not sess:
            self.store.ensure_book(book)
            nums = self.store.chapter_numbers(book)
            sess = self.sessions[book] = Session(book=book, chapter=nums[0] if nums else 1)
            if self.journal:
                sess.summary = self.journal.summary(book)
                sess.history = [{"role": r["role"], "content": r["text"]} for r in self.journal.recent(book, 30) if r["role"] in ("user", "assistant")]
                sess.history = self._window(sess)
        return sess

    # ------------------------------------------------------------------
    async def handle(self, sid: str, user_text: str, book: str | None = None, via: str = "typed", interrupted: bool = False) -> Reply:
        """One turn. Everything said is written to the journal (so nothing is ever lost when working memory is trimmed)."""
        sess = self.session(sid, book)
        sess.via, sess.interrupted = via, interrupted
        reply = await self._handle(sid, user_text, book)
        if self.journal and reply.text and user_text.strip():
            sess = self.session(sid, book)
            self.journal.append(sess.book, "user", user_text, via)
            self.journal.append(sess.book, "assistant", reply.text[:1200])
            if self.index:
                self.index.schedule(sess.book)
            self._spawn(self._compact(sess))
        return reply

    def stt_hint(self, book: str | None = None) -> str:
        """Names the writer is likely to say, so the speech model spells them as the book does."""
        book = book or self.s.default_book
        try:
            b = self.store.bible(book)
            names = [n for kind in ("characters", "world") for n in list(b.entries(kind))[:25]]
        except Exception:
            names = []
        return ("Story about " + ", ".join(names) + ".") if names else ""

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _handle(self, sid: str, user_text: str, book: str | None = None) -> Reply:
        sess = self.session(sid, book)
        self.kg.refresh(sess.book)
        text = user_text.strip()
        if not text:
            return self._reply(sess, "I didn't catch that. Try again.")
        items, sess.last_items = sess.last_items, []  # only the very next thing the writer says can pick one of the numbered ideas
        if items and not sess.pending and (k := intent.pick_number(text)) and 1 <= k <= len(items):
            text = f"Let's use this {sess.last_items_kind}: {items[k - 1]}"

        if sess.pending and intent.undo_request(text) and sess.pending.kind not in ("voice",):  # "undo that" while a proposal waits: drop the proposal
            sess.pending = None
            return self._reply(sess, "Okay, I dropped that. Nothing in your book was changed.")
        # Answers to a pending proposal come first, so "okay" confirms rather than being discarded as noise.
        if sess.pending and sess.pending.kind == "file":
            done = await self._file_pending(sess, text)
            if done is not None:
                return done
        if sess.pending and sess.pending.kind == "voice":
            done = await self._voice_pending(sess, text)
            if done is not None:
                return done
        if sess.pending:
            kind, rest = intent.confirmation(text)
            if kind == "hold":
                return self._reply(sess, "Okay, I'll wait. Say yes to save it, or tell me what to change.")
            if kind == "yes" and not rest:
                return await self._accept(sess)
            if kind == "no" and not rest:
                self.memory.add(f"Writer rejected a proposed {sess.pending.kind}: '{sess.pending.text[:120]}'", "preference")
                sess.pending = None
                return self._reply(sess, "Okay, dropped it. What would you like instead?")
            if kind and rest and sess.pending.kind != "confirm_target":
                return await self._revise(sess, rest)  # "yes but make it shorter" / "no, make it longer": never save or drop blindly
            if not kind and (r := self._ui_reply(sess, text)):  # "bigger" while reviewing a proposal: change the screen, keep the proposal
                return r
            if not kind and sess.pending.kind != "voice" and not self._is_new_command(text):
                # a question, a look at the book, small talk, a vague remark, goodbye: none of these is an answer or a new job, so the change
                # keeps waiting (the card stays up) and I remind the writer, out loud, that it is still waiting
                keep = sess.pending
                if intent.goto_chapter(text) is not None or intent.step_direction(text) or intent.is_read_aloud(text):
                    sess.pending = None  # navigation runs as usual, then the change is put back
                    r = await self._navigate_keeping(sess, text)
                else:
                    r = await self._converse(sess, text)
                sess.pending = keep
                r.pending = {"kind": keep.kind, "text": keep.text, "paragraph_id": keep.paragraph_id, "ask": keep.ask}
                if keep.ask:
                    reminder = f"I am still holding this for you: {keep.ask} Say yes or no."
                    r.text = f"{r.text} {reminder}".strip()
                    r.speech = (r.speech or []) + [{"speaker": vx.ASSISTANT, "text": reminder}]
                return r
            if kind == "no" and rest:  # "no, I mean chapter 3 paragraph 2": the guess was wrong; keep the instruction, change the target
                original, sess.pending = sess.pending.text, None
                target, sure = await self._resolve(sess, rest)
                if target:
                    return await self._propose_edit(sess, original, target=target, sure=sure)
                text = rest
            else:
                sess.pending = None  # anything else is a new command; the unsaved proposal is abandoned, and we say so
                sess.notices.append("I dropped the unsaved proposal.")

        if not sess.pending and (draft := self._draft_to_rescue(sess, text)):  # "yes, add it" to a draft that was shown without a proposal
            sess.pending = Pending("append", draft, sess.chapter)
            sess.history[-1]["content"] = "[the draft paragraph was saved]"  # so a later "yes" can never save it a second time
            r = await self._accept(sess)
            r.text += " (That was the paragraph I just showed you.)"
            return r

        if intent.is_noise(text):  # silence that the speech model turned into "Thank you."
            return Reply(text="", mode=sess.mode, chapter=sess.chapter, focus=sess.focus)

        self._learn(text)

        if mode := intent.detect_mode(text):
            sess.mode = mode
            return self._reply(sess, "Switching to feedback mode. Tell me what you're working on." if mode == "therapist" else "Back to narrator mode.")

        if r := self._ui_reply(sess, text):  # "I can't see it": instant, no model call
            return r
        if (req := intent.ideas_request(text)) is not None:
            return self._ideas_reply(sess, req)
        if intent.undo_request(text):
            return self._undo_reply(sess)
        if (n := intent.new_chapter(text)) is not None:
            return self._new_chapter_reply(sess, n)
        if (ref := intent.delete_request(text)) is not None:
            return self._delete_reply(sess, ref, text)
        if (mv := intent.move_request(text)) is not None:
            return self._move_reply(sess, mv, text)
        if (what := intent.open_request(text)) and (r := self._open_reply(sess, what)):  # "open the harbour notes"
            return r
        if intent.file_request(text):  # "I just downloaded a new PDF, can you open it"
            return self._find_reply(sess, text)

        if self.voices and (cmd := intent.voice_command(text)):
            if (r := await self._voice_command(sess, text, cmd)) is not None:
                return r

        if (n := intent.goto_chapter(text)) is not None:
            if n in self.store.chapter_numbers(sess.book):
                sess.chapter, sess.focus = n, None
                return self._reply(sess, f"Chapter {n}.")
            nxt = max(self.store.chapter_numbers(sess.book), default=0) + 1
            return self._reply(sess, f"There is no chapter {n} yet." + (f" Say start chapter {n} and I will begin it." if n == nxt else ""))
        if d := intent.step_direction(text):
            return await self._step(sess, d)
        if intent.is_read_aloud(text):
            target = self._by_ref(sess, intent.parse_ref(text))
            if target:
                sess.chapter, sess.focus = target.chapter, target.id
                r = self._reply(sess, target.text, await self._speech_for(sess, target))
                r.highlights = self._hl(sess, [target.id])
                return r
            return self._reply(sess, "I couldn't find that paragraph.")

        if sess.mode == "narrator" and intent.is_edit(text):
            return await self._propose_edit(sess, text)
        if sess.mode == "narrator" and intent.is_draft(text):
            return await self._propose_draft(sess, text)
        return await self._converse(sess, text)

    # --- files: opening what is in the library, finding what was just downloaded, the inbox folder ---------------------------------
    def _finder(self, book: str) -> Finder:
        """The inbox folder first, then whatever folders the administrator shared (FILE_SEARCH_DIRS)."""
        return Finder([str(self.inbox.dir(book))] + list(self.s.search_dirs), MAX_BYTES)

    def _open_reply(self, sess: Session, what: str) -> Reply | None:
        if not self.library:
            return None
        want = set(tokens(what)) - {"file", "pdf", "document", "picture", "photo", "one"}
        best = None
        for m in self.library.items(sess.book):
            title = set(tokens(m["title"]))
            if want and want <= title or (title and title <= set(tokens(what))):
                best = best or m
        if not best:
            return None
        r = self._reply(sess, f"Opening {best['title']}.")
        r.open_item = {"slug": best["slug"], "title": best["title"]}
        return r

    _OFFER = re.compile(r"(?:add|save|put|use|keep|go ahead|shall i|should i|want me to|happy with)[^.?!]*\?\s*$", re.I)

    def _draft_to_rescue(self, sess: Session, text: str) -> str:
        """If the writer says yes and my last message was a draft paragraph that asked to be saved but never became a proposal, recover it."""
        kind, rest = intent.confirmation(text)
        if kind != "yes" or rest or not sess.history or sess.history[-1]["role"] != "assistant":
            return ""
        last = sess.history[-1]["content"].strip()
        if len(last) < 200 or not self._OFFER.search(last) or last.startswith("["):
            return ""
        body = re.sub(r"(?:\s*[^.?!]*\?)\s*$", "", last).strip()  # drop the closing question
        body = re.sub(r"^[^\n]{0,90}:\s*\n?", "", body, count=1).strip() if re.match(r"^[^\n]{0,90}:\s", body) else body  # drop "Here's the next paragraph:"
        return body if len(body) >= 120 else ""

    def _outline(self, book: str, limit: int = 45) -> str:
        """One short line per saved paragraph, so a new draft continues the story instead of retelling a scene that is already in the book."""
        try:
            paras = self.store.all_paragraphs(book)
        except Exception:
            return ""
        if not paras:
            return ""
        lines = [f"ch{p.chapter} p{i}: {' '.join(p.text.split()[:14])}…" for i, p in enumerate(paras, 1)][-limit:]
        return "THE BOOK SO FAR (one line per saved paragraph; continue forward, do not retell a scene that is already here):\n" + "\n".join(lines)

    def _book_state(self, sess: Session) -> str:
        nums = self.store.chapter_numbers(sess.book)
        counts = ", ".join(f"chapter {n}: {len(self.store.paragraphs(sess.book, n))} saved paragraphs" for n in nums[-8:]) or "no chapters yet"
        wait = (f"A proposal is waiting for the writer's yes or no ({sess.pending.ask or sess.pending.kind}). " if sess.pending else "Nothing is waiting for an answer. ")
        return (f"BOOK STATE (what is really saved): {counts}. The writer is looking at chapter {sess.chapter}. {wait}"
                "Text you write in chat is NOT in the book: only a confirmed proposal is saved. The app shows its own big YES and NO buttons and the writer can also answer by voice, "
                "so never tell them to press, click, tap or look at anything on the screen (they may not be able to see it).")

    def _ask_for(self, sess: Session, p: Pending) -> str:
        """The short question shown in big letters on the page (and read out)."""
        try:
            if p.kind == "edit":
                return f"Save this change to {self._where(sess, self.store.find(sess.book, p.paragraph_id))}?"
            if p.kind == "append":
                return f"Add this paragraph to chapter {p.chapter}?" if p.chapter in self.store.chapter_numbers(sess.book) else f"Start chapter {p.chapter} with this paragraph?"
            if p.kind == "wiki":
                d = p.data or {}
                if d.get("batch"):
                    return f"Save these {len(d['batch'])} notes to the wiki?"
                return "Save this to the book's premise?" if d.get("kind") == "premise" else f"Save this note under {d.get('name', 'the wiki')}?"
            if p.kind == "delete":
                return f"Delete {self._where(sess, self.store.find(sess.book, p.paragraph_id))}?"
            if p.kind == "move":
                return f"Move {self._where(sess, self.store.find(sess.book, p.paragraph_id))} to chapter {(p.data or {}).get('to')}?"
            if p.kind == "confirm_target":
                return "Is that the paragraph you mean?"
            if p.kind == "voice":
                return "Keep this voice?"
        except Exception:
            return ""
        return ""

    async def _navigate_keeping(self, sess: Session, text: str) -> Reply:
        return await self._handle("", text, sess.book)

    def _new_chapter_reply(self, sess: Session, n: int) -> Reply:
        nums = self.store.chapter_numbers(sess.book)
        nxt = max(nums, default=0) + 1
        target = n or nxt
        if target in nums:
            sess.chapter, sess.focus = target, None
            return self._reply(sess, f"Chapter {target} already exists. I have gone there.")
        if target != nxt:
            return self._reply(sess, f"That would skip a chapter. The next one is chapter {nxt}. Say start chapter {nxt}.")
        self.store.create_chapter(sess.book, target)
        sess.chapter, sess.focus = target, None
        if self.index:
            self.index.schedule(sess.book)
        return self._reply(sess, f"Chapter {target} is ready. Tell me what happens in it, or say write the first paragraph.")

    def _delete_reply(self, sess: Session, ref, text: str) -> Reply:
        target = self._by_ref(sess, ref)
        if not target:
            return self._reply(sess, "I couldn't find that paragraph.")
        sess.pending = Pending("delete", target.text, target.chapter, paragraph_id=target.id)
        sess.chapter, sess.focus = target.chapter, target.id
        r = self._reply(sess, f"Delete {self._where(sess, target)}? It begins: {excerpt(target.text, 12)} You can say undo afterwards.")
        r.highlights = self._hl(sess, [target.id])
        return r

    def _move_reply(self, sess: Session, mv, text: str) -> Reply:
        target = self._by_ref(sess, mv.ref)
        if not target:
            return self._reply(sess, "I couldn't find that paragraph.")
        nums = self.store.chapter_numbers(sess.book)
        to = mv.chapter or target.chapter
        if to not in nums and to != max(nums, default=0) + 1:
            return self._reply(sess, f"There is no chapter {to}, and it is not the next one. The next is chapter {max(nums, default=0) + 1}.")
        if to == target.chapter and mv.where == "end" and self.store.paragraphs(sess.book, to)[-1].id == target.id:
            return self._reply(sess, "It is already at the end of that chapter.")
        sess.pending = Pending("move", target.text, target.chapter, paragraph_id=target.id, data={"to": to, "where": mv.where})
        sess.chapter, sess.focus = target.chapter, target.id
        place = {"end": f"the end of chapter {to}", "start": f"the start of chapter {to}"}.get(mv.where) if isinstance(mv.where, str) else f"{mv.where[0]} paragraph {mv.where[1]} of chapter {to}"
        r = self._reply(sess, f"Move {self._where(sess, target)} to {place}? It begins: {excerpt(target.text, 12)}")
        r.highlights = self._hl(sess, [target.id])
        return r

    def _is_new_command(self, text: str) -> bool:
        """Is this a new job for me (so a waiting change is put aside), rather than a question, small talk or a look at the book?"""
        return bool(intent.is_edit(text) or intent.is_draft(text) or intent.delete_request(text) or intent.move_request(text) or intent.new_chapter(text) is not None
                    or intent.ideas_request(text) is not None or intent.file_request(text) or intent.undo_request(text) or intent.detect_mode(text)
                    or (self.voices and intent.voice_command(text)) or intent.open_request(text))

    def _commit_change(self, sess: Session) -> Reply:
        p, sess.pending = sess.pending, None
        try:
            if p.kind == "delete":
                self.store.delete_paragraph(sess.book, p.paragraph_id)
                text, changed = "Deleted. Say undo if you want it back.", []
            else:
                new = self.store.move_paragraph(sess.book, p.paragraph_id, (p.data or {})["to"], (p.data or {}).get("where", "end"))
                sess.chapter, sess.focus = new.chapter, new.id
                text, changed = f"Moved to the end of chapter {new.chapter}. Say undo to put it back.", [new.id]
        except BookError as e:
            return self._reply(sess, f"I couldn't do that: {e}")
        self.kg.refresh(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        r = self._reply(sess, text)
        r.changed = changed
        return r

    def _undo_reply(self, sess: Session) -> Reply:
        """Put the book back as it was before the last saved change."""
        try:
            done = self.store.undo_last(sess.book)
        except BookError as e:
            return self._reply(sess, str(e))
        if not done:
            return self._reply(sess, "There is nothing to undo.")
        sess.chapter, sess.focus = done["chapter"], done["paragraph"] if done["what"] in ("edit", "delete") else None
        self.kg.refresh(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        said = {"edit": "I took back the change I just saved", "append": "I took back the paragraph I just added", "delete": "I put back the paragraph I deleted",
                "move": "I moved the paragraph back where it was"}.get(done["what"], "I took back the last change")
        r = self._reply(sess, f"Done. {said}. Chapter {done['chapter']} is as it was before.")
        r.changed = [done["paragraph"]] if done["what"] in ("edit", "delete") and done["paragraph"] else []
        return r

    def _describe_found(self, f) -> str:
        return f"{f.name}, {human_size(f.size)}, in {f.where}, from {ago(time.time() - f.mtime)}"

    def _set_found_pending(self, sess: Session) -> str:
        f = sess.found[sess.found_i]
        sess.pending = Pending(kind="file", text=self._describe_found(f), chapter=sess.chapter, data={"root": f.root, "rel": f.rel, "name": f.name})
        return self._describe_found(f)

    def _find_reply(self, sess: Session, text: str) -> Reply:
        finder = self._finder(sess.book)
        found = finder.search(text, limit=5)
        if not found:
            shared = len(finder.roots) > 1
            return self._reply(sess, "I looked in " + ("your inbox folder and the folders shared with me" if shared else "your inbox folder") + " and could not find a matching file. "
                               + ("Tell me part of its name and I will look again." if shared else "Drop the file into the inbox folder, or press Import and choose it, and I will read it."))
        sess.found, sess.found_i = found, 0
        what = self._set_found_pending(sess)
        more = f" I have {len(found) - 1} other possible matches if this is not it." if len(found) > 1 else ""
        return self._reply(sess, f"I found {what}. Is that the one?{more}")

    async def _file_pending(self, sess: Session, text: str) -> Reply | None:
        """Yes adds the file and opens it; no moves on to the next match."""
        kind, _rest = intent.confirmation(text)
        if kind == "yes":
            return await self._commit_file(sess)
        if kind == "hold":
            return self._reply(sess, "Okay, I'll wait. Say yes to add the file, or no to try another.")
        if kind == "no":
            was, sess.pending = sess.pending, None
            if (was.data or {}).get("inbox"):
                return self._reply(sess, "Okay. I will leave it in the folder. Press Import any time if you want it added.")
            sess.found_i += 1
            if sess.found and sess.found_i < len(sess.found):
                what = self._set_found_pending(sess)
                return self._reply(sess, f"Okay. How about {what}? Is that the one?")
            sess.found = []
            return self._reply(sess, "Okay, I'll leave it. If you are looking for something else, tell me part of its name.")
        return None

    async def _commit_file(self, sess: Session) -> Reply:
        p, sess.pending = sess.pending, None
        d, book = p.data or {}, sess.book
        if not self.library:
            return self._reply(sess, "The library is not available.")
        try:
            if d.get("inbox"):
                items = [(self.inbox.path(book, n), n, True) for n in d["inbox"]]
            else:
                finder = self._finder(book)
                items = [(finder.resolve(d["root"], d["rel"]), d["name"], d["root"] == 0)]
        except (FileNotFoundError, OSError, IndexError):
            return self._reply(sess, "I cannot find that file any more.")
        msgs, first = [], None
        for path, name, from_inbox in items:
            try:
                if path.stat().st_size > MAX_BYTES:
                    raise Unreadable(f"{name} is too big. The limit is {MAX_BYTES // 1048576} MB.")
                meta = await ingest.add_file(book, name, path.read_bytes(), self.library, self.llm, self.stt, self.journal)
            except Unreadable as e:
                msgs.append(str(e))
                continue
            except OSError:
                msgs.append(f"I could not read {name}.")
                continue
            msgs.append(meta["message"])
            first = first or meta
            if from_inbox:
                self.inbox.done(book, path.name)
        if self.index:
            self.index.schedule(book)
        r = self._reply(sess, " ".join(msgs) + (" I have opened it for you." if first else ""))
        if first:
            r.open_item = {"slug": first["slug"], "title": first["title"]}
        return r

    def inbox_check(self, book: str | None = None) -> Reply | None:
        """Called by the page now and then: if new files have appeared in the inbox folder, ask whether to add them (once per file)."""
        sess = self.session("", book)
        if sess.pending:  # never interrupt a question that is waiting for an answer
            return None
        new = self.inbox.fresh(sess.book)
        if not new:
            return None
        names = [f["name"] for f in new][:5]
        sess.pending = Pending(kind="file", text=", ".join(names), chapter=sess.chapter, data={"inbox": names})
        what = names[0] if len(names) == 1 else f"{len(names)} new files, {', '.join(names[:3])}{'…' if len(names) > 3 else ''}"
        text = f"I see {'a new file' if len(names) == 1 else ''} in your inbox folder: {what}. Shall I add {'it' if len(names) == 1 else 'them'} to the library and open {'it' if len(names) == 1 else 'the first'}?"
        if self.journal:
            self.journal.append(sess.book, "assistant", text)
        return self._reply(sess, text)

    # --- the screen, and random ideas -----------------------------------------------------------------
    def _ui_reply(self, sess: Session, text: str) -> Reply | None:
        res = uisettings.parse(text, self._look())
        if not res:
            return None
        look, say = res
        self._save_look(look)
        r = self._reply(sess, say)
        r.ui = look
        return r

    def _setting_cultures(self, book: str) -> list[str]:
        """Name cultures that fit this book's own setting, read from its premise, wiki and opening pages."""
        try:
            b = self.store.bible(book)
            bits = [b.read("premise")] + [f"{e.name} {e.summary()}" for k in ("characters", "world") for e in list(b.entries(k).values())[:40]]
            bits += [p.text for p in self.store.all_paragraphs(book)[:12]]
        except Exception:
            return []
        from ideas import cultures_for
        return cultures_for(" ".join(bits)[:20000])

    def _ideas_reply(self, sess: Session, req: dict) -> Reply:
        if req.get("repeat"):
            req = sess.last_ideas or {"kind": "name", "n": 5, "opts": {}}
        sess.last_ideas = req
        items = self.ideas.generate(req["kind"], req["n"], fits=self._setting_cultures(sess.book), **req["opts"])
        label = KINDS[req["kind"]]["label"].lower()
        sess.last_items, sess.last_items_kind = (items if len(items) > 1 else []), KINDS[req["kind"]]["one"]
        if len(items) == 1:
            say = items[0]
        else:
            say = f"Here are {len(items)} {label}: " + ". ".join(f"Number {i}, {x}" for i, x in enumerate(items, 1)) + "."
        spoken = say + (" Say the number you like, or say more." if len(items) > 1 else " Say more for another.")
        self._remember(sess, "[asked for random " + label + "]", say)
        r = self._reply(sess, spoken)
        r.ideas = items
        return r

    # --- navigation ----------------------------------------------------
    async def _step(self, sess: Session, d: int) -> Reply:
        paras = self.store.paragraphs(sess.book, sess.chapter) if sess.chapter in self.store.chapter_numbers(sess.book) else []
        if not paras:
            return self._reply(sess, "This chapter is empty.")
        ids = [p.id for p in paras]
        if sess.focus in ids:
            i = ids.index(sess.focus) + d
            if not 0 <= i < len(ids):  # run off the end of the chapter: say so instead of repeating the same paragraph
                nxt = sess.chapter + d
                if nxt in self.store.chapter_numbers(sess.book):
                    sess.chapter, sess.focus = nxt, None
                    return self._reply(sess, f"That was the {'last' if d > 0 else 'first'} paragraph. Moving to chapter {nxt}.")
                return self._reply(sess, f"That is the {'end' if d > 0 else 'start'} of the book.")
        else:
            i = 0 if d > 0 else len(ids) - 1
        sess.focus = ids[i]
        r = self._reply(sess, paras[i].text, await self._speech_for(sess, paras[i]))
        r.highlights = self._hl(sess, [ids[i]])
        self._prefetch(sess, paras[i + 1 : i + 3])  # work out who speaks the next lines while this one is being read
        return r

    # --- target resolution -----------------------------------------------
    def _by_ref(self, sess: Session, ref: intent.Ref) -> Paragraph | None:
        if ref.para is None:
            return None
        return self.store.nth_paragraph(sess.book, ref.chapter or sess.chapter, ref.para, ref.scene)

    async def _resolve(self, sess: Session, text: str) -> tuple[Paragraph | None, bool]:
        """-> (paragraph, sure). sure=False means "this is my best guess; read it back before touching it".

        Order: an explicit reference ("chapter 3 paragraph 2"), then the focused paragraph, then word matching. When word matching is
        not confident, meaning-based search (the embedding index) is consulted; it can improve the *guess*, but the agent only acts
        without asking when both methods agree on the same paragraph."""
        ref = intent.parse_ref(text)
        if ref.explicit:
            return self._by_ref(sess, ref), True
        if sess.focus and re.search(r"\b(this|that|the current|it)\b", intent.norm(text)) and not self.kg.named_in(text):
            return self.store.find(sess.book, sess.focus), True
        query = intent.locator_query(text)  # describe-where words only; the instruction ("more sarcastic") must not steer the search
        hits = self.kg.search(query, limit=5, chapter=ref.chapter)
        if hits and self.kg.confident(hits, len(tokens(query))):
            return hits[0].para, True
        if self.index and self.index.status(sess.book)["ready"]:
            by_id = {p.id: p for p in self.store.all_paragraphs(sess.book)}
            only = (lambda pid: pid in by_id and by_id[pid].chapter == ref.chapter) if ref.chapter else None
            sem = await self.index.search(sess.book, query, k=5, chapter_of=only)
            if sem:
                score: dict[str, float] = {}
                for rank, h in enumerate(hits):
                    score[h.para.id] = score.get(h.para.id, 0) + 1 / (60 + rank)  # reciprocal rank fusion
                for rank, (pid, _) in enumerate(sem):
                    score[pid] = score.get(pid, 0) + 1 / (60 + rank)
                best = max(score, key=score.get)
                agree = bool(hits) and hits[0].para.id == sem[0][0]
                return by_id.get(best), agree
        if not hits:
            return None, False
        return hits[0].para, False

    def _where(self, sess: Session, para: Paragraph) -> str:
        """How the writer counts: the paragraph's position in the chapter, not its within-scene id suffix."""
        ids = [p.id for p in self.store.paragraphs(sess.book, para.chapter)]
        return f"chapter {para.chapter}, paragraph {ids.index(para.id) + 1}"

    # --- LLM helpers -----------------------------------------------------
    def _manuscript(self, sess: Session, target: Paragraph | None = None) -> str:
        """The actual text around the work: neighbours of the target, else the focus, else the end of the current chapter.
        Without this the model only ever sees the bible and cannot keep a rewrite consistent with what is on the page."""
        try:
            paras = self.store.paragraphs(sess.book, (target.chapter if target else sess.chapter))
        except BookError:
            return ""
        ids = [p.id for p in paras]
        anchor = target.id if target else (sess.focus if sess.focus in ids else None)
        if anchor:
            i = ids.index(anchor)
            before, after = paras[max(0, i - 2) : i], paras[i + 1 : i + 2]
            parts = [("PRECEDING", p.text) for p in before] + [("FOLLOWING", p.text) for p in after]
        else:
            parts = [("RECENT", p.text) for p in paras[-3:]]
        return "\n\n".join(f"[{k}] {v}" for k, v in parts)[:3000]

    async def _canon_by_meaning(self, sess: Session, text: str) -> str:
        """Bible entries that match the *meaning* of the request but are not named in it ("the old man with the luggage" -> the porter)."""
        if not (self.index and text.strip()):
            return ""
        try:
            if not self.index.status(sess.book)["ready"]:
                return ""
            hits = await self.index.search(sess.book, text, k=3, bible=True)
        except Exception:
            return ""
        b = self.store.bible(sess.book)
        have = {e.slug for k in ("characters", "threads", "world") for e in b.mentioned(k, text)}
        lines = []
        floor = max(0.2, hits[0][1] * 0.85) if hits else 1.0  # only entries nearly as close as the best match: unrelated text still scores ~0.3
        for key, score in hits:
            _, kind, slug = (key.split(":", 2) + [""])[:3] if key != "bible:premise" else ("", "", "")
            if not kind or slug in have or score < floor:
                continue
            e = b.entries(kind).get(next((n for n, x in b.entries(kind).items() if x.slug == slug), ""))
            if e:
                lines.append(f"{kind[:-1].upper() if kind != 'world' else 'PLACE'} {e.name}:\n{e.summary()}")
        return ("ALSO RELEVANT (matched by meaning):\n" + "\n\n".join(lines[:3])) if lines else ""

    def _sparks(self, sess: Session, text: str, drafting: bool = False) -> list[str]:
        """Random seeds that keep drafts out of the model's usual groove. How many depends on the writer's Surprise setting."""
        level = self._pref("surprise", "low")
        if level == "off" or sess.mode != "narrator":
            return []
        if level == "high" or drafting or intent.is_creative_ask(text):
            return self.ideas.sparks(3 if level == "high" else 2)
        return []

    def _temperature(self) -> float:
        return {"off": 0.7, "low": 0.8, "high": 1.0}.get(self._pref("surprise", "low"), 0.8)

    def _context(self, sess: Session, text: str, extra: str = "", target: Paragraph | None = None, canon: str = "",
                 sparks: list[str] | None = None, tools: bool = False, recall: str = "") -> list[dict]:
        prefs = self.memory.search(text, limit=6)
        sys = [SYSTEM, MODES[sess.mode], self._book_state(sess)]
        if sess.mode == "narrator" and (outline := self._outline(sess.book)):
            sys.append(outline)
        age = (self.prefs.get("writer_age") if self.prefs else "") or self.s.writer_age
        if age:
            sys.append(f"The writer is {age}; let that subtly shape idiom and description density. Never mention it.")
        if prefs:
            sys.append("WRITER PREFERENCES / MEMORY:\n- " + "\n- ".join(prefs))
        bible = self.store.bible(sess.book).context_for(text + " " + extra)
        if bible:
            sys.append("STORY BIBLE (canon; never contradict):\n" + bible)
        if canon:
            sys.append(canon)
        if sess.via == "voice":  # the voice and the thinking are one agent: what I write is what will be said aloud
            sys.append("THE WRITER IS TALKING TO YOU OUT LOUD and your reply will be spoken. Answer in one to three short, natural sentences. No lists, no markdown, "
                       "no headings. If you must give several options, say two or three in a sentence. Ask at most one question." + (
                           " They just cut off your last reply by speaking over it: do not repeat it, answer what they said now." if sess.interrupted else ""))
        if sess.summary:
            sys.append("EARLIER IN THIS LONG CONVERSATION (a summary; the full record can be searched with the recall tool):\n" + sess.summary)
        if recall:
            sys.append(recall)
        if sparks:
            sys.append("OPTIONAL SPARKS (random seeds; use at most one, only if it makes the writing better, and ignore the rest):\n- " + "\n- ".join(sparks))
        if tools:
            sys.append(harness.TOOLS_PROMPT.replace("{tools}", self.toolbox.describe()))
            skills = tools_mod.skills_prompt(self)
            if skills:
                sys.append(skills)  # replace, not format: the prompt has literal braces
        rel = self.graph.context_for(sess.book, text + " " + extra) if self.graph else ""
        if rel:
            sys.append("STORY GRAPH (EXTRACTED = stated in the text, INFERRED = a guess):\n" + rel)
        ms = self._manuscript(sess, target)
        if ms:
            sys.append("MANUSCRIPT AROUND THE WORK (for continuity; do not repeat it):\n" + ms)
        return [{"role": "system", "content": "\n\n".join(sys)}] + self._window(sess)

    async def _gated(self, messages: list[dict], ctx: "harness.Ctx | None" = None, tools: bool = False) -> tuple[str, list[str]]:
        strict, limit = self._pref("strictness", "gentle"), self._max_sentence()
        if tools and ctx is not None:
            raw, _ = await harness.run(self.llm, messages, self.toolbox, ctx, max_steps=2, temperature=self._temperature())
        else:
            raw = await self.llm.chat(messages)
        out = clean_output(raw)
        issues = check(out, strict, limit)
        for _ in range(self.s.max_rewrite_attempts):
            if not issues:
                break
            out = clean_output(await self.llm.chat(
                messages + [{"role": "assistant", "content": out},
                            {"role": "user", "content": "Rewrite it fixing these problems, output only the text:\n- " + "\n- ".join(issues)}]
            ))
            issues = check(out, strict, limit)
        return out, issues

    def _attach(self, sess: Session, reply: Reply, ctx: "harness.Ctx", text: str, heard: str = "") -> Reply:
        """Whatever a turn's tools and the reply's own words point at: passages to highlight, a new look, tap-to-use ideas."""
        ids = list(dict.fromkeys(ctx.highlights + self.refs.find(sess.book, text)))[:5]
        reply.highlights = self._hl(sess, ids)
        if ctx.notes:
            reply.notes += ctx.notes
        if ctx.open_item:
            reply.open_item = ctx.open_item
        if ctx.ui:
            self._save_look(ctx.ui)
            reply.ui = ctx.ui
        if ctx.ideas:
            reply.ideas = ctx.ideas
        if ctx.calls:
            self.traces.append({"at": int(time.time()), "tools": ctx.calls, "highlighted": len(reply.highlights)})
        return reply

    # --- working memory: a window of recent turns, a rolling summary, and the journal behind them ----------------------------
    def _window(self, sess: Session) -> list[dict]:
        """The newest turns that fit the context budget (and begin with the writer, as chat models expect)."""
        out, size = [], 0
        for m in reversed(sess.history):
            if size + len(m["content"]) > self.s.context_chars and len(out) >= 2:
                break
            out.insert(0, m)
            size += len(m["content"])
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    async def _compact(self, sess: Session, force: bool = False) -> None:
        """Fold the oldest turns into the rolling summary once working memory outgrows its budget. The writer never notices: the turns stay
        in the journal, and the recall tool and meaning-search can bring any of them back."""
        budget = self.s.context_chars
        size = sum(len(m["content"]) for m in sess.history)
        if sess.compacting or not sess.history or (not force and size <= budget * 1.4):
            return
        sess.compacting = True
        try:
            count = len(sess.history)
            keep: list[dict] = []
            if not force:
                used = 0
                for m in reversed(sess.history):
                    if used + len(m["content"]) > budget * 0.6 and len(keep) >= 2:
                        break
                    keep.insert(0, m)
                    used += len(m["content"])
                while keep and keep[0]["role"] != "user":
                    keep.pop(0)
            old = sess.history[: count - len(keep)]
            if not old:
                return
            sess.summary = await self._summarise(sess.summary, old)
            if self.journal:
                self.journal.set_summary(sess.book, sess.summary)
            sess.history = keep + sess.history[count:]  # turns that arrived while the summary was being written stay
        finally:
            sess.compacting = False

    async def _summarise(self, previous: str, turns: list[dict]) -> str:
        convo = "\n".join(f"{'Writer' if m['role'] == 'user' else 'Co-writer'}: {m['content'][:700]}" for m in turns)
        try:
            return (await self.llm.chat([
                {"role": "system", "content": "You keep the running memory of a long conversation between a writer and their AI co-writer. Reply with only the updated summary."},
                {"role": "user", "content": f"CURRENT SUMMARY:\n{previous or '(none yet)'}\n\nNEW CONVERSATION TO FOLD IN:\n{convo}\n\n"
                 "Write the updated summary in under 250 words. Keep: decisions made, facts about the story and its people, the writer's preferences, "
                 "ideas still open, and what they were working on. Drop small talk."}], temperature=0.2, max_tokens=500)).strip()[:3000]
        except Exception:  # no model right now: keep something rather than let memory grow without bound
            bits = [m["content"][:110].replace("\n", " ") for m in turns if m["role"] == "user"]
            return ((previous + " | " if previous else "") + "Earlier the writer said: " + "; ".join(bits))[-1800:]

    async def clear_chat(self, book: str | None = None) -> None:
        """The writer's Clear button: empties the screen and working memory, but folds it into the summary first, so the agent still remembers."""
        sess = self.session("", book)
        await self._compact(sess, force=True)
        sess.history, sess.pending = [], None
        if self.journal:
            self.journal.clear_screen(sess.book)

    _MEMORY_ASK = re.compile(r"\b(remember|remind|earlier|before|last (time|week|month|year)|ages ago|a while ago|back when|i (said|mentioned|told)|you (said|mentioned|told)|forgot|forgotten|what was (that|the)|did (i|we)|we (talked|discussed|decided))\b", re.I)

    async def _recall(self, sess: Session, text: str) -> str:
        """Older conversation and library passages that are close in meaning to what the writer just said ("that old lighthouse I described")."""
        if not (self.journal or self.library) or len(text.split()) < 4:
            return ""
        b, found = sess.book, []
        if self.index and self.index.status(b)["ready"]:
            try:
                hits = await self.index.search(b, text, k=6, scope="mem|lib")
            except Exception:
                hits = []
            floor = max(0.3, hits[0][1] * 0.88) if hits else 1.0
            docs = {**(self.journal.chunks(b) if self.journal else {}), **(self.library.chunks(b) if self.library else {})}
            found = [docs[k] for k, s in hits if s >= floor and k in docs][:4]
        elif self._MEMORY_ASK.search(text):
            found = [x for _, x in (self.journal.search_words(b, text, 3) if self.journal else []) + (self.library.search_words(b, text, 2) if self.library else [])]
        recent = [m["content"][:60] for m in sess.history[-6:]]
        found = [f for f in found if not any(r and r in f for r in recent)]  # skip what is already in the visible conversation
        if not found:
            return ""
        return "FROM EARLIER CONVERSATIONS AND THE LIBRARY (may or may not matter; use only if it does, and say where it came from):\n- " + "\n- ".join(f[:450] for f in found)

    def _remember(self, sess: Session, user: str, assistant: str) -> None:
        sess.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        del sess.history[:-60]  # compaction (below) normally trims far sooner; this is only a hard ceiling

    # --- flows -----------------------------------------------------------
    async def _converse(self, sess: Session, text: str) -> Reply:
        sess.history.append({"role": "user", "content": text})
        ctx = harness.Ctx(sess=sess)
        canon = await self._canon_by_meaning(sess, text)
        msgs = self._context(sess, text, canon=canon, sparks=self._sparks(sess, text), tools=True, recall=await self._recall(sess, text))
        out, _ = await harness.run(self.llm, msgs, self.toolbox, ctx, max_steps=3, temperature=self._temperature())
        out = out or "I'm not sure what to say to that. Could you tell me a little more?"
        if ctx.drafted and sess.pending and sess.pending.kind == "append":  # the model proposed a paragraph: show it, hear it, and ask
            tail = f"Should I add it to chapter {sess.pending.chapter}?" if sess.pending.chapter in self.store.chapter_numbers(sess.book) else f"Should I start chapter {sess.pending.chapter} with it?"
            lead = (out.strip() + " ") if out.strip() and len(out.strip()) <= 220 and out.strip() != "I'm not sure what to say to that. Could you tell me a little more?" else ""
            sess.history.append({"role": "assistant", "content": "[proposed a new paragraph]"})
            reply = self._reply(sess, f"{lead}Here's a draft: {ctx.drafted} {tail}", await self._speech_text(sess, ctx.drafted, f"{lead}Here's a draft:", tail))
            return self._attach(sess, reply, ctx, "", text)
        sess.history.append({"role": "assistant", "content": out})
        return self._attach(sess, self._reply(sess, out), ctx, out, text)

    async def _propose_edit(self, sess: Session, text: str, target: Paragraph | None = None, sure: bool = True) -> Reply:
        if target is None:
            target, sure = await self._resolve(sess, text)
        if not target:
            return self._reply(sess, "Which part? Say the chapter and paragraph, or describe the scene.")
        where = self._where(sess, target)
        if not sure:  # a guess: read it back and wait, rather than rewriting something the writer didn't mean
            sess.chapter, sess.focus = target.chapter, target.id
            sess.pending = Pending("confirm_target", text, target.chapter, paragraph_id=target.id)
            r = self._reply(sess, f"Do you mean {where}, starting: {excerpt(target.text)} Say yes to go ahead, or tell me which part.")
            r.highlights = self._hl(sess, [target.id])
            return r
        msgs = self._context(sess, text, target.text, target) + [
            {"role": "user", "content": f"Rewrite ONLY this paragraph.\nInstruction: {text}\n\nORIGINAL:\n{target.text}\n\nOutput only the new paragraph."}
        ]
        new, issues = await self._gated(msgs)
        sess.chapter, sess.focus = target.chapter, target.id
        sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id)
        self._remember(sess, text, f"[proposed a rewrite of {where}]")
        lead = f"Changing {where}, which starts: {excerpt(target.text, 6)} Here is the new version:"
        reply = self._reply(sess, f"{lead} {new} Should I save it?", await self._speech_text(sess, new, lead, "Should I save it?", target))
        reply.notes += [f"Style note: {i}" for i in issues]
        reply.highlights = self._hl(sess, [target.id])
        return reply

    async def _propose_draft(self, sess: Session, text: str) -> Reply:
        asked = intent.parse_ref(text).chapter
        nums = self.store.chapter_numbers(sess.book)
        if asked and (asked in nums or asked == (max(nums, default=0) + 1)):
            sess.chapter = asked  # "write paragraph 2 of chapter 3": the chapter the writer named, not the one on screen (a new chapter is made on save)
        if sess.chapter not in nums and sess.chapter != max(nums, default=0) + 1:
            sess.chapter = max(nums, default=1)
        recent = "\n\n".join(p.text for p in self.store.paragraphs(sess.book, sess.chapter)[-4:]) if sess.chapter in self.store.chapter_numbers(sess.book) else ""
        ctx = harness.Ctx(sess=sess)
        canon = await self._canon_by_meaning(sess, text)
        msgs = self._context(sess, text, recent, canon=canon, sparks=self._sparks(sess, text, drafting=True), tools=True) + [
            {"role": "user", "content": f"{text}\n\nStory so far in this chapter (most recent):\n{recent or '(nothing yet)'}\n\nWrite ONE new paragraph of prose. Output only the paragraph."}
        ]
        new, issues = await self._gated(msgs, ctx, tools=True)
        if ctx.drafted:  # the model already proposed its paragraph through propose_append: that is the draft
            new = ctx.drafted
        sess.pending = Pending("append", new, sess.chapter)
        self._remember(sess, text, "[proposed a new paragraph]")
        tail = f"Should I add it to chapter {sess.chapter}?" if sess.chapter in self.store.chapter_numbers(sess.book) else f"Should I start chapter {sess.chapter} with it?"
        reply = self._reply(sess, f"Here's a draft: {new} {tail}", await self._speech_text(sess, new, "Here's a draft:", tail))
        reply.notes += [f"Style note: {i}" for i in issues]
        return self._attach(sess, reply, ctx, "", text)

    async def _revise(self, sess: Session, instruction: str) -> Reply:
        """The writer answered a proposal with changes ("yes but shorter", "no, make it longer"): redo it, don't save or drop."""
        p = sess.pending
        if p.kind in ("family", "page"):
            sess.pending = None
            return self._reply(sess, "Okay, I won't add it yet. Ask me again with the person and how many generations you want.")
        if p.kind == "wiki" and (p.data or {}).get("batch"):
            return self._reply(sess, "Say yes to save all of them, or no to drop them. To change one, say no and tell me which.")
        if p.kind == "wiki":
            new = (await self.llm.chat([{"role": "system", "content": "Rewrite the note for a story wiki. Plain facts, at most 80 words. Output only the note."},
                                        {"role": "user", "content": f"NOTE:\n{p.text}\n\nChange requested: {instruction}"}], temperature=0.3, max_tokens=250)).strip()
            p.text = new or p.text
            return self._reply(sess, f"Revised: {p.text} Should I save it to the wiki?")
        if p.kind == "edit":
            orig = self.store.find(sess.book, p.paragraph_id)
            msgs = self._context(sess, instruction, orig.text, orig) + [{"role": "user", "content":
                f"Rewrite ONLY this paragraph.\n\nORIGINAL:\n{orig.text}\n\nYOUR PREVIOUS ATTEMPT (not saved):\n{p.text}\n\n"
                f"The writer now says: {instruction}\nRevise the previous attempt accordingly. Output only the new paragraph."}]
        else:
            msgs = self._context(sess, instruction, p.text) + [{"role": "user", "content":
                f"YOUR PREVIOUS DRAFT (not saved):\n{p.text}\n\nThe writer now says: {instruction}\nRevise the draft accordingly. Output only the paragraph."}]
        new, issues = await self._gated(msgs)
        p.text = new
        self._remember(sess, instruction, "[revised the proposal]")
        reply = self._reply(sess, f"Revised: {new} Should I save it?", await self._speech_text(sess, new, "Revised:", "Should I save it?"))
        reply.notes += [f"Style note: {i}" for i in issues]
        return reply

    async def _accept(self, sess: Session) -> Reply:
        if sess.pending.kind == "page":
            return self._commit_page(sess)
        if sess.pending.kind in ("delete", "move"):
            return self._commit_change(sess)
        if sess.pending.kind == "wiki":
            return self._commit_wiki(sess)
        if sess.pending.kind == "family":
            p, sess.pending = sess.pending, None
            res = bible_import.commit(self.store, self.library, sess.book, p.data["plan"])
            if self.index:
                self.index.schedule(sess.book)
            return self._reply(sess, f"Done. I added {res['characters_added']} characters" + (f", extended {res['characters_extended']} existing ones" if res["characters_extended"] else "")
                               + f", {res['places_added']} places and {res['timeline_entries']} timeline entries to the story wiki." + (f" {res['aliased']} living relatives were given made-up names." if res["aliased"] else ""))
        if sess.pending.kind == "confirm_target":
            p, sess.pending = sess.pending, None
            return await self._propose_edit(sess, p.text, target=self.store.find(sess.book, p.paragraph_id), sure=True)
        return self._commit(sess)

    def _commit_page(self, sess: Session) -> Reply:
        """Keep a web page's text in the library (with its address), after the writer said yes."""
        p, sess.pending = sess.pending, None
        d = p.data or {}
        if not self.library:
            return self._reply(sess, "The library is not available.")
        text = f"Source: {d.get('url', '')}\n\n{d.get('text', '')}"
        meta = self.library.add(sess.book, f"{d.get('title', 'page')}.txt", text.encode("utf-8"), d.get("title") or "Web page", "document", text)
        if self.journal:
            self.journal.append(sess.book, "note", f"The writer kept a web page in the library: {meta['title']} ({d.get('url', '')}).")
        if self.index:
            self.index.schedule(sess.book)
        r = self._reply(sess, f"Saved {meta['title']} to the library, with its web address.")
        r.open_item = {"slug": meta["slug"], "title": meta["title"]}
        return r

    @staticmethod
    def _save_note(bible, kind: str, name: str, note: str) -> str:
        """Append one note to the wiki (never overwriting). -> where it went, for the reply."""
        if kind == "premise":
            cur = bible.read("premise").rstrip()
            bible.write_doc("premise", (cur if cur.strip("# \nPremise") else "# Premise") + "\n\n" + note)
            return "the book's premise"
        old = next((e for e in bible.entries(kind).values() if e.name.lower() == name.lower()), None)
        body = ((old.body.strip() + "\n\n") if old and old.body.strip() else "") + note
        bible.upsert(kind, old.name if old else name, None, body)
        return name

    def _commit_wiki(self, sess: Session) -> Reply:
        """Save the note (or the several notes) the model proposed for the story wiki, after the writer said yes. Notes are appended; nothing is overwritten."""
        p, sess.pending = sess.pending, None
        d = p.data or {}
        batch = d.get("batch") or [{"kind": d.get("kind"), "name": d.get("name", ""), "note": p.text}]
        saved = []
        try:
            bible = self.store.bible(sess.book)
            for item in batch:
                saved.append(self._save_note(bible, item["kind"], item["name"], item["note"]))
        except Exception as e:
            return self._reply(sess, f"I couldn't save that: {e}" + (f" (I did save: {', '.join(saved)}.)" if saved else ""))
        if self.index:
            self.index.schedule(sess.book)
        return self._reply(sess, f"Saved to the wiki: {', '.join(saved)}." if len(saved) > 1 else f"Saved to {saved[0] if saved[0].startswith('the book') else 'the wiki under ' + saved[0]}.")

    def _commit(self, sess: Session) -> Reply:
        p, sess.pending = sess.pending, None
        try:
            if p.kind == "edit":
                para = self.store.replace_paragraph(sess.book, p.paragraph_id, p.text)
            else:
                if p.chapter not in self.store.chapter_numbers(sess.book):
                    self.store.create_chapter(sess.book, p.chapter)
                para = self.store.append_paragraph(sess.book, p.chapter, p.text, p.scene)
        except BookError as e:
            return self._reply(sess, f"I couldn't save that: {e}")
        sess.focus = para.id
        self.kg.refresh(sess.book)
        self._after_save(sess, para)
        r = self._reply(sess, "Saved.")
        r.changed = [para.id]
        return r

    # --- speech: who says what -----------------------------------------------------
    async def _segments(self, sess: Session, text: str, before: list[Paragraph]) -> list[dict]:
        """Narration and dialogue split by speaker. `before` = the preceding paragraphs, which tell us who was talking."""
        chars = self.store.bible(sess.book).characters()
        forms = speech.forms_for(chars)
        genders = {n: vx.guess_traits(n, e.state, e.body)["gender"] for n, e in chars.items()}
        recent: list[str] = []
        for pr in before[-12:]:  # speakers chain through untagged lines, so look back far enough to find the last tagged ones
            _, recent = speech.attribute(pr.text, forms, recent, genders)
        segs, _ = speech.attribute(text, forms, recent, genders)
        if self.llm and self.s.speech_llm and chars and any(s.quoted and not s.resolved for s in segs):
            key = speech.cache_key(before[-1].text if before else "", text)
            if key not in self._speaker_cache:
                try:
                    self._speaker_cache[key] = await asyncio.wait_for(speech.llm_attribute(self.llm, list(chars), before[-1].text if before else "", segs, text), 3)
                except Exception:
                    self._speaker_cache[key] = None
            names = self._speaker_cache[key]
            if names:
                it = iter(names)
                for s in segs:
                    if s.quoted:
                        who = next(it)
                        if not s.resolved and who:
                            s.speaker, s.resolved = who, True
        return [{"speaker": s.speaker, "text": s.text} for s in segs]

    def _prefetch(self, sess: Session, paras: list[Paragraph]) -> None:
        """Warm the speaker cache for upcoming paragraphs so "next paragraph" never waits on a model."""
        async def go():
            for p in paras:
                try:
                    await self._speech_for(sess, p)
                except Exception:
                    return
        if paras and self.llm and self.s.speech_llm:
            task = asyncio.create_task(go())
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    async def _speech_for(self, sess: Session, para: Paragraph) -> list[dict]:
        paras = self.store.paragraphs(sess.book, para.chapter)
        i = next((k for k, p in enumerate(paras) if p.id == para.id), 0)
        return await self._segments(sess, para.text, paras[max(0, i - 12) : i])

    async def _speech_text(self, sess: Session, body: str, lead: str, tail: str, target: Paragraph | None = None) -> list[dict]:
        try:
            paras = self.store.paragraphs(sess.book, target.chapter if target else sess.chapter)
        except BookError:
            paras = []
        if target:
            i = next((k for k, p in enumerate(paras) if p.id == target.id), len(paras))
            before = paras[max(0, i - 12) : i]
        else:
            before = paras[-12:]
        return [{"speaker": vx.ASSISTANT, "text": lead}, *await self._segments(sess, body, before), {"speaker": vx.ASSISTANT, "text": tail}]

    # --- voices: auditioning and editing with the writer ---------------------------------
    def _label(self, speaker: str) -> str:
        return {vx.NARRATOR: "the narrator", vx.ASSISTANT: "my voice"}.get(speaker, speaker)

    def _voice_target(self, sess: Session, text: str) -> str | None:
        n = intent.norm(text)
        if re.search(r"\b(narrator|narration|storyteller)\b", n):
            return vx.NARRATOR
        if re.search(r"\b(your|you|assistant)\b", n) and re.search(r"\b(voice|sound|speak|talk)\b", n):
            return vx.ASSISTANT
        if names := self.kg.named_in(text):
            return names[0]
        if sess.voice_focus and re.search(r"\b(his|her|their|that|this|him|it|the)\b.*\b(voice|sound)\b|\bvoice\b|\b(deeper|lower|higher|slower|faster|softer|older|younger|warmer|colder|drier)\b", n):
            return sess.voice_focus
        return None

    def _sample_line(self, sess: Session, speaker: str) -> str:
        if speaker == vx.ASSISTANT:
            return "Hello. This is how I will sound when we talk."
        if speaker == vx.NARRATOR:
            paras = self.store.all_paragraphs(sess.book)
            first = next((p.text for p in paras if not speech.split_quotes(p.text)[0][1]), "")
            sentence = re.split(r"(?<=[.!?])\s+", first)[0] if first else ""
            return sentence if 4 <= len(sentence.split()) <= 35 else "The evening was quiet, and the road ahead was long."
        chars = self.store.bible(sess.book).characters()
        forms = speech.forms_for(chars)
        paras = {p.id: p for p in self.store.all_paragraphs(sess.book)}
        for pid in sorted(self.kg.mentions.get(speaker, ()), key=lambda i: list(paras).index(i) if i in paras else 0)[:15]:
            segs, _ = speech.attribute(paras[pid].text, forms)
            for s in segs:
                words = len(s.text.split())
                line = s.text.strip("“”\" ")
                if s.quoted and s.speaker == speaker and 5 <= words <= 30 and line[:1].isupper() and line[-1:] in ".!?,":  # a whole line, not a mid-sentence fragment
                    return line.rstrip(",") + ("." if line.endswith(",") else "")
        return "I had not expected to see you here tonight."

    def _taken(self, sess: Session, tts) -> dict[str, str]:
        return {who: pr["pinned"].get(tts.id) or pr["assigned"].get(tts.id) for who, pr in self.voices.all_profiles(sess.book).items()}

    def _audition(self, sess: Session, speaker: str, prof: dict, said: str) -> Reply:
        tts = self.voices.primary()
        label = self._label(speaker)
        did = self.voices.new_draft(speaker, prof)
        sess.pending = Pending("voice", label, sess.chapter, data={"speaker": speaker, "profile": prof, "draft": did})
        sess.voice_focus = speaker
        sample = self._sample_line(sess, speaker)
        lead = f"Here is {label}." + (f" {said}" if said else "")
        tail = "Say keep it, or tell me what to change. For example deeper, slower, more sarcastic, or try another."
        heard = self.voices.resolve(sess.book, speaker, tts, did)  # the voice actually used for this audition
        text = f"{lead} ({vx.describe(prof, tts, heard.voice)}) “{sample}” {tail}"
        return self._reply(sess, text, [{"speaker": vx.ASSISTANT, "text": lead}, {"speaker": speaker, "text": sample, "draft": did}, {"speaker": vx.ASSISTANT, "text": tail}])

    async def _adjust(self, speaker: str, prof: dict, feedback: str, tts) -> tuple[dict, str] | None:
        n = intent.norm(feedback)
        for v in vx.parse_pool(tts):  # the writer named a voice: "use onyx for the narrator"
            if re.search(rf"\b{re.escape(intent.norm(v.name))}\b", n):
                pinned = vx.pin_voice(prof, tts, v.name)
                if pinned:
                    return pinned, f"Using {v.name}."
        new, said = vx.heuristic_edit(prof, feedback)
        words = vx.residual(feedback, set(intent.norm(speaker).split()) | {"narrator", "assistant"} | {w for n in self.kg.named_in(feedback) for w in intent.norm(n).split()})
        if said and not (words and self.llm):  # every word was understood by the rules: no need to wait for a model
            return new, "Made it " + ", ".join(said) + "."
        if self.llm:  # open-ended feedback ("sound like a tired professor"): let the model translate it
            pool = ", ".join(f"{v.name} ({'/'.join(sorted(v.tags)) or 'untagged'})" for v in vx.parse_pool(tts))
            user = (f"Available voices: {pool or tts.voice or 'default only'}\nCurrent: voice={prof['pinned'].get(tts.id) or prof['assigned'].get(tts.id)}, "
                    f"traits={prof['traits']}, speed={prof['speed']}, style={prof['style']!r}\nSpeaker: {speaker}\nWriter's feedback: {feedback}")
            try:
                raw = await asyncio.wait_for(self.llm.chat([{"role": "system", "content": vx.EDIT_PROMPT}, {"role": "user", "content": user}], temperature=0.2, max_tokens=250), 12)
                got = vx.parse_llm_edit(raw, prof, tts)
            except Exception:
                got = None
            if got and (got[0]["traits"], got[0]["speed"], got[0]["style"], got[0]["pinned"]) != (prof["traits"], prof["speed"], prof["style"], prof["pinned"]):
                return got[0], got[1] or "Adjusted."
        if said:
            return new, "Made it " + ", ".join(said) + "."
        return None

    async def _voice_command(self, sess: Session, text: str, cmd: dict) -> Reply | None:
        tts = self.voices.primary()
        action = cmd["action"]
        if action == "list":
            pool = vx.parse_pool(tts) if tts else []
            if not pool:
                return self._reply(sess, "I only know the default voice. Add your voices in the admin menu, under text-to-speech, to give characters different ones.")
            return self._reply(sess, "Available voices: " + ", ".join(v.name + (f" ({'/'.join(sorted(v.tags))})" if v.tags else "") for v in pool) + ".")
        target = self._voice_target(sess, text)
        if target is None:
            if action == "audition":
                return None  # "play the next paragraph": not about a voice after all
            return self._reply(sess, "Whose voice? Say a character's name, the narrator, or my voice.")
        if tts is None:
            return self._reply(sess, "No text-to-speech is set up yet, so I can't play voices. Add one in the admin menu.")
        label = self._label(target)
        if action == "reset":
            self.voices.forget(sess.book, target)
            return self._reply(sess, f"Forgot {label}'s voice. I'll pick a fresh one the next time it's needed.")
        prof = self.voices.profile(sess.book, target)
        if action == "audition":
            return self._audition(sess, target, prof, "")
        if action == "another":
            return self._audition(sess, target, vx.next_voice(prof, tts, self._taken(sess, tts), target), "Trying another.")
        got = await self._adjust(target, prof, cmd.get("feedback", text), tts)
        if not got:
            return self._reply(sess, f"I'm not sure what to change about {label}. Try deeper, slower, more sarcastic, or name a voice like {(vx.parse_pool(tts) or [None])[0].name if vx.parse_pool(tts) else 'alloy'}.")
        return self._audition(sess, target, got[0], got[1])

    async def _voice_pending(self, sess: Session, text: str) -> Reply | None:
        """The writer is auditioning a voice. Returns the reply, or None to drop the draft and handle `text` normally."""
        d = sess.pending.data
        speaker, prof, label = d["speaker"], d["profile"], self._label(d["speaker"])
        kind, rest = intent.confirmation(text)
        if kind == "hold":
            return self._reply(sess, "Okay. Say keep it when you're happy, or tell me what to change.")
        if kind == "yes" and not rest:
            self.voices.freeze(sess.book, speaker, d["draft"], auto=False)
            sess.pending = None
            return self._reply(sess, f"Saved. I'll use that for {label} from now on.")
        if kind == "no" and not rest:
            sess.pending = None
            return self._reply(sess, f"Okay, I'll leave {label} as it was.")
        tts = self.voices.primary()
        feedback = rest if kind else text
        if (cmd := intent.voice_command(feedback)) and cmd["action"] in ("another", "reset", "list"):
            sess.pending = None
            return await self._voice_command(sess, feedback, cmd)
        if cmd and cmd["action"] == "audition" and self._voice_target(sess, feedback) not in (None, speaker):
            sess.pending = None
            return await self._voice_command(sess, feedback, cmd)
        if tts and (got := await self._adjust(speaker, prof, feedback, tts)):
            return self._audition(sess, speaker, got[0], got[1])
        if kind:
            return self._reply(sess, "I'm not sure what to change. Try deeper, slower, more sarcastic, or say keep it.")
        sess.pending = None  # not about the voice at all: leave the draft and carry on
        sess.notices.append("I dropped the voice draft without saving it.")
        return None

    # --- background work after a save ---------------------------------------
    def _after_save(self, sess: Session, para: Paragraph) -> None:
        if self.graph:
            self.graph.schedule(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        if self.check_drift:
            task = asyncio.create_task(self._drift(sess, para))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    async def _drift(self, sess: Session, para: Paragraph) -> None:
        try:
            bible = self.store.bible(sess.book).context_for(para.text)
            graph = self.graph.context_for(sess.book, para.text) if self.graph else ""
            issues = await drift_review(self.llm, bible, graph, para.id, para.text)
        except Exception:
            return  # a failed background check must never disturb the writing session
        for i in issues:
            self.memory.add(f"Drift from canon: {i}", "drift")
            sess.notices.append(f"Heads up, this may contradict the story bible. {i}")

    # --- memory ----------------------------------------------------------
    def _learn(self, text: str) -> None:
        if PREF.search(text):
            self.memory.add(f"Writer said: {text[:300]}", "preference")

    def _reply(self, sess: Session, text: str, speech: list[dict] | None = None) -> Reply:
        if sess.pending and not sess.pending.ask:
            sess.pending.ask = self._ask_for(sess, sess.pending) or text
        pend = {"kind": sess.pending.kind, "text": sess.pending.text, "paragraph_id": sess.pending.paragraph_id, "ask": sess.pending.ask} if sess.pending else None
        notices, sess.notices = sess.notices, []
        if speech is None:
            speech = [{"speaker": vx.ASSISTANT, "text": text}] if text else []
        return Reply(text=text, mode=sess.mode, chapter=sess.chapter, focus=sess.focus, pending=pend, notes=notices, speech=speech)
