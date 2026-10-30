"""Conversation orchestrator: intent routing, targeted edits with confirmation, memory, quality gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from book_store import BookError, BookStore, Paragraph
import asyncio

from clients import LLM
from drift import review as drift_review
import intent
from kg import StoryKG, tokens
from modes import MODES
from quality import check, clean_output
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


@dataclass
class Session:
    book: str
    mode: str = "narrator"
    chapter: int = 1
    focus: str | None = None
    history: list[dict] = field(default_factory=list)
    pending: Pending | None = None
    notices: list[str] = field(default_factory=list)  # background findings, delivered on the next reply


@dataclass
class Reply:
    text: str
    mode: str
    chapter: int
    focus: str | None
    changed: list[str] = field(default_factory=list)
    pending: dict | None = None
    notes: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, s: Settings, store: BookStore, llm: LLM, memory, kg: StoryKG, graph=None, check_drift: bool = False):
        self.s, self.store, self.llm, self.memory, self.kg = s, store, llm, memory, kg
        self.graph, self.check_drift = graph, check_drift
        self.sessions: dict[str, Session] = {}
        self._bg: set[asyncio.Task] = set()

    def session(self, sid: str, book: str | None = None) -> Session:
        book = book or self.s.default_book
        sess = self.sessions.get(sid)
        if not sess or sess.book != book:
            self.store.ensure_book(book)
            nums = self.store.chapter_numbers(book)
            sess = self.sessions[sid] = Session(book=book, chapter=nums[0] if nums else 1)
        return sess

    # ------------------------------------------------------------------
    async def handle(self, sid: str, user_text: str, book: str | None = None) -> Reply:
        sess = self.session(sid, book)
        self.kg.refresh(sess.book)
        text = user_text.strip()
        if not text:
            return self._reply(sess, "I didn't catch that. Try again.")

        # Answers to a pending proposal come first, so "okay" confirms rather than being discarded as noise.
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
            if kind == "no" and rest:  # "no, I mean chapter 3 paragraph 2": the guess was wrong; keep the instruction, change the target
                original, sess.pending = sess.pending.text, None
                target, sure = self._resolve(sess, rest)
                if target:
                    return await self._propose_edit(sess, original, target=target, sure=sure)
                text = rest
            else:
                sess.pending = None  # anything else is a new command; the unsaved proposal is abandoned, and we say so
                sess.notices.append("I dropped the unsaved proposal.")

        if intent.is_noise(text):  # silence that the speech model turned into "Thank you."
            return Reply(text="", mode=sess.mode, chapter=sess.chapter, focus=sess.focus)

        self._learn(text)

        if mode := intent.detect_mode(text):
            sess.mode = mode
            return self._reply(sess, "Switching to feedback mode. Tell me what you're working on." if mode == "therapist" else "Back to narrator mode.")

        if (n := intent.goto_chapter(text)) is not None:
            if n in self.store.chapter_numbers(sess.book):
                sess.chapter, sess.focus = n, None
                return self._reply(sess, f"Chapter {n}.")
            return self._reply(sess, f"There is no chapter {n} yet.")
        if d := intent.step_direction(text):
            return self._step(sess, d)
        if intent.is_read_aloud(text):
            target = self._by_ref(sess, intent.parse_ref(text))
            if target:
                sess.chapter, sess.focus = target.chapter, target.id
                return self._reply(sess, target.text)
            return self._reply(sess, "I couldn't find that paragraph.")

        if sess.mode == "narrator" and intent.is_edit(text):
            return await self._propose_edit(sess, text)
        if sess.mode == "narrator" and intent.is_draft(text):
            return await self._propose_draft(sess, text)
        return await self._converse(sess, text)

    # --- navigation ----------------------------------------------------
    def _step(self, sess: Session, d: int) -> Reply:
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
        return self._reply(sess, paras[i].text)

    # --- target resolution -----------------------------------------------
    def _by_ref(self, sess: Session, ref: intent.Ref) -> Paragraph | None:
        if ref.para is None:
            return None
        return self.store.nth_paragraph(sess.book, ref.chapter or sess.chapter, ref.para, ref.scene)

    def _resolve(self, sess: Session, text: str) -> tuple[Paragraph | None, bool]:
        """-> (paragraph, sure). sure=False means "this is my best guess; read it back before touching it"."""
        ref = intent.parse_ref(text)
        if ref.explicit:
            return self._by_ref(sess, ref), True
        if sess.focus and re.search(r"\b(this|that|the current|it)\b", intent.norm(text)) and not self.kg.named_in(text):
            return self.store.find(sess.book, sess.focus), True
        query = intent.locator_query(text)  # describe-where words only; the instruction ("more sarcastic") must not steer the search
        hits = self.kg.search(query, limit=3, chapter=ref.chapter)
        if not hits:
            return None, False
        return hits[0].para, self.kg.confident(hits, len(tokens(query)))

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

    def _context(self, sess: Session, text: str, extra: str = "", target: Paragraph | None = None) -> list[dict]:
        prefs = self.memory.search(text, limit=6)
        sys = [SYSTEM, MODES[sess.mode]]
        if self.s.writer_age:
            sys.append(f"The writer is {self.s.writer_age}; let that subtly shape idiom and description density. Never mention it.")
        if prefs:
            sys.append("WRITER PREFERENCES / MEMORY:\n- " + "\n- ".join(prefs))
        bible = self.store.bible(sess.book).context_for(text + " " + extra)
        if bible:
            sys.append("STORY BIBLE (canon; never contradict):\n" + bible)
        rel = self.graph.context_for(sess.book, text + " " + extra) if self.graph else ""
        if rel:
            sys.append("STORY GRAPH (EXTRACTED = stated in the text, INFERRED = a guess):\n" + rel)
        ms = self._manuscript(sess, target)
        if ms:
            sys.append("MANUSCRIPT AROUND THE WORK (for continuity; do not repeat it):\n" + ms)
        return [{"role": "system", "content": "\n\n".join(sys)}] + sess.history[-10:]

    async def _gated(self, messages: list[dict]) -> tuple[str, list[str]]:
        out = clean_output(await self.llm.chat(messages))
        issues = check(out)
        for _ in range(self.s.max_rewrite_attempts):
            if not issues:
                break
            out = clean_output(await self.llm.chat(
                messages + [{"role": "assistant", "content": out},
                            {"role": "user", "content": "Rewrite it fixing these problems, output only the text:\n- " + "\n- ".join(issues)}]
            ))
            issues = check(out)
        return out, issues

    def _remember(self, sess: Session, user: str, assistant: str) -> None:
        sess.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        del sess.history[:-12]

    # --- flows -----------------------------------------------------------
    async def _converse(self, sess: Session, text: str) -> Reply:
        sess.history.append({"role": "user", "content": text})
        out = await self.llm.chat(self._context(sess, text))
        sess.history.append({"role": "assistant", "content": out})
        return self._reply(sess, out)

    async def _propose_edit(self, sess: Session, text: str, target: Paragraph | None = None, sure: bool = True) -> Reply:
        if target is None:
            target, sure = self._resolve(sess, text)
        if not target:
            return self._reply(sess, "Which part? Say the chapter and paragraph, or describe the scene.")
        where = self._where(sess, target)
        if not sure:  # a guess: read it back and wait, rather than rewriting something the writer didn't mean
            sess.chapter, sess.focus = target.chapter, target.id
            sess.pending = Pending("confirm_target", text, target.chapter, paragraph_id=target.id)
            return self._reply(sess, f"Do you mean {where}, starting: {excerpt(target.text)} Say yes to go ahead, or tell me which part.")
        msgs = self._context(sess, text, target.text, target) + [
            {"role": "user", "content": f"Rewrite ONLY this paragraph.\nInstruction: {text}\n\nORIGINAL:\n{target.text}\n\nOutput only the new paragraph."}
        ]
        new, issues = await self._gated(msgs)
        sess.chapter, sess.focus = target.chapter, target.id
        sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id)
        self._remember(sess, text, f"[proposed a rewrite of {where}]")
        reply = self._reply(sess, f"Changing {where}, which starts: {excerpt(target.text, 6)} Here is the new version: {new} Should I save it?")
        reply.notes += [f"Style note: {i}" for i in issues]
        return reply

    async def _propose_draft(self, sess: Session, text: str) -> Reply:
        if sess.chapter not in self.store.chapter_numbers(sess.book):
            self.store.create_chapter(sess.book, sess.chapter)
        recent = "\n\n".join(p.text for p in self.store.paragraphs(sess.book, sess.chapter)[-4:])
        msgs = self._context(sess, text, recent) + [
            {"role": "user", "content": f"{text}\n\nStory so far in this chapter (most recent):\n{recent or '(nothing yet)'}\n\nWrite ONE new paragraph of prose. Output only the paragraph."}
        ]
        new, issues = await self._gated(msgs)
        sess.pending = Pending("append", new, sess.chapter)
        self._remember(sess, text, "[proposed a new paragraph]")
        reply = self._reply(sess, f"Here's a draft: {new} Should I add it to chapter {sess.chapter}?")
        reply.notes += [f"Style note: {i}" for i in issues]
        return reply

    async def _revise(self, sess: Session, instruction: str) -> Reply:
        """The writer answered a proposal with changes ("yes but shorter", "no, make it longer"): redo it, don't save or drop."""
        p = sess.pending
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
        reply = self._reply(sess, f"Revised: {new} Should I save it?")
        reply.notes += [f"Style note: {i}" for i in issues]
        return reply

    async def _accept(self, sess: Session) -> Reply:
        if sess.pending.kind == "confirm_target":
            p, sess.pending = sess.pending, None
            return await self._propose_edit(sess, p.text, target=self.store.find(sess.book, p.paragraph_id), sure=True)
        return self._commit(sess)

    def _commit(self, sess: Session) -> Reply:
        p, sess.pending = sess.pending, None
        try:
            if p.kind == "edit":
                para = self.store.replace_paragraph(sess.book, p.paragraph_id, p.text)
            else:
                para = self.store.append_paragraph(sess.book, p.chapter, p.text, p.scene)
        except BookError as e:
            return self._reply(sess, f"I couldn't save that: {e}")
        sess.focus = para.id
        self.kg.refresh(sess.book)
        self._after_save(sess, para)
        r = self._reply(sess, "Saved.")
        r.changed = [para.id]
        return r

    # --- background work after a save ---------------------------------------
    def _after_save(self, sess: Session, para: Paragraph) -> None:
        if self.graph:
            self.graph.schedule(sess.book)
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

    def _reply(self, sess: Session, text: str) -> Reply:
        pend = {"kind": sess.pending.kind, "text": sess.pending.text, "paragraph_id": sess.pending.paragraph_id} if sess.pending else None
        notices, sess.notices = sess.notices, []
        return Reply(text=text, mode=sess.mode, chapter=sess.chapter, focus=sess.focus, pending=pend, notes=notices)
