"""Conversation orchestrator: intent routing, targeted edits with confirmation, memory, quality gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from book_store import BookError, BookStore, Paragraph
from clients import LLM
from kg import StoryKG
from modes import MODES
from quality import check
from settings import Settings

SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text()

YES = re.compile(r"^\s*(yes|yeah|yep|sure|ok(ay)?|do it|save( it)?|go ahead|confirm|sounds good)\b", re.I)
NO = re.compile(r"^\s*(no|nope|cancel|never mind|don'?t|discard|scrap that)\b", re.I)
EDIT = re.compile(r"\b(change|make|rewrite|rephrase|fix|edit|tweak|soften|tighten|revise|replace)\b", re.I)
DRAFT = re.compile(r"\b(write|draft|continue|next scene|add a (scene|paragraph)|keep going)\b", re.I)
PREF = re.compile(r"\b(i (prefer|like|love|hate|don'?t like|dislike)|always|never|too (long|flowery|wordy|formal)|remember (that|this))\b", re.I)
REF = re.compile(
    r"chapter\s+(\d+)(?:\D+?scene\s+(\d+))?(?:\D+?(?:paragraph|para)\s+(\d+))?", re.I
)
GOTO = re.compile(r"\b(?:go to|open|show|read)\s+chapter\s+(\d+)\b", re.I)
NUMWORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


@dataclass
class Pending:
    kind: str  # "edit" | "append"
    text: str
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
    def __init__(self, s: Settings, store: BookStore, llm: LLM, memory, kg: StoryKG):
        self.s, self.store, self.llm, self.memory, self.kg = s, store, llm, memory, kg
        self.sessions: dict[str, Session] = {}

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

        self._learn(text)

        low = text.lower()
        if "therapist" in low and "mode" in low or "quick feedback" in low:
            sess.mode = "therapist"
            return self._reply(sess, "Switching to feedback mode. Tell me what you're working on.")
        if "narrator" in low and "mode" in low:
            sess.mode = "narrator"
            return self._reply(sess, "Back to narrator mode.")

        if sess.pending:
            if YES.match(text):
                return self._commit(sess)
            if NO.match(text):
                self.memory.add(f"Writer rejected a proposed {sess.pending.kind}: '{sess.pending.text[:120]}'", "preference")
                sess.pending = None
                return self._reply(sess, "Okay, dropped it. What would you like instead?")
            sess.pending = None  # anything else abandons the proposal and is handled normally

        if m := GOTO.search(text):
            n = int(m.group(1))
            if n in self.store.chapter_numbers(sess.book):
                sess.chapter, sess.focus = n, None
                return self._reply(sess, f"Chapter {n}.")
            return self._reply(sess, f"There is no chapter {n} yet.")
        if re.search(r"\bnext (paragraph|section)\b", low):
            return self._step(sess, +1)
        if re.search(r"\b(previous|last) (paragraph|section)\b", low):
            return self._step(sess, -1)

        if sess.mode == "narrator" and EDIT.search(text):
            return await self._propose_edit(sess, text)
        if sess.mode == "narrator" and DRAFT.search(text):
            return await self._propose_draft(sess, text)
        return await self._converse(sess, text)

    # --- navigation ----------------------------------------------------
    def _step(self, sess: Session, d: int) -> Reply:
        paras = self.store.paragraphs(sess.book, sess.chapter) if sess.chapter in self.store.chapter_numbers(sess.book) else []
        if not paras:
            return self._reply(sess, "This chapter is empty.")
        ids = [p.id for p in paras]
        i = ids.index(sess.focus) + d if sess.focus in ids else (0 if d > 0 else len(ids) - 1)
        i = max(0, min(len(ids) - 1, i))
        sess.focus = ids[i]
        return self._reply(sess, paras[i].text)

    # --- target resolution -----------------------------------------------
    def _resolve(self, sess: Session, text: str) -> Paragraph | None:
        if m := REF.search(text):
            ch = int(m.group(1))
            scene = int(m.group(2) or 1)
            if m.group(3):
                return self.store.resolve_ref(sess.book, ch, scene, int(m.group(3)))
        for word, n in NUMWORDS.items():
            if re.search(rf"\b{word} paragraph\b", text, re.I):
                return self.store.resolve_ref(sess.book, sess.chapter, 1, n)
        if re.search(r"\b(this|that|the current) (part|paragraph|bit|section)\b", text, re.I) and sess.focus:
            return self.store.find(sess.book, sess.focus)
        hits = self.kg.find_paragraphs(text, limit=1)
        return hits[0] if hits else None

    # --- LLM helpers -----------------------------------------------------
    def _context(self, sess: Session, text: str, extra: str = "") -> list[dict]:
        prefs = self.memory.search(text, limit=6)
        sys = [SYSTEM, MODES[sess.mode]]
        if self.s.writer_age:
            sys.append(f"The writer is {self.s.writer_age}; let that subtly shape idiom and description density. Never mention it.")
        if prefs:
            sys.append("WRITER PREFERENCES / MEMORY:\n- " + "\n- ".join(prefs))
        bible = self.store.read_doc(sess.book, "bible").strip()
        if bible:
            sys.append("STORY BIBLE:\n" + bible[:4000])
        kg_ctx = self.kg.context_for(text + " " + extra)
        if kg_ctx:
            sys.append("CHARACTERS IN PLAY:\n" + kg_ctx)
        return [{"role": "system", "content": "\n\n".join(sys)}] + sess.history[-10:]

    async def _gated(self, messages: list[dict]) -> tuple[str, list[str]]:
        out = await self.llm.chat(messages)
        issues = check(out)
        for _ in range(self.s.max_rewrite_attempts):
            if not issues:
                break
            out = await self.llm.chat(
                messages + [{"role": "assistant", "content": out},
                            {"role": "user", "content": "Rewrite it fixing these problems, output only the text:\n- " + "\n- ".join(issues)}]
            )
            issues = check(out)
        return out, issues

    # --- flows -----------------------------------------------------------
    async def _converse(self, sess: Session, text: str) -> Reply:
        sess.history.append({"role": "user", "content": text})
        out = await self.llm.chat(self._context(sess, text))
        sess.history.append({"role": "assistant", "content": out})
        return self._reply(sess, out)

    async def _propose_edit(self, sess: Session, text: str) -> Reply:
        target = self._resolve(sess, text)
        if not target:
            return self._reply(sess, "Which part? Say the chapter and paragraph, or describe the scene.")
        msgs = self._context(sess, text, target.text) + [
            {"role": "user", "content": f"Rewrite ONLY this paragraph.\nInstruction: {text}\n\nORIGINAL:\n{target.text}\n\nOutput only the new paragraph."}
        ]
        new, issues = await self._gated(msgs)
        sess.chapter, sess.focus = target.chapter, target.id
        sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id)
        reply = self._reply(sess, f"Here is the new version of chapter {target.chapter}, paragraph {target.id.split('_p')[-1]}: {new} Should I save it?")
        reply.notes = issues
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
        reply = self._reply(sess, f"Here's a draft: {new} Should I add it to chapter {sess.chapter}?")
        reply.notes = issues
        return reply

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
        r = self._reply(sess, "Saved.")
        r.changed = [para.id]
        return r

    # --- memory ----------------------------------------------------------
    def _learn(self, text: str) -> None:
        if PREF.search(text):
            self.memory.add(f"Writer said: {text[:300]}", "preference")

    def _reply(self, sess: Session, text: str) -> Reply:
        pend = {"kind": sess.pending.kind, "text": sess.pending.text, "paragraph_id": sess.pending.paragraph_id} if sess.pending else None
        return Reply(text=text, mode=sess.mode, chapter=sess.chapter, focus=sess.focus, pending=pend)
