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
import speech
import voices as vx
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
    data: dict | None = None  # kind "voice": {speaker, profile, draft}


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


class Agent:
    def __init__(self, s: Settings, store: BookStore, llm: LLM, memory, kg: StoryKG, graph=None, check_drift: bool = False, voices=None, prefs=None, index=None):
        self.s, self.store, self.llm, self.memory, self.kg = s, store, llm, memory, kg
        self.graph, self.check_drift, self.voices, self.prefs, self.index = graph, check_drift, voices, prefs, index
        self._speaker_cache: dict[str, list[str | None]] = {}
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
            if kind == "no" and rest:  # "no, I mean chapter 3 paragraph 2": the guess was wrong; keep the instruction, change the target
                original, sess.pending = sess.pending.text, None
                target, sure = await self._resolve(sess, rest)
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

        if self.voices and (cmd := intent.voice_command(text)):
            if (r := await self._voice_command(sess, text, cmd)) is not None:
                return r

        if (n := intent.goto_chapter(text)) is not None:
            if n in self.store.chapter_numbers(sess.book):
                sess.chapter, sess.focus = n, None
                return self._reply(sess, f"Chapter {n}.")
            return self._reply(sess, f"There is no chapter {n} yet.")
        if d := intent.step_direction(text):
            return await self._step(sess, d)
        if intent.is_read_aloud(text):
            target = self._by_ref(sess, intent.parse_ref(text))
            if target:
                sess.chapter, sess.focus = target.chapter, target.id
                return self._reply(sess, target.text, await self._speech_for(sess, target))
            return self._reply(sess, "I couldn't find that paragraph.")

        if sess.mode == "narrator" and intent.is_edit(text):
            return await self._propose_edit(sess, text)
        if sess.mode == "narrator" and intent.is_draft(text):
            return await self._propose_draft(sess, text)
        return await self._converse(sess, text)

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
        return self._reply(sess, paras[i].text, await self._speech_for(sess, paras[i]))

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

    def _context(self, sess: Session, text: str, extra: str = "", target: Paragraph | None = None) -> list[dict]:
        prefs = self.memory.search(text, limit=6)
        sys = [SYSTEM, MODES[sess.mode]]
        age = (self.prefs.get("writer_age") if self.prefs else "") or self.s.writer_age
        if age:
            sys.append(f"The writer is {age}; let that subtly shape idiom and description density. Never mention it.")
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
            target, sure = await self._resolve(sess, text)
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
        lead = f"Changing {where}, which starts: {excerpt(target.text, 6)} Here is the new version:"
        reply = self._reply(sess, f"{lead} {new} Should I save it?", await self._speech_text(sess, new, lead, "Should I save it?", target))
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
        tail = f"Should I add it to chapter {sess.chapter}?"
        reply = self._reply(sess, f"Here's a draft: {new} {tail}", await self._speech_text(sess, new, "Here's a draft:", tail))
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
        reply = self._reply(sess, f"Revised: {new} Should I save it?", await self._speech_text(sess, new, "Revised:", "Should I save it?"))
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
                    self._speaker_cache[key] = await asyncio.wait_for(speech.llm_attribute(self.llm, list(chars), before[-1].text if before else "", segs, text), 8)
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
        pend = {"kind": sess.pending.kind, "text": sess.pending.text, "paragraph_id": sess.pending.paragraph_id} if sess.pending else None
        notices, sess.notices = sess.notices, []
        if speech is None:
            speech = [{"speaker": vx.ASSISTANT, "text": text}] if text else []
        return Reply(text=text, mode=sess.mode, chapter=sess.chapter, focus=sess.focus, pending=pend, notes=notices, speech=speech)
