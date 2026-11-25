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
import images as imgs
from images import PictureError
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


_CHATTER_TAIL = re.compile(r"(?:\n\s*\n|(?<=[.!?\"”’])\s+)(?:I |Let me know|Would you|Should I|Shall I|Do you|Want me|Does that|Is that|Hope |Note:|\(|—)[^\n]*$", re.I)
_ADDRESS = re.compile(r"\b(?:let me know|would you like|shall i|should i|do you want|want me to|hope (?:this|that)|i (?:liked|gave|tried|made|kept|wrote|added|used|went with|chose|thought))\b", re.I)


_ABBR = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "mt", "capt", "col", "gen", "lt", "sgt", "no", "vs", "etc", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "prof", "rev", "hon", "fr", "gov", "sen", "rep", "co", "inc", "ft", "u.s", "a.m", "p.m"}


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences without losing closing quotes and without splitting after Mr., St., Dr., month abbreviations and the like."""
    parts = re.split(r"(?<=[.!?])([\"\u201d\u2019)]*)\s+(?=[\"\u201c\u2018(]?[A-Z0-9])", text.strip())
    # re.split with a group gives [sentence, closers, sentence, closers, ...]
    pieces: list[str] = []
    i = 0
    while i < len(parts):
        piece = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        pieces.append(piece)
        i += 2
    out: list[str] = []
    for pc in pieces:
        if out:
            last_word = re.findall(r"([A-Za-z.]+)\.$", out[-1])
            if last_word and last_word[0].lower().rstrip(".") in _ABBR:
                out[-1] = out[-1] + " " + pc
                continue
        out.append(pc)
    return [o for o in out if o.strip()]


_DRAFT_JUNK_HEAD = re.compile(r"^(?:\s*(?:propose_append|propose_edit_paragraph|start_chapter|make_picture|text)\s*:\s*|\s*(?:the writer (?:asked|wants|said)|you asked|as requested)[^\n.]{0,200}[.:]\s*)+", re.I)
_DRAFT_JUNK_TAIL = re.compile(r"(?:\s+(?:how does (?:this|that) sound|does (?:this|that) (?:work|sound)|is (?:this|that) (?:what|alright|okay|ok)|what do you think|let me know|would you like|shall i|should i|want me to)[^.!?]*[.!?]?\s*)+$", re.I)


def clean_draft(text: str) -> str:
    """Strip the model's own talk to the writer ("I gave Arthur a small wink. Want me to change it?") from a paragraph meant for the book. Only a trailing
    block that clearly addresses the writer is removed; the paragraph itself is never cut into."""
    t = text.strip().strip("`").strip()
    m_ = re.search(r"\b(?:propose_\w+|start_chapter|make_picture)\s*\(\s*(?:text|paragraph)\s*=\s*(['\"\u201c])(.+?)\1\s*(?:,[^()]*)?\)", t, flags=re.S)
    if m_:
        t = m_.group(2).replace("\\'", "'").replace('\\"', '"')  # propose_append(text='…', chapter=1): only the paragraph inside is prose
    t = re.sub(r"\[\[[^\[\]]{0,4000}\]\]|\((?:note for me|note)[^)]{0,600}\)|^\s*\[[^\]\n]{0,120}\]\s*", " ", t, flags=re.I).strip()  # tool lines and notes are never prose
    for _ in range(3):  # talk before the paragraph: "Let me try again, using only what we know." / "Here's another version."
        t2 = re.sub(r"^(?:(?:okay|ok|alright|sure|of course)[,.!]?\s+)?(?:let me (?:try|write|redo|fix|give|have)|here'?s (?:a|the|my|another|that)\b[^.!?:]{0,40}[.!]|how about (?:this|that)[.!]|i'?ll (?:try|write|redo|give|have another)|using only what|this (?:version|time)|try(?:ing)? again)[^.!?:]{0,120}[.!]\s+", "", t, flags=re.I).strip()
        if t2 == t:
            break
        t = t2
    for _ in range(2):  # a lead-in line ("I can put this in the book for you:", "How about this: \"", "Here's a draft for chapter 2:")
        t = re.sub(r"^(?:[^.!?\n:]{0,90}\b(?:i|i'?ll|i can|here'?s|here is|how about|for you|draft|version|let me)\b[^.!?\n:]{0,40}:)\s*", "", t, flags=re.I).strip()
    t = _DRAFT_JUNK_HEAD.sub("", t).strip()
    if len(t) > 2 and t[0] in "\"\u201c" and t[-1] in "\"\u201d" and len(re.findall(r"[\"\u201c\u201d]", t)) == 2:
        t = t.strip("\"\u201c\u201d ").strip()  # a paragraph wrapped in quotes
    t = _DRAFT_JUNK_TAIL.sub("", t).strip()  # "… How does this sound for your book?"
    # an apology or explanation before the paragraph ("My apologies. I had proposed it, but... Here is the paragraph I drafted:")
    t = re.sub(r"^(?:(?:my apologies|i apologi[sz]e|sorry|i'?m sorry|okay|ok|sure|of course|certainly|alright|understood|you'?re right)\b[^\n]{0,300}?(?:here(?:'s| is)[^\n:]{0,80}:|\n\n))\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^(?:here(?:'s| is)[^\n:]{0,60}:|sure[,!.][^\n]{0,40}\n)\s*", "", t, flags=re.I).strip()
    for _ in range(3):  # a few trailing remarks at most
        m = _CHATTER_TAIL.search(t)
        if not m or not _ADDRESS.search(t[m.start():]) and not t[m.start():].strip().endswith("?"):
            break
        if len(t[:m.start()].strip()) < 60:
            break
        t = t[: m.start()].rstrip()
    return t.strip()


def excerpt(text: str, n: int = EXCERPT_WORDS) -> str:
    w = imgs.plain(text).split()
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
    research: dict = field(default_factory=dict)  # every page looked up this session, by title: what the writer heard counts as a source
    last_edit: tuple | None = None  # (instruction, paragraph id) of a change that found nothing to change, for one turn: "no, I meant paragraph three"
    work: int = 0  # the chapter new writing goes into (set by "start / go to chapter", by naming a chapter, by saving): looking at other paragraphs never moves it
    viewing_now: int | None = None  # the chapter the page showed when this request was sent
    last_viewing: int | None = None  # the chapter the page last said it was showing
    screen_wins: bool = False  # the writer navigated since they last wrote: new pictures go to the chapter on screen, not the one being written
    last_dropped: object = None  # the proposal that was put aside when the writer moved on, so "bring it back" can restore it
    dropped_stale: bool = False  # something was saved or undone since the proposal was put aside: "bring it back" now means undo
    say_first: str = ""  # something to say before the next reply (a dropped proposal)
    last_user: str = ""  # what the writer last asked for, so a draft can be checked against it
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
    picture: dict | None = None  # a picture waiting for the writer's yes or no: {"url", "caption"}
    download: dict | None = None  # a file for the page to save: {"url", "label"}


class Agent:
    def __init__(self, s: Settings, store: BookStore, llm: LLM, memory, kg: StoryKG, graph=None, check_drift: bool = False, voices=None, prefs=None, index=None, journal=None, library=None, addons=None):
        self.s, self.store, self.llm, self.memory, self.kg = s, store, llm, memory, kg
        self.graph, self.check_drift, self.voices, self.prefs, self.index = graph, check_drift, voices, prefs, index
        self.journal, self.library, self.addons = journal, library, addons  # long-term record, imported files, administrator addons
        self.inbox, self.stt = Inbox(store), None  # the drop folder; speech-to-text for audio files (set by main)
        self.images, self.imagegen = imgs.ImageStore(store), None  # pictures in the book; the generation server (set by main)
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
    async def handle(self, sid: str, user_text: str, book: str | None = None, via: str = "typed", interrupted: bool = False, viewing: int | None = None) -> Reply:
        """One turn. Everything said is written to the journal (so nothing is ever lost when working memory is trimmed)."""
        sess = self.session(sid, book)
        sess.via, sess.interrupted = via, interrupted
        sess.viewing_now = viewing  # only what THIS request says: a client that does not send it is never assumed to be looking at an old chapter
        if viewing:  # the page says which chapter it shows: what the writer says next is about the chapter they are looking at
            if viewing != sess.last_viewing and viewing in self.store.chapter_numbers(sess.book):
                sess.chapter = viewing
            sess.last_viewing = viewing
            sess.screen_wins = True
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

    def _work(self, sess: Session) -> int:
        """The chapter the writer is writing in. Reading, editing or moving something elsewhere does not change it."""
        nums = self.store.chapter_numbers(sess.book)
        if sess.work and (sess.work in nums or sess.work == max(nums, default=0) + 1):
            return sess.work
        return max(nums, default=1)

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
        sess.last_user = text
        self._learn_genre(text)
        last_edit, sess.last_edit = sess.last_edit, None
        if last_edit and not sess.pending and re.match(r"^(?:(?:oh|sorry|no|wait|um|uh)\s+)*(?:i (?:meant|mean)|it'?s|its|i said)\b", intent.norm(text)) \
                and ((rf_ := intent.parse_ref(text)).para is not None or re.search(r"\b(?:before|after|next|previous)\b", intent.norm(text))):
            try:
                old = self.store.find(sess.book, last_edit[1])
            except BookError:
                old = None
            sess.pending = Pending("confirm_target", last_edit[0], old.chapter if old else sess.chapter, paragraph_id=last_edit[1])
            return await self._retarget(sess, text)  # the change the writer asked for a moment ago, on the paragraph they meant
        items, sess.last_items = sess.last_items, []  # only the very next thing the writer says can pick one of the numbered ideas
        if items and not sess.pending and (k := intent.pick_number(text)) and 1 <= k <= len(items):
            text = f"Let's use this {sess.last_items_kind}: {items[k - 1]}"

        nb_ = intent.norm(text)
        save_back = bool(re.search(r"\b(?:bring|get) (?:that|it|that 1|the \w+(?: 1)?) back\b", nb_) and re.search(r"\b(?:said yes|save it|put it in|add it|keep it)\b", nb_))
        if sess.last_dropped is not None and (save_back and sess.last_dropped.kind in ("append", "picture") or not sess.dropped_stale and ( re.match(r"^(?:please )?(?:bring|put|get) (?:that|it|the (?:last|other|first|dropped|old) (?:one|paragraph|draft|change)) back$|^i (?:want|wanted) (?:that|it|the (?:other|first|last) one) back$|^(?:restore|undrop) (?:that|it)$", nb_))):
            if save_back and sess.last_dropped.kind in ("append", "picture"):  # "No, I said yes to that one. Bring it back and put it in chapter two."
                keep_ = sess.pending if sess.pending and sess.pending.kind in ("append", "edit", "picture") else None
                sess.pending, sess.last_dropped = sess.last_dropped, None
                if (mch_ := re.search(r"\bchapter (\d+)\b", nb_)) and sess.pending.kind == "append":
                    ch_ = int(mch_.group(1))
                    nums_ = self.store.chapter_numbers(sess.book)
                    if ch_ in nums_ or ch_ == max(nums_, default=0) + 1:
                        sess.pending.chapter, sess.pending.ask = ch_, ""
                done = await self._accept(sess)
                if keep_ is not None:
                    sess.pending = keep_  # the newer draft that was on screen is still waiting
                    done.pending = {"kind": keep_.kind, "text": keep_.text, "paragraph_id": keep_.paragraph_id, "ask": keep_.ask}
                    done.text += f" The other one is still waiting: {keep_.ask or self._ask_for(sess, keep_)} Say yes or no."
                return done
            if sess.pending and sess.pending.kind in ("append", "edit", "picture"):
                sess.pending, sess.last_dropped = sess.last_dropped, sess.pending  # swap: the one on screen is put aside in turn
            elif not sess.pending:
                sess.pending, sess.last_dropped = sess.last_dropped, None
            else:
                return self._reply(sess, "I am holding something else for your yes or no right now. Answer that first.")
            body = f" {sess.pending.text}" if sess.pending.kind in ("append", "edit") else ""
            r = self._reply(sess, f"Here it is again:{body} {sess.pending.ask or self._ask_for(sess, sess.pending)} Say yes or no.")
            if sess.pending.kind == "picture":
                d_ = sess.pending.data or {}
                try:
                    self.images.path(sess.book, d_.get("candidate", ""), candidate=True)
                    r.picture = self.images.info(sess.book, d_["candidate"], d_.get("caption", ""), candidate=True)
                except FileNotFoundError:
                    sess.pending = None
                    return self._reply(sess, "That picture is gone. Ask me for it again.")
            return r
        if intent.describe_request(text):  # "describe it to me": words for the picture, and a waiting picture keeps waiting. Never a new paragraph.
            return self._describe_picture(sess, text) or self._reply(sess, "I do not know which picture you mean. Say which chapter it is in, or what it shows, and I will describe it.")
        if sess.pending and intent.undo_request(text) and sess.pending.kind not in ("voice",):  # "undo that" while a proposal waits: drop the proposal
            sess.pending = None
            return self._reply(sess, "Okay, I dropped that. Nothing in your book was changed.")
        if sess.pending and (mode := intent.autosave_mode(text)) is not None:  # "ask me first from now on" with a draft waiting: the setting changes, the draft keeps waiting
            if self.prefs:
                self.prefs.update({"auto_save": mode})
            return self._reply(sess, "Okay. I will save each paragraph straight away from now on. The one waiting still needs your yes." if mode == "on" else "Okay. I will ask you before I save anything I write. The one waiting is still waiting for your yes.")
        # Answers to a pending proposal come first, so "okay" confirms rather than being discarded as noise.
        if sess.pending and sess.pending.kind == "file":
            done = await self._file_pending(sess, text)
            if done is not None:
                return done
        if sess.pending and sess.pending.kind == "voice":
            done = await self._voice_pending(sess, text)
            if done is not None:
                return done
        if sess.pending and sess.pending.kind in ("append", "edit") and re.match(r"^(?:please )?(?:read|say|play) (?:it|that|this|them|the (?:whole|rest|draft|paragraph|thing))(?: (?:all|out|aloud|again|to me|of it|through|please|out loud|the whole thing|the rest))*$|^(?:read )?(?:all of it|the whole thing|the rest)$", intent.norm(text)):
            r = self._reply(sess, f"{sess.pending.text} {sess.pending.ask or self._ask_for(sess, sess.pending)} Say yes or no.",
                            await self._speech_text(sess, sess.pending.text, "", f"{sess.pending.ask or ''} Say yes, or no, or what to change.".strip()))
            return r
        if (dr := re.match(r"^(?:please )?(?:read (?:my )?drafts? (?:to me )?(?:in full|all the way|completely|all of it)|read the whole (?:draft|paragraph) (?:every time|each time)|(?:read|only read) (?:me )?(?:just )?the (?:start|beginning) of (?:my )?drafts)(?: from now on| please)?$", intent.norm(text))):
            if self.prefs:
                self.prefs.update({"draft_read": "full" if "start" not in intent.norm(text) and "beginning" not in intent.norm(text) else "short"})
            return self._reply(sess, "Okay. I will read each whole draft to you." if "start" not in intent.norm(text) and "beginning" not in intent.norm(text) else "Okay. I will read you just the start of a long draft.")
        if sess.pending and sess.pending.kind not in ("voice", "file"):
            nt = intent.norm(text)
            if (re.match(r"^(?:ok(?:ay)?|yes|yeah|sure|alright|right|well)?,? ?(?:hold on|wait|wait a (?:moment|minute|sec(?:ond)?)|hang on|one (?:moment|second|sec|minute)|let me think(?: about it)?|give me a (?:moment|minute|sec(?:ond)?)|not yet|not now|later)\b", nt)
                    and len(re.sub(r"^(?:ok(?:ay)?|yes|yeah|sure|alright|right|well)?,? ?(?:hold on|wait(?: a (?:moment|minute|sec(?:ond)?))?|hang on|one (?:moment|second|sec|minute)|let me think(?: about it)?|give me a (?:moment|minute|sec(?:ond)?)|not yet|not now|later)\b", "", nt).split()) <= 3) \
                    or re.match(r"^good (?:night|morning|evening|afternoon)\b", nt):
                return self._reply(sess, f"Okay, I'll wait. I am still holding this for you: {sess.pending.ask or self._ask_for(sess, sess.pending)} Say yes or no.")
            if sess.pending.kind in ("append", "edit") and re.match(r"^(?:ok(?:ay)?|yes|yeah|sure|please)?,? ?(?:please )?(?:read|say|play) (?:it|that|this|them)(?: (?:to me|out|aloud|again|all|out loud|please))*$", nt):
                return self._reply(sess, f"{sess.pending.text} {sess.pending.ask or self._ask_for(sess, sess.pending)} Say yes or no.",
                                   await self._speech_text(sess, sess.pending.text, "", f"{sess.pending.ask or ''} Say yes, or no, or what to change.".strip()))
            if sess.pending.kind in ("append", "edit") and (sr_ := intent.sentence_removal(text)) is not None and sr_[0] is None:
                return self._shorten_waiting(sess, sr_[1])  # "take the last sentence out" while a draft waits: shorten THAT draft
        if sess.pending:
            kind, rest = intent.confirmation(text)
            nt_ = intent.norm(text)
            first_ = (intent._words(text) or [""])[0]
            if sess.pending.kind in ("delete", "move") and re.search(r"\b(?:keep|leave|alone|as it is|where it is|stay|stays|don'?t|do not)\b", nt_):
                kind, rest = "no", ""  # "keep it", "fine, leave it", "keep it where it is": the writer is refusing the delete or move
            if kind == "yes" and rest and (intent.is_question(rest) or text.strip().endswith("?")) and first_ not in ("yes", "yeah", "yep", "yup", "save"):
                kind, rest = None, text  # "okay, who was waiting at the station?", "good, what happens next?": conversation, not a yes
            if (two := self._answer_then_more(sess, text)) is not None:
                first_kind, after = two  # "Yes, add it. We'll just say 1918." / "No! Don't take it out. I want to change it…": answer first, then the rest is its own request
                if first_kind == "yes":
                    done = await self._accept(sess)
                else:
                    was = sess.pending.kind
                    if was == "picture":
                        self.images.discard(sess.book, (sess.pending.data or {}).get("candidate", ""))
                    sess.pending = None
                    done = self._reply(sess, "Okay, I left it as it is." if was in ("delete", "move") else "Okay, I let that go.")
                if sess.pending is None and after:
                    more = await self._handle(sid, after, sess.book)
                    more.text = f"{done.text} {more.text}".strip()
                    more.speech = (done.speech or []) + (more.speech or [])
                    more.changed = list(dict.fromkeys((done.changed or []) + (more.changed or [])))
                    return more
                return done
            if kind == "yes" and sess.pending.kind not in ("confirm_target", "voice", "file") and (cont := self._continue_after(text, rest)):
                done = await self._accept(sess)  # "yes, save it and write the next one", "keep going": save this one, then write on
                if sess.pending is None and done.text.startswith(("Saved", "Added", "Done", "Changed")):
                    nxt = await self._propose_draft(sess, cont)
                    nxt.text = f"{done.text} {nxt.text}".strip()
                    nxt.speech = (done.speech or []) + (nxt.speech or [])
                    return nxt
                return done
            if sess.pending.kind == "append" and not intent.is_question(text) and intent.goto_chapter(text) is None and (mch := re.search(r"\b(?:in|to|into|for|on) chapter (\d+)\b", intent.norm(rest if kind else text))) \
                    and len((rest if kind else text).split()) <= 12 and not re.search(r"\b(?:make|different|another|instead|shorter|longer|rewrite|sadder|change|like|as in|same)\b", intent.norm(rest if kind else text)):
                ch_ = int(mch.group(1))  # "yes, add it to chapter 2" / "no, put it in chapter 2": the paragraph is fine, it belongs in another chapter
                nums_ = self.store.chapter_numbers(sess.book)
                if ch_ not in nums_ and ch_ != max(nums_, default=0) + 1:
                    return self._reply(sess, f"There is no chapter {ch_}. The paragraph is still waiting. Tell me one of chapters 1 to {max(nums_, default=1)}.")
                sess.pending.chapter, sess.pending.ask = ch_, ""
                pos_c = intent.picture_position(rest if kind else text)
                sess.pending.data = {**(sess.pending.data or {}), "where": pos_c}
                sess.pending.ask = self._ask_for(sess, sess.pending)
                if kind == "yes":
                    return await self._accept(sess)
                r = self._reply(sess, f"Okay, it will go in chapter {ch_}. Say yes to add it.")
                r.pending = {"kind": "append", "text": sess.pending.text, "paragraph_id": None, "ask": sess.pending.ask}
                return r
            if kind == "no" and rest and sess.pending.kind in ("edit", "confirm_target", "delete", "move") and self._standalone_command(rest):
                sess.pending = None  # "No. Call chapter four Home Again": the guess was wrong and the writer said what they want instead
                return await self._handle(sid, rest, sess.book)
            if sess.pending.kind in ("edit", "confirm_target") and kind == "no" and rest and self._names_other_target(rest):
                return await self._retarget(sess, rest)  # "no, I meant paragraph three": keep what the writer asked, change where
            if sess.pending.kind == "confirm_target" and kind == "yes" and rest and not intent.is_question(rest):
                p_, sess.pending = sess.pending, None  # "yes, but make it sadder too": that paragraph, with both wishes
                return await self._propose_edit(sess, f"{p_.text}. Also: {rest}", target=self.store.find(sess.book, p_.paragraph_id), sure=True)
            if kind == "yes" and len(rest.split()) > 8 and not intent._CHANGE.search(rest) and sess.pending.kind != "confirm_target":
                kind = None  # a long remark that merely starts with "right" or "yes" is conversation, not an answer and not a change request
            if sess.pending.kind == "append" and not intent.is_question(text):
                said_p = rest if kind else text
                pos_p = intent.picture_position(said_p)
                if pos_p is not None and intent.placement_command(said_p) and not re.search(r"\b(?:make|different|another|instead|shorter|longer|rewrite|sadder|change|new)\b", intent.norm(said_p)) and len(said_p.split()) <= 10:
                    sess.pending.data = {**(sess.pending.data or {}), "where": pos_p}  # "no, put it first" / "yes, put it first": the paragraph is fine, only its place was wrong
                    sess.pending.ask = ""
                    if kind == "yes":
                        return await self._accept(sess)
                    place = "at the very start" if pos_p == 0 else "at the end" if pos_p < 0 else f"after paragraph {pos_p}"
                    return self._reply(sess, f"Okay, it will go {place} of chapter {sess.pending.chapter}. Say yes to add it.")
            if sess.pending.kind == "picture" and not intent.is_question(text):
                said = rest if kind else text
                if not kind:  # "That sounds right. Yes, put it after paragraph two.": the yes and the place are in a later sentence
                    for x in re.split(r"(?<=[.!?])\s+", text.strip())[1:]:
                        k2, r2 = intent.confirmation(x)
                        if k2 == "yes" and intent.picture_position(r2 or x) is not None:
                            kind, said = "yes", (r2 or x)
                            break
                if len(said.split()) > 16:  # "Hm. I don't like it, you can't see anything. Well, put it in anyway, at the end of the chapter."
                    said = next((x for x in reversed(re.split(r"(?<=[.!?])\s+", said.strip())) if re.search(r"\b(?:put|add|place|use)\b", intent.norm(x))), said)
                nums_ = self.store.chapter_numbers(sess.book)
                ch_new = intent.parse_ref(said).chapter if re.search(r"\bchapter \d+", intent.norm(said)) and intent.new_chapter(said) is None and intent.new_chapter_titled(said) is None else None
                if ch_new is not None and ch_new not in nums_:
                    if re.search(r"\bchapter \d+", intent.norm(said)) and not intent.picture_position(said) is None or ch_new == max(nums_, default=0) + 1:
                        return self._reply(sess, f"Chapter {ch_new} does not exist yet. Say start chapter {ch_new} first, or tell me one of chapters 1 to {max(nums_, default=1)}. The picture is still waiting.")
                    ch_new = None
                pos = intent.picture_position(said) if intent.placement_command(said) or ch_new is not None else None
                other = re.search(r"\b(?:make|different|another|instead|bigger|smaller|new|find|draw)\b", intent.norm(said))
                if kind is None and pos is None and re.search(r"\b(?:at|to) the (?:very )?end\b", intent.norm(said)) and re.search(r"\b(?:put|add|use|place)\b", intent.norm(said)):
                    pos = -1  # "well, put it in anyway, at the end of the chapter. It's real at least."
                if kind is None and re.match(r"^(?:(?:well|oh|okay|ok|alright|then|just|so)\s+)*(?:put|add|use|place) (?:it|that|this|that one|this one) in\b", intent.norm(said)) and not other:
                    kind = "yes"  # "put it in anyway": that is a yes
                if (ch_new is not None or pos is not None) and not other and len(said.split()) <= 16:
                    # "no, put it in chapter 4 after paragraph 1" / "yes, at the end": where it goes is part of the answer (a "no" about the place keeps the picture)
                    if ch_new is not None:
                        sess.pending.chapter = ch_new  # only the picture moves; where the writer is writing does not change
                        sess.pending.data["where"] = pos
                    elif pos is not None:
                        sess.pending.data["where"] = pos
                    sess.pending.ask = ""
                    sess.pending.ask = self._ask_for(sess, sess.pending)
                    if kind == "yes":
                        return await self._accept(sess)
                    r = self._reply(sess, f"Okay, it will go in chapter {sess.pending.chapter}{self._where_words(sess.pending.data.get('where')) or ', at the end'}. Say yes to put it in the book.")
                    r.pending = {"kind": "picture", "text": sess.pending.text, "paragraph_id": None, "ask": sess.pending.ask}
                    r.picture = self.images.info(sess.book, sess.pending.data["candidate"], sess.pending.data.get("caption", ""), candidate=True)
                    return r
            if kind == "hold":
                return self._reply(sess, "Okay, I'll wait. Say yes to save it, or tell me what to change.")
            if kind == "yes" and not rest:
                return await self._accept(sess)
            if kind == "yes" and rest and sess.pending.kind in ("picture", "wiki", "delete", "move", "file", "family", "page") and not intent.is_question(rest) \
                    and not intent._CHANGE.search(rest) and len(rest.split()) <= 5:
                return await self._accept(sess)  # "yes, put that one in" / "yes, save them" / "yes delete all four": more yes, not a change
            if kind == "yes" and rest and (intent.is_question(rest) or text.strip().endswith("?")) and sess.pending.kind in ("edit", "append", "wiki", "picture", "delete", "move"):
                done = await self._accept(sess)  # "yes, and who was Kay's father?": save it, then answer
                r2 = await self._converse(sess, rest)
                done.text = f"{done.text} {r2.text}".strip()
                done.speech = (done.speech or []) + (r2.speech or [])
                return done
            if kind == "no" and not rest:
                if sess.pending.kind == "picture":
                    self.images.discard(sess.book, (sess.pending.data or {}).get("candidate", ""))
                self.memory.add(f"Writer rejected a proposed {sess.pending.kind}: '{sess.pending.text[:120]}'", "preference")
                was = sess.pending.kind
                sess.pending = None
                if was in ("edit", "confirm_target") and self._work(sess) != sess.chapter:  # the look at an older chapter ends with the "no": back to where the writing is
                    sess.chapter = self._work(sess)
                    return self._reply(sess, f"Okay, I left it as it was. I'm back in chapter {sess.chapter}, where you are writing. What next?")
                if was in ("delete", "move"):
                    return self._reply(sess, "Okay, I left it as it is. Nothing was " + ("deleted." if was == "delete" else "moved."))
                return self._reply(sess, "Okay, dropped it. What would you like instead?")
            if kind and rest and sess.pending.kind != "confirm_target":
                return await self._revise(sess, rest, said_yes=(kind == "yes"))  # "yes but make it shorter" / "no, make it longer": never save or drop blindly
            if not kind and not (sess.pending.kind == "picture" and re.search(r"\byou can'?t see\b|\bcan'?t see (?:anything|much|a thing|what)\b|\b(?:picture|photo|that 1|this 1)\b", intent.norm(text))) \
                    and (r := self._ui_reply(sess, text)):  # "bigger" while reviewing a proposal: change the screen, keep the proposal
                return r
            if not kind and sess.pending.kind in ("append", "edit") and intent.is_edit(text) and (rf0 := intent.parse_ref(text)).para is None and rf0.chapter is None \
                    and not intent.is_question(text) and not intent.sentence_removal(text):
                return await self._revise(sess, text)  # "leave out the part about her living there": a change to the draft on the table, not to an old paragraph
            if not kind and sess.pending.kind != "voice" and not self._is_new_command(text):
                # a question, a look at the book, small talk, a vague remark, goodbye: none of these is an answer or a new job, so the change
                # keeps waiting (the card stays up) and I remind the writer, out loud, that it is still waiting
                keep = sess.pending
                if intent.goto_chapter(text) is not None or intent.step_direction(text) or intent.is_read_aloud(text):
                    sess.pending = None  # navigation runs as usual, then the change is put back
                    r = await self._navigate_keeping(sess, text)
                elif intent.norm(text) in ("hello", "hi", "hey", "hello there", "hi there", "are you there", "you there", "are you still there"):
                    r = self._reply(sess, "I'm here.")  # never ask the model: it can mix up which draft is waiting
                else:
                    r = await self._converse(sess, text)
                    if sess.pending is not None and sess.pending is not keep and sess.pending.kind in ("append", "edit", "picture"):
                        return r  # the model made a NEWER proposal (a revised draft): that is what a "yes" now answers, never the older one that was waiting
                sess.pending = keep
                r.pending = {"kind": keep.kind, "text": keep.text, "paragraph_id": keep.paragraph_id, "ask": keep.ask}
                if keep.ask and keep.kind != "page" and not re.search(r"\b(?:still waiting|waiting for your (?:yes|answer)|say yes or no)\b", r.text, re.I):
                    beg = f" It begins: {excerpt(keep.text, 10)}" if keep.kind in ("append", "edit") and keep.text else ""
                    reminder = f"I am still holding this for you: {keep.ask}{beg} Say yes or no."
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
                dropped, sess.pending = sess.pending, None  # anything else is a new command; the unsaved proposal is abandoned, and we say so
                sess.notices.append("I let go of the draft you had not saved.")
                sess.last_dropped = dropped
                sess.say_first = f"I dropped the {'paragraph' if dropped.kind == 'append' else 'change'} I showed you before, because you moved on. Say bring it back if you wanted it."

        if not sess.pending and (draft := self._draft_to_rescue(sess, text)):  # "yes, add it" to a draft that was shown without a proposal
            sess.pending = Pending("append", draft, self._work(sess))
            sess.history[-1]["content"] = "That paragraph is in the book now."  # so a later "yes" can never save it a second time (plain words: a model copies brackets)
            r = await self._accept(sess)
            r.text += " (That was the paragraph I just showed you.)"
            return r

        sents_ = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
        if not sess.pending and len(sents_) >= 2 and not self._standalone_command(sents_[0]) \
                and (k_ := next((i for i, x in enumerate(sents_) if self._standalone_command(x)), None)) \
                and (all(len(x.split()) <= 8 and not intent.is_question(x) for x in sents_[:k_]) or (k_ == len(sents_) - 1 and intent.book_title(sents_[k_]) is not None)):
            r0 = await self._handle(sid, " ".join(sents_[:k_]), sess.book)  # the remark first ("Buck. I like Buck."), then the command and what follows
            r1 = await self._handle(sid, " ".join(sents_[k_:]), sess.book)
            r1.text = f"{r0.text} {r1.text}".strip()
            r1.speech = (r0.speech or []) + (r1.speech or [])
            return r1
        if not sess.pending and (parts := re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)) and len(parts) == 2 and parts[1].strip() \
                and self._standalone_command(parts[0]) and not intent.is_question(parts[0]):
            first, more_text = parts[0].strip(), parts[1].strip()  # "Call chapter three Over There. Then go to chapter one." / "Start chapter four. He comes home in 1919."
            started = intent.new_chapter_titled(first) is not None or intent.new_chapter(first) is not None
            r1 = await self._handle(sid, first, sess.book)
            more_first = re.split(r"(?<=[.!?])\s+", more_text, maxsplit=1)[0]
            want_n = (intent.new_chapter_titled(first) or (None,))[0] or intent.new_chapter(first)
            if started and want_n and want_n not in self.store.chapter_numbers(sess.book):
                return r1  # the chapter was not started (it would skip one): do not rename another chapter with "call it"
            if (ci := re.match(r"(?i)^(?:and |then |um |uh )*(?:call|name|title) (?:it|this one|that one)\s+(.+)$", more_first.strip())):
                more_text = f"Call chapter {sess.chapter} {ci.group(1)}" + more_text[len(more_first):]  # "Start chapter five. Call it The Old Soldier."
            if started and sess.mode == "narrator" and not self._standalone_command(re.split(r"(?<=[.!?])\s+", more_text, maxsplit=1)[0]) and not intent.is_question(more_text) and not intent.is_draft(more_text):
                more_text = f"Write the first paragraph of this chapter: {more_text}"
            r2 = await self._handle(sid, more_text, sess.book)
            if started and sess.pending and sess.pending.kind == "append":
                r1.text = r1.text.split(". Tell me what happens")[0].rstrip(".") + "."
                r1.speech = [{"speaker": vx.ASSISTANT, "text": r1.text}]
            r2.text = f"{r1.text} {r2.text}".strip()
            r2.speech = (r1.speech or []) + (r2.speech or [])
            r2.ui = r2.ui or r1.ui
            return r2
        if not sess.pending and re.match(r"^(?:(?:oh|ok|okay|no|well|just)\s+)*(?:never ?mind|cancel(?: that| it| this)?|forget (?:it|that|about it)|scratch that|stop that)(?:\s+(?:never ?mind|cancel that|please|then|for now|it doesnt matter|it does not matter))*$", intent.norm(text)):
            sess.last_edit = None
            return self._reply(sess, "Okay, I stopped. Nothing is waiting and nothing was changed. If you want to take back the last thing I saved, say undo.")
        if intent.norm(text) in ("goodbye", "good bye", "bye bye", "goodnight", "good night"):  # a real farewell gets a farewell, never an empty answer
            return self._reply(sess, "Goodbye for now. Everything you saved is in your book.")
        if intent.is_noise(text):  # silence that the speech model turned into "Thank you."
            return Reply(text="", mode=sess.mode, chapter=sess.chapter, focus=sess.focus)

        self._learn(text)

        if (mode := intent.detect_mode(text)) and (mode != sess.mode or len(intent.norm(text).split()) <= 5):
            sess.mode = mode
            return self._reply(sess, "Switching to feedback mode. Tell me what you're working on." if mode == "therapist" else "Back to narrator mode.")

        if r := self._ui_reply(sess, text):  # "I can't see it": instant, no model call
            return r
        if (bt := intent.book_title(text)) is not None:  # "call the book Samuel's War": the title on the exported book
            b_ = self.store.bible(sess.book)
            prem = b_.read("premise") or ""
            prem = re.sub(r"(?im)^\s*title:.*\n?", "", prem).strip()
            b_.write_doc("premise", f"Title: {bt}\n\n{prem}".strip() + "\n")
            return self._reply(sess, f"Done. The book is called {bt}.")
        if (req := intent.ideas_request(text)) is not None:
            return self._ideas_reply(sess, req)
        if intent.undo_request(text):
            return self._undo_reply(sess)
        if (nt := intent.new_chapter_titled(text)) is not None:
            r = self._new_chapter_reply(sess, nt[0])
            if (more := intent.chapter_story(text)) and sess.mode == "narrator" and sess.chapter == sess.work and sess.chapter in self.store.chapter_numbers(sess.book):
                if not re.match(r"(?i)(?:write|draft)\b", more):
                    more = f"Write the first paragraph of this chapter, {more}"  # "start chapter two where they arrive in Kansas"
                d = await self._propose_draft(sess, more)
                d.text = f"{r.text.split('.')[0]}. {d.text}"
                return d
            if nt[1] and self.store.chapter_title(sess.book, sess.chapter) == "" and sess.chapter in self.store.chapter_numbers(sess.book):
                t = self.store.set_chapter_title(sess.book, sess.chapter, nt[1])
                r.text = f"Chapter {sess.chapter}, {t}, is ready. Tell me what happens in it, or say write the first paragraph."
                r.speech = [{"speaker": vx.ASSISTANT, "text": r.text}]
            return r
        if (n := intent.new_chapter(text)) is not None:
            return self._new_chapter_reply(sess, n)
        if (fmt := intent.export_request(text)) is not None:
            return self._export_reply(sess, fmt)
        if (mode := intent.autosave_mode(text)) is not None:
            if self.prefs:
                self.prefs.update({"auto_save": mode})
            return self._reply(sess, "Okay. From now on I will save each paragraph I write straight away and read it to you. Say undo to take one back, or change paragraph and the number. Say ask me first to go back to asking."
                               if mode == "on" else "Okay. I will ask you before I save anything I write.")
        if (tc := intent.title_chapter(text)) is not None:
            return self._title_reply(sess, tc[0], tc[1])
        if (dm := intent.delete_many(text)) is not None:
            return self._delete_many_reply(sess, dm[0], dm[1])
        if (sr := intent.sentence_removal(text)) is not None and sess.mode == "narrator":
            return self._remove_sentence(sess, sr[0], sr[1])
        if (ref := intent.delete_request(text)) is not None:
            return self._delete_reply(sess, ref, text)
        if (mv := intent.move_request(text)) is not None:
            return self._move_reply(sess, mv, text)
        if (what := intent.open_request(text)) and (r := self._open_reply(sess, what)):  # "open the harbour notes"
            return r
        if (lib_pic := await self._library_picture_reply(sess, text)) is not None:  # "put the old tintype from 1862 after the first paragraph"
            return lib_pic
        if intent.picture_request(text):  # "make a picture of the lake", "find a map of Camelot": a picture for the book, not an edit or a file on the computer
            return await self._converse(sess, text)
        if intent.file_request(text):  # "I just downloaded a new PDF, can you open it"
            return self._find_reply(sess, text)

        if self.voices and (cmd := intent.voice_command(text)):
            if (r := await self._voice_command(sess, text, cmd)) is not None:
                return r

        if (n := intent.goto_chapter(text)) is not None:
            if n in self.store.chapter_numbers(sess.book):
                if re.match(r"^(?:(?:please|ok|okay|now|and)\s+)*read\b", intent.norm(text)):
                    paras = self.store.paragraphs(sess.book, n)  # "read me chapter one": read all of it; the chapter being written stays where it is
                    words = [p for p in paras if not p.image]
                    sess.chapter, sess.focus = n, words[0].id if words else None
                    if not words:
                        return self._reply(sess, f"Chapter {n} has no writing yet.")
                    title = self.store.chapter_title(sess.book, n)
                    head = f"Chapter {n}" + (f", {title}." if title else ".")
                    speech: list[dict] = [{"speaker": vx.ASSISTANT, "text": head}]
                    texts = [head]
                    for p in paras:
                        if p.image:
                            cap = f"There is a picture here: {p.image[0]}."
                            texts.append(cap); speech.append({"speaker": vx.ASSISTANT, "text": cap})
                        else:
                            texts.append(imgs.plain(p.text)); speech += await self._speech_for(sess, p)
                    r = self._reply(sess, "\n\n".join(texts), speech)
                    r.highlights = self._hl(sess, [p.id for p in words])
                    return r
                sess.chapter, sess.focus, sess.work = n, None, n
                return self._reply(sess, f"Chapter {n}.")
            nxt = max(self.store.chapter_numbers(sess.book), default=0) + 1
            return self._reply(sess, f"There is no chapter {n} yet." + (f" Say start chapter {n} and I will begin it." if n == nxt else ""))
        if d := intent.step_direction(text):
            return await self._step(sess, d)
        if intent.is_read_aloud(text):
            target = self._by_ref(sess, intent.parse_ref(text))
            if target:
                sess.chapter, sess.focus = target.chapter, target.id
                r = self._reply(sess, imgs.plain(target.text), await self._speech_for(sess, target))
                r.highlights = self._hl(sess, [target.id])
                return r
            return self._reply(sess, "I couldn't find that paragraph.")

        if not sess.pending and sess.mode == "narrator" and re.match(r"^(?:no|nope|not that one),? (?:make|find|try|show|give) (?:it|me|one|another|a different)\b", intent.norm(text)) \
                and any(m["role"] == "assistant" and re.search(r"\b(?:picture|illustration|photo|image)\b", m["content"], re.I) for m in sess.history[-4:]):
            return await self._converse(sess, f"I do not want that picture. {text} (This is about the PICTURE, not the screen and not a paragraph.)")

        if sess.mode == "narrator" and intent.draft_first(text):
            return await self._propose_draft(sess, text)
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
        if not re.match(r"^(?:yes|yeah|yep|save|add|put|keep)\b", intent.norm(text)) or last.startswith("Here is your picture") \
                or re.search(r"(?i)\b(?:picture|photo|photograph|portrait|tintype|image|illustration)\b", last[:200]):
            return ""  # only a plain yes saves a paragraph that was shown without a card, and never a picture's description
        if (para_ := self._paragraph_in(last)) and not last.startswith(("Here's a draft for chapter", "Here is your picture")) and not re.search(r"\bShould I (?:add|save) it\?$", last) \
                and not re.search(r"(?i)\b(?:picture|photo|photograph|portrait|tintype|image)\b", last[:200]) and re.match(r"^(?:yes|yeah|yep|save|add|put)\b", intent.norm(text)):
            return para_  # a story paragraph was read out without a card, and the writer said yes to it
        if len(last) < 200 or not self._OFFER.search(last) or last.startswith("[") \
                or re.match(r"^(?:i can'?t|i cannot|i couldn'?t|i could not|i'm sorry|i am sorry|sorry|i do not|i don'?t|unfortunately|i'm not|i am not|okay, i|i did not|i didn'?t)\b", last.lower()):
            return ""  # only a story paragraph that was shown is ever rescued, never my own apology or explanation
        q = (re.search(r"[^.?!]*\?\s*$", last) or re.search(r"$", last)).group(0).lower()
        if not re.search(r"\b(?:add|save|put|keep|use)\b", q) or re.search(r"\b(?:picture|photo|image|illustration|wiki|note|notes|idea|ideas|page|draw|find|make|write|name|names)\b", q) \
                or re.search(r"(?m)^\s*(?:\d+[.)]|[-*\u2022])\s", last) or re.search(r"\b(?:one|1)[,:.] .+\b(?:two|2)[,:.] ", last.lower()):
            return ""  # only an offer to put THIS paragraph in the book; a list of ideas, a picture offer or a wiki offer is not a paragraph
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
        seen: dict[int, int] = {}
        lines = []
        for p in paras:  # numbered inside each chapter, the way the writer says it ("paragraph 3 of chapter 2")
            if p.image:
                lines.append(f"chapter {p.chapter} picture (not numbered): {imgs.plain(p.text)[:60]}")
                continue
            seen[p.chapter] = seen.get(p.chapter, 0) + 1
            lines.append(f"chapter {p.chapter} paragraph {seen[p.chapter]}: {' '.join(imgs.plain(p.text).split()[:14])}…")
        lines = lines[-limit:]
        return "THE BOOK SO FAR (one line per saved paragraph; continue forward, do not retell a scene that is already here):\n" + "\n".join(lines)

    async def _autosave(self, sess: Session, text: str, lead: str = "", warn: str = "") -> Reply:
        """"Save as you go": the paragraph is saved at once, read to the writer, and can still be undone with one word."""
        ch = sess.pending.chapter if sess.pending else self._work(sess)
        r = self._commit(sess)
        if not r.changed:
            return r
        said_lead, tail = f"{lead}I saved this to chapter {ch}:", f"{warn} Say undo to take it back.".strip()
        r.text = f"{said_lead} {text} {tail}"
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        short = " ".join(sentences[:2]) if len(sentences) > 2 else text  # tired ears: the first two sentences are spoken; the whole paragraph is on screen and can be read in full
        more = " I have read you the start. Say read it to me for all of it." if len(sentences) > 2 else ""
        r.speech = await self._speech_text(sess, short, said_lead, f"{more} {tail}".strip())
        return r

    def _book_state(self, sess: Session) -> str:
        nums = self.store.chapter_numbers(sess.book)
        def _count(n: int) -> str:
            ps = self.store.paragraphs(sess.book, n)
            pics = sum(1 for p in ps if p.image)
            return f"chapter {n}: {len(ps) - pics} saved paragraphs" + (f" and {pics} picture{'s' if pics != 1 else ''}" if pics else "")
        counts = ", ".join(_count(n) for n in nums[-8:]) or "no chapters yet"
        wait = (f"A proposal is waiting for the writer's yes or no ({sess.pending.ask or sess.pending.kind}). " if sess.pending else "Nothing is waiting for an answer. ")
        return (f"BOOK STATE (what is really saved): {counts}. The writer is looking at chapter {sess.chapter} and is writing in chapter {self._work(sess)}: new paragraphs and pictures go into chapter {self._work(sess)} unless they name another. {wait}"
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
                    names = ", ".join(b["name"] for b in d["batch"][:10])
                    return f"Save these {len(d['batch'])} notes to the wiki: {names}?"
                return "Save this to the book's premise?" if d.get("kind") == "premise" else f"Save this note under {d.get('name', 'the wiki')}?"
            if p.kind == "picture":
                return f"Put this picture in chapter {p.chapter}{self._where_words((p.data or {}).get('where'))}?"
            if p.kind == "delete":
                if (p.data or {}).get("ids"):
                    return f"Delete these {len(p.data['ids'])} paragraphs of chapter {p.chapter}?"
                return f"Delete {self._where(sess, self.store.find(sess.book, p.paragraph_id))}?"
            if p.kind == "move":
                w, to = (p.data or {}).get("where", "end"), (p.data or {}).get("to")
                place = {"end": f"the end of chapter {to}", "start": f"the start of chapter {to}"}.get(w) if isinstance(w, str) else f"{w[0]} paragraph {w[1]} of chapter {to}"
                return f"Move {self._where(sess, self.store.find(sess.book, p.paragraph_id))} to {place}?"
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
            sess.chapter, sess.focus, sess.work = target, None, target
            return self._reply(sess, f"Chapter {target} already exists. I have gone there.")
        if target != nxt:
            return self._reply(sess, f"That would skip a chapter. The next one is chapter {nxt}. Say start chapter {nxt}.")
        self.store.create_chapter(sess.book, target)
        sess.chapter, sess.focus, sess.work = target, None, target
        if self.index:
            self.index.schedule(sess.book)
        return self._reply(sess, f"Chapter {target} is ready. Tell me what happens in it, or say write the first paragraph.")

    def _remove_sentence(self, sess: Session, ref, which: int) -> Reply:
        """"take the last sentence of paragraph 2 out": no model needed. The shortened paragraph is shown and only a yes replaces the old one (undoable)."""
        if sess.pending and not self._supersede(sess):
            return self._reply(sess, "I am holding something for your yes or no. Answer that first, then ask again.")
        if ref is not None and ref.para is None and ref.chapter is not None:
            return self._reply(sess, f"Which paragraph of chapter {ref.chapter}? Say, for example: take the last sentence of paragraph 2 in chapter {ref.chapter} out.")
        if ref is not None and ref.para is not None:
            target = self._by_ref(sess, ref)
        elif sess.focus:
            try:
                target = self.store.find(sess.book, sess.focus)
            except BookError:
                target = None
        else:
            target = self.store.nth_paragraph(sess.book, self._work(sess), -1)
        if not target or target.image:
            return self._reply(sess, "I could not tell which paragraph. Say the chapter and paragraph, for example: take the last sentence of paragraph 2 in chapter 2 out.")
        sentences = split_sentences(target.text)
        if len(sentences) < 2:
            return self._reply(sess, f"{self._where(sess, target)[0].upper() + self._where(sess, target)[1:]} is only one sentence. Say delete the paragraph if you want it gone.")
        i = len(sentences) - 1 if which == -1 else which - 1
        if not 0 <= i < len(sentences):
            return self._reply(sess, f"That paragraph has {len(sentences)} sentences.")
        removed = sentences.pop(i)
        new = " ".join(sentences).strip()
        sess.chapter, sess.focus = target.chapter, target.id
        sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id)
        self._remember(sess, "take a sentence out", f"Shall I take that sentence out of {self._where(sess, target)}?")
        lead = f"I will take out this sentence from {self._where(sess, target)}: {removed}"
        r = self._reply(sess, f"{lead} It would read: {new} Should I save it?", [{"speaker": vx.ASSISTANT, "text": lead}, {"speaker": vx.ASSISTANT, "text": "Should I save the shorter paragraph?"}])
        r.highlights = self._hl(sess, [target.id])
        return r

    def _shorten_waiting(self, sess: Session, which: int) -> Reply:
        """"take the last sentence out" while a draft waits: shorten the draft on the table (the saved book is not touched)."""
        sentences = split_sentences(sess.pending.text)
        if len(sentences) < 2:
            return self._reply(sess, "The paragraph waiting is only one sentence. Say no to drop it, or tell me what to change.")
        i = len(sentences) - 1 if which == -1 else which - 1
        if not 0 <= i < len(sentences):
            return self._reply(sess, f"The paragraph waiting has {len(sentences)} sentences.")
        removed = sentences.pop(i)
        sess.pending.text = " ".join(sentences)
        sess.pending.ask = ""
        return self._reply(sess, f"I took out: {removed} It now reads: {sess.pending.text} {self._ask_for(sess, sess.pending)} Say yes or no.")

    def _delete_reply(self, sess: Session, ref, text: str) -> Reply:
        target = self._by_ref(sess, ref)
        if not target:
            return self._reply(sess, "I couldn't find that paragraph.")
        sess.pending = Pending("delete", target.text, target.chapter, paragraph_id=target.id)
        sess.chapter, sess.focus = target.chapter, target.id
        r = self._reply(sess, f"Delete {self._where(sess, target)}? It begins: {excerpt(target.text, 12)} You can say undo afterwards.")
        r.highlights = self._hl(sess, [target.id])
        return r

    def _delete_many_reply(self, sess: Session, chapter: int | None, numbers: list[int]) -> Reply:
        ch = chapter or sess.chapter
        if ch not in self.store.chapter_numbers(sess.book):
            return self._reply(sess, f"There is no chapter {ch}.")
        paras = [p for p in self.store.paragraphs(sess.book, ch) if not p.image]
        picked = []
        for k in numbers:
            i = k - 1 if k > 0 else len(paras) + k
            if 0 <= i < len(paras) and paras[i].id not in picked:
                picked.append(paras[i].id)
        if len(picked) < 2:
            return self._reply(sess, f"Chapter {ch} does not have those paragraphs. It has {len(paras)}.")
        sess.pending = Pending("delete", "", ch, paragraph_id=picked[0], data={"ids": picked})
        sess.chapter = ch
        r = self._reply(sess, f"Delete {len(picked)} paragraphs of chapter {ch}? The first begins: {excerpt(self.store.find(sess.book, picked[0]).text, 10)} You can say undo afterwards.")
        r.highlights = self._hl(sess, picked[:5])
        return r

    def _move_reply(self, sess: Session, mv, text: str) -> Reply:
        target = self._by_ref(sess, mv.ref)
        if not target:
            return self._reply(sess, "I couldn't find that paragraph.")
        nums = self.store.chapter_numbers(sess.book)
        to = mv.chapter or target.chapter
        if to not in nums and to != max(nums, default=0) + 1:
            return self._reply(sess, f"There is no chapter {to}, and it is not the next one. The next is chapter {max(nums, default=0) + 1}.")
        if to == target.chapter and mv.where == "end" and mv.ref.chapter is None and mv.chapter is not None:
            return self._reply(sess, f"I am looking at chapter {to}, and that paragraph is already in it. Say which chapter it is in now, for example: move paragraph {mv.ref.para} of chapter 1 to chapter {to}.")
        if to == target.chapter and mv.where == "end" and self.store.paragraphs(sess.book, to)[-1].id == target.id:
            return self._reply(sess, "It is already at the end of that chapter.")
        sess.pending = Pending("move", target.text, target.chapter, paragraph_id=target.id, data={"to": to, "where": mv.where})
        sess.chapter, sess.focus = target.chapter, target.id
        place = {"end": f"the end of chapter {to}", "start": f"the start of chapter {to}"}.get(mv.where) if isinstance(mv.where, str) else f"{mv.where[0]} paragraph {mv.where[1]} of chapter {to}"
        r = self._reply(sess, f"Move {self._where(sess, target)} to {place}? It begins: {excerpt(target.text, 12)}")
        r.highlights = self._hl(sess, [target.id])
        return r

    _CONTINUE = re.compile(r"^(?:(?:and|then|now|so|please)\s+)*(?:(?:save|add|keep) (?:it|that|this)\s+(?:and\s+)?(?:then\s+)?(?:now\s+)?)?(?:going|writing|keep going|keep writing|carry on|go on|continue|write (?:on|more|the next|another|a next|one more)\b|(?:the )?next (?:one|paragraph|bit|part)\b|(?:one|1) more\b|another (?:one|paragraph)\b|(?:and )?(?:then )?(?:the )?(?:part|scene|bit|moment) (?:where|when|in which|with)\b|write\b)")

    def _continue_after(self, text: str, rest: str) -> str:
        """After a yes: is the rest a request for MORE writing? -> the drafting request, or ""."""
        r = intent.norm(rest)
        m = self._CONTINUE.match(r)
        if not m or intent._CHANGE.search(r[: m.end()]) and not re.match(r"^(?:\w+ )*?(?:one|1) more|another", r[m.start():]):
            return ""
        tail = r[m.end():].strip()
        core = m.group(0)
        if re.search(r"\b(?:going|writing|carry on|go on|continue|next|one more|1 more|another|write on|write more)\b", core) and len(tail.split()) < 2:
            return "Write the next paragraph."
        out = re.sub(r"^(?:(?:and|then|now|so|please)\s+)*(?:(?:save|add|keep) (?:it|that|this)\s+(?:and\s+)?(?:then\s+)?(?:now\s+)?)?", "", rest.strip() or r, flags=re.I)
        if re.match(r"(?i)(?:part|scene|bit|moment)\b", out):
            out = "Write the " + out  # "yes, and then the part where he buys a ticket"
        return out or "Write the next paragraph."

    _NOT_AFTER = re.compile(r"^(?:but|except|only|though|however|make|change|can you make|could you make|shorter|longer|cut|take|remove|drop|leave|instead|not|no|without|less|more|add (?:a|that|some|more)|try)\b")

    _REQ_VERB = re.compile(r"\b(?:write|make|change|look|find|start|call|name|go|read|delete|remove|move|put|add|save|keep|tell|show|search|undo|draw|use|rewrite|fix|take|cut)\b")
    _SAVE_PHRASE = re.compile(r"^(?:(?:yes|yeah|ok|okay|so|well|then|now|please|just|and|oh)\s+)*(?:go ahead and\s+)?(?:put|add|save|use|keep) (?:it|that|this|them|that 1|this 1|the(?: \w+){1,2})(?: in| in there| into the book| in the book| on)?(?: anyway| then| now| please| for me| all the same)*$")

    def _sentence_kind(self, sess: Session, s: str) -> str:
        n = intent.norm(s)
        if not n:
            return "neutral"
        if sess.pending.kind in ("delete", "move") and re.search(r"\b(?:keep|leave|don'?t|do not|wait|stop)\b", n):
            return "no"  # "No, wait. Keep it." refuses the delete
        if self._SAVE_PHRASE.match(n) and not intent._CHANGE.search(re.sub(r"\bin\b", "", n)):
            return "yes"
        k, r = intent.confirmation(s)
        if k == "no" and not r:
            return "no"
        if k == "yes" and not r and re.search(r"\b(?:yes|yeah|yep|yup|save|add|keep|put|go ahead|do it|please do|sure)\b", n):
            return "yes"
        if k == "yes" and r and re.match(r"^(?:yes|yeah|yep|yup)\b", n) and len(r.split()) <= 8 and not self._REQ_VERB.search(r) and not intent._CHANGE.search(r) \
                and not intent.is_question(s):
            return "yes"  # "Yes, that's him all the same."
        if len(n.split()) <= 7 and not intent.is_question(s) and not self._REQ_VERB.search(n) and k != "no" and not re.search(r"\b(?:not|no|don'?t|never)\b", n):
            return "neutral"  # "That's good.", "Oh that's nice.", "Hm."
        return "other"

    def _answer_then_more(self, sess: Session, text: str) -> tuple[str, str] | None:
        """The answer among several sentences: "That's good. Yes, put the draft one in. Then write about the camp." -> ("yes", "Then write about the camp.");
        "Yes. Put the picture in." -> ("yes", ""). None when there is no clear answer, or the rest is about the thing waiting ("Yes. But make it shorter.")."""
        sents = [x for x in re.split(r"(?<=[.!?])\s+|,\s+but\s+(?=(?:yes|yeah|no|okay|ok|save|put|add|keep)\b)|;\s+", text.strip(), flags=re.I) if x.strip()]
        if len(sents) < 2:
            return None
        kinds = [self._sentence_kind(sess, x) for x in sents]
        at = next((i for i, k in enumerate(kinds) if k in ("yes", "no")), None)
        if at is None or {"yes", "no"} <= set(kinds):
            return None
        if at > 0 and not re.match(r"^(?:yes|yeah|yep|yup|no|nope)\b", intent.norm(sents[at])) and not self._SAVE_PHRASE.match(intent.norm(sents[at])) \
                and not (kinds[at] == "no" and sess.pending.kind in ("delete", "move")):
            return None  # after other talk, only a plain "Yes, …" / "Put it in." counts as the answer
        k = kinds[at]
        # what comes before the answer: a remark ("I can't see his face so good") is dropped; a request ("Put her in too, and Karin") is done afterwards
        before = [x for x, kk in zip(sents[:at], kinds[:at]) if kk == "other" and self._REQ_VERB.search(intent.norm(x)) and not intent.is_question(x)]
        after = " ".join(before + [x for x, kk in zip(sents[at + 1:], kinds[at + 1:]) if kk != "yes" and not (kk == "neutral" and len(x.split()) <= 3)]).strip()
        if not after:
            return k, ""
        if self._NOT_AFTER.match(intent.norm(after)) and sess.pending.kind in ("append", "edit", "picture"):
            return None  # "No. Make it shorter." / "Yes. But at the end.": about the thing waiting
        if k == "no" and sess.pending.kind in ("append", "edit"):
            return None  # "No. He never said that." is a revision of the draft
        return k, after

    def _standalone_command(self, rest: str) -> bool:
        """Something that is clearly a request of its own (not a description of which paragraph)."""
        return bool(intent.title_chapter(rest) or intent.goto_chapter(rest) is not None or intent.new_chapter(rest) is not None or intent.undo_request(rest)
                    or intent.new_chapter_titled(rest) is not None or intent.book_title(rest) is not None
                    or intent.export_request(rest) is not None or uisettings.parse(rest, None) is not None)

    def _names_other_target(self, rest: str) -> bool:
        n = intent.norm(rest)
        rf = intent.parse_ref(rest)
        return rf.para is not None or rf.chapter is not None or bool(re.search(
            r"\b(?:i meant|i mean|not that (?:one|1)|wrong (?:one|1|paragraph)|the (?:one|1|paragraph) (?:before|after|above|below)(?: (?:that|it|this))?|the (?:other|previous|next|last|first) (?:one|1)|(?:one|1|paragraph) (?:before|after) (?:that|it|this))\b", n))

    async def _retarget(self, sess: Session, rest: str) -> Reply:
        """The writer said no to WHICH paragraph, not to the change: find the paragraph they meant (never the rejected one) and do the same change there."""
        p, sess.pending = sess.pending, None
        original = p.text if p.kind == "confirm_target" else (p.data or {}).get("instruction", "")
        if not original:
            return self._reply(sess, "Okay, I left it as it was. Tell me the change again, with the chapter and paragraph.")
        n = intent.norm(rest)
        rejected = self.store.find(sess.book, p.paragraph_id) if p.paragraph_id else None
        target = None
        if rejected is not None:
            texts = [q for q in self.store.paragraphs(sess.book, rejected.chapter) if not q.image]
            i = next((k for k, q in enumerate(texts) if q.id == rejected.id), None)
            if i is not None and not re.search(r"\b(?:paragraph|chapter|line) \d+\b", n):
                if re.search(r"\b(?:before|above|previous|earlier)\b", n):
                    target = texts[i - 1] if i > 0 else None
                elif re.search(r"\b(?:after|below|next|following)\b", n):
                    target = texts[i + 1] if i + 1 < len(texts) else None
                elif re.search(r"\blast (?:one|1)\b", n):
                    target = texts[-1]
                elif re.search(r"\bfirst (?:one|1)\b", n):
                    target = texts[0]
                elif re.search(r"\bother (?:one|1)\b", n) and len(texts) == 2:
                    target = texts[1 - i]
        if target is None:
            rf = intent.parse_ref(rest)
            if rf.para is not None:
                if rf.chapter is None and rejected is not None:
                    rf.chapter = rejected.chapter
                target = self._by_ref(sess, rf)
            elif rf.chapter is not None:
                return self._reply(sess, f"Which paragraph of chapter {rf.chapter}? Say, for example, paragraph 2 of chapter {rf.chapter}.")
        if target is None or (rejected is not None and target.id == rejected.id):
            return self._reply(sess, "Okay, not that one. Which paragraph do you mean? Say the chapter and paragraph number.")
        return await self._propose_edit(sess, original, target=target, sure=True)

    def _is_new_command(self, text: str) -> bool:
        """Is this a new job for me (so a waiting change is put aside), rather than a question, small talk or a look at the book?"""
        return bool(intent.is_edit(text) or intent.is_draft(text) or intent.delete_request(text) or intent.move_request(text) or intent.new_chapter(text) is not None
                    or intent.ideas_request(text) is not None or intent.file_request(text) or intent.undo_request(text) or intent.detect_mode(text)
                    or intent.export_request(text) is not None or intent.book_title(text) is not None
                    or (self.voices and intent.voice_command(text)) or intent.open_request(text))

    def _commit_change(self, sess: Session) -> Reply:
        sess.dropped_stale = True
        p, sess.pending = sess.pending, None
        try:
            if p.kind == "delete":
                ids = (p.data or {}).get("ids")
                if ids:
                    self.store.delete_paragraphs(sess.book, ids)
                    text, changed = f"Deleted {len(ids)} paragraphs. Say undo if you want them back.", []
                else:
                    self.store.delete_paragraph(sess.book, p.paragraph_id)
                    text, changed = "Deleted. Say undo if you want it back.", []
            else:
                new = self.store.move_paragraph(sess.book, p.paragraph_id, (p.data or {})["to"], (p.data or {}).get("where", "end"))
                sess.chapter, sess.focus = new.chapter, new.id
                w = (p.data or {}).get("where", "end")
                place = {"end": f"the end of chapter {new.chapter}", "start": f"the start of chapter {new.chapter}"}.get(w) if isinstance(w, str) else f"{w[0]} paragraph {w[1]} of chapter {new.chapter}"
                text, changed = f"Moved to {place}. Say undo to put it back.", [new.id]
        except BookError as e:
            return self._reply(sess, f"I couldn't do that: {e}")
        self.kg.refresh(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        r = self._reply(sess, text)
        r.changed = changed
        return r

    def book_voice(self, sess: Session) -> str:
        """"third" or "first": the person the book is told in, from the premise the writer gave, else from what is already written. "" when unknown."""
        try:
            premise = self.store.bible(sess.book).read("premise").lower()
        except Exception:
            premise = ""
        if re.search(r"third[- ]person|in the third\b", premise):
            return "third"
        if re.search(r"first[- ]person|in the first\b", premise):
            return "first"
        texts = [imgs.plain(p.text) for p in self.store.all_paragraphs(sess.book) if not p.image][-6:]
        if len(texts) >= 4:
            fp = sum(len(re.findall(r"\b(?:I|my|me)\b", re.sub(r'["“”][^"“”]*["“”]|(?<!\w)[\'‘][^\'’]{2,}?[\'’](?!\w)', " ", t))) >= 2 for t in texts)
            return "first" if fp >= len(texts) - 1 else "third" if fp == 0 else ""
        return ""

    def check_voice(self, sess: Session, text: str) -> str:
        """A draft that slips into the wrong person ("I" in a book told in the third person) -> a short reason, else ""."""
        voice = self.book_voice(sess)
        bare = re.sub(r'["“”][^"“”]*["“”]|(?<!\w)[\'‘][^\'’]{2,}?[\'’](?!\w)', " ", text)  # what people say in quotes (double or single) may use I
        bare = re.sub(r"\b(?:Part|Henry|Edward|Richard|George|James|John|Louis|Charles|Chapter|Book|Act|Scene|World War|Elizabeth|Pope) [IVX]+\b", " ", bare)
        bare = re.sub(r"(?:thought|wondered|said to (?:himself|herself)|asked (?:himself|herself)|whispered|told (?:himself|herself)):\s*[^.!?]*[.!?]", " ", bare)
        hits = len(re.findall(r"\b(?:I|me|my|mine|myself|we)\b", bare))
        has_i = hits >= 2
        if voice == "third" and has_i:
            return "This book is told in the third person (he, she, Everett), but the paragraph speaks as I or we."
        if voice == "first" and not has_i and len(bare.split()) > 25:
            return "This book is told in the first person (I, my), but the paragraph tells it as he or she."
        return ""

    def check_repeat(self, sess: Session, text: str) -> str:
        """A new paragraph that mostly re-says one already in the book (any chapter) -> a short reason, else "". Plain word-overlap, no model."""
        def shingles(t: str) -> set:
            w = re.findall(r"[a-z0-9']+", imgs.plain(t).lower())
            return {" ".join(w[i:i + 4]) for i in range(max(0, len(w) - 3))}
        def content(t: str) -> set:
            return {w for w in re.findall(r"[a-z]{5,}", imgs.plain(t).lower())}
        new, cnew = shingles(text), content(text)
        if len(new) < 8:
            return ""
        for p in self.store.all_paragraphs(sess.book):
            if p.image:
                continue
            old, cold = shingles(p.text), content(p.text)
            if len(old) >= 8 and len(new & old) / min(len(new), len(old)) >= 0.35:
                return f"This repeats much of {self._where(sess, p)}, which already tells it."
            if len(cnew) >= 12 and len(cold) >= 12 and len(cnew & cold) / len(cnew | cold) >= 0.4:  # the same story in other words
                return f"This tells the same things as {self._where(sess, p)}."
        return ""

    def ungrounded(self, sess: Session, text: str) -> str:
        """Specific facts (years, dates, numbers, names) that appear nowhere in the book, the wiki, the library, or what the writer said -> a plain list, else "".
        A paragraph about a real family should not carry details nobody gave."""
        corpus = [imgs.plain(p.text) for p in self.store.all_paragraphs(sess.book) if not p.image]
        corpus += [m["content"] for m in sess.history if m.get("role") == "user" and isinstance(m.get("content"), str)]
        try:
            b = self.store.bible(sess.book)
            corpus += [b.read("premise")] + [f"{n} {e.body} {' '.join(map(str, e.state.values()))}" for k in ("characters", "world", "threads") for n, e in b.entries(k).items()]
        except Exception:
            pass
        corpus += [str(h.get("text", "")) for h in list(sess.research.values())[-12:]]  # history looked up this session (Wikipedia, web pages)
        if self.journal:  # everything the writer has said, also before the conversation was summarised ("my name is Walter")
            try:
                corpus += [str(r.get("text", "")) for r in self.journal.recent(sess.book, 400, since_screen=False) if r.get("role") == "user"]
            except Exception:
                pass
        if self.library:
            for m in self.library.items(sess.book)[:40]:
                try:
                    corpus.append(self.library.text(sess.book, m["slug"])[:60000])
                except Exception:
                    pass
            corpus.append(sess.last_user or "")
        if not (self.library and self.library.items(sess.book)):
            return ""  # a book with no documents (a novel) has nothing to ground a detail in
        story = self._pref("genre", "") == "story"  # the writer wants imagined scenes: only dates and numbers must match the papers
        hay = " ".join(corpus).lower()
        if len(hay) < 200:
            return ""
        bare = re.sub(r'["“”][^"“”]*["“”]', " ", text)
        found: list[str] = []
        for m in re.finditer(r"\b(1[5-9]\d\d|20\d\d)\b|\b\d{2,}\b", bare):
            tok = m.group(0)
            if tok.lower() not in hay and tok not in found:
                found.append(tok)
        common = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "june", "july", "august",
                  "september", "october", "november", "december", "christmas", "easter", "god", "lord", "america", "american", "europe", "english", "swedish",
                  "atlantic", "pacific", "ocean", "africa", "asia", "earth", "bible", "church", "mother", "father", "grandfather", "grandmother", "mama", "papa",
                  "jesus", "sundays", "saturdays", "christmases", "midwest", "prairie", "heaven"}
        hay_stems = set(re.findall(r"[a-z]{3,}", hay))
        for m in ([] if story else re.finditer(r"(?<![.!?\"“]\s)(?<!^)\b([A-Z][a-z]{3,})\b", bare)):
            tok = m.group(1)
            low = tok.lower()
            stem = re.sub(r"(?:'s|s|es)$", "", low)
            if low not in hay and low not in common and stem not in hay_stems and low.rstrip("s") not in common and tok not in found \
                    and not any(len(w) >= 5 and low.startswith(w) for w in hay_stems):
                found.append(tok)
        if found:
            return "I could not find " + ", ".join(found[:4]) + " in your documents, your notes or what you told me."
        if self.library and self.library.items(sess.book) and not story:  # sentences made of things the documents never mention
            hayw = set(re.findall(r"[a-z]{5,}", hay))
            weak = []
            for sen in re.split(r"(?<=[.!?])\s+", bare.strip()):
                ws = [w for w in re.findall(r"[a-z]{5,}", sen.lower())]
                if len(sen.split()) >= 12 and len(ws) >= 6 and sum(w in hayw for w in ws) / len(ws) < 0.15:
                    weak.append(sen.strip())
            if weak:
                return "These details are not in your documents or notes: " + " ".join(f"\u201c{excerpt(w, 14)}\u201d" for w in weak[:2])
        return ""

    def _supersede(self, sess: Session) -> bool:
        """A new draft or picture while an older draft/picture waits: the older one is put aside (and said out loud, and "bring it back" restores it), so the
        writer is never told to "say no first". Changes that delete, move or save notes are never replaced this way."""
        p = sess.pending
        if p is None or p.kind not in ("append", "edit", "picture"):
            return False
        sess.last_dropped, sess.pending, sess.dropped_stale = p, None, False  # (a picture's file is kept, so "bring it back" can show it again)
        sess.say_first = f"I put aside the {'picture' if p.kind == 'picture' else 'paragraph' if p.kind == 'append' else 'change'} I showed you before. Say bring it back if you want it."
        return True

    def _find_picture(self, sess: Session, text: str) -> Paragraph | None:
        """The picture a writer means by "the railway station picture" or "the picture in chapter 4": by chapter, by words in its caption, else the newest."""
        pics = [p for p in self.store.all_paragraphs(sess.book) if p.image]
        if not pics:
            return None
        ref = intent.parse_ref(text)
        if ref.chapter and (inch := [p for p in pics if p.chapter == ref.chapter]):
            return inch[-1]
        words = {w for w in re.findall(r"[a-z]{4,}", intent.norm(text)) if w not in {"describe", "picture", "photo", "photograph", "image", "about", "tell", "please", "chapter", "what", "that", "this", "again", "illustration"}}
        scored = sorted(((len(words & set(re.findall(r"[a-z]{4,}", p.image[0].lower()))), i, p) for i, p in enumerate(pics)), key=lambda t: (t[0], t[1]), reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][2]
        return None

    def _describe_picture(self, sess: Session, text: str = "") -> Reply | None:
        """Put a picture into words: the one waiting for a yes, else the one the writer was last looking at."""
        p = sess.pending
        if p and p.kind in ("append", "edit") and not self._find_picture(sess, text):
            return self._reply(sess, "That is a paragraph waiting for your yes, not a picture. Say read it all to hear it again, or yes, or tell me what to change.")
        if p and p.kind == "picture":
            d = p.data or {}
            said = (d.get("describe") or f"It is a picture with the caption: {d.get('caption', '')}.").strip()
            short = " ".join(re.split(r"(?<=[.!?])\s+", said)[:3])  # tired ears: the first three sentences
            r = self._reply(sess, short + " Say yes to put it in the book, or no for a different one.")
            r.picture = self.images.info(sess.book, d["candidate"], d.get("caption", ""), candidate=True)
            return r
        rf = intent.parse_ref(text) if text else None
        if rf and rf.chapter and not re.search(r"[a-z]{4,}", re.sub(r"\b(?:describe|picture|photo|photograph|image|chapter|please|again|tell|about|what|that|this|illustration)\b", " ", intent.norm(text))):
            inch = [p for p in self.store.all_paragraphs(sess.book) if p.image and p.chapter == rf.chapter]
            if len(inch) > 1:  # "describe the picture in chapter 2" when there are two: say all of them
                parts = []
                for k, p_ in enumerate(inch, 1):
                    kept = self.images.note(sess.book, p_.image[1]) if len(p_.image) > 1 else ""
                    parts.append(f"Picture {k}, {p_.image[0]}." + (f" {' '.join(re.split(r'(?<=[.!?])\s+', kept)[:2])}" if kept else ""))
                return self._reply(sess, f"Chapter {rf.chapter} has {len(inch)} pictures. " + " ".join(parts))
        named = self._find_picture(sess, text) if text else None
        if named:
            sess.chapter, sess.focus = named.chapter, named.id
            kept = self.images.note(sess.book, named.image[1]) if len(named.image) > 1 else ""
            r = self._reply(sess, f"It is the picture called {named.image[0]}." + (f" {' '.join(re.split(r'(?<=[.!?])\s+', kept)[:2])}" if kept else ""))
            r.highlights = self._hl(sess, [named.id])
            return r
        if sess.focus:
            try:
                para = self.store.find(sess.book, sess.focus)
            except BookError:
                para = None
            if para is not None and para.image:
                kept = self.images.note(sess.book, para.image[1]) if len(para.image) > 1 else ""
                return self._reply(sess, f"It is a picture with the caption: {para.image[0]}." + (f" {' '.join(re.split(r'(?<=[.!?])\s+', kept)[:3])}" if kept else ""))
        if self.library:  # "describe the photograph": the picture the writer added most recently, in the words the app has for it
            pics = [m for m in self.library.items(sess.book) if m.get("kind") == "image"]
            if pics:
                m = max(pics, key=lambda x: x.get("added", 0))
                seen = self.library.text(sess.book, m["slug"]).strip() if m.get("status") == "read" else ""
                if seen:
                    return self._reply(sess, f"{m['title']}: {seen[:500]}")
                return self._reply(sess, f"I have your picture, {m['title']}, but I cannot see pictures, so I cannot describe it. If you tell me what is in it, I will keep your words with it and use them in the book.")
        return None

    def _export_reply(self, sess: Session, fmt: str) -> Reply:
        label = {"html": "web page file (open it and print it to make a PDF)", "epub": "e-book file", "md": "plain text folder (a zip file)"}[fmt]
        r = self._reply(sess, f"I am saving your whole book as a {label}, with its pictures. It will appear in your downloads.")
        r.download = {"url": f"/api/books/{sess.book}/export?format={fmt}", "label": label}
        return r

    async def check_continuity(self, sess: Session, text: str, ask: str = "") -> str:
        """Does a new paragraph contradict, or retell, something already in the book? -> a short reason, or "" when it is fine (or cannot be checked)."""
        if self._pref("continuity", "on") == "off" or len(self.store.all_paragraphs(sess.book)) < 4:
            return ""
        try:
            self.kg.refresh(sess.book)
            hits = [h.para for h in self.kg.search(text, limit=5)]
            recent = self.store.all_paragraphs(sess.book)[-2:]
            seen, parts = set(), []
            for p in hits + recent:
                if p.id not in seen and not p.image:
                    seen.add(p.id)
                    parts.append(f"(chapter {p.chapter}) {p.text[:500]}")
            bible = self.store.bible(sess.book).context_for(text, budget=1500)
            outline = self._outline(sess.book, 60)
            wanted = f"\n\nTHE WRITER ASKED FOR: {ask[:300]}" if ask else ""
            out = await asyncio.wait_for(self.llm.chat([
                {"role": "system", "content": "You are a strict continuity editor for a novel. Answer in one line."},
                {"role": "user", "content": "PARTS OF THE BOOK SO FAR:\n" + "\n".join(parts) + f"\n\n{outline}\n\nWIKI FACTS:\n{bible}{wanted}\n\nNEW PARAGRAPH:\n{text}\n\n"
                 "Does the new paragraph contradict a fact above (who knows or did what, descriptions, the order of events), retell an event that was already told (in any chapter), "
                 "or fail to write what the writer asked for? Reply exactly OK, or CONFLICT: \"the exact words from the book or wiki above that it contradicts or retells\" then one short sentence, "
                 "or MISSED: (not what was asked) followed by one short sentence. Only say CONFLICT when you can quote those words exactly."}],
                temperature=0, max_tokens=100), 60)
        except Exception:
            return ""
        out = (out or "").strip()
        up = out.upper()
        if up.startswith("CONFLICT") and ":" in out:  # a contradiction must quote the book; a made-up one is ignored rather than argued over
            q = re.search(r"[\"\u201c]([^\"\u201d]{12,300})[\"\u201d]", out)
            hay = re.sub(r"\s+", " ", ("\n".join(parts) + " " + bible).lower())
            if not q or re.sub(r"\s+", " ", q.group(1).lower().strip(" .,")) not in hay:
                return ""
            out = "CONFLICT: " + (out[q.end():].strip(" .,;:-\u2014") or f"it does not fit: {q.group(1)}")  # the explanation, without the quoted words
        if (up.startswith("CONFLICT") or up.startswith("MISSED")) and ":" in out:
            reason = out.split(":", 1)[1].strip()[:240]
            return f"it does not do what you asked: {reason}" if up.startswith("MISSED") else reason
        return ""

    def _title_reply(self, sess: Session, n: int | None, title: str) -> Reply:
        n = n or self._work(sess)
        if n not in self.store.chapter_numbers(sess.book):
            return self._reply(sess, f"There is no chapter {n} yet. Say start chapter {n} first.")
        try:
            t = self.store.set_chapter_title(sess.book, n, title)
        except BookError as e:
            return self._reply(sess, str(e))
        sess.chapter = n
        return self._reply(sess, f"Chapter {n} is now called {t}.")

    describe_made = True  # look at a made picture too, so the writer hears what came out, not what was asked for

    async def propose_picture(self, sess: Session, ctx, data: bytes, caption: str, chapter: int | None, hint: str = "", describe: bool = False, after: int | None = None, made: bool = False) -> str:
        """Check a picture, keep it as a candidate, and ask the writer. -> a plain message for the model."""
        if sess.pending and not self._supersede(sess):
            return "A proposal is already waiting for the writer's answer. Do not propose another."
        nums = self.store.chapter_numbers(sess.book)
        named = intent.parse_ref(sess.last_user).chapter if re.search(r"\bchapter \d+", intent.norm(sess.last_user)) else None
        ch = named or (sess.viewing_now if sess.viewing_now in nums else (chapter or self._work(sess)))  # the chapter the writer named, else the one the page shows (not the model's guess), else where they are writing
        if ch not in nums and ch != max(nums, default=0) + 1:
            ch = self._work(sess)
        try:
            name = self.images.stage(sess.book, data, hint or caption)
        except PictureError as e:
            return f"Error: {e}"
        cap = imgs.clean_caption(caption)
        where = after if isinstance(after, int) else intent.picture_position(sess.last_user, strict=True)  # the writer's own words win over the model's guess
        if isinstance(after, int) and (said := intent.picture_position(sess.last_user, strict=True)) is not None:
            where = said
        sess.pending = Pending("picture", cap, ch, data={"candidate": name, "caption": cap, "where": where})
        ctx.picture = {"candidate": name, "caption": cap, "chapter": ch}
        seen = ""
        if made:
            describe = self.describe_made
        if describe and self.llm:  # a picture that was found, not made: let a vision model look at it, so it can be described to the writer
            try:
                seen = await asyncio.wait_for(ingest.describe_picture(self.llm, self.images.path(sess.book, name, candidate=True).read_bytes()), 90)
            except Exception:
                seen = ""
        sess.pending.data["describe"] = seen
        if not describe or made:
            sess.pending.data["gen_prompt"] = (hint or "")[:300]
        how = (f" A vision model says the picture shows: {seen}" if seen else ("" if made else " You cannot see this picture and no vision model could: say so honestly and describe only what its title and caption tell you.")) if describe else ""
        return ("Proposed. The app shows the picture to the writer and asks yes or no by itself. In your reply, DESCRIBE the picture in one or two plain sentences "
                "(the writer may not look at the screen) and do not ask about saving." + how)

    def _place_image(self, sess: Session, chapter: int, name: str, caption: str, where: int | None) -> Paragraph:
        """The picture goes after `where` paragraphs (0 = the very start); anywhere else, or nothing said, it goes at the end of the chapter."""
        paras = self.store.paragraphs(sess.book, chapter)
        texts = [i for i, p in enumerate(paras) if not p.image]  # "after paragraph 2" counts the writing, not other pictures
        if where is None or where < 0 or not paras:
            return self.store.append_image(sess.book, chapter, name, caption)
        at = 0 if where == 0 else (texts[where - 1] + 1 if where <= len(texts) else len(paras))
        if at >= len(paras):
            return self.store.append_image(sess.book, chapter, name, caption)
        return self.store.insert_paragraph(sess.book, chapter, imgs.markdown(caption, name), before_id=paras[at].id)

    @staticmethod
    def _where_words(where: int | None) -> str:
        return "" if where is None or where < 0 else (", at the very start" if where == 0 else f", after paragraph {where}")

    def _commit_picture(self, sess: Session) -> Reply:
        sess.dropped_stale = True
        p, sess.pending = sess.pending, None
        d = p.data or {}
        try:
            if p.chapter not in self.store.chapter_numbers(sess.book):
                self.store.create_chapter(sess.book, p.chapter, auto=True)
            name = self.images.accept(sess.book, d["candidate"])
            self.images.note(sess.book, name, d.get("describe") or d.get("gen_prompt") or "")
            para = self._place_image(sess, p.chapter, name, d.get("caption", ""), d.get("where"))
        except (FileNotFoundError, BookError, PictureError) as e:
            return self._reply(sess, f"I couldn't add the picture: {e}")
        sess.focus = para.id
        self.kg.refresh(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        r = self._reply(sess, f"The picture is in chapter {p.chapter}{self._where_words(d.get('where'))}.")
        r.changed = [para.id]
        r.highlights = self._hl(sess, [para.id])
        return r

    def _undo_reply(self, sess: Session) -> Reply:
        """Put the book back as it was before the last saved change."""
        sess.dropped_stale = True
        try:
            done = self.store.undo_last(sess.book)
        except BookError as e:
            return self._reply(sess, str(e))
        if not done:
            return self._reply(sess, "There is nothing to undo.")
        sess.chapter, sess.focus = done["chapter"], done["paragraph"] if done["what"] in ("edit", "delete") else None
        gone = done.get("removed")
        if gone:
            last = max(self.store.chapter_numbers(sess.book), default=0) or None
            if sess.chapter == gone:
                sess.chapter, sess.focus = last, None
            if sess.work == gone:
                sess.work = last or 0
        self.kg.refresh(sess.book)
        if self.index:
            self.index.schedule(sess.book)
        said = {"edit": "I took back the change I just saved", "append": "I took back the paragraph I just added", "delete": "I put back the paragraph I deleted",
                "move": "I moved the paragraph back where it was", "delete_many": "I put back the paragraphs I deleted"}.get(done["what"], "I took back the last change")
        after = f"Chapter {gone} was new, so it is gone again." if gone and done["chapter"] == gone else f"Chapter {done['chapter']} is as it was before."
        if gone and done["chapter"] != gone:
            after += f" Chapter {gone} was new, so it is gone again."
        r = self._reply(sess, f"Done. {said}. {after}")
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
            if self.journal:
                bits += [r["text"] for r in self.journal.recent(book, 40, since_screen=False) if r["role"] == "user"]  # what the writer has said about the book so far
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
        r = self._reply(sess, imgs.plain(paras[i].text), await self._speech_for(sess, paras[i]))
        r.highlights = self._hl(sess, [ids[i]])
        self._prefetch(sess, paras[i + 1 : i + 3])  # work out who speaks the next lines while this one is being read
        return r

    # --- target resolution -----------------------------------------------
    def _by_ref(self, sess: Session, ref: intent.Ref) -> Paragraph | None:
        if ref.para is None:
            return None
        return self.store.nth_paragraph(sess.book, ref.chapter or sess.chapter, ref.para, ref.scene, picture=ref.picture)

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
        if para.image:
            return f"the picture in chapter {para.chapter}"
        ids = [p.id for p in self.store.paragraphs(sess.book, para.chapter) if not p.image]  # pictures are not counted: the writer counts the writing
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
        if ctx.download:
            reply.download = ctx.download
        if ctx.notes:
            reply.notes += ctx.notes
        if ctx.open_item:
            reply.open_item = ctx.open_item
        if ctx.ui:
            self._save_look(ctx.ui)
            reply.ui = ctx.ui
        if ctx.ideas and not self._ANIMAL.search(intent.norm(heard or "")):  # the name lists are people's names: no buttons for the farm dog
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

    _DOC_TALK = re.compile(r"\b(?:papers?|documents?|files?|letters?|obituar(?:y|ies)|card|cards|pension|muster|enlist\w*|records?|certificates?|newspaper|clipping|family tree|tree|what (?:does|do|did) (?:it|they) say|word for word|read (?:me )?(?:the|that|his|her))\b")

    def _doc_extracts(self, sess: Session, text: str, budget: int = 7000) -> str:
        """The writer asks what their papers say: hand the model the papers' actual text (the ones named, else all of them, shortest first)."""
        if not self.library or not self._DOC_TALK.search(intent.norm(text)):
            return ""
        items = [m for m in self.library.items(sess.book) if m.get("status") == "read" and m.get("kind") in ("document", "text", "scan", "family_tree")]
        if not items:
            return ""
        n = set(re.findall(r"[a-z]{4,}", intent.norm(text)))
        named = [m for m in items if n & set(re.findall(r"[a-z]{4,}", (m.get("title", "") + " " + m.get("filename", "")).lower().replace("_", " ")))]
        pick = named or sorted(items, key=lambda m: m.get("chars", 0))
        out, left = [], budget
        for m in pick:
            try:
                t = self.library.text(sess.book, m["slug"])
            except Exception:
                continue
            piece = t[: max(400, min(len(t), left // max(1, len(pick))))]
            out.append(f"=== {m['title']} ===\n{piece}")
            left -= len(piece)
            if left <= 400:
                break
        return ("THE WRITER'S OWN PAPERS (the real text; when you say what they say, use only these words, never guess or fill in):\n" + "\n\n".join(out)) if out else ""

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
        recall = await self._recall(sess, text)
        if docs := self._doc_extracts(sess, text):  # "what do the papers say": the real words in front of the model, so it cannot make them up
            recall = (recall + "\n\n" if recall else "") + docs
        msgs = self._context(sess, text, canon=canon, sparks=self._sparks(sess, text), tools=True, recall=recall)
        had_pending, sig0 = sess.pending is not None, self._book_sig(sess.book)
        out, _ = await harness.run(self.llm, msgs, self.toolbox, ctx, max_steps=3, temperature=self._temperature())
        if re.fullmatch(r"\s*[\[(][^\]\)]{0,160}[\])]\s*", out or "") and not ctx.drafted and not ctx.picture:
            # the model wrote a note ("[proposed a new paragraph]") instead of doing it: ask once more, plainly
            out, _ = await harness.run(self.llm, msgs + [{"role": "assistant", "content": out}, {"role": "user", "content":
                    "That was only a note, and nothing happened. Do what the writer asked now: to offer a paragraph call propose_append with the whole paragraph as its text; "
                    "for a picture call make_picture or find_pictures; otherwise answer in plain spoken words."}], self.toolbox, ctx, max_steps=2, temperature=self._temperature())
            if re.fullmatch(r"\s*[\[(][^\]\)]{0,160}[\])]\s*", out or ""):
                out = ""
        if not (out or "").strip() and not ctx.drafted and not ctx.picture and not sess.pending:
            try:  # the model gave nothing: ask once more without tools rather than say "I'm not sure what to say"
                out = harness.parse_calls(await self.llm.chat(msgs + [{"role": "user", "content": "Answer the writer now in plain spoken words (1-3 sentences), or write the paragraph they asked for."}], temperature=self._temperature()))[1]
            except Exception:
                out = ""
        out = (out or "").strip() or ("Goodbye for now. Everything you saved is in your book." if re.search(r"\b(?:good ?bye|bye|see you|talk (?:to you )?later)\b", text.lower()) else "I'm not sure what to say to that. Could you tell me a little more?")
        if ctx.picture and sess.pending and sess.pending.kind == "picture":  # the model made or found a picture: show it, describe it, ask
            cap = ctx.picture["caption"]
            gp = (sess.pending.data or {}).get("gen_prompt", "")
            described = len(out.strip()) >= 40 and out.strip().lower().rstrip('.') != cap.lower().rstrip('.')  # a real description, not just "Here you go" or the caption
            out = " ".join(x for x in re.split(r"(?<=[.!?])\s+", out.strip()) if not re.search(r"(?i)\bI(?:'ve| have)? (?:placed|put|added|inserted|saved)\b|\bshould i put\b|\byes or no\b", x))
            described = len(out.strip()) >= 40 and out.strip().lower().rstrip('.') != cap.lower().rstrip('.')
            seen_ = ((sess.pending.data or {}).get("describe") or "").strip()
            if seen_ and len(seen_) >= 40:  # what a vision model saw in THIS picture beats the chat model's words (it once described the wrong picture)
                out, described = excerpt(seen_, 50), True
            lead = (out.strip() + " ") if described and len(out.strip()) <= 420 else (f"I made a picture: {excerpt(gp, 30).rstrip('.… ')}. " if gp else f"I have a picture for you, called {cap}. ")
            if sess.pending.data is not None:  # what was said about the picture, kept so "describe it to me" can say it again
                sess.pending.data["describe"] = out.strip() if described else (gp or sess.pending.data.get("describe", ""))
            sess.history.append({"role": "assistant", "content": f"{lead}Should I put it in chapter {sess.pending.chapter}? (The picture \u201c{cap}\u201d is waiting for a yes.)"})
            reply = self._reply(sess, f"{lead}Should I put it in chapter {sess.pending.chapter}?")
            reply.picture = self.images.info(sess.book, ctx.picture["candidate"], cap, candidate=True)
            return self._attach(sess, reply, ctx, "", text)
        if ctx.drafted and sess.pending and sess.pending.kind == "append":  # the model proposed a paragraph: show it, hear it, and ask
            tail = f"Should I add it to chapter {sess.pending.chapter}?" if sess.pending.chapter in self.store.chapter_numbers(sess.book) else f"Should I start chapter {sess.pending.chapter} with it?"
            lead = (out.strip() + " ") if out.strip() and len(out.strip()) <= 220 and out.strip() != "I'm not sure what to say to that. Could you tell me a little more?" else ""
            sess.history.append({"role": "assistant", "content": f"Here's a draft for chapter {sess.pending.chapter}: {ctx.drafted} Should I add it?"})
            warn = f" A warning: {ctx.warning}" if ctx.warning else ""
            if ctx.warning:
                sess.pending.data = {**(sess.pending.data or {}), "warning": ctx.warning}
                tail = "It may not be what you asked for. Say yes to keep it anyway, or tell me what to change."
            if self._pref("auto_save", "off") == "on" and not ctx.warning:  # a paragraph the app itself flags is never saved behind the writer's back
                if re.search(r"yes or no|ready for|waiting|say yes|should i|shall i|would you like|your answer|once it'?s saved|after (?:you|it)", lead, re.I):
                    lead = ""  # it is being saved, not asked about: never say both
                return self._attach(sess, await self._autosave(sess, ctx.drafted, lead, warn), ctx, "", text)
            reply = self._reply(sess, f"{lead}Here's a draft: {ctx.drafted}{warn} {tail}", await self._draft_speech(sess, ctx.drafted, f"{lead}Here's a draft:", f"{warn} {tail}".strip()))
            return self._attach(sess, reply, ctx, "", text)
        if (not ctx.drafted and not sess.pending and sess.mode == "narrator" and self._looks_like_paragraph(out) and not intent.is_question(text)
                and re.match(r"^(?:and |ok |okay |now |please )*(?:one more|another|next|more|then|continue|write|and then)\b", intent.norm(text))):
            sess.history.pop()  # the model wrote prose into the chat instead of proposing it: ask again through the proper draft path, so it is checked, shown and can be saved
            return await self._propose_draft(sess, text)
        if not sess.pending and not ctx.drafted and sess.mode == "narrator" and re.search(r"\b(?:write|paragraph|draft|story|scene|next part|chapter)\b", intent.norm(text)) \
                and not intent.is_question(text) and (para_ := self._paragraph_in(out)):
            await self.toolbox.call("propose_append", {"text": para_}, ctx)  # the model wrote the paragraph into the chat: offer it properly, with a card and a yes
            if ctx.drafted and sess.pending and sess.pending.kind == "append":
                tail = f"Should I add it to chapter {sess.pending.chapter}?"
                sess.history.append({"role": "assistant", "content": f"Here's a draft: {ctx.drafted} {tail}"})
                return self._attach(sess, self._reply(sess, f"Here's a draft: {ctx.drafted} {tail}", await self._draft_speech(sess, ctx.drafted, "Here's a draft:", tail)), ctx, "", text)
        if not sess.pending and self._book_sig(sess.book) == sig0 and self._CLAIM.search(out) and not set(ctx.calls) & {"start_chapter", "set_chapter_title", "set_ui", "export_book"} \
                and sess.mode == "narrator" and re.search(r"\b(?:write|tell|paragraph|part where|the part|scene|story|draft)\b", intent.norm(text)) and not intent.is_question(text):
            return await self._propose_draft(sess, text)  # a request for writing that the model only claimed to do: write it through the draft path
        if not sess.pending and self._book_sig(sess.book) == sig0 and self._CLAIM.search(out) and not set(ctx.calls) & {"start_chapter", "set_chapter_title", "set_ui", "export_book"}:
            again, _ = await harness.run(self.llm, msgs + [{"role": "assistant", "content": out}, {"role": "user", "content":
                    "You said it is done, but nothing in the book changed and nothing is waiting for a yes. Do it now by calling the right tool "
                    "(propose_append, propose_edit_paragraph, propose_wiki_notes, use_library_picture, make_picture, export_book…). Do not say it is done."}],
                self.toolbox, ctx, max_steps=2, temperature=self._temperature())
            if sess.pending or self._book_sig(sess.book) != sig0 or ctx.download or ctx.picture or ctx.drafted:
                return await self._after_retry(sess, ctx, again or "", text)
            # the model says it saved, moved or undid something, but nothing happened: never let the writer believe it
            out = ("I have not changed anything yet" + (" — nothing was waiting for your yes" if not had_pending else "") +
                   ". Tell me again what you want, and I will show it to you first.")
        sess.history.append({"role": "assistant", "content": out})
        return self._attach(sess, self._reply(sess, out), ctx, out, text)

    def _book_sig(self, book: str) -> tuple:
        """Changes anywhere in the book's files (chapters, wiki, library) show up here: to tell a real change from a claimed one."""
        try:
            d = self.store.book_dir(book)
            return tuple(sorted((str(p.relative_to(d)), p.stat().st_mtime_ns) for sub in ("chapters", "bible", "library") if (d / sub).is_dir()
                                for p in (d / sub).rglob("*") if p.is_file()))
        except Exception:
            return ()

    async def _after_retry(self, sess: Session, ctx, out: str, text: str) -> Reply:
        """The model did it after all on the second ask: show it the usual way."""
        if ctx.drafted and sess.pending and sess.pending.kind == "append":
            tail = f"Should I add it to chapter {sess.pending.chapter}?"
            return self._attach(sess, self._reply(sess, f"Here's a draft: {ctx.drafted} {tail}", await self._draft_speech(sess, ctx.drafted, "Here's a draft:", tail)), ctx, "", text)
        if ctx.picture and sess.pending and sess.pending.kind == "picture":
            r = self._reply(sess, f"{(out.strip() + ' ') if out.strip() else ''}Should I put it in chapter {sess.pending.chapter}?")
            r.picture = self.images.info(sess.book, ctx.picture["candidate"], ctx.picture["caption"], candidate=True)
            return self._attach(sess, r, ctx, "", text)
        msg = out.strip() if out.strip() and not self._CLAIM.search(out) else ""
        if sess.pending and not msg:
            msg = sess.pending.ask or self._ask_for(sess, sess.pending)
        return self._attach(sess, self._reply(sess, msg or "Done."), ctx, msg, text)

    _ANIMAL = re.compile(r"\b(?:dog|dogs|horse|horses|cat|cats|cow|cows|mule|mules|ox|oxen|pony|pet|puppy|calf|goat|pig|hen|rooster)\b")

    async def _library_picture_reply(self, sess: Session, text: str) -> Reply | None:
        """The writer asks to put one of THEIR pictures in the book: find it in the library by the words they use, and show it with a yes/no."""
        n = intent.norm(text)
        if not self.library or not re.search(r"\b(?:put|add|use|insert|place)\b", n) or re.search(r"\b(?:make|draw|paint|find|search|new|write|scene|tell|story)\b", n) \
                or not re.search(r"\b(?:picture|photo|photograph|tintype|portrait|snapshot|image|daguerreotype|ambrotype)\b", n):
            return None
        stop = {"the", "a", "an", "of", "my", "picture", "photo", "photograph", "file", "old", "and", "in", "at", "jpg", "png", "put", "add", "use", "insert", "place",
                "it", "after", "before", "first", "paragraph", "chapter", "this", "that", "please", "from", "into", "book", "end", "start", "there", "one", "you", "can", "i"}
        def toks(t: str) -> set:
            return {w for w in intent.norm(t.replace("_", " ").replace("-", " ")).split() if w not in stop and not w.isdigit() or (w.isdigit() and len(w) == 4)}
        want = toks(text)
        imgs_ = [m for m in self.library.items(sess.book) if m.get("kind") in ("image", "scan")]
        best = max(imgs_, key=lambda m: len(want & toks(m["title"] + " " + m.get("filename", ""))), default=None)
        if not best or not want & toks(best["title"] + " " + best.get("filename", "")) or sess.pending:
            return None
        ctx = harness.Ctx(sess=sess)
        pos = intent.picture_position(text, strict=True)
        ch = intent.parse_ref(text).chapter if re.search(r"\bchapter \d+", n) else None
        out = await self.toolbox.call("use_library_picture", {"title": best["title"], "caption": best["title"], "after": pos if pos is not None and pos >= 0 else "", "chapter": ch or ""}, ctx)
        if not (sess.pending and sess.pending.kind == "picture" and ctx.picture):
            return self._reply(sess, out if not out.startswith("Error") else "I could not put that picture in just now.")
        seen = (sess.pending.data or {}).get("describe", "")
        where = self._where_words((sess.pending.data or {}).get("where")) or ", at the end"
        r = self._reply(sess, f"Here is your picture, {best['title']}. {excerpt(seen, 45) if seen else ''} Should I put it in chapter {sess.pending.chapter}{where}?".replace("  ", " "))
        r.picture = self.images.info(sess.book, ctx.picture["candidate"], ctx.picture["caption"], candidate=True)
        sess.history.append({"role": "user", "content": text})
        sess.history.append({"role": "assistant", "content": r.text})
        return self._attach(sess, r, ctx, "", text)

    def _paragraph_in(self, out: str) -> str:
        """A story paragraph written into a chat reply, without the talk around it; "" when there is none."""
        blocks = [b.strip() for b in re.split(r"\n\s*\n", out or "") if b.strip()]
        prose = []
        for b in blocks:
            c = clean_draft(b)
            outside = re.sub(r"[\"\u201c][^\"\u201d]*[\"\u201d]", " ", c)  # what is said in dialogue may speak to "you"; the narration may not
            if len(c.split()) >= 15 and not c.rstrip().endswith("?") and not self._is_talk(c) \
                    and not re.search(r"\b(?:your|walter|i can|i will|i'?ll|shall i|would you|should i)\b", outside[:160].lower()):
                prose.append(c)
            elif prose:
                break  # the story is one run of blocks; talk after it ends it
        text = "\n\n".join(prose)
        return text if len(text.split()) >= 45 and (len(prose) > 1 or self._looks_like_paragraph(text)) else ""

    _TALK = re.compile(r"(?i)^(?:i(?:'m|'ve|'ll|'d| am| have| will| can| noticed| understand| could| couldn'?t| apologi[sz]e| see| think| made| was (?:not|unable))|you(?: are|'re)|my apologies|sorry|okay|ok|sure|here'?s|here is|let me|of course|great|wonderful|revised)\b"
                       r"|\b(?:story bible|your notes or documents|not in your documents|the writer asked|should i (?:save|add) it|shall i (?:save|add))\b")

    def _is_talk(self, text: str) -> bool:
        """The model's own talk, never story: "I can put this in, but I noticed…", "I am still having trouble with that…"."""
        t = (text or "").strip().lstrip("\"\u201c ")
        return bool(self._TALK.search(t[:200])) and not re.match(r"^[\"\u201c]", (text or "").strip())

    _CLAIM = re.compile(r"\b(?:i(?:'ve| have)? (?:now )?(?:saved|added|updated|changed|moved|deleted|removed|put (?:it|that|the|this|them)\b[^.]{0,40}\bin|undone|undid|corrected|fixed|replaced|taken out|took out)|(?:is|are|has been|have been) (?:now )?(?:saved|added|updated|in (?:the|your) book|back in|undone|moved|removed)|(?:now|is) back (?:in|where)|it'?s (?:saved|in the book|done)|(?:successfully )?exported|ready (?:for you )?to (?:download|print)|on your screen now)\b", re.I)

    @staticmethod
    def _looks_like_paragraph(out: str) -> bool:
        t = out.strip()
        sentences = re.split(r"(?<=[.!?])\s+", t)
        return len(t.split()) >= 45 and len(sentences) >= 3 and not t.endswith("?") and "[[" not in t and "?" not in sentences[-1]

    async def _propose_edit(self, sess: Session, text: str, target: Paragraph | None = None, sure: bool = True) -> Reply:
        if target is None:
            target, sure = await self._resolve(sess, text)
        if not target:
            return self._reply(sess, "Which part? Say the chapter and paragraph, or describe the scene.")
        where = self._where(sess, target)
        if target.image:
            return self._reply(sess, f"{where[0].upper() + where[1:]} is a picture, not writing. I can delete it, move it, or you can ask me for a different picture.")
        if not sure:  # a guess: read it back and wait, rather than rewriting something the writer didn't mean
            sess.chapter, sess.focus = target.chapter, target.id
            sess.pending = Pending("confirm_target", text, target.chapter, paragraph_id=target.id)
            r = self._reply(sess, f"Do you mean {where}, starting: {excerpt(target.text)} Say yes to go ahead, or tell me which part.")
            r.highlights = self._hl(sess, [target.id])
            return r
        swap = re.search(r"(?i)\b(?:change|replace|swap|make)\s+[\"\u201c']?(.{2,60}?)[\"\u201d']?\s+(?:to|with|into|for)\s+[\"\u201c']?(.{2,60}?)[\"\u201d']?\s*(?:[.,;!]|$|\s+(?:nothing|and nothing|only|just|please))", text)
        if swap and target.text.lower().count(swap.group(1).lower().strip()) == 1 and len(swap.group(1).split()) <= 6:
            old_w, new_w = swap.group(1).strip(), swap.group(2).strip()  # "just change Licking River to Raccoon Creek": a plain word swap, no model, nothing else touched
            k = target.text.lower().index(old_w.lower())
            new = target.text[:k] + new_w + target.text[k + len(old_w):]
            sess.chapter, sess.focus = target.chapter, target.id
            sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id, data={"instruction": text})
            self._remember(sess, text, f"Here is the new version of {where}: {new} Should I save it?")
            lead = f"Changing {where}: {old_w} becomes {new_w}. Here is the new version:"
            r = self._reply(sess, f"{lead} {new} Should I save it?", await self._draft_speech(sess, new, lead, "Should I save it?", target))
            r.highlights = self._hl(sess, [target.id])
            return r
        msgs = self._context(sess, text, target.text, target) + [
            {"role": "user", "content": f"Rewrite ONLY this paragraph.\nInstruction: {text}\n\nORIGINAL:\n{target.text}\n\nOutput only the new paragraph. "
                                        "If the instruction refers to something that is NOT in this paragraph (a word, a detail), do not invent it: reply with exactly NOT_FOUND: followed by a short, plain reason."}
        ]
        new, issues = await self._gated(msgs)
        if new.strip().upper().startswith("NOT_FOUND") and (rf := intent.parse_ref(text)).chapter and rf.para is None and not rf.picture:
            # "take that part out of the paragraph in chapter 2": no paragraph was named, so the guess may be the wrong one; try the others in that chapter
            for other in [p for p in self.store.paragraphs(sess.book, rf.chapter) if not p.image and p.id != target.id][:6]:
                m2 = self._context(sess, text, other.text, other) + [
                    {"role": "user", "content": f"Rewrite ONLY this paragraph.\nInstruction: {text}\n\nORIGINAL:\n{other.text}\n\nOutput only the new paragraph. If the instruction refers to something that is NOT in this paragraph (a word, a detail), reply with exactly NOT_FOUND: and a short reason."}]
                n2, i2 = await self._gated(m2)
                if not n2.strip().upper().startswith("NOT_FOUND"):
                    target, new, issues, msgs = other, n2, i2, m2
                    where = self._where(sess, target)
                    break
        if new.strip().upper().startswith("NOT_FOUND"):
            sess.last_edit = (text, target.id)
            self._remember(sess, text, "I did not change anything; that is not in " + where + ".")
            return self._reply(sess, "I did not change anything. " + (new.split(":", 1)[1].strip() if ":" in new else "That is not in this paragraph.") + " Tell me what you want instead.")
        new = clean_draft(new)
        shorten = re.search(r"\b(?:shorter|shorten|cut|trim|remove|take out|leave out|delete|drop|tighten|less)\b", intent.norm(text))
        if re.search(r"\b(?:propose_\w+|paragraph_id|text\s*=|\w+\(\s*\w+\s*=)", new) or (not re.search(r"[.!?][\"\u201d\u2019')\]]*\s*$", new) and re.search(r"[.!?][\"\u201d\u2019')\]]*\s*$", target.text)):
            # machine text, or a rewrite that lost much of the paragraph without being asked to: ask once more, plainly
            new2, _ = await self._gated(msgs + [{"role": "assistant", "content": new}, {"role": "user", "content":
                    "That is not right: give the WHOLE paragraph as plain prose, with only the asked-for change made. No code, no tool names."}])
            new = clean_draft(new2)
            if re.search(r"\b(?:propose_\w+|paragraph_id|text\s*=)", new) or (not re.search(r"[.!?][\"\u201d\u2019')\]]*\s*$", new) and re.search(r"[.!?][\"\u201d\u2019')\]]*\s*$", target.text)):
                return self._reply(sess, "I could not make that change cleanly, so I left the paragraph as it is. Tell me the exact words to change, for example: change Licking River to Raccoon Creek.")
        if re.sub(r"\s+", " ", new).strip() == re.sub(r"\s+", " ", target.text).strip():  # an "edit" that changes nothing: try once more, then say so
            msgs2 = msgs + [{"role": "assistant", "content": new}, {"role": "user", "content": "That is identical to the original. Make a real change that does what the writer asked. Output only the new paragraph."}]
            new, _ = await self._gated(msgs2)
            new = clean_draft(new)
            if re.sub(r"\s+", " ", new).strip() == re.sub(r"\s+", " ", target.text).strip():
                return self._reply(sess, "I tried, but I could not find a way to change it. Tell me exactly what you want different, for example a word to replace or a feeling to add.")
        sess.chapter, sess.focus = target.chapter, target.id
        sess.pending = Pending("edit", new, target.chapter, paragraph_id=target.id, data={"instruction": text})
        self._remember(sess, text, f"Here is the new version of {where}: {new} Should I save it?")
        lead = f"Changing {where}, which starts: {excerpt(target.text, 6)} Here is the new version:"
        reply = self._reply(sess, f"{lead} {new} Should I save it?", await self._draft_speech(sess, new, lead, "Should I save it?", target))
        reply.notes += [f"Style note: {i}" for i in issues]
        reply.highlights = self._hl(sess, [target.id])
        return reply

    async def _propose_draft(self, sess: Session, text: str) -> Reply:
        asked = intent.parse_ref(text).chapter
        nums = self.store.chapter_numbers(sess.book)
        if asked and (asked in nums or asked == (max(nums, default=0) + 1)):
            sess.work = asked  # "write paragraph 2 of chapter 3": the chapter the writer named (a new chapter is made on save)
        sess.screen_wins = False
        sess.chapter = sess.work = self._work(sess)  # otherwise the chapter being written in, not whatever paragraph was last looked at
        recent = "\n\n".join(p.text for p in self.store.paragraphs(sess.book, sess.chapter)[-4:]) if sess.chapter in self.store.chapter_numbers(sess.book) else ""
        ctx = harness.Ctx(sess=sess)
        canon = await self._canon_by_meaning(sess, text)
        msgs = self._context(sess, text, recent, canon=canon, sparks=self._sparks(sess, text, drafting=True), tools=True) + [
            {"role": "user", "content": f"{text}\n\nStory so far in this chapter (most recent):\n{recent or '(nothing yet)'}\n\nWrite ONE new paragraph of prose. Output only the paragraph."}
        ]
        new, issues = await self._gated(msgs, ctx, tools=True)
        new = clean_draft(new)
        if ctx.drafted:  # the model already proposed its paragraph through propose_append: that is the draft
            new = ctx.drafted
        if not ctx.drafted and re.match(r"(?i)^(?:my apologies|i apologi[sz]e|i'?m sorry|sorry|you'?re (?:absolutely )?right|i made a mistake|i see)\b", new or ""):
            new = ""  # an apology is not a paragraph: ask for the prose below
        if not ctx.drafted and (re.fullmatch(r"\s*[\[(][^\]\)]{0,200}[\])]\s*", new or "") or len((new or "").split()) < 4):
            again, _ = await self._gated(msgs + [{"role": "assistant", "content": new or ""}, {"role": "user", "content": "That is not a paragraph. Write the paragraph itself now: only the prose, nothing else."}])
            new = clean_draft(again)
            if re.fullmatch(r"\s*[\[(][^\]\)]{0,200}[\])]\s*", new or "") or len((new or "").split()) < 4:
                return self._reply(sess, "I could not write that paragraph just now. Please ask me again in a moment.")
        sess.pending = Pending("append", new, sess.chapter)
        self._remember(sess, text, f"Here's a draft for chapter {sess.pending.chapter}: {new} Should I add it?")
        tail = f"Should I add it to chapter {sess.chapter}?" if sess.chapter in self.store.chapter_numbers(sess.book) else f"Should I start chapter {sess.chapter} with it?"
        issue = ctx.warning if ctx.drafted else await self.check_continuity(sess, new, text)  # a paragraph proposed through the tool was already checked there
        warn = f" A warning: {issue}" if issue else ""
        if issue:
            tail = "It may not be what you asked for. Say yes to keep it anyway, or tell me what to change."
        if self._pref("auto_save", "off") == "on" and not issue:
            return self._attach(sess, await self._autosave(sess, new, "", warn), ctx, "", text)
        reply = self._reply(sess, f"Here's a draft: {new}{warn} {tail}", await self._draft_speech(sess, new, "Here's a draft:", f"{warn} {tail}".strip()))
        reply.notes += [f"Style note: {i}" for i in issues]
        return self._attach(sess, reply, ctx, "", text)

    async def _revise(self, sess: Session, instruction: str, said_yes: bool = False) -> Reply:
        """The writer answered a proposal with changes ("yes but shorter", "no, make it longer"): redo it, don't save or drop."""
        p = sess.pending
        if p.kind in ("delete", "move"):
            sess.pending = None
            return self._reply(sess, "Okay, I will not do that. Tell me again what you want.")
        if p.kind == "picture":  # "no, make it an empty lake": drop that picture for good and ask the model for a different one
            self.images.discard(sess.book, (p.data or {}).get("candidate", ""))
            sess.pending = None
            if re.search(r"\b(?:forget|never ?mind|no picture|drop|leave it|skip|not now|no more pictures|without (?:a|the) picture)\b", intent.norm(instruction)):
                return self._reply(sess, "Okay, no picture. Nothing was added.")
            return await self._converse(sess, f"I do not want that picture. {instruction}")
        if p.kind in ("family", "page"):
            sess.pending = None
            return self._reply(sess, "Okay, I won't add it yet. Ask me again with the person and how many generations you want.")
        if p.kind == "wiki" and (p.data or {}).get("batch"):
            batch = p.data["batch"]
            m = re.search(r"\b(?:leave|take|keep)\s+(.+?)\s+(?:out|off)\b|\b(?:without|except|skip|exclude|ignore|drop|not)\s+(.+)$", intent.norm(instruction))
            gone = {w for w in re.findall(r"[a-z]{3,}", (m.group(1) or m.group(2)) if m else "") if w not in {"the", "page", "pages", "note", "notes", "and", "his", "her"}}
            kept = [b for b in batch if not (gone & set(re.findall(r"[a-z]{3,}", b["name"].lower())))] if gone else batch
            if gone and 0 < len(kept) < len(batch):
                p.data["batch"] = kept
                p.ask = ""
                p.ask = self._ask_for(sess, p)
                left = [b["name"] for b in batch if b not in kept]
                if said_yes:  # "yes, but leave Erik out": the rest is saved now, no second yes
                    done = self._commit_wiki(sess)
                    done.text = f"I left out {', '.join(left)}. {done.text}"
                    return done
                r = self._reply(sess, f"Okay, I left out {', '.join(left)}. {p.ask} Say yes or no.")
                r.pending = {"kind": p.kind, "text": p.text, "paragraph_id": p.paragraph_id, "ask": p.ask}
                return r
            return self._reply(sess, "Say yes to save all of them, or no to drop them. To leave one out, say which, for example: leave Erik out.")
        if p.kind == "wiki":
            new = (await self.llm.chat([{"role": "system", "content": "Rewrite the note for a story wiki. Plain facts, at most 80 words. Output only the note."},
                                        {"role": "user", "content": f"NOTE:\n{p.text}\n\nChange requested: {instruction}"}], temperature=0.3, max_tokens=250)).strip()
            p.text = new or p.text
            return self._reply(sess, f"Revised: {p.text} Should I save it to the wiki?")
        if re.fullmatch(r"(?:please )?(?:fix|correct|change|sort|mend) (?:it|that|this|that one)(?: please| then| now| up)?|(?:do|make) (?:that|it) right", intent.norm(instruction)) and (p.data or {}).get("warning"):
            instruction = f"Fix this problem: {p.data['warning']}"  # "Yes, fix it": the problem I warned about
        if p.kind == "edit":
            orig = self.store.find(sess.book, p.paragraph_id)
            msgs = self._context(sess, instruction, orig.text, orig) + [{"role": "user", "content":
                f"Rewrite ONLY this paragraph.\n\nORIGINAL:\n{orig.text}\n\nYOUR PREVIOUS ATTEMPT (not saved):\n{p.text}\n\n"
                f"The writer now says: {instruction}\nRevise the previous attempt accordingly. Output only the new paragraph."}]
        else:
            msgs = self._context(sess, instruction, p.text) + [{"role": "user", "content":
                f"YOUR PREVIOUS DRAFT (not saved):\n{p.text}\n\nThe writer now says: {instruction}\nRevise the draft accordingly. Output only the paragraph."}]
        new, issues = await self._gated(msgs)
        new = clean_draft(new)
        if not new or re.match(r"(?i)^(?:my apologies|i apologi[sz]e|i'?m sorry|sorry|i couldn'?t|i could not|i can'?t|i cannot)\b", new):
            return self._reply(sess, f"I could not change it that way. The draft is still waiting: {p.ask or self._ask_for(sess, p)} Say yes, or tell me what to change.")
        p.text = new
        self._remember(sess, instruction, f"Here is the revised draft: {new} Should I save it?")
        reply = self._reply(sess, f"Revised: {new} Should I save it?", await self._draft_speech(sess, new, "Revised:", "Should I save it?"))
        reply.notes += [f"Style note: {i}" for i in issues]
        return reply

    async def _accept(self, sess: Session) -> Reply:
        if sess.pending.kind == "picture":
            return self._commit_picture(sess)
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
        def base(n: str) -> str:
            return re.sub(r"^(?:sir|lady|lord|king|queen|dame|master|mr|mrs|miss|old|young|saint|st|the|prince|princess)\.?\s+", "", n.strip().lower())
        old = next((e for e in bible.entries(kind).values() if e.name.lower() == name.lower()), None) or \
            next((e for e in bible.entries(kind).values() if base(e.name) == base(name)), None)  # "Sir Kay" and "Kay" are one page
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
        sess.dropped_stale = True
        p, sess.pending = sess.pending, None
        if p.kind in ("edit", "append") and not imgs.parse(p.text.strip()):
            p.text = clean_draft(p.text) or p.text  # the last guard: no lead-in, no "Should I save it?", no wrapping quotes ever reach the book
            if self._is_talk(p.text):
                return self._reply(sess, "That was my own talk, not a paragraph for the book, so I did not save it. Ask me again and I will write the paragraph itself.")
            if p.kind == "append" and (re.match(r"(?i)^author'?s note\b", p.text) or re.search(r"\bauthor'?s note\b", intent.norm(sess.last_user or "") + " " + " ".join(m["content"] for m in sess.history[-4:] if m.get("role") == "user").lower())):
                note = re.sub(r"(?i)^author'?s note[:.\s-]*", "", p.text).strip()
                self.store.bible(sess.book).write_doc("authors_note", note + "\n")
                return self._reply(sess, "Saved as the author's note. It is printed at the front of the book, before chapter 1.")
        try:
            if p.kind == "edit":
                para = self.store.replace_paragraph(sess.book, p.paragraph_id, p.text)
            else:
                if p.chapter not in self.store.chapter_numbers(sess.book):
                    self.store.create_chapter(sess.book, p.chapter, auto=True)
                where = (p.data or {}).get("where")
                paras_ = self.store.paragraphs(sess.book, p.chapter)
                texts_ = [i for i, q in enumerate(paras_) if not q.image]
                pieces = [x.strip() for x in re.split(r"\n\s*\n", p.text) if x.strip()] or [p.text]  # a draft of several paragraphs is saved as several paragraphs
                if where is not None and where >= 0 and paras_ and (at := 0 if where == 0 else (texts_[where - 1] + 1 if 0 < where <= len(texts_) else len(paras_))) < len(paras_):
                    for piece in pieces:
                        para = self.store.insert_paragraph(sess.book, p.chapter, piece, before_id=paras_[at].id)
                else:
                    for piece in pieces:
                        para = self.store.append_paragraph(sess.book, p.chapter, piece, p.scene)
                sess.work = p.chapter
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
        if para.image:  # a picture is read out as its caption
            return [{"speaker": vx.NARRATOR, "text": imgs.plain(para.text)}]
        paras = self.store.paragraphs(sess.book, para.chapter)
        i = next((k for k, p in enumerate(paras) if p.id == para.id), 0)
        return await self._segments(sess, para.text, paras[max(0, i - 12) : i])

    async def _draft_speech(self, sess: Session, body: str, lead: str, tail: str, target: Paragraph | None = None) -> list[dict]:
        """A draft waiting for a yes is read out loud. A long one is read from its start only (tired ears), and "read it all" gives the rest; "read my drafts in full"
        turns that off. The whole text is always on the screen."""
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        if len(sentences) <= 3 or self._pref("draft_read", "short") == "full":
            return await self._speech_text(sess, body, lead, tail, target)
        start = " ".join(sentences[:2])
        return await self._speech_text(sess, start, lead, f"That is only the start. Say read it all to hear the rest. {tail}".strip(), target)

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
            # (not shown to the writer: a background guess with internal ids, and too often wrong to read out to someone listening)

    # --- memory ----------------------------------------------------------
    def _learn_genre(self, text: str) -> None:
        n = intent.norm(text)
        if self.prefs and re.search(r"\b(?:story based on|based on (?:his|her|their|the) (?:life|papers|documents)|historical fiction|(?:fill|filled) in (?:the )?scenes|scenes filled in|some of it imagined|make up (?:the )?scenes)\b", n):
            self.prefs.update({"genre": "story"})  # a story based on the papers: imagined scenes are wanted, so they are not flagged as "not in your documents"
        elif self.prefs and re.search(r"\b(?:only what the papers say|just the facts|nothing made up|true family history|only the facts|family history only)\b", n):
            self.prefs.update({"genre": "history"})

    def _learn(self, text: str) -> None:
        if PREF.search(text):
            self.memory.add(f"Writer said: {text[:300]}", "preference")

    @staticmethod
    def _plain_talk(text: str) -> str:
        """Spoken replies carry no markdown: no stars, no bullet marks, no headings (they would be read out or shown as symbols)."""
        t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text or "")
        t = re.sub(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+", "", t)
        t = re.sub(r"(?m)^#{1,6}\s*", "", t)
        return re.sub(r"(?<!\*)\*(?!\*)", "", t)

    def _reply(self, sess: Session, text: str, speech: list[dict] | None = None) -> Reply:
        if sess.pending and not sess.pending.ask:
            sess.pending.ask = self._ask_for(sess, sess.pending) or text
        pend = {"kind": sess.pending.kind, "text": sess.pending.text, "paragraph_id": sess.pending.paragraph_id, "ask": sess.pending.ask} if sess.pending else None
        notices, sess.notices = sess.notices, []
        text = self._plain_talk(text) if not (sess.pending and sess.pending.kind in ("append", "edit") and sess.pending.text in (text or "")) else text
        lead = sess.say_first
        if lead and lead.startswith("I put aside") and sess.pending and sess.last_dropped is not None and sess.pending.kind == sess.last_dropped.kind \
                and sess.pending.kind in ("append", "edit"):
            a_, b_ = set(re.findall(r"[a-z]{4,}", sess.pending.text.lower())), set(re.findall(r"[a-z]{4,}", sess.last_dropped.text.lower()))
            if a_ and b_ and len(a_ & b_) / min(len(a_), len(b_)) > 0.5:
                lead = sess.say_first = ""  # the new draft is a revision of the one it replaced: no need to say anything was put aside
        if lead:  # said out loud as well as shown, because the writer may not be looking at the screen
            sess.say_first = ""
            text = f"{lead} {text}".strip()
            if speech is not None:
                speech = [{"speaker": vx.ASSISTANT, "text": lead}] + speech
        if speech is None:
            speech = [{"speaker": vx.ASSISTANT, "text": text}] if text else []
        return Reply(text=text, mode=sess.mode, chapter=sess.chapter, focus=sess.focus, pending=pend, notes=notices, speech=speech)
