"""The tools the model can use (see harness.py). All read-only with respect to the manuscript.

Each tool is a small async function; `build(agent)` wires them to one Agent. They return short text the model reads, and may leave
things in the turn's Ctx for the writer's screen (highlights, a new look, tap-to-use idea buttons).
"""
from __future__ import annotations

import asyncio

import uisettings
from harness import Ctx, Tool, Toolbox
from ideas import KINDS, Ideas


def _int(v, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def build(agent) -> Toolbox:
    store, kg = agent.store, agent.kg

    def book(ctx: Ctx) -> str:
        return ctx.sess.book

    async def random_ideas(ctx: Ctx, kind="name", n=5, culture="any", gender="any", type="village"):
        if kind not in KINDS:
            return f"Unknown kind {kind!r}. Kinds: {', '.join(KINDS)}."
        out = agent.ideas.generate(kind, _int(n, 5, 1, 12), culture=culture, gender=gender, type=type)
        ctx.ideas = out
        return out

    async def roll_sparks(ctx: Ctx, n=3):
        return agent.ideas.sparks(_int(n, 3, 1, 5))

    async def search_book(ctx: Ctx, query="", k=4):
        k = _int(k, 4, 1, 8)
        hits = [h.para for h in kg.search(query, limit=k)]
        if agent.index and agent.index.status(book(ctx))["ready"]:
            sem = await agent.index.search(book(ctx), query, k=k)
            by_id = {p.id: p for p in store.all_paragraphs(book(ctx))}
            for pid, _ in sem:
                if pid in by_id and by_id[pid].id not in {h.id for h in hits}:
                    hits.append(by_id[pid])
        if not hits:
            return "Nothing in the book matches that."
        return [{"where": agent._where(ctx.sess, p), "id": p.id, "text": p.text[:240]} for p in hits[:k]]

    async def read_passage(ctx: Ctx, chapter=1, paragraph=1):
        p = store.nth_paragraph(book(ctx), _int(chapter, 1, 1, 9999), _int(paragraph, 1, -1, 9999))
        return p.text if p else "There is no such paragraph."

    async def bible_lookup(ctx: Ctx, query=""):
        b = store.bible(book(ctx))
        out = []
        for kind in ("characters", "threads", "world"):
            for e in b.mentioned(kind, query):
                out.append(f"{kind[:-1].upper()} {e.name}: {e.summary()}")
        canon = await agent._canon_by_meaning(ctx.sess, query)
        if canon:
            out.append(canon)
        return "\n\n".join(out) or "The story bible has nothing about that yet."

    async def book_info(ctx: Ctx):
        b = book(ctx)
        paras = store.all_paragraphs(b)
        bible = store.bible(b)
        return {
            "chapters": len(store.chapter_numbers(b)), "paragraphs": len(paras), "words": sum(len(p.text.split()) for p in paras),
            "current_chapter": ctx.sess.chapter, "characters": list(bible.characters()),
            "open_threads": [t.name for t in bible.threads().values() if str(t.state.get("status", "open")) != "resolved"],
        }

    async def show_passage(ctx: Ctx, ids=None, chapter=None, paragraph=None):
        want = [ids] if isinstance(ids, str) else list(ids or [])
        if chapter is not None and paragraph is not None:
            p = store.nth_paragraph(book(ctx), _int(chapter, 1, 1, 9999), _int(paragraph, 1, -1, 9999))
            if p:
                want.append(p.id)
        known = {p.id for p in store.all_paragraphs(book(ctx))}
        good = [i for i in want if i in known][:5]
        ctx.highlights += [i for i in good if i not in ctx.highlights]
        return f"Highlighted {len(good)} passage(s) on the writer's screen." if good else "Nothing to highlight: those passages do not exist."

    async def set_ui(ctx: Ctx, setting="", value=""):
        look = dict(ctx.ui) if ctx.ui else agent._look()  # build on this turn's earlier change, not the saved look
        if setting == "size":
            look["size"] = look["size"] + 1 if value == "bigger" else look["size"] - 1 if value == "smaller" else _int(value, look["size"], 0, uisettings.SIZE_MAX)
        elif setting in ("contrast", "theme", "layout", "bold", "simple", "spacing", "rate"):
            look[setting] = {"true": True, "false": False}.get(str(value).lower(), value)
            if setting in ("spacing",):
                look[setting] = _int(value, 0, 0, 2)
            if setting == "rate":
                try:
                    look[setting] = float(value)
                except (TypeError, ValueError):
                    return "rate must be a number from 0.6 to 1.5."
        else:
            return "setting must be one of: size, contrast, bold, spacing, simple, rate, theme, layout."
        ctx.ui = uisettings.clean(look)
        return "The screen has been changed."

    async def brainstorm(ctx: Ctx, topic="", n=3):
        """Several *different* takes at once: each call gets its own random seed, so the answers do not collapse into one."""
        n = _int(n, 3, 2, 4)
        if ctx.llm_calls + n > ctx.llm_budget:
            return "Brainstorm budget used up this turn: choose from what you already have."
        ctx.llm_calls += n
        seeds = [Ideas(seed=None).sparks(2) for _ in range(n)]

        async def one(sp):
            msg = [{"role": "system", "content": "You are a bold, original fiction writer. Reply with ONE specific idea in at most 40 words. Avoid the obvious first idea."},
                   {"role": "user", "content": f"Topic: {topic}\nStrange ingredients you may use: {'; '.join(sp)}"}]
            return (await agent.llm.chat(msg, temperature=1.0, max_tokens=120)).strip()

        return list(await asyncio.gather(*(one(sp) for sp in seeds)))

    T = Tool
    return Toolbox([
        T("random_ideas", "Real randomness for names, places, character sketches, problems, twists, titles, objects, sensory details, weather, jobs, quirks and writing challenges. "
          "USE THIS instead of inventing names or plot turns yourself: your own first choices are the predictable ones.",
          {"kind": "|".join(KINDS), "n": "1-12", "culture": "for names/characters, e.g. irish, japanese, fantasy_elvish, any", "gender": "female|male|neutral|any", "type": "for places: village|town|inn|ship|street|wild"},
          random_ideas, required=("kind",)),
        T("roll_sparks", "A few unrelated random ingredients to push a draft somewhere unexpected.", {"n": "1-5"}, roll_sparks),
        T("search_book", "Find passages in the writer's book by words or meaning. Use before answering questions about what the book says.", {"query": "what to look for", "k": "1-8"}, search_book, required=("query",)),
        T("read_passage", "Read one paragraph of the book.", {"chapter": "number", "paragraph": "number, -1 for the last"}, read_passage, required=("chapter", "paragraph")),
        T("bible_lookup", "Look up what the story bible says about a character, place, rule or thread.", {"query": "a name or topic"}, bible_lookup, required=("query",)),
        T("book_info", "How long the book is, who is in it, which threads are open.", {}, book_info),
        T("show_passage", "Light up passages on the writer's screen while you talk about them. Use whenever you point at part of the book.",
          {"ids": "list of paragraph ids from search_book", "chapter": "number", "paragraph": "number"}, show_passage),
        T("set_ui", "Change how the screen looks if the writer says they cannot see well. Only when asked or clearly struggling.",
          {"setting": "size|contrast|bold|spacing|simple|rate|theme|layout", "value": "size: 0-7, bigger or smaller; contrast: normal|high|dark; bold/simple: true|false; spacing: 0-2; rate: 0.6-1.5; theme: auto|light|dark; layout: side|stack|chat|book"},
          set_ui, required=("setting", "value")),
        T("brainstorm", "Get several genuinely different ideas on a topic at once (costs extra time; use sparingly).", {"topic": "what to brainstorm", "n": "2-4"}, brainstorm, required=("topic",)),
    ])
