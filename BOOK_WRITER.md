# BookWriter UI + Book Structure Guide

## Part A: The Split-Screen Layout

```
┌──────────────────────────────┬─────────────────────────────┐
│                              │                             │
│   [BOOK HALF]                │   [CHAT / AGENT HALF]       │
│                                │                               │
│   Chapter 3                    │   🤖 Narrator/Agent          │
│   Section 1                    │   ────────────────────────  │
│                                │   "She steps off the train, │
│   <prose text here>            │    rain soaking her coat…>   │
│                                │                               │
│   ...more prose...             │   💬 Your turn (voice button) │
│                                │                                 │
│                                │   ●●● [🎙️ TALK] ●●●         │
│                                │                                     │
└──────────────────────────────┴─────────────────────────────┘
```

**Two halves:**
- **Left = manuscript.** Read-only by default. Shows current chapter/section/prose in big type. No menus to navigate — just scroll or say "next section."
- **Right = agent conversation.** This is where all interaction happens. Voice-in mic button always visible at bottom.

### Accessibility-first rules (creator can barely see/type)
- **Voice as primary input.** A large 🎙️ button stays pinned at the bottom of the chat half. Hold to talk, release to send. No keyboard needed.
- **Huge font (20–24px)** on book half, high-contrast dark theme.
- **Minimal taps.** Only these buttons matter: mic, and maybe "save" / "next". Everything else is voice command ("go to next paragraph", "change that part").
- **Agent confirms aloud** before writing anything new.
- **No drag-drop, no file pickers, no nested menus.**

---

## Part B: How to Structure Books

All reference repos follow a similar pattern. Here's the convention:

```
book/
├── bible/                    # Story Bible (canonical facts, never changes lightly)
│   ├── premise.md            # Core story logline + themes
│   ├── world.md              # World rules, magic systems, geography
│   ├── characters/           # One file per character
│   │   ├── hero.md
│   │   └── mentor.md
│   ├── timeline.md           # Chronological events
│   └── threads/              # Plot threads, foreshadowing, mysteries
│       └── revenge_thread.md
│
├── manuscript/               # The actual novel
│   └── volume_1/
│       ├── chapter_01.md
│       ├── chapter_02.md
│       └── ...
│
└── memory/                   # Auto-generated continuity data
    ├── summaries/            # Rolling chapter summaries
    └── graph/                # Neo4j or mem0 export
```

### Chapter structure (inside each `.md`)

Use **frontmatter** for metadata and **stable paragraph IDs** for targeted edits:

```markdown
---
chapter: 1
title: "The Departure"
scene_count: 3
characters: [hero, mentor]
location: train_station
---

# Chapter 1: The Departure

## Scene 1
<!-- paragraph_id: ch1_s1_p1 -->
She stepped onto the platform, her breath fogging in the cold air.

<!-- paragraph_id: ch1_s1_p2 -->
The express was late. It was always late.

## Scene 2
<!-- paragraph_id: ch1_s2_p1 -->
...
```

**Why stable paragraph IDs?** This is what makes "change page 32, second paragraph" work. The agent resolves natural language ("the part where she waits for the train") OR structured refs (ch1 §1 p2) to a specific `paragraph_id`, then rewrites only that span — never regenerating the whole chapter.

---

## Part C: Targeted Edits Flow

1. Creator says: *"In chapter 3, when they meet at the café, make him more nervous."*
2. Agent resolves → finds `paragraph_id` via KG search or scene title match.
3. Agent loads ONLY those paragraphs + relevant memory graph nodes (characters, prior scenes).
4. LLM rewrites just that span via llama-swap endpoint.
5. New text appears inline; agent confirms aloud.

This keeps edits surgical — no accidental rewriting of good prose elsewhere.

---

## Part D: Git Reference Repos (study/fork these)

### Full novel-writing frameworks
- **doctoroyy/novel-copilot** ⭐ *most similar to your vision*
  - Story Bible + plot graph + character state + timeline tracking
  - Deployable on Cloudflare Workers
  - Multi-provider model support
- **Lanerra/saga** ⭐ 110★
  - projects/{story}/chapters/ , summaries/, outline/structure.yaml
  - Neo4j knowledge graph + LangGraph orchestration
  - Consistency checks + contradiction detection
- **LvPengfei1/novelforge-agent**
  - raw/ (imports), novel/03_plot/chapters/ (spines), llm-wiki/ (bible + continuity ledger)
  - Structured long-form fiction pipeline
- **heider-x/vela** ⭐ 541★
  - AI Novel Writing IDE with local RAG + worldbuilding MCP integration
- **Ckokoski/AuthorAgent** → autonomous book pipeline, runs locally

### Story bible / KG patterns (lighter weight)
- **Anshler/graphify-novel**
  - chapters/ , bible/(characters/, threads/, world/) , graphify-out/
  - Keeps a knowledge graph synced with your manuscript
- **neo4j-labs/llm-graph-builder** → convert any text → KG (nodes+relationships)

### Memory layers
- **mem0ai/mem0** ⭐ *recommended* — self-hostable OSS memory layer, plug into any LLM
- **guirguispierre/memoryvault** — Cloudflare MCP memory server (40+ tools)
- **cashcon57/recall** — hybrid vector+BM25 memory on Cloudflare Workers

---

## Part E: Minimal Viable Structure for You

For a light novel with voice-first input, start simple:

```
book/
├── bible.md              # premise + world rules (one file is fine early)
├── characters.md         # character roster w/ relationships
├── chapters/             # chapter_01.md ... chapter_NN.md
│   └── chapter_01.md     # frontmatter + paragraph_ids inside
└── memory/               # mem0 export + rolling summaries
```

Grow it into the full `bible/` subfolder structure once you have >3 chapters.

---

*Next step: scaffold the FastAPI agent service that reads this structure and talks to llama-swap.*
