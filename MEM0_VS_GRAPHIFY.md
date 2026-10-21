# Can You Combine mem0 + graphify-novel? And Which Framework Is Best

## Short answer: **Yes, use both.** They solve different problems and complement each other perfectly.

---

## What each one does

### graphify-novel (Anshler/graphify-novel)
**Purpose:** The **story-internal knowledge graph** — characters, relationships, plot threads, world rules, contradictions, cross-chapter references.

- Reads your `chapters/` + `bible/` and extracts a KG using the `graphify` skill under the hood
- Maps relationships between characters, locations, events, themes **across the full manuscript**
- Powers: cross-chapter queries ("where was Character X last seen?"), contradiction detection ("you said the vault key was in the north tower, but chapter 7 says south tower"), structural hub detection, implicit connection hints
- Structure you already saw: `chapters/`, `bible/` (premise.md, timeline.md, characters/, threads/, world/), `graphify-out/` (generated KG)
- **Commands:** `/graphify-novel init`, `/graphify-novel review ch03.md`
- **Best for:** Keeping the story itself consistent. It's about **"what happened in the book."**

### mem0ai/mem0
**Purpose:** The **agent's conversational memory** — what the writer has said across sessions, writer preferences, cross-session continuity, things the agent needs to remember about THE WRITER.

- Multi-level memory: **User, Session, Agent** (adaptive personalization)
- Self-hostable (Apache-2.0). Works with any LLM + vector store
- Auto memory extraction, dedup, conflict resolution
- Skills mode: agent controls memory through Triage/Recall/Dream
- Auto-recall before each turn + auto-capture after each turn
- Session-scoped + user-scoped memories
- MCP + LangGraph integrations
- **Best for:** Keeping the **writer** in the loop across sessions. It's about **"what the writer told the agent."**

---

## Can they be combined? Absolutely.

**Recommended architecture: BOTH, in parallel.**

| Layer | What it tracks | Tool |
|---|---|---|
| **Story KG** | Characters, relationships, plot threads, world rules, contradictions, timeline consistency across chapters | `graphify-novel` (runs on your manuscript + bible) |
| **Agent memory** | Writer preferences, style guidelines, cross-session continuity, things the writer has taught the agent, recurring notes/ideas | `mem0` (persistent memory layer, plug into any LLM) |
| **Voice/STT** | Transcribe speech → text | Your working whisper image + faster-whisper proxy |
| **LLM inference** | Generate prose, agent replies, edit spans | llama-swap OpenAI-compatible endpoint |
| **Embeddings** | Semantic search over manuscript + mem0 facts | llama-swap embedding model or sentence-transformers |

**Data flow example:**

1. Writer speaks: *"In chapter 3, make the protagonist more nervous."*
2. Agent resolves via **graphify-novel**: finds `paragraph_id` + checks character state in the story KG
3. Agent also checks **mem0**: "Does the writer have a preference for how nervousness is portrayed? Do they like certain metaphors?" (cross-session memory)
4. LLM rewrites just that paragraph span, respecting both the story KG constraints AND the writer's preferences from mem0
5. New text is committed; mem0 stores the fact that "writer preferred nervous → trembling hands" for next time; graphify-novel updates the character state snapshot

---

## Which framework is "best" for your use case?

Depends on what you prioritize:

### If you want **story consistency + contradiction detection** first
→ Start with **graphify-novel**. It's purpose-built for writing a book, has the init/review commands, and keeps the KG automatically synced with your manuscript. You can add mem0 later for cross-session continuity.

### If you want **conversational continuity + writer memory** first
→ Start with **mem0**. It's the more general-purpose memory layer but is exactly what you need for a "continuous agent that remembers the whole book" + therapist/narrator back-and-forth. Add graphify-novel later when the manuscript grows and you need contradiction detection.

### **Best of both: combine them from the start.**

That's the path I'd recommend. Here's the minimal setup:

```
backend/
├── agent/           # FastAPI + mem0 + llama-swap integration
│   └── memory/      # mem0 config + vector store
├── book/            # manuscript + bible + graphify-novel
│   ├── chapters/
│   ├── bible/
│   └── graphify-out/
└── memoryvault/     # optional: Cloudflare MCP server if you want other agents to call you
```

**Why this combo works for your specific constraints:**

- **Creator can't type / can't see well** → Voice is primary input. mem0 gives the agent continuous memory of the writer's preferences across sessions, so the agent "gets to know" the writer over time.
- **Agent has full context of the book** → graphify-novel gives the agent the story KG (characters, plot threads, world rules) so it never contradicts itself and can answer questions like "where was the vault key mentioned?"
- **Light novel, just words** → Both are text-first; no heavy UI dependency.
- **Hosted as container + Cloudflare HTTPS** → Both mem0 and graphify-novel are self-hostable; graphify-novel runs as a skill/command; mem0 has Docker deployment.
- **MCP connection to other agents** → mem0 has MCP integrations (40+ tools in memoryvault); graphify-novel could be wrapped as an MCP tool too if you want other writing agents to query your story KG.

---

## Quickstart: combine them in one session

1. **Set up graphify-novel** for story KG:
   ```bash
   # Initialize from a premise
   /graphify-novel init "A disgraced archivist discovers the empire's founding records were forged..."
   
   # Or initialize from existing chapters
   /graphify-novel init --from-chapters
   ```
   This creates `bible/`, `chapters/`, and the initial `graphify-out/` KG.

2. **Set up mem0** for agent-writer memory:
   ```bash
   # Pull mem0 Docker + configure
   # Point embedder at llama-swap embedding endpoint
   # Point LLM at llama-swap text endpoint
   ```
   mem0 will auto-capture facts from each conversation and make them available on later turns.

3. **Wire them together in your FastAPI agent:**
   - Before each LLM call: mem0 retrieves relevant user memories + graphify-novel retrieves relevant story KG nodes
   - Pass both as context to the LLM
   - LLM generates prose or edit, respecting both the story constraints AND the writer's preferences
   - After each turn: mem0 auto-captures new facts; graphify-novel can optionally run a review check

---

## Which do I recommend you start with?

**If you already have a manuscript in progress** → Start with **graphify-novel**. It'll immediately give you contradiction detection, character/thread tracking, and the structure to make "change page 32, second paragraph" work. Add mem0 in session 2.

**If you're starting from scratch / want the agent to "remember you" across sessions** → Start with **mem0**. It'll give you a conversational agent that learns your style, preferences, and remembers you from session to session. Add graphify-novel when you have >3 chapters.

**Best: both from day one.** The architecture is cleanly separated (story KG vs. agent memory) and they complement each other rather than overlap.

---

*Next step in your task: scaffold the FastAPI agent that ties llama-swap + whisper + mem0 + graphify-novel together, with the split-screen UI on top.*