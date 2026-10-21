# BookWriter Research — Open-Source & Existing Solutions

Research into existing tools, frameworks, and backends relevant to building a
voice-to-book AI co-writer with memory graph and conversational editing.

## Target Concept

A hosted web app where a writer talks (voice or text) to a continuous AI agent
that:
- Transcribes speech → manuscript
- Maintains a memory graph (characters, relationships, plot threads) with embeddings
- Converses back as a fluid narrator/therapist
- Supports targeted edits ("page 32, second paragraph, change this")
- Runs as a container behind Cloudflare HTTPS
- Exposes its API as an MCP server for other agents

---

## 1. Novel Writing Frameworks / Apps

### Full Pipelines
```
NousResearch/autonovel
├── FRAMEWORK/  (program.md, CRAFT.md, ANTI-SLOP.md, PIPELINE.md)
├── TEMPLATES/  (voice.md, world.md, characters.md, outline.md, canon.md)
├── TYPESETTING/ (LaTeX → trade paperback, ePub)
└── CONFIG/ (.env, pyproject)
→ Autonomous novel writing pipeline. MIT. Reference for craft + guardrails.

NovelForge-Agent (LvPengfei1)
├── raw/  (imported source material)
├── novel/03_plot/chapters/  (chapter spines)
└── llm-wiki/  (story bible, continuity ledger)
→ Reusable agent framework for long-form fiction. Uses llm-wiki for canon.

doctoroyy/novel-copilot  ⭐ **MOST RELEVANT**
├── Story Bible (world, rules, constraints)
├── Rolling Summary + Recent Chapters
├── Character State snapshots
├── Plot Graph (foreshadowing/dependency)
├── Timeline Tracking (event dedup)
├── Semantic Cache
└── Deployable on Cloudflare Workers + D1
→ Full-stack AI novel platform, 3-layer memory, relationship graph.
→ Multi-provider (Gemini, OpenAI, DeepSeek). This is essentially your backend.

heider-x/vela  ⭐ 541★
├── Local RAG knowledge base (SQLite + vector)
├── Worldbuilding + character management
├── Auto outline → chapter drafting → rewrite pipeline
├── BYOK (OpenAI, DeepSeek, Gemini, Claude, Ollama)
└── MCP protocol integration
→ AI Novel Writing IDE. GPL-3.0. Local-first, privacy-focused.

Ckokoski/AuthorAgent
→ Autonomous AI writing agent, runs book pipeline locally. MIT.
```

### Story Bible / Graph-Aware (lighter weight)
```
Anshler/graphify-novel
├── chapters/  (manuscript)
├── bible/  (premise, timeline, characters/, threads/, world/)
├── graphify-out/  (generated knowledge graph)
└── .graphifyignore
→ Writing assistant that keeps a KG synced with your manuscript.

Lanerra/saga  ⭐ 110★
├── projects/{story}/
│   ├── chapters/  (finalized markdown)
│   ├── summaries/
│   └── outline/  (structure.yaml)
├── Neo4j knowledge graph (characters, locations, relationships, events)
├── LangGraph orchestration (checkpointed, resumable)
├── Consistency checks + contradiction detection
└── Apache-2.0
→ Autonomous agentic story writing with stored embeddings + KG.
```

### Commercial Alternatives (for feature parity)
- **Sudowrite** — prose-first (Describe, Rewrite, Brainstorm), Muse model, Story Bible
- **NovelCrafter** — worldbuilding-heavy, Codex system, BYOK, scene generation
- **NovelAI** — Kayra-XL model, Lorebook, long-term character memory
- **InkWeaver** — Story Matrix, OpenRouter key, genre modules

---

## 2. Memory & Knowledge Graph Backends (reuse these)

### Dedicated Memory Layers
```
mem0ai/mem0  ⭐ 64.5★ **RECOMMEND**
├── Multi-level memory (User, Session, Agent)
├── Self-hostable OSS (Apache-2.0)
├── Works with any LLM + vector store
├── Auto memory extraction, dedup, conflict resolution
└── MCP + LangGraph + OpenClaw integrations
→ Drop-in memory for your agent. Handles the "continuous agent with memory" part.

OSU-NLP-Group/HippoRAG  ⭐ 3.98★
├── HippoRAG 2 memory framework (NeurIPS'24)
├── Human long-term memory inspired
├── KG + Personalized PageRank retrieval
├── MIT, cost/latency efficient
└── OpenIE triple extraction
→ Best-in-class for multi-hop narrative reasoning.

langchain-ai/langmem
├── Core memory API (works with any storage)
├── Background memory manager (extract/consolidate/update)
├── LangGraph native integration
└── Hot-path + background memory tools
→ If you're already in LangGraph.
```

### Cloudflare-Native Memory (matches your hosting plan)
```
guirguispierre/memoryvault  ⭐ **PERFECT FIT**
├── Self-hosted, graph-aware memory server
├── Cloudflare Workers + D1 + Vectorize + Workers AI
├── 40+ MCP tools (CRUD, graph linking, conflict detection)
├── Hybrid search (lexical + semantic via RRF)
├── Knowledge graph (typed relationships, path finding)
├── Multi-tenant (isolated "brains")
└── OAuth + PKCE auth for MCP clients
→ A memory graph server that IS an MCP endpoint on Cloudflare. Deploy as-is, build book UI on top.

cashcon57/recall
├── MCP memory server, Cloudflare Workers
├── Hybrid (vector + BM25) + cross-encoder reranking
├── Tiered recency decay, auto-relationship graph
├── Three backends: CF Workers / local / Docker
└── Provenance + lifecycle controls
→ More enterprise/trust features; same Cloudflare pattern.

rahilp/second-brain-cloudflare  ⭐ 765★
├── MCP memory layer, Cloudflare Workers
├── Memory graph + vector search (Vectorize)
├── Notion sync, desktop app option
└── MIT, runs on CF free tier
→ Established MCP memory backbone.
```

### Knowledge Graph Builders
```
neo4j-labs/llm-graph-builder
→ Transform unstructured text → KG (nodes, relationships, properties)
→ Great for building your story KG from manuscript.

rahulnyk/knowledge_graph
→ Convert any text → knowledge graph. Simple, GraphRAG-ready.

LlamaIndex Property Graph Index
├── Labeled property graphs
├── Vector + Cypher retrieval
├── Neo4j backing
└── Hybrid search built-in
→ If you want the most flexible graph querying.
```

---

## 3. Voice Transcription (speech → manuscript)

```
SYSTRAN/faster-whisper  ⭐ **RECOMMEND**
├── 4× faster than OpenAI Whisper, less memory
├── GPU/CPU, 8-bit quantization
├── Whisper-Streaming (real-time)
├── WhisperLive (near-live)
└── speaches (OpenAI-compatible streaming server)
→ Self-hosted real-time STT. Runs in your container.

AssemblyAI Self-Hosted Streaming
├── Real-time WebSocket transcription
├── Universal-3.5 Pro model
├── Needs GPU + license (proprietary)
└── Complex deployment
→ Best quality, but proprietary + heavy. Only if you need production-grade ASR.
```

---

## 4. Related Research (papers)

```
ComoRAG  (arXiv:2508.10419)
→ Cognitive-inspired, memory-organized RAG for narrative comprehension.
→ Multi-hop story reasoning; better than static RAG by ~8%.

Guiding Generative Storytelling with Knowledge Graphs  (arXiv:2505.24803)
→ KG-augmented story generation + user-editable KG.
→ Shows editing the graph increases writer agency.
```

---

## 5. Recommended Architecture (synthesis)

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (your web UI)                                       │
│  - Voice mic → transcript                                     │
│  - Manuscript editor (chapters/sections/paragraphs)           │
│  - Graph view (characters, relationships)                     │
│  - Edit UI (select paragraph → ask agent to change)           │
└─────────────────────┬─────────────────────────────────────┘
                      │ HTTP/WebSocket
┌──────────────────────────────────────────────────────────┐
│  BookWriter Backend (FastAPI/Flask, in container)           │
│                                                            │
│  Agent Orchestrator (LangGraph or custom)                  │
│  ├─ Conversational loop (narrator + therapist modes)       │
│  ├─ Targeted edit engine (paragraph ID → LLM rewrite)       │
│  └─ Retrieval (RAG over memory)                            │
│                                                            │
│  Memory Layer                                              │
│  ├─ mem0 (or memoryvault)  → persistent memory graph        │
│  ├─ Neo4j (or LlamaIndex Property Graph) → character/plot KG│
│  └─ faster-whisper (self-hosted STT) → voice input          │
└─────────────────────┬─────────────────────────────────────┘
                      │ REST + MCP
┌──────────────────────────────────────────────────────────┐
│  Cloudflare Tunnel → HTTPS (or deploy backend on CF)        │
└─────────────────────────────────────────────────────────────┘
```

### Key Decisions
1. **Memory**: Use `mem0` (most battle-tested) OR `memoryvault` (already Cloudflare-MCP, saves infra work).
2. **Story KG**: `graphify-novel` or `novel-copilot` patterns (bible + rolling summary + plot graph).
3. **Backend reuse**: `novel-copilot` is closest to your full vision — study it, fork or integrate.
4. **Voice**: `faster-whisper` (self-host, no external dependency).
5. **MCP**: Expose your agent's memory + edit APIs as MCP tools so other agents can call you.

### What's Novel (your IP)
- Voice-first conversational writing (therapist + narrator hybrid)
- Targeted paragraph-level edits via natural language
- Light-novel specific UX (Japanese web-fiction style)
- Container + Cloudflare hosting with MCP exposure

---

## 6. Quickstart Recommendations

**If you want to move fast:**
1. Fork `doctoroyy/novel-copilot` (has Story Bible + plot graph + timeline)
2. Add `faster-whisper` for voice
3. Add `mem0` for cross-session memory
4. Build your own conversational edit layer on top
5. Deploy behind Cloudflare Tunnel

**If you want modular control:**
1. Backend: `novelforge-agent` framework (structured, uses llm-wiki)
2. Memory: `mem0`
3. KG: `graphify-novel` or `neo4j-labs/llm-graph-builder`
4. Voice: `faster-whisper`
5. Host: Docker Compose + Cloudflare Tunnel

---

*Research date: 2026-09-27*

---

## Running it (current state)

```bash
cp .env.example .env        # set ADMIN_TOKEN and the first-run local URLs
docker compose up -d        # UI on :8080, agent API behind it at /api
```

- **Models and tiers:** tap ⚙️ (admin token required). Chat, speech-to-text, text-to-speech and the story graph each have
  three ordered tiers, named by you (default: **T1 Local ROCm → T2 CPU fallback → T3 Frontier**). A request goes to T1; T2 is
  tried only after every T1 provider failed, and T3 (the paid one) only after T2 failed too. A failed provider is skipped for
  a cooldown (default 30 s) so a dead GPU box costs one timeout, not one per request. Any error counts as failure
  (down, timeout, 5xx, bad key, garbled reply). The UI tells the writer when a fallback answered. Providers can be saved
  unassigned; "Test" checks one provider directly. **Embeddings don't fail over** (different models make incompatible vectors,
  which would corrupt memory), so that role uses its highest-priority provider only.
  Book Scribe runs no models. Each provider is just an OpenAI-compatible base URL + model + optional key for one of four
  roles: **chat**, **embeddings** (its own provider, separate from chat), **speech-to-text** and **text-to-speech**. Optionally
  pre-fill them with `CHAT_URL`, `EMBED_URL`, `STT_URL`, `TTS_URL` (and `*_MODEL`) in `.env`; blank ones are simply not
  created. The *story graph* role is optional and reuses the chat tiers when left empty.
  Settings, including API keys, are stored in `data/books/.state/providers.json` (mode 0600; keep it out of git).
- **Cloudflare Tunnel:** run it in its own container and point it at `book-ui:80` (or host port 8080).
- **Memory model:** three layers with separate jobs.
  **Bible** (`data/books/<book>/bible/`, graphify-novel layout) = exact canon, read directly, only changed with the writer's confirmation.
  **Story graph** (`graphify-out/graph.json`, built by vendored graphify in the background after saves) = relationships, with each edge tagged EXTRACTED/INFERRED.
  **mem0** = writer preferences, rejected edits and *drift* notes (new text that contradicts the bible), recalled into later prompts.
- **Vendored dependencies:** `vendor/mem0`, `vendor/graphify`, `vendor/graphify-novel` — see `vendor/VENDOR.md`.
- **Tests:** `cd agent && pip install -r requirements-dev.txt ../vendor/mem0 && pytest`

