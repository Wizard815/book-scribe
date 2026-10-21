# Book Scribe — Plan & Roadmap

## What We Have Right Now

| File | Description |
|---|---|
| `README.md` | Research summary of 20+ open-source novel writing frameworks, memory backends, voice STT options, and Cloudflare-native deployment patterns |
| `HOST_PLAN.md` | Full container stack (Docker Compose) for this host + Unraid split, architecture diagram, 6-step deploy order. Points llama-swap/whisper to Unraid over LAN. |
| `BOOK_WRITER.md` | Split-screen UI design (book half + chat half), book structure conventions (bible.md + characters.md + chapters/), targeted edit flow via stable paragraph_ids, Git reference repos (novel-copilot, saga, graphify-novel, mem0) |
| `MEM0_VS_GRAPHIFY.md` | Deep comparison: can combine both, what each solves, which to prioritize, combined architecture diagram |

---

## Options Listed (from research)

| Category | Options Evaluated |
|---|---|
| **Full novel frameworks** | `doctoroyy/novel-copilot` ⭐ (closest, Cloudflare Workers + D1), `Lanerra/saga` (Neo4j+LangGraph), `LvPengfei1/novelforge-agent` (llm-wiki based), `heider-x/vela` (local RAG + MCP), `Ckokoski/AuthorAgent` (autonomous pipeline) |
| **Memory / KG layers** | `mem0ai/mem0` (battle-tested OSS, multi-level user/session/agent memory), `guirguispierre/memoryvault` (Cloudflare MCP server, 40+ tools), `cashcon57/recall`, `rahilp/second-brain-cloudflare`; `graphify-novel` (story KG), `neo4j-labs/llm-graph-builder`, `LlamaIndex Property Graph Index` |
| **Voice STT** | `faster-whisper` (self-hosted, 4× faster than OpenAI Whisper), `AssemblyAI` self-hosted streaming (proprietary, GPU + license), `sysytem Whisper image` (already working on Unraid) |
| **TTS / Voice output** | OpenAI TTS (6 built-in voices), edge-tts (local, reference-audio cloning), custom voice profile from creator clip, single voice vs. multiple character voices |
| **Agent conversational modes** | Narrator (default, long-form) + Therapist/Quick-Feedback (on-demand), with system-prompt constraints for normal English, quality gate, writer-preference learning |
| **Creator profile enhancements** | Voice clip enrollment (embedding stored in mem0), optional age reference (influences sentence length, idiom frequency, description density) |

---

## What Was Picked (the stack)

| Layer | Choice | Why |
|---|---|---|
| **Compute split** | App layer on this Ubuntu host; llama-swap + whisper on Unraid over LAN | This box has Intel iGPU only; model stack already proven on Unraid |
| **LLM endpoint** | llama-swap (OpenAI-compatible) on Unraid | Already installed and working; no API keys needed |
| **STT** | `faster-whisper` proxy on Unraid `/v1/audio/transcriptions` | Already working whisper image; faster-whisper for real-time |
| **Memory (agent-writer)** | `mem0ai/mem0` (self-hosted, Apache-2.0) | Persistent cross-session memory, auto-capture, MCP integrations |
| **Story KG** | `graphify-novel` + Neo4j (optional) | Book-internal consistency, contradiction detection, paragraph_id resolution |
| **TTS** | OpenAI TTS + custom voice profile (creator clip) + optional age reference | Creator can record their voice; optional age flag subtly shapes style |
| **Quality gate** | System prompt constraints + optional post-check (repeated words, passive voice, "telling" phrases) + mem0 preference learning | Keeps prose grounded in normal English, not AI-isms |
| **UI** | Split-screen (book half + chat half), big fonts, voice-only input (🎙️ pinned mic), high-contrast dark theme, minimal taps | Accessibility-first for creator who can't type/see well |
| **Hosting** | Docker Compose on this host + Cloudflare Tunnel (`cloudflared`) → HTTPS on public internet | `cloudflared` config dir already present at `/home/codebox/UnraidAppdata/cloudflared`; Unraid box handles inference, this box handles app |

---

## Still to Work On (next concrete steps)

| Priority | Step | File(s) affected |
|---|---|---|
| 1 | Scaffold FastAPI `book-agent` service: tie together llama-swap + whisper + mem0 + graphify-novel + TTS | `agent/` dir, `agent/main.py`, `agent/requirements.txt` |
| 2 | Build split-screen UI prototype (React or static HTML + nginx) with big fonts, voice mic button, minimal taps | `ui/` dir, `nginx.conf` additions |
| 3 | Implement voice clip enrollment → mem0 `voice_profile` embedding + TTS passthrough | `agent/memory/`, `config.yaml` additions |
| 4 | Add optional `writer_age` config flag and subtle style adjustments in system prompt | `config.yaml`, `agent/system_prompt.md` |
| 5 | End-to-end test: whisper → agent → TTS read-back → mem0 stores preference → graphify-novel KG update | Full `docker compose up` cycle |
| 6 | Deploy behind Cloudflare Tunnel; verify public HTTPS + MCP endpoint exposure | `cloudflared/config.yml`, `HOST_PLAN.md` update |

---

## What's Up in the Air (choices yet to finalize)

| Item | Options | Decision needed |
|---|---|---|
| **TTS voice mode** | Single voice (all narration) vs. Multiple voices (character differentiation) | Start with single; add character voices later if requested |
| **Age reference** | Include optional `writer_age` in config + system prompt | Omit unless creator volunteers it; keep optional |
| **KG backend** | `graphify-novel` alone + text-based checks vs. Full Neo4j online + Cypher queries | Graphify-novel sufficient for MVP; Neo4j added when cross-chapter queries become frequent |
| **MCP exposure** | Expose agent memory + edit APIs as Model Context Protocol server vs. Keep internal only | Memoryvault already provides 40+ MCP tools; could adopt rather than build fresh |
| **Embedding model** | Use llama-swap embeddings vs. sentence-transformers all-MiniLM-L6-vs | Llama-swap already on Unraid; sentence-transformers smaller/faster if needed |
| **Editor temp storage** | Scratch dir `/home/codebox/.hermes/cache/scratch` (auto-prune 24h) vs. Persistent `/data/books` | Persistent for cross-session; scratch for temp drafts only |

---

## How to Self-Host It

### Prerequisites (on your machine)

| What | Where |
|---|---|
| **Docker + Docker Compose** | This Ubuntu box: Docker 29 + Compose v5 ✅ |
| **llama-swap + Whisper + embedders** | Unraid box (your model stack) |
| **Cloudflare Tunnel** | `cloudflared` config dir already at `/home/codebox/UnraidAppdata/cloudflared` (empty — fill in tunnel UUID later) |
| **Ports** | This box: `:8080` (nginx/ UI), `:8081` (book-agent API), `:8648` (already Hermes gateway) |
| **SSH / network** | Unraid box reachable over LAN (for llama-swap/whisper URLs) |

### Quickstart (6 steps, from `HOST_PLAN.md`)

1. **Fill in env vars** — edit `HOST_PLAN.md` or create `.env` with:
   - `LLAMA_SWAP_URL=http://<UNRAID_IP>:8080` (OpenAI-compatible LLM+embeddings)
   - `WHISPER_URL=http://<UNRAID_IP>:8000/v1/audio/transcriptions`
   - `MEM0_EMBEDDING_MODEL=...` (point at llama-swap embedding endpoint)
   - `PROJECT_DIR=/data/books` (persistent manuscript store)
2. **Start the stack** — `docker compose up -d` from the compose dir
3. **Verify agent chat** — open http://localhost:8080, speak into mic, see prose appear in book half
4. **Add your voice clip** — upload 30s audio → system extracts embedding → stores in mem0 as `voice_profile`
5. **(Optional) Set writer age** — add `age: "42"` to `config.yaml writer_profile` for subtle style shaping
6. **Cloudflare Tunnel** — `cloudflared tunnel run <UUID>` → point at `http://localhost:8080` → HTTPS on `book.example.com`

### Directory Structure (what you'll have after step 1)

```
book-scribe/              ← this repo
├── docker-compose.yml    ← orchestration (points Unraid URLs)
├── .env                  ← filled by you (secrets, URLs)
├── config.yaml           ← TTS mode, voice profile, optional age
├── agent/                ← FastAPI service (llama-swap + mem0 + graphify)
│   ├── main.py
│   ├── requirements.txt
│   └── memory/
│       └── mem0_config.json
├── ui/                   ← static book UI (nginx reverse-proxy)
│   ├── index.html
│   └── styles.css
├── data/                 ← persistent manuscript store (chapters/ bible/)
│   └── books/
│       └── light_novel/
├── cloudflared/          ← tunnel config (or separate)
└── plan.md               ← this file
```

---

## Quick-Reference: One-Sentence Summary

> **Book Scribe** is a voice-first, self-hosted AI co-writer that runs a conversational agent (llama-swap + mem0 + graphify-novel) behind Cloudflare HTTPS, with a split-screen UI for a creator who can't type or see well, maintaining normal English prose via quality gates and learning writer preferences across sessions, optionally cloned to the creator's own voice.

---

*Plan maintained on Studio task card `b6e43b6d`. Next: scaffold the FastAPI agent service (step 1 above).*