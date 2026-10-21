# BookWriter — Self-Host Plan & Container Stack

**Goal:** voice-first AI co-writer for a light novel. Creator can't type well / can't see well → least amount of computer use: talk, tap, maybe hold one mic button. Agent has full context of the book + a memory graph.

---

## 0. Reality check on your hardware (probed today)

| Thing | Where it is | Notes |
|---|---|---|
| **llama-swap (models)** | Unraid box, NOT this host | Probed this Ubuntu box: no llama-swap install, no `.gguf` anywhere reachable here. Your `OnePiece/odysseus/docker-compose.gpu-nvidia.yml` proves the model stack lives on the other machine. |
| **embeddors** | Unraid box | Same story — point at the llama-swap/embedding endpoint over LAN. |
| **whisper** | Unraid box (`.../UnraidAppdata/cloudflared` dir exists; whisper image confirmed working there) | This box has NO docker containers running whisper. |
| **This Ubuntu box** | Here | Hermes Web UI on **:8648**, Intel iGPU only (no good GPU), **14 GB RAM / 4 cores**, Docker 29 + Compose v5 installed, cloudflared config dir present but empty. |

**Conclusion:** The heavy compute (LLM infer + embedding + STT) stays on the **Unraid box**. This Ubuntu box runs the **application layer**: book UI + agent + memory graph + proxy + Cloudflare tunnel. That's a clean split and actually ideal. If you'd rather run everything on one box you need to move llama-swap/whisper onto it too (or use this box as the single host). Tell me which and I'll adjust.

---

## 1. What is self-hostable right now (free, no API keys needed)

Everything below can run fully local against your llama-swap + whisper endpoints:

- **Manuscript editor** — custom web app, Markdown/JSON-backed. ✅ built by us
- **Conversational agent (narrator/therapist)** — calls your llama-swap OpenAI-compatible endpoint. ✅
- **Targeted paragraph edits** ("page 32, 2nd para") — resolves natural language → stable paragraph ID → rewrites that span via llama-swap. ✅
- **Story bible + KG** — `graphify-novel` / `novel-copilot` patterns, or build small ourselves. ✅
- **Memory layer** — `mem0` OSS (Apache-2.0) OR `memoryvault` (Cloudflare MCP server) — both self-hostable. ✅
- **Voice input** — your existing whisper image; wrap in an OpenAI-compatible `/v1/audio/transcriptions` proxy if needed. ✅
- **Embeddings** — reuse llama-swap embedding model already running on Unraid. ✅
- **HTTPS to internet** — Cloudflare Tunnel (`cloudflared`), config dir already present at `/home/codebox/UnraidAppdata/cloudflared`. ✅
- **Container orchestration** — Docker Compose. ✅

You only need API keys if you decide to fall back to a hosted LLM for any piece. With llama-swap live you don't.

---

## 2. Architecture diagram

```
                         ┌─────────────────────────────┐
   Creator (poor vision / │    this Ubuntu host         │
   limited typing)        │                             │
   • speaks into mic      │   [5] cloudflared tunnel ───┼──> https on public internet
   • big buttons / voice  │   command                  │     behind Cloudflare
   ◄── agent replies TTS  │                             │
                          │   nginx (TLS + path split)  │
        browser            │   /              /app │ mem │ db │ tts
       ════════             │              │   │   │    │    │
        │                  ▼                  ▼   ▼    ▼    ▼
   ┌────▼─────┐     ┌──────────────┐  ┌───────────┐ ┌─────────┐
   │ Book UI  │────▶│ BookWriter    │  │ memory:   │ │ vector/ │
   │ (React,  │ WS  │ Agent + KB    │  │ mem0 or   │ │ KG:     │
   │  huge    │◀────│  service      │  │ vault     │ │ sqlite/ │
   │ fonts/  │ HTTP │  - chat loop  │  │           │ │ neo4j   │
   │ A11y ok │─────▶│  - edit engine│  └────┬──────┘ └─────────┘
   └────┬─────┘     │  - RAG recall │       │
        │          └─────┬──────────┘       ▼
        │                │ OpenAI-compatible            ┌───────────────────┐
        ▼                ▼  endpoint (llama-swap)         │ UNRAID BOX        │
   whisper proxy        embedding + LLM calls             │ • llama-swap      │
        ▲                └──────────────▲─────────────────│ • embeddors       │
        │                               │                │ • whisper image   │
        └─────────── /v1/audio/transcriptions ◄──────────┘                   │
                                                                           └───────────────────┘
```

**Data flow (the "least computer use" path):**
1. Creator holds mic button → browser streams audio → **whisper proxy** on Unraid transcribes to text.
2. Text goes to **agent** with full book context loaded from the memory/KB layer.
3. Agent replies as narrator/therapist; optionally spoken back via TTS (optional, add later).
4. New prose is committed to the manuscript at a stable paragraph ID automatically — creator never navigates menus.

---

## 3. Container stack (Docker Compose) — rough, for this box

Each service talks to your off-box llama-swap/whisper endpoints via env vars (`LLAMA_SWAP_URL`, `WHISPER_URL`, `EMBED_URL`). Swap in real hostnames/IPs before deploy.

### `docker-compose.yml` (this host)

```yaml
services:
  # [1] Book UI — big-font, minimal buttons, voice-driven. Served static build.
  book-ui:
    image: nginx:stable-alpine
    ports: ["8080:80"]
    volumes: ["./ui/dist:/usr/share/nginx/html:ro"]
    depends_on: [book-agent]
    restart: unless-stopped

  # [2] The agent + knowledge base backend (FastAPI). Your core logic.
  book-agent:
    build: ./agent        # FastAPI app; see agent/Dockerfile
    environment:
      LLAMA_SWAP_URL: http://<UNRAID_HOST>:8080   # OpenAI-compatible LLM+embeddings
      WHISPER_URL: http://<UNRAID_HOST>:8000/v1/audio/transcriptions
      MEM_BACKEND: mem0            # or memoryvault
      PROJECT_DIR: /data/books     # persistent manuscript store
    ports: ["8081:8000"]            # internal API
    volumes: ["./data:/data"]
    restart: unless-stopped

  # [3] Memory layer — pick ONE:
  # Option A (recommended OSS): Mem0
  mem0:
    image: ghcr.io/mem0ai/mem0:latest
    environment: { VECTORDB_URI: ..., EMBEDDING_MODEL: ... }
    # point embedder at llama-swap embeddings endpoint if using self-hosted vectors
  # Option B (already Cloudflare-MCP, saves infra work):
  # memoryvault: build from guirguispierre/memoryvault (Workers-based) -> not a plain container

  # [4] Vector / KG store (if you don't use mem0's built-in vector + Neo4j graph)
  neo4j:
    image: neo4j:5
    environment: { NEO4J_AUTH: "neo4j/bookwriter" }
    volumes: ["./data/neo4j:/data"]
    expose: ["7687"]

  # [5] Reverse proxy with TLS for the Cloudflare tunnel.
  # Simpler: let cloudflared connect outbound to your public domain; nginx serves on :8080.
  cloudflared:
    image: cloudflare/cloudflared:latest
    command: ["run", "--config", "/etc/cloudflared/config.yml"]
    volumes: ["./cloudflared/config.yml:/etc/cloudflared/config.yml:ro"]
    depends_on: [book-ui]
    restart: unless-stopped
```

### `agent/Dockerfile` (minimal, FastAPI + faster-whisper if you want local STT fallback)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends portaudio19-dev \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `agent/requirements.txt`
```
fastapi, uvicorn[standard], httpx
python-multipart, websockets
openai          # uses OpenAI-compatible client for llama-swap
mem0ai          # self-hostable memory layer
sentence-transformers   # only if you need local embeddings; else use llama-swap
sqlalchemy      # manuscript + paragraph ID store
```

### `cloudflared/config.yml`
```yaml
tunnel: <YOUR_TUNNEL_UUID>         # or use "ingress" with your origin
originRequest:
  noTLSVerify: true
  connectTimeout: 30s
ingress:
  - hostname: book.example.com
    service: http://book-ui:8080     # proxy to the big-font UI behind nginx
```

---

## 4. The two most important design choices (call these)

1. **Where does the stack live?**
   - **(A)** Run app layer on this Ubuntu box, point at Unraid's llama-swap+whisper over LAN — *my recommendation given current hardware.* No GPU needed here, inference stays where it already works.
   - **(B)** Move llama-swap + whisper onto this box and run everything together — only do this if Unraid is unreliable or you want one machine. This Intel iGPU won't help LLM speed; CPU-only infer will be slow.

2. **Memory implementation:**
   - **Mem0** (recommended): battle-tested OSS, self-hostable, plug into any LLM/embedder, handles cross-session continuity = exactly the "continuous agent that remembers the whole book."
   - **memoryvault**: already an MCP server on Cloudflare — nice if you lean into the MCP-for-other-agents requirement from day one, but Workers-based so it's a different runtime than your containers.

For a light novel with a memory graph of characters/relationships/side-notes, pair **mem0** for conversational memory with a small **Neo4j** story KG for canonical character/plot facts (novel-copilot / graphify-novel both do exactly this).

---

## 5. Accessibility-first UI rules (the creator can barely see/type)

- Voice-in as the ONLY way to add content (mic button always visible, large).
- Big type (18–22px body), high-contrast dark theme, keyboard-free navigation.
- Everything reachable by voice command ("next paragraph", "change that", "save").
- Agent confirms aloud what it did before writing anything.
- No file browsers, no menus, no drag-drop. Minimal taps only.

---

## 6. Deployment order (rough)

1. Point `LLAMA_SWAP_URL` / `WHISPER_URL` at real Unraid host:ports; verify OpenAI-compatible endpoints respond (`curl $LLAMA_SWAP_URL/v1/models`).
2. Stand up `book-agent` + a minimal text-only UI; confirm chat → manuscript round-trip.
3. Add mem0 + Neo4j; load first chapter, test RAG recall.
4. Wire whisper proxy for live mic input; add TTS reply (optional).
5. Add nginx + cloudflared; get public HTTPS.
6. Expose agent memory/edit endpoints as an MCP server so other agents can call you (guariguispierre/memoryvault already does this — consider adopting its tool surface instead of building fresh).

---

*Date: 2026-09-27 · Plan is rough/iterative — replace `<UNRAID_HOST>` and tunnel UUID before running.*
