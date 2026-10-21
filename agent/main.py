"""Book Scribe agent API (FastAPI)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import hmac
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Agent
from book_store import BookError, BookStore
from clients import LLM, STT, TTS, AllTiersFailed, NotConfigured, Router, probe
from kg import StoryKG
from memory.store import MemoryManager
from providers import ProviderError, Registry
from story_graph import StoryGraph
from settings import load_settings


WEAK_TOKENS = {"change-me", "changeme", "admin", "password", "token", "secret"}


class ChatIn(BaseModel):
    text: str
    session_id: str | None = None
    book: str | None = None


class EditIn(BaseModel):
    text: str


class TiersIn(BaseModel):
    tiers: list[list[str]]


class TierSettingsIn(BaseModel):
    labels: list[str] | None = None
    cooldown: int | None = None


class ProfileIn(BaseModel):
    id: str | None = None
    label: str = ""
    kind: str = "local"
    base_url: str = ""
    api_key: str = ""
    clear_key: bool = False
    model: str = ""
    voice: str = ""
    timeout: float = 0
    tier: int | None = None


def create_app(settings=None, llm=None, stt=None, tts=None, memory=None, graph=None, check_drift=True) -> FastAPI:
    s = settings or load_settings()
    store = BookStore(s.project_dir)
    store.ensure_book(s.default_book)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry(s)
    mem = memory or MemoryManager(s, reg)
    router = Router(reg)
    llm, stt, tts = llm or LLM(router), stt or STT(router), tts or TTS(router)
    graph = graph or StoryGraph(store, reg)
    agent = Agent(s, store, llm, mem, StoryKG(store), graph, check_drift)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for c in (llm, stt, tts):
            await c.http.aclose()

    app = FastAPI(title="Book Scribe Agent", lifespan=lifespan)

    @app.exception_handler(NotConfigured)
    async def _not_configured(_, e: NotConfigured):
        return JSONResponse({"detail": str(e)}, status_code=503)

    @app.exception_handler(AllTiersFailed)
    async def _all_failed(_, e: AllTiersFailed):
        return JSONResponse({"detail": str(e), "errors": e.errors}, status_code=502)

    def admin(x_admin_token: str = Header("")):
        if not s.admin_token:
            raise HTTPException(503, "Admin is disabled: set ADMIN_TOKEN in .env and restart the agent.")
        if len(s.admin_token) < 16 or s.admin_token.lower() in WEAK_TOKENS:
            raise HTTPException(503, "Admin is disabled: ADMIN_TOKEN is too weak. Use 16+ random characters (e.g. `openssl rand -hex 24`).")
        if not hmac.compare_digest(x_admin_token.encode(), s.admin_token.encode()):
            raise HTTPException(401, "Wrong admin token.")

    @app.get("/api/health")
    async def health():
        try:
            models = await llm.models()
        except Exception as e:
            models = f"unreachable: {e.__class__.__name__}"
        mem.search("ping", 1)  # lets the manager settle on a backend before we report it
        return {"ok": True, "memory": getattr(mem, "backend", "injected"), "memory_error": getattr(mem, "error", ""),
                "tts": tts.enabled, "llm_models": models, "kg": agent.kg.stats(), "admin_enabled": len(s.admin_token) >= 16 and s.admin_token.lower() not in WEAK_TOKENS,
                "graph": graph.info(s.default_book)}

    @app.post("/api/chat")
    async def chat(body: ChatIn):
        sid = body.session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, body.text, body.book)
        return {"session_id": sid, "user_text": body.text, **asdict(reply), "served_by": router.served("chat")}

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        return {"text": await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")}

    @app.post("/api/voice")
    async def voice(audio: UploadFile = File(...), session_id: str = Form(""), book: str = Form("")):
        """Mic → whisper → agent in one round trip (fewest taps for the writer)."""
        text = await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")
        sid = session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, text, book or None)
        return {"session_id": sid, "user_text": text, **asdict(reply), "served_by": router.served("stt", "chat")}

    @app.post("/api/tts")
    async def speak(body: EditIn):
        if not tts.enabled:
            raise HTTPException(501, "TTS_URL not configured; the UI falls back to browser speech")
        audio = await tts.speak(body.text)
        return Response(audio, media_type="audio/mpeg", headers={"X-Served-Tier": router.last["tts"].label.encode("ascii", "replace").decode()})

    @app.post("/api/voice-profile")
    async def voice_profile(audio: UploadFile = File(...)):
        """Store the creator's reference clip. Cloning itself is done by the TTS backend that reads this file."""
        d = s.state_dir / "voice"
        d.mkdir(parents=True, exist_ok=True)
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty file")
        (d / "reference.wav").write_bytes(data)
        mem.add("Writer enrolled a custom voice reference clip.", "voice_profile")
        return {"stored": True, "bytes": len(data)}

    # --- admin: model providers (token-protected; holds API keys) ------------
    @app.get("/api/admin/providers", dependencies=[Depends(admin)])
    async def admin_list():
        return reg.snapshot(router.health_snapshot())

    @app.put("/api/admin/providers/{role}", dependencies=[Depends(admin)])
    async def admin_upsert(role: str, body: ProfileIn):
        try:
            return reg.upsert(role, body.model_dump(exclude_unset=True)).public()
        except ProviderError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/admin/providers/{role}/{pid}", dependencies=[Depends(admin)])
    async def admin_delete(role: str, pid: str):
        try:
            reg.delete(role, pid)
        except ProviderError as e:
            raise HTTPException(400, str(e))
        return reg.snapshot(router.health_snapshot())

    @app.put("/api/admin/tiers/{role}", dependencies=[Depends(admin)])
    async def admin_tiers(role: str, body: TiersIn):
        """Replace a role's three ordered tiers: {"tiers": [[ids], [ids], [ids]]}."""
        try:
            reg.set_tiers(role, body.tiers)
        except ProviderError as e:
            raise HTTPException(400, str(e))
        return reg.snapshot(router.health_snapshot())

    @app.put("/api/admin/tier-settings", dependencies=[Depends(admin)])
    async def admin_tier_settings(body: TierSettingsIn):
        try:
            reg.set_settings(body.labels, body.cooldown)
        except (ProviderError, ValueError) as e:
            raise HTTPException(400, str(e))
        return reg.snapshot(router.health_snapshot())

    @app.post("/api/admin/providers/{role}/{pid}/test", dependencies=[Depends(admin)])
    async def admin_test(role: str, pid: str):
        p = next((p for p in reg.profiles.get(role, []) if p.id == pid), None)
        if not p:
            raise HTTPException(404, "unknown profile")
        return await probe(llm.http, role, p)

    @app.post("/api/admin/books/{book}/graph/rebuild", dependencies=[Depends(admin)])
    async def graph_rebuild(book: str):
        """Rebuild now (admin only: it spends LLM tokens). Returns immediately; poll the status endpoint."""
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        graph.schedule_now(book)
        return graph.info(book)

    @app.get("/api/books/{book}/graph")
    async def graph_status(book: str):
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        return graph.info(book)

    # --- manuscript (read by the UI's book half) -----------------------
    @app.get("/api/books/{book}/chapters")
    async def chapters(book: str):
        try:
            return {"chapters": store.chapter_numbers(book)}
        except BookError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/books/{book}/chapters/{n}")
    async def chapter(book: str, n: int):
        try:
            return {"chapter": n, "paragraphs": [asdict(p) for p in store.paragraphs(book, n)]}
        except BookError as e:
            raise HTTPException(404, str(e))

    @app.put("/api/books/{book}/paragraphs/{pid}")
    async def put_paragraph(book: str, pid: str, body: EditIn):
        try:
            return asdict(store.replace_paragraph(book, pid, body.text))
        except BookError as e:
            raise HTTPException(404, str(e))

    ui_dir = os.getenv("UI_DIR")  # dev convenience; in Docker nginx serves the UI
    if ui_dir:
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app

