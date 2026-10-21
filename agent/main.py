"""Book Scribe agent API (FastAPI)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from agent import Agent
from book_store import BookError, BookStore
from clients import LLM, STT, TTS
from kg import StoryKG
from memory.store import make_memory
from settings import load_settings


class ChatIn(BaseModel):
    text: str
    session_id: str | None = None
    book: str | None = None


class EditIn(BaseModel):
    text: str


def create_app(settings=None, llm=None, stt=None, tts=None, memory=None) -> FastAPI:
    s = settings or load_settings()
    store = BookStore(s.project_dir)
    store.ensure_book(s.default_book)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    mem, backend = (memory, "injected") if memory else make_memory(s)
    llm, stt, tts = llm or LLM(s), stt or STT(s), tts or TTS(s)
    agent = Agent(s, store, llm, mem, StoryKG(store))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for c in (llm, stt, tts):
            await c.http.aclose()

    app = FastAPI(title="Book Scribe Agent", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        try:
            models = await llm.models()
        except Exception as e:
            models = f"unreachable: {e.__class__.__name__}"
        return {"ok": True, "memory": backend, "tts": tts.enabled, "llm_models": models, "kg": agent.kg.stats()}

    @app.post("/api/chat")
    async def chat(body: ChatIn):
        sid = body.session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, body.text, body.book)
        return {"session_id": sid, "user_text": body.text, **asdict(reply)}

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        return {"text": await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")}

    @app.post("/api/voice")
    async def voice(audio: UploadFile = File(...), session_id: str = Form(""), book: str = Form("")):
        """Mic → whisper → agent in one round trip (fewest taps for the writer)."""
        text = await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")
        sid = session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, text, book or None)
        return {"session_id": sid, "user_text": text, **asdict(reply)}

    @app.post("/api/tts")
    async def speak(body: EditIn):
        if not tts.enabled:
            raise HTTPException(501, "TTS_URL not configured; the UI falls back to browser speech")
        return Response(await tts.speak(body.text), media_type="audio/mpeg")

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

    return app

