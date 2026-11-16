"""Book Scribe agent API (FastAPI)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import hashlib
import hmac
import json

import httpx
import os
import time

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import Agent
from auth import (ADMIN_TTL, COOKIE, WRITER_TTL, AccountStore, AuthError, LoginLimiter, Sessions, admin_password_ok,
                  check_password_rules, check_username_rules)
from book_store import BookError, BookStore
from clients import LLM, STT, TTS, AllTiersFailed, NotConfigured, Router, probe
from kg import StoryKG
from memory.store import MemoryManager
from providers import ProviderError, Registry
from bible import BibleError
from import_text import import_text
from prefs import FIELDS as PREF_FIELDS, Prefs
import uisettings
from embeddings import ParagraphIndex
import extract as extract_mod
import ingest
from addons import Addons
from journal import Journal
from library import Library
from ideas import describe_kinds
from story_graph import StoryGraph
from voices import VoiceBook
from settings import load_settings


CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
       "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
WEAK_TOKENS = {"change-me", "changeme", "admin", "password", "token", "secret"}


class ChatIn(BaseModel):
    text: str
    session_id: str | None = None
    book: str | None = None


class PersonIn(BaseModel):
    person_id: str = Field(max_length=60)


class EditIn(BaseModel):
    text: str


class EntryIn(BaseModel):
    slug: str | None = Field(default=None, max_length=80)
    name: str = Field(min_length=1, max_length=80)
    state: dict = {}
    body: str = Field(default="", max_length=20000)


class DocIn(BaseModel):
    text: str = Field(max_length=100000)


class NewBookIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class ImportIn(BaseModel):
    text: str = Field(max_length=5_000_000)
    max_chapters: int | None = None


class PrefsIn(BaseModel):
    writer_age: str | None = Field(default=None, max_length=20)
    display_name: str | None = Field(default=None, max_length=60)
    surprise: str | None = Field(default=None, max_length=8)      # off | low | high: how much random inspiration the model gets
    strictness: str | None = Field(default=None, max_length=8)    # off | gentle | strict: how hard the style check pushes back
    look: dict | None = None                                      # how the screen looks (validated by uisettings.clean)


class IdeasIn(BaseModel):
    kind: str = Field(max_length=20)
    n: int = Field(default=5, ge=1, le=12)
    opts: dict = {}


def clean_state(raw: dict) -> dict:
    """Bible front-matter from a form: short plain values only, so a pasted blob can't bloat or break the YAML."""
    out = {}
    for k, v in list(raw.items())[:30]:
        key = str(k).strip().lower().replace(" ", "_")[:40]
        if not key or key == "name" or not key.replace("_", "").isalnum():
            continue
        if isinstance(v, list):
            vals = [str(x).strip()[:80] for x in v[:30] if str(x).strip()]
            if vals:
                out[key] = vals
        elif v not in (None, ""):
            out[key] = str(v).strip()[:300]
    return out


class RebuildIn(BaseModel):
    full: bool = False


class LoginIn(BaseModel):
    username: str = Field(max_length=80)
    password: str = Field(max_length=300)


class PasswordIn(BaseModel):
    current: str = Field(max_length=300)
    new_password: str = Field(max_length=300)
    new_username: str | None = Field(default=None, max_length=80)


class SpeakIn(BaseModel):
    text: str
    speaker: str = "assistant"  # assistant | narrator | a character's name
    draft: str | None = None  # an unsaved voice being auditioned
    book: str | None = None


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
    voices: str = ""
    style_field: str = ""
    timeout: float = 0
    tier: int | None = None


def create_app(settings=None, llm=None, stt=None, tts=None, memory=None, graph=None, check_drift=True, require_auth=True, embed_http=None) -> FastAPI:
    s = settings or load_settings()
    store = BookStore(s.project_dir)
    store.ensure_book(s.default_book)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry(s)
    mem = memory or MemoryManager(s, reg)
    router = Router(reg)
    voices = VoiceBook(store, reg)
    llm, stt, tts = llm or LLM(router), stt or STT(router), tts or TTS(router, voices=voices, default_book=s.default_book)
    graph = graph or StoryGraph(store, reg)
    accounts, sessions, limiter = AccountStore(s.state_dir), Sessions(s.state_dir), LoginLimiter()
    if accounts.get() is None and s.writer_username and s.writer_password:
        try:
            accounts.set(s.writer_username, s.writer_password)  # first-run seed from the environment
        except AuthError as e:
            print(f"[auth] WRITER_USERNAME/WRITER_PASSWORD ignored: {e}")
    admin_fp = hashlib.sha256(s.admin_token.encode()).hexdigest()[:12]  # an ADMIN_TOKEN change ends admin sessions
    prefs = Prefs(s.state_dir)
    emb_http = embed_http or httpx.AsyncClient()
    index = ParagraphIndex(store, reg, router, emb_http, s.state_dir, debounce=s.index_debounce)
    journal, library = Journal(store), Library(store)
    index.sources = [journal.chunks, library.chunks]  # the long conversation and the added files are searched by meaning too
    addons = Addons(s, s.addons_dir).load()
    agent = Agent(s, store, llm, mem, StoryKG(store), graph, check_drift, voices, prefs, index, journal, library, addons)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for c in (llm, stt, tts):
            await c.http.aclose()
        await emb_http.aclose()

    app = FastAPI(title="Book Scribe Agent", lifespan=lifespan)

    @app.exception_handler(NotConfigured)
    async def _not_configured(_, e: NotConfigured):
        return JSONResponse({"detail": str(e)}, status_code=503)

    @app.exception_handler(AllTiersFailed)
    async def _all_failed(_, e: AllTiersFailed):
        return JSONResponse({"detail": str(e), "errors": e.errors}, status_code=502)

    def client_ip(request: Request) -> str:
        if s.trust_proxy:
            fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            if fwd:
                return fwd
        return request.client.host if request.client else "?"

    def session_of(request: Request) -> dict | None:
        return sessions.parse(request.cookies.get(COOKIE))

    def writer(request: Request) -> dict:
        if not require_auth:
            return {"r": "writer", "u": "local"}
        sess = session_of(request)
        if not sess or sess.get("r") != "writer" or sess.get("v") != accounts.version():
            raise HTTPException(401, "Please log in.")
        return sess

    def admin(request: Request, x_admin_token: str = Header("")):
        if not s.admin_token:
            raise HTTPException(503, "Admin is disabled: set ADMIN_TOKEN in .env and restart the agent.")
        if not admin_password_ok(s.admin_token):
            raise HTTPException(503, "Admin is disabled: ADMIN_TOKEN is too weak. Use 16+ random characters (e.g. `openssl rand -hex 24`).")
        sess = session_of(request)
        if sess and sess.get("r") == "admin" and sess.get("v") == admin_fp:
            return
        if x_admin_token and hmac.compare_digest(x_admin_token.encode(), s.admin_token.encode()):  # scripts
            return
        raise HTTPException(401, "Please log in as the admin.")

    @app.middleware("http")
    async def hardening(request: Request, call_next):
        # State-changing requests must come from this site (cookies are SameSite=Strict too; this is the second lock).
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and (origin := request.headers.get("origin")):
            host = request.headers.get("x-forwarded-host") if s.trust_proxy and request.headers.get("x-forwarded-host") else request.headers.get("host", "")
            if origin.split("://", 1)[-1] != host:
                return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/api/"):
            resp.headers.setdefault("Content-Security-Policy", CSP)
        if request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    def set_session(resp: Response, request: Request, role: str, username: str, version, ttl: int) -> None:
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        resp.set_cookie(COOKIE, sessions.issue(role, username, version, ttl), max_age=ttl, httponly=True, samesite="strict", secure=secure, path="/")

    def attempt(request: Request, kind: str, ok_fn) -> bool:
        key = f"{kind}:{client_ip(request)}"
        wait = limiter.wait(key)
        if wait:
            raise HTTPException(429, f"Too many wrong tries. Please wait {wait // 60 + 1} minute(s).", headers={"Retry-After": str(wait)})
        if ok_fn():
            limiter.ok(key)
            return True
        limiter.fail(key)
        raise HTTPException(401, "Wrong username or password.")

    # --- login -----------------------------------------------------------------------------------------------
    @app.get("/api/auth/status")
    async def auth_status():
        """What the login page needs to know before anyone is logged in."""
        return {"writer_account": accounts.get() is not None, "admin_enabled": bool(s.admin_token) and admin_password_ok(s.admin_token), "auth_required": require_auth}

    @app.post("/api/auth/login")
    async def login(body: LoginIn, request: Request):
        attempt(request, "writer", lambda: accounts.verify(body.username.strip(), body.password))
        acc = accounts.get()
        resp = JSONResponse({"role": "writer", "username": acc["username"]})
        set_session(resp, request, "writer", acc["username"], acc["version"], WRITER_TTL)
        return resp

    @app.post("/api/auth/admin-login")
    async def admin_login(body: LoginIn, request: Request):
        if not s.admin_token or not admin_password_ok(s.admin_token):
            raise HTTPException(503, "Admin login is off: set a strong ADMIN_TOKEN (16+ characters) in .env and restart.")
        attempt(request, "admin", lambda: hmac.compare_digest(body.username.strip().encode(), s.admin_user.encode()) & hmac.compare_digest(body.password.encode(), s.admin_token.encode()))
        resp = JSONResponse({"role": "admin", "username": s.admin_user})
        set_session(resp, request, "admin", s.admin_user, admin_fp, ADMIN_TTL)
        return resp

    @app.post("/api/auth/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    @app.get("/api/auth/me")
    async def me(request: Request):
        if not require_auth:
            return {"role": "writer", "username": "local"}
        sess = session_of(request)
        if sess and sess.get("r") == "writer" and sess.get("v") == accounts.version():
            return {"role": "writer", "username": sess["u"]}
        if sess and sess.get("r") == "admin" and sess.get("v") == admin_fp:
            return {"role": "admin", "username": sess["u"]}
        raise HTTPException(401, "Please log in.")

    @app.post("/api/account/password", dependencies=[Depends(writer)])
    async def change_password(body: PasswordIn, request: Request):
        acc = accounts.get()
        if not acc:
            raise HTTPException(404, "No account.")
        attempt(request, "writer", lambda: accounts.verify(acc["username"], body.current))
        try:
            new = accounts.set((body.new_username or acc["username"]).strip(), body.new_password)
        except AuthError as e:
            raise HTTPException(400, str(e))
        resp = JSONResponse({"ok": True, "username": new["username"]})
        set_session(resp, request, "writer", new["username"], new["version"], WRITER_TTL)  # keep this device logged in; all others end
        return resp

    # --- admin: the one writer account -------------------------------------------------------------------------
    @app.get("/api/admin/account", dependencies=[Depends(admin)])
    async def admin_account():
        acc = accounts.get()
        return {"exists": acc is not None, "username": acc["username"] if acc else None, "updated": acc["updated"] if acc else None}

    @app.put("/api/admin/account", dependencies=[Depends(admin)])
    async def admin_set_account(body: LoginIn):
        """Create or reset the writer's login. Existing writer sessions end."""
        try:
            new = accounts.set(body.username.strip(), body.password)
        except AuthError as e:
            raise HTTPException(400, str(e))
        return {"exists": True, "username": new["username"]}

    @app.delete("/api/admin/account", dependencies=[Depends(admin)])
    async def admin_delete_account():
        accounts.remove()
        return {"exists": False}

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/admin/status", dependencies=[Depends(admin)])
    async def admin_status():
        try:
            models = await llm.models()
        except Exception as e:
            models = f"unreachable: {e.__class__.__name__}"
        mem.search("ping", 1)  # lets the manager settle on a backend before we report it
        return {"ok": True, "memory": getattr(mem, "backend", "injected"), "memory_error": getattr(mem, "error", ""),
                "tts": tts.enabled, "llm_models": models, "kg": agent.kg.stats(), "admin_enabled": len(s.admin_token) >= 16 and s.admin_token.lower() not in WEAK_TOKENS,
                "graph": graph.info(s.default_book)}

    @app.post("/api/chat", dependencies=[Depends(writer)])
    async def chat(body: ChatIn):
        sid = body.session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, body.text, body.book, via="typed")
        return {"session_id": sid, "user_text": body.text, **asdict(reply), "served_by": router.served("chat")}

    @app.post("/api/transcribe", dependencies=[Depends(writer)])
    async def transcribe(audio: UploadFile = File(...)):
        return {"text": await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")}

    @app.post("/api/voice", dependencies=[Depends(writer)])
    async def voice(audio: UploadFile = File(...), session_id: str = Form(""), book: str = Form("")):
        """Mic → whisper → agent in one round trip (fewest taps for the writer)."""
        text = await stt.transcribe(await audio.read(), audio.filename or "audio.webm", audio.content_type or "audio/webm")
        sid = session_id or uuid.uuid4().hex
        reply = await agent.handle(sid, text, book or None, via="voice")
        return {"session_id": sid, "user_text": text, **asdict(reply), "served_by": router.served("stt", "chat")}

    @app.post("/api/tts", dependencies=[Depends(writer)])
    async def speak(body: SpeakIn):
        if not tts.enabled:
            raise HTTPException(501, "TTS_URL not configured; the UI falls back to browser speech")
        audio = await tts.speak(body.text, body.speaker, body.book, body.draft)
        return Response(audio, media_type="audio/mpeg", headers={"X-Served-Tier": router.last["tts"].label.encode("ascii", "replace").decode()})

    @app.post("/api/voice-profile", dependencies=[Depends(writer)])
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

    @app.get("/api/admin/embeddings", dependencies=[Depends(admin)])
    async def admin_embeddings(book: str | None = None):
        """Embedding provider, the paragraph index for a book, and long-term memory (mem0) status."""
        b = book_or_400(book or s.default_book)
        mem.search("ping", 1)  # lets the memory manager settle on a backend before we report it
        names = sorted(d.name for d in s.project_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / "chapters").is_dir())
        return {"book": b, "books": names, "index": index.status(b), "memory": {"backend": getattr(mem, "backend", "injected"), "error": getattr(mem, "error", "")}}

    @app.post("/api/admin/embeddings/test", dependencies=[Depends(admin)])
    async def admin_embeddings_test():
        try:
            (v,) = await index.embed(["Book Scribe embedding test."])
        except Exception as e:
            return {"ok": False, "detail": f"{e.__class__.__name__}: {str(e)[:200]}"}
        return {"ok": True, "dims": len(v)}

    @app.post("/api/admin/books/{book}/embeddings/rebuild", dependencies=[Depends(admin)])
    async def admin_embeddings_rebuild(book: str, body: RebuildIn):
        index.start(book_or_400(book), body.full)
        return index.status(book)

    @app.get("/api/admin/addons", dependencies=[Depends(admin)])
    async def admin_addons():
        return {"directory": s.addons_dir or "", "loaded": addons.status, "readers": [{"addon": n, "extensions": list(e)} for n, e in addons.readers],
                "tools": [t.name for t in addons.tools]}

    @app.get("/api/admin/traces", dependencies=[Depends(admin)])
    async def admin_traces():
        """Recent turns in which the model used tools: what it called, and how many passages it pointed at."""
        return {"traces": list(agent.traces)[::-1]}

    @app.get("/api/admin/graph/{book}", dependencies=[Depends(admin)])
    async def admin_graph(book: str):
        return graph.info(book_or_400(book))

    @app.post("/api/admin/books/{book}/graph/rebuild", dependencies=[Depends(admin)])
    async def graph_rebuild(book: str):
        """Rebuild now (admin only: it spends LLM tokens). Returns immediately; poll the status endpoint."""
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        graph.schedule_now(book)
        return graph.info(book)

    @app.get("/api/books/{book}/graph", dependencies=[Depends(writer)])
    async def graph_status(book: str):
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        return graph.info(book)

    # --- manuscript (read by the UI's book half) -----------------------
    @app.get("/api/books", dependencies=[Depends(writer)])
    async def books():
        found = sorted(d.name for d in s.project_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / "chapters").is_dir()) if s.project_dir.is_dir() else []
        return {"books": found, "default": s.default_book}

    def book_or_400(book: str) -> str:
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        return book

    # --- writer settings ------------------------------------------------------------------------------------------
    def settings_view() -> dict:
        d = prefs.all()
        d.update(look=agent._look(), surprise=d["surprise"] or "low", strictness=d["strictness"] or "gentle", default_book=s.default_book)
        return d

    @app.get("/api/settings", dependencies=[Depends(writer)])
    async def get_settings():
        return settings_view()

    @app.put("/api/settings", dependencies=[Depends(writer)])
    async def put_settings(body: PrefsIn):
        changes = body.model_dump(exclude={"look"})
        if body.look is not None:
            changes["look"] = json.dumps(uisettings.clean(body.look))
        prefs.update(changes)
        return settings_view()

    # --- random ideas (names, places, twists...) for the writer's Ideas panel -------------------------------------
    @app.get("/api/ideas/kinds", dependencies=[Depends(writer)])
    async def ideas_kinds():
        return {"kinds": describe_kinds()}

    @app.post("/api/ideas", dependencies=[Depends(writer)])
    async def ideas_generate(body: IdeasIn):
        try:
            return {"items": agent.ideas.generate(body.kind, body.n, **body.opts)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    # --- books -------------------------------------------------------------------------------------------------
    @app.post("/api/books", dependencies=[Depends(writer)])
    async def new_book(body: NewBookIn):
        name = body.name.strip().replace(" ", "-")
        book_or_400(name)
        if (s.project_dir / name).exists():
            raise HTTPException(409, "There is already a book with that name.")
        store.ensure_book(name)
        return {"name": name}

    @app.post("/api/books/{book}/import", dependencies=[Depends(writer)])
    async def import_book(book: str, body: ImportIn):
        """Split a pasted/uploaded draft into chapters and paragraphs (see import_text.py). Refuses a book that already has chapters."""
        book_or_400(book)
        store.ensure_book(book)
        if store.chapter_numbers(book):
            raise HTTPException(409, "That book already has chapters. Import into a new book instead.")
        chapters, paragraphs = import_text(store, book, body.text, body.max_chapters)
        index.schedule(book)
        if not chapters:
            raise HTTPException(400, "I could not find any chapters. Start each chapter with a line like: Chapter 1")
        return {"chapters": chapters, "paragraphs": paragraphs}

    # --- the one long conversation ---------------------------------------------------------------------------------
    @app.get("/api/books/{book}/chat", dependencies=[Depends(writer)])
    async def chat_log(book: str, n: int = 30):
        """What the chat window shows after a reload: the latest turns since the writer last cleared the screen."""
        book_or_400(book)
        rows = [r for r in journal.recent(book, max(1, min(n, 100))) if r["role"] in ("user", "assistant")]
        return {"messages": [{"role": r["role"], "text": r["text"], "at": r["at"]} for r in rows], "total": journal.count(book), "days": journal.days(book)}

    @app.post("/api/books/{book}/chat/clear", dependencies=[Depends(writer)])
    async def chat_clear(book: str):
        """Clear the screen. The agent folds the conversation into its summary first, and the journal keeps every word."""
        await agent.clear_chat(book_or_400(book))
        return {"ok": True}

    # --- the library: files the writer adds -----------------------------------------------------------------------------
    UPLOAD_LIMIT = extract_mod.MAX_BYTES

    @app.get("/api/books/{book}/library", dependencies=[Depends(writer)])
    async def library_list(book: str):
        return {"items": library.items(book_or_400(book)), "formats": {"documents": "txt md html rtf docx odt pptx xlsx csv json epub pdf (also old doc/xls/ppt)",
                "pictures": "jpg png gif webp bmp tiff (read by the chat model if it can see pictures)", "audio": "mp3 wav m4a ogg flac webm (turned into words)",
                "family trees": "ged gdz gramps ftm ftmb", "addons": [{"extensions": list(e), "addon": n} for n, e in addons.readers]}}

    @app.post("/api/books/{book}/library", dependencies=[Depends(writer)])
    async def library_add(book: str, files: list[UploadFile] = File(...)):
        book_or_400(book)
        store.ensure_book(book)
        out = []
        for f in files[:10]:
            data = await f.read(UPLOAD_LIMIT + 1)
            name = (f.filename or "file")[:120]
            try:
                out.append(await ingest.add_file(book, name, data, library, llm, stt, journal))
            except extract_mod.Unreadable as e:
                out.append({"title": name, "error": str(e), "message": str(e)})
            except Exception:
                out.append({"title": name, "error": "I could not add that file.", "message": f"I could not add {name}."})
        index.schedule(book)
        return {"items": out}

    def lib_item(book: str, slug: str) -> dict:
        try:
            return library.meta(book_or_400(book), slug)
        except KeyError:
            raise HTTPException(404, "No such file in the library.")

    @app.get("/api/books/{book}/library/{slug}", dependencies=[Depends(writer)])
    async def library_get(book: str, slug: str, part: int = 1):
        meta = lib_item(book, slug)
        parts = __import__("library").chunk_text(library.text(book, slug), 4000) or [""]
        part = max(1, min(part, len(parts)))
        out = {**meta, "text": parts[part - 1], "part": part, "parts": len(parts)}
        tree = library.tree(book, slug)
        if tree:
            out["people"] = [{"id": p.id, "name": p.name or "Unknown", "label": tree.label(p.id)} for p in sorted(tree.people.values(), key=lambda p: p.name)[:1500]]
        return out

    FILE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}

    @app.get("/api/books/{book}/library/{slug}/file", dependencies=[Depends(writer)])
    async def library_file(book: str, slug: str):
        """The original file. Only pictures are shown in the page; everything else downloads, and nothing is ever run as a web page."""
        lib_item(book, slug)
        path = library.source(book, slug)
        if not path:
            raise HTTPException(404, "The original file is gone.")
        ctype = FILE_TYPES.get(path.suffix.lower())
        name = library.meta(book, slug)["filename"].replace('"', "")
        return Response(path.read_bytes(), media_type=ctype or "application/octet-stream", headers={
            "Content-Disposition": ("inline" if ctype else "attachment") + f'; filename="{name.encode("ascii", "replace").decode()}"',
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox; default-src 'none'; img-src 'self'"})

    @app.delete("/api/books/{book}/library/{slug}", dependencies=[Depends(writer)])
    async def library_delete(book: str, slug: str):
        lib_item(book, slug)
        library.delete(book, slug)
        index.schedule(book)
        return {"ok": True}

    @app.post("/api/books/{book}/library/{slug}/retry", dependencies=[Depends(writer)])
    async def library_retry(book: str, slug: str):
        """Try again to read a picture or recording that could not be read the first time (for example after the admin picked a vision model)."""
        meta = lib_item(book, slug)
        path = library.source(book, slug)
        data = path.read_bytes() if path else b""
        try:
            ex = extract_mod.extract(data, meta["filename"])
            if ex.kind in ("image", "scan"):
                text = await ingest._look(llm, ex.images)
            elif ex.kind == "audio":
                text = await stt.transcribe(ex.audio or b"", ex.audio_name)
            else:
                text = ex.text
        except Exception:
            raise HTTPException(502, "It still could not be read. Check that a model that can look at pictures (or hear audio) is set up.")
        library.update_text(book, slug, text, "read", [])
        index.schedule(book)
        return {"ok": True}

    @app.post("/api/books/{book}/library/{slug}/character", dependencies=[Depends(writer)])
    async def library_person_to_character(book: str, slug: str, body: PersonIn):
        """Turn someone in a family tree into a character in the story wiki (the writer pressed the button, so this is the confirmation)."""
        lib_item(book, slug)
        tree = library.tree(book, slug)
        person = tree.people.get(body.person_id) if tree else None
        if not person:
            raise HTTPException(404, "That person is not in the tree.")
        state = {k: v for k, v in {"gender": person.sex, "born": " ".join(x for x in person.birth if x), "died": " ".join(x for x in person.death if x),
                                   "from_family_tree": library.meta(book, slug)["title"]}.items() if v}
        e = store.bible(book).upsert("characters", person.name or "Unknown", state, tree.describe(person.id))
        index.schedule(book)
        return {"slug": e.slug, "name": e.name}

    @app.get("/api/books/{book}/memory/search", dependencies=[Depends(writer)])
    async def memory_search(book: str, q: str = ""):
        """One search box over everything: the book, the wiki, old conversation and the library, by meaning when the index is ready."""
        book_or_400(book)
        q = q.strip()[:300]
        if not q:
            return {"hits": []}
        hits: list[dict] = []
        if index.status(book)["ready"]:
            docs = index._docs(book)
            for key, score in await index.search(book, q, k=12, scope="para|bible|mem|lib"):
                if key in docs and score >= 0.25:
                    src = "book" if not key.split(":")[0] in ("bible", "mem", "lib") else {"bible": "wiki", "mem": "conversation", "lib": "library"}[key.split(":")[0]]
                    hits.append({"source": src, "text": docs[key][:300], "key": key, "score": round(score, 2)})
        else:
            for key, text in journal.search_words(book, q, 4) + library.search_words(book, q, 4):
                hits.append({"source": "conversation" if key.startswith("mem:") else "library", "text": text[:300], "key": key, "score": 0})
            agent.kg.refresh(book)
            for h in agent.kg.search(q, limit=4):
                hits.append({"source": "book", "text": h.para.text[:300], "key": h.para.id, "score": 0})
        return {"hits": hits[:12]}

    # --- story bible -------------------------------------------------------------------------------------------
    @app.get("/api/books/{book}/bible", dependencies=[Depends(writer)])
    async def get_bible(book: str):
        b = store.bible(book_or_400(book))
        ent = lambda kind: [{"slug": e.slug, "name": e.name, "state": {k: v for k, v in e.state.items() if k != "name"}, "body": e.body.strip()} for e in b.entries(kind).values()]
        return {"premise": b.read("premise"), "timeline": b.read("timeline"), "characters": ent("characters"), "threads": ent("threads"), "world": ent("world")}

    @app.put("/api/books/{book}/bible/doc/{name}", dependencies=[Depends(writer)])
    async def put_bible_doc(book: str, name: str, body: DocIn):
        try:
            store.bible(book_or_400(book)).write_doc(name, body.text)
        except BibleError as e:
            raise HTTPException(400, str(e))
        index.schedule(book)
        return {"ok": True}

    @app.put("/api/books/{book}/bible/{kind}", dependencies=[Depends(writer)])
    async def put_bible_entry(book: str, kind: str, body: EntryIn):
        try:
            e = store.bible(book_or_400(book)).save_entry(kind, body.name, clean_state(body.state), body.body, body.slug)
        except BibleError as err:
            raise HTTPException(400, str(err))
        index.schedule(book)  # the bible is searched by meaning too
        return {"slug": e.slug, "name": e.name}

    @app.delete("/api/books/{book}/bible/{kind}/{slug}", dependencies=[Depends(writer)])
    async def delete_bible_entry(book: str, kind: str, slug: str):
        try:
            gone = store.bible(book_or_400(book)).delete(kind, slug)
        except BibleError as err:
            raise HTTPException(400, str(err))
        if not gone:
            raise HTTPException(404, "No such entry.")
        index.schedule(book)
        return {"ok": True}

    @app.delete("/api/books/{book}/voices/{speaker}", dependencies=[Depends(writer)])
    async def forget_voice(book: str, speaker: str):
        return {"forgotten": voices.forget(book_or_400(book), speaker)}

    @app.get("/api/books/{book}/voices", dependencies=[Depends(writer)])
    async def book_voices(book: str):
        """Saved voice profiles for a book plus the voices the TTS provider offers."""
        try:
            store.book_dir(book)
        except BookError as e:
            raise HTTPException(400, str(e))
        from voices import parse_pool
        prim = voices.primary()
        return {"profiles": voices.all_profiles(book), "pool": [{"name": v.name, "tags": sorted(v.tags)} for v in parse_pool(prim)] if prim else []}

    @app.get("/api/books/{book}/chapters", dependencies=[Depends(writer)])
    async def chapters(book: str):
        try:
            return {"chapters": store.chapter_numbers(book)}
        except BookError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/books/{book}/chapters/{n}", dependencies=[Depends(writer)])
    async def chapter(book: str, n: int):
        try:
            return {"chapter": n, "paragraphs": [asdict(p) for p in store.paragraphs(book, n)]}
        except BookError as e:
            raise HTTPException(404, str(e))

    @app.put("/api/books/{book}/paragraphs/{pid}", dependencies=[Depends(writer)])
    async def put_paragraph(book: str, pid: str, body: EditIn):
        try:
            return asdict(store.replace_paragraph(book, pid, body.text))
        except BookError as e:
            raise HTTPException(404, str(e))

    ui_dir = os.getenv("UI_DIR")  # dev convenience; in Docker nginx serves the UI
    if ui_dir:
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app

