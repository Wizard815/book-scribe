"""Async clients for OpenAI-compatible services. Every call goes through the tier router, so a dead T1
(ROCm box down, model unloaded) falls through to T2 and T3 without the caller noticing."""
from __future__ import annotations

import httpx

from providers import Profile
from router import AllTiersFailed, NotConfigured, Router  # noqa: F401  (re-exported for main.py)

DEFAULT_TIMEOUT = {"chat": 180.0, "stt": 120.0, "tts": 120.0, "embed": 60.0, "image": 120.0}


def headers(p: Profile) -> dict:
    return {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}


def timeout_for(role: str, p: Profile) -> httpx.Timeout:
    # Short connect timeout: an unreachable box should fall through quickly. Long read: cold model loads are slow.
    return httpx.Timeout(p.timeout or DEFAULT_TIMEOUT.get(role, 120.0), connect=5.0)


class _Base:
    role = ""

    def __init__(self, router: Router, client: httpx.AsyncClient | None = None):
        self.router = router
        self.http = client or httpx.AsyncClient()


class LLM(_Base):
    role = "chat"

    async def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        async def go(p: Profile) -> str:
            r = await self.http.post(
                f"{p.base_url}/v1/chat/completions", headers=headers(p), timeout=timeout_for("chat", p),
                json={"model": p.model or "default", "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        return await self.router.call("chat", go)

    native_ok: bool | None = None  # None: not tried yet; False: the server refused function definitions once, so the text format is used

    async def chat_tools(self, messages: list[dict], tools: list[dict], temperature: float = 0.7, max_tokens: int = 1200) -> dict:
        """One turn with native function calling. -> the assistant message {"content", "tool_calls"}."""
        import harness

        async def go(p: Profile) -> dict:
            r = await self.http.post(
                f"{p.base_url}/v1/chat/completions", headers=headers(p), timeout=timeout_for("chat", p),
                json={"model": p.model or "default", "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "tools": tools, "tool_choice": "auto"})
            if r.status_code in (400, 404, 422, 501) and "tool" in r.text.lower():
                self.native_ok = False
                raise harness.NativeToolsUnsupported(r.text[:200])
            r.raise_for_status()
            self.native_ok = True
            return r.json()["choices"][0]["message"]

        try:
            return await self.router.call("chat", go)
        except harness.NativeToolsUnsupported:
            raise
        except Exception as e:
            if isinstance(getattr(e, "__cause__", None), harness.NativeToolsUnsupported) or "NativeToolsUnsupported" in repr(e):
                raise harness.NativeToolsUnsupported(str(e)[:200])
            raise

    async def models(self) -> list[str]:
        return await self.router.call("chat", lambda p: list_models(self.http, p))


class STT(_Base):
    role = "stt"

    async def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str = "audio/webm", hint: str = "") -> str:
        """`hint` is a short list of words the speech model should expect (character and place names): it makes it spell them right."""
        async def go(p: Profile) -> str:
            data = {"model": p.model or "whisper-1", "response_format": "json"}
            if hint:
                data["prompt"] = hint[:600]
            r = await self.http.post(
                f"{p.base_url}/v1/audio/transcriptions", headers=headers(p), timeout=timeout_for("stt", p),
                files={"file": (filename, audio, content_type)}, data=data)
            r.raise_for_status()
            return (r.json().get("text") or "").strip()

        return await self.router.call("stt", go)


class TTS(_Base):
    role = "tts"

    def __init__(self, router: Router, client: httpx.AsyncClient | None = None, voices=None, default_book: str = "demo"):
        super().__init__(router, client)
        self.voices, self.default_book = voices, default_book

    @property
    def enabled(self) -> bool:
        return bool(self.router.reg.chain("tts"))

    async def speak(self, text: str, speaker: str = "assistant", book: str | None = None, draft: str | None = None) -> bytes:
        """Speak `text` as `speaker` (assistant, narrator or a character). The voice is resolved per provider *inside* the
        failover loop: if T1 is down and T3 answers, the same character gets T3's closest voice, not a T1 voice name."""
        async def go(p: Profile) -> bytes:
            voice, speed, style = (p.voice or "alloy", 1.0, "")
            if self.voices:
                r = self.voices.resolve(book or self.default_book, speaker, p, draft)
                voice, speed, style = r.voice or voice, r.speed, r.style
            body = {"model": p.model or "tts-1", "input": text, "voice": voice}
            if abs(speed - 1.0) > 0.01:
                body["speed"] = speed
            if style and p.style_field:
                body[p.style_field] = style
            r = await self.http.post(f"{p.base_url}/v1/audio/speech", headers=headers(p), timeout=timeout_for("tts", p), json=body)
            r.raise_for_status()
            return r.content

        return await self.router.call("tts", go)


class ImageGen(_Base):
    """Makes a picture from a description: any OpenAI-compatible /v1/images/generations server (a local Stable Diffusion / Flux box, or a cloud model)."""
    role = "image"

    @property
    def enabled(self) -> bool:
        return bool(self.router.reg.chain("image"))

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        import base64

        async def go(p: Profile) -> bytes:
            r = await self.http.post(f"{p.base_url}/v1/images/generations", headers=headers(p), timeout=timeout_for("image", p),
                                     json={"model": p.model or "image", "prompt": prompt[:1000], "n": 1, "size": size, "response_format": "b64_json"})
            r.raise_for_status()
            d = (r.json().get("data") or [{}])[0]
            if d.get("b64_json"):
                return base64.b64decode(d["b64_json"])
            if d.get("url"):  # some servers only return an address on their own host
                g = await self.http.get(d["url"] if d["url"].startswith("http") else f"{p.base_url}{d['url']}", headers=headers(p), timeout=timeout_for("image", p))
                g.raise_for_status()
                return g.content
            raise ValueError("the image server returned no picture")

        return await self.router.call("image", go)


async def list_models(http: httpx.AsyncClient, p: Profile) -> list[str]:
    r = await http.get(f"{p.base_url}/v1/models", headers=headers(p), timeout=8)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


async def probe(http: httpx.AsyncClient, role: str, p: Profile) -> dict:
    """Connectivity check for the admin 'Test' button. Tests this one profile, bypassing the tiers. Never raises."""
    out: dict = {"ok": False, "models": []}
    try:
        try:
            out["models"] = await list_models(http, p)
            out["ok"] = True
        except httpx.HTTPStatusError as e:  # many whisper/TTS servers have no /v1/models
            out["ok"] = e.response.status_code in (404, 405)
            out["detail"] = f"/v1/models → {e.response.status_code}"
        if role in ("chat", "graph") and out["ok"]:
            r = await http.post(f"{p.base_url}/v1/chat/completions", headers=headers(p), timeout=60,
                                json={"model": p.model or "default", "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 5})
            out["ok"] = r.status_code == 200
            out["detail"] = "completion ok" if out["ok"] else f"completion → {r.status_code}: {r.text[:160]}"
        elif role == "embed" and out["ok"]:
            r = await http.post(f"{p.base_url}/v1/embeddings", headers=headers(p), timeout=60, json={"model": p.model or "embed", "input": "ok"})
            out["ok"] = r.status_code == 200
            out["detail"] = f"{len(r.json()['data'][0]['embedding'])} dimensions" if out["ok"] else f"embeddings → {r.status_code}: {r.text[:160]}"
    except Exception as e:
        out["detail"] = f"{e.__class__.__name__}: {str(e)[:160]}"
    return out
