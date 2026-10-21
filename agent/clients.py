"""Async clients for OpenAI-compatible services. Each call resolves the *active* provider from the
registry, so switching models in the admin menu takes effect on the next request with no restart."""
from __future__ import annotations

import httpx

from providers import Profile, Registry


def headers(p: Profile) -> dict:
    return {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}


class NotConfigured(Exception):
    pass


class _Base:
    role = ""

    def __init__(self, reg: Registry, client: httpx.AsyncClient | None = None):
        self.reg = reg
        self.http = client or httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))

    def profile(self) -> Profile:
        p = self.reg.get(self.role)
        if not p:
            raise NotConfigured(f"No {self.role} provider configured. Open the admin menu.")
        return p


class LLM(_Base):
    role = "chat"

    async def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        p = self.profile()
        r = await self.http.post(
            f"{p.base_url}/v1/chat/completions", headers=headers(p),
            json={"model": p.model or "default", "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def models(self) -> list[str]:
        return await list_models(self.http, self.profile())


class STT(_Base):
    role = "stt"

    async def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> str:
        p = self.profile()
        r = await self.http.post(
            f"{p.base_url}/v1/audio/transcriptions", headers=headers(p),
            files={"file": (filename, audio, content_type)},
            data={"model": p.model or "whisper-1", "response_format": "json"},
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


class TTS(_Base):
    role = "tts"

    @property
    def enabled(self) -> bool:
        return self.reg.get("tts") is not None

    async def speak(self, text: str, voice: str | None = None) -> bytes:
        p = self.profile()
        r = await self.http.post(
            f"{p.base_url}/v1/audio/speech", headers=headers(p),
            json={"model": p.model or "tts-1", "input": text, "voice": voice or p.voice or "alloy"},
        )
        r.raise_for_status()
        return r.content


async def list_models(http: httpx.AsyncClient, p: Profile) -> list[str]:
    r = await http.get(f"{p.base_url}/v1/models", headers=headers(p), timeout=8)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


async def probe(http: httpx.AsyncClient, role: str, p: Profile) -> dict:
    """Connectivity check for the admin 'Test' button. Never raises."""
    out: dict = {"ok": False, "models": []}
    try:
        try:
            out["models"] = await list_models(http, p)
            out["ok"] = True
        except httpx.HTTPStatusError as e:  # many whisper/TTS servers have no /v1/models
            out["ok"] = e.response.status_code in (404, 405)
            out["detail"] = f"/v1/models → {e.response.status_code}"
        if role == "chat" and out["ok"]:
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
