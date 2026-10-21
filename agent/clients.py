"""Thin async clients for the Unraid-hosted OpenAI-compatible services (llama-swap, whisper, TTS)."""
from __future__ import annotations

import httpx

from settings import Settings


class LLM:
    def __init__(self, s: Settings, client: httpx.AsyncClient | None = None):
        self.s = s
        self.http = client or httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))

    async def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        r = await self.http.post(
            f"{self.s.llama_swap_url}/v1/chat/completions",
            json={"model": self.s.llm_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def models(self) -> list[str]:
        r = await self.http.get(f"{self.s.llama_swap_url}/v1/models", timeout=5)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


class STT:
    def __init__(self, s: Settings, client: httpx.AsyncClient | None = None):
        self.s = s
        self.http = client or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10))

    async def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> str:
        r = await self.http.post(
            self.s.whisper_url,
            files={"file": (filename, audio, content_type)},
            data={"model": self.s.whisper_model, "response_format": "json"},
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


class TTS:
    def __init__(self, s: Settings, client: httpx.AsyncClient | None = None):
        self.s = s
        self.http = client or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10))

    @property
    def enabled(self) -> bool:
        return bool(self.s.tts_url)

    async def speak(self, text: str, voice: str | None = None) -> bytes:
        payload = {"model": self.s.tts_model, "input": text, "voice": voice or self.s.voice_profile.get("voice") or self.s.tts_voice}
        r = await self.http.post(self.s.tts_url, json=payload)
        r.raise_for_status()
        return r.content
