"""Model provider registry: which backend serves each role, editable at runtime from the admin menu.

Roles: chat (LLM), embed (embeddings, used by mem0), stt (speech-to-text), tts (text-to-speech),
graph (the model graphify uses to extract the story graph; a stronger/cheaper model than chat is fine).
Every provider is an OpenAI-compatible endpoint, so "local" (llama-swap, faster-whisper on Unraid) and
"cloud" (OpenAI, OpenRouter, Anthropic's OpenAI-compatible API, ...) are the same shape — just a base URL,
an optional API key and a model name. State lives in <state_dir>/providers.json.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from settings import Settings

ROLES = ("chat", "embed", "stt", "tts", "graph")

# Suggested starting points for the admin form. Purely UI sugar; any URL works.
PRESETS = [
    {"id": "local", "label": "Local (llama-swap / whisper on your LAN)", "kind": "local", "base_url": "http://UNRAID_IP:8080"},
    {"id": "openai", "label": "OpenAI", "kind": "cloud", "base_url": "https://api.openai.com"},
    {"id": "openrouter", "label": "OpenRouter", "kind": "cloud", "base_url": "https://openrouter.ai/api"},
    {"id": "anthropic", "label": "Anthropic (OpenAI-compatible)", "kind": "cloud", "base_url": "https://api.anthropic.com"},
    {"id": "custom", "label": "Custom OpenAI-compatible URL", "kind": "cloud", "base_url": ""},
]


@dataclass
class Profile:
    id: str
    label: str
    kind: str = "local"  # local | cloud (display only)
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    voice: str = ""  # tts only

    def public(self) -> dict:
        d = asdict(self)
        key = d.pop("api_key")
        d["has_key"] = bool(key)
        d["key_hint"] = ("…" + key[-4:]) if len(key) >= 8 else ("set" if key else "")
        return d


def _strip(url: str, *suffixes: str) -> str:
    url = url.rstrip("/")
    for sfx in suffixes:
        if url.endswith(sfx):
            url = url[: -len(sfx)]
    return url.rstrip("/")


def _slug(label: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "profile") + "-" + uuid.uuid4().hex[:4]


class ProviderError(Exception):
    pass


class Registry:
    def __init__(self, s: Settings):
        self.path = s.state_dir / "providers.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.version = 0
        self.profiles: dict[str, list[Profile]] = {r: [] for r in ROLES}
        self.active: dict[str, str] = {}
        if self.path.exists():
            self._read()
            missing = [r for r in ROLES if not self.profiles[r]]  # roles added after this file was written
            if missing:
                fresh = Registry.__new__(Registry)
                fresh.profiles, fresh.active = {r: [] for r in ROLES}, {}
                fresh._seed(s)
                for r in missing:
                    self.profiles[r], self.active[r] = fresh.profiles[r], fresh.active[r]
                self._write()
        else:
            self._seed(s)
            self._write()

    # --- persistence ----------------------------------------------------
    def _seed(self, s: Settings) -> None:
        seeds = {
            "chat": Profile("local-chat", "Local chat", "local", s.llama_swap_url, "", s.llm_model),
            "embed": Profile("local-embed", "Local embeddings", "local", s.llama_swap_url, "", s.embed_model),
            "stt": Profile("local-stt", "Local whisper", "local", _strip(s.whisper_url, "/v1/audio/transcriptions"), "", s.whisper_model),
            "tts": Profile("local-tts", "Local TTS", "local", _strip(s.tts_url, "/v1/audio/speech"), "", s.tts_model, s.tts_voice),
            "graph": Profile("local-graph", "Local graph extraction", "local", s.llama_swap_url, "", s.llm_model),
        }
        for role, p in seeds.items():
            self.profiles[role] = [p]
            self.active[role] = p.id
        self._seeds = seeds

    def _read(self) -> None:
        raw = json.loads(self.path.read_text())
        for role in ROLES:
            self.profiles[role] = [Profile(**p) for p in raw.get("profiles", {}).get(role, [])]
        self.active = raw.get("active", {})

    def _write(self) -> None:
        data = {"active": self.active, "profiles": {r: [asdict(p) for p in ps] for r, ps in self.profiles.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.chmod(0o600)  # holds API keys
        tmp.replace(self.path)
        self.version += 1

    # --- queries ----------------------------------------------------------
    def get(self, role: str) -> Profile | None:
        """Active profile for a role, or None if the role is unconfigured (e.g. no TTS)."""
        for p in self.profiles[self._role(role)]:
            if p.id == self.active.get(role):
                return p if p.base_url else None
        return None

    def snapshot(self) -> dict:
        return {
            "roles": {
                r: {"active": self.active.get(r), "profiles": [p.public() for p in self.profiles[r]]}
                for r in ROLES
            },
            "presets": PRESETS,
        }

    @staticmethod
    def _role(role: str) -> str:
        if role not in ROLES:
            raise ProviderError(f"unknown role {role!r}")
        return role

    # --- mutations ----------------------------------------------------------
    def upsert(self, role: str, data: dict) -> Profile:
        """Create or update a profile. An empty api_key on update keeps the stored key."""
        with self._lock:
            self._role(role)
            base = _strip(str(data.get("base_url", "")).strip(), "/v1/chat/completions", "/v1/audio/transcriptions", "/v1/audio/speech", "/v1/embeddings", "/v1")
            if base and not re.match(r"^https?://", base):
                raise ProviderError("base_url must start with http:// or https://")
            kind = data.get("kind", "local")
            if kind not in ("local", "cloud"):
                raise ProviderError("kind must be local or cloud")
            pid = data.get("id")
            existing = next((p for p in self.profiles[role] if p.id == pid), None) if pid else None
            if pid and not existing:
                raise ProviderError("unknown profile id")
            p = existing or Profile(_slug(str(data.get("label") or "profile")), "")
            p.label = str(data.get("label") or p.label or "Unnamed")[:60]
            p.kind, p.base_url = kind, base
            p.model = str(data.get("model", p.model)).strip()
            p.voice = str(data.get("voice", p.voice)).strip()
            if data.get("api_key"):
                p.api_key = str(data["api_key"]).strip()
            if data.get("clear_key"):
                p.api_key = ""
            if not existing:
                self.profiles[role].append(p)
                self.active.setdefault(role, p.id)
            self._write()
            return p

    def delete(self, role: str, pid: str) -> None:
        with self._lock:
            ps = self.profiles[self._role(role)]
            if len(ps) <= 1:
                raise ProviderError("keep at least one profile per role")
            self.profiles[role] = [p for p in ps if p.id != pid]
            if self.active.get(role) == pid:
                self.active[role] = self.profiles[role][0].id
            self._write()

    def activate(self, role: str, pid: str) -> None:
        with self._lock:
            if not any(p.id == pid for p in self.profiles[self._role(role)]):
                raise ProviderError("unknown profile id")
            self.active[role] = pid
            self._write()
