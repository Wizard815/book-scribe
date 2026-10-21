"""Model provider registry with tiered fallback, editable at runtime from the admin menu.

Roles: chat (LLM), embed (embeddings, used by mem0), stt (speech-to-text), tts (text-to-speech),
graph (the model graphify uses to extract the story graph; empty tiers = reuse the chat tiers).

Every provider is an OpenAI-compatible endpoint (base URL + optional key + model), so "local" (Argos,
llama-swap, whisper.cpp, a CPU container) and "cloud" (OpenAI, OpenRouter, ...) are the same shape.

Each role has three ordered **tiers** (default labels: T1 Local ROCm, T2 CPU fallback, T3 Frontier). A call tries
T1's profiles first; only when all of them fail does it try T2, then T3 (see router.py). A profile that is in no
tier is saved but never called. State lives in <state_dir>/providers.json.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from settings import Settings

ROLES = ("chat", "embed", "stt", "tts", "graph")
N_TIERS = 3
DEFAULT_LABELS = ["T1 · Local ROCm", "T2 · CPU fallback", "T3 · Frontier"]

# Suggested starting points for the admin form. Purely UI sugar; any URL works.
PRESETS = [
    {"id": "local", "label": "Local / LAN server (any OpenAI-compatible)", "kind": "local", "base_url": "http://UNRAID_IP:8080"},
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
    timeout: float = 0  # seconds to wait for a reply; 0 = role default

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
        self.tiers: dict[str, list[list[str]]] = {r: [[] for _ in range(N_TIERS)] for r in ROLES}
        self.labels: list[str] = list(DEFAULT_LABELS)
        self.cooldown: int = 30  # seconds a failed profile is skipped before it is tried first again
        if self.path.exists():
            self._read()
            # roles added to the code since this file was written get seeded once; roles already in the file never are
            added = [self._seed_role(r, s) for r in ROLES if r not in self._known and not self.profiles[r]]
            if any(added):
                self._write()
        else:
            for r in ROLES:
                self._seed_role(r, s)
            self._write()

    # --- persistence ----------------------------------------------------
    @staticmethod
    def _seed_profile(role: str, s: Settings) -> Profile | None:
        """A starting provider from the environment, or None when that backend's URL isn't set."""
        url, model = {
            "chat": (s.chat_url, s.chat_model),
            "embed": (s.embed_url or s.chat_url, s.embed_model),
            "stt": (_strip(s.stt_url, "/v1/audio/transcriptions"), s.stt_model),
            "tts": (_strip(s.tts_url, "/v1/audio/speech"), s.tts_model),
        }.get(role, ("", ""))  # graph is never seeded: with empty tiers it reuses the chat tiers
        if not url:
            return None
        label = {"chat": "Chat", "embed": "Embeddings", "stt": "Speech-to-text", "tts": "Text-to-speech"}[role]
        return Profile(f"env-{role}", f"{label} (from .env)", "local", url, "", model, s.tts_voice if role == "tts" else "")

    def _seed_role(self, role: str, s: Settings) -> bool:
        p = self._seed_profile(role, s)
        if not p:
            return False
        self.profiles[role] = [p]
        self.tiers[role] = [[p.id], [], []]
        return True

    def _read(self) -> None:
        raw = json.loads(self.path.read_text())
        self._known = set(raw.get("profiles", {}))
        for role in ROLES:
            self.profiles[role] = [Profile(**p) for p in raw.get("profiles", {}).get(role, [])]
        self.labels = (raw.get("labels") or DEFAULT_LABELS)[:N_TIERS]
        self.cooldown = int(raw.get("cooldown", 30))
        for role in ROLES:
            ids = {p.id for p in self.profiles[role]}
            if "tiers" in raw:
                tiers = raw["tiers"].get(role) or [[] for _ in range(N_TIERS)]
            else:  # file from before tiers existed: the single active profile becomes T1
                a = raw.get("active", {}).get(role)
                tiers = [[a] if a else [], [], []]
            tiers = (list(tiers) + [[] for _ in range(N_TIERS)])[:N_TIERS]
            self.tiers[role] = [[i for i in t if i in ids] for t in tiers]

    def _write(self) -> None:
        data = {"labels": self.labels, "cooldown": self.cooldown, "tiers": self.tiers,
                "profiles": {r: [asdict(p) for p in ps] for r, ps in self.profiles.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.chmod(0o600)  # holds API keys
        tmp.replace(self.path)
        self.version += 1

    # --- queries ----------------------------------------------------------
    @staticmethod
    def _role(role: str) -> str:
        if role not in ROLES:
            raise ProviderError(f"unknown role {role!r}")
        return role

    def chain(self, role: str) -> list[tuple[int, Profile]]:
        """Callable providers in priority order as (tier index, profile): every T1 profile, then T2, then T3."""
        by_id = {p.id: p for p in self.profiles[self._role(role)]}
        return [(ti, by_id[i]) for ti, ids in enumerate(self.tiers[role]) for i in ids if i in by_id and by_id[i].base_url]

    def get(self, role: str) -> Profile | None:
        """Highest-priority provider for a role. Used where failover is impossible (embeddings -> mem0)."""
        c = self.chain(role)
        return c[0][1] if c else None

    def snapshot(self, health: dict | None = None) -> dict:
        health = health or {}
        return {
            "labels": self.labels, "cooldown": self.cooldown,
            "roles": {r: {"tiers": self.tiers[r], "profiles": [{**p.public(), "health": health.get(p.id)} for p in self.profiles[r]]} for r in ROLES},
            "presets": PRESETS,
        }

    # --- mutations ----------------------------------------------------------
    def _place(self, role: str, pid: str, tier: int) -> None:
        """tier 0 = unassigned, 1..N = that tier (appended at the end)."""
        for t in self.tiers[role]:
            if pid in t:
                t.remove(pid)
        if tier:
            self.tiers[role][tier - 1].append(pid)

    def upsert(self, role: str, data: dict) -> Profile:
        """Create or update a profile. An empty api_key on update keeps the stored key."""
        with self._lock:
            self._role(role)
            pid = data.get("id")
            existing = next((p for p in self.profiles[role] if p.id == pid), None) if pid else None
            if pid and not existing:
                raise ProviderError("unknown profile id")
            base = _strip(str(data.get("base_url", existing.base_url if existing else "")).strip(), "/v1/chat/completions", "/v1/audio/transcriptions", "/v1/audio/speech", "/v1/embeddings", "/v1")
            if base and not re.match(r"^https?://", base):
                raise ProviderError("base_url must start with http:// or https://")
            kind = data.get("kind", existing.kind if existing else "local")
            if kind not in ("local", "cloud"):
                raise ProviderError("kind must be local or cloud")
            try:
                timeout = float(data.get("timeout", existing.timeout if existing else 0) or 0)
            except (TypeError, ValueError):
                raise ProviderError("timeout must be a number of seconds")
            if not 0 <= timeout <= 3600:
                raise ProviderError("timeout must be between 0 and 3600 seconds")
            tier = data.get("tier")
            if tier is not None and tier not in range(N_TIERS + 1):
                raise ProviderError(f"tier must be 0 (unassigned) to {N_TIERS}")
            p = existing or Profile(_slug(str(data.get("label") or "profile")), "")
            p.label = str(data.get("label") or p.label or "Unnamed")[:60]
            p.kind, p.base_url, p.timeout = kind, base, timeout
            p.model = str(data.get("model", p.model)).strip()
            p.voice = str(data.get("voice", p.voice)).strip()
            if data.get("api_key"):
                p.api_key = str(data["api_key"]).strip()
            if data.get("clear_key"):
                p.api_key = ""
            if not existing:
                self.profiles[role].append(p)
            if tier is not None:
                self._place(role, p.id, tier)
            self._write()
            return p

    def delete(self, role: str, pid: str) -> None:
        with self._lock:
            self.profiles[self._role(role)] = [p for p in self.profiles[role] if p.id != pid]
            self._place(role, pid, 0)
            self._write()

    def set_tiers(self, role: str, tiers: list[list[str]]) -> None:
        with self._lock:
            ids = {p.id for p in self.profiles[self._role(role)]}
            if not isinstance(tiers, list) or len(tiers) != N_TIERS or not all(isinstance(t, list) for t in tiers):
                raise ProviderError(f"tiers must be a list of {N_TIERS} lists")
            flat = [i for t in tiers for i in t]
            if len(flat) != len(set(flat)):
                raise ProviderError("a profile can be in only one tier")
            if any(i not in ids for i in flat):
                raise ProviderError("unknown profile id in tiers")
            self.tiers[role] = [list(t) for t in tiers]
            self._write()

    def set_settings(self, labels: list[str] | None = None, cooldown: int | None = None) -> None:
        with self._lock:
            if labels is not None:
                if len(labels) != N_TIERS or not all(isinstance(l, str) and 0 < len(l.strip()) <= 30 for l in labels):
                    raise ProviderError(f"labels must be {N_TIERS} names of 1-30 characters")
                self.labels = [l.strip() for l in labels]
            if cooldown is not None:
                if not 0 <= int(cooldown) <= 3600:
                    raise ProviderError("cooldown must be 0-3600 seconds")
                self.cooldown = int(cooldown)
            self._write()
