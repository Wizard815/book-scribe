"""Runtime settings: environment variables first, then config.yaml for writer-facing options."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Settings:
    # Optional first-run seeds for the four OpenAI-compatible backends (base URLs, no /v1). Anything left empty
    # is simply not pre-filled; add or change providers in the admin menu at any time.
    chat_url: str = ""
    chat_model: str = ""
    embed_url: str = ""  # defaults to chat_url when unset
    embed_model: str = ""
    stt_url: str = ""
    stt_model: str = ""
    tts_url: str = ""
    tts_model: str = ""
    tts_voice: str = ""
    project_dir: Path = Path("/data/books")
    default_book: str = "demo"
    mem_backend: str = "mem0"  # mem0 | file
    user_id: str = "writer"
    admin_token: str = ""  # required for /api/admin/*; empty disables the admin API
    # config.yaml → writer_profile
    writer_age: str | None = None
    tts_mode: str = "single"  # single | multi
    voice_profile: dict = field(default_factory=dict)
    # quality gate
    max_rewrite_attempts: int = 1

    @property
    def state_dir(self) -> Path:
        return self.project_dir / ".state"


def _env(*names: str) -> str:
    """First set variable wins; later names are legacy aliases."""
    return next((os.environ[n] for n in names if os.environ.get(n)), "")


def load_settings(config_path: str | None = None) -> Settings:
    s = Settings(
        chat_url=_env("CHAT_URL", "LLAMA_SWAP_URL").rstrip("/"),
        chat_model=_env("CHAT_MODEL", "LLM_MODEL"),
        embed_url=_env("EMBED_URL").rstrip("/"),
        embed_model=_env("EMBED_MODEL"),
        stt_url=_env("STT_URL", "WHISPER_URL").rstrip("/"),
        stt_model=_env("STT_MODEL", "WHISPER_MODEL"),
        tts_url=_env("TTS_URL").rstrip("/"),
        tts_model=_env("TTS_MODEL"),
        tts_voice=_env("TTS_VOICE"),
        project_dir=Path(os.getenv("PROJECT_DIR", "/data/books")),
        default_book=os.getenv("DEFAULT_BOOK", "demo"),
        mem_backend=os.getenv("MEM_BACKEND", "mem0"),
        user_id=os.getenv("WRITER_ID", "writer"),
        admin_token=os.getenv("ADMIN_TOKEN", ""),
    )
    path = Path(config_path or os.getenv("CONFIG_PATH", "/app/config.yaml"))
    if path.is_file():
        cfg = yaml.safe_load(path.read_text()) or {}
        wp = cfg.get("writer_profile") or {}
        age = wp.get("age")
        s.writer_age = str(age) if age else None
        tts = cfg.get("tts") or {}
        s.tts_mode = tts.get("mode", s.tts_mode)
        s.voice_profile = cfg.get("voice_profile") or {}
    return s
