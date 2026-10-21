"""Runtime settings: environment variables first, then config.yaml for writer-facing options."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Settings:
    llama_swap_url: str = "http://localhost:8080"
    llm_model: str = "default"
    embed_model: str = "embed"
    whisper_url: str = "http://localhost:8000/v1/audio/transcriptions"
    whisper_model: str = "whisper-1"
    tts_url: str = ""  # OpenAI-compatible /v1/audio/speech; empty disables server TTS
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    project_dir: Path = Path("/data/books")
    default_book: str = "demo"
    mem_backend: str = "mem0"  # mem0 | file
    user_id: str = "writer"
    # config.yaml → writer_profile
    writer_age: str | None = None
    tts_mode: str = "single"  # single | multi
    voice_profile: dict = field(default_factory=dict)
    # quality gate
    max_rewrite_attempts: int = 1

    @property
    def state_dir(self) -> Path:
        return self.project_dir / ".state"


def load_settings(config_path: str | None = None) -> Settings:
    s = Settings(
        llama_swap_url=os.getenv("LLAMA_SWAP_URL", Settings.llama_swap_url).rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embed_model=os.getenv("EMBED_MODEL", Settings.embed_model),
        whisper_url=os.getenv("WHISPER_URL", Settings.whisper_url),
        whisper_model=os.getenv("WHISPER_MODEL", Settings.whisper_model),
        tts_url=os.getenv("TTS_URL", ""),
        tts_model=os.getenv("TTS_MODEL", Settings.tts_model),
        tts_voice=os.getenv("TTS_VOICE", Settings.tts_voice),
        project_dir=Path(os.getenv("PROJECT_DIR", "/data/books")),
        default_book=os.getenv("DEFAULT_BOOK", "demo"),
        mem_backend=os.getenv("MEM_BACKEND", "mem0"),
        user_id=os.getenv("WRITER_ID", "writer"),
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
