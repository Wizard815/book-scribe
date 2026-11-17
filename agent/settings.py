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
    image_url: str = ""  # an OpenAI-compatible image generation server (IMAGE_URL), for pictures in the book
    image_model: str = ""
    project_dir: Path = Path("/data/books")
    default_book: str = "demo"
    mem_backend: str = "mem0"  # mem0 | file
    user_id: str = "writer"
    admin_token: str = ""  # the admin's password (16+ random characters); empty disables the admin login and API
    admin_user: str = "admin"
    writer_username: str = ""  # optional first-run seed for the single writer account (the admin can also create it)
    writer_password: str = ""
    trust_proxy: bool = False  # take the client address from CF-Connecting-IP / X-Forwarded-For (only behind your tunnel/proxy)
    # config.yaml → writer_profile
    writer_age: str | None = None
    tts_mode: str = "single"  # single | multi
    voice_profile: dict = field(default_factory=dict)
    context_chars: int = 9000  # recent conversation kept word for word; older turns are folded into a summary
    search_dirs: list[str] = field(default_factory=list)  # folders shared with the file finder (FILE_SEARCH_DIRS), read-only
    searxng_url: str = ""  # a SearXNG server for web search (SEARXNG_URL); off when empty
    wikipedia_url: str = ""  # Wikipedia, or an offline mirror such as Kiwix (WIKIPEDIA_URL, e.g. https://en.wikipedia.org); off when empty
    host_books_path: str = ""  # where the books folder is on the writer's computer, shown beside the inbox folder (HOST_BOOKS_PATH)
    addons_dir: str | None = None  # folder of the administrator's addon .py files (see addons.py)
    index_debounce: float = 15.0  # seconds to wait after a save before updating the embedding index
    # reading aloud
    speech_llm: bool = True  # let the chat model assign speakers to dialogue the rules can't place (cached)
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
        image_url=_env("IMAGE_URL").rstrip("/"),
        image_model=_env("IMAGE_MODEL"),
        project_dir=Path(os.getenv("PROJECT_DIR", "/data/books")),
        default_book=os.getenv("DEFAULT_BOOK", "demo"),
        mem_backend=os.getenv("MEM_BACKEND", "mem0"),
        user_id=os.getenv("WRITER_ID", "writer"),
        admin_token=os.getenv("ADMIN_TOKEN", ""),
        admin_user=os.getenv("ADMIN_USER", "admin"),
        writer_username=os.getenv("WRITER_USERNAME", ""),
        writer_password=os.getenv("WRITER_PASSWORD", ""),
        trust_proxy=os.getenv("TRUST_PROXY", "").lower() in ("1", "true", "yes"),
        addons_dir=os.getenv("ADDONS_DIR") or None,
        searxng_url=os.getenv("SEARXNG_URL", "").strip().rstrip("/"),
        wikipedia_url=os.getenv("WIKIPEDIA_URL", "").strip().rstrip("/"),
        host_books_path=os.getenv("HOST_BOOKS_PATH", "").rstrip("/\\"),
        search_dirs=[d for d in os.getenv("FILE_SEARCH_DIRS", "").split(os.pathsep) if d.strip()],
    )
    if os.getenv("CONTEXT_CHARS", "").isdigit():
        s.context_chars = max(2000, int(os.environ["CONTEXT_CHARS"]))
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
