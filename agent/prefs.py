"""Writer preferences kept on the server (so they follow the writer to any device): <state>/prefs.json."""
from __future__ import annotations

import json
import threading
from pathlib import Path

FIELDS = {"writer_age": 20, "display_name": 60, "surprise": 8, "strictness": 8, "look": 400, "chat_engine": 6, "listen_engine": 6, "voice_engine": 6,
          "voice_mode": 10, "narrator_reader": 8, "live_interrupt": 4, "web_lookup": 4, "auto_save": 3, "continuity": 3, "genre": 8}  # name -> max length
ENGINE = ("auto", "local", "cloud")
CHOICES = {"surprise": ("off", "low", "high"), "strictness": ("off", "gentle", "strict"), "chat_engine": ENGINE, "listen_engine": ENGINE, "voice_engine": ENGINE,
           "voice_mode": ("same", "separate", "characters"), "narrator_reader": ("female", "male"), "live_interrupt": ("on", "off"), "web_lookup": ("on", "off"), "auto_save": ("on", "off"), "continuity": ("on", "off"), "genre": ("story", "history")}


class Prefs:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "prefs.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def all(self) -> dict:
        try:
            d = json.loads(self.path.read_text()) if self.path.exists() else {}
        except (OSError, ValueError):
            d = {}
        return {k: str(d.get(k, "")) for k in FIELDS}

    def get(self, key: str, default: str = "") -> str:
        return self.all().get(key) or default

    def update(self, changes: dict) -> dict:
        with self._lock:
            cur = self.all()
            for k, v in changes.items():
                if k in FIELDS and v is not None:
                    v = str(v).strip()[: FIELDS[k]]
                    if k in CHOICES and v not in CHOICES[k]:
                        continue  # only the listed settings are accepted
                    cur[k] = v
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cur))
            tmp.replace(self.path)
            return cur
