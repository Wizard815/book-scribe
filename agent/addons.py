"""Addons: the administrator's own Python files that teach Book Scribe new tricks.

Set ADDONS_DIR to a folder. Every *.py file in it is loaded at start-up and may define

    def register(api):
        api.add_reader((".ftm",), my_reader)          # my_reader(data: bytes, filename: str) -> extract.Extracted | None
        api.add_tool("my_tool", "what it does", {"arg": "what it is"}, my_async_fn, required=("arg",))   # read-only helpers for the model
        api.settings                                   # the app Settings, read-only

A reader that returns None passes the file on to the built-in readers. Tools are shown to the model next to the built-in ones; they
receive (ctx, **args) and return text. Addons run with the server's full rights, so only the administrator (who sets ADDONS_DIR, outside the
web) can add them; the writer cannot upload code. See docs/ADDONS.md.
"""
from __future__ import annotations

import importlib.util
import logging
import traceback
from pathlib import Path

import extract
from harness import Tool

log = logging.getLogger("addons")


class AddonAPI:
    def __init__(self, settings, name: str, owner: "Addons"):
        self.settings, self._name, self._owner = settings, name, owner

    def add_reader(self, extensions, fn) -> None:
        exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in extensions)
        extract.ADDON_READERS.append((exts, fn))
        self._owner.readers.append((self._name, exts))

    def add_tool(self, name: str, description: str, args: dict, fn, required: tuple = ()) -> None:
        self._owner.tools.append(Tool(name, description, args, fn, required))


class Addons:
    def __init__(self, settings, directory: str | None):
        self.settings = settings
        self.dir = Path(directory).expanduser() if directory else None
        self.tools: list[Tool] = []
        self.readers: list[tuple[str, tuple]] = []
        self.status: list[dict] = []

    def load(self) -> "Addons":
        extract.ADDON_READERS.clear()
        if not self.dir or not self.dir.is_dir():
            return self
        for path in sorted(self.dir.glob("*.py")):
            entry = {"name": path.stem, "ok": False, "error": ""}
            try:
                spec = importlib.util.spec_from_file_location(f"bs_addon_{path.stem}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "register"):
                    raise RuntimeError("the file has no register(api) function")
                mod.register(AddonAPI(self.settings, path.stem, self))
                entry["ok"] = True
            except Exception as e:
                entry["error"] = f"{e.__class__.__name__}: {str(e)[:160]}"
                log.warning("addon %s failed: %s", path.name, traceback.format_exc())
            self.status.append(entry)
        return self
