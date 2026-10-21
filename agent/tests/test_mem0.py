"""Exercises the vendored mem0 for real (Qdrant on disk) against a stub OpenAI-compatible embedder."""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("mem0")

from memory.store import MemoryManager
from providers import Registry
from settings import Settings


def embed(text, dims=16):
    h = hashlib.sha256(text.lower().encode()).digest()
    return [b / 255 for b in h[:dims]]


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        out = json.dumps({"object": "list", "model": "e", "usage": {"prompt_tokens": 1, "total_tokens": 1},
                          "data": [{"object": "embedding", "index": i, "embedding": embed(t)} for i, t in enumerate(inputs)]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


def test_vendored_mem0_roundtrip(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = Settings(project_dir=tmp_path, llama_swap_url=f"http://127.0.0.1:{srv.server_port}", mem_backend="mem0")
    mgr = MemoryManager(s, Registry(s))
    mgr.add("Writer said: I prefer short sentences", "preference")
    hits = mgr.search("Writer said: I prefer short sentences", 3)
    assert mgr.backend == "mem0", mgr.error
    assert hits and "short sentences" in hits[0]
    srv.shutdown()
