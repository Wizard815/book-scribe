import shutil, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]


class FakeLLM:
    def __init__(self, replies=None):
        self.replies, self.calls = list(replies or []), []
        self.http = type("H", (), {"aclose": staticmethod(lambda: _noop())})()

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "Okay."

    async def models(self):
        return ["fake"]


async def _noop():
    return None


@pytest.fixture
def books(tmp_path):
    shutil.copytree(ROOT / "data/books", tmp_path / "books")
    return tmp_path / "books"
