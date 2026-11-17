"""Adding a file to the library: read it, (look at it / listen to it if needed), file it, and tell the journal.

Reading is done by extract.py. Pictures and scanned pages go to the chat model as images (it must be vision-capable; the admin chooses
the model). Audio goes to the speech-to-text backend. If the needed model is missing or cannot do it, the original file is still kept,
marked "not read yet", and the writer is told plainly: nothing is silently dropped.
"""
from __future__ import annotations

import asyncio
import base64
import re

import extract as ex_mod
from extract import Extracted, Unreadable

VISION_PROMPT = (
    "This picture was added to a writer's book project. First, write out every word of text you can see, exactly as written "
    "(handwriting too; put [?] after any word you cannot make out). Then describe what the picture shows in plain sentences: people, "
    "places, objects, clothes, the era, the mood. If it is a document, chart or family tree, list the names, dates and relationships. "
    "No introduction and no commentary."
)
CANT_SEE = re.compile(r"\b(can(?:'|no)?t|cannot|unable to|not able to|don't have the ability to)\b[^.]{0,40}\b(see|view|process|access|analy[sz]e|look at)\b[^.]{0,30}\b(image|picture|photo|attachment)s?\b", re.I)


async def _look(llm, images: list[bytes]) -> str:
    sem = asyncio.Semaphore(2)

    async def one(i: int, jpg: bytes) -> str:
        async with sem:
            url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
            msg = [{"role": "user", "content": [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]
            out = (await llm.chat(msg, temperature=0.2, max_tokens=1500)).strip()
            if CANT_SEE.search(out[:300]):
                raise RuntimeError("the chat model cannot read pictures")
            return out

    parts = await asyncio.gather(*(one(i, im) for i, im in enumerate(images)))
    return "\n\n".join((f"Page {i + 1}:\n{p}" if len(parts) > 1 else p) for i, p in enumerate(parts))


async def describe_picture(llm, jpg: bytes) -> str:
    """Two plain sentences about a picture, for a writer who may not look at it. "" when no vision model is available."""
    url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
    msg = [{"role": "user", "content": [{"type": "text", "text": "Describe this picture in two plain sentences for someone who cannot see it: what is in it, the main colours, the mood. No introduction."},
                                        {"type": "image_url", "image_url": {"url": url}}]}]
    out = (await llm.chat(msg, temperature=0.2, max_tokens=160)).strip()
    return "" if CANT_SEE.search(out[:300]) else out


async def add_file(book: str, filename: str, data: bytes, library, llm, stt, journal=None) -> dict:
    """-> the library item's meta, plus "message": one plain sentence about what happened. Raises Unreadable for a file that cannot be read."""
    import hashlib

    digest = hashlib.sha1(data).hexdigest()
    for m in library.items(book):  # the same file again: say so, rather than keeping two copies
        if m.get("sha1") == digest:
            m = dict(m)
            m["message"] = f"You already added this one: {m['title']}. I will use that copy."
            m["duplicate"] = True
            return m
    ex: Extracted = await asyncio.to_thread(ex_mod.extract, data, filename)
    text, status, notes = ex.text, "read", list(ex.notes)

    if ex.kind in ("image", "scan"):
        try:
            seen = await _look(llm, ex.images)
            text = (text + "\n\n" if text.strip() else "") + seen if ex.kind == "scan" else seen
        except Exception as e:
            status = "unread"
            notes.append("I kept your picture, and you can still put it in the book, but I cannot see pictures yet, so I cannot tell you what is in it."
                         if "cannot read pictures" in str(e) or "400" in str(e) or "422" in str(e) else "I kept your picture, but I could not look at it just now. You can still put it in the book.")
    elif ex.kind == "audio":
        try:
            text = await stt.transcribe(ex.audio or b"", ex.audio_name or filename)
            if not text.strip():
                status = "unread"
                notes.append("I could not hear any speech in that recording.")
        except Exception:
            status = "unread"
            notes.append("I kept the recording but could not turn it into words just now.")

    meta = library.add(book, filename, data, ex.title, ex.kind, text, status=status, notes=notes, pages=ex.pages, tree=ex.tree, sha1=digest)
    meta["message"] = _message(meta, ex)
    if journal:
        journal.append(book, "note", f"The writer added a file to the library: {meta['title']} ({meta['kind'].replace('_', ' ')}). " + (text[:300].replace("\n", " ") if status == "read" else ""))
    return meta


def _message(meta: dict, ex: Extracted) -> str:
    t = meta["title"]
    if meta["status"] != "read":
        return f"I added {t}. " + (meta["notes"][-1] if meta["notes"] else "I could not read it yet.")
    if meta["kind"] == "family_tree" and ex.tree:
        return f"I added the family tree {t}: {ex.tree.summary()}. Ask me about anyone in it."
    if meta["kind"] == "database":
        return f"I added {t}. " + (ex.notes[0] if ex.notes else "")
    words = len(ex.text.split()) if ex.text else meta["chars"] // 6
    return f"I added {t} to the library and read it: about {words:,} words. You can ask me about it any time."
