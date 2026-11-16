"""Turn almost any file into text (or into pictures to be read by a vision model, or audio to be transcribed).

    extract(data, filename) -> Extracted

Text and documents: .txt .md .html .rtf .docx .odt .pptx .epub .pdf .csv .tsv .json .xlsx .xml, and old Office files when LibreOffice exists.
Family trees: .ged .gdz .gramps .ftm .ftmb (see genealogy.py).
Pictures: .jpg .png .gif .webp .bmp .tiff become downscaled JPEGs for a vision model. Scanned PDFs are rendered to pictures too.
Audio: .mp3 .wav .m4a .ogg .flac .webm .opus are handed back for speech-to-text.
Zip files: every readable member is read.
Addons (addons.py) can add more formats, or take over one (such as a custom Family Tree Maker reader).

Nothing here runs code from a file; documents are only parsed.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import genealogy

MAX_BYTES = 60 * 1024 * 1024
MAX_TEXT = 2_000_000
MAX_IMAGES = 10
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".webm", ".opus", ".aac", ".mp4"}


class Unreadable(Exception):
    """A file this app cannot read. The message is written for the writer to read aloud."""


@dataclass
class Extracted:
    text: str = ""
    kind: str = "text"  # text | document | family_tree | database | image | scan | audio
    title: str = ""
    pages: int = 0
    images: list[bytes] = field(default_factory=list)  # JPEG pictures for a vision model
    audio: bytes | None = None
    audio_name: str = ""
    tree: genealogy.Tree | None = None
    notes: list[str] = field(default_factory=list)


# addon-provided readers: [(extensions tuple, fn(bytes, filename) -> Extracted | None)]; later entries win
ADDON_READERS: list[tuple[tuple[str, ...], object]] = []


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _html_text(raw: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style", "noscript", "head"]):
        t.decompose()
    for br in soup.find_all(["br"]):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote"]):
        blk.append("\n\n" if blk.name != "li" and blk.name != "tr" else "\n")
    return _tidy(soup.get_text())


def _rtf_text(raw: str) -> str:
    raw = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", "replace"), raw)
    raw = re.sub(r"\{\\\*[^{}]*\}", "", raw)
    raw = re.sub(r"\\par[d]?\b", "\n", raw)
    raw = re.sub(r"\\(?:tab)\b", "\t", raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    return _tidy(re.sub(r"[{}]", "", raw))


def _docx(raw: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(raw))
    out = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            out.append(" | ".join(c.text.strip() for c in row.cells))
    return _tidy("\n\n".join(x for x in out if x.strip()))


def _zip_xml_text(raw: bytes, members: list[str], para_tags: tuple[str, ...]) -> str:
    from lxml import etree

    z = zipfile.ZipFile(io.BytesIO(raw))
    out = []
    for name in members:
        if name not in z.namelist():
            continue
        root = etree.fromstring(z.read(name), parser=etree.XMLParser(resolve_entities=False, no_network=True))
        for el in root.iter():
            if isinstance(el.tag, str) and el.tag.split("}")[-1] in para_tags:
                out.append("".join(el.itertext()))
    return _tidy("\n\n".join(x for x in out if x.strip()))


def _odt(raw: bytes) -> str:
    return _zip_xml_text(raw, ["content.xml"], ("p", "h"))


def _pptx(raw: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(raw))
    slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
    return _zip_xml_text(raw, slides, ("p",))


def _epub(raw: bytes) -> tuple[str, str]:
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
    title = ""
    for n in z.namelist():
        if n.lower().endswith(".opf"):
            m = re.search(r"<dc:title[^>]*>([^<]+)</dc:title>", z.read(n).decode("utf-8", "replace"))
            title = m.group(1).strip() if m else ""
    parts = [_html_text(z.read(n).decode("utf-8", "replace")) for n in names]
    return _tidy("\n\n".join(p for p in parts if p)), title


def _pdf(raw: bytes) -> Extracted:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise Unreadable("That PDF is locked with a password.")
        pages = [(pg.extract_text() or "") for pg in reader.pages]
    except Unreadable:
        raise
    except Exception:
        pages = []
    text = _tidy("\n\n".join(pages))
    ex = Extracted(text=text, kind="document", pages=len(pages), title="")
    if len(text) < 40 * max(1, min(len(pages), 3)):  # scanned: no real text layer, so read the page pictures instead
        imgs = _render_pdf(raw)
        if imgs:
            ex.kind, ex.images, ex.text = "scan", imgs, text
            if len(pages) > len(imgs):
                ex.notes.append(f"Only the first {len(imgs)} of {len(pages)} pages were read as pictures.")
    return ex


def _render_pdf(raw: bytes, limit: int = MAX_IMAGES) -> list[bytes]:
    exe = shutil.which("pdftoppm")
    if not exe:
        return []
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.pdf"
        src.write_bytes(raw)
        try:
            subprocess.run([exe, "-jpeg", "-r", "110", "-f", "1", "-l", str(limit), str(src), str(Path(d) / "pg")], check=True, timeout=120, capture_output=True)
        except Exception:
            return []
        return [p.read_bytes() for p in sorted(Path(d).glob("pg*.jpg"))][:limit]


def _image(raw: bytes) -> bytes:
    from PIL import Image, ImageOps

    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im)
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        im = im.convert("RGB")
    except Exception:
        raise Unreadable("I could not open that picture. HEIC photos from some phones are not supported: share it as a JPEG instead.")
    im.thumbnail((1800, 1800))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=82)
    return out.getvalue()


def _sheet(raw: bytes) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"Sheet {ws.title}")
        for n, row in enumerate(ws.iter_rows(values_only=True)):
            if n >= 3000:
                out.append("(more rows not shown)")
                break
            cells = [str(c) for c in row if c not in (None, "")]
            if cells:
                out.append(" | ".join(cells))
    return _tidy("\n".join(out))


def _delimited(text: str, delim: str) -> str:
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))[:5000]
    if not rows:
        return ""
    head = rows[0]
    if len(rows) > 1 and all(head):
        return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(head, r) if v.strip()) for r in rows[1:])
    return "\n".join(" | ".join(r) for r in rows)


def _office_via_libreoffice(raw: bytes, ext: str) -> str:
    exe = shutil.which("libreoffice") or shutil.which("soffice")
    if not exe:
        raise Unreadable(f"I cannot read {ext} files here. Save it as .docx or .pdf and add it again.")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"in{ext}"
        src.write_bytes(raw)
        try:
            subprocess.run([exe, "--headless", "--convert-to", "txt:Text", "--outdir", d, str(src)], check=True, timeout=90, capture_output=True,
                           env={"HOME": d, "PATH": "/usr/bin:/bin"})
            return _tidy((Path(d) / "in.txt").read_text(errors="replace"))
        except Exception:
            raise Unreadable(f"I could not read that {ext} file. Save it as .docx or .pdf and add it again.")


def _printable_text(raw: bytes) -> str | None:
    sample = raw[:4096]
    if not sample or b"\x00" in sample:
        return None
    text = genealogy.decode_text(raw)
    ok = sum(1 for c in text[:4096] if c.isprintable() or c in "\n\r\t")
    return text if ok / max(1, len(text[:4096])) > 0.95 else None


def _stem_title(filename: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(filename).stem).strip().title() or "Untitled"


def extract(data: bytes, filename: str, depth: int = 0) -> Extracted:
    if len(data) > MAX_BYTES:
        raise Unreadable(f"That file is too big ({len(data) // 1_000_000} MB). The limit is {MAX_BYTES // 1_000_000} MB.")
    if not data:
        raise Unreadable("That file is empty.")
    name = Path(filename or "file").name
    ext = Path(name).suffix.lower()
    title = _stem_title(name)

    for exts, fn in reversed(ADDON_READERS):  # an addon that claims the extension goes first
        if ext in exts:
            got = fn(data, name)
            if got is not None:
                got.title = got.title or title
                return got

    ex = _builtin(data, name, ext, depth)
    ex.title = ex.title or title
    ex.text = ex.text[:MAX_TEXT]
    return ex


def _builtin(data: bytes, name: str, ext: str, depth: int) -> Extracted:
    # family trees
    if ext in (".ged", ".gdz", ".gramps", ".ftm", ".ftmb") or genealogy.looks_like_gedcom(data[:800]):
        try:
            tree, text = genealogy.read_tree(data, name)
        except ValueError as e:
            raise Unreadable(str(e))
        except Exception:
            raise Unreadable("I could not read that family tree. Export it as a GEDCOM (.ged) file from your genealogy program and add that instead.")
        if tree:
            return Extracted(text=text, kind="family_tree", tree=tree, notes=[f"Family tree: {tree.summary()}."])
        return Extracted(text=text, kind="database",
                         notes=["I read this as a raw database, so I can show what is in it but not the family links. For the full tree, export a GEDCOM (.ged) file from the program."])
    # pictures and audio
    if ext in IMAGE_EXT or data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
        return Extracted(kind="image", images=[_image(data)])
    if ext in AUDIO_EXT:
        return Extracted(kind="audio", audio=data, audio_name=name)
    # documents
    if ext == ".pdf" or data[:5] == b"%PDF-":
        return _pdf(data)
    if ext == ".docx":
        return Extracted(text=_docx(data), kind="document")
    if ext in (".odt", ".odp", ".ods") and zipfile.is_zipfile(io.BytesIO(data)):
        return Extracted(text=_odt(data), kind="document")
    if ext == ".pptx":
        return Extracted(text=_pptx(data), kind="document")
    if ext in (".xlsx", ".xlsm"):
        return Extracted(text=_sheet(data), kind="document")
    if ext == ".epub":
        text, title = _epub(data)
        return Extracted(text=text, kind="document", title=title)
    if ext in (".doc", ".ppt", ".xls", ".wpd", ".wps", ".pages", ".numbers", ".key"):
        return Extracted(text=_office_via_libreoffice(data, ext), kind="document")
    if ext == ".zip" or (data[:2] == b"PK" and not ext):
        return _zip(data, depth)
    # plain text and friends
    text = _printable_text(data)
    if text is None:
        raise Unreadable(f"I do not know how to read {ext or 'that kind of'} file yet. If you tell the administrator, an addon can be added for it.")
    if ext in (".html", ".htm", ".xhtml") or re.match(r"\s*<(!doctype html|html)", text[:200], re.I):
        return Extracted(text=_html_text(text))
    if ext == ".rtf" or text.startswith("{\\rtf"):
        return Extracted(text=_rtf_text(text), kind="document")
    if ext == ".csv":
        return Extracted(text=_delimited(text, ","))
    if ext == ".tsv":
        return Extracted(text=_delimited(text, "\t"))
    if ext == ".json":
        try:
            return Extracted(text=json.dumps(json.loads(text), indent=1, ensure_ascii=False))
        except ValueError:
            pass
    if ext in (".xml", ".opml"):
        return Extracted(text=_tidy(re.sub(r"<[^>]+>", " ", text)))
    return Extracted(text=_tidy(text))


def _zip(data: bytes, depth: int) -> Extracted:
    if depth >= 1:
        raise Unreadable("That zip file has another zip file inside it. Unzip it first.")
    z = zipfile.ZipFile(io.BytesIO(data))
    parts, notes, images = [], [], []
    for info in z.infolist()[:60]:
        if info.is_dir() or info.file_size > MAX_BYTES or Path(info.filename).name.startswith((".", "__")):
            continue
        try:
            with z.open(info) as fh:
                blob = fh.read(MAX_BYTES + 1)  # the size in the header can lie; never read more than the limit
            sub = extract(blob, info.filename, depth + 1)
        except Unreadable:
            notes.append(f"Skipped {Path(info.filename).name}: could not read it.")
            continue
        parts.append(f"=== {Path(info.filename).name} ===\n{sub.text}")
        images += sub.images
    if not parts:
        raise Unreadable("I could not read anything inside that zip file.")
    return Extracted(text="\n\n".join(parts), kind="document", images=images[:MAX_IMAGES], notes=notes[:5])
