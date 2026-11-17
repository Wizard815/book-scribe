"""Getting the finished book out: one self-contained HTML file (pictures embedded, opens anywhere, prints to PDF), an EPUB for e-readers,
or a Markdown folder (zip) with the pictures beside it.

Chapter titles, the book's title (a "Title:" line in the premise, else the book's name) and every picture with its caption are included.
"""
from __future__ import annotations

import base64
import html
import io
import re
import zipfile
from pathlib import Path

import images as imgs


def book_title(store, book: str) -> str:
    """The title the writer gave, found in the premise ("Title: ...", "The book is called ...", "The title is ..."), else the book's name."""
    prem = store.bible(book).read("premise")
    word = r"(?:[\"“]([^\"”\n]{2,100})[\"”]|([^\n]+?)(?:\.\s|\.?$))"  # a quoted title, or the words up to the end of the sentence
    for rx in (rf"^\s*(?:\*\*)?Title(?:\*\*)?\s*[:\-]\s*{word}", rf"\b(?:called|titled|entitled|named)\s+{word}", rf"\btitle (?:is|will be|should be)\s+{word}"):
        m = re.search(rx, prem, re.M | re.I)
        if m:
            t = (m.group(1) or m.group(2) or "").strip().strip("\"'“”")
            if 2 <= len(t) <= 100:
                return t
    return book.replace("-", " ").replace("_", " ").title()


def _chapters(store, book: str):
    for n in store.chapter_numbers(book):
        yield n, store.chapter_title(book, n), store.paragraphs(book, n)


def _mime(name: str) -> str:
    return imgs.TYPES.get(Path(name).suffix.lower(), "image/jpeg")


def to_html(store, istore, book: str) -> str:
    title = book_title(store, book)
    toc, body = [], []
    for n, ct, paras in _chapters(store, book):
        head = f"Chapter {n}" + (f": {ct}" if ct else "")
        toc.append(f'<li><a href="#c{n}">{html.escape(head)}</a></li>')
        parts = [f'<section id="c{n}"><h2>{html.escape(head)}</h2>']
        for p in paras:
            img = imgs.parse(p.text)
            if img:
                try:
                    data = istore.path(book, img[1]).read_bytes()
                    src = f"data:{_mime(img[1])};base64,{base64.b64encode(data).decode()}"
                    parts.append(f'<figure><img src="{src}" alt="{html.escape(img[0])}"><figcaption>{html.escape(img[0])}</figcaption></figure>')
                except FileNotFoundError:
                    parts.append(f'<p class="missing">[Picture missing: {html.escape(img[0])}]</p>')
            else:
                parts.append(f"<p>{html.escape(p.text)}</p>")
        parts.append("</section>")
        body.append("\n".join(parts))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 1.25rem; line-height: 1.7; max-width: 40rem; margin: 0 auto; padding: 2rem 1.2rem 5rem; color: #1b1b1b; background: #fbfaf7; }}
h1 {{ font-size: 2.4rem; text-align: center; margin: 3rem 0 .4rem; }} h2 {{ font-size: 1.7rem; margin: 3.5rem 0 1.2rem; page-break-before: always; }}
nav {{ margin: 2rem 0; }} nav li {{ margin: .3rem 0; }} p {{ margin: 0 0 1.1rem; text-indent: 1.4em; }} section p:first-of-type {{ text-indent: 0; }}
figure {{ margin: 2rem 0; text-align: center; }} figure img {{ max-width: 100%; height: auto; border-radius: 6px; }} figcaption {{ font-size: .95rem; color: #555; margin-top: .5rem; font-style: italic; }}
@media print {{ body {{ background: #fff; }} nav {{ display: none; }} }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<nav aria-label="Contents"><ol>{"".join(toc)}</ol></nav>
{"".join(body)}
</body></html>"""


def to_markdown_zip(store, istore, book: str) -> bytes:
    title = book_title(store, book)
    md = [f"# {title}\n"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, ct, paras in _chapters(store, book):
            md.append(f"\n## Chapter {n}" + (f": {ct}" if ct else "") + "\n")
            for p in paras:
                img = imgs.parse(p.text)
                if img:
                    md.append(f"![{img[0]}](images/{img[1]})\n")
                    try:
                        z.write(istore.path(book, img[1]), f"images/{img[1]}")
                    except FileNotFoundError:
                        pass
                else:
                    md.append(p.text + "\n")
        z.writestr("book.md", "\n".join(md))
    return buf.getvalue()


def to_epub(store, istore, book: str) -> bytes:
    title = book_title(store, book)
    esc = html.escape
    chapters, manifest, spine, nav, files = [], [], [], [], {}
    for n, ct, paras in _chapters(store, book):
        head = f"Chapter {n}" + (f": {ct}" if ct else "")
        parts = [f"<h2>{esc(head)}</h2>"]
        for p in paras:
            img = imgs.parse(p.text)
            if img:
                try:
                    files[img[1]] = istore.path(book, img[1]).read_bytes()
                    parts.append(f'<figure><img src="images/{esc(img[1])}" alt="{esc(img[0])}"/><figcaption>{esc(img[0])}</figcaption></figure>')
                except FileNotFoundError:
                    pass
            else:
                parts.append(f"<p>{esc(p.text)}</p>")
        doc = (f'<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>{esc(head)}</title>'
               f'<style>p{{text-indent:1.2em;margin:0 0 .8em}}figure{{text-align:center;margin:1.2em 0}}img{{max-width:100%}}figcaption{{font-style:italic;font-size:.9em}}</style></head><body>{"".join(parts)}</body></html>')
        chapters.append((f"c{n}.xhtml", doc))
        manifest.append(f'<item id="c{n}" href="c{n}.xhtml" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{n}"/>')
        nav.append(f'<li><a href="c{n}.xhtml">{esc(head)}</a></li>')
    for i, name in enumerate(files):
        manifest.append(f'<item id="img{i}" href="images/{esc(name)}" media-type="{_mime(name)}"/>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("nav.xhtml", f'<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head><body><nav epub:type="toc"><h1>{esc(title)}</h1><ol>{"".join(nav)}</ol></nav></body></html>')
        z.writestr("content.opf", f'<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">book-scribe-{esc(book)}</dc:identifier><dc:title>{esc(title)}</dc:title><dc:language>en</dc:language></metadata><manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>{"".join(manifest)}</manifest><spine>{"".join(spine)}</spine></package>')
        for name, doc in chapters:
            z.writestr(name, doc)
        for name, data in files.items():
            z.writestr(f"images/{name}", data)
    return buf.getvalue()
