"""Looking things up: Wikipedia (or an offline mirror such as Kiwix) and a SearXNG search server.

Both are OFF unless the administrator sets a URL (WIKIPEDIA_URL, SEARXNG_URL in .env), because a lookup sends words to another server. Only the
short query the model chose leaves the machine, never the book. The writer can also switch lookups off in Settings.

    wikipedia(query)  -> the best matching article's text (plain, first few thousand characters) and its address
    search(query)     -> a few results (title, address, snippet) from SearXNG, which itself asks many search engines
    read(url)         -> the readable text of one page

Everything that comes back is UNTRUSTED TEXT: the model is told it is data to read, never instructions to follow, and what it writes into the
book or wiki still needs the writer's yes.

Fetching an arbitrary page is the risky part (a result could point at a machine on your own network). So read() only fetches public internet
addresses over http/https on the ordinary ports, checks every redirect the same way, caps the size, and never sends cookies. The two servers the
administrator named are the only exception.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

UA = "BookScribe/1.0 (writing assistant; contact: the administrator of this server)"
MAX_BYTES = 2_000_000


class WebError(Exception):
    """A lookup that could not be done. The message is fit to show the writer."""


def _is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    ips = {i[4][0] for i in infos}
    return bool(ips) and all(ipaddress.ip_address(ip.split("%")[0]).is_global for ip in ips)


def _text_of(html: str, chars: int) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", main.get_text("\n"))).strip()
    return text[:chars]


class Web:
    def __init__(self, searxng_url: str = "", wikipedia_url: str = "", http: httpx.AsyncClient | None = None, timeout: float = 15.0):
        self.searxng = searxng_url.rstrip("/")
        self.wikipedia_url = wikipedia_url.rstrip("/")
        self.http = http or httpx.AsyncClient(headers={"User-Agent": UA}, trust_env=True)
        self.timeout = timeout
        self._trusted = {urlparse(u).hostname for u in (self.searxng, self.wikipedia_url) if u}

    @property
    def can_search(self) -> bool:
        return bool(self.searxng)

    @property
    def can_wikipedia(self) -> bool:
        return bool(self.wikipedia_url)

    @property
    def enabled(self) -> bool:
        return self.can_search or self.can_wikipedia

    # --- the two servers the administrator chose ---------------------------------------------------------------------
    async def _json(self, url: str, params: dict) -> object:
        try:
            r = await self.http.get(url, params=params, timeout=self.timeout, headers={"User-Agent": UA, "Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise WebError(f"The lookup server did not answer ({e.__class__.__name__}).")

    async def search(self, query: str, n: int = 5) -> list[dict]:
        if not self.can_search:
            raise WebError("Web search is not set up.")
        data = await self._json(f"{self.searxng}/search", {"q": query[:200], "format": "json", "safesearch": 1})
        out = []
        for r in (data.get("results") if isinstance(data, dict) else None) or []:
            if isinstance(r, dict) and r.get("url") and r.get("title"):
                out.append({"title": str(r["title"])[:140], "url": str(r["url"])[:400], "snippet": re.sub(r"\s+", " ", str(r.get("content") or ""))[:300]})
            if len(out) >= max(1, min(n, 8)):
                break
        return out

    async def wikipedia(self, query: str, chars: int = 3500) -> dict | None:
        if not self.can_wikipedia:
            raise WebError("Wikipedia lookup is not set up.")
        api = f"{self.wikipedia_url}/w/api.php"
        found = await self._json(api, {"action": "opensearch", "search": query[:120], "limit": 1, "namespace": 0, "format": "json", "redirects": "resolve"})
        titles = found[1] if isinstance(found, list) and len(found) > 1 else []
        if not titles:
            return None
        data = await self._json(api, {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain", "titles": titles[0], "redirects": 1, "format": "json"})
        pages = (data.get("query") or {}).get("pages") or {} if isinstance(data, dict) else {}
        page = next(iter(pages.values()), {})
        text = re.sub(r"\n{3,}", "\n\n", str(page.get("extract") or "")).strip()
        if not text:
            return None
        title = str(page.get("title") or titles[0])
        return {"title": title, "url": f"{self.wikipedia_url}/wiki/{title.replace(' ', '_')}", "text": text[:chars], "more": len(text) > chars}

    async def image_search(self, query: str, n: int = 6) -> list[dict]:
        """Picture results from SearXNG: [{title, img (the picture's address), page (where it was found)}]."""
        if not self.can_search:
            raise WebError("Web search is not set up.")
        data = await self._json(f"{self.searxng}/search", {"q": query[:200], "format": "json", "categories": "images", "safesearch": 1})
        out = []
        for r in (data.get("results") if isinstance(data, dict) else None) or []:
            img = r.get("img_src") if isinstance(r, dict) else None
            if img and str(img).startswith(("http://", "https://")):
                out.append({"title": re.sub(r"\s+", " ", str(r.get("title") or "picture"))[:120], "img": str(img)[:500], "page": str(r.get("url") or "")[:500]})
            if len(out) >= max(1, min(n, 10)):
                break
        return out

    async def fetch_image(self, url: str, max_bytes: int = 12_000_000) -> bytes:
        """The bytes of one picture, from a public address only (every redirect is checked). The caller still validates it as a picture."""
        for _ in range(4):
            u = urlparse(url)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise WebError("That is not a web address I can open.")
            if u.port not in (None, 80, 443) and u.hostname not in self._trusted:
                raise WebError("I only open ordinary web pages.")
            if u.hostname not in self._trusted and not await asyncio.to_thread(_is_public, u.hostname):
                raise WebError("I will not open that address: it is not on the public internet.")
            try:
                async with self.http.stream("GET", url, timeout=self.timeout, follow_redirects=False, headers={"User-Agent": UA}) as r:
                    if 300 <= r.status_code < 400 and r.headers.get("location"):
                        url = urljoin(url, r.headers["location"])
                        continue
                    r.raise_for_status()
                    if not r.headers.get("content-type", "").lower().startswith("image/"):
                        raise WebError("That address is not a picture.")
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > max_bytes:
                            raise WebError("That picture is too big.")
                    return body
            except httpx.HTTPError as e:
                raise WebError(f"I could not fetch that picture ({e.__class__.__name__}).")
        raise WebError("That address keeps redirecting.")

    # --- any other page: public addresses only ----------------------------------------------------------------------------
    async def read(self, url: str, chars: int = 6000) -> str:
        for _ in range(4):  # follow at most three redirects, checking each one
            u = urlparse(url)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise WebError("That is not a web address I can open.")
            if u.port not in (None, 80, 443):
                raise WebError("I only open ordinary web pages.")
            if u.hostname not in self._trusted and not await asyncio.to_thread(_is_public, u.hostname):
                raise WebError("I will not open that address: it is not on the public internet.")
            try:
                async with self.http.stream("GET", url, timeout=self.timeout, follow_redirects=False, headers={"User-Agent": UA}, cookies=None) as r:
                    if 300 <= r.status_code < 400 and r.headers.get("location"):
                        url = urljoin(url, r.headers["location"])
                        continue
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype not in ("text/html", "application/xhtml+xml", "text/plain", ""):
                        raise WebError("That page is not text I can read (it may be a PDF or a picture).")
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            break
            except httpx.HTTPError as e:
                raise WebError(f"I could not open that page ({e.__class__.__name__}).")
            html = body.decode(r.encoding or "utf-8", errors="replace")
            return _text_of(html, chars) if ctype != "text/plain" else html[:chars]
        raise WebError("That page keeps redirecting.")
