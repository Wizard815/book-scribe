"""Backends for the Civil War story test.
:8090  /v1/chat/completions -> OpenRouter (real model, capped spend; see orlib.py), every call logged to ../calls.jsonl
       /v1/images/generations -> OpenRouter image model
:80    mock Wikipedia (/w/api.php, /wiki/...) and SearXNG (/search) from ../wiki/*.txt; picture results are ../webimg/*.jpg
"""
import base64, json, os, re, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote
import orlib

RIG = orlib.RIG
WIKI, IMG = os.path.join(RIG, "wiki"), os.path.join(RIG, "webimg")
MODEL = os.environ.get("TEST_MODEL", "google/gemini-2.5-flash")
LOG = os.path.join(RIG, "calls.jsonl")
lock = threading.Lock()


def log(rec):
    with lock:
        open(LOG, "a").write(json.dumps(rec)[:20000] + "\n")


def articles():
    out = {}
    for f in sorted(os.listdir(WIKI)):
        if f.endswith(".txt"):
            title, _, body = open(os.path.join(WIKI, f), encoding="utf-8").read().partition("\n")
            out[title.strip()] = body.strip()
    return out


def words(q):
    return [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) > 2 and w not in ("the", "and", "war", "civil", "american", "history", "of")]


def find(q):
    ws, best, score = words(q), None, 0
    for t, b in articles().items():
        s = sum(3 for w in ws if w in t.lower()) + sum(1 for w in ws if w in b.lower())
        if s > score:
            best, score = t, s
    return best if score >= max(2, len(ws) // 2) else None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


class Model(H):
    def do_GET(self):
        self.send(200, {"data": [{"id": MODEL}]}) if self.path.startswith("/v1/models") else self.send(404, {})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t0 = time.time()
        try:
            if self.path.startswith("/v1/images/generations"):
                b = orlib.image(data.get("prompt", ""))
                log({"t": t0, "kind": "image", "prompt": data.get("prompt", "")[:300], "spent": orlib.spent()})
                return self.send(200, {"data": [{"b64_json": base64.b64encode(b).decode()}]})
            body = {"model": MODEL, "messages": data.get("messages", []), "max_tokens": min(int(data.get("max_tokens") or 1200), 4000),
                    "temperature": data.get("temperature", 0.7)}
            if data.get("tools"):
                body["tools"], body["tool_choice"] = data["tools"], data.get("tool_choice", "auto")
            d = orlib.chat(body)
            m = d["choices"][0]["message"]
            text = m.get("content") or ""
            if m.get("tool_calls"):
                log({"t": t0, "kind": "chat", "secs": round(time.time() - t0, 1), "tool_calls": [tc.get("function") for tc in m["tool_calls"]], "answer": text[:1000], "spent": orlib.spent()})
                return self.send(200, {"id": "x", "object": "chat.completion", "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "tool_calls": m["tool_calls"]}, "finish_reason": "tool_calls"}]})
            msgs = data.get("messages", [])
            last = msgs[-1]["content"] if msgs else ""
            log({"t": t0, "kind": "chat", "secs": round(time.time() - t0, 1), "last_user": (last if isinstance(last, str) else "[parts]")[-1500:], "answer": text[:3000],
                 "tokens": d.get("usage", {}).get("prompt_tokens"), "spent": orlib.spent()})
            return self.send(200, {"id": "x", "object": "chat.completion", "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})
        except Exception as e:
            log({"t": t0, "kind": "error", "error": str(e)[:500]})
            return self.send(503, {"error": {"message": str(e)[:300]}})


class Web(H):
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        host = "http://localhost"
        a = articles()
        if u.path == "/w/api.php":
            if q.get("action") == "opensearch":
                t = find(q.get("search", ""))
                return self.send(200, [q.get("search", ""), [t] if t else [], [""] if t else [], [f"{host}/wiki/{quote(t)}"] if t else []])
            t = q.get("titles", "")
            return self.send(200, {"query": {"pages": {"1": {"title": t, "extract": a[t]}}}} if t in a else {"query": {"pages": {"-1": {"missing": ""}}}})
        if u.path.startswith("/wiki/"):
            t = unquote(u.path[6:]).replace("_", " ")
            if t not in a:
                return self.send(404, b"not found", "text/plain")
            return self.send(200, (f"<html><head><title>{t}</title></head><body><h1>{t}</h1>" + "".join(f"<p>{p}</p>" for p in a[t].split("\n\n")) + "</body></html>").encode(), "text/html; charset=utf-8")
        if u.path.startswith("/img/"):
            p = os.path.join(IMG, os.path.basename(unquote(u.path[5:])))
            return self.send(200, open(p, "rb").read(), "image/jpeg") if os.path.isfile(p) else self.send(404, b"", "text/plain")
        if u.path == "/search":
            ws = words(q.get("q", ""))
            if q.get("categories") == "images":
                imgs = sorted(os.listdir(IMG), key=lambda f: -sum(1 for w in ws if w[:5] in f))
                captions = {"union-camp.jpg": "Union army camp, 1863 (archive photograph)", "ringgold-gap.jpg": "Ringgold Gap, Georgia, 1864 (archive photograph)",
                            "gar-reunion.jpg": "G.A.R. veterans' reunion, Ohio, 1890s (archive photograph)"}
                return self.send(200, {"results": [{"title": captions.get(f, f), "img_src": f"{host}/img/{f}", "url": f"{host}/wiki/Grand_Army_of_the_Republic"} for f in imgs]})
            ranked = sorted(a, key=lambda t: -sum(1 for w in ws if w in (t + " " + a[t]).lower()))[:5]
            return self.send(200, {"results": [{"title": t, "url": f"{host}/wiki/{quote(t.replace(' ', '_'))}", "content": a[t][:280]} for t in ranked]})
        self.send(404, {})


if __name__ == "__main__":
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", 8090), Model).serve_forever, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 80), Web).serve_forever()
