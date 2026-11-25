"""OpenRouter calls with a hard spending cap for the test (budget.json in the rig folder)."""
import base64, json, os, threading, time
import httpx

RIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = os.path.join(RIG, "budget.json")
CAP = 2.00
_lock = threading.Lock()
URL = "https://openrouter.ai/api/v1/chat/completions"
HEAD = {"Authorization": "Bearer from-proxy", "HTTP-Referer": "https://localhost/book-scribe-test", "X-Title": "Book Scribe test"}


def spent() -> float:
    try:
        return json.load(open(BUDGET))["spent"]
    except Exception:
        return 0.0


def add(cost: float, what: str) -> float:
    with _lock:
        d = {"spent": 0.0, "calls": 0, "log": []}
        try:
            d = json.load(open(BUDGET))
        except Exception:
            pass
        d["spent"] = round(d["spent"] + cost, 6)
        d["calls"] = d.get("calls", 0) + 1
        d["log"] = (d.get("log", []) + [[int(time.time()), what, round(cost, 6)]])[-400:]
        json.dump(d, open(BUDGET, "w"))
        return d["spent"]


def chat(body: dict, what: str = "chat") -> dict:
    if spent() >= CAP:
        raise RuntimeError(f"budget of ${CAP} used up")
    body = dict(body)
    body["usage"] = {"include": True}
    r = httpx.post(URL, headers=HEAD, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:400]}")
    d = r.json()
    cost = float((d.get("usage") or {}).get("cost") or 0)
    add(cost, f"{what}:{body.get('model')}")
    return d


def image(prompt: str, model: str = "google/gemini-2.5-flash-image") -> bytes:
    d = chat({"model": model, "modalities": ["image", "text"],
              "messages": [{"role": "user", "content": f"Create one picture, no text in it: {prompt}"}]}, "image")
    msg = d["choices"][0]["message"]
    for im in msg.get("images") or []:
        url = im.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
    raise RuntimeError("no image in the answer: " + str(msg.get("content"))[:200])
