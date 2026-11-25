"""The writer's hands on the app. Usage:
  python bs.py say "what you say out loud"      talk to Book Scribe (as if spoken; the chapter on screen is sent along)
  python bs.py view N                           turn the book on screen to chapter N
  python bs.py upload FILE [FILE...]            add family files (PDF, GEDCOM, photos) to the library
  python bs.py book                             the whole book as saved (paragraph numbers as a writer counts them; pictures marked)
  python bs.py shot [ZOOM]                      a 1920x1080 screenshot of the app (book zoom 100..500), saved in ../shots/
  python bs.py library                          what is in the library
"""
import json, os, sys, time
import httpx

RIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE, BOOK = "http://127.0.0.1:8077", "samuel"
ST = os.path.join(RIG, "writer_state.json")


def state():
    try:
        return json.load(open(ST))
    except Exception:
        return {}


def save(s):
    json.dump(s, open(ST, "w"))


def client():
    c = httpx.Client(base_url=BASE, timeout=1600)
    s = state()
    if s.get("cookie"):
        c.cookies.set("bs_session", s["cookie"])
    if c.get("/api/auth/me").status_code != 200:
        r = c.post("/api/auth/login", json={"username": "walter", "password": "licking county 1862"})
        r.raise_for_status()
        s["cookie"] = r.cookies.get("bs_session") or c.cookies.get("bs_session")
        save(s)
    return c


def say(text):
    s, c = state(), client()
    body = {"text": text, "book": BOOK, "session_id": s.get("sid")}
    if s.get("viewing"):
        body["viewing"] = s["viewing"]
    t0 = time.time()
    r = c.post("/api/chat", json=body)
    if r.status_code != 200:
        print("ERROR", r.status_code, r.text[:400]); return
    d = r.json()
    s["sid"] = d["session_id"]
    s["viewing"] = d.get("chapter") or s.get("viewing")
    save(s)
    print(f"BOOK SCRIBE ({time.time() - t0:.0f}s): {d['text']}")
    if d.get("pending"):
        print(f"  [card on screen: {d['pending'].get('ask') or d['pending'].get('kind')}]")
    if d.get("picture"):
        print(f"  [picture shown: {d['picture'].get('caption', '')}]")
    for n in d.get("notes") or []:
        print(f"  [note: {n}]")
    if d.get("ui"):
        print(f"  [screen changed: {d['ui']}]")
    if d.get("ideas"):
        print("  [idea buttons: " + " | ".join(d["ideas"]) + "]")
    if d.get("download"):
        print(f"  [download offered: {d['download']}]")
    print(f"  [book now shows chapter {s['viewing']}]")


def book():
    c = client()
    nums = c.get(f"/api/books/{BOOK}/chapters").json().get("chapters", [])
    for n in nums:
        d = c.get(f"/api/books/{BOOK}/chapters/{n}").json()
        print(f"\n=== Chapter {n}{': ' + d['title'] if d.get('title') else ''}")
        k = 0
        for p in d["paragraphs"]:
            if p["text"].startswith("![") or p.get("image"):
                print(f"  [PICTURE] {p['text'][:160]}")
            else:
                k += 1
                print(f"  {k}. {p['text']}")


def upload(paths):
    c = client()
    files = [("files", (os.path.basename(p), open(p, "rb").read())) for p in paths]
    for it in c.post(f"/api/books/{BOOK}/library", files=files).json().get("items", []):
        print(it.get("message") or it)


def library():
    c = client()
    for it in c.get(f"/api/books/{BOOK}/library").json().get("items", []):
        print(f"- {it['title']} ({it['kind']}, {it['status']})")


def shot(zoom=None):
    import subprocess
    client(); s = state()
    out = os.path.join(RIG, "shots", f"shot_{int(time.time())}.png")
    subprocess.run(["node", os.path.join(RIG, "tools", "shot.js"), s["cookie"], str(zoom or "-"), out, str(s.get("viewing") or 1)], check=True, timeout=120)
    print("saved", out, "- look at it with your Read tool")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "say":
        say(" ".join(a[1:]))
    elif a[0] == "view":
        s = state(); s["viewing"] = int(a[1]); save(s); print("book shows chapter", a[1])
    elif a[0] == "upload":
        upload(a[1:])
    elif a[0] == "book":
        book()
    elif a[0] == "library":
        library()
    elif a[0] == "shot":
        shot(a[1] if len(a) > 1 else None)
    else:
        print(__doc__)
