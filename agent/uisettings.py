"""How the screen looks, and understanding the writer when they say they can't see it.

The look is a small record kept on the server (so it follows the writer to any device) and applied by the page:

    zoom      100..500  percent: the words on the book's pages (100 normal, up to 500: bigger words, so more pages)
    size      0..7   word size steps (18, 22, 26, 32, 40, 52, 64, 80 px); 2 is the everyday default
    contrast  normal | high (black on white, strong lines) | dark (yellow on black)
    bold      heavier letters
    spacing   0..2   more room between lines and letters
    simple    hide the toolbar, one thing at a time
    rate      0.6..1.5  how fast replies are read aloud
    theme     auto | light | dark
    layout    "" | side | stack | chat | book

`parse()` turns what a person says into a change of look *without a model call*, so "I can't see it" takes effect immediately. The same
function backs the model's `set_ui` tool. Everything is validated here; nothing the model or a request sends can set an unknown value.
"""
from __future__ import annotations

import re

from intent import norm

DEFAULT = {"zoom": 100, "size": 2, "contrast": "normal", "bold": False, "spacing": 0, "simple": False, "rate": 1.0, "theme": "auto", "layout": ""}
SIZE_MAX = 7
ZOOMS = (100, 125, 150, 200, 250, 300, 400, 500)  # the book page's own zoom, in percent: 100 is normal, 500 is five times the words and five times the pages
CONTRASTS = ("normal", "high", "dark")
THEMES = ("auto", "light", "dark")
LAYOUTS = ("", "side", "stack", "chat", "book")


def nearest_zoom(v: float) -> int:
    return min(ZOOMS, key=lambda z: abs(z - v))


def step_zoom(cur: int, steps: int) -> int:
    i = ZOOMS.index(nearest_zoom(cur)) + steps
    return ZOOMS[max(0, min(len(ZOOMS) - 1, i))]


def clean(look: dict | None) -> dict:
    """Any dict in, a valid complete look out (unknown keys and bad values are dropped)."""
    d, out = look or {}, dict(DEFAULT)
    if isinstance(d.get("size"), (int, float)) and not isinstance(d.get("size"), bool):
        out["size"] = max(0, min(SIZE_MAX, int(d["size"])))
    if isinstance(d.get("zoom"), (int, float)) and not isinstance(d.get("zoom"), bool):
        out["zoom"] = nearest_zoom(d["zoom"])
    if d.get("contrast") in CONTRASTS:
        out["contrast"] = d["contrast"]
    if d.get("theme") in THEMES:
        out["theme"] = d["theme"]
    if d.get("layout") in LAYOUTS:
        out["layout"] = d["layout"]
    for k in ("bold", "simple"):
        if isinstance(d.get(k), bool):
            out[k] = d[k]
    if isinstance(d.get("spacing"), (int, float)) and not isinstance(d.get("spacing"), bool):
        out["spacing"] = max(0, min(2, int(d["spacing"])))
    if isinstance(d.get("rate"), (int, float)) and not isinstance(d.get("rate"), bool):
        out["rate"] = round(max(0.6, min(1.5, float(d["rate"]))), 2)
    return out


_UI_NOUN = r"(?:text|words?|letters|print|font|writing|type|screen|display|buttons?|everything|page|it|them|these|things|lines?|colou?rs?)"
_BIGGER = re.compile(r"\b(?:bigger|larger|enlarge|increase|big(?:ger)? words|zoom in|make (?:it|them|everything) big)\b")
_SMALLER = re.compile(r"\b(?:smaller|shrink|zoom out|too (?:big|large|huge)|not so big)\b")
_MUCH = re.compile(r"\b(?:much|way|a lot|lot|really|very|so much|huge|hugely|far|a great deal)\b")
_BIGGEST = re.compile(r"\b(?:biggest|largest|maximum|as big as (?:it )?(?:goes|gets|can)|as large as)\b")
_NORMAL_SIZE = re.compile(r"\b(?:normal|regular|default|usual|original) (?:size|sized|words|text|print)\b")
_CANT_SEE = re.compile(
    # "can't see" counts only when it ends the sentence or is followed by something on the screen or a plain "well"/"at all",
    # so "I can't see her doing that" (the story) is left alone.
    r"\b(?:i )?(?:can ?not|cant|can't|couldn't|cannot|do not|dont|don't) (?:really |quite )?(?:see|read)"
    r"(?=\s*$| (?:it|this|that|these|those|them|anything|well|properly|clearly|much|a thing|at all|very|enough|the (?:words|text|screen|letters|print|page|buttons|book))\b)"
    r"|\bhard to (?:see|read)\b|\btoo (?:small|tiny|faint|dim|light|busy|cluttered)\b|\b(?:my )?eyes? (?:are|is) (?:tired|bad|hurting|sore|strained|watering)\b|\bblurry\b|\bcan't make (?:it|that|this|anything) out\b")
_CANT_SEE_NOT_UI = re.compile(r"\b(?:see|read) (?:where|how|why|what|whether|if|the point|the ending|the plot|the story|the connection|a way|a future|your)\b")
_HIGH = re.compile(r"\b(?:high contrast|more contrast|stronger contrast|black (?:text )?on white|white background|easier to (?:see|read)|clearer|stronger colou?rs?|darker text)\b")
_DARK_HC = re.compile(r"\b(?:white on black|yellow on black|yellow text|black background|night mode|dark high contrast)\b")
_NORMAL_CONTRAST = re.compile(r"\b(?:normal|regular|usual|ordinary) (?:colou?rs?|contrast)\b")
_DARK_MODE = re.compile(r"\b(?:dark mode|dark theme|darker screen|too bright|the glare|glare)\b")
_LIGHT_MODE = re.compile(r"\b(?:light mode|light theme|bright mode|lighter screen)\b")
_BOLD = re.compile(r"\b(?:bold|bolder|thicker|heavier|thick letters|fatter)\b")
_NOT_BOLD = re.compile(r"\b(?:not bold|no bold|thinner|lighter letters|normal weight|stop the bold)\b")
_SPACE_MORE = re.compile(r"\b(?:more (?:space|spacing|room)|space (?:it |them )?out|spread (?:it |them )?out|lines? (?:are )?too close|cramped|too crowded)\b")
_SPACE_LESS = re.compile(r"\b(?:less (?:space|spacing)|tighter|closer together|too (?:spread|spaced))\b")
_SIMPLE = re.compile(r"\b(?:simple(?:r)? (?:screen|mode|page|view)|too many buttons|too busy|too cluttered|hide the buttons|keep it simple|just the chat|only the chat|simplify (?:the |this )?(?:screen|page|view|display|app))\b")
_NOT_SIMPLE = re.compile(r"\b(?:show everything|all the buttons|full screen|normal screen|not simple|bring back the buttons|show the buttons)\b")
_SLOWER = re.compile(r"\b(?:slower|slow down|too fast|more slowly)\b")
_FASTER = re.compile(r"\b(?:faster|speed up|too slow|more quickly)\b")
_SPEECH = re.compile(r"\b(?:talk|talking|speak|speaking|read|reading|speech|voice|you|your|speed)\b")
_RESET = re.compile(r"\b(?:back to normal|reset (?:the )?(?:screen|look|display|size|everything)|default (?:screen|look|settings|display)|undo (?:the )?(?:screen|size|look|changes to the screen))\b")
_LAYOUT = [
    (re.compile(r"\b(?:only|just) (?:the )?chat\b|\bchat only\b|\bhide the book\b"), "chat"),
    (re.compile(r"\b(?:only|just) (?:the )?book\b|\bbook only\b|\bhide the chat\b"), "book"),
    (re.compile(r"\bside by side\b"), "side"),
    (re.compile(r"\btop and bottom\b|\bone above the other\b"), "stack"),
]


_BOOKISH = re.compile(r"\b(?:book|page|pages|story|chapter|chapters|reading|manuscript)\b")
_ZOOM_WORDS = re.compile(r"\b(?:zoom|text|words?|font|print|letters?|type|size|bigger|smaller|larger|magnif\w*)\b")
_ZOOM_PCT = re.compile(r"\b(\d{2,3})\s*(?:%|percent|per cent)")
_ZOOM_RESET = re.compile(r"\bzoom (?:back|reset|normal)\b|\b(?:normal|regular|usual|default) (?:zoom|page size|page zoom)\b|\breset (?:the )?zoom\b")
_NOT_ZOOM_PCT = re.compile(r"\b(?:sure|certain|agree|right|confident|done|finished)\b")


def _story_size_talk(n: str):
    """"make it a bigger ship": the writer is talking about the story, not asking for bigger words."""
    return re.search(r"\b(?:a|an|the|some|one|that|this)\s+(?:much |a lot |far )?(?:bigger|smaller|larger)\s+(?!(?:please|now|again|than|for|so|still|thanks|thank|if)\b)\w+", n)


def _book_zoom(n: str, cur: int, raw: str = "") -> tuple[int, str] | None:
    """The book page's zoom: "make the book text bigger", "zoom in", "200 percent", "zoom back to normal". None if this is not about it."""
    if _ZOOM_RESET.search(n):
        return 100, "the book is back to 100 percent"
    m = _ZOOM_PCT.search(raw.lower() or n)
    if m and (_ZOOM_WORDS.search(n) or _BOOKISH.search(n)) and not _NOT_ZOOM_PCT.search(n) and 25 <= int(m.group(1)) <= 900:
        z = nearest_zoom(int(m.group(1)))
        return z, f"the book is at {z} percent"
    whole = re.search(r"\b(?:screen|everything|buttons?|chat|whole|display|menu|controls?|the app)\b", n)
    picture = re.search(r"\b(?:picture|pictures|photo|photos|photograph|image|images|map|illustration|drawing)\b", n)
    story_talk = _story_size_talk(n)
    plain = (_BIGGER.search(n) or _SMALLER.search(n)) and not story_talk and (re.search(rf"\b{_UI_NOUN}\b", n) or len(n.split()) <= 3) and not whole and not picture and not _CANT_SEE.search(n)  # "make the words bigger": the page is what is being read
    zoomish = plain or re.search(r"\bzoom(?: in| out)?\b", n) and not re.search(r"\bzoom (?:call|meeting|link)\b", n)
    bookish = _BOOKISH.search(n) and _ZOOM_WORDS.search(n)
    if not (zoomish or bookish):
        return None
    out = re.search(r"\bzoom out\b", n) or _SMALLER.search(n)
    up = re.search(r"\bzoom in\b", n) or _BIGGER.search(n) or re.search(r"\bzoom\b", n) and not out
    if _BIGGEST.search(n) and not out:
        z = ZOOMS[-1]
    elif up and not out:
        z = step_zoom(cur, 2 if _MUCH.search(n) else 1)
    elif out and not (up and not _SMALLER.search(n)):
        z = step_zoom(cur, -2 if _MUCH.search(n) else -1)
    else:
        return None
    return z, f"the book is at {z} percent"


def _has_ui_target(n: str, words: int) -> bool:
    return bool(re.search(rf"\b{_UI_NOUN}\b", n)) or words <= 2


def parse(text: str, look: dict | None) -> tuple[dict, str] | None:
    """-> (new look, what to say) when `text` is a request about the screen, else None."""
    n = norm(text)
    if not n:
        return None
    cur, new, said = clean(look), clean(look), []
    words = len(n.split())
    screen_words = re.search(r"\b(?:words|letters|text|font|print|writing|type)\b", n) and re.search(r"\b(?:bigger|larger|smaller|tinier|cant see|can't see|cannot see|hard to (?:see|read))\b", n) \
        and re.match(r"^(?:(?:please|can you|could you|would you|so|and|now|oh)\s+)*(?:make|turn|put|set|i cant|i can't|i cannot|i can not)\b", n)  # a request, not "the letters were bigger than his hands"
    if not screen_words and re.search(r"\bparagraphs? \d+\b|\b(?:he|she|they|him|them|his|her|their)\b|\bchapter \d+\b|\b(?:the|a|an) \w+ (?:is|was|looked|seemed|felt|were)\b", n) and not re.search(r"\b(?:screen|buttons?|display|everything)\b", n):
        return None  # about a paragraph or a person in the story, not the screen: "simplify paragraph 2", "make him speak slower", "he was bold"

    def size(delta: int | None = None, to: int | None = None):
        new["size"] = max(0, min(SIZE_MAX, (to if to is not None else new["size"] + delta)))

    bz = _book_zoom(n, cur["zoom"], text)
    if bz is not None:
        if bz[0] == cur["zoom"]:
            return cur, f"The book is already at {cur['zoom']} percent, as far as it goes." if bz[0] in (ZOOMS[0], ZOOMS[-1]) and not _ZOOM_PCT.search(text.lower()) else f"The book is already at {cur['zoom']} percent."
        new["zoom"] = bz[0]
        return new, f"Done: {bz[1]}. Pages are {'longer' if bz[0] < cur['zoom'] else 'shorter'} and there are {'fewer' if bz[0] < cur['zoom'] else 'more'} of them."
    if _RESET.search(n) or (n in ("back to normal", "normal", "go back to normal")):
        return clean(None), "Back to normal."
    if _CANT_SEE.search(n) and not _CANT_SEE_NOT_UI.search(n) and not _BIGGER.search(n) and not _SMALLER.search(n):
        size(+2); new["contrast"] = "high" if cur["contrast"] == "normal" else cur["contrast"]; new["bold"] = True; new["spacing"] = max(cur["spacing"], 1)
        return new, "I made the words bigger and bolder, and the colours stronger. Say bigger if you need more, or back to normal."
    if _story_size_talk(n):
        pass
    elif _BIGGEST.search(n) and _has_ui_target(n, words):
        size(to=SIZE_MAX); said.append("as big as they go")
    elif _NORMAL_SIZE.search(n):
        size(to=DEFAULT["size"]); said.append("back to the usual size")
    elif _BIGGER.search(n) and _has_ui_target(n, words) and not _SMALLER.search(n):
        step = 3 if _MUCH.search(n) else 1
        size(+step); said.append("everything on the screen is much bigger" if step > 1 else "everything on the screen is bigger")
    elif _SMALLER.search(n) and _has_ui_target(n, words):
        size(-(2 if _MUCH.search(n) else 1)); said.append("everything on the screen is smaller")
    if _NORMAL_CONTRAST.search(n):
        new["contrast"] = "normal"; said.append("normal colours")
    elif _DARK_HC.search(n):
        new["contrast"] = "dark"; said.append("yellow on black")
    elif _HIGH.search(n):
        new["contrast"] = "high"; said.append("stronger colours")
    elif _DARK_MODE.search(n):
        new["theme"] = "dark"; said.append("a dark screen")
    elif _LIGHT_MODE.search(n):
        new["theme"] = "light"; said.append("a light screen")
    if _NOT_BOLD.search(n):
        new["bold"] = False; said.append("lighter letters")
    elif _BOLD.search(n) and _has_ui_target(n, words):
        new["bold"] = True; said.append("bolder letters")
    if _SPACE_MORE.search(n):
        new["spacing"] = min(2, cur["spacing"] + 1); said.append("more space between lines")
    elif _SPACE_LESS.search(n):
        new["spacing"] = max(0, cur["spacing"] - 1); said.append("less space between lines")
    if _NOT_SIMPLE.search(n):
        new["simple"] = False; new["layout"] = ""; said.append("everything showing")
    elif _SIMPLE.search(n):
        new["simple"] = True; said.append("a simpler screen")
    for rx, lay in _LAYOUT:
        if rx.search(n):
            new["layout"] = lay; said.append({"chat": "just the chat", "book": "just the book", "side": "side by side", "stack": "one above the other"}[lay]); break
    if _SPEECH.search(n) and (_SLOWER.search(n) or _FASTER.search(n)):
        new["rate"] = round(max(0.6, min(1.5, cur["rate"] + (-0.15 if _SLOWER.search(n) else 0.15))), 2); said.append("slower speech" if _SLOWER.search(n) else "faster speech")
    if not said or new == cur and not any(s in ("as big as they go",) for s in said):
        if not said:
            return None
        return cur, "That is already as far as it goes."
    return new, "Done: " + ", ".join(said) + "."
