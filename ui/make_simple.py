"""Build ui/simple.html (the standalone page for your own website) from ui/simple.fragment.html.

The fragment is also exactly what gets published as the Claude artifact, so there is one source for both:

    python ui/make_simple.py
"""
from pathlib import Path

here = Path(__file__).parent
fragment = (here / "simple.fragment.html").read_text()
page = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{color-scheme:light}[hidden]{display:none!important}body{margin:0}</style>\n</head>\n<body>\n"
    + fragment + "\n</body>\n</html>\n"
)
(here / "simple.html").write_text(page)
print("wrote", here / "simple.html", len(page), "bytes")
