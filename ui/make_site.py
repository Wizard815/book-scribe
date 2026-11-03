"""Build the website's main page.

    python ui/make_site.py

ui/index.html = the shared workspace (simple.fragment.html, which is also published as the Claude artifact)
              + the site-only parts (site.extra.html: writer login and Settings).
"""
from pathlib import Path

here = Path(__file__).parent
fragment = (here / "simple.fragment.html").read_text()
extra = (here / "site.extra.html").read_text()
page = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{color-scheme:light}[hidden]{display:none!important}body{margin:0}</style>\n</head>\n<body>\n"
    + fragment + "\n" + extra + "\n</body>\n</html>\n"
)
(here / "index.html").write_text(page)
print("wrote", here / "index.html", len(page), "bytes")
