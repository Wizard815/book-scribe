# Addons: teach Book Scribe your own file types and tools

An addon is a Python file you keep in a folder of your own. Set `ADDONS_DIR` in `.env` to that folder and restart the agent. The admin page
(Models, then Embeddings and memory) shows which addons loaded and which failed, with the error.

Addons run with the server's full rights, so only the administrator can add them (the writer cannot upload code, and `ADDONS_DIR` is
set outside the web). Only install addons you wrote or have read.

```python
# /addons/my_ftm.py
import extract

def read_ftm(data: bytes, filename: str):
    """Return an extract.Extracted, or None to let the built-in readers try."""
    if not data.startswith(b"SQLite format 3"):
        return None
    tree_text = my_own_code_to_describe_the_tree(data)       # your logic
    return extract.Extracted(text=tree_text, kind="document", title="My family tree")

async def find_relative(ctx, name=""):
    return "…"                                                # a read-only helper for the model

def register(api):
    api.add_reader([".ftm", ".ftmb"], read_ftm)               # yours is asked first for these extensions
    api.add_tool("find_relative", "Find a relative in my tree", {"name": "a name"}, find_relative, required=("name",))
```

* `Extracted(text=…, kind="family_tree", tree=genealogy.Tree(...))` also gives the library the people and families, so the agent's
  `family_lookup` tool and the "Use as character" buttons work for your format. Build the `Tree` from `genealogy.Person` and `genealogy.Family`.
* Skills: drop `.md` files into `ADDONS_DIR/skills/` (same format as `agent/skills/`).
* Tools receive `(ctx, **args)`; `ctx.sess.book` is the open book. Keep them read-only: changes to the book go through the writer's yes.

## Family Tree Maker and other genealogy programs

* **Dependable route:** in the program choose File, Export, GEDCOM (`.ged`). Family Tree Maker (2017 to 2026), Ancestry, MyHeritage, FamilySearch,
  RootsMagic, Legacy, Gramps and Reunion all do this. Add the file to the Library: Book Scribe reads people, dates, places, parents, spouses,
  children, siblings and notes, and the agent can answer questions about anyone.
* **Gramps** `.gramps` files are read directly. A `.ftmb` backup or a `.gdz` package is opened and the tree inside is read.
* **A raw `.ftm` file** is a SQLite database whose tables Family Tree Maker does not document and which can change between versions. Book Scribe
  reads it as plain tables (so you can see and search what is inside) but cannot rebuild the family links reliably. Use the GEDCOM export, or
  write an addon above that knows your version's tables.
