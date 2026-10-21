# Vendored code

Copied in-tree so Book Scribe builds without fetching these at deploy time. Licenses are kept alongside the code.

| Path | Upstream | Commit | License | What was copied |
|---|---|---|---|---|
| `vendor/mem0/` | https://github.com/mem0ai/mem0 (v2.2.1) | `2c81af3bbe27d9bccb09c772d583df9e9be72ee5` | Apache-2.0 | The `mem0/` Python package, `pyproject.toml`, `LICENSE`, `README.md`. Unmodified. Not copied: server, docs, tests, TypeScript SDK. |
| `vendor/graphify/` | https://github.com/Graphify-Labs/graphify (v0.9.93) | `b90b9df7949bd04a15b40a1624af85d0a9cfdf04` | Apache-2.0 (+ `LICENSE-MIT`, `NOTICE`) | The `graphify/` Python package, `pyproject.toml`, licenses. Unmodified. Not copied: tests, docs, benchmarks, worked examples. |
| `vendor/graphify-novel/` | https://github.com/Anshler/graphify-novel | `124c9abc473508e081a625e4d2a24b24071581a2` | MIT | Whole repo (`SKILL.md`, READMEs, `LICENSE`). Unmodified. |

**mem0** is installed in the agent image with `pip install --no-deps ./vendor/mem0` (see `agent/Dockerfile`);
its runtime deps are listed in `agent/requirements.txt`. Used by `agent/memory/store.py`. Telemetry is disabled.

**graphify** is installed in the agent image with `pip install ./vendor/graphify` (its deps include ~25 tree-sitter
grammars, which are only needed for code; they make the image larger but are harmless). It is run headless as
`python -m graphify extract` by `agent/story_graph.py` against each book folder, using the **Story graph** provider from
the admin menu (any OpenAI-compatible endpoint, passed as `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `GRAPHIFY_OPENAI_MODEL`).

**graphify-novel** is *not code*: it is a skill (`SKILL.md` prompts) layered on graphify for novels. Its **bible layout**
(`bible/premise.md`, `characters/<slug>.md` with YAML state + arc log, `threads/`, `world/`, `timeline.md`) is what
`agent/bible.py` implements. Its slash-command prompts are not executed by the agent; they are kept as reference.

To update: re-clone upstream, replace the directory, and update the commit above.
