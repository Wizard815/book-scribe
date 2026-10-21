# Vendored code

Copied in-tree so Book Scribe builds without fetching these at deploy time. Licenses are kept alongside the code.

| Path | Upstream | Commit | License | What was copied |
|---|---|---|---|---|
| `vendor/mem0/` | https://github.com/mem0ai/mem0 (v2.2.1) | `2c81af3bbe27d9bccb09c772d583df9e9be72ee5` | Apache-2.0 | The `mem0/` Python package, `pyproject.toml`, `LICENSE`, `README.md`. Unmodified. Not copied: server, docs, tests, TypeScript SDK. |
| `vendor/graphify-novel/` | https://github.com/Anshler/graphify-novel | `124c9abc473508e081a625e4d2a24b24071581a2` | MIT | Whole repo (`SKILL.md`, READMEs, `LICENSE`). Unmodified. |

**mem0** is installed in the agent image with `pip install --no-deps ./vendor/mem0` (see `agent/Dockerfile`);
its runtime deps are listed in `agent/requirements.txt`. Used by `agent/memory/store.py`. Telemetry is disabled.

**graphify-novel** is *not code*: it is an AI-assistant skill (`SKILL.md` prompts) that drives the separate
`graphify` tool to build a story graph from a `bible/` + `chapters/` tree. It is kept here as the reference
for the bible layout and review/update prompts. The running agent uses `agent/kg.py` (an in-process graph)
behind the same interface, until the graphify pipeline is wired in.

To update: re-clone upstream, replace the directory, and update the commit above.
