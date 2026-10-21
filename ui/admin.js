"use strict";
// Admin menu: pick which model serves chat / embeddings / speech-to-text / text-to-speech,
// local or cloud. Talks to /api/admin/* with the ADMIN_TOKEN (kept in sessionStorage only).
(() => {
  const $ = (id) => document.getElementById(id);
  const ROLE_TITLES = {
    chat: ["Chat agent", "The model that writes and talks."],
    embed: ["Embeddings", "Used by long-term memory (mem0)."],
    stt: ["Speech-to-text", "Whisper or any /v1/audio/transcriptions server."],
    tts: ["Text-to-speech", "Optional. Without one, the browser's built-in voice reads replies."],
  };
  let snap = null;
  let models = {}; // `${role}:${id}` → model names found by the last Test
  const tokenKey = "bs_admin";
  const msg = (t) => ($("admin-msg").textContent = t || "");

  async function call(method, path, body) {
    const r = await fetch("/api/admin" + path, {
      method,
      headers: { "Content-Type": "application/json", "X-Admin-Token": sessionStorage.getItem(tokenKey) || "" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (r.status === 401) { sessionStorage.removeItem(tokenKey); $("gate").hidden = false; $("roles").replaceChildren(); throw new Error("Wrong admin token."); }
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.status);
    return j;
  }

  function h(tag, props = {}, ...kids) {
    const el = Object.assign(document.createElement(tag), props);
    el.append(...kids.filter((k) => k != null));
    return el;
  }

  async function load() {
    msg("");
    try { snap = await call("GET", "/providers"); $("gate").hidden = true; render(); }
    catch (e) { msg(e.message); if (/disabled/.test(e.message)) $("gate").hidden = true; else $("gate").hidden = false; }
  }

  function render() {
    $("roles").replaceChildren(...Object.keys(ROLE_TITLES).map(roleSection));
  }

  function roleSection(role) {
    const { active, profiles } = snap.roles[role];
    const box = h("section", {}, h("h3", { textContent: ROLE_TITLES[role][0] }), h("small", { textContent: ROLE_TITLES[role][1] }));
    for (const p of profiles) box.append(profileRow(role, p, p.id === active));
    const formHost = h("div");
    box.append(h("button", { textContent: "+ Add provider", onclick: () => formHost.replaceChildren(form(role, null, formHost)) }), formHost);
    return box;
  }

  function profileRow(role, p, isActive) {
    const result = h("div", { className: "test" });
    const row = h("div", { className: "prof" + (isActive ? " active" : "") },
      h("div", { className: "meta" },
        h("strong", { textContent: p.label }),
        h("span", { className: "badge " + p.kind, textContent: p.kind }),
        isActive ? h("span", { className: "badge", textContent: "ACTIVE" }) : null,
        h("small", { textContent: `${p.base_url || "(not set)"} · ${p.model || "default model"}${p.has_key ? " · key " + p.key_hint : ""}` })),
      isActive ? null : h("button", { className: "primary", textContent: "Use this", onclick: () => act(() => call("POST", `/providers/${role}/${p.id}/activate`)) }),
      h("button", { textContent: "Test", onclick: async (ev) => {
        ev.target.disabled = true; result.className = "test"; result.textContent = "Testing…";
        try {
          const r = await call("POST", `/providers/${role}/${p.id}/test`);
          models[`${role}:${p.id}`] = r.models || [];
          result.className = "test " + (r.ok ? "ok" : "bad");
          result.textContent = (r.ok ? "✓ Reachable. " : "✗ ") + (r.detail || "") + (r.models.length ? ` ${r.models.length} models: ${r.models.slice(0, 8).join(", ")}${r.models.length > 8 ? "…" : ""}` : "");
        } catch (e) { result.className = "test bad"; result.textContent = e.message; }
        ev.target.disabled = false;
      } }),
      h("button", { textContent: "Edit", onclick: (ev) => { const host = row.nextSibling; host.replaceChildren(form(role, p, host)); } }),
      h("button", { className: "danger", textContent: "Delete", onclick: () => confirm(`Delete "${p.label}"?`) && act(() => call("DELETE", `/providers/${role}/${p.id}`)) }),
      result);
    return h("div", {}, row, h("div")); // second div hosts the edit form
  }

  function form(role, p, host) {
    const f = h("form", { className: "form" });
    const field = (label, name, type = "text", { list, ...extra } = {}) => {
      const input = h("input", { name, type, value: (p && p[name]) || "", autocomplete: "off", ...extra });
      if (list) input.setAttribute("list", list); // input.list is read-only, so it can't be assigned as a property
      return h("label", {}, label, input);
    };
    const preset = h("select", { onchange: () => {
      const x = snap.presets.find((q) => q.id === preset.value);
      if (!x) return;
      f.base_url.value = x.base_url; f.kind.value = x.kind;
      if (!f.label.value) f.label.value = x.label.split(" (")[0];
    } }, h("option", { value: "", textContent: "Start from a preset…" }), ...snap.presets.map((x) => h("option", { value: x.id, textContent: x.label })));
    const dl = h("datalist", { id: "models-" + role });
    for (const m of models[`${role}:${p && p.id}`] || []) dl.append(h("option", { value: m }));
    const kind = h("select", { name: "kind" }, h("option", { value: "local", textContent: "Local" }), h("option", { value: "cloud", textContent: "Cloud / frontier" }));
    kind.value = (p && p.kind) || "local";
    f.append(
      p ? null : h("label", {}, "Preset", preset),
      field("Name", "label", "text", { required: true }),
      h("div", { className: "row" }, h("label", {}, "Type", kind), field("Base URL (no /v1)", "base_url", "url", { required: true, placeholder: "http://192.168.1.10:8080" })),
      h("div", { className: "row" },
        field("Model", "model", "text", { list: "models-" + role, placeholder: role === "stt" ? "whisper-1" : "model name" }),
        field(p && p.has_key ? `API key (stored ${p.key_hint}; leave blank to keep)` : "API key (cloud only)", "api_key", "password", { value: "", autocomplete: "new-password" })),
      role === "tts" ? field("Voice", "voice") : null,
      dl,
      h("button", { className: "primary", textContent: "Save" }),
      h("button", { type: "button", textContent: "Cancel", onclick: () => host.replaceChildren() }));
    f.onsubmit = (ev) => {
      ev.preventDefault();
      const d = Object.fromEntries(new FormData(f));
      act(() => call("PUT", "/providers/" + role, { ...d, id: p ? p.id : null }));
    };
    return f;
  }

  async function act(fn) {
    msg("");
    try { await fn(); snap = await call("GET", "/providers"); render(); }
    catch (e) { msg(e.message); }
  }

  $("gear").onclick = () => { $("admin").hidden = false; load(); };
  $("admin-close").onclick = () => ($("admin").hidden = true);
  addEventListener("keydown", (e) => { if (e.key === "Escape") $("admin").hidden = true; });
  $("gate").onsubmit = (e) => { e.preventDefault(); sessionStorage.setItem(tokenKey, $("token").value); $("token").value = ""; load(); };
})();
