"use strict";
// Admin menu: for each of chat / embeddings / speech-to-text / text-to-speech / story graph, three ordered tiers
// (default: T1 Local ROCm, T2 CPU fallback, T3 Frontier). A call tries T1 first and only reaches T2, then T3,
// when everything before it has failed. Talks to /api/admin/* with the ADMIN_TOKEN (sessionStorage only).
(() => {
  const $ = (id) => document.getElementById(id);
  const ROLE_TITLES = {
    chat: ["Chat agent", "The model that writes and talks."],
    stt: ["Speech-to-text", "Whisper or any /v1/audio/transcriptions server."],
    tts: ["Text-to-speech", "Optional. If every tier fails or none is set, the browser's built-in voice reads replies."],
    graph: ["Story graph (graphify)", "Builds the story's relationship graph in the background. Tried tier by tier, run by run."],
    embed: ["Embeddings", "Used by long-term memory (mem0). No failover: different models make incompatible vectors, so only the highest-priority provider is used."],
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
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) k.startsWith("aria-") ? el.setAttribute(k, v) : (el[k] = v); // aria-* are attributes, not properties
    el.append(...kids.filter((k) => k != null));
    return el;
  }

  async function load() {
    msg("");
    try { snap = await call("GET", "/providers"); $("gate").hidden = true; render(); }
    catch (e) { msg(e.message); $("gate").hidden = /disabled/.test(e.message); }
  }

  function render() {
    $("roles").replaceChildren(settingsBox(), ...Object.keys(ROLE_TITLES).map(roleSection));
  }

  // --- tier names + cooldown ------------------------------------------------
  function settingsBox() {
    const inputs = snap.labels.map((l, i) => h("input", { value: l, maxLength: 30, "aria-label": `Tier ${i + 1} name` }));
    const cool = h("input", { type: "number", min: 0, max: 3600, value: snap.cooldown });
    const box = h("div", { className: "form" },
      h("strong", { textContent: "Tiers" }),
      h("small", { textContent: " A request goes to T1 first. T2 is tried only if every T1 provider failed, and T3 only after T2 failed too." }),
      h("div", { className: "row3" }, ...inputs),
      h("label", {}, "Skip a failed provider for this many seconds (so a dead server doesn't slow every request)", cool),
      h("button", { className: "primary", textContent: "Save tier names", onclick: () => act(() => call("PUT", "/tier-settings", { labels: inputs.map((i) => i.value), cooldown: Number(cool.value) })) }));
    return box;
  }

  // --- one role: three lanes + unassigned --------------------------------------
  function roleSection(role) {
    const { tiers, profiles } = snap.roles[role];
    const byId = Object.fromEntries(profiles.map((p) => [p.id, p]));
    const placed = new Set(tiers.flat());
    const box = h("section", {}, h("h3", { textContent: ROLE_TITLES[role][0] }), h("small", { textContent: ROLE_TITLES[role][1] }));
    tiers.forEach((ids, ti) => {
      const lane = h("div", { className: "lane" }, h("div", { className: "lane-title", textContent: snap.labels[ti] }));
      if (!ids.length) lane.append(h("div", { className: "empty", textContent: "nothing here" }));
      ids.forEach((id, i) => lane.append(profileRow(role, byId[id], tiers, ti, i)));
      box.append(lane);
    });
    const loose = profiles.filter((p) => !placed.has(p.id));
    if (loose.length) {
      const lane = h("div", { className: "lane off" }, h("div", { className: "lane-title", textContent: "Unassigned (never called)" }));
      loose.forEach((p) => lane.append(profileRow(role, p, tiers, -1, 0)));
      box.append(lane);
    }
    const formHost = h("div");
    box.append(h("button", { textContent: "+ Add provider", onclick: () => formHost.replaceChildren(form(role, null, formHost)) }), formHost);
    return box;
  }

  const putTiers = (role, tiers) => act(() => call("PUT", `/tiers/${role}`, { tiers }));

  function moveTo(role, tiers, p, from, to) {
    const t = tiers.map((x) => x.filter((i) => i !== p.id)); // to: -1 = unassigned
    if (to >= 0) t[to].push(p.id);
    putTiers(role, t);
  }

  function shift(role, tiers, ti, i, d) {
    const t = tiers.map((x) => [...x]);
    [t[ti][i], t[ti][i + d]] = [t[ti][i + d], t[ti][i]];
    putTiers(role, t);
  }

  function healthChip(p) {
    const x = p.health;
    if (!x) return h("span", { className: "badge", textContent: "not called yet" });
    if (x.cooling) return h("span", { className: "badge bad", textContent: `failing, skipped ${x.failed_ago}s ago`, title: x.error });
    if (x.error) return h("span", { className: "badge bad", textContent: "last call failed", title: x.error });
    return h("span", { className: "badge ok", textContent: `ok ${x.ok_ago}s ago` });
  }

  function profileRow(role, p, tiers, ti, i) {
    const result = h("div", { className: "test" });
    const lane = ti >= 0 ? tiers[ti] : [];
    const where = h("select", { "aria-label": "Tier", onchange: (e) => moveTo(role, tiers, p, ti, Number(e.target.value)) },
      ...snap.labels.map((l, k) => h("option", { value: k, textContent: l })), h("option", { value: -1, textContent: "Unassigned" }));
    where.value = ti;
    const row = h("div", { className: "prof" },
      h("div", { className: "meta" },
        h("strong", { textContent: p.label }),
        h("span", { className: "badge " + p.kind, textContent: p.kind }),
        ti >= 0 ? healthChip(p) : null,
        h("small", { textContent: `${p.base_url || "(not set)"} · ${p.model || "default model"}${p.has_key ? " · key " + p.key_hint : ""}${p.timeout ? " · " + p.timeout + "s timeout" : ""}` }),
        p.health && p.health.error ? h("small", { className: "err", textContent: p.health.error }) : null),
      ti >= 0 && i > 0 ? h("button", { textContent: "↑", title: "Try earlier within this tier", onclick: () => shift(role, tiers, ti, i, -1) }) : null,
      ti >= 0 && i < lane.length - 1 ? h("button", { textContent: "↓", title: "Try later within this tier", onclick: () => shift(role, tiers, ti, i, 1) }) : null,
      where,
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
      h("button", { textContent: "Edit", onclick: () => { const host = wrap.lastChild; host.replaceChildren(form(role, p, host)); } }),
      h("button", { className: "danger", textContent: "Delete", onclick: () => confirm(`Delete "${p.label}"?`) && act(() => call("DELETE", `/providers/${role}/${p.id}`)) }),
      result);
    const wrap = h("div", {}, row, h("div")); // second div hosts the edit form
    return wrap;
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
      if (x.kind === "cloud" && f.tier.value === "1") f.tier.value = "3";
      if (!f.label.value) f.label.value = x.label.split(" (")[0];
    } }, h("option", { value: "", textContent: "Start from a preset…" }), ...snap.presets.map((x) => h("option", { value: x.id, textContent: x.label })));
    const dl = h("datalist", { id: "models-" + role });
    for (const m of models[`${role}:${p && p.id}`] || []) dl.append(h("option", { value: m }));
    const kind = h("select", { name: "kind" }, h("option", { value: "local", textContent: "Local" }), h("option", { value: "cloud", textContent: "Cloud / frontier" }));
    kind.value = (p && p.kind) || "local";
    const cur = p ? snap.roles[role].tiers.findIndex((t) => t.includes(p.id)) + 1 : 1; // findIndex -1 → 0 = unassigned
    const tier = h("select", { name: "tier" }, h("option", { value: 0, textContent: "Unassigned (saved, never called)" }),
      ...snap.labels.map((l, k) => h("option", { value: k + 1, textContent: l })));
    tier.value = cur;
    f.append(
      p ? null : h("label", {}, "Preset", preset),
      field("Name", "label", "text", { required: true }),
      h("div", { className: "row" }, h("label", {}, "Type", kind), h("label", {}, "Tier", tier)),
      field("Base URL (no /v1). For Argos use its address, e.g. http://argos:8080", "base_url", "url", { required: true, placeholder: "http://192.168.1.10:8080" }),
      h("div", { className: "row" },
        field("Model", "model", "text", { list: "models-" + role, placeholder: role === "stt" ? "whisper-1" : "model name" }),
        field(p && p.has_key ? `API key (stored ${p.key_hint}; leave blank to keep)` : "API key (cloud only)", "api_key", "password", { value: "", autocomplete: "new-password" })),
      h("div", { className: "row" },
        field("Give up after (seconds, 0 = default)", "timeout", "number", { min: 0, max: 3600, step: 1 }),
        role === "tts" ? field("Voice", "voice") : h("div")),
      dl,
      h("button", { className: "primary", textContent: "Save" }),
      h("button", { type: "button", textContent: "Cancel", onclick: () => host.replaceChildren() }));
    f.onsubmit = (ev) => {
      ev.preventDefault();
      const d = Object.fromEntries(new FormData(f));
      d.tier = Number(d.tier); d.timeout = Number(d.timeout || 0);
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
