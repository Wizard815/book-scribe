"use strict";
const $ = (id) => document.getElementById(id);
const sid = localStorage.getItem("bs_sid") || crypto.randomUUID().replace(/-/g, "");
localStorage.setItem("bs_sid", sid);
const state = { chapter: null, rec: null, chunks: [], speaking: null, book: "demo", holding: false, ttsOff: false };

function say(kind, text) {
  const d = document.createElement("div");
  d.className = "msg " + kind;
  d.textContent = text;
  $("log").append(d);
  $("log").scrollTop = $("log").scrollHeight;
}
const status = (t) => ($("status").textContent = t);

async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) {
    const j = await r.clone().json().catch(() => ({}));  // the server explains itself, e.g. "No chat provider is in any tier"
    throw new Error(j.detail || path + " → " + r.status);
  }
  return r;
}

async function loadChapter(n, focus, flash) {
  const { paragraphs } = await (await api(`/books/${state.book}/chapters/${n}`)).json();
  state.chapter = n;
  $("chapter-title").textContent = "Chapter " + n;
  const root = $("prose");
  root.replaceChildren(...paragraphs.map((p) => {
    const el = document.createElement("p");
    el.id = p.id;
    el.textContent = p.text;
    if (p.id === focus) el.className = "focus" + (flash ? " flash" : "");
    return el;
  }));
  const f = focus && document.getElementById(focus);
  if (f) f.scrollIntoView({ block: "center", behavior: "smooth" });
}

async function speak(text) {
  stopSpeaking();
  try {
    if (state.ttsOff) throw new Error("no server TTS");
    const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (r.ok) {
      const a = new Audio(URL.createObjectURL(await r.blob()));
      state.speaking = a;
      return a.play();
    }
    if (r.status === 501) state.ttsOff = true;  // not configured: stop asking, use the browser's voice
  } catch (_) { /* fall through to browser voice */ }
  if ("speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}
function stopSpeaking() {
  if (state.speaking) state.speaking.pause();
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

async function render(res) {
  if (!res.text) { status("Ready"); return; }  // silence the speech model mis-heard as "Thank you." is ignored server-side: no bubble, no reply
  if (res.user_text) say("you", res.user_text);
  say("bot", res.text);
  $("mode").textContent = res.mode === "therapist" ? "Feedback" : "Narrator";
  $("pending").hidden = !res.pending;
  (res.notes || []).forEach((n) => say("note", n));
  await loadChapter(res.chapter, res.focus, res.changed && res.changed.length);
  // Tell the writer when a lower tier answered, so a quality or latency change is never a mystery.
  const fb = Object.entries(res.served_by || {}).filter(([, s]) => s.fell_back);
  fb.forEach(([role, s]) => say("note", `Using fallback for ${role}: ${s.label}`));
  status(fb.length ? "Ready (fallback)" : "Ready");
  speak(res.text);
}

async function send(promise) {
  status("Thinking…");
  try { await render(await (await promise).json()); }
  catch (e) { status("Error"); say("note", String(e)); }
}

const sendText = (text) => send(api("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, session_id: sid }) }));

$("yes").onclick = () => sendText("yes, save it");
$("no").onclick = () => sendText("no");

// Hold-to-talk. Pointer events cover mouse, touch and pen; Space also works for keyboard users.
async function startRec() {
  stopSpeaking();
  if (state.rec || state.holding) return;
  state.holding = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (!state.holding) { stream.getTracks().forEach((t) => t.stop()); return; }  // released before the mic was ready
    state.chunks = [];
    state.rec = new MediaRecorder(stream);
    state.rec.ondataavailable = (e) => state.chunks.push(e.data);
    state.rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(state.chunks, { type: state.rec.mimeType || "audio/webm" });
      state.rec = null;
      if (blob.size < 1500) return status("Too short — hold longer");
      const fd = new FormData();
      fd.append("audio", blob, "speech.webm");
      fd.append("session_id", sid);
      send(api("/voice", { method: "POST", body: fd }));
    };
    state.rec.start();
    $("mic").classList.add("rec");
    $("mic").textContent = "🔴 Listening… release to send";
    status("Listening");
  } catch (e) { state.holding = false; status("Mic blocked"); say("note", "Allow microphone access in the browser."); }
}
function stopRec() {
  state.holding = false;
  $("mic").classList.remove("rec");
  $("mic").textContent = "🎙️ Hold to talk";
  if (state.rec && state.rec.state === "recording") state.rec.stop();
}
const mic = $("mic");
mic.addEventListener("pointerdown", (e) => { mic.setPointerCapture(e.pointerId); startRec(); });
mic.addEventListener("pointerup", stopRec);
mic.addEventListener("pointercancel", stopRec);
mic.addEventListener("contextmenu", (e) => e.preventDefault());
addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat) { e.preventDefault(); startRec(); } });
addEventListener("keyup", (e) => { if (e.code === "Space") stopRec(); });

(async () => {
  try {
    state.book = (await (await api("/books")).json()).default;
    const { chapters } = await (await api(`/books/${state.book}/chapters`)).json();
    if (chapters.length) await loadChapter(chapters[0]);
    say("note", "Hold the big button and talk.");
  } catch (e) { status("Offline"); say("note", String(e)); }
})();
