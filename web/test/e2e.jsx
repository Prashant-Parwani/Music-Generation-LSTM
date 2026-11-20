// End-to-end smoke test of the website against the REAL running server (Tone.js is stubbed: jsdom has no audio).
//   1) start the server:  python server.py
//   2) run:               npm run test:e2e          (optional: API_URL=http://host:port)
import { JSDOM } from "jsdom";

(async () => {
  const BASE = process.env.API_URL || "http://127.0.0.1:8000";
  const dom = new JSDOM("<!doctype html><html><body><div id='root'></div></body></html>", { url: `${BASE}/`, pretendToBeVisual: true });
  const { window } = dom;
  globalThis.window = window; globalThis.document = window.document;
  Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
  globalThis.HTMLElement = window.HTMLElement; globalThis.Node = window.Node;
  globalThis.requestAnimationFrame = (f) => setTimeout(() => f(Date.now()), 16);
  globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const ctx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
  window.HTMLCanvasElement.prototype.getContext = () => ctx;
  window.HTMLCanvasElement.prototype.setPointerCapture = () => {};
  Object.defineProperty(window.HTMLElement.prototype, "clientWidth", { get() { return 700; } });
  const realFetch = globalThis.fetch;
  globalThis.fetch = (u, o) => realFetch(String(u).startsWith("/") ? `${BASE}${u}` : u, o);

  const React = (await import("react")).default;
  const { createRoot } = await import("react-dom/client");
  const { act } = await import("react");
  const App = (await import("../src/App.jsx")).default;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const text = () => document.body.textContent;
  const btn = (label) => [...document.querySelectorAll("button")].find((b) => b.textContent.includes(label));
  const waitFor = async (cond, tries = 80, ms = 250) => { for (let i = 0; i < tries && !cond(); i++) await act(async () => { await sleep(ms); }); return cond(); };
  let failed = 0;
  const check = (name, ok, extra = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); if (!ok) failed++; };
  const errors = [];
  const origErr = console.error;
  console.error = (...a) => errors.push(a.join(" "));

  const root = createRoot(document.getElementById("root"));
  await act(async () => { root.render(React.createElement(App)); });
  check("connects to the server and loads the controls", await waitFor(() => text().includes("Generate music"), 60, 200), "(is `python server.py` running?)");
  check("server badge says Online", /Online/.test(text()));
  if (!btn("Generate music")) { origErr("Cannot continue: controls did not load.\n" + text().slice(0, 400)); process.exit(1); }

  await act(async () => { btn("Rock band").click(); });
  await act(async () => { btn("Short").click(); });
  await act(async () => { btn("Generate music").click(); });
  check("clicking Generate shows progress", /Composing|Waiting in queue/.test(text()));
  check("a result appears (piano roll rendered)", await waitFor(() => Boolean(document.querySelector("canvas")), 160, 250));
  const t = text();
  check("stats are shown", /\d+(\.\d+)?s · \d+ bars · \d+ BPM/.test(t), (t.match(/\d+(\.\d+)?s · \d+ bars · \d+ BPM/) || [""])[0]);
  check("mixer lists the requested instruments (Rock band = drums, bass, guitar)", ["Drums", "Bass", "Guitar"].every((n) => t.includes(n)));
  check("MIDI download and share buttons exist", Boolean(btn("MIDI")) && Boolean(btn("Share")));
  await act(async () => { btn("New variation").click(); });
  check("a new variation is added to the session history", await waitFor(() => document.querySelectorAll("ul li").length >= 2, 160, 250));
  await act(async () => { document.querySelector("button[aria-label^='Mute']").click(); });
  check("mute toggle works", Boolean(document.querySelector("button[aria-label^='Unmute']")));
  const real = errors.filter((e) => !/not wrapped in act|ReactDOMTestUtils|Warning:/.test(e));
  check("no console errors", real.length === 0, real.slice(0, 2).join(" | ").slice(0, 300));
  console.log(failed ? `\n${failed} check(s) FAILED` : "\nAll checks passed");
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error("TEST CRASHED:", e); process.exit(1); });
