import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Music4, RotateCw } from "lucide-react";
import { generate, getConfig, waitForServer } from "./api.js";
import { MusicEngine } from "./audio/engine.js";
import { APP_NAME, INSTRUMENT_ORDER, TAGLINE } from "./config.js";
import Controls from "./components/Controls.jsx";
import Footer from "./components/Footer.jsx";
import Header from "./components/Header.jsx";
import History from "./components/History.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import Player from "./components/Player.jsx";

const engine = new MusicEngine();
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));

/** Defaults from the server, overridden by a shared link (?m=&b=&d=&i=&l=&s=) when valid. */
function initialSettings(cfg) {
  const d = cfg.defaults;
  const q = new URLSearchParams(window.location.search);
  const num = (k) => (q.has(k) && !Number.isNaN(Number(q.get(k))) ? Number(q.get(k)) : null);
  const model = cfg.models.some((m) => m.id === q.get("m")) ? q.get("m") : d.model;
  const instr = (q.get("i") || "").split(",").filter((g) => cfg.instruments.includes(g));
  const bpm = num("b"), dens = num("d"), len = num("l"), seed = q.get("s");
  return {
    model,
    bpm: bpm && bpm >= cfg.bpm.min && bpm <= cfg.bpm.max ? bpm : d.bpm,
    density: dens !== null && dens >= 0 && dens < cfg.density_levels.length ? dens : d.density,
    instruments: instr.length ? instr : d.instruments,
    length: cfg.lengths.some((l) => l.tokens === len) ? len : d.length,
    temperature: d.temperature,
    top_p: d.top_p,
    hard_instruments: d.hard_instruments,
    seed: seed && /^\d+$/.test(seed) ? seed : "",
  };
}

export default function App() {
  const [server, setServer] = useState({ state: "connecting", seconds: 0, device: "", queue: 0 });
  const [config, setConfig] = useState(null);
  const [settings, setSettings] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const abortRef = useRef(null);

  const connect = useCallback(() => {
    const ac = new AbortController();
    abortRef.current = ac;
    setServer({ state: "connecting", seconds: 0, device: "", queue: 0 });
    (async () => {
      try {
        const health = await waitForServer(
          (s) => setServer((o) => ({ ...o, state: s > 6 ? "waking" : "connecting", seconds: s })), ac.signal);
        const cfg = await getConfig();
        setConfig(cfg);
        setSettings(initialSettings(cfg));
        setServer({ state: "online", seconds: 0, device: health.device, queue: health.queue });
      } catch {
        if (!ac.signal.aborted) setServer((o) => ({ ...o, state: "offline" }));
      }
    })();
    return () => ac.abort();
  }, []);
  useEffect(() => connect(), [connect]);

  const update = useCallback((patch) => setSettings((s) => ({ ...s, ...patch })), []);

  const run = useCallback(async (override = {}) => {
    if (busy || !settings) return;
    engine.unlock();                                   // must happen inside the click gesture
    const s = { ...settings, ...override };
    setSettings(s);
    setBusy(true); setError(""); setJob({ status: "queued", progress: 0 });
    engine.stop(); setPlaying(false);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const payload = {
        model: s.model, bpm: s.bpm, density: s.density,
        instruments: INSTRUMENT_ORDER.filter((g) => s.instruments.includes(g)),
        length: s.length, temperature: s.temperature, top_p: s.top_p,
        seed: s.seed === "" || s.seed == null ? null : Number(s.seed),
        hard_instruments: s.hard_instruments,
      };
      const res = await generate(payload, setJob, ac.signal);
      await engine.load(res.tracks, res.duration);
      const item = { id: uid(), result: res, settings: s };
      setResult(res); setActiveId(item.id);
      setHistory((h) => [item, ...h].slice(0, 8));
      await engine.play(); setPlaying(true);
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || String(e));
    } finally {
      setBusy(false); setJob(null);
    }
  }, [busy, settings]);

  const loadFromHistory = async (item) => {
    if (busy) return;
    engine.stop(); setPlaying(false);
    await engine.load(item.result.tracks, item.result.duration);
    setResult(item.result); setActiveId(item.id); setSettings(item.settings);
  };

  const shareUrl = useMemo(() => {
    const item = history.find((h) => h.id === activeId);
    if (!item) return "";
    const s = item.settings;
    const p = new URLSearchParams({
      m: s.model, b: String(s.bpm), d: String(s.density),
      i: INSTRUMENT_ORDER.filter((g) => s.instruments.includes(g)).join(","),
      l: String(s.length), s: String(item.result.seed),
    });
    return `${window.location.origin}${window.location.pathname}?${p.toString()}`;
  }, [history, activeId]);

  return (
    <div className="aurora min-h-screen">
      <Header server={server} />
      <main className="mx-auto max-w-6xl px-4 pb-8">
        <section className="py-8 text-center sm:py-12">
          <h1 className="bg-gradient-to-r from-indigo-300 via-fuchsia-300 to-sky-300 bg-clip-text text-4xl font-bold tracking-tight text-transparent sm:text-5xl">
            {APP_NAME}
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-base text-slate-400 sm:text-lg">{TAGLINE}</p>
        </section>

        {error && (
          <div role="alert" className="mb-6 flex items-start gap-3 rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
            <AlertTriangle size={18} className="mt-0.5 shrink-0" />
            <div>{error}</div>
          </div>
        )}

        {server.state === "offline" && (
          <div className="glass mb-6 flex flex-col items-center gap-3 rounded-2xl p-8 text-center">
            <p className="text-slate-300">The music server isn't answering. If it runs on a free host it may be asleep or restarting.</p>
            <button type="button" onClick={connect}
              className="flex items-center gap-2 rounded-lg bg-white/10 px-4 py-2 text-sm hover:bg-white/15"><RotateCw size={15} /> Try again</button>
          </div>
        )}

        {!config && server.state !== "offline" && (
          <div className="glass rounded-2xl p-10 text-center text-slate-400">
            <p>{server.state === "waking"
              ? "The server is waking up after a nap. This can take up to a minute on the free tier…"
              : "Connecting to the music server…"}</p>
          </div>
        )}

        {config && settings && (
          <div className="grid items-start gap-6 lg:grid-cols-[400px_minmax(0,1fr)]">
            <div className="min-w-0">
              <Controls config={config} settings={settings} update={update} busy={busy} job={job} onGenerate={() => run()} />
            </div>
            <div className="flex min-w-0 flex-col gap-6">
              {result ? (
                <Player result={result} engine={engine} playing={playing} setPlaying={setPlaying} loop={loop} setLoop={setLoop}
                  busy={busy} shareUrl={shareUrl} onVariation={() => run({ seed: "" })} />
              ) : (
                <div className="glass flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-2xl p-8 text-center">
                  <Music4 size={42} className="text-indigo-300" />
                  <p className="text-lg font-medium text-slate-200">Your music will appear here</p>
                  <p className="max-w-sm text-sm text-slate-400">Choose a quick-start style on the left, or set your own instruments, tempo and density, then press Generate.</p>
                </div>
              )}
              <History items={history} activeId={activeId} onPick={loadFromHistory} />
            </div>
          </div>
        )}

        <HowItWorks />
      </main>
      <Footer />
    </div>
  );
}
