import { useEffect, useMemo, useState } from "react";
import { Clock, Download, Link2, Pause, Play, Repeat, RefreshCw, Square, Volume2, VolumeX, Check } from "lucide-react";
import PianoRoll from "./PianoRoll.jsx";
import { INSTRUMENTS, INSTRUMENT_ORDER } from "../config.js";

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function Player({ result, engine, playing, setPlaying, loop, setLoop, onVariation, shareUrl, busy }) {
  const [muted, setMuted] = useState(new Set());
  const [volumes, setVolumes] = useState({});
  const [time, setTime] = useState(0);
  const [copied, setCopied] = useState(false);
  const groups = useMemo(() => INSTRUMENT_ORDER.filter((g) => result.tracks[g]?.length), [result]);

  // new result -> reset mixer
  useEffect(() => {
    setMuted(new Set()); setVolumes({}); setTime(0);
    groups.forEach((g) => { engine.setMuted(g, false); engine.setVolume(g, 0); });
  }, [result, engine, groups]);

  // clock + end detection
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setTime(engine.getTime());
      if (engine.isFinished()) { engine.stop(); setPlaying(false); setTime(0); }
    }, 100);
    return () => clearInterval(id);
  }, [playing, engine, setPlaying]);

  const togglePlay = async () => {
    if (playing) { engine.pause(); setPlaying(false); }
    else { await engine.play(); setPlaying(true); }
  };
  const stop = () => { engine.stop(); setPlaying(false); setTime(0); };
  const seek = (t) => { engine.seek(t); setTime(t); };
  const toggleMute = (g) => {
    const next = new Set(muted);
    next.has(g) ? next.delete(g) : next.add(g);
    setMuted(next);
    engine.setMuted(g, next.has(g));
  };
  const setVol = (g, v) => { setVolumes((o) => ({ ...o, [g]: v })); engine.setVolume(g, v); };

  const downloadMidi = () => {
    const bytes = Uint8Array.from(atob(result.midi_base64), (c) => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
    const a = document.createElement("a");
    a.href = url; a.download = `ai-music-${result.model}-${result.bpm}bpm-seed${result.seed}.mid`;
    a.click(); URL.revokeObjectURL(url);
  };
  const copyLink = async () => {
    let ok = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl);
        ok = true;
      }
    } catch { /* fallback */ }
    if (!ok) {
      try {
        const ta = document.createElement("textarea");
        ta.value = shareUrl;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch { /* ignore */ }
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="glass flex flex-col gap-5 rounded-2xl p-5 sm:p-6">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={togglePlay} disabled={busy}
          className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-500/30 transition hover:brightness-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-fuchsia-300"
          aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={22} /> : <Play size={22} className="ml-0.5" />}
        </button>
        <button type="button" onClick={stop} className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-slate-200 hover:bg-white/15" aria-label="Stop"><Square size={16} /></button>
        <button type="button" onClick={() => { const v = !loop; setLoop(v); engine.setLoop(v); }} aria-pressed={loop}
          className={`flex h-10 w-10 items-center justify-center rounded-full transition ${loop ? "bg-indigo-500/80 text-white" : "bg-white/10 text-slate-300 hover:bg-white/15"}`} aria-label="Loop"><Repeat size={16} /></button>
        <div className="ml-2 font-mono text-sm tabular-nums text-slate-300">{fmt(time)} <span className="text-slate-500">/ {fmt(result.duration)}</span></div>
        <div className="ml-auto flex flex-wrap gap-2">
          <button type="button" onClick={onVariation} disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-sm text-slate-100 hover:bg-white/15 disabled:opacity-50"><RefreshCw size={15} />New variation</button>
          <button type="button" onClick={downloadMidi}
            className="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-sm text-slate-100 hover:bg-white/15"><Download size={15} />MIDI</button>
          <button type="button" onClick={copyLink}
            className="flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-sm text-slate-100 hover:bg-white/15">
            {copied ? <Check size={15} className="text-emerald-400" /> : <Link2 size={15} />}{copied ? "Copied" : "Share"}</button>
        </div>
      </div>

      <PianoRoll tracks={result.tracks} duration={result.duration} getTime={() => engine.getTime()}
        playing={playing} onSeek={seek} muted={muted} />

      <div className="grid gap-2 sm:grid-cols-2">
        {groups.map((g) => {
          const meta = INSTRUMENTS[g];
          const isMuted = muted.has(g);
          return (
            <div key={g} className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2">
              <button type="button" onClick={() => toggleMute(g)} aria-pressed={isMuted} aria-label={`${isMuted ? "Unmute" : "Mute"} ${meta.label}`}
                className="text-slate-300 hover:text-white">{isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}</button>
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: meta.color, opacity: isMuted ? 0.3 : 1 }} />
              <span className={`w-16 text-sm ${isMuted ? "text-slate-500" : "text-slate-200"}`}>{meta.label}</span>
              <input type="range" min={-24} max={6} step={1} value={volumes[g] ?? 0} onChange={(e) => setVol(g, Number(e.target.value))}
                aria-label={`${meta.label} volume`} className="flex-1" />
              <span className="w-10 text-right text-xs tabular-nums text-slate-500">{result.stats.notes[g] ?? 0}</span>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-400">
        <span className="flex items-center gap-1"><Clock size={13} /> {result.duration}s · {result.bars} bars · {result.bpm} BPM</span>
        <span>{result.stats.total_notes} notes</span>
        <span>{result.model === "lstm" ? "LSTM" : "Transformer"} · seed {result.seed}</span>
        <span>made in {result.elapsed}s</span>
      </div>
    </div>
  );
}
