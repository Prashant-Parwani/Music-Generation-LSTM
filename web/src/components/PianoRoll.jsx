import { useEffect, useRef } from "react";
import { INSTRUMENTS } from "../config.js";

const DRUM_ROWS = [
  { name: "cymbal", pitches: [49, 51, 52, 53, 55, 56, 57, 59] },
  { name: "hat", pitches: [42, 44, 46] },
  { name: "tom", pitches: [41, 43, 45, 47, 48, 50] },
  { name: "snare", pitches: [37, 38, 39, 40] },
  { name: "kick", pitches: [35, 36] },
];
const drumRow = (p) => {
  const i = DRUM_ROWS.findIndex((r) => r.pitches.includes(p));
  return i === -1 ? 1 : i;
};

/** Canvas piano roll with a moving playhead. Click or drag to seek. */
export default function PianoRoll({ tracks, duration, getTime, playing, onSeek, muted, height = 260 }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const staticRef = useRef(null);       // offscreen layer with the notes
  const sizeRef = useRef({ w: 0, h: height });
  const rafRef = useRef(0);
  const draggingRef = useRef(false);

  const paintStatic = () => {
    const { w, h } = sizeRef.current;
    if (!w) return;
    const dpr = window.devicePixelRatio || 1;
    const off = document.createElement("canvas");
    off.width = w * dpr; off.height = h * dpr;
    const g = off.getContext("2d");
    g.scale(dpr, dpr);

    const hasDrums = Boolean(tracks.drums?.length);
    const pitched = Object.entries(tracks).filter(([k]) => k !== "drums");
    const all = pitched.flatMap(([, rows]) => rows.map((r) => r[2]));
    const lo = all.length ? Math.min(...all) - 2 : 48;
    const hi = all.length ? Math.max(...all) + 2 : 72;
    const drumH = hasDrums ? Math.min(70, h * 0.28) : 0;
    const gap = hasDrums && pitched.length ? 8 : 0;
    const pitchH = h - drumH - gap;
    const x = (t) => (t / Math.max(duration, 0.1)) * w;

    // background grid (every second, stronger every 4 s)
    g.fillStyle = "rgba(255,255,255,0.025)"; g.fillRect(0, 0, w, h);
    for (let s = 0; s <= duration; s += 1) {
      g.fillStyle = s % 4 === 0 ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.04)";
      g.fillRect(Math.round(x(s)), 0, 1, h);
    }
    // octave lines
    for (let p = Math.ceil(lo / 12) * 12; p <= hi; p += 12) {
      const y = pitchH - ((p - lo) / (hi - lo)) * pitchH;
      g.fillStyle = "rgba(255,255,255,0.06)"; g.fillRect(0, y, w, 1);
    }

    const noteH = Math.max(2, pitchH / (hi - lo + 1) - 0.5);
    for (const [group, rows] of pitched) {
      const dim = muted?.has(group);
      const color = INSTRUMENTS[group]?.color || "#fff";
      for (const [s, e, p, v] of rows) {
        g.globalAlpha = dim ? 0.12 : 0.45 + (v / 127) * 0.55;
        g.fillStyle = color;
        const y = pitchH - ((p - lo + 1) / (hi - lo + 1)) * pitchH;
        g.beginPath();
        g.roundRect(x(s), y, Math.max(2, x(e) - x(s) - 0.5), noteH, 1.5);
        g.fill();
      }
    }
    if (hasDrums) {
      const top = pitchH + gap, rowH = drumH / DRUM_ROWS.length;
      g.globalAlpha = 1;
      g.fillStyle = "rgba(255,255,255,0.04)"; g.fillRect(0, top, w, drumH);
      const dim = muted?.has("drums");
      for (const [s, , p, v] of tracks.drums) {
        g.globalAlpha = dim ? 0.12 : 0.5 + (v / 127) * 0.5;
        g.fillStyle = INSTRUMENTS.drums.color;
        g.fillRect(x(s), top + drumRow(p) * rowH + 1, 3, rowH - 2);
      }
    }
    g.globalAlpha = 1;
    staticRef.current = off;
  };

  const paint = () => {
    const canvas = canvasRef.current, off = staticRef.current;
    if (!canvas || !off) return;
    const { w, h } = sizeRef.current;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0);
    ctx.scale(dpr, dpr);
    const t = Math.min(getTime?.() ?? 0, duration);
    const px = (t / Math.max(duration, 0.1)) * w;
    ctx.fillStyle = "rgba(255,255,255,0.07)"; ctx.fillRect(0, 0, px, h);          // played part
    ctx.fillStyle = "#ffffff"; ctx.fillRect(px - 1, 0, 2, h);                       // playhead
  };

  const resize = () => {
    const wrap = wrapRef.current, canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const w = Math.max(200, Math.floor(wrap.clientWidth));
    const dpr = window.devicePixelRatio || 1;
    sizeRef.current = { w, h: height };
    canvas.width = w * dpr; canvas.height = height * dpr;
    canvas.style.width = "100%"; canvas.style.height = `${height}px`;
    paintStatic(); paint();
  };

  useEffect(() => {
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, duration, height]);

  useEffect(() => { paintStatic(); paint(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [muted]);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    if (!playing) { paint(); return undefined; }
    const loop = () => { paint(); rafRef.current = requestAnimationFrame(loop); };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, tracks, duration]);

  const seekFromEvent = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek?.(frac * duration);
    paint();
  };

  return (
    <div ref={wrapRef} className="w-full overflow-hidden rounded-xl border border-white/10 bg-black/30">
      <canvas
        ref={canvasRef}
        className="block cursor-pointer touch-none"
        role="img"
        aria-label="Piano roll of the generated music. Click to jump to a position."
        onPointerDown={(e) => { draggingRef.current = true; e.currentTarget.setPointerCapture(e.pointerId); seekFromEvent(e); }}
        onPointerMove={(e) => draggingRef.current && seekFromEvent(e)}
        onPointerUp={() => { draggingRef.current = false; }}
      />
    </div>
  );
}
