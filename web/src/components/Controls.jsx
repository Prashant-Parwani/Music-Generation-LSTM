import { useState } from "react";
import { ChevronDown, Dices, Loader2, Sparkles } from "lucide-react";
import { INSTRUMENTS, INSTRUMENT_ORDER, PRESETS } from "../config.js";

function Section({ title, hint, children }) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</h3>
        {hint && <span className="text-xs text-slate-500">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function Segmented({ options, value, onChange, disabled }) {
  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }} role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`rounded-lg px-2 py-2 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
            value === o.value ? "bg-indigo-500/90 text-white shadow shadow-indigo-500/30" : "bg-white/5 text-slate-300 hover:bg-white/10"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export default function Controls({ config, settings, update, busy, job, onGenerate }) {
  const [advanced, setAdvanced] = useState(false);
  const toggleInstrument = (g) => {
    const on = settings.instruments.includes(g);
    if (on && settings.instruments.length === 1) return;          // keep at least one
    update({ instruments: on ? settings.instruments.filter((x) => x !== g) : [...settings.instruments, g] });
  };
  const applyPreset = (p) => update({ bpm: p.bpm, density: p.density, instruments: p.instruments });
  const pct = Math.round((job?.progress || 0) * 100);

  return (
    <div className="glass flex flex-col gap-6 rounded-2xl p-5 sm:p-6">
      <Section title="Quick start">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button key={p.name} type="button" disabled={busy} onClick={() => applyPreset(p)}
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-slate-200 transition hover:border-indigo-400/60 hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">
              <span className="mr-1.5">{p.emoji}</span>{p.name}
            </button>
          ))}
        </div>
      </Section>

      <Section title="Model" hint={config.models.length > 1 ? "compare both!" : undefined}>
        <div className={`grid gap-2 ${config.models.length > 1 ? "sm:grid-cols-2" : ""}`}>
          {config.models.map((m) => (
            <button key={m.id} type="button" disabled={busy} onClick={() => update({ model: m.id })}
              className={`rounded-xl border p-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
                settings.model === m.id ? "border-indigo-400/70 bg-indigo-500/15" : "border-white/10 bg-white/5 hover:bg-white/10"
              }`}>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-100">{m.label}</span>
                <span className="text-xs text-slate-400">{m.params_m}M params</span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">{m.blurb}</p>
            </button>
          ))}
        </div>
      </Section>

      <Section title="Instruments" hint="pick at least one">
        <div className="grid grid-cols-5 gap-2">
          {INSTRUMENT_ORDER.filter((g) => config.instruments.includes(g)).map((g) => {
            const on = settings.instruments.includes(g);
            const meta = INSTRUMENTS[g];
            return (
              <button key={g} type="button" disabled={busy} aria-pressed={on} onClick={() => toggleInstrument(g)}
                className={`flex flex-col items-center gap-1 rounded-xl border px-1 py-3 text-xs font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
                  on ? "text-white" : "border-white/10 bg-white/5 text-slate-400 hover:bg-white/10"
                }`}
                style={on ? { borderColor: meta.color, background: `${meta.color}26` } : undefined}>
                <span className="text-xl leading-none">{meta.emoji}</span>
                {meta.label}
              </button>
            );
          })}
        </div>
      </Section>

      <Section title="Tempo" hint={`${settings.bpm} BPM`}>
        <input type="range" min={config.bpm.min} max={config.bpm.max} step={1} value={settings.bpm}
          disabled={busy} onChange={(e) => update({ bpm: Number(e.target.value) })} aria-label="Tempo in beats per minute" />
        <div className="mt-1 flex justify-between text-[11px] text-slate-500"><span>Slow</span><span>Medium</span><span>Fast</span></div>
      </Section>

      <Section title="Density" hint="how busy the music is">
        <Segmented options={config.density_levels} value={settings.density} onChange={(v) => update({ density: v })} disabled={busy} />
      </Section>

      <Section title="Length">
        <Segmented options={config.lengths.map((l) => ({ value: l.tokens, label: l.label }))}
          value={settings.length} onChange={(v) => update({ length: v })} disabled={busy} />
      </Section>

      <div>
        <button type="button" onClick={() => setAdvanced((a) => !a)}
          className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 hover:text-slate-200">
          Advanced
          <ChevronDown size={16} className={`transition ${advanced ? "rotate-180" : ""}`} />
        </button>
        {advanced && (
          <div className="mt-4 flex flex-col gap-4">
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-400"><span>Creativity (temperature)</span><span>{settings.temperature.toFixed(2)}</span></div>
              <input type="range" min={0.6} max={1.3} step={0.05} value={settings.temperature} disabled={busy}
                onChange={(e) => update({ temperature: Number(e.target.value) })} aria-label="Temperature" />
              <p className="mt-1 text-[11px] text-slate-500">Lower = safer and more repetitive, higher = more surprising.</p>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-400"><span>Top-p</span><span>{settings.top_p.toFixed(2)}</span></div>
              <input type="range" min={0.7} max={1} step={0.01} value={settings.top_p} disabled={busy}
                onChange={(e) => update({ top_p: Number(e.target.value) })} aria-label="Top p" />
            </div>
            <div className="flex items-end gap-2">
              <label className="flex-1 text-xs text-slate-400">
                Seed (empty = random)
                <input type="number" min={0} value={settings.seed} disabled={busy}
                  onChange={(e) => update({ seed: e.target.value })} placeholder="random"
                  className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-slate-100 outline-none focus:border-indigo-400" />
              </label>
              <button type="button" disabled={busy} onClick={() => update({ seed: String(Math.floor(Math.random() * 1e9)) })}
                className="rounded-lg border border-white/10 bg-white/5 p-2.5 text-slate-300 hover:bg-white/10" aria-label="Random seed" title="Random seed">
                <Dices size={18} />
              </button>
            </div>
            <label className="flex items-start gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={settings.hard_instruments} disabled={busy}
                onChange={(e) => update({ hard_instruments: e.target.checked })} className="mt-1 accent-indigo-500" />
              <span>Strict instruments
                <span className="block text-xs text-slate-500">Never let an instrument you switched off sneak in.</span></span>
            </label>
          </div>
        )}
      </div>

      <div>
        <button type="button" onClick={onGenerate} disabled={busy}
          className={`flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3.5 text-base font-semibold text-white shadow-lg transition focus:outline-none focus-visible:ring-2 focus-visible:ring-fuchsia-300 ${
            busy ? "shimmer cursor-wait bg-gradient-to-r from-indigo-600 via-fuchsia-500 to-indigo-600" : "bg-gradient-to-r from-indigo-500 to-fuchsia-500 shadow-indigo-500/30 hover:brightness-110"
          }`}>
          {busy ? <Loader2 className="animate-spin" size={20} /> : <Sparkles size={20} />}
          {busy ? (job?.status === "queued" ? `Waiting in queue${job.position ? ` (${job.position} ahead)` : ""}…` : `Composing… ${pct}%`) : "Generate music"}
        </button>
        {busy && (
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-fuchsia-400 transition-all duration-500" style={{ width: `${Math.max(4, pct)}%` }} />
          </div>
        )}
      </div>
    </div>
  );
}
