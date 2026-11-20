import { History as HistoryIcon } from "lucide-react";
import { INSTRUMENTS, INSTRUMENT_ORDER } from "../config.js";

export default function History({ items, activeId, onPick }) {
  if (!items.length) return null;
  return (
    <div className="glass rounded-2xl p-5">
      <h3 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        <HistoryIcon size={14} /> This session
      </h3>
      <ul className="flex flex-col gap-1.5">
        {items.map((it) => (
          <li key={it.id}>
            <button type="button" onClick={() => onPick(it)}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
                activeId === it.id ? "bg-indigo-500/20 text-white" : "text-slate-300 hover:bg-white/5"}`}>
              <span className="w-24 shrink-0 truncate">{it.result.model === "lstm" ? "LSTM" : "Transformer"}</span>
              <span className="text-slate-400">{it.result.bpm} BPM</span>
              <span className="ml-auto flex gap-1">
                {INSTRUMENT_ORDER.filter((g) => it.result.tracks[g]).map((g) => (
                  <span key={g} className="h-2 w-2 rounded-full" style={{ background: INSTRUMENTS[g].color }} title={INSTRUMENTS[g].label} />
                ))}
              </span>
              <span className="w-10 text-right text-xs text-slate-500">{it.result.duration}s</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
