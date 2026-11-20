import { Loader2 } from "lucide-react";

export default function StatusBadge({ state, seconds, device, queue }) {
  const map = {
    online: { dot: "bg-emerald-400", text: `Online${device ? ` · ${device === "cuda" ? "GPU" : "CPU"}` : ""}${queue ? ` · ${queue} in queue` : ""}` },
    connecting: { dot: "bg-amber-400 animate-pulse", text: "Connecting…" },
    waking: { dot: "bg-amber-400 animate-pulse", text: `Waking the server up… ${seconds}s` },
    offline: { dot: "bg-rose-500", text: "Server offline" },
  };
  const s = map[state] || map.connecting;
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
      {state === "waking" ? <Loader2 size={12} className="animate-spin text-amber-400" /> : <span className={`h-2 w-2 rounded-full ${s.dot}`} />}
      {s.text}
    </span>
  );
}
