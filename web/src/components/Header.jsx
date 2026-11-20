import { Github, Music2 } from "lucide-react";
import { APP_NAME, GITHUB_URL } from "../config.js";
import StatusBadge from "./StatusBadge.jsx";

export default function Header({ server }) {
  return (
    <header className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-5">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-indigo-500/30">
        <Music2 size={20} className="text-white" />
      </div>
      <span className="text-lg font-semibold tracking-tight">{APP_NAME}</span>
      <div className="ml-auto flex items-center gap-3">
        <StatusBadge {...server} />
        {GITHUB_URL && (
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub"
            className="rounded-lg p-2 text-slate-400 transition hover:bg-white/10 hover:text-white"><Github size={18} /></a>
        )}
      </div>
    </header>
  );
}
