import { APP_NAME, GITHUB_URL, MODEL_REPO_URL } from "../config.js";

export default function Footer() {
  return (
    <footer className="mx-auto mt-16 max-w-6xl border-t border-white/10 px-4 py-8 text-xs leading-relaxed text-slate-500">
      <p>
        {APP_NAME} is an educational project. Models trained on the Clean MIDI subset of the Lakh MIDI Dataset (Colin Raffel).
        Built with PyTorch, MidiTok, FastAPI, React and Tone.js. Generated clips are short and the models are small, so expect experiments, not hits.
      </p>
      <p className="mt-2 flex gap-4">
        {GITHUB_URL && <a className="underline hover:text-slate-300" href={GITHUB_URL} target="_blank" rel="noreferrer">Source code</a>}
        {MODEL_REPO_URL && <a className="underline hover:text-slate-300" href={MODEL_REPO_URL} target="_blank" rel="noreferrer">Models on Hugging Face</a>}
      </p>
    </footer>
  );
}
