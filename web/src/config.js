// ---- Branding: change these freely -------------------------------------------------------------
export const APP_NAME = "AI Music Studio";
export const TAGLINE = "Pick the instruments, set the mood, and let two neural networks compose for you.";
export const GITHUB_URL = ""; // e.g. "https://github.com/your-name/music-gen" (hidden if empty)
export const MODEL_REPO_URL = ""; // e.g. "https://huggingface.co/your-name/music-gen-lstm-transformer" (hidden if empty)

// ---- Backend address ------------------------------------------------------------------------------
// "" = same server (server.py serves this website) or the Vite proxy in dev.
// For a separate deployment set VITE_API_URL, e.g. https://your-name-music-gen.hf.space
export const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

// ---- Instruments ----------------------------------------------------------------------------------
export const INSTRUMENTS = {
  drums:   { label: "Drums",   emoji: "🥁", color: "#f59e0b" },
  bass:    { label: "Bass",    emoji: "🎸", color: "#10b981" },
  piano:   { label: "Piano",   emoji: "🎹", color: "#38bdf8" },
  guitar:  { label: "Guitar",  emoji: "🎸", color: "#fb7185" },
  strings: { label: "Strings", emoji: "🎻", color: "#a78bfa" },
};
export const INSTRUMENT_ORDER = ["drums", "bass", "piano", "guitar", "strings"];

// ---- One-click presets ----------------------------------------------------------------------------
export const PRESETS = [
  { name: "Chill lo-fi",    emoji: "☕", bpm: 78,  density: 1, instruments: ["drums", "bass", "piano"] },
  { name: "Rock band",      emoji: "🤘", bpm: 132, density: 2, instruments: ["drums", "bass", "guitar"] },
  { name: "Dance floor",    emoji: "🪩", bpm: 126, density: 3, instruments: ["drums", "bass", "piano", "guitar"] },
  { name: "Cinematic",      emoji: "🎬", bpm: 88,  density: 1, instruments: ["strings", "piano", "bass"] },
  { name: "Full orchestra", emoji: "🎼", bpm: 104, density: 2, instruments: ["drums", "bass", "piano", "guitar", "strings"] },
];
