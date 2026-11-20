import { Binary, Brain, Library, SlidersHorizontal } from "lucide-react";

const CARDS = [
  { icon: Binary, title: "Music as a language",
    text: "Every song is turned into a sequence of tokens (bar, position, instrument, pitch, loudness, length), just like words in a sentence. The networks learn to predict the next token." },
  { icon: SlidersHorizontal, title: "Controls are part of the prompt",
    text: "Each piece starts with eight control tokens: tempo, density and an on/off switch for every instrument. During training the networks always saw these first, so they learned to follow them." },
  { icon: Brain, title: "Two different brains",
    text: "An LSTM reads the music step by step and keeps a running memory. A Transformer looks back over everything at once with attention, which helps with structure. Try both with the same settings." },
  { icon: Library, title: "What it learned from",
    text: "About 7,700 songs from the Clean MIDI subset of the Lakh MIDI dataset (4/4 time), merged into five instrument groups: drums, bass, piano, guitar and strings." },
];

export default function HowItWorks() {
  return (
    <section className="mt-16" aria-labelledby="how">
      <h2 id="how" className="text-2xl font-semibold tracking-tight">How it works</h2>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {CARDS.map(({ icon: Icon, title, text }) => (
          <div key={title} className="glass rounded-2xl p-5">
            <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-300"><Icon size={18} /></div>
            <h3 className="font-semibold text-slate-100">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{text}</p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Good to know: notes are written in beats and bars, so the tempo slider changes how fast the piece is played back.
        The sound you hear comes from simple built-in synthesizers; download the MIDI file to play it with real instruments in any music software.
      </p>
    </section>
  );
}
