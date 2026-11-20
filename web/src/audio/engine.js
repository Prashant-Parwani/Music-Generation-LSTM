import * as Tone from "tone";

/**
 * Browser synthesiser: plays the generated notes with Tone.js (no samples / soundfonts to download, works offline).
 * Notes arrive from the API already in seconds: { group: [[start, end, pitch, velocity], ...] }
 */
const midiToFreq = (p) => 440 * 2 ** ((p - 69) / 12);

function polyVoice(options, cutoff, baseDb) {
  const synth = new Tone.PolySynth(Tone.Synth, options);
  const filter = new Tone.Filter(cutoff, "lowpass");
  const out = new Tone.Volume(baseDb);
  synth.chain(filter, out);
  return { kind: "poly", synth, out, baseDb };
}

function drumVoice(baseDb) {
  const out = new Tone.Volume(baseDb);
  const membrane = (octaves, decay, pitchDecay) =>
    new Tone.MembraneSynth({ pitchDecay, octaves, envelope: { attack: 0.001, decay, sustain: 0, release: 0.1 } }).connect(out);
  const noise = (decay, type, freq) => {
    const s = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay, sustain: 0, release: 0.02 } });
    s.chain(new Tone.Filter(freq, type), out);
    return s;
  };
  return {
    kind: "drums", out, baseDb,
    kit: {
      kick: membrane(6, 0.32, 0.05), tom: membrane(3, 0.26, 0.03),
      snare: noise(0.17, "bandpass", 1900), clap: noise(0.12, "bandpass", 1300),
      hat: noise(0.045, "highpass", 8000), openHat: noise(0.24, "highpass", 7000),
      cymbal: noise(0.9, "highpass", 5200), ride: noise(0.4, "highpass", 6500),
    },
  };
}

// General-MIDI drum note -> voice
function drumTarget(p) {
  if (p === 35 || p === 36) return ["kick", 52];
  if (p === 38 || p === 40 || p === 37) return ["snare"];
  if (p === 39) return ["clap"];
  if (p === 42 || p === 44) return ["hat"];
  if (p === 46) return ["openHat"];
  if ([41, 43, 45, 47, 48, 50].includes(p)) return ["tom", 70 + (p - 41) * 14];
  if ([49, 52, 55, 57].includes(p)) return ["cymbal"];
  if ([51, 53, 56, 59].includes(p)) return ["ride"];
  return ["hat"];
}

export class MusicEngine {
  constructor() {
    this.ready = false;
    this.voices = {};
    this.part = null;
    this.duration = 0;
    this.loop = false;
  }

  /** Call this synchronously inside a click handler: browsers only allow audio after a user gesture. */
  unlock() { return Tone.start(); }

  async init() {
    if (this.ready) return;
    await Tone.start();
    const master = new Tone.Limiter(-1).toDestination();
    const reverb = new Tone.Freeverb({ roomSize: 0.55, dampening: 3500, wet: 0.16 }).connect(master);
    this.voices = {
      drums: drumVoice(-5),
      bass: polyVoice({ maxPolyphony: 8, oscillator: { type: "sawtooth" },
        envelope: { attack: 0.01, decay: 0.25, sustain: 0.5, release: 0.25 } }, 520, -3),
      piano: polyVoice({ maxPolyphony: 32, oscillator: { type: "triangle8" },
        envelope: { attack: 0.004, decay: 0.45, sustain: 0.15, release: 1.1 } }, 5000, -9),
      guitar: polyVoice({ maxPolyphony: 24, oscillator: { type: "fatsawtooth", count: 2, spread: 12 },
        envelope: { attack: 0.004, decay: 0.3, sustain: 0.12, release: 0.5 } }, 2800, -13),
      strings: polyVoice({ maxPolyphony: 24, oscillator: { type: "fatsawtooth", count: 3, spread: 25 },
        envelope: { attack: 0.18, decay: 0.3, sustain: 0.8, release: 1.0 } }, 2600, -15),
    };
    Object.values(this.voices).forEach((v) => v.out.connect(reverb));
    this.ready = true;
  }

  _trigger(ev, time) {
    const v = this.voices[ev.group];
    if (!v) return;
    try {
      if (v.kind === "drums") {
        const [name, freq] = drumTarget(ev.pitch);
        const s = v.kit[name];
        if (freq) s.triggerAttackRelease(freq, 0.15, time, ev.vel);
        else s.triggerAttackRelease(0.15, time, ev.vel);
      } else {
        v.synth.triggerAttackRelease(midiToFreq(ev.pitch), Math.max(0.06, ev.dur), time, ev.vel);
      }
    } catch {
      /* two notes at exactly the same instant on a monophonic drum voice: ignore */
    }
  }

  async load(tracks, duration) {
    await this.init();
    this.stop();
    if (this.part) this.part.dispose();
    const events = [];
    for (const [group, rows] of Object.entries(tracks)) {
      for (const [s, e, p, v] of rows) events.push([s, { group, pitch: p, dur: e - s, vel: Math.min(1, Math.max(0.05, v / 127)) }]);
    }
    events.sort((a, b) => a[0] - b[0]);
    this.part = new Tone.Part((time, ev) => this._trigger(ev, time), events).start(0);
    this.duration = duration;
    this._applyLoop();
  }

  _applyLoop() {
    const t = Tone.getTransport();
    t.loop = this.loop;
    t.loopStart = 0;
    t.loopEnd = this.duration + 0.3;
  }

  setLoop(on) { this.loop = on; this._applyLoop(); }

  async play() {
    await Tone.start();
    const t = Tone.getTransport();
    if (t.seconds >= this.duration) t.seconds = 0;
    t.start();
  }

  pause() { Tone.getTransport().pause(); this._releaseAll(); }

  stop() {
    const t = Tone.getTransport();
    t.stop();
    t.seconds = 0;
    this._releaseAll();
  }

  _releaseAll() {
    Object.values(this.voices).forEach((v) => v.kind === "poly" && v.synth.releaseAll());
  }

  seek(seconds) { Tone.getTransport().seconds = Math.max(0, Math.min(seconds, this.duration)); }
  getTime() { return Tone.getTransport().seconds; }
  isFinished() { return !this.loop && this.getTime() > this.duration + 0.4; }
  setMuted(group, muted) { if (this.voices[group]) this.voices[group].out.mute = muted; }
  setVolume(group, relDb) {
    const v = this.voices[group];
    if (v) v.out.volume.value = v.baseDb + relDb;
  }
}
