// Minimal stand-in for Tone.js so the React flow can run in jsdom (no Web Audio there).
const calls = { triggers: 0, started: 0, parts: 0 };
globalThis.__toneCalls = calls;
const node = () => ({ connect() { return this; }, chain() { return this; }, toDestination() { return this; }, triggerAttackRelease() { calls.triggers++; }, releaseAll() {}, mute: false, volume: { value: 0 } });
export const start = async () => { calls.started++; };
export const Limiter = function () { return node(); };
export const Freeverb = function () { return node(); };
export const Filter = function () { return node(); };
export const Volume = function () { return node(); };
export const PolySynth = function () { return node(); };
export const Synth = function () {};
export const MembraneSynth = function () { return node(); };
export const NoiseSynth = function () { return node(); };
let seconds = 0;
const transport = { loop: false, loopStart: 0, loopEnd: 0, get seconds() { return seconds; }, set seconds(v) { seconds = v; }, start() {}, pause() {}, stop() {} };
export const getTransport = () => transport;
export class Part { constructor(cb, events) { calls.parts++; this.events = events; this.cb = cb; } start() { return this; } dispose() {} }
