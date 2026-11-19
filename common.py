"""
Shared helpers used by every step:
  * building / loading the MidiTok tokenizer
  * merging raw MIDI tracks into our 5 instrument groups
  * the ControlSpec: control tokens (tempo, density, instrument ON/OFF) + token statistics
  * converting generated token ids back to a MIDI file

Token layout seen by the models
-------------------------------
  [ BOS, TEMPO_bin, DENSITY_bin, drums_ON/OFF, piano_ON/OFF, guitar_ON/OFF, bass_ON/OFF, strings_ON/OFF,
    <music tokens ... (REMI)> ]
The first PREFIX_LEN = 8 tokens are the "control prefix". Control token ids sit AFTER the
MidiTok vocabulary (ids >= base_vocab), so we never have to modify MidiTok's own vocab.
"""
import json
from pathlib import Path

import numpy as np

import config as C

PREFIX_LEN = 1 + 1 + 1 + len(C.ALL_GROUPS)   # BOS + tempo + density + one flag per group = 8
CTX_LEN = PREFIX_LEN - 1                     # control tokens only (without BOS)
N_TEMPO_BINS = len(C.TEMPO_EDGES) + 1
N_DENS_BINS = C.N_DENS_BINS
N_CONTROL_TOKENS = N_TEMPO_BINS + N_DENS_BINS + 2 * len(C.ALL_GROUPS)


# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------
def build_tokenizer():
    from miditok import REMI, TokenizerConfig
    return REMI(TokenizerConfig(**C.TOKENIZER_KWARGS))


def load_tokenizer(path=None):
    from miditok import REMI
    return REMI(params=Path(path or C.TOKENIZER_PATH))


# ----------------------------------------------------------------------------
# Merge a raw score into <=5 tracks (one per instrument group)
# ----------------------------------------------------------------------------
def _group_of_program(program: int):
    for name, rng in C.INSTRUMENT_GROUPS.items():
        if program in rng:
            return name
    return None


def merge_to_groups(score):
    """Return a new symusic Score with one track per instrument group (drums/piano/guitar/bass/strings)."""
    from symusic import Score, Track
    from symusic.core import NoteTickList

    new = Score(score.ticks_per_quarter)
    for t in score.tempos:
        new.tempos.append(t)
    for ts in score.time_signatures:
        new.time_signatures.append(ts)

    buckets = {g: [] for g in C.ALL_GROUPS}
    for tr in score.tracks:
        if len(tr.notes) == 0:
            continue
        g = "drums" if tr.is_drum else _group_of_program(tr.program)
        if g is None:
            continue                      # brass, synth leads, etc. are ignored
        buckets[g].append(tr.notes.numpy())

    for g, parts in buckets.items():
        if not parts:
            continue
        time = np.concatenate([p["time"] for p in parts])
        dur = np.concatenate([p["duration"] for p in parts])
        pitch = np.concatenate([p["pitch"] for p in parts])
        vel = np.concatenate([p["velocity"] for p in parts])
        keep = dur > 0
        time, dur, pitch, vel = time[keep], dur[keep], pitch[keep], vel[keep]
        order = np.lexsort((pitch, time))
        trk = Track(program=C.GROUP_PROGRAM[g], is_drum=(g == "drums"), name=g)
        trk.notes = NoteTickList.from_numpy(
            np.ascontiguousarray(time[order], dtype=np.int32),
            np.ascontiguousarray(dur[order], dtype=np.int32),
            np.ascontiguousarray(pitch[order], dtype=np.int8),
            np.ascontiguousarray(vel[order], dtype=np.int8))
        new.tracks.append(trk)
    return new


# ----------------------------------------------------------------------------
# ControlSpec
# ----------------------------------------------------------------------------
class ControlSpec:
    """Everything about token ids + control tokens. Saved as data/processed/controls.json."""

    def __init__(self, d: dict):
        self.d = d
        self.base = int(d["base_vocab"])
        self.pad_id, self.bos_id = int(d["pad_id"]), int(d["bos_id"])
        self.eos_id, self.mask_id = int(d["eos_id"]), int(d["mask_id"])
        self.bar_id = int(d["bar_id"])
        self.tempo_edges = list(d["tempo_edges"])
        self.dens_edges = list(d["dens_edges"])
        self.groups = list(d["groups"])
        self.vocab_size = self.base + N_CONTROL_TOKENS

        # lookup tables over the base vocabulary
        self._is_note = np.zeros(self.base, dtype=bool)
        self._prog_group = np.full(self.base, -1, dtype=np.int64)
        self._pitch_val = np.full(self.base, -1, dtype=np.int64)
        self._tempo_val = np.full(self.base, np.nan, dtype=np.float64)
        for i in d["note_ids"]:
            self._is_note[int(i)] = True
        for i, g in d["program_group"].items():
            self._prog_group[int(i)] = int(g)
        for i, p in d["pitch_values"].items():
            self._pitch_val[int(i)] = int(p)
        for i, bpm in d["tempo_values"].items():
            self._tempo_val[int(i)] = float(bpm)

    # ---- construction / io ----
    @classmethod
    def build(cls, tok, dens_edges=(0.0, 0.0, 0.0)):
        v = tok.vocab
        note_ids, pitch_values, program_group, tempo_values = [], {}, {}, {}
        for name, i in v.items():
            if name.startswith("Pitch_"):
                note_ids.append(i); pitch_values[str(i)] = int(name.split("_")[1])
            elif name.startswith("PitchDrum_"):
                note_ids.append(i)
            elif name.startswith("Program_"):
                p = int(name.split("_")[1])
                g = "drums" if p == -1 else next(
                    (gn for gn in C.INSTRUMENT_GROUPS if C.GROUP_PROGRAM[gn] == p), None)
                if g is not None:
                    program_group[str(i)] = C.ALL_GROUPS.index(g)
            elif name.startswith("Tempo_"):
                tempo_values[str(i)] = float(name.split("_")[1])
        d = dict(
            base_vocab=len(v), pad_id=v["PAD_None"], bos_id=v["BOS_None"], eos_id=v["EOS_None"],
            mask_id=v["MASK_None"], bar_id=v["Bar_None"], note_ids=note_ids,
            pitch_values=pitch_values, program_group=program_group, tempo_values=tempo_values,
            tempo_edges=C.TEMPO_EDGES, dens_edges=list(dens_edges), groups=C.ALL_GROUPS)
        return cls(d)

    def save(self, path=None):
        Path(path or C.CONTROLS_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(path or C.CONTROLS_JSON, "w") as f:
            json.dump(self.d, f)

    @classmethod
    def load(cls, path=None):
        with open(path or C.CONTROLS_JSON) as f:
            return cls(json.load(f))

    def with_dens_edges(self, edges):
        d = dict(self.d); d["dens_edges"] = [float(e) for e in edges]
        return ControlSpec(d)

    # ---- control token ids ----
    def tempo_token(self, b):          return self.base + int(b)
    def dens_token(self, b):           return self.base + N_TEMPO_BINS + int(b)
    def inst_token(self, g_idx, on):   return self.base + N_TEMPO_BINS + N_DENS_BINS + 2 * g_idx + (0 if on else 1)

    @property
    def control_ids(self):
        return list(range(self.base, self.vocab_size))

    def prefix(self, tempo_bin, dens_bin, flags):
        """flags: iterable of 5 bools, in C.ALL_GROUPS order."""
        ids = [self.bos_id, self.tempo_token(tempo_bin), self.dens_token(dens_bin)]
        ids += [self.inst_token(i, bool(f)) for i, f in enumerate(flags)]
        assert len(ids) == PREFIX_LEN
        return ids

    # ---- binning ----
    def tempo_bin(self, bpm):
        return int(np.searchsorted(self.tempo_edges, bpm, side="right"))

    def dens_bin(self, notes_per_bar):
        return int(np.searchsorted(self.dens_edges, notes_per_bar, side="right"))

    # ---- statistics computed from music token ids ----
    def _current_group(self, ids):
        pg = self._prog_group[ids]
        idx = np.where(pg >= 0, np.arange(len(ids)), -1)
        idx = np.maximum.accumulate(idx)
        return np.where(idx >= 0, pg[np.maximum(idx, 0)], -1)

    def group_counts(self, ids):
        """Number of note events per instrument group."""
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.base)]
        if len(ids) == 0:
            return np.zeros(len(C.ALL_GROUPS), dtype=np.int64)
        cur = self._current_group(ids)
        notes = self._is_note[ids] & (cur >= 0)
        return np.bincount(cur[notes], minlength=len(C.ALL_GROUPS))

    def notes_per_bar(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.base)]
        n_notes = int(self._is_note[ids].sum())
        n_bars = int((ids == self.bar_id).sum())
        return n_notes / max(1, n_bars)

    def crop_controls(self, ids):
        """-> (flags[5] bool, notes_per_bar)"""
        counts = self.group_counts(ids)
        return counts >= C.MIN_NOTES_FOR_ON, self.notes_per_bar(ids)

    def median_tempo(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.base)]
        t = self._tempo_val[ids]
        t = t[~np.isnan(t)]
        return float(np.median(t)) if len(t) else None

    def pitch_class_hist(self, ids):
        """Normalised 12-bin histogram of pitched (non-drum) notes."""
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.base)]
        pv = self._pitch_val[ids]
        pv = pv[pv >= 0]
        h = np.bincount(pv % 12, minlength=12).astype(float)
        return h / h.sum() if h.sum() > 0 else h

    def program_ids_of_group(self, g_idx):
        return [int(i) for i, g in self.d["program_group"].items() if int(g) == g_idx]

    @property
    def banned_ids(self):
        """Tokens that must never be generated as music."""
        return [self.pad_id, self.bos_id, self.mask_id, self.eos_id] + self.control_ids


# ----------------------------------------------------------------------------
# Token ids -> MIDI
# ----------------------------------------------------------------------------
def _make_tempo(bpm):
    """Build a symusic Tempo at tick 0. The constructor differs between symusic versions, so try each and verify."""
    from symusic import Tempo
    builders = (lambda: Tempo(0, float(bpm)),
                lambda: Tempo.from_qpm(0, float(bpm)),
                lambda: Tempo(0, int(round(60_000_000 / bpm))))
    for build in builders:
        try:
            t = build()
            if abs(float(t.qpm) - bpm) < 1.0:
                return t
        except Exception:
            continue
    return None


def ids_to_score(ids, tok, spec: ControlSpec, tempo_bpm=None):
    """Decode generated music token ids to a symusic Score (control/special tokens are stripped)."""
    music = [int(i) for i in ids if 0 <= int(i) < spec.base
             and int(i) not in (spec.pad_id, spec.bos_id, spec.eos_id, spec.mask_id)]
    score = tok.decode(music)
    if sum(len(t.notes) for t in score.tracks) == 0:
        raise ValueError("The generated token sequence contains no valid notes "
                         "(model is under-trained or the settings are too extreme). Try another seed.")
    if tempo_bpm:
        # Music tokens are measured in beats/bars, so tempo is a playback setting: apply the requested BPM.
        tempo = _make_tempo(float(tempo_bpm))
        if tempo is not None:
            score.tempos.clear()
            score.tempos.append(tempo)
    return score


def ids_to_midi(ids, tok, spec: ControlSpec, path, tempo_bpm=None):
    score = ids_to_score(ids, tok, spec, tempo_bpm)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(path))
    return score