"""
PyTorch Dataset: random bar-aligned crops of songs, each prefixed with its control tokens.

Every training example =  [BOS, tempo, density, 5 x instrument ON/OFF]  +  crop of music tokens
The controls are computed from the crop itself (so the labels are always truthful):
  * instrument ON/OFF : does the group play >= C.MIN_NOTES_FOR_ON notes in the crop?
  * density bin       : note events per bar in the crop
  * tempo bin         : median tempo of the whole song

Training-only extras (never applied to val/test, so validation numbers stay clean):
  * PITCH TRANSPOSITION: every crop is shifted by a random number of semitones in `transpose`
    (default -5..+6). Only pitched notes move; drum tokens are untouched. Notes that would leave the
    tokenizer's pitch range are folded back by an octave. Controls do not depend on pitch, so they stay valid.
  * CONTROL DROPOUT (for classifier-free guidance): with probability `cfg_drop` the 7 control tokens are
    replaced by the MASK token, which teaches the model an "unconditional" mode.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from common import PREFIX_LEN, ControlSpec


class MusicDataset(Dataset):
    def __init__(self, split="train", seq_len=1024, crops_per_song=8, train=True, spec=None,
                 transpose=(-5, 6), cfg_drop=0.0):
        self.spec = spec or ControlSpec.load()
        with open(C.TOKENS_DIR / f"{split}.pkl", "rb") as f:
            d = pickle.load(f)
        self.ids, self.bars, self.tempo = d["ids"], d["bars"], d["tempo"]
        if len(self.ids) == 0:
            raise RuntimeError(f"Split '{split}' is empty. Use more songs or adjust C.SPLIT_FRACTIONS.")
        self.seq_len, self.crops, self.train = seq_len, crops_per_song, train
        self.music_len = seq_len - PREFIX_LEN

        self.transpose = tuple(transpose) if (train and transpose) else None
        self.cfg_drop = float(cfg_drop) if train else 0.0
        self.shift_tables = self._build_shift_tables(*self.transpose) if self.transpose else {}

    def _build_shift_tables(self, lo, hi):
        """For each shift k: an array mapping token id -> transposed token id (identity for non-pitch tokens)."""
        pv = self.spec._pitch_val
        p2id = {int(p): i for i, p in enumerate(pv) if p >= 0}
        pmin, pmax = min(p2id), max(p2id)
        tables = {}
        for k in range(lo, hi + 1):
            if k == 0:
                continue
            t = np.arange(self.spec.base, dtype=np.int64)
            for p, i in p2id.items():
                q = p + k
                while q < pmin:
                    q += 12
                while q > pmax:
                    q -= 12
                t[i] = p2id.get(q, i)
            tables[k] = t
        return tables

    def __len__(self):
        return len(self.ids) * self.crops

    def __getitem__(self, idx):
        spec, L = self.spec, self.music_len
        song = idx % len(self.ids)
        rng = np.random.default_rng() if self.train else np.random.default_rng(idx)
        ids, bars = self.ids[song], self.bars[song]

        # pick a bar-aligned start; 10% of the time pick one near the end so the model sees EOS
        cand = bars[bars <= max(0, len(ids) - L)]
        if len(cand) == 0:
            cand = bars[:1] if len(bars) else np.array([0])
        start = int(cand[rng.integers(len(cand))])
        tail = bars[bars >= len(ids) - L]
        if len(tail) and rng.random() < 0.1:
            start = int(tail[0])

        crop = ids[start:start + L].astype(np.int64)
        if len(crop) < L:
            crop = np.concatenate([crop, [spec.eos_id]])
            crop = np.concatenate([crop, np.full(L - len(crop), spec.pad_id)])[:L]

        flags, npb = spec.crop_controls(crop)           # computed BEFORE transposing (they don't depend on pitch)

        if self.shift_tables:
            k = int(rng.integers(self.transpose[0], self.transpose[1] + 1))
            if k != 0:
                crop = self.shift_tables[k][crop]

        if self.cfg_drop > 0 and rng.random() < self.cfg_drop:
            prefix = [spec.bos_id] + [spec.mask_id] * (PREFIX_LEN - 1)     # "no controls" example
        else:
            prefix = spec.prefix(spec.tempo_bin(self.tempo[song]), spec.dens_bin(npb), flags)
        return torch.from_numpy(np.concatenate([prefix, crop]).astype(np.int64))