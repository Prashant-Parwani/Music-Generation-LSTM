"""
STEP 2: Tokenize the filtered MIDI files and create train / val / test splits.

Run (after step1_scan_filter.py):
    python step2_tokenize.py

What it does
  1. reads data/processed/manifest.csv (files that passed the filters)
  2. merges every song into 5 tracks (drums, piano, guitar, bass, strings)
  3. tokenizes with MidiTok REMI (+ Program tokens, so one stream holds all instruments)
  4. splits BY ARTIST into train/val/test (avoids near-duplicate songs leaking across splits)
  5. computes density bin edges (note events per bar) from the TRAIN split
  6. saves tokenizer.json, controls.json and tokens/{train,val,test}.pkl
"""
import argparse
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from common import ControlSpec, build_tokenizer, merge_to_groups

_tok = None


def _init_worker():
    global _tok
    _tok = build_tokenizer()


def _tokenize(args):
    path, tempo = args
    try:
        from symusic import Score
        score = merge_to_groups(Score(path))
        if len(score.tracks) == 0:
            return None
        seq = _tok(score)
        seq = seq[0] if isinstance(seq, list) else seq
        ids = np.asarray(seq.ids, dtype=np.int32)
        return path, ids, float(tempo)
    except Exception:
        return None


def split_by_artist(paths, seed):
    artists = sorted({Path(p).parent.name for p in paths})
    random.Random(seed).shuffle(artists)
    n = len(artists)
    n_val = max(1, round(n * C.SPLIT_FRACTIONS[1])) if n >= 3 else 0
    n_test = max(1, round(n * C.SPLIT_FRACTIONS[2])) if n >= 3 else 0
    val, test = set(artists[:n_val]), set(artists[n_val:n_val + n_test])
    out = {}
    for p in paths:
        a = Path(p).parent.name
        out[p] = "val" if a in val else "test" if a in test else "train"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only use first N filtered songs (testing)")
    args = ap.parse_args()

    if not C.MANIFEST_CSV.exists():
        raise SystemExit("manifest.csv not found - run step1_scan_filter.py first.")
    df = pd.read_csv(C.MANIFEST_CSV)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} songs passed step 1. Tokenizing...")

    tok = build_tokenizer()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(C.TOKENIZER_PATH)
    print(f"Base vocabulary size: {len(tok.vocab)}")

    jobs = list(zip(df["path"], df["tempo"]))
    workers = C.NUM_WORKERS or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        results = list(tqdm(ex.map(_tokenize, jobs, chunksize=8), total=len(jobs)))
    results = [r for r in results if r is not None and len(r[1]) >= C.MIN_TOKENS_PER_SONG]
    print(f"{len(results)} songs tokenized successfully (>= {C.MIN_TOKENS_PER_SONG} tokens).")

    spec = ControlSpec.build(tok)
    split_of = split_by_artist([r[0] for r in results], C.SEED)
    data = {"train": [], "val": [], "test": []}
    for path, ids, tempo in results:
        data[split_of[path]].append((path, ids, tempo))

    # density edges from TRAIN songs (quartiles of note-events-per-bar)
    npb = [spec.notes_per_bar(ids) for _, ids, _ in data["train"]]
    qs = np.quantile(npb, np.linspace(0, 1, C.N_DENS_BINS + 1)[1:-1])
    spec = spec.with_dens_edges(qs)
    spec.save()
    print("Density bin edges (note events / bar):", [round(float(q), 1) for q in qs])

    C.TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in data.items():
        out = {
            "paths": [p for p, _, _ in items],
            "ids": [ids for _, ids, _ in items],
            "tempo": [t for _, _, t in items],
            "bars": [np.where(ids == spec.bar_id)[0] for _, ids, _ in items],
        }
        with open(C.TOKENS_DIR / f"{split}.pkl", "wb") as f:
            pickle.dump(out, f)
        n_tok = sum(len(i) for i in out["ids"])
        print(f"{split:5s}: {len(items):5d} songs, {n_tok:,} tokens")

    print(f"\nModel vocabulary size (music + {len(spec.control_ids)} control tokens): {spec.vocab_size}")
    print(f"Saved: {C.TOKENIZER_PATH.name}, {C.CONTROLS_JSON.name}, tokens/*.pkl in {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
