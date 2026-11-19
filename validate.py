"""
VALIDATION: is the model actually good? Produces a small report you can read (or send to someone) without sharing the models.

    python validate.py                          # ~60 generated clips per model
    python validate.py --n-samples 100          # tighter error bars (slower)

It answers 5 questions
  1. Did training converge, and is there overfitting?            (reads checkpoints/*_log.csv)
  2. Do the models obey the controls?                            (instrument / density accuracy with 95% confidence intervals)
  3. Does generated music look like REAL music?                  (same metrics measured on real test-set clips = the baseline)
        key clarity, in-key note ratio, repetition between bars, notes per bar
  4. Is the model copying the training data?                     (bar-level overlap with the training set, vs the real-music baseline)
  5. Does it SOUND good?                                         (blind listening test files + a rating sheet are created for you)

Outputs: results/validation_report.md, results/validation_report.json, results/listening_test/
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as C
from common import PREFIX_LEN, ControlSpec, ids_to_midi, load_tokenizer
from dataset import MusicDataset
from generate import generate_ids, load_model

KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR, MINOR = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/max(1,n):.0f}% (95% CI {100*lo:.0f}-{100*hi:.0f}%)"


def clean(ids, spec):
    ids = np.asarray(ids, dtype=np.int64)
    bad = [spec.pad_id, spec.bos_id, spec.eos_id, spec.mask_id]
    return ids[(ids >= 0) & (ids < spec.base) & ~np.isin(ids, bad)]


def key_stats(hist):
    if hist.sum() == 0 or np.std(hist) == 0:
        return float("nan"), float("nan")
    best, arg = -2, None
    for k in range(12):
        for name, prof in (("maj", KS_MAJ), ("min", KS_MIN)):
            r = np.corrcoef(hist, np.roll(prof, k))[0, 1]
            if r > best:
                best, arg = r, (k, name)
    k, name = arg
    scale = MAJOR if name == "maj" else MINOR
    return best, float(sum(hist[(k + s) % 12] for s in scale) / hist.sum())


def clip_metrics(ids, spec):
    ids = clean(ids, spec)
    pos = np.where(ids == spec.bar_id)[0]
    bars = [ids[a:b] for a, b in zip(pos, list(pos[1:]) + [len(ids)])]
    full = bars[:-1] if len(bars) > 1 else bars                    # ignore the cut-off last bar
    nonempty = [b for b in full if spec._is_note[b].sum() > 0]
    h = spec.pitch_class_hist(ids)
    kc, ik = key_stats(h)
    return dict(
        notes_per_bar=spec.notes_per_bar(ids),
        n_bars=len(bars),
        key_corr=kc, in_key=ik,
        bar_repeat=(1 - len({b.tobytes() for b in nonempty}) / len(nonempty)) if len(nonempty) >= 2 else float("nan"),
        empty_bars=(1 - len(nonempty) / max(1, len(full))),
    )


def summarize(rows):
    df = pd.DataFrame(rows)
    return {c: (float(np.nanmean(df[c])), float(np.nanstd(df[c]))) for c in df.columns}


def bar_hashes(ids, spec, W=48, min_notes=6):
    ids = np.asarray(ids, dtype=np.int16)
    out = []
    for p in np.where(ids == spec.bar_id)[0]:
        w = ids[p:p + W]
        if len(w) == W and spec._is_note[w].sum() >= min_notes:
            out.append(hash(w.tobytes()))
    return out


def build_train_index(spec, W=48):
    with open(C.TOKENS_DIR / "train.pkl", "rb") as f:
        d = pickle.load(f)
    index = set()
    for ids, bars in tqdm(zip(d["ids"], d["bars"]), total=len(d["ids"]), desc="indexing training bars"):
        for p in bars:
            w = ids[p:p + W]
            if len(w) == W and spec._is_note[w].sum() >= 6:
                index.add(hash(w.tobytes()))
    return index


def copy_rate(clips, index, spec):
    hit = tot = 0
    for ids in clips:
        hs = bar_hashes(ids, spec)
        tot += len(hs); hit += sum(h in index for h in hs)
    return hit, tot


def training_summary():
    out = {}
    for log in sorted(C.CHECKPOINT_DIR.glob("*_log.csv")):
        df = pd.read_csv(log)
        if df.empty:
            continue
        best = df.loc[df["val_loss"].idxmin()]
        last3 = df["val_loss"].tail(3).tolist()
        out[log.stem.replace("_log", "")] = dict(
            epochs=int(len(df)), best_epoch=int(best["epoch"]), best_val_loss=float(best["val_loss"]),
            final_train_loss=float(df["train_loss"].iloc[-1]), final_val_loss=float(df["val_loss"].iloc[-1]),
            gap_val_minus_train=float(df["val_loss"].iloc[-1] - df["train_loss"].iloc[-1]),
            last3_val=last3, still_improving=bool(len(last3) >= 2 and last3[-1] < last3[0] - 0.005))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--listening-clips", type=int, default=5, help="clips per source for the blind listening test")
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec, tok = ControlSpec.load(), load_tokenizer()
    ckpts = sorted(C.CHECKPOINT_DIR.glob("*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints found. Train first.")
    report = {"settings": vars(args), "training": training_summary(), "models": {}}

    # ---- real test-set clips = baseline + the controls we ask the models for ----
    ds = MusicDataset("test", args.seq_len, crops_per_song=4, train=False, spec=spec)
    rng = np.random.default_rng(C.SEED)
    real, controls = [], []
    for idx in rng.integers(0, len(ds), size=args.n_samples):
        x = ds[int(idx)].numpy()
        crop = x[PREFIX_LEN:]
        flags, npb = spec.crop_controls(crop)
        controls.append((int(x[1] - spec.base), spec.dens_bin(npb), flags))
        real.append(crop)
    report["real_baseline"] = summarize([clip_metrics(c, spec) for c in real])

    print("Building the training-set bar index (for the copying check)...")
    index = build_train_index(spec)
    h, t = copy_rate(real, index, spec)
    report["real_baseline"]["train_bar_overlap"] = (h / max(1, t), None)

    listening = []                                              # (source, ids, tempo_bin)
    for i in range(min(args.listening_clips, len(real))):
        listening.append(("real", real[i], controls[i][0]))

    for ck in ckpts:
        name = ck.stem
        model, mtype = load_model(ck, device)
        gens = []
        for k, (tb, db, flags) in enumerate(tqdm(controls, desc=f"generating ({name})")):
            groups = [g for g, f in zip(C.ALL_GROUPS, flags) if f]
            gens.append(generate_ids(model, mtype, spec, tb, db, groups, args.length,
                                     args.temperature, 0, args.top_p, False, device, seed=C.SEED + k))
        n = len(gens)
        req = np.array([c[2] for c in controls], dtype=bool)
        got = np.array([spec.group_counts(g) >= C.MIN_NOTES_FOR_ON for g in gens])
        dgot = np.array([spec.dens_bin(spec.notes_per_bar(g)) for g in gens])
        dreq = np.array([c[1] for c in controls])
        per_inst = {}
        for gi, gname in enumerate(C.ALL_GROUPS):
            asked = req[:, gi]
            per_inst[gname] = dict(requested=int(asked.sum()), played=int((got[:, gi] & asked).sum()),
                                   played_when_not_requested=int((got[:, gi] & ~asked).sum()),
                                   not_requested=int((~asked).sum()))
        hit, tot = copy_rate(gens, index, spec)
        report["models"][name] = dict(
            n=n,
            inst_exact=pct(int((got == req).all(axis=1).sum()), n),
            density_acc=pct(int((dgot == dreq).sum()), n),
            density_within_1_bin=pct(int((np.abs(dgot - dreq) <= 1).sum()), n),
            density_mae_bins=float(np.abs(dgot - dreq).mean()),
            per_instrument=per_inst,
            quality=summarize([clip_metrics(g, spec) for g in gens]),
            train_bar_overlap=hit / max(1, tot), bars_checked=tot)
        for i in range(min(args.listening_clips, n)):
            listening.append((name, gens[i], controls[i][0]))

    # ---- blind listening test files ----
    out_dir = C.RESULTS_DIR / "listening_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("clip_*.mid"):
        old.unlink()
    order = np.random.default_rng(C.SEED + 1).permutation(len(listening))
    key = []
    for n_, j in enumerate(order, 1):
        src, ids, tb = listening[j]
        fname = f"clip_{n_:02d}.mid"
        try:
            ids_to_midi(ids, tok, spec, out_dir / fname, C.TEMPO_BIN_CENTERS[tb])
            key.append((fname, src))
        except Exception as e:
            print(f"  (could not write {fname}: {e})")
    pd.DataFrame(key, columns=["clip", "source"]).to_csv(out_dir / "ANSWER_KEY_do_not_open_until_done.csv", index=False)
    pd.DataFrame({"clip": [k[0] for k in key], "rating_1to5": "", "sounds_like_real_music_yes_no": "", "comment": ""}).to_csv(
        out_dir / "rating_sheet.csv", index=False)

    # ---- write report ----
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    lines = ["# Validation report\n", f"Clips per model: {args.n_samples}, tokens per clip: {args.length}\n",
             "## 1. Training convergence"]
    for m, t in report["training"].items():
        lines.append(f"- **{m}**: best val loss {t['best_val_loss']:.3f} at epoch {t['best_epoch']}/{t['epochs']}; "
                     f"final train {t['final_train_loss']:.3f} vs val {t['final_val_loss']:.3f} "
                     f"(gap {t['gap_val_minus_train']:+.3f}); last 3 val {['%.3f' % v for v in t['last3_val']]}; "
                     f"still improving: {t['still_improving']}")
    lines.append("\n## 2. Control accuracy (95% confidence intervals; with ~60 clips differences < 10 points are noise)")
    for m, r in report["models"].items():
        lines.append(f"- **{m}**: instrument set exact {r['inst_exact']}; density bin exact {r['density_acc']}, "
                     f"within one bin {r['density_within_1_bin']}, mean error {r['density_mae_bins']:.2f} bins (chance is about 25% exact)")
        for g, v in r["per_instrument"].items():
            lines.append(f"    - {g}: played {v['played']}/{v['requested']} when requested; "
                         f"played {v['played_when_not_requested']}/{v['not_requested']} when NOT requested")
    lines.append("\n## 3. Musical-structure metrics: generated vs REAL test clips (mean ± std)")
    names = ["real_baseline"] + list(report["models"])
    cols = ["notes_per_bar", "key_corr", "in_key", "bar_repeat", "empty_bars"]
    lines.append("| source | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for nm in names:
        q = report["real_baseline"] if nm == "real_baseline" else report["models"][nm]["quality"]
        lines.append(f"| {nm} | " + " | ".join(f"{q[c][0]:.2f} ± {q[c][1]:.2f}" for c in cols) + " |")
    lines.append("\n(key_corr / in_key: how clearly the notes sit in one musical key. bar_repeat: share of repeated bars. "
                 "Values close to the real row = music that is structured like real music.)")
    lines.append("\n## 4. Copying check: share of bars (first 48 tokens) found verbatim in the training set")
    lines.append(f"- real test clips (natural baseline): {100*report['real_baseline']['train_bar_overlap'][0]:.1f}%")
    for m, r in report["models"].items():
        lines.append(f"- {m}: {100*r['train_bar_overlap']:.1f}% of {r['bars_checked']} bars")
    lines.append("\nIf a model's number is far ABOVE the real baseline it is memorising songs; "
                 "at or below the baseline means it is producing new material.")
    lines.append(f"\n## 5. Listening test\nFiles in `{out_dir}`: rate each clip 1-5 in `rating_sheet.csv` "
                 "(ask 3-5 friends, do NOT open the answer key first), then average the ratings per source.")
    (C.RESULTS_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved results/validation_report.md and .json, plus the listening test in {out_dir}")


if __name__ == "__main__":
    main()