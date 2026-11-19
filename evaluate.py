"""
STEP 5: Evaluate and compare the LSTM and the Transformer.

    python evaluate.py                       # evaluates every checkpoint found in checkpoints/
    python evaluate.py --n-samples 100 --save-midi 5

Metrics
  Likelihood : test-set negative log-likelihood and perplexity (lower = better)
  Controllability (generated with controls taken from real test crops):
     inst_exact      generated instrument set == requested set
     inst_recall     requested instruments that actually appear
     inst_extra      generated instruments that were NOT requested
     density_acc     generated density bin == requested bin ; density_mae = mean |bin difference|
  Validity
     decode_ok       fraction of generated sequences that decode to a MIDI file without error
  (Tempo is applied at playback, so it is not scored. For musical-quality checks run validate.py.)
Results -> results/metrics.csv  (+ results/samples/<model>/*.mid)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from common import ControlSpec, ids_to_midi, load_tokenizer, PREFIX_LEN
from dataset import MusicDataset
from generate import generate_ids, load_model
from train import evaluate_loss


def real_reference(ds, spec, n):
    """Pick n real test crops -> (controls, average pitch-class histogram)."""
    rng = np.random.default_rng(C.SEED)
    items, hist = [], np.zeros(12)
    for idx in rng.integers(0, len(ds), size=n):
        x = ds[int(idx)].numpy()
        crop = x[PREFIX_LEN:]
        flags, npb = spec.crop_controls(crop)
        tb = int(x[1] - spec.base)                       # tempo token -> bin
        items.append((tb, spec.dens_bin(npb), flags))
        hist += spec.pitch_class_hist(crop)
    return items, hist / max(1, len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None, help="checkpoint files (default: all in checkpoints/)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--length", type=int, default=512, help="tokens per generated sample")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--save-midi", type=int, default=3, help="MIDI samples to save per model")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec, tok = ControlSpec.load(), load_tokenizer()
    ckpts = args.ckpts or sorted(str(p) for p in C.CHECKPOINT_DIR.glob("*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints found. Run train.py first.")

    test_ds = MusicDataset("test", args.seq_len, crops_per_song=4, train=False, spec=spec)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False)
    ref_items, ref_pch = real_reference(test_ds, spec, args.n_samples)

    rows = []
    for ck in ckpts:
        model, mtype = load_model(ck, device)
        name = Path(ck).stem
        nll = evaluate_loss(model, test_dl, spec.pad_id, device)
        print(f"\n[{name}] test NLL {nll:.3f} | perplexity {math.exp(nll):.2f}")

        exact, recall, extra, dacc, dmae, ok = [], [], [], [], [], []
        out_dir = C.RESULTS_DIR / "samples" / name
        for k, (tb, db, flags) in enumerate(tqdm(ref_items, desc=f"generating ({name})")):
            groups = [g for g, f in zip(C.ALL_GROUPS, flags) if f]
            ids = generate_ids(model, mtype, spec, tb, db, groups, args.length, args.temperature,
                               0, args.top_p, False, device, seed=C.SEED + k)
            gen_flags = spec.group_counts(ids) >= C.MIN_NOTES_FOR_ON
            req = np.asarray(flags, dtype=bool)
            exact.append(bool((gen_flags == req).all()))
            recall.append(float((gen_flags & req).sum() / max(1, req.sum())))
            extra.append(float((gen_flags & ~req).sum() / max(1, gen_flags.sum())))
            gb = spec.dens_bin(spec.notes_per_bar(ids))
            dacc.append(gb == db); dmae.append(abs(gb - db))
            try:
                if k < args.save_midi:
                    ids_to_midi(ids, tok, spec, out_dir / f"sample{k}.mid", C.TEMPO_BIN_CENTERS[tb])
                else:
                    from common import ids_to_score
                    ids_to_score(ids, tok, spec, C.TEMPO_BIN_CENTERS[tb])
                ok.append(True)
            except Exception:
                ok.append(False)

        rows.append(dict(model=name, params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                         test_nll=round(nll, 3), test_ppl=round(math.exp(nll), 2),
                         inst_exact=round(np.mean(exact), 3), inst_recall=round(np.mean(recall), 3),
                         inst_extra=round(np.mean(extra), 3), density_acc=round(np.mean(dacc), 3),
                         density_mae=round(np.mean(dmae), 3), decode_ok=round(np.mean(ok), 3)))

    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved results/metrics.csv and MIDI samples in results/samples/")


if __name__ == "__main__":
    main()