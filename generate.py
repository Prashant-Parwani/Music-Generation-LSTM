"""
STEP 4: Controllable generation.

    python generate.py --ckpt checkpoints/transformer.pt --bpm 120 --density 2 \
        --instruments drums,bass,piano,guitar --length 1024 --out out/song.mid

Controls
  --bpm          target tempo (mapped to one of 5 tempo bins)
  --density      0 (sparse) .. 3 (busy)   [note events per bar, quartile bins]
  --instruments  comma list from: drums,piano,guitar,bass,strings
  --hard-instruments  additionally forbid instruments you did not ask for (hard constraint)
"""
import argparse
from pathlib import Path

import torch

import config as C
from common import PREFIX_LEN, ControlSpec, ids_to_midi, load_tokenizer
from models import build_model


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["model_type"], ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["model_type"]


def _sample(logits, banned, temperature, top_k, top_p, recent_ids=None, rep_penalty=1.0):
    logits = logits.float().clone()
    logits[banned] = float("-inf")
    # Repetition penalty: discourage tokens recently generated
    if recent_ids is not None and rep_penalty > 1.0:
        for tid in set(recent_ids):
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p and top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


@torch.no_grad()
def generate_ids(model, model_type, spec: ControlSpec, tempo_bin, dens_bin, groups, n_tokens=1024,
                 temperature=0.90, top_k=0, top_p=0.92, hard_instruments=False, device="cpu", seed=None,
                 rep_penalty=1.15, rep_window=64):
    """Returns the generated MUSIC token ids (control prefix removed). Starts with a Bar token."""
    if seed is not None:
        torch.manual_seed(seed)
    flags = [g in groups for g in C.ALL_GROUPS]
    prefix = spec.prefix(tempo_bin, dens_bin, flags) + [spec.bar_id]

    banned = list(spec.banned_ids)
    if hard_instruments:
        for gi, on in enumerate(flags):
            if not on:
                banned += spec.program_ids_of_group(gi)
    banned = torch.tensor(sorted(set(banned)), device=device)

    out = [spec.bar_id]
    if model_type == "lstm":
        logits, hidden, ctx = model.step(torch.tensor([prefix], device=device))
        for _ in range(n_tokens - 1):
            recent = out[-rep_window:] if rep_window else None
            nxt = _sample(logits[0, -1], banned, temperature, top_k, top_p, recent, rep_penalty)
            out.append(nxt)
            logits, hidden, ctx = model.step(torch.tensor([[nxt]], device=device), hidden, ctx)
    else:
        seq = list(prefix)
        for _ in range(n_tokens - 1):
            # sliding window that always keeps the control prefix
            ctx_ids = seq if len(seq) <= model.max_len else seq[:PREFIX_LEN] + seq[-(model.max_len - PREFIX_LEN):]
            logits = model(torch.tensor([ctx_ids], device=device))[0, -1]
            recent = out[-rep_window:] if rep_window else None
            nxt = _sample(logits, banned, temperature, top_k, top_p, recent, rep_penalty)
            seq.append(nxt); out.append(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.CHECKPOINT_DIR / "transformer.pt"))
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--density", type=int, default=2, choices=range(C.N_DENS_BINS))
    ap.add_argument("--instruments", default="drums,bass,piano,guitar")
    ap.add_argument("--length", type=int, default=1024, help="number of music tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--hard-instruments", action="store_true")
    ap.add_argument("--rep-penalty", type=float, default=1.15, help="repetition penalty factor (1.0 = off)")
    ap.add_argument("--rep-window", type=int, default=64, help="look-back window for repetition penalty")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out/generated.mid")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec, tok = ControlSpec.load(), load_tokenizer()
    model, mtype = load_model(args.ckpt, device)
    groups = [g.strip() for g in args.instruments.split(",") if g.strip()]
    bad = [g for g in groups if g not in C.ALL_GROUPS]
    if bad:
        raise SystemExit(f"Unknown instruments {bad}. Choose from {C.ALL_GROUPS}")

    tb = spec.tempo_bin(args.bpm)
    ids = generate_ids(model, mtype, spec, tb, args.density, groups, args.length, args.temperature,
                       args.top_k, args.top_p, args.hard_instruments, device, args.seed,
                       args.rep_penalty, args.rep_window)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ids_to_midi(ids, tok, spec, args.out, tempo_bpm=args.bpm)

    counts = spec.group_counts(ids)
    print(f"Saved {args.out}")
    print("Requested:", groups, "| tempo bin", tb, "| density bin", args.density)
    print("Notes generated per group:", {g: int(c) for g, c in zip(C.ALL_GROUPS, counts)})
    print(f"Measured density: {spec.notes_per_bar(ids):.1f} note events/bar -> bin {spec.dens_bin(spec.notes_per_bar(ids))}")


if __name__ == "__main__":
    main()