"""
STEP 3: Train the LSTM or the Transformer.

    python train.py --model lstm
    python train.py --model transformer
    python train.py --model transformer --tiny --epochs 1      # 1-minute smoke test

Best checkpoint (lowest validation loss) -> checkpoints/<model>.pt ; log -> checkpoints/<model>_log.csv
"""
import argparse
import csv
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from common import PREFIX_LEN, ControlSpec
from dataset import MusicDataset
from models import build_model, count_params


def compute_loss(model, x, pad_id):
    """Next-token cross-entropy. The control prefix is given, not predicted, so it is masked out."""
    inp, tgt = x[:, :-1], x[:, 1:].clone()
    tgt[:, :PREFIX_LEN - 1] = -100
    tgt[tgt == pad_id] = -100
    logits = model(inp)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                           ignore_index=-100, label_smoothing=0.1)


@torch.no_grad()
def evaluate_loss(model, loader, pad_id, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for i, x in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        loss = compute_loss(model, x, pad_id)
        total += loss.item(); n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--crops-per-song", type=int, default=8, help="random crops per song per epoch")
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 transformer, 2e-3 lstm")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10, help="early stopping on val loss")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train-batches", type=int, default=None, help="debug: cap batches per epoch")
    ap.add_argument("--no-ctx", action="store_true", help="LSTM only: disable per-step control vector")
    ap.add_argument("--tiny", action="store_true", help="very small model for a smoke test")
    ap.add_argument("--name", default=None, help="checkpoint name (default: model type)")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ControlSpec.load()
    name = args.name or args.model

    train_ds = MusicDataset("train", args.seq_len, args.crops_per_song, train=True, spec=spec)
    val_ds = MusicDataset("val", args.seq_len, max(2, args.crops_per_song // 4), train=False, spec=spec)
    pin = device == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    if args.model == "lstm":
        cfg = dict(vocab_size=spec.vocab_size, use_ctx=not args.no_ctx)
        cfg.update(dict(emb=64, hidden=128, layers=2) if args.tiny else {})
    else:
        cfg = dict(vocab_size=spec.vocab_size, max_len=args.seq_len)
        cfg.update(dict(d_model=64, n_layers=2, n_heads=2) if args.tiny else {})
    model = build_model(args.model, cfg).to(device)
    print(f"{args.model}: {count_params(model)/1e6:.2f}M parameters | device={device} | "
          f"{len(train_ds.ids)} train songs, {len(val_ds.ids)} val songs")

    lr = args.lr or (3e-4 if args.model == "transformer" else 2e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_dl), args.max_train_batches or len(train_dl))
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))) * 0.98 + 0.02)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = C.CHECKPOINT_DIR / f"{name}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_ppl", "seconds"])

    best, bad = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        bar = tqdm(train_dl, total=steps_per_epoch, desc=f"epoch {epoch}/{args.epochs}")
        for i, x in enumerate(bar):
            if i >= steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(model, x, spec.pad_id)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); n += 1
            bar.set_postfix(loss=f"{run/n:.3f}")

        val = evaluate_loss(model, val_dl, spec.pad_id, device, max_batches=args.max_train_batches)
        print(f"  train {run/max(1,n):.3f} | val {val:.3f} | val ppl {math.exp(val):.2f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(run / max(1, n), 4), round(val, 4),
                                    round(math.exp(val), 3), round(time.time() - t0, 1)])
        if val < best:
            best, bad = val, 0
            torch.save(dict(model_type=args.model, config=model.config, state_dict=model.state_dict(),
                            val_loss=val, epoch=epoch, args=vars(args)), C.CHECKPOINT_DIR / f"{name}.pt")
            print(f"  saved best checkpoint -> checkpoints/{name}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping"); break
    print(f"Done. Best val loss {best:.3f} (ppl {math.exp(best):.2f})")


if __name__ == "__main__":
    main()
