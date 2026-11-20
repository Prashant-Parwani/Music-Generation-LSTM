"""
STEP 3: Train the LSTM or the Transformer   (v3: classifier-free-guidance dropout, pitch transposition,
                                              resumable, gradient accumulation, clean validation loss)

    python train.py --model lstm
    python train.py --model transformer
    python train.py --model transformer --resume            # continue after a crash / shutdown
    python train.py --model transformer --size small        # the older, smaller architecture (faster, less memory)
    python train.py --model transformer --tiny --epochs 1   # 1-minute smoke test
    python train.py --model transformer --name transformer_v3 --size small --batch-size 4 --grad-accum 4 --seq-len 512

Best checkpoint (lowest validation loss) -> checkpoints/<name>.pt       (used by generate / evaluate / export)
Latest state (for --resume)              -> checkpoints/<name>_last.pt  (deleted automatically when training finishes)
Log                                      -> checkpoints/<name>_log.csv

v3 additions
  --cfg-drop 0.1    10% of training examples have their 7 control tokens replaced by MASK. This teaches the model an
                    "unconditional" mode, which generate.py needs for classifier-free guidance. The best checkpoint
                    stores cfg=True so generate.py turns guidance on automatically.
  --no-transpose    disable the random pitch-transposition augmentation (-5..+6 semitones, drums untouched).
"""
import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from common import PREFIX_LEN, ControlSpec
from dataset import MusicDataset
from models import build_model, count_params


def compute_loss(model, x, pad_id, label_smoothing=0.0):
    """Next-token cross-entropy. The control prefix is given, not predicted, so it is masked out."""
    inp, tgt = x[:, :-1], x[:, 1:].clone()
    tgt[:, :PREFIX_LEN - 1] = -100
    tgt[tgt == pad_id] = -100
    logits = model(inp)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                           ignore_index=-100, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate_loss(model, loader, pad_id, device, max_batches=None):
    """Validation/test loss. ALWAYS plain cross-entropy (no smoothing) so perplexity is a true perplexity."""
    model.eval()
    total, n = 0.0, 0
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    for i, x in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            loss = compute_loss(model, x, pad_id)
        total += loss.item(); n += 1
    return total / max(1, n)


def make_optimizer(model, lr, weight_decay):
    """AdamW; weight decay only on weight matrices (not biases / LayerNorm)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)        # a crash / power cut during saving can never corrupt the old file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8, help="per-step batch (lower it if the GPU runs out of memory)")
    ap.add_argument("--grad-accum", type=int, default=1, help="accumulate N steps -> effective batch = batch-size x N")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--crops-per-song", type=int, default=4, help="random crops per song per epoch")
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 transformer, 2e-3 lstm")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="default OFF: smoothing adds random-token noise when sampling and distorts validation loss")
    ap.add_argument("--patience", type=int, default=8, help="early stopping on val loss (epochs without improvement)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", choices=["base", "small"], default="base",
                    help="base = sizes in models.py (LSTM 768 / Transformer 512x8); small = older smaller sizes")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>_last.pt")
    ap.add_argument("--keep-last", action="store_true", help="keep <name>_last.pt after training finishes")
    ap.add_argument("--max-train-batches", type=int, default=None, help="debug: cap batches per epoch")
    ap.add_argument("--stop-after", type=int, default=None, help="debug: stop after N epochs (to test --resume)")
    ap.add_argument("--no-ctx", action="store_true", help="LSTM only: disable per-step control vector")
    ap.add_argument("--cfg-drop", type=float, default=0.1,
                    help="probability of replacing the control tokens with MASK (enables classifier-free guidance; 0 = off)")
    ap.add_argument("--no-transpose", action="store_true", help="disable pitch-transposition augmentation")
    ap.add_argument("--tiny", action="store_true", help="very small model for a smoke test")
    ap.add_argument("--name", default=None, help="checkpoint name (default: model type)")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    spec = ControlSpec.load()
    name = args.name or args.model

    train_ds = MusicDataset("train", args.seq_len, args.crops_per_song, train=True, spec=spec,
                            transpose=None if args.no_transpose else (-5, 6), cfg_drop=args.cfg_drop)
    val_ds = MusicDataset("val", args.seq_len, max(2, args.crops_per_song // 4), train=False, spec=spec)
    pin, persist = device == "cuda", args.workers > 0
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                          pin_memory=pin, persistent_workers=persist)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=pin, persistent_workers=persist)

    if args.model == "lstm":
        cfg = dict(vocab_size=spec.vocab_size, use_ctx=not args.no_ctx)
        if args.tiny:
            cfg.update(emb=64, hidden=128, layers=2)
        elif args.size == "small":
            cfg.update(emb=256, hidden=512, layers=3)
    else:
        cfg = dict(vocab_size=spec.vocab_size, max_len=args.seq_len, rope=True)
        if args.tiny:
            cfg.update(d_model=64, n_layers=2, n_heads=2)
        elif args.size == "small":
            cfg.update(d_model=384, n_layers=6, n_heads=6)
    model = build_model(args.model, cfg).to(device)

    accum = max(1, args.grad_accum)
    micro_steps = (min(len(train_dl), args.max_train_batches or len(train_dl)) // accum) * accum
    opt_steps_per_epoch = max(1, micro_steps // accum)
    total_steps = args.epochs * opt_steps_per_epoch
    print(f"{args.model} ({'tiny' if args.tiny else args.size}): {count_params(model)/1e6:.2f}M parameters | device={device}")
    print(f"{len(train_ds.ids)} train songs, {len(val_ds.ids)} val songs | batch {args.batch_size} x accum {accum} "
          f"= effective {args.batch_size * accum} | {opt_steps_per_epoch} optimizer steps/epoch | "
          f"label smoothing {args.label_smoothing}")
    print(f"cfg_drop {args.cfg_drop} | transpose {'off' if args.no_transpose else '-5..+6'}")

    lr = args.lr or (3e-4 if args.model == "transformer" else 2e-3)
    opt = make_optimizer(model, lr, args.weight_decay)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))) * 0.98 + 0.02)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = C.CHECKPOINT_DIR / f"{name}_log.csv"
    last_path = C.CHECKPOINT_DIR / f"{name}_last.pt"
    best_path = C.CHECKPOINT_DIR / f"{name}.pt"

    start_epoch, best, bad = 1, float("inf"), 0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"Resuming after epoch {ck['epoch']} (best val loss so far {best:.3f})")
    else:
        if args.resume:
            print("No _last.pt found, starting from scratch.")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_ppl", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_dl, total=micro_steps, desc=f"epoch {epoch}/{args.epochs}")
        for i, x in enumerate(bar):
            if i >= micro_steps:
                break
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(model, x, spec.pad_id, args.label_smoothing)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run += loss.item(); n += 1
            if n % 25 == 0:
                bar.set_postfix(loss=f"{run/n:.3f}")

        val = evaluate_loss(model, val_dl, spec.pad_id, device, max_batches=args.max_train_batches)
        print(f"  train {run/max(1,n):.3f} | val {val:.3f} | val ppl {math.exp(val):.2f} | "
              f"gap {val - run/max(1,n):+.3f} | {(time.time()-t0)/60:.1f} min")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(run / max(1, n), 4), round(val, 4),
                                    round(math.exp(val), 3), round(time.time() - t0, 1)])
        if val < best:
            best, bad = val, 0
            save_atomic(dict(model_type=args.model, config=model.config, state_dict=model.state_dict(),
                             val_loss=val, epoch=epoch, args=vars(args), cfg=args.cfg_drop > 0), best_path)
            print(f"  saved best checkpoint -> checkpoints/{name}.pt")
        else:
            bad += 1
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best=best, bad=bad), last_path)
        if bad >= args.patience:
            print(f"  early stopping (no improvement for {args.patience} epochs)")
            break
        if args.stop_after and epoch >= args.stop_after:
            print(f"  stopping after epoch {epoch} (--stop-after); continue with --resume")
            return

    print(f"Done. Best val loss {best:.3f} (ppl {math.exp(best):.2f})")
    if last_path.exists() and not args.keep_last:
        last_path.unlink()


if __name__ == "__main__":
    main()