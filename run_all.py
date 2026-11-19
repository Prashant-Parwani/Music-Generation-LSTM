"""
Run the whole pipeline with one command.

    python run_all.py                       # everything, default settings
    python run_all.py --epochs 30           # longer training
    python run_all.py --skip step1 step2    # reuse earlier outputs
    python run_all.py --smoke               # tiny/fast end-to-end test (a few minutes)
"""
import argparse
import subprocess
import sys

PY = sys.executable


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"Step failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["step1", "step2", "lstm", "transformer", "evaluate"])
    ap.add_argument("--smoke", action="store_true", help="tiny models, few songs, 1 epoch")
    args = ap.parse_args()

    limit = ["--limit", "300"] if args.smoke else []
    tr = ["--epochs", "1", "--tiny", "--seq-len", "256", "--batch-size", "8", "--crops-per-song", "2",
          "--max-train-batches", "20", "--workers", "0"] if args.smoke else \
         ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len)]

    if "step1" not in args.skip:
        run([PY, "step1_scan_filter.py"] + limit)
    if "step2" not in args.skip:
        run([PY, "step2_tokenize.py"] + limit)
    for m in ("lstm", "transformer"):
        if m not in args.skip:
            run([PY, "train.py", "--model", m] + tr)
    if "evaluate" not in args.skip:
        ev = ["--n-samples", "6", "--length", "256", "--seq-len", "256"] if args.smoke else []
        run([PY, "evaluate.py"] + ev)
    print("\nAll done. Try:  python generate.py --ckpt checkpoints/transformer.pt   or   python app.py")


if __name__ == "__main__":
    main()
