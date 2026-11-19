"""
STEP 1: Scan the raw MIDI files, filter out unusable ones, and build a manifest.

Run:
    python step1_scan_filter.py
    python step1_scan_filter.py --limit 500      # quick test on 500 files

Output:
    data/processed/manifest.csv   (one row per file, with status + stats)
    data/processed/plots/*.png    (dataset overview plots)
"""
import argparse
import os
import re
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

import config as C


def program_to_group(program: int):
    for name, rng in C.INSTRUMENT_GROUPS.items():
        if program in rng:
            return name
    return None  # other instruments (synth leads, brass, etc.) are ignored


def analyse(path_str: str) -> dict:
    """Load one MIDI file and return its stats + whether it passes the filters."""
    warnings.filterwarnings("ignore")
    row = {"path": path_str, "status": "rejected", "reason": ""}

    try:
        pm = pretty_midi.PrettyMIDI(path_str)
    except Exception:
        row["reason"] = "corrupt"
        return row

    duration = pm.get_end_time()
    row["duration"] = round(float(duration), 2)
    if duration < C.MIN_DURATION_S:
        row["reason"] = "too_short"
        return row
    if duration > C.MAX_DURATION_S:
        row["reason"] = "too_long"
        return row

    # Time signature: only allow 4/4 (or none specified, which defaults to 4/4)
    for ts in pm.time_signature_changes:
        if (ts.numerator, ts.denominator) not in C.ALLOWED_TIME_SIGS:
            row["reason"] = "time_signature"
            return row

    # Tempo
    _, tempi = pm.get_tempo_changes()
    tempo = float(np.median(tempi)) if len(tempi) else 120.0
    row["tempo"] = round(tempo, 1)
    if not (C.TEMPO_RANGE[0] <= tempo <= C.TEMPO_RANGE[1]):
        row["reason"] = "tempo_out_of_range"
        return row

    # Count notes per instrument group
    counts = {g: 0 for g in C.ALL_GROUPS}
    for inst in pm.instruments:
        if inst.is_drum:
            counts["drums"] += len(inst.notes)
        else:
            g = program_to_group(inst.program)
            if g is not None:
                counts[g] += len(inst.notes)

    present = [g for g, n in counts.items() if n >= C.MIN_NOTES_PER_GROUP]
    for g in C.ALL_GROUPS:
        row[f"notes_{g}"] = counts[g]
    row["groups"] = ",".join(present)
    row["n_groups"] = len(present)

    if len(present) < C.MIN_GROUPS_PRESENT:
        row["reason"] = "too_few_instrument_groups"
        return row

    total_notes = sum(counts[g] for g in present)
    row["n_notes"] = total_notes
    row["density"] = round(total_notes / duration, 2)  # notes per second
    row["status"] = "ok"
    return row


def song_key(path: Path):
    """Same artist folder + filename without the '.1', '.2' version suffix."""
    stem = re.sub(r"\.\d+$", "", path.stem).lower()
    return (path.parent.name.lower(), stem)


def make_plots(df_ok: pd.DataFrame, df_all: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Rejection reasons
    reasons = df_all["reason"].replace("", "kept").value_counts()
    plt.figure(figsize=(8, 4))
    reasons.plot(kind="bar")
    plt.title("Files kept vs rejected (by reason)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "rejection_reasons.png", dpi=120)
    plt.close()

    # Instrument group presence
    group_counts = Counter()
    for gs in df_ok["groups"]:
        group_counts.update(gs.split(","))
    plt.figure(figsize=(6, 4))
    plt.bar(list(group_counts.keys()), list(group_counts.values()))
    plt.title("Songs containing each instrument group")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "instrument_groups.png", dpi=120)
    plt.close()

    # Tempo, duration, density histograms
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(df_ok["tempo"], bins=30); ax[0].set_title("Tempo (BPM)")
    ax[1].hist(df_ok["duration"], bins=30); ax[1].set_title("Duration (s)")
    ax[2].hist(df_ok["density"], bins=30); ax[2].set_title("Density (notes/s)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "tempo_duration_density.png", dpi=120)
    plt.close()


def find_midi_files(root):
    """Find all .mid/.midi files. Folders Windows cannot open (e.g. names ending in a dot) are skipped."""
    skipped = []

    def on_error(err):
        skipped.append(str(getattr(err, "filename", err)))

    files = []
    for dirpath, _, names in os.walk(root, onerror=on_error):
        for n in names:
            if n.lower().endswith((".mid", ".midi")):
                files.append(Path(dirpath) / n)
    if skipped:
        print(f"Skipped {len(skipped)} unreadable folder(s) (invalid Windows names), e.g. {skipped[:3]}")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only scan first N files (for testing)")
    args = parser.parse_args()

    if not C.RAW_MIDI_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {C.RAW_MIDI_DIR}\n"
                         "Extract the Clean MIDI download there (data/clean_midi/<Artist>/<Song>.mid).")

    files = find_midi_files(C.RAW_MIDI_DIR)
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} MIDI files. Analysing...")

    workers = C.NUM_WORKERS or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(tqdm(ex.map(analyse, [str(f) for f in files], chunksize=16), total=len(files)))

    df = pd.DataFrame(rows)

    # Remove duplicate versions of the same song (keep the first one that passed)
    if C.DEDUPE_VERSIONS:
        seen, dup_idx = set(), []
        for i, r in df.iterrows():
            if r["status"] != "ok":
                continue
            k = song_key(Path(r["path"]))
            if k in seen:
                dup_idx.append(i)
            else:
                seen.add(k)
        df.loc[dup_idx, ["status", "reason"]] = ["rejected", "duplicate_version"]

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.MANIFEST_CSV, index=False)

    ok = df[df["status"] == "ok"].copy()
    print("\n===== SUMMARY =====")
    print(f"Total files scanned : {len(df)}")
    print(f"Kept                : {len(ok)}")
    print("Rejected by reason  :")
    print(df[df['status'] != 'ok']['reason'].value_counts().to_string())
    if len(ok):
        print(f"\nTempo   median/min/max : {ok['tempo'].median():.0f} / {ok['tempo'].min():.0f} / {ok['tempo'].max():.0f}")
        print(f"Density median (notes/s): {ok['density'].median():.1f}")
        gc = Counter(g for gs in ok["groups"] for g in gs.split(","))
        print("Songs per instrument group:", dict(gc))
        make_plots(ok, df)
        print(f"\nManifest saved to {C.MANIFEST_CSV}")
        print(f"Plots saved to    {C.PLOTS_DIR}")


if __name__ == "__main__":
    main()