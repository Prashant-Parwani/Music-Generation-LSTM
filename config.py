"""Central config for the multi-instrument music generation project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths -------------------------------------------------------------
# Extract the Clean MIDI download so that artist folders sit inside this dir:
#   data/clean_midi/<Artist>/<Song>.mid
RAW_MIDI_DIR = ROOT / "data" / "clean_midi"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
PLOTS_DIR = PROCESSED_DIR / "plots"
TOKENS_DIR = PROCESSED_DIR / "tokens"
TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.json"
CONTROLS_JSON = PROCESSED_DIR / "controls.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# --- Instrument groups (same idea as the Lakh Pianoroll LPD-5 setup) ----
# General MIDI program ranges (0-indexed). Drums are detected via is_drum.
INSTRUMENT_GROUPS = {
    "piano":   range(0, 8),     # acoustic/electric pianos, harpsichord, clavinet...
    "guitar":  range(24, 32),   # nylon/steel/electric/distortion guitars
    "bass":    range(32, 40),   # acoustic/electric/fretless/synth bass
    "strings": range(40, 56),   # violin, viola, cello, ensembles, choir/orchestra hits
}
ALL_GROUPS = ["drums"] + list(INSTRUMENT_GROUPS.keys())

# After merging, every group becomes ONE track with this representative program.
# (drums use program 0 with is_drum=True; MidiTok turns that into Program_-1)
GROUP_PROGRAM = {"drums": 0, "piano": 0, "guitar": 25, "bass": 33, "strings": 48}

# --- Step 1 filtering rules ---------------------------------------------
MIN_DURATION_S = 30          # drop tiny clips
MAX_DURATION_S = 600         # drop very long songs
MIN_NOTES_PER_GROUP = 30     # a group only "counts" if it has at least this many notes
MIN_GROUPS_PRESENT = 3       # need at least 3 of the 5 groups (e.g. drums+bass+piano)
ALLOWED_TIME_SIGS = {(4, 4)} # 4/4 only keeps bar structure consistent
TEMPO_RANGE = (40, 220)      # BPM sanity range
DEDUPE_VERSIONS = True       # Clean MIDI has "Song.mid", "Song.1.mid"... keep one per song

# --- Step 2 tokenization ------------------------------------------------
# REMI with Program tokens, one token stream for all instruments.
TOKENIZER_KWARGS = dict(
    num_velocities=16,
    use_tempos=True, num_tempos=16, tempo_range=(50, 200),
    use_programs=True, programs=[-1, 0, 25, 33, 48],   # drums, piano, guitar, bass, strings
    one_token_stream_for_programs=True,
    use_time_signatures=False,
    use_chords=False,
    beat_res={(0, 4): 4, (4, 12): 4},                  # uniform 16th-note grid
)
MIN_TOKENS_PER_SONG = 600
SPLIT_FRACTIONS = (0.90, 0.05, 0.05)   # train / val / test, split BY ARTIST to avoid leakage

# --- Control tokens -----------------------------------------------------
TEMPO_EDGES = [90, 110, 130, 150]      # -> 5 tempo bins
TEMPO_BIN_CENTERS = [75, 100, 120, 140, 165]
N_DENS_BINS = 4                        # density = note events per bar, quartile-based bins
MIN_NOTES_FOR_ON = 4                   # a group is "ON" in a crop if it has >= this many notes

# --- Misc ---------------------------------------------------------------
NUM_WORKERS = 2              # limit parallel processes to avoid Windows paging file errors
SEED = 42
