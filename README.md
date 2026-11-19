# Controllable Multi-Instrument Music Generation (LSTM vs Transformer)

Symbolic (MIDI) music generation with 5 instrument groups (drums, piano, guitar, bass, strings),
controlled by **tempo**, **density** (note events per bar) and **instrument ON/OFF** tokens.

## Folder layout
```
music-gen/
├── config.py              all settings (paths, filters, tokenizer, control bins)
├── requirements.txt
├── step1_scan_filter.py   scan raw MIDI, filter, write manifest.csv + plots
├── common.py              tokenizer, instrument merging, control tokens, token -> MIDI
├── step2_tokenize.py      MidiTok REMI tokenization + artist-level train/val/test split
├── dataset.py             PyTorch Dataset (bar-aligned crops + control prefix)
├── models.py              LSTMLM and TransformerLM
├── train.py               training loop (python train.py --model lstm|transformer)
├── generate.py            controllable generation -> .mid
├── evaluate.py            perplexity + controllability metrics, LSTM vs Transformer table
├── app.py                 Gradio demo
├── run_all.py             runs steps 1,2, both trainings, evaluation in one command
└── data/clean_midi/<Artist>/<Song>.mid     <- put the Clean MIDI dataset here
```
All files go in ONE folder (they import each other).

## Run order
```bash
pip install -r requirements.txt

python step1_scan_filter.py        # 1. filter dataset      -> data/processed/manifest.csv
python step2_tokenize.py           # 2. tokenize + split    -> data/processed/tokens/*.pkl
python train.py --model lstm       # 3a. train LSTM         -> checkpoints/lstm.pt
python train.py --model transformer# 3b. train Transformer  -> checkpoints/transformer.pt
python evaluate.py                 # 4. compare models      -> results/metrics.csv
python generate.py --ckpt checkpoints/transformer.pt --bpm 120 --density 2 --instruments drums,bass,piano
python app.py                      # 5. web demo
```
Or everything at once: `python run_all.py` (quick end-to-end test first: `python run_all.py --smoke`).

## Tips
* **GPU**: training on CPU is very slow. Use Colab/Kaggle (T4). Upload the folder, `pip install -r requirements.txt`.
* **Out of memory**: lower `--batch-size` (8) or `--seq-len` (512).
* **Too few songs after step 1**: relax `MIN_GROUPS_PRESENT` / `ALLOWED_TIME_SIGS` in config.py.
* **Quick experiments**: `--crops-per-song 4` and `--epochs 10`.
* **Ablation ideas**: `--no-ctx` (LSTM without per-step control vector), different `--seq-len`,
  `--hard-instruments` in generate.py, temperature / top-p sweeps.
* **Audio in the demo**: install FluidSynth + a SoundFont, then `SOUNDFONT=/path/font.sf2 python app.py`.
