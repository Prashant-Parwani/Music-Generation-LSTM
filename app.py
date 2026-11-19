"""
STEP 6: Web demo (Gradio).

    python app.py            # then open the printed local URL
    python app.py --share    # public temporary link (handy from Colab)

Pick the model, tempo, density and instruments -> get a MIDI file, a piano-roll plot and (if
FluidSynth + a SoundFont are installed) playable audio. Set SOUNDFONT=/path/to/font.sf2 for audio.
"""
import argparse
import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch

import config as C
from common import ControlSpec, ids_to_midi, load_tokenizer
from generate import generate_ids, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPEC, TOK = ControlSpec.load(), load_tokenizer()
CKPTS = {p.stem: p for p in sorted(C.CHECKPOINT_DIR.glob("*.pt"))}
_cache = {}


def get_model(name):
    if name not in _cache:
        _cache[name] = load_model(CKPTS[name], DEVICE)
    return _cache[name]


def pianoroll_figure(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors
    for k, inst in enumerate(pm.instruments):
        label = "drums" if inst.is_drum else pretty_midi.program_to_instrument_name(inst.program)
        for n in inst.notes:
            ax.plot([n.start, n.end], [n.pitch, n.pitch], color=colors[k % 10], linewidth=2,
                    label=label)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch")
    fig.tight_layout()
    return fig


def render_audio(midi_path):
    sf2 = os.environ.get("SOUNDFONT")
    if not sf2:
        return None
    try:
        from scipy.io import wavfile
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=22050, sf2_path=sf2)
        out = midi_path.replace(".mid", ".wav")
        wavfile.write(out, 22050, (audio / max(1e-6, np.abs(audio).max()) * 32767).astype(np.int16))
        return out
    except Exception:
        return None


def run(model_name, bpm, density, instruments, length, temperature, top_p, hard, seed):
    if not instruments:
        raise gr.Error("Select at least one instrument.")
    model, mtype = get_model(model_name)
    tb = SPEC.tempo_bin(bpm)
    ids = generate_ids(model, mtype, SPEC, tb, int(density), list(instruments), int(length),
                       temperature, 0, top_p, hard, DEVICE, seed=int(seed))
    path = os.path.join(tempfile.mkdtemp(), "generated.mid")
    try:
        ids_to_midi(ids, TOK, SPEC, path, tempo_bpm=bpm)
    except ValueError as e:
        raise gr.Error(str(e))
    counts = SPEC.group_counts(ids)
    info = " | ".join(f"{g}: {int(c)} notes" for g, c in zip(C.ALL_GROUPS, counts) if c > 0)
    return path, pianoroll_figure(path), render_audio(path), info


with gr.Blocks(title="Controllable Multi-Instrument Music Generation") as demo:
    gr.Markdown("# Controllable Multi-Instrument Music Generation\nLSTM vs Transformer, "
                "conditioned on tempo, density and instruments.")
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(list(CKPTS), value=list(CKPTS)[0] if CKPTS else None, label="Model")
            bpm = gr.Slider(60, 180, value=120, step=1, label="Tempo (BPM)")
            density = gr.Slider(0, C.N_DENS_BINS - 1, value=2, step=1,
                                label="Density (0 sparse .. 3 busy)")
            inst = gr.CheckboxGroup(C.ALL_GROUPS, value=["drums", "bass", "piano"], label="Instruments")
            length = gr.Slider(256, 2048, value=1024, step=128, label="Length (tokens)")
            temp = gr.Slider(0.5, 1.5, value=0.95, step=0.05, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
            hard = gr.Checkbox(False, label="Hard instrument constraint")
            seed = gr.Number(42, label="Seed", precision=0)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            midi = gr.File(label="MIDI file")
            plot = gr.Plot(label="Piano roll")
            audio = gr.Audio(label="Audio (needs SOUNDFONT env var)", type="filepath")
            info = gr.Textbox(label="Notes per instrument")
    btn.click(run, [model, bpm, density, inst, length, temp, top_p, hard, seed], [midi, plot, audio, info])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    demo.launch(share=ap.parse_args().share)