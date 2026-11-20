"""
PART 1 - Web API for the music generator (FastAPI).

    pip install -r requirements-web.txt
    python server.py                 # API: http://localhost:8000/api/health   Website: http://localhost:8000 (if web/dist is built)

Where the models come from (first match wins):
    RELEASE_DIR=hf_release           folder produced by hf_export.py
    HF_REPO=your-name/your-repo      downloaded from the Hugging Face Hub (set HF_TOKEN if the repo is private)
    (nothing set)                    checkpoints/lstm.pt and checkpoints/transformer.pt from your training runs

Optional environment variables:
    CORS_ORIGINS=https://my-site.vercel.app,http://localhost:5173     (default "*")
    MAX_QUEUE=6      MAX_LENGTH=2048      PORT=8000      WEB_DIST=web/dist      TORCH_THREADS=4

Endpoints
    GET  /api/health            -> status, device, loaded models, queue size
    GET  /api/config            -> everything the website needs to build its controls
    POST /api/generate          -> {"job_id": "..."}      (generation runs in a background queue, one job at a time)
    GET  /api/jobs/{job_id}     -> {"status": queued|running|done|error, "progress": 0..1, "position": n, "result": {...}}
"""
import base64
import inspect
import json
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config as C
from common import ControlSpec, ids_to_score, load_tokenizer
from generate import generate_ids
from models import build_model, count_params

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 2048))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 6))
if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

# older copies of generate.py may not have the progress callback; the server still works (just without a live progress bar)
_SUPPORTS_PROGRESS = "progress_cb" in inspect.signature(generate_ids).parameters

MODEL_BLURBS = {
    "lstm": "Recurrent network. Fast, tends to wander a little, never repeats itself too much.",
    "transformer": "Attention-based network. Better long-range structure and cleaner grooves, a bit slower.",
}
DENSITY_LABELS = ["Sparse", "Light", "Busy", "Dense"]
LENGTHS = [{"label": "Short", "tokens": 512}, {"label": "Medium", "tokens": 1024}, {"label": "Long", "tokens": 2048}]
PROGRAM_TO_GROUP = {C.GROUP_PROGRAM[g]: g for g in C.ALL_GROUPS if g != "drums"}

# filled at startup
SPEC: Optional[ControlSpec] = None
TOK = None
REGISTRY: dict = {}          # name -> dict(model, type, params_m, val_loss, epoch)
JOBS: dict = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)      # one generation at a time (GPU / CPU friendly)
_counter = 0


# ----------------------------------------------------------------------------- model loading
def _register(name, model, mtype, val_loss=None, epoch=None):
    model.eval().to(DEVICE)
    REGISTRY[name] = dict(model=model, type=mtype, params_m=round(count_params(model) / 1e6, 2),
                          val_loss=None if val_loss is None else round(float(val_loss), 3), epoch=epoch)


def _load_release(root: Path):
    from safetensors.torch import load_model
    global SPEC, TOK
    SPEC = ControlSpec.load(root / "controls.json")
    TOK = load_tokenizer(root / "tokenizer.json")
    for name in ("lstm", "transformer"):
        d = root / name
        if (d / "config.json").exists() and (d / "model.safetensors").exists():
            info = json.loads((d / "config.json").read_text())
            model = build_model(info["model_type"], info["model_config"])
            load_model(model, str(d / "model.safetensors"))
            _register(name, model, info["model_type"], info.get("val_loss"), info.get("epoch"))


def _load_checkpoints():
    global SPEC, TOK
    SPEC, TOK = ControlSpec.load(), load_tokenizer()
    for name in ("lstm", "transformer"):
        path = C.CHECKPOINT_DIR / f"{name}.pt"
        if path.exists():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(ck["model_type"], ck["config"])
            model.load_state_dict(ck["state_dict"])
            _register(name, model, ck["model_type"], ck.get("val_loss"), ck.get("epoch"))


def load_models():
    release, repo = os.environ.get("RELEASE_DIR"), os.environ.get("HF_REPO")
    if repo and not release:
        from huggingface_hub import snapshot_download
        release = snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN"))
    if release:
        print(f"Loading models from release folder: {release}")
        _load_release(Path(release))
    else:
        print(f"Loading models from {C.CHECKPOINT_DIR}")
        _load_checkpoints()
    if not REGISTRY:
        raise RuntimeError("No models found. Train first (checkpoints/lstm.pt, transformer.pt) or set RELEASE_DIR / HF_REPO.")
    summary = ", ".join(f"{k} ({v['params_m']}M)" for k, v in REGISTRY.items())
    print(f"Ready on {DEVICE}: {summary}")


@asynccontextmanager
async def lifespan(app):
    load_models()
    yield


app = FastAPI(title="Music Generator API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
                   allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- request / helpers
class GenerateRequest(BaseModel):
    model: str = "transformer"
    bpm: float = Field(120, ge=40, le=220)
    density: int = Field(2, ge=0, le=C.N_DENS_BINS - 1)
    instruments: list[str] = Field(default_factory=lambda: ["drums", "bass", "piano"], min_length=1, max_length=5)
    length: int = Field(1024, ge=256, le=MAX_LENGTH)
    temperature: float = Field(0.95, ge=0.5, le=1.5)
    top_p: float = Field(0.95, ge=0.5, le=1.0)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    hard_instruments: bool = True


def _extract_notes(score, bpm):
    """Notes in SECONDS, grouped by instrument: {group: [[start, end, pitch, velocity], ...]}"""
    sec_per_tick = 60.0 / (bpm * score.ticks_per_quarter)
    tracks = {}
    for tr in score.tracks:
        if len(tr.notes) == 0:
            continue
        group = "drums" if tr.is_drum else PROGRAM_TO_GROUP.get(int(tr.program), "piano")
        n = tr.notes.numpy()
        start = n["time"].astype(np.float64) * sec_per_tick
        end = (n["time"].astype(np.float64) + n["duration"]) * sec_per_tick
        rows = np.stack([np.round(start, 3), np.round(end, 3), n["pitch"], n["velocity"]], axis=1)
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        tracks.setdefault(group, []).extend(rows.tolist())
    return tracks


def _run_job(job_id: str, req: GenerateRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        entry = REGISTRY[req.model]
        seed = req.seed if req.seed is not None else random.randrange(1, 2**31 - 1)
        tempo_bin = SPEC.tempo_bin(req.bpm)
        t0 = time.time()

        def progress(done, total):
            job["progress"] = round(min(0.99, done / total), 3)

        ids = generate_ids(entry["model"], entry["type"], SPEC, tempo_bin, req.density, req.instruments,
                           req.length, req.temperature, 0, req.top_p, req.hard_instruments, DEVICE,
                           seed=seed, **({"progress_cb": progress} if _SUPPORTS_PROGRESS else {}))
        score = ids_to_score(ids, TOK, SPEC, tempo_bpm=req.bpm)      # raises ValueError if no valid notes
        try:
            midi_bytes = score.dumps_midi()
        except AttributeError:
            tmp = ROOT / f".tmp_{job_id}.mid"
            score.dump_midi(str(tmp)); midi_bytes = tmp.read_bytes(); tmp.unlink()
        tracks = _extract_notes(score, req.bpm)
        all_notes = [n for rows in tracks.values() for n in rows]
        duration = max((n[1] for n in all_notes), default=0.0)
        counts = SPEC.group_counts(ids)
        npb = SPEC.notes_per_bar(ids)
        job["result"] = {
            "seed": seed, "model": req.model, "bpm": req.bpm, "duration": round(float(duration), 2),
            "bars": int(np.sum(np.asarray(ids) == SPEC.bar_id)), "tracks": tracks,
            "stats": {"notes": {g: int(c) for g, c in zip(C.ALL_GROUPS, counts) if c > 0},
                      "total_notes": len(all_notes), "notes_per_bar": round(float(npb), 1),
                      "measured_density": int(SPEC.dens_bin(npb))},
            "midi_base64": base64.b64encode(midi_bytes).decode("ascii"),
            "elapsed": round(time.time() - t0, 1),
        }
        job["progress"] = 1.0
        job["status"] = "done"
    except ValueError as e:
        job["status"], job["error"] = "error", str(e)
    except Exception as e:                     # never let the worker thread die silently
        job["status"], job["error"] = "error", f"Generation failed: {type(e).__name__}: {e}"


def _cleanup():
    cutoff = time.time() - 30 * 60
    for k in [k for k, j in JOBS.items() if j["created"] < cutoff and j["status"] in ("done", "error")]:
        JOBS.pop(k, None)


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    active = sum(j["status"] in ("queued", "running") for j in JOBS.values())
    return {"status": "ok", "device": DEVICE, "models": list(REGISTRY), "queue": active}


@app.get("/api/config")
def config():
    return {
        "instruments": C.ALL_GROUPS,
        "density_levels": [{"value": i, "label": DENSITY_LABELS[i]} for i in range(C.N_DENS_BINS)],
        "bpm": {"min": 60, "max": 180, "default": 120},
        "lengths": [l for l in LENGTHS if l["tokens"] <= MAX_LENGTH],
        "models": [{"id": k, "label": k.upper() if k == "lstm" else k.capitalize(), "params_m": v["params_m"],
                    "val_loss": v["val_loss"], "epoch": v["epoch"], "blurb": MODEL_BLURBS.get(k, "")}
                   for k, v in REGISTRY.items()],
        "defaults": {"model": "transformer" if "transformer" in REGISTRY else next(iter(REGISTRY)),
                     "instruments": ["drums", "bass", "piano"], "density": 2, "bpm": 120, "length": 1024,
                     "temperature": 0.95, "top_p": 0.95, "hard_instruments": True},
    }


@app.post("/api/generate", status_code=202)
def create_job(req: GenerateRequest):
    global _counter
    if req.model not in REGISTRY:
        raise HTTPException(422, f"Unknown model '{req.model}'. Available: {list(REGISTRY)}")
    bad = [g for g in req.instruments if g not in C.ALL_GROUPS]
    if bad:
        raise HTTPException(422, f"Unknown instruments {bad}. Choose from {C.ALL_GROUPS}")
    with LOCK:
        _cleanup()
        if sum(j["status"] in ("queued", "running") for j in JOBS.values()) >= MAX_QUEUE:
            raise HTTPException(429, "The server is busy. Please try again in a minute.")
        _counter += 1
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "progress": 0.0, "created": time.time(), "order": _counter}
    EXECUTOR.submit(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    out = {"status": job["status"], "progress": job["progress"]}
    if job["status"] == "queued":
        out["position"] = sum(1 for j in JOBS.values() if j["status"] == "queued" and j["order"] < job["order"])
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job.get("error", "Unknown error")
    return out


# website (after `npm run build` in web/); mounted last so /api/* keeps priority
DIST = Path(os.environ.get("WEB_DIST", ROOT / "web" / "dist"))
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
