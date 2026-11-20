# AI Music Studio website: setup (Windows + VS Code)

## What gets added to your project folder (`D:\Music Generation Project`)
```
server.py               NEW   FastAPI backend (loads your models, generates music, serves the website)
requirements-web.txt    NEW   backend packages
generate.py             REPLACE  same as yours + an optional progress callback (used by the progress bar)
web\                    NEW   the React website (src\, test\, dist\ = already built, package.json ...)
```
Nothing else changes. No retraining.

## Run it (no Node.js needed, the site is prebuilt in `web\dist`)
```powershell
cd "D:\Music Generation Project"
.\.venv\Scripts\Activate.ps1
pip install -r requirements-web.txt
python server.py
```
Open http://localhost:8000 . Check http://localhost:8000/api/health : it should list `lstm` and `transformer`,
and `"device": "cuda"` if your RTX is used.

## Change the website (needs Node.js 18+ or 20 LTS)
```powershell
cd web
npm install
npm run dev        # live-reload site on http://localhost:5173 (talks to the server on :8000)
npm run build      # rebuilds web\dist (what server.py serves)
```
Name, tagline, links and the one-click presets are in `web\src\config.js`.

## Automatic check (server must be running)
```powershell
cd web
npm run test:e2e
```
Prints PASS/FAIL for: connect, load controls, generate, piano roll, mixer, history, mute.
(Audio is stubbed in this test, so it cannot check the sound itself. Always listen once in Chrome.)

## Where the models come from (first match wins)
| Setting | Meaning |
|---|---|
| `$env:RELEASE_DIR="hf_release"` | folder made by `python hf_export.py` (safetensors) |
| `$env:HF_REPO="your-name/repo"` (+ `$env:HF_TOKEN` if private) | download from Hugging Face |
| nothing | `checkpoints\lstm.pt` and `checkpoints\transformer.pt` |

Other settings: `PORT`, `MAX_LENGTH` (default 2048 tokens), `MAX_QUEUE` (default 6), `CORS_ORIGINS`.

## API (for reference)
- `GET /api/health`, `GET /api/config`
- `POST /api/generate` body `{model, bpm, density 0-3, instruments[], length 256-2048, temperature, top_p, seed?, hard_instruments}` -> `{job_id}`
- `GET /api/jobs/{id}` -> `{status: queued|running|done|error, progress, position?, result?, error?}`
- `result` = `{seed, model, bpm, duration, bars, tracks:{group:[[start,end,pitch,velocity]...]}, stats, midi_base64, elapsed}`

## Good to know
- Tempo is applied at playback (the models write music in beats and bars), so the BPM slider changes playback speed.
- The Transformer is slower than the LSTM. On a CPU-only host, keep clips short (`MAX_LENGTH=512`).
- Free hosts sleep when idle; the site shows "waking the server up" and retries by itself.
