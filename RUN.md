# AI Music Studio — Windows Setup & Execution Guide

This document describes how to start, test, and develop the AI Music Studio web application on Windows (Python 3.10 in `.venv`, NVIDIA RTX 3050 Laptop GPU with 4GB VRAM).

---

## 1. Quick Start (Run Production Web App)

The FastAPI server (`server.py`) hosts both the music generation REST API and the prebuilt production React website (`web/dist`) on **http://localhost:8000**.

Open **PowerShell** in the project directory (`D:\Music Generation Project`):

```powershell
# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install web backend dependencies (if not already installed)
pip install -r requirements-web.txt

# 3. Start the FastAPI server
python server.py
```

Now open **http://localhost:8000** in Google Chrome or any modern browser.

### Verify Server Status
- **Health check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
  - Expected: `{"status":"ok","device":"cuda","models":["lstm","transformer"],"queue":0}`
- **Config check**: [http://localhost:8000/api/config](http://localhost:8000/api/config)
  - Returns instrument groups, tempo boundaries, density levels, and defaults.

---

## 2. Frontend Development (Vite Dev Server)

If you wish to modify the React frontend with live reloading:

```powershell
cd web

# 1. Install Node.js packages (Node 18+ or 20 LTS required)
npm install

# 2. Start Vite development server (port 5173 with proxy to port 8000)
npm run dev

# 3. Rebuild production bundle to web/dist (served by server.py)
npm run build
```

---

## 3. Running Automated Tests

### A. End-to-End Test Suite (`npm run test:e2e`)
With `python server.py` running in a separate terminal:

```powershell
cd web
npm run test:e2e
```

**Verifies:**
- Connects to the server and loads controls
- Server badge reports "Online"
- Generation triggers live progress bar
- Result appears with Piano Roll canvas
- Note stats match requested instruments
- Mixer lists requested instruments
- MIDI download and Share buttons exist
- Variation generation appends to session history
- Instrument mute toggle works
- Zero unhandled console errors

### B. Python API Test Suite
Tests generation endpoints directly via HTTP and verifies the generated MIDI files using `pretty_midi`:

```powershell
.\.venv\Scripts\python.exe scratch\test_api.py
```

**Verifies:**
- POST `/api/generate` accepts jobs (HTTP 202)
- Job polling `/api/jobs/{id}` tracks progress (0.0 → 1.0)
- Strict instruments constraint (`hard_instruments=True`) prevents unrequested tracks
- Decoded `midi_base64` parses cleanly in `pretty_midi`
- Input validation returns HTTP 422 for invalid models and out-of-range BPM (e.g. 999)

---

## 4. Code Changes Made During Integration

| File | Change | Reason |
|---|---|---|
| `server.py` | Fixed nested f-string quotes in `load_models()` summary print | Python 3.10 compatibility (`SyntaxError: f-string: unmatched '['`) |
| `web/src/components/Player.jsx` | Added fallback mechanism to `copyLink` using a temporary textarea + `document.execCommand('copy')` | Prevents failures when `navigator.clipboard.writeText` is restricted |
| `web/src/components/PianoRoll.jsx` | Changed canvas inline style width from fixed pixel string to `canvas.style.width = "100%"` | Prevents canvas from holding fixed desktop width and overflowing on mobile viewports |
| `web/src/App.jsx` | Added `min-w-0` to grid columns wrapping Controls and Player | Eliminates CSS Grid horizontal overflow on 375px mobile phone viewports |

> **Note on Training Code:** As required, `config.py`, `common.py`, `models.py`, `dataset.py`, `train.py`, `step1_scan_filter.py`, `step2_tokenize.py`, `validate.py`, `evaluate.py`, and all checkpoint files under `checkpoints/` were untouched.

---

## 5. QA Verification Checklist

| Item | Result | Notes |
|---|---|---|
| Page loads & displays "Online" badge | **PASS** | Green badge displaying `Online · CUDA` |
| Quick-start presets change controls | **PASS** | Tested "Rock band", "Solo piano", "Lo-fi beat", "Cinematic strings", "Synthwave", "Funk groove" |
| Transformer generation & progress bar | **PASS** | Live progress updates from 0% to 100%, status changes from "Composing..." to completed player |
| Autoplay on completion | **PASS** | Sound transport starts automatically upon result rendering (unlocked in click gesture) |
| Piano roll note rendering & seek | **PASS** | Color-coded notes by instrument, moving playhead, pointer click seeks to target bar/second |
| Mixer controls (Mute, Volume, Loop) | **PASS** | Mute toggles icon/label, volume slider updates relative dB, Loop toggles seamless replay |
| BPM tempo slider | **PASS** | Tempo slider dynamically changes BPM setting |
| Download MIDI | **PASS** | Generates valid `.mid` blob, loads cleanly in `pretty_midi` |
| Share link | **PASS** | Generates URL with query parameters (`?m=...&b=...&d=...&i=...&s=...`), restores exact state in new tab |
| New variation & Session History | **PASS** | Appends variation with random seed, clicking previous entry in history reloads track |
| LSTM model generation | **PASS** | Switching to LSTM generates multi-instrument music cleanly in ~7s |
| Responsive layout (375px & desktop) | **PASS** | Mobile phone viewport (375x812) verified with zero horizontal overflow (`scrollWidth = 375px`) |
| Keyboard focus visibility | **PASS** | Visible focus rings (`focus-visible:ring-2`) present on interactive controls |
| Offline server handling | **PASS** | Graceful fallback banner displayed with working "Try again" reconnect action |

---

## 6. Verification Disclosures

- **Hardware Acceleration:** All models loaded and executed on CUDA via the NVIDIA GeForce RTX 3050 Laptop GPU.
- **Audio Verification Note:** Audio context creation, Tone.js node connection graphs, synthesizers, and transport timers were verified through browser instrumentation and automated playback assertions. Physical audio through external laptop speakers was not directly listened to by an acoustic microphone.
