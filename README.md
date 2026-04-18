# Connect 4 AlphaZero on RTX 4070

[![Deploy to GCP VM](https://github.com/kv244/MLTrain/actions/workflows/deploy.yml/badge.svg)](https://github.com/kv244/MLTrain/actions/workflows/deploy.yml)
[![Lint Codebase](https://github.com/kv244/MLTrain/actions/workflows/lint.yml/badge.svg)](https://github.com/kv244/MLTrain/actions/workflows/lint.yml)

This project implements the AlphaZero algorithm to train a neural network to play Connect 4. The training loop is heavily optimized for modern NVIDIA GPUs (like the RTX 4070) by using batched MCTS, mixed-precision training, and Tensor Core acceleration.

## Live Demo

**Game:** https://c4star.com

**Admin analytics dashboard:** https://c4star.com/admin/`<ADMIN_TOKEN>`
_(token is set via the `ADMIN_TOKEN` environment variable on the server — see `.env`)_

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for training)
- NVIDIA Drivers: Version 525+

Install the required Python libraries:
```bash
pip install torch numpy openvino
```

> **Note:** `openvino` is only required if you want to run inference on CPU using the ONNX export path. Training uses PyTorch + CUDA only.

## How It Works

The project is split into six files:
- `model.py`: Defines the `AlphaNet` dual-headed neural network architecture (policy and value heads) using PyTorch.

### Why ResNet blocks?
- ResNet-style residual blocks (the `ResBlock` class) enable deeper networks while avoiding vanishing gradients.
- Skip connections help stabilize training and speed up convergence by letting the network learn residual corrections.
- For a structured game like Connect 4, the deeper representation helps capture local and global board patterns more effectively.

- `mcts.py`: Contains the Connect 4 game logic and the Monte Carlo Tree Search (MCTS) implementation.
- `self_play.py`: Implements batched self-play — runs 128 games in parallel, collecting one batched GPU call per MCTS simulation step.
- `train.py`: The main training script. It orchestrates self-play data generation and network training.
- `play.py`: A script to play against a trained model checkpoint (PyTorch `.pt` or ONNX via OpenVINO) in your terminal.
- `export_onnx.py`: Converts a trained PyTorch checkpoint to ONNX format for CPU inference.

## 1. Training the Model

The training process involves the AI playing games against itself to generate data, and then learning from that data.

### How to Start Training

To begin the training loop, simply run:
```bash
python train.py
```
The script will automatically detect and use your CUDA-enabled GPU. All output is mirrored to `train_recovery.log` in the working directory (appended on restart, so history is preserved across runs).

### 🧠 Training & Memory Management

The training loop is optimized for efficiency on high-end consumer GPUs:
- **VRAM Recovery**: After every champion evaluation, the script explicitly deletes the evaluation model and clears the CUDA cache via `torch.cuda.empty_cache()`. This prevents OOM (Out of Memory) errors during long-running training sessions.
- **Batched MCTS**: Self-play is performed in parallel batches to maximize Tensor Core utilization.
- **Batched Evaluation**: Evaluation against the champion is now batched (100+ games at once), reducing gating time by ~15x compared to sequential play.
- **High-Throughput Self-Play**: `PARALLEL_GAMES` is tuned to 128 to fully saturate the RTX 4070's compute units.

### 🚀 Hardware Acceleration & Benchmarking

The project now includes a suite of tools to utilize heterogeneous hardware (RTX 4070 + Intel NPU/CPU):

- **Inference Benchmarking**: Use `benchmark_inference.py` to identify the lowest-latency backend for your specific hardware.
- **ONNX Runtime GPU**: Leveraging `onnxruntime-gpu` for CUDA-accelerated inference.
- **Multi-Backend support**: `play.py` automatically selects the fastest backend (NPU/CPU for single moves, GPU for batched search) based on real-world latency profiles.

### What to Expect

You will see output indicating the training progress for each iteration:
```
[2026-04-18 12:00:00] Using device: cuda
[2026-04-18 12:00:00] torch.compile: disabled on Windows due to Triton compatibility
[2026-04-18 12:00:01] [  0] +4,096 states  buffer=4,096  eps=0.25  |  loss=1.9876  policy=1.9455  value=0.0421
[2026-04-18 12:00:01]           → saved checkpoint_0000.pt
[2026-04-18 12:00:02] [  1] +4,096 states  buffer=8,192  eps=0.25  |  loss=1.8123  policy=1.7901  value=0.0222
...
```
- `+4,096 states`: Number of new game states added to the replay buffer in this iteration.
- `buffer=...`: Total size of the replay buffer.
- `loss=...`: The average training loss, which should generally decrease over time.

Checkpoints are saved periodically (e.g., `checkpoint_0000.pt`, `checkpoint_0010.pt`, etc.) in the same directory.

### When is it Done?

The training script runs for a fixed number of cycles, defined by `TOTAL_ITERATIONS` in `train.py` (default is 1500). It will stop automatically when finished. The "best" model is typically one of the later checkpoints, where the training loss has stabilized at a low value.

## 2. Playing a Game

Once you have a trained checkpoint file, you can play against the AI.

### Option A — PyTorch (GPU or CPU)

Use the `play.py` script, pointing it to the checkpoint you want to use. For example, to play against the model from iteration 190:

```bash
python play.py --model checkpoint_0190.pt
```

By default, the AI plays first as 'X'. If you want to play first, use the `--human-first` flag:

```bash
python play.py --model checkpoint_0190.pt --human-first
```

You will be prompted to enter a column number (0-6) to make your move.

### Option B — ONNX / OpenVINO (CPU inference, no GPU required)

You can export a trained checkpoint to ONNX and run it on any machine — no NVIDIA GPU needed.

**Step 1: Export to ONNX**
```bash
python export_onnx.py --checkpoint checkpoint_0190.pt --output model.onnx
```

**Step 2: Play using OpenVINO**
```bash
python play.py --model model.onnx
```

When `play.py` detects a `.onnx` file it automatically switches to OpenVINO for inference on CPU.

## Deploy

CPU-only inference with OpenVINO on a GCP VM is the right call — that's exactly what the ONNX export is for.

The deployment flow is:

```
Local: train → export_onnx.py → model.onnx
                                     ↓
                         Upload to GCS bucket (or commit to Flask repo)
                                     ↓
GCP VM: pull Flask app → download model → restart service
```

### Step 1: Export & store the model

On your local machine after training:
```bash
python export_onnx.py --checkpoint checkpoint_0190.pt --output model.onnx
gsutil cp model.onnx gs://<YOUR_BUCKET>/model.onnx
```

### Step 2: Set up the Flask app as a systemd service on the VM

SSH into the VM and create `/etc/systemd/system/connect4.service`:

```ini
[Unit]
Description=Connect4 Flask App
After=network.target

[Service]
User=<VM_USER>
WorkingDirectory=/home/<VM_USER>/mltrain
ExecStart=/home/<VM_USER>/mltrain/.venv/bin/gunicorn --workers 1 --bind 127.0.0.1:5000 --timeout 120 --graceful-timeout 60 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable it:
```bash
sudo systemctl enable connect4
sudo systemctl start connect4
```

> **Important — keep `--workers 1`**: `_bg_update_state` (admin background regeneration status) and the flask-limiter rate counters are stored in process memory. With `--workers 2+`, each worker has its own independent copy — a POST to worker A sets state that worker B never sees, causing the admin UI to show "Generation failed" even when the image generated successfully. If traffic grows enough to need multiple workers, migrate both to Redis (`LIMITER_STORAGE_URI`) before bumping the worker count.

> **Note:** The service uses the full venv path for `ExecStart` — systemd does not source `activate`. Ubuntu's system-managed Python requires a venv; the venv only needs to be created once on the VM.

### Step 3: GitHub Actions workflow

In your Flask web app repo, use the `.github/workflows/deploy.yml` provided in the codebase.

**Key Hardening**:
- **SSH Fingerprints**: The workflow uses hardcoded fingerprints for the production VM (`34.124.251.132`) to ensure stable CI/CD connections and prevent MITM verification flakes.

### GitHub Secrets

Go to your Flask repo → **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret | Value |
|---|---|
| `GCP_SSH_PRIVATE_KEY` | Full contents of your local `~/.ssh/id_ed25519` (including `-----BEGIN...-----` and `-----END...-----` lines) |
| `GCP_VM_HOST` | VM's external IP |
| `GCP_VM_USER` | Your Unix username on the VM (GCP converts dots to underscores, e.g. `user.name@gmail.com` → `user_name`) |

### Setting up the SSH key

If you don't have a key pair yet, generate one:
```bash
ssh-keygen -t ed25519 -C "connect4-deploy"
# Press Enter for all prompts (no passphrase)
cat ~/.ssh/id_ed25519      # private key → paste into GCP_SSH_PRIVATE_KEY secret
cat ~/.ssh/id_ed25519.pub  # public key → add to GCP VM
```

Add the public key to the VM via **GCP Console → Compute Engine → VM instances → click your VM → Edit → SSH Keys → Add item**.

### Troubleshooting Common Deployment Errors

#### ❌ Error: Permission denied (publickey)
This usually means the SSH key in GitHub Secrets does not match the `authorized_keys` file on the VM, or there is a **Username Mismatch**.
1.  **Check the User**: In GitHub Secrets, `GCP_VM_USER` must be `razvan_petrescu` (underscore).
2.  **Verify the Key**: Ensure the private key in `GCP_SSH_PRIVATE_KEY` matches the public key in `~/.ssh/authorized_keys` on the VM.
3.  **Rotate the key** if needed — see _Setting up the SSH key_ above. Never commit private keys to the repository; `id_*` and `*.pem` are blocked by `.gitignore`.

#### ❌ Error: 502 Bad Gateway
This means the Gunicorn/Flask app failed to start.
1.  **Missing Dependencies**: Run `sudo journalctl -u connect4 -n 50` on the VM. If you see `ModuleNotFoundError: No module named 'PIL'`, ensure `Pillow` is in your `requirements.txt`.
2.  **Port 5000 Bind**: Ensure no other process is using port 5000 (`sudo ss -tlnp | grep 5000`).

### 🛡️ Production Authentication & User Management

**The Dot vs. Underscore Issue**:
GCP often maps email-based SSH keys to a user with a dot (e.g., `razvan.petrescu`). However, the application and service are configured to run as `razvan_petrescu` (underscore).
*   **Best Practice**: Always target the **underscore** user for deployment.
*   **Manual Fix**: Always append the public key to `/home/razvan_petrescu/.ssh/authorized_keys` manually rather than relying solely on the GCP Console's "SSH Keys" metadata table, which can be inconsistent with OS Login.

### Firewall rule

**GCP Console → VPC Network → Firewall → Create Firewall Rule:**

| Field | Value |
|---|---|
| Name | `allow-connect4` |
| Direction | Ingress |
| Source IPv4 ranges | Cloudflare IP ranges only (see `cloudflare.com/ips`) |
| Protocols and ports | TCP, port `443` (HTTPS via Cloudflare proxy) |

### VM spec notes

- **2 vCPUs, 4 GB RAM (e2-medium)**: OpenVINO is efficient on CPU-only — inference will be fast for Connect 4.
- **No GPU**: Make sure the Flask app loads the model via OpenVINO (`openvino.runtime`) or `onnxruntime`, not PyTorch.
- **Checkpoints**: Do not commit `.pt` files to git — use a GCS bucket instead.

### Rate limiting & DoS protection

Flask has no built-in DoS protection — every request triggers MCTS inference and burns CPU. Protect the VM with two layers:

#### Layer 1: nginx (one-time VM setup)

Install and configure nginx as a reverse proxy in front of Flask. SSH into the VM once and run:

```bash
sudo apt install nginx -y
sudo nano /etc/nginx/sites-available/connect4
```

Paste this config:

```nginx
# Resolve real visitor IP from Cloudflare proxy
set_real_ip_from 173.245.48.0/20;
set_real_ip_from 103.21.244.0/22;
set_real_ip_from 103.22.200.0/22;
set_real_ip_from 103.31.4.0/22;
set_real_ip_from 141.101.64.0/18;
set_real_ip_from 108.162.192.0/18;
set_real_ip_from 190.93.240.0/20;
set_real_ip_from 188.114.96.0/20;
set_real_ip_from 197.234.240.0/22;
set_real_ip_from 198.41.128.0/17;
set_real_ip_from 162.158.0.0/15;
set_real_ip_from 104.16.0.0/13;
set_real_ip_from 104.24.0.0/14;
set_real_ip_from 172.64.0.0/13;
set_real_ip_from 131.0.72.0/22;
real_ip_header CF-Connecting-IP;

limit_req_zone $binary_remote_addr zone=one:10m rate=60r/m;

server {
    listen 80;
    server_name c4star.com www.c4star.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name c4star.com www.c4star.com;

    ssl_certificate     /etc/ssl/cloudflare/origin.pem;
    ssl_certificate_key /etc/ssl/cloudflare/origin.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    limit_req zone=one burst=30 nodelay;

    location / {
        proxy_pass         http://127.0.0.1:5000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $remote_addr;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
}
```

Then enable it:

```bash
sudo ln -s /etc/nginx/sites-available/connect4 /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl start nginx
```

This limits each IP to 60 requests/minute (burst of 30). Update the GCP firewall to open port `80` and remove public access to port `5000` — only nginx should be publicly reachable.

> **Note:** nginx is infrastructure, not app code — it only needs to be set up once on the VM and is unaffected by GitHub Actions deploys.

#### Layer 2: flask-limiter (in the Flask app)

Add `flask-limiter` to your Flask app as a backstop. 

> **Important (Production):** If using multi-worker Gunicorn, `memory://` storage will not sync limits across workers. You **must** use a Redis URL (e.g., `redis://localhost:6379`) in your `LIMITER_STORAGE_URI` environment variable for consistent rate limiting.

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])
```

Add `flask-limiter` to your `requirements.txt`.

#### Layer 3: GCP billing alert

Set a budget alert so you're notified before costs spiral:
**GCP Console → Billing → Budgets & alerts → Create budget**

Set a threshold (e.g. $20/month) — GCP will email you before you're surprised. The e2-medium is a fixed ~$25/month with no autoscaling, so the real risk is CPU overload slowing the app, not an unbounded bill.

### 🚀 Future Roadmap & TODOs
- [x] **Secure Connection (HTTPS):** Domain `c4star.com` registered and proxied through Cloudflare. Origin secured with a Cloudflare Origin Certificate on nginx (Full Strict SSL mode). HTTP redirects to HTTPS.
- [x] **Monetization:** Ko-fi donation button (`ko-fi.com/c4star`) shown post-game. Facebook share button with pre-populated result message. Carbon Ads application pending.
- [ ] **User Authentication:** Add a login system to restrict access and save player game statistics.
- [ ] **Move History:** Add a feature to download or replay past games from the UI.
- [x] **Real-time Analytics:** BigQuery player analytics implemented — tracks visits, games, win/loss, moves per IP. Admin dashboard at `/admin/<token>`.
- [x] **Global Player Map:** Geo-IP welcome message now done browser-side (geolocation-db.com whitelisted in CSP) so the client's real IP is used. Includes wallpaper renewal countdown.
- [x] **Database Integration:** BigQuery replaces the CSV-only telemetry system for structured per-player analytics.
- [x] **Dynamic Environment:** Successfully implemented via `background_manager.py`. The system automatically generates and rotates high-fidelity cyberpunk backgrounds using the Gemini API and Vertex AI Imagen once a week, keeping the UI fresh and modern.
- [x] **Root Cause Analysis (RCA):** Completed detailed analysis of the April 2026 deployment outages.

## Version History

### [v2.1.2] - 2026-04-18

#### Web App — Bug Fixes
- **"Generation failed" false negative** (`app.py`, systemd service): root cause was `--workers 2` in the gunicorn service file. `_bg_update_state` is in-process memory — with two workers, admin POST hits worker A but `bg_status` polls hit worker B (which has no state), producing a spurious failure message even when the image generated successfully. Fixed by enforcing `--workers 1`. Documented the constraint in the README with a note on the Redis migration path for future scaling.
- **Concurrent startup/admin background race** (`app.py`): the startup stale-background check called `update_background()` directly without setting `_bg_update_state["running"]`, so a manual admin trigger arriving during boot would race with the startup update over the same `.tmp` file. Fixed: startup update now routes through `_bg_update_state` so the admin endpoint correctly sees it as already running and blocks the duplicate.
- **`num_workers` DataLoader crash** (`train.py`): reverted `num_workers` from 4 to 0. On Windows, `spawn`-based DataLoader workers re-import the `train` module, re-executing module-level CUDA initialisation (model allocation, checkpoint load, replay buffer load) in each worker — causing OOM/conflicts and silent crashes with no iteration output in the log.
- **Iteration log line lost on crash** (`train.py`): added `flush=True` to the `end=""` per-iteration print so the line is written to `train_recovery.log` immediately, even if the process crashes before the loss figures are appended.

### [v2.1.1] - 2026-04-18

#### Web App
- **Background image cache-bust** (`app.py`, `script.js`): `/api/geoip` now returns `bg_mtime` (the file's unix mtime). On every page load, `script.js` overrides `document.body.style.background` with `/static/cyberpunk_bg.png?v={mtime}`, so the browser fetches a fresh image whenever `update_background()` writes a new file. Previously the browser cached the old image indefinitely even after a successful admin regeneration.

#### Training (`train.py`)
- **`persistent_workers=True`** on the DataLoader: prevents 4 worker processes from being spawned and torn down on every iteration (costly `spawn` overhead on Windows). Workers now stay alive across iterations.
- **`scheduler.step()` runs unconditionally**: moved outside the `if len(memory) >= BATCH_SIZE` block so the LR schedule advances every iteration regardless of buffer size. Previously the milestones at iter 700/1000 could fire one step late on a fresh run.
- **Line-buffered log file** (`buffering=1`): `train_recovery.log` is opened with line buffering so each line is flushed immediately; no output is lost if the process crashes mid-buffer.
- **Unused imports removed**: `random`, `_history_to_training_data`, `Connect4`, `print_board`, `numpy` were imported but never referenced.

#### Docs (`README.md`)
- Fixed stale values: `self_play.py` description updated from 64 → 128 parallel games; `TOTAL_ITERATIONS` default updated from 200 → 1500; nginx rate-limit description corrected to 60 req/min burst 30; sample output updated with timestamps and `eps=` field.

### [v2.1.0] - 2026-04-18

#### Hardware & Performance
- **RTX 4070 acceleration via ONNX Runtime** (`play.py`, `self_play.py`, `requirements.txt`): added support for `onnxruntime-gpu`. The `play.py` script now supports multiple backends (`--backend auto|pytorch|openvino|onnx-gpu|onnx-cpu`) and intelligently selects the fastest one for single-move play (preferring NPU/CPU to avoid PCIe latency).
- **Batched Evaluation** (`self_play.py`, `train.py`): Moved champion evaluation from a sequential loop to a fully batched implementation (`run_batched_evaluation`). This plays all 100+ gating games simultaneously on the GPU, yielding a ~15x speedup for the evaluation phase.
- **Dell XPS 16 / RTX 4070 tuning** (`train.py`): Increased `PARALLEL_GAMES` from 64 to 128 and enabled 4-worker data loading (`num_workers=4`) to fully saturate high-core-count CPUs and high-end mobile GPUs.
- **Inference Benchmarking** (`benchmark_inference.py`): New utility to measure and compare latency across PyTorch (CUDA/CPU), OpenVINO (NPU/GPU/CPU), and ONNX Runtime (CUDA/CPU).
- **Rigor update**: Increased `EVAL_SIMS` from 50 to 200 in `train.py` to break tactical plateaus and ensure only significantly stronger models are promoted to "Champion."

### [v2.0.0] - 2026-04-18

#### Web App — Features

- **Think Intensity slider** (`index.html`, `script.js`, `app.py`): New slider in the control panel lets players choose how many MCTS simulations the AI runs (100–2000, step 100, default 400). Works independently of the Easy/Medium/Hard difficulty setting — difficulty controls move randomness, intensity controls search depth. Slider is locked during a game. Server-side cap raised to 2000 (was 1200) in both `/api/move` and `/api/assess`; adaptive boost ceiling updated to match.

- **Hall of Fame** (`bigquery_tracker.py`, `app.py`, `index.html`, `script.js`, `style.css`): When a player wins, a modal appears after 1.8 s offering to save their result to a hall-of-fame table (`connect4.win_records` in BigQuery). Saved record contains name, difficulty, simulations, moves, IP address, and timestamp. Name is pre-filled from `localStorage` on return visits. The most recent winner is fetched on every page load (`/api/recent_winner`, 60-second cache) and displayed in the welcome toast in gold with a 🏆 prefix.

- **Welcome message localisation** (`app.py`, `script.js`): The welcome toast now renders in the visitor's primary language. A new `/api/welcome_strings?country=France` endpoint asks Gemini Flash to translate a fixed set of UI strings (greeting, subtitle, stat labels, winner line) into the appropriate language. Translations are cached in-memory per country for the lifetime of the process — Gemini is only called once per country. Singapore and unknown/empty countries default to English without touching Gemini. The `initWelcomeMessage` function is restructured as two phases: geo lookup first (Phase 1), then translations + stats + winner + geoip in parallel (Phase 2).

- **Piece style variety** (`style.css`, `script.js`): Six inner-ring styles are now randomly selected at the start of each game by adding a class to `#c4Board`. Styles: `circuit` (current dashed ring, default), `scanner` (radar sweep arc + centre blip), `cross` (crosshair lines in a circle ring), `hex` (spinning semi-transparent hexagon with `filter:drop-shadow` outline), `diamond` (spinning kite shape), `pulse` (sonar-ping ring that expands outward and fades, no rotation — player 1 at 1.7 s, player 2 at 2.0 s). Style changes on every Restart. All breathing/sizzle/drop animations and the `chip-charged` electric-flash override are unaffected.

#### Web App — Bug Fixes

- **Menace sting now fires regardless of soundtrack state** (`script.js`): `playMenace()` was gated on `AudioEngine.isPlaying`, but the soundtrack auto-stops 10 seconds after game start — so the sting effectively never played during real games. Guard removed from both call sites (bad-move assessment score ≤ 2, and AI confidence > 0.65); `setIntensity()` retains its guard since it requires the live drone filter. Consistent with `playSwoosh()` which has never had an `isPlaying` guard.

- **Facebook share button** (`index.html`, `script.js`): `fbShare()` was defined in an inline `<script>` block and called via `onclick=""`. Both are blocked by the `script-src 'self'` CSP header (no `'unsafe-inline'`), so the function was never defined and clicks silently failed. Moved to `script.js` and wired via `addEventListener`. Also removed `onclick` attribute.

- **Wallpaper renewal countdown** (`app.py`, `script.js`): Backend changed `math.ceil` to `int()` (floor) — previously a freshly updated image (age 0.1 days) would incorrectly round up to 7 days. Frontend now suppresses the note entirely when `days_left === 0` (image is stale, renewal already attempted at startup).

- **`win_records` BigQuery table not created** (`bigquery_tracker.py`): `_win_table_ref` was missing from the `global` declaration in `init()`, causing Python to create a local variable instead of updating the module-level one. The module-level `_win_table_ref` stayed `None` for the lifetime of the process, so `_ensure_win_table()` silently failed, `record_win()` generated malformed SQL, and `/api/recent_winner` always returned `{"winner": null}`. Fixed: `_win_table_ref` added to the `global` statement.

#### Training

- **Log file encoding** (`train.py`): `open("train_recovery.log", "a")` raised `UnicodeEncodeError` on Windows (cp1252 default) when log lines contained Unicode characters such as `→`. Fixed by adding `encoding="utf-8"`.

### [v1.9.3] - 2026-04-17

#### Training (`train.py`)
- **Persistent logging restored**: Training output is now mirrored to `train_recovery.log` via a `_Tee` stdout wrapper instead of relying on shell redirection (`>> train_recovery.log`), which had silently stopped working. Log is opened in append mode so restarts preserve history.
- **Code comments**: Added targeted inline comments explaining non-obvious implementation choices — `weights_only=True` security rationale, `last_epoch` fast-forward on resume, `set_to_none=True` GPU perf benefit, dataloader wrap-around, and the `end=""` print pattern.

### [v1.9.2] - 2026-04-13

#### Web App (`script.js`)
- **"AI is thinking" permanent hang — 3 root causes fixed:**
  - **Missing UI cleanup on invalid AI move**: `triggerAiMove()` did a bare `return` when `getLowestEmptyRow(col) === -1` (AI returned a full column), leaving the board permanently disabled with the "AI is thinking…" badge. Fixed: now calls `endGame("AI Error")`.
  - **Stale auto-hint timer race (Easy difficulty)**: `updateTurnUI()` scheduled `getHint()` 500 ms after the human's turn started. If `fetchAssessment` (MCTS + Gemini) took longer than 500 ms — which it routinely does — the timer fired while `currentPlayer` hadn't flipped yet, causing a spurious `/api/move` request to reach the server concurrently with the real AI move. Fixed: timer ID saved in `_hintTimerId`; `clearTimeout(_hintTimerId)` added at the top of `handleColumnClick` alongside the existing `AbortController` abort.
  - **No fetch timeout on `/api/move`**: Without a timeout, any server-side hang (e.g. OpenVINO stall) left the client waiting forever. Fixed: added `signal: AbortSignal.timeout(90000)` — 90-second watchdog on the AI move fetch.

### [v1.9.1] - 2026-04-13

#### Web App
- **Error 502 fix**: cap MCTS sims at 1200 server-side (was 5000); adaptive boost ceiling now matches; frontend hard drops from 2000→800, medium 800→400 to prevent gunicorn timeouts on contested mid-game positions.
- **Facebook share button fix**: replaced deprecated `FB.ui()` + SDK dependency with plain `window.open()` sharer URL; removed FB SDK script tag, `fb-root` div, `fbAsyncInit`; cleaned up CSP accordingly.
- **Difficulty lock mid-game**: AI difficulty selector is now disabled during a game (`startGame` disables, `endGame` re-enables); added `:disabled` visual style (opacity 0.4).

### [v1.9.0] - 2026-04-12

#### Web App
- **Dynamic drop audio** (`audio-engine.js`): `playSwoosh(row)` now scales with drop depth. Bottom-row pieces land with a deep sub-thud (60–180 Hz sine, 0.43 s decay); top-row pieces are a light airy hiss. ±10% pitch jitter makes every drop sound unique.
- **Random piece visuals** (`style.css`, `script.js`): ring spin speed randomised per piece (2.5–5.5 s) via `--ring-speed` CSS custom property. 15% of pieces trigger a `.chip-charged` electric-flash effect on placement (white-blue filter pulse + fast ring burst).
- **Facebook JS SDK share** (`index.html`, `script.js`, `app.py`): Share button posts a pre-populated win/loss/draw message with move count and `#Connect4AI` hashtag via `FB.ui()`. CSP updated for `connect.facebook.net`, `graph.facebook.com`, and `frame-src https://www.facebook.com`.
- **HTTPS & Custom Domain:** Site live at `https://c4star.com` via Cloudflare proxy with Full Strict SSL. Cloudflare Origin Certificate installed on nginx. HTTP auto-redirects to HTTPS.
- **nginx Rate Limiting Fix:** Rate limit raised from 5 req/min to 60 req/min (burst 30) to support full games without 503 errors. Real visitor IP now extracted from `CF-Connecting-IP` header — previously all users shared one Cloudflare IP bucket.
- **Difficulty levels** (`app.py`, `script.js`): Easy (2/3 random moves, auto-hints), Medium (1/3 random), Hard (full MCTS). Level description shown below selector.
- **BigQuery difficulty tracking** (`bigquery_tracker.py`, `app.py`): `easy_games`, `medium_games`, `hard_games` columns added via `ALTER TABLE … ADD COLUMN IF NOT EXISTS`. Admin dashboard updated with 3 new stat cards and columns.
- **Ko-fi + Facebook share post-game:** Ko-fi donation button and FB share shown together after each game ends.
- **Model cold-start fix** (`app.py`): ONNX preload moved to module level so Gunicorn workers compile the model at startup, not on first request.
- **"AI is thinking" loop fix** (`script.js`): `AbortController` cancels any pending hint fetch when the player clicks a column, eliminating the race between hint and move requests on a single Gunicorn sync worker.
- **Bug fixes** (`app.py`): `mcts_probs` field name unified to `probs` (random-move path was returning a different key, breaking the heatmap). Stats cache write made thread-safe with a dedicated lock. Removed unused `urllib.request` / `json as _json` imports and dead `OpenVINOModel.eval()` / `policy_output` / `value_output` attributes. `activeEffect` dead state removed from `win-effects.js`.
- **Deploy backup fix** (`deploy.yml`): Cleanup runs before backup. Retains last backup. Excludes `.venv.backup` and `*.pt.*` files to prevent 800 MB+ bloat.

### [v1.8.0] - 2026-04-11/12

#### Training
- **MCTS `expand()` uniform fallback**: Fixed a silent bug where `sum_p == 0` (float32 softmax underflow on policy logits) caused `expand()` to add no children, leaving a non-terminal node as a perpetual leaf — caused a 5-hour evaluation hang at iter 480.
- **`evaluate_model` hang guard**: Replaced `while True` with `for _ in range(42)` + explicit validity and `None` guards to prevent infinite loops.
- **MCTS tree reuse**: Removed `and chosen_child.children` guard in `self_play.py` — previously discarded the entire accumulated tree after almost every move because newly chosen children are unexpanded leaves. Tree now correctly carries over visit counts across moves.
- **LR recovery**: Lowered base LR to `5e-4` and added `MultiStepLR` schedule (×0.1 at iter 700 and 1000) to stabilise late-stage training.
- **Training resumed** from `checkpoint_best.pt` (iter 460) after value head saturation detected at iter 910.

#### Web App
- **Piece FX overhaul** (`style.css`, `script.js`): spinning dashed inner ring (CW/CCW per player), drop-in + sizzle animation on placement, breathing glow on settled pieces, fixed-position spark particle burst on each drop, randomised landing bounce using CSS custom properties `--bounce-h1` / `--bounce-h2`.
- **Audio Engine v2** (`audio-engine.js`): replaced clangy random square-wave arpeggios with a structured atmospheric soundscape — sparse triangle pluck melody (8-step pattern with rests), chord pad wash (E3+B3 → A3+D4), soft sine kick + whisper hi-hat, 3-oscillator sub drone (E1 sawtooth + detuned square + B1 triangle fifth). BPM lowered to 128.
- **Move assessment** (`app.py`): fixed star rating collapse (was `temperature=0` → only 1 or 5 stars; now `temperature=1.0`). Gemini prompt now includes played column, optimal column, and % confidence match for generative, context-aware comments.
- **Geo-IP** (`app.py`, `script.js`): geolocation-db.com added to CSP `connect-src`; browser now calls it directly so the client's real IP is used. `/api/geoip` simplified to return only the wallpaper renewal countdown. Welcome toast shows on every page load (sessionStorage guard removed).
- **BigQuery analytics** (`bigquery_tracker.py`, `app.py`): new `connect4.player_stats` table on `gen-lang-client-0269772194`. Tracks IP, country, first/last seen, visits, games, wins, draws, total moves. `MERGE` upsert on every page load and game end. All BQ calls are fire-and-forget daemon threads.
- **Admin dashboard** (`/admin/<token>`): protected by `ADMIN_TOKEN` env var (404 on mismatch). Shows grand-total stat cards, daily new-visitor table (last 30 days), and full per-IP breakdown with win% bar and new/returning badge.
- **`google.genai` migration** (`app.py`, `background_manager.py`): migrated from the deprecated `google.generativeai` and `vertexai` packages to the unified `google.genai` SDK. This resolves the June 2025 deprecation of the legacy Vertex AI image generation SDK. The background renewal process now uses the unified client with `vertexai=True` to access Imagen 3.0.
- **Security**: removed accidentally committed SSH private key (`id_deploy_final`); added `id_*` and `*.pem` patterns to `.gitignore`.

### [v1.7.0] - 2026-04-10
- **Training Recovery**: Diagnosed value head saturation in checkpoint 910 (value always ~1.0, uniform policy). Rolled back to `checkpoint_best.pt` (iter 460), cleared the contaminated replay buffer, and lowered base LR to `5e-4` for a clean recovery run.
- **Geo-IP Proxy**: Added `/api/geoip` server-side proxy route in `app.py` to resolve the geolocation welcome message failing due to CSP `default-src 'self'` blocking direct client-side calls to `geolocation-db.com`. Frontend now calls `/api/geoip` (same-origin).
- **Restored `print_board`**: Re-added the missing `print_board` utility to `mcts.py`, which is imported by `visualize.py` and `play.py`.

### [v1.6.0] - 2026-04-09
- **Pipeline & API Stability pass**: Implemented 15 critical bug fixes (including 8 suggested by Claude):
    - **Visualisation Guard**: Added `root is None` check in `visualize_mcts.py` to prevent crashes.
    - **LR Schedule Recovery**: Fixed `train.py` to correctly restore the `MultiStepLR` state on resume.
    - **Optimizer Consistency**: Removed hardcoded LR resets on resume.
    - **Improved Gating**: Updated `train.py` to prefer `checkpoint_best.pt` on resume.
    - **Draw Accounting**: Fixed binomial test truncation by using `round(wins)`.
    - **Training Logic**: Moved `scheduler.step()` inside the training block; removed redundant checkpoint save block.
    - **Windows Atomic Save**: Switched to `Path.replace()` in `background_manager.py`.
    - **Redundancy Cleanup**: Streamlined `export_onnx.py` and `app.py` (removed unused variables).
    - **API Robustness**: Added 8s timeout to Gemini API calls and input validation guards (Full Board / Invalid Move) in `app.py`.
    - **Code Cleanliness**: Consolidated imports (global `re`) and removed redundant logic branches.
 
### [v1.5.0] - 2026-04-09
- **Generative AI Assessments**: Switched to Gemini-powered evaluation comments for 1-5 star ratings, replacing hardcoded strings with atmospheric, atmospheric personality.
- **Visual Legibility**: Added a pulsing neon blue halo to highlight the most recent moves for both computer and player.
- **Audio Housekeeping**: Implemented an auto-stop timer (10s) for the background soundtrack to ensure a clean atmospheric intro without repetitive loops.

### [v1.4.0] - 2026-04-06
- **Deployment Self-Backup**: Added an automatic `tar` archive step to the GCP deployment harness (`deploy.yml`) to capture the "live" state before pulling new code.
- **Archive Rotation**: Implemented a retention policy to keep only the 5 most recent source-code backups on the VM, preventing storage bloat.
- **Expert-Tier Diagnostics**: Integrated `visualize.py` and `eval_models.py` into the testing workflow to verify Checkpoint 410's stabilized value trajectory and sharpened board focus.

### [v1.3.1] - 2026-04-06
- **Runtime Strength Boost**: Raised inference simulation depth to **800** (Medium) and **2000** (Hard).
- **Dynamic Simulation Cap**: Increased server-side hard cap from 2048 to **5000** (NPU-optimized) to allow deeper search in high-complexity positions.
- **Improved Engine Defaults**: Standardized **800** simulations across CLI tools (`play.py`, `mcts.py`).

### [v1.3.0] - 2026-04-05
- **AlphaZero Tuning**: Scaled training parallelization to **128 parallel games** and **400 sims** per self-play move.
- **Tuned Exploration**: Reduced `c_puct` from 1.5 to **1.0** for sharper tactical exploitation and set `temp_threshold` to **12** for more competitive endgame data.
- **Balanced Opening Curriculum**: Pre-seeded self-play games with **2–4 random moves** for diverse state coverage.
- **Robust Champion Gating**: Increased model evaluation threshold to **100 games** at a 55% win-rate for promotion.

### [v1.2.0] - 2023-04-05
- **GCP Environment Upgrades**: Implemented **Vertex AI Imagen** integration in `background_manager.py` with atomic `.tmp` → `rename` file saving.
- **Environment Parity**: Added comprehensive security and configuration documentation to `.env.example`.

### [v1.1.0] - 2026-04-05
- **Interactive UI Overhaul**: Added **win-cell highlighting**, **move history** (2-move undo), and **hints**.
- **Engine Visuals**: Implemented real-time **MCTS Probability Heatmap** and **Adaptive Simulation Budget** (dynamic sims).
- **Multi-Level Difficulty**: Integrated a custom difficulty selector into the frontend.

### [v1.0.0] - 2026-04-04
- **Initial Release**: Core AlphaZero Connect 4 training loop with RTX 4070 optimizations.