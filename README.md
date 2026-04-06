# Connect 4 AlphaZero on RTX 4070

[![Deploy to GCP VM](https://github.com/kv244/MLTrain/actions/workflows/deploy.yml/badge.svg)](https://github.com/kv244/MLTrain/actions/workflows/deploy.yml)
[![Lint Codebase](https://github.com/kv244/MLTrain/actions/workflows/lint.yml/badge.svg)](https://github.com/kv244/MLTrain/actions/workflows/lint.yml)

This project implements the AlphaZero algorithm to train a neural network to play Connect 4. The training loop is heavily optimized for modern NVIDIA GPUs (like the RTX 4070) by using batched MCTS, mixed-precision training, and Tensor Core acceleration.

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
- `self_play.py`: Implements batched self-play — runs 64 games in parallel, collecting one batched GPU call per MCTS simulation step.
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
The script will automatically detect and use your CUDA-enabled GPU.

### 🧠 Training & Memory Management

The training loop is optimized for efficiency on high-end consumer GPUs:
- **VRAM Recovery**: After every champion evaluation, the script explicitly deletes the evaluation model and clears the CUDA cache via `torch.cuda.empty_cache()`. This prevents OOM (Out of Memory) errors during long-running training sessions.
- **Batched MCTS**: Self-play is performed in parallel batches to maximize Tensor Core utilization.

### What to Expect

You will see output indicating the training progress for each iteration:
```
Using device: cuda
torch.compile: enabled
[  0] +4,096 states  buffer=4,096  |  (warming up, need 512 samples)
[  1] +4,096 states  buffer=8,192  |  loss=1.9876  policy=1.9455  value=0.0421
          → saved checkpoint_0000.pt
[  2] +4,096 states  buffer=12,288 |  loss=1.8123  policy=1.7901  value=0.0222
...
```
- `+4,096 states`: Number of new game states added to the replay buffer in this iteration.
- `buffer=...`: Total size of the replay buffer.
- `loss=...`: The average training loss, which should generally decrease over time.

Checkpoints are saved periodically (e.g., `checkpoint_0000.pt`, `checkpoint_0010.pt`, etc.) in the same directory.

### When is it Done?

The training script runs for a fixed number of cycles, defined by `TOTAL_ITERATIONS` in `train.py` (default is 200). It will stop automatically when finished. The "best" model is typically one of the later checkpoints, where the training loss has stabilized at a low value.

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
ExecStart=/home/<VM_USER>/mltrain/.venv/bin/gunicorn -w 1 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable it:
```bash
sudo systemctl enable connect4
sudo systemctl start connect4
```

> **Note:** The service uses the full venv path for `ExecStart` — systemd does not source `activate`. Ubuntu's system-managed Python requires a venv; the venv only needs to be created once on the VM.

### Step 3: GitHub Actions workflow

In your Flask web app repo, use the `.github/workflows/deploy.yml` provided in the codebase.

**Key Hardening**:
- **SSH Fingerprints**: The workflow uses hardcoded fingerprints for the production VM (`34.124.246.40`) to ensure stable CI/CD connections and prevent MITM verification flakes.

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
2.  **Verify the Key**: Ensure the private key in `GCP_SSH_PRIVATE_KEY` was copied directly from the local `id_deploy_final` file (not the chat terminal).
3.  **Manual Authorization**: If still failing, re-run the manual authorization command:
    ```bash
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJUOGRtwL6yFjNCAA0e/v+ttBM9gzwzfTH3Tk/rLVUQD connect4-deploy" >> ~/.ssh/authorized_keys
    ```

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
| Source IPv4 ranges | `0.0.0.0/0` |
| Protocols and ports | TCP, port `5000` (or `80` if using nginx) |

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
limit_req_zone $binary_remote_addr zone=connect4:10m rate=5r/m;

server {
    listen 80;

    location / {
        limit_req zone=connect4 burst=10 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:5000;
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

This limits each IP to 5 requests/minute (burst of 10). Update the GCP firewall to open port `80` and remove public access to port `5000` — only nginx should be publicly reachable.

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
- [x] **Secure Connection (HTTPS):** Registered a domain name and set up SSL certificates via Let's Encrypt (Certbot).
- [ ] **User Authentication:** Add a login system to restrict access and save player game statistics.
- [ ] **Monetization Strategy:** Consider adding subtle ads or a premium model to cover hosting costs.
- [ ] **Move History:** Add a feature to download or replay past games from the UI.
- [ ] **Real-time Analytics:** Show how many users are currently connected and playing.
- [ ] **Global Player Map:** Add a window showing which country players are connecting from (GeoIP).
- [ ] **SQLite Integration:** Implement a robust database for game results and telemetry to replace the current CSV system.
- [x] **Dynamic Environment:** Successfully implemented via `background_manager.py`. The system automatically generates and rotates high-fidelity cyberpunk backgrounds using the Gemini API and Vertex AI Imagen once a week, keeping the UI fresh and modern.
- [x] **Root Cause Analysis (RCA):** Completed detailed analysis of the April 2026 deployment outages.

## Version History

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