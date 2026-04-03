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
gsutil cp model.onnx gs://your-bucket/model.onnx
```

### Step 2: Set up the Flask app as a systemd service on the VM

SSH into the VM and create `/etc/systemd/system/connect4.service`:

```ini
[Unit]
Description=Connect4 Flask App
After=network.target

[Service]
User=razvan_petrescu
WorkingDirectory=/home/razvan_petrescu/connect4-web
ExecStart=/home/razvan_petrescu/connect4-web/.venv/bin/python app.py
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

In your Flask web app repo, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GCP VM

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up SSH
        uses: webfactory/ssh-agent@v0.9.0
        with:
          ssh-private-key: ${{ secrets.GCP_SSH_PRIVATE_KEY }}

      - name: Add VM to known hosts
        run: ssh-keyscan -H ${{ secrets.GCP_VM_HOST }} >> ~/.ssh/known_hosts

      - name: Deploy
        run: |
          ssh ${{ secrets.GCP_VM_USER }}@${{ secrets.GCP_VM_HOST }} << 'EOF'
            cd ~/connect4-web
            git pull origin main
            python3 -m venv .venv --system-site-packages 2>/dev/null || true
            .venv/bin/pip install -r requirements.txt
            sudo systemctl restart connect4
          EOF

      - name: Update model (if changed)
        run: |
          ssh ${{ secrets.GCP_VM_USER }}@${{ secrets.GCP_VM_HOST }} \
            "gsutil cp gs://your-bucket/model.onnx ~/connect4-web/model.onnx"
```

### GitHub Secrets

Go to your Flask repo → **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret | Value |
|---|---|
| `GCP_SSH_PRIVATE_KEY` | Full contents of your local `~/.ssh/id_ed25519` (including `-----BEGIN...-----` and `-----END...-----` lines) |
| `GCP_VM_HOST` | VM's external IP |
| `GCP_VM_USER` | Your Unix username on the VM (GCP converts dots to underscores, e.g. `razvan.petrescu@gmail.com` → `razvan_petrescu`) |

### Setting up the SSH key

If you don't have a key pair yet, generate one:
```bash
ssh-keygen -t ed25519 -C "connect4-deploy"
# Press Enter for all prompts (no passphrase)
cat ~/.ssh/id_ed25519      # private key → paste into GCP_SSH_PRIVATE_KEY secret
cat ~/.ssh/id_ed25519.pub  # public key → add to GCP VM
```

Add the public key to the VM via **GCP Console → Compute Engine → VM instances → click your VM → Edit → SSH Keys → Add item**.

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

Add `flask-limiter` to your Flask app as a backstop:

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