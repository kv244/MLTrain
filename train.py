"""
train.py — Master loop ("Study Session")

Alternates between:
    Phase 1 — Self-Play: 128 games run in parallel; every MCTS simulation step
              batches all leaf evaluations into one model(states) call of shape
              (≤128, 3, 6, 7), saturating the RTX 4070 Tensor Cores.
    Phase 2 — Training:  sample from the replay buffer and update weights.

RTX 4070 optimisations:
    • Batched MCTS leaf evaluation (128 games × num_sims GPU calls → N× throughput)
    • AMP (torch.amp.autocast) for FP16 Tensor Core throughput
    • GradScaler for numerically stable FP16 training
    • Large replay buffer (50 k) to decorrelate training samples
"""

import os
import glob
import datetime
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

from model import AlphaNet
from self_play import run_batched_self_play, run_batched_evaluation
import bigquery_tracker

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PARALLEL_GAMES       = 128       # Optimized for RTX 4070 throughput; increased from 64 to better saturate Tensor Cores.
TRAIN_STEPS_PER_ITER = 10        # gradient updates per iteration
BATCH_SIZE           = 512
NUM_SIMS             = 400       # Higher-quality trees for stronger models
REPLAY_BUFFER_SIZE   = 50_000
LEARNING_RATE        = 5e-4  # Lowered from 1e-3 for recovery run from checkpoint_best (iter 460)
WEIGHT_DECAY         = 1e-4
TOTAL_ITERATIONS     = 1500
CHECKPOINT_EVERY     = 10
EVAL_GAMES           = 100       # Reduced variance in champion gating
EVAL_EVERY           = 20
EVAL_SIMS            = 800       # Increased from 200 to 800; use deeper search to distinguish between very strong models.
EPSILON_FLOOR        = 0.12      # Raised from 0.05: keeps self-play games diverse in late training.

# Set to a specific checkpoint path to override auto-latest resume (e.g. rollback recovery).
# Set to None to resume from the most recent checkpoint as normal.
RESUME_FROM          = None

# ── Device & model ────────────────────────────────────────────────────────────
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"[{get_timestamp()}] Using device: {device}")

model     = AlphaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler    = torch.amp.GradScaler(device.type)

start_iteration = 0
pretrained_checkpoints = sorted(
    glob.glob("checkpoint_[0-9]*.pt"),
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)

resume_ckpt = RESUME_FROM if (RESUME_FROM and os.path.exists(RESUME_FROM)) else (
    pretrained_checkpoints[-1] if pretrained_checkpoints else None
)

if resume_ckpt:
    print(f"[{get_timestamp()}] Resuming from {resume_ckpt}...")
    ckpt_data = torch.load(resume_ckpt, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt_data:
        model.load_state_dict(ckpt_data['model_state_dict'])
    if 'optimizer_state_dict' in ckpt_data:
        optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
    if 'scaler_state_dict' in ckpt_data:
        scaler.load_state_dict(ckpt_data['scaler_state_dict'])
    if 'iteration' in ckpt_data:
        start_iteration = ckpt_data['iteration'] + 1

if os.path.exists("checkpoint_best.pt"):
    best_ckpt_path = "checkpoint_best.pt"
elif pretrained_checkpoints:
    best_ckpt_path = pretrained_checkpoints[-1]
else:
    best_ckpt_path = None
# LR schedule: drop by 10× at iteration 700, again at 1000.
# Keeps updates aggressive early, then stabilises loss in late training.
# last_epoch=start_iteration-1 fast-forwards the scheduler state on resume so
# the LR reflects where we actually are in training, not epoch 0.
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[700, 1000], gamma=0.1,
    last_epoch=start_iteration - 1
)

import sys

# Mirror every print() to train_recovery.log without changing call sites.
# Append mode so restarts don't clobber earlier runs.
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files: f.write(data)
    def flush(self):
        for f in self.files: f.flush()

_log_fh = open("train_recovery.log", "a", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)

# torch.compile gives ~20-30% extra throughput via kernel fusion on PyTorch 2.x.
# Falls back gracefully on older installs.
if sys.platform != "win32":
    try:
        model = torch.compile(model)
        print(f"[{get_timestamp()}] torch.compile: enabled")
    except Exception:
        print(f"[{get_timestamp()}] torch.compile: unavailable (PyTorch < 2.0), skipping")
else:
    print(f"[{get_timestamp()}] torch.compile: disabled on Windows due to Triton compatibility")

memory: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

BUFFER_PATH = "replay_buffer.pt"
if os.path.exists(BUFFER_PATH):
    try:
        print(f"[{get_timestamp()}] Loading replay buffer from {BUFFER_PATH}...")
        saved_memory = torch.load(BUFFER_PATH, map_location="cpu", weights_only=True)
        memory.extend(saved_memory)
        print(f"[{get_timestamp()}] Buffer loaded: {len(memory):,} states")
    except Exception as e:
        print(f"[{get_timestamp()}] Warning: Could not load replay buffer: {e}")

# ── Utilities ─────────────────────────────────────────────────────────────────

def _human_games_to_training_data(games_data: list) -> list:
    """Convert human-win game records from BigQuery to (state, policy, value) tuples.

    Uses a one-hot policy (the actual move played) as a weak supervised signal.
    These samples are lower quality than MCTS-derived data but cover positions
    the model lost from, which self-play never generates.
    """
    from mcts import Connect4, board_to_tensor
    all_data = []
    for g in games_data:
        moves      = g["move_sequence"]
        winner_str = g["winner"]
        human_pl   = g["human_player"]
        game       = Connect4()
        history    = []

        for col in moves:
            if not isinstance(col, int) or col < 0 or col > 6:
                break
            # For AI-turn positions in a human-win game the AI played a losing move.
            # One-hot policy would train the model to reproduce that mistake, so use
            # uniform policy and let the value head (-1.0) carry the corrective signal.
            if game.current_player != human_pl and winner_str == "human":
                probs = np.ones(7, dtype=np.float32) / 7
            else:
                probs = np.zeros(7, dtype=np.float32)
                probs[col] = 1.0
            history.append((board_to_tensor(game), probs, game.current_player))
            result = game.play(col)
            if result is None:
                break
            r, c = result
            if game.check_win(r, c):
                break

        winner = 0 if winner_str == "draw" else (human_pl if winner_str == "human" else -human_pl)
        for state_t, mcts_p, player in history:
            value = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
            all_data.append((
                state_t,
                torch.from_numpy(mcts_p).float(),
                torch.tensor([value], dtype=torch.float32),
            ))
    return all_data


# Seed the buffer with human-win games from BigQuery so the model trains on
# positions it lost from — positions self-play never generates.
bigquery_tracker.init()
if bigquery_tracker._enabled:
    try:
        print(f"[{get_timestamp()}] Loading human games from BigQuery...")
        _hg_raw  = bigquery_tracker.get_human_games(limit=2000)
        _hg_data = _human_games_to_training_data(_hg_raw)
        if _hg_data:
            memory.extend(_hg_data)
            print(f"[{get_timestamp()}] Added {len(_hg_data):,} states from {len(_hg_raw)} human games")
        del _hg_raw, _hg_data
    except Exception as _e:
        print(f"[{get_timestamp()}] Warning: Could not load human games: {_e}")


class ReplayBufferDataset(torch.utils.data.Dataset):
    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]


# ── Single training step ──────────────────────────────────────────────────────

def train_step(states, target_p, target_v):
    """
    One gradient update.

    Policy loss  — cross-entropy with soft MCTS targets.
                   Uses log_softmax on raw logits (numerically stable;
                   avoids log(softmax(x)) = log(p) where p → 0 can be -inf).
    Value loss   — MSE between predicted and actual game outcome.
    """
    model.train()
    # set_to_none=True skips the memset-to-zero step — marginally faster on GPU.
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device.type):
        policy_logits, value = model(states)

        log_probs   = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_p * log_probs, dim=1).mean()
        value_loss  = F.mse_loss(value.float(), target_v)
        loss        = policy_loss + value_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_model(current_model, best_model, device, n_games=EVAL_GAMES):
    """
    Play games between current model (+1) and best model (-1).
    Returns (wins, n_games) for statistical testing.
    
    OPTIMIZATION: Previously sequential (one-by-one), which had high PCIe overhead on GPU.
    Now uses batched inference to play all games simultaneously, yielding ~15x speedup.
    """
    wins = run_batched_evaluation(
        current_model, best_model, device,
        num_games=n_games, num_sims=EVAL_SIMS
    )
    return wins, n_games

# ── Master loop ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.stats import binomtest

    for iteration in range(start_iteration, TOTAL_ITERATIONS):
        # A) Dirichlet Epsilon Decay
        # Linear decay from 0.25 to EPSILON_FLOOR
        epsilon = max(EPSILON_FLOOR, 0.25 * (1.0 - iteration / TOTAL_ITERATIONS))

        # ── Phase 1: Self-Play (64 games, batched GPU calls) ────────────────
        game_data = run_batched_self_play(
            model, device,
            num_games=PARALLEL_GAMES,
            num_sims=NUM_SIMS,
            epsilon=epsilon
        )
        memory.extend(game_data)

        # end="" so the loss figures are appended on the same line when training runs.
        print(f"[{get_timestamp()}] [{iteration:3d}] +{len(game_data):,} states  buffer={len(memory):,} eps={epsilon:.2f}", end="", flush=True)

        # ── Phase 2: Training ─────────────────────────────────────────────────
        if len(memory) >= BATCH_SIZE:
            total_loss = total_p = total_v = 0.0
            
            # Setup data loader for the current iteration
            dataset = ReplayBufferDataset(list(memory))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,  # Windows spawn would re-execute module-level CUDA init in each worker → OOM.
            )
            
            data_iter = iter(dataloader)
            for _ in range(TRAIN_STEPS_PER_ITER):
                try:
                    states, target_p, target_v = next(data_iter)
                except StopIteration:
                    # Buffer smaller than TRAIN_STEPS * BATCH_SIZE: wrap around.
                    data_iter = iter(dataloader)
                    states, target_p, target_v = next(data_iter)
                    
                states   = states.to(device)
                target_p = target_p.to(device)
                target_v = target_v.to(device)

                loss, p_loss, v_loss = train_step(states, target_p, target_v)
                total_loss += loss
                total_p    += p_loss
                total_v    += v_loss

            n = TRAIN_STEPS_PER_ITER
            print(f"  |  loss={total_loss/n:.4f}"
                  f"  policy={total_p/n:.4f}"
                  f"  value={total_v/n:.4f}")

            # ── Phase 3: Checkpointing & Gating ────────────────────────────────
            if iteration % CHECKPOINT_EVERY == 0:
                path = f"checkpoint_{iteration:04d}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, path)
                print(f"[{get_timestamp()}]           → saved {path}")

                # Persistent Buffer
                torch.save(list(memory), BUFFER_PATH)
                print(f"[{get_timestamp()}]           → updated {BUFFER_PATH}")

            if iteration > 0 and iteration % EVAL_EVERY == 0 and best_ckpt_path:
                print(f"[{get_timestamp()}] Evaluation against {best_ckpt_path}...")
                eval_model = AlphaNet().to(device)
                eval_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
                eval_model.load_state_dict(eval_ckpt['model_state_dict'] if 'model_state_dict' in eval_ckpt else eval_ckpt)
                
                wins, total = evaluate_model(model, eval_model, device)
                # B) Binomial Gating
                # One-sided test: is the current model significant better than 50/50?
                # We round wins to integer for binomtest; draws count as 0.5 but binomtest needs counts.
                # For fairness in binomtest, we treat draws as half-wins.
                res = binomtest(int(wins + 0.5), total, p=0.5, alternative='greater')
                win_rate = wins / total
                print(f"[{get_timestamp()}] Evaluation result: {win_rate*100:.1f}% win rate (p={res.pvalue:.4f})")
                
                if res.pvalue < 0.05:
                    best_ckpt_path = "checkpoint_best.pt"
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, best_ckpt_path)
                    print(f"[{get_timestamp()}]           → NEW CHAMPION SAVED: {best_ckpt_path}")
                del eval_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            elif not best_ckpt_path:
                best_ckpt_path = "checkpoint_best.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                }, best_ckpt_path)

        scheduler.step()

