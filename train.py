"""
train.py — Master loop ("Study Session")

Alternates between:
    Phase 1 — Self-Play: 64 games run in parallel; every MCTS simulation step
              batches all leaf evaluations into one model(states) call of shape
              (≤64, 3, 6, 7), saturating the RTX 4070 Tensor Cores.
    Phase 2 — Training:  sample from the replay buffer and update weights.

RTX 4070 optimisations:
    • Batched MCTS leaf evaluation (64 games × num_sims GPU calls → N× throughput)
    • AMP (torch.amp.autocast) for FP16 Tensor Core throughput
    • GradScaler for numerically stable FP16 training
    • Large replay buffer (50 k) to decorrelate training samples
"""

import os
import glob
import random
import datetime
from collections import deque

import torch
import torch.nn.functional as F

from model import AlphaNet
from self_play import run_batched_self_play, _history_to_training_data
from mcts import run_mcts_simulations, Connect4

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PARALLEL_GAMES       = 128       # games played simultaneously per self-play phase
TRAIN_STEPS_PER_ITER = 10        # gradient updates per iteration
BATCH_SIZE           = 512
NUM_SIMS             = 400       # Higher-quality trees for stronger models
REPLAY_BUFFER_SIZE   = 50_000
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
TOTAL_ITERATIONS     = 1500
CHECKPOINT_EVERY     = 10
EVAL_GAMES           = 100       # Reduced variance in champion gating
EVAL_EVERY           = 20

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
pretrained_checkpoints = sorted(glob.glob("checkpoint_*.pt"))
if pretrained_checkpoints:
    latest_ckpt = pretrained_checkpoints[-1]
    print(f"[{get_timestamp()}] Resuming from {latest_ckpt}...")
    ckpt_data = torch.load(latest_ckpt, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt_data:
        model.load_state_dict(ckpt_data['model_state_dict'])
    if 'optimizer_state_dict' in ckpt_data:
        optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
    if 'scaler_state_dict' in ckpt_data:
        scaler.load_state_dict(ckpt_data['scaler_state_dict'])
    if 'iteration' in ckpt_data:
        start_iteration = ckpt_data['iteration'] + 1

    # Reset LR — the saved LR of 1e-5 is too low to continue learning effectively
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

best_ckpt_path = pretrained_checkpoints[-1] if pretrained_checkpoints else None
import numpy as np

# LR schedule: drop by 10× at iteration 100, again at 150.
# Keeps updates aggressive early, then stabilises loss in late training.
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[700, 1000], gamma=0.1
)

import sys
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
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device.type):
        policy_logits, value = model(states)

        log_probs   = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_p * log_probs, dim=1).mean()
        value_loss  = F.mse_loss(value.float(), target_v) # FIX 12
        loss        = policy_loss + value_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_model(current_model, best_model, device, n_games=EVAL_GAMES):
    """
    Play games between current model (+1) and best model (-1).
    Returns win rate for current model (wins=1, draws=0.5).
    """
    current_model.eval()
    best_model.eval()
    wins = 0

    for i in range(n_games):
        game = Connect4()
        # Alternate who goes first
        current_player_at_start = 1 if i % 2 == 0 else -1
        game.current_player = current_player_at_start # FIX 11
        
        while True:
            # Use 50 sims for fast evaluation
            active_model = current_model if game.current_player == 1 else best_model
            mcts_probs = run_mcts_simulations(
                game, active_model, device,
                num_sims=50, temperature=0, add_dirichlet_noise=False
            )
            move = int(np.argmax(mcts_probs))
            r, c = game.play(move)

            if game.check_win(r, c):
                if game.current_player == -1: # Current model just moved and won
                    wins += 1
                break
            if not game.get_valid_moves():
                wins += 0.5
                break
                
    return wins / n_games

# ── Master loop ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for iteration in range(start_iteration, TOTAL_ITERATIONS):

        # ── Phase 1: Self-Play (64 games, batched GPU calls) ────────────────
        # Each simulation step evaluates up to 64 leaf nodes in one model() call.
        game_data = run_batched_self_play(
            model, device,
            num_games=PARALLEL_GAMES,
            num_sims=NUM_SIMS,
        )
        memory.extend(game_data)

        print(f"[{get_timestamp()}] [{iteration:3d}] +{len(game_data):,} states  buffer={len(memory):,}", end="")

        # ── Phase 2: Training ─────────────────────────────────────────────────
        if len(memory) >= BATCH_SIZE:
            total_loss = total_p = total_v = 0.0
            
            dataset = ReplayBufferDataset(memory)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )
            
            data_iter = iter(dataloader)
            for _ in range(TRAIN_STEPS_PER_ITER):
                try:
                    states, target_p, target_v = next(data_iter)
                except StopIteration:
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

            if iteration % CHECKPOINT_EVERY == 0:
                path = f"checkpoint_{iteration:04d}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, path)
                print(f"[{get_timestamp()}]           → saved {path}")

                # Persistent Buffer: ensures self-play data survives restarts
                torch.save(list(memory), BUFFER_PATH)
                print(f"[{get_timestamp()}]           → updated {BUFFER_PATH}")
            # ── Phase 3: Champion Gating ──────────────────────────────────────
            if iteration > 0 and iteration % EVAL_EVERY == 0 and best_ckpt_path:
                print(f"[{get_timestamp()}] Evaluation against {best_ckpt_path}...")
                eval_model = AlphaNet().to(device)
                eval_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
                eval_model.load_state_dict(eval_ckpt['model_state_dict'])
                
                win_rate = evaluate_model(model, eval_model, device)
                print(f"[{get_timestamp()}] Evaluation result: {win_rate*100:.1f}% win rate")
                
                if win_rate > 0.55:
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
                # First iteration saves as the initial best
                best_ckpt_path = "checkpoint_best.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, best_ckpt_path)

        scheduler.step()

