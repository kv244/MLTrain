import time
import torch
import numpy as np
from mcts import Connect4, run_mcts_simulations
from model import AlphaNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = AlphaNet().to(device)
model.eval()

game = Connect4()
start = time.time()
print("Starting 400 simulations...")
try:
    probs = run_mcts_simulations(game, model, device, num_sims=400, add_dirichlet_noise=True)
    end = time.time()
    print(f"400 simulations took {end - start:.4f} seconds.")
    print(f"Probabilities: {probs}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
