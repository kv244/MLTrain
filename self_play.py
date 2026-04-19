"""
self_play.py — The "Gym"

Two modes:
    run_self_play_game()      — single game, batch size 1 (baseline / debug)
    run_batched_self_play()   — N games in parallel, batch size N per GPU call
                                This is the RTX 4070 path.

Batching strategy (leaf parallelism):
    Each simulation step, advance *every* active game's MCTS tree through
    selection until each reaches a leaf node.  Collect all non-terminal
    leaves, stack them into one tensor, call model() once, then distribute
    policy/value back to each game and backpropagate.

    Result: one model(states) call of shape (N, 3, 6, 7) per simulation step
    instead of N calls of shape (1, 3, 6, 7) — N× better Tensor Core utilisation.
"""

import numpy as np
import torch
import torch.nn.functional as F

from mcts import Connect4, MCTSNode, board_to_tensor, _add_dirichlet_noise


# ── Batched self-play ─────────────────────────────────────────────────────────

def run_batched_self_play(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = 64,
    num_sims: int = 400,
    c_puct: float = 1.0,
    temp_threshold: int = 12,
    epsilon: float = 0.25,
) -> list:
    """
    Play `num_games` games simultaneously using one batched GPU call per MCTS
    simulation step.
    """
    model.eval()
    games = [Connect4() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))
    roots = {i: None for i in range(num_games)}
    all_data = []

    while active:
        to_expand = [i for i in active if roots[i] is None]
        if to_expand:
            new_roots = _expand_roots_batched(games, to_expand, model, device, epsilon=epsilon)
            for i, node in new_roots.items():
                roots[i] = node
        
        # NOTE: do NOT re-add Dirichlet noise to reused roots here.
        # Noise is applied once in _expand_roots_batched when each root is first
        # created.  Adding it again to an already partially-explored tree corrupts
        # accumulated visit counts and Q-values (AlphaZero spec: noise at root
        # creation only, not on every subsequent search).

        for _ in range(num_sims - 1):
            to_evaluate = []
            for i in active:
                node = roots[i]
                while not node.is_leaf():
                    _, node = node.select_child(c_puct)
                if node.is_terminal():
                    node.backpropagate(node.terminal_value)
                else:
                    to_evaluate.append((i, node))

            if to_evaluate:
                leaf_states = torch.stack([board_to_tensor(nd.game) for _, nd in to_evaluate]).to(device)
                with torch.inference_mode(), torch.amp.autocast(device.type):
                    policy_logits, values = model(leaf_states)
                leaf_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                for j, (i, node) in enumerate(to_evaluate):
                    node.expand(leaf_probs[j])
                    node.backpropagate(values[j].float().item())

        newly_done = []
        for i in active:
            temperature = 1.0 if move_counts[i] < temp_threshold else 0.01
            probs = _visits_to_probs(roots[i], temperature)

            # Tactical override: immediate win or forced block
            win_move = games[i].get_winning_move(games[i].current_player)
            if win_move is not None:
                probs = np.zeros(7, dtype=np.float64)
                probs[win_move] = 1.0
            else:
                opp = -games[i].current_player
                block_move = games[i].get_winning_move(opp)
                if block_move is not None:
                    probs = np.zeros(7, dtype=np.float64)
                    probs[block_move] = 1.0

            histories[i].append((board_to_tensor(games[i]), probs, games[i].current_player))
            move = int(np.random.choice(7, p=probs))
            r, c = games[i].play(move)
            move_counts[i] += 1

            if games[i].check_win(r, c):
                winner = -games[i].current_player
                all_data.extend(_history_to_training_data(histories[i], winner))
                newly_done.append(i)
            elif not games[i].get_valid_moves():
                all_data.extend(_history_to_training_data(histories[i], 0))
                newly_done.append(i)
            else:
                chosen_child = roots[i].children.get(move)
                # v1.8.0 (2026-04-11): removed `and chosen_child.children` guard.
                # Previously, tree reuse was skipped whenever the child had not yet
                # been expanded (no .children dict) — i.e., every normal leaf case.
                # That meant roots[i] was reset to None after almost every move,
                # throwing away all accumulated visit counts.  Removing the guard
                # lets an unexpanded child become the new root; _expand_roots_batched
                # will expand it at the top of the next iteration as expected.
                if chosen_child:
                    chosen_child.parent = None
                    roots[i] = chosen_child
                else:
                    roots[i] = None
            active = [i for i in active if i not in newly_done]
    return all_data

def run_batched_evaluation(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    device: torch.device,
    num_games: int = 100,
    num_sims: int = 200,
    c_puct: float = 1.0,
) -> float:
    """
    Play `num_games` between two models simultaneously.
    
    RATIONALE: Small-model inference on discrete GPUs is dominated by kernel launch 
    and PCIe overhead. By batching all match games (default 100) together, we
    saturate the GPU and reduce total evaluation time from 5+ mins to <30 seconds.
    """
    model1.eval()
    model2.eval()
    games = [Connect4() for _ in range(num_games)]
    # Match colors: model1 is P1 for even games, P2 for odd games to ensure fairness
    m1_colors = [1 if i % 2 == 0 else -1 for i in range(num_games)]
    
    active = list(range(num_games))
    roots = {i: None for i in range(num_games)}
    wins = 0.0

    while active:
        # 1. Expand roots if needed
        to_expand = [i for i in active if roots[i] is None]
        if to_expand:
            # Batch expansion for all games that need it
            states_batch = torch.stack([board_to_tensor(games[i]) for i in to_expand]).to(device)
            with torch.inference_mode(), torch.amp.autocast(device.type):
                # We need to know which model is active for each game's root expansion
                # But to save GPU calls, we sort them by model
                m1_expand = [idx for idx, game_idx in enumerate(to_expand) if games[game_idx].current_player == m1_colors[game_idx]]
                m2_expand = [idx for idx, game_idx in enumerate(to_expand) if games[game_idx].current_player != m1_colors[game_idx]]
                
                final_probs = np.zeros((len(to_expand), 7), dtype=np.float32)
                final_values = np.zeros(len(to_expand), dtype=np.float32)
                
                if m1_expand:
                    p, v = model1(states_batch[m1_expand])
                    final_probs[m1_expand] = F.softmax(p, dim=1).cpu().numpy()
                    final_values[m1_expand] = v.cpu().numpy().flatten()
                if m2_expand:
                    p, v = model2(states_batch[m2_expand])
                    final_probs[m2_expand] = F.softmax(p, dim=1).cpu().numpy()
                    final_values[m2_expand] = v.cpu().numpy().flatten()

            for j, i in enumerate(to_expand):
                root = MCTSNode(games[i].clone())
                root.expand(final_probs[j])
                root.backpropagate(final_values[j].item())
                roots[i] = root

        # 2. Sequential Simulations (Batched Inference)
        for _ in range(num_sims - 1):
            to_evaluate_m1 = []
            to_evaluate_m2 = []
            games_needing_eval = []
            
            for i in active:
                node = roots[i]
                while not node.is_leaf():
                    _, node = node.select_child(c_puct)
                if node.is_terminal():
                    node.backpropagate(node.terminal_value)
                else:
                    # Determine which model should evaluate this leaf
                    if games[i].current_player == m1_colors[i]:
                        to_evaluate_m1.append(node)
                    else:
                        to_evaluate_m2.append(node)
                    games_needing_eval.append((i, node))

            if games_needing_eval:
                # To maximize throughput, we could stack M1 and M2 and do one call, 
                # but they are different models, so we do two batched calls.
                if to_evaluate_m1:
                    states = torch.stack([board_to_tensor(n.game) for n in to_evaluate_m1]).to(device)
                    with torch.inference_mode(), torch.amp.autocast(device.type):
                        p, v = model1(states)
                    probs = F.softmax(p, dim=1).cpu().numpy()
                    vals = v.cpu().numpy().flatten()
                    for j, node in enumerate(to_evaluate_m1):
                        node.expand(probs[j])
                        node.backpropagate(vals[j].item())
                        
                if to_evaluate_m2:
                    states = torch.stack([board_to_tensor(n.game) for n in to_evaluate_m2]).to(device)
                    with torch.inference_mode(), torch.amp.autocast(device.type):
                        p, v = model2(states)
                    probs = F.softmax(p, dim=1).cpu().numpy()
                    vals = v.cpu().numpy().flatten()
                    for j, node in enumerate(to_evaluate_m2):
                        node.expand(probs[j])
                        node.backpropagate(vals[j].item())

        # 3. Make Moves
        newly_done = []
        for i in active:
            # For evaluation, always use temperature 0 (greedy)
            visits = np.zeros(7, dtype=np.int32)
            for move, child in roots[i].children.items():
                visits[move] = child.visit_count
            
            move = int(np.argmax(visits))
            r, c = games[i].play(move)
            
            if games[i].check_win(r, c):
                # Last player moved and won. 
                # If last player was m1_colors[i], Model 1 wins.
                if -games[i].current_player == m1_colors[i]:
                    wins += 1.0
                newly_done.append(i)
            elif not games[i].get_valid_moves():
                wins += 0.5 # Draw
                newly_done.append(i)
            else:
                # Tree reuse
                chosen_child = roots[i].children.get(move)
                if chosen_child:
                    chosen_child.parent = None
                    roots[i] = chosen_child
                else:
                    roots[i] = None
        
        active = [i for i in active if i not in newly_done]

    return wins

def _expand_roots_batched(games, active, model, device, epsilon=0.25):
    """Batched expansion of MCTS roots with Dirichlet noise initialization."""
    states_batch = torch.stack([board_to_tensor(games[i]) for i in active]).to(device)
    with torch.inference_mode(), torch.amp.autocast(device.type):
        policy_logits, values = model(states_batch)
    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()

    roots = {}
    for j, i in enumerate(active):
        root = MCTSNode(games[i].clone())
        root.expand(policy_probs[j])
        root.backpropagate(values[j].float().item())
        if root.children:
            _add_dirichlet_noise(root, epsilon=epsilon)
        roots[i] = root
    return roots


def _visits_to_probs(root: MCTSNode, temperature: float) -> np.ndarray:
    """Convert MCTS visit counts to a move probability vector of length 7."""
    visits = np.zeros(7, dtype=np.float64)
    for move, child in root.children.items():
        visits[move] = child.visit_count

    if temperature <= 0.01: # Use 0.01 for greedy-like sampling
        probs = np.zeros(7, dtype=np.float64)
        probs[int(np.argmax(visits))] = 1.0
        return probs

    visits = visits ** (1.0 / temperature)
    total = visits.sum()
    if total == 0:
        # Fallback: uniform over all columns (should never happen in normal play)
        return np.ones(7, dtype=np.float64) / 7
    return visits / total


def _history_to_training_data(history, winner):
    """
    Convert a game's move history to labelled training tuples.
    winner: +1, -1, or 0 (draw).
    """
    data = []
    for state_tensor, mcts_probs, player in history:
        value = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
        policy_tensor = torch.from_numpy(mcts_probs).float()

        # Original
        data.append((
            state_tensor,
            policy_tensor,
            torch.tensor([value], dtype=torch.float32),
        ))

        # Horizontal mirror — flip spatial dim (W) of board, reverse policy columns
        # state_tensor shape is (3, 6, 7), flip along dim 2 (width)
        mirrored_state = torch.flip(state_tensor, dims=[2])
        # policy_probs is a 1D tensor of 7 columns, flip it
        mirrored_policy = torch.flip(policy_tensor, dims=[0])
        data.append((
            mirrored_state,
            mirrored_policy,
            torch.tensor([value], dtype=torch.float32),
        ))

    return data
