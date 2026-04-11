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
import random

from mcts import Connect4, MCTSNode, board_to_tensor, _add_dirichlet_noise


# ── Single-game self-play (kept for reference / debugging) ───────────────────

def run_self_play_game(
    model: torch.nn.Module,
    device: torch.device,
    num_sims: int = 400,
    c_puct: float = 1.0,
    temp_threshold: int = 12,
    epsilon: float = 0.25,
) -> list:
    """
    Simulates a full Connect 4 game from start to finish
    applying the search parameters provided.
    """
    from mcts import run_mcts_simulations
    game = Connect4()
    states = []
    
    # ── Game Loop ─────────────────────────────────────────────────────────────
    while True:
        # Determine current search temperature:
        # - High (1.0) during opening (up to temp_threshold) for diversity
        # - Low (0.01) thereafter for tactical precision
        temp = 1.0 if game.move_count < temp_threshold else 0.01
        
        # Run MCTS simulations for current position
        probs = run_mcts_simulations(
            game,
            model,
            device,
            num_sims=num_sims,
            temperature=temp,
            add_dirichlet_noise=True,
            epsilon=epsilon
        )
        
        # Save state: (normalized_board, mcts_probs, current_player)
        states.append((board_to_tensor(game), probs, game.current_player))
        
        # Sample move from the MCTS probability distribution
        move = np.random.choice(7, p=probs)
        r, c = game.play(move)
        
        if game.check_win(r, c):
            winner = -game.current_player; break
        if not game.get_valid_moves():
            winner = 0; break

    return _history_to_training_data(states, winner)


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
        
        for i in active:
            if i not in to_expand and roots[i] is not None:
                if roots[i].children:
                    _add_dirichlet_noise(roots[i], epsilon=epsilon)

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
    return visits / visits.sum()


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
