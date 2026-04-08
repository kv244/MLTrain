"""
mcts.py — The "Brain" (Monte Carlo Tree Search)

Optimized for Connect 4:
    • Vectorized board representation (numpy)
    • Policy/Value network integration (AlphaNet)
    • UCB1-based selection (c_puct for balance)
    • Dirichlet noise for exploration in self-play
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── Internal board logic ──────────────────────────────────────────────────────

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0

    def play(self, col):
        if col < 0 or col > 6: return None
        for row in range(5, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player *= -1
                self.move_count += 1
                return row, col
        return None

    def get_valid_moves(self):
        valid = []
        for col in range(7):
            if self.board[0, col] == 0:
                valid.append(col)
        return valid

    def check_win(self, row, col):
        # Last player who moved was the previous player
        player = self.board[row, col]
        if player == 0: return False
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Forward
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            # Backward
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            if count >= 4: return True
        return False

    def clone(self):
        new_game = Connect4()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game


def board_to_tensor(game: Connect4) -> torch.Tensor:
    """
    Encodes the board state into a (3, 6, 7) tensor for AlphaNet.
    Channel 0: Player 1 pieces
    Channel 1: Player 2 pieces
    Channel 2: All ones if Player 1 to move, zeroes otherwise
    """
    tensor = np.zeros((3, 6, 7), dtype=np.float32)
    tensor[0] = (game.board == 1)
    tensor[1] = (game.board == -1)
    if game.current_player == 1:
        tensor[2] = 1.0
    return torch.from_numpy(tensor)


# ── MCTS Tree Nodes ───────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, game: Connect4, parent=None):
        self.game = game # Full game state
        self.parent = parent
        self.children = {} # move -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.terminal_value = None # None if not terminal else float

    @property
    def q_value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.terminal_value is not None

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_move = -1
        best_node = None
        
        sqrt_parent_visits = np.sqrt(self.visit_count) if self.visit_count > 0 else 1.0

        for move, child in self.children.items():
            # Standard AlphaZero PUCT: -Q + U
            # (where Q is child's value from child's perspective, so -Q is parent's perspective)
            # OR as in our current backprop, if child.q_value is child's perspective, parent needs -child.q_value.
            score = -child.q_value + c_puct * child.prior * (sqrt_parent_visits / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_move = move
                best_node = child
                
        return best_move, best_node

    def expand(self, policy_probs):
        valid_moves = self.game.get_valid_moves()
        # Mask and re-normalize policy
        sum_p = 0.0
        for move in valid_moves:
            sum_p += policy_probs[move]
            
        if sum_p > 0:
            for move in valid_moves:
                child_game = self.game.clone()
                r, c = child_game.play(move) 
                child = MCTSNode(child_game, parent=self)
                child.prior = policy_probs[move] / sum_p
                
                # Check terminal status during expansion
                if child_game.check_win(r, c):
                    # Previous player (parent_player) won.
                    # Value from perspective of current player (child_game.current_player) is -1.0.
                    child.terminal_value = -1.0
                elif not child_game.get_valid_moves():
                    child.terminal_value = 0.0 # Draw
                    
                self.children[move] = child

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            # AlphaZero flips the value for the parent (their perspective)
            self.parent.backpropagate(-value)


# ── Dirichlet noise helper ────────────────────────────────────────────────────

def _add_dirichlet_noise(node: MCTSNode, epsilon: float = 0.25):
    """Adds Dirichlet noise to root priors to encourage exploration (AlphaZero)."""
    if not node.children:
        return

    alpha = 1.0
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    
    for i, move in enumerate(actions):
        node.children[move].prior = (1 - epsilon) * node.children[move].prior + epsilon * noise[i]


# ── MCTS search ───────────────────────────────────────────────────────────────

def run_mcts_simulations(
    game: Connect4, 
    model: torch.nn.Module, 
    device: torch.device, 
    num_sims: int = 400, 
    c_puct: float = 1.41,
    temperature: float = 1.0, 
    add_dirichlet_noise: bool = True, 
    return_root: bool = False,
    epsilon: float = 0.25,
) -> np.ndarray:
    """Runs N simulations from the current game state and returns move probabilities."""
    root = MCTSNode(game.clone())
    
    # 1. Expand the root with a network call
    state = board_to_tensor(root.game).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(device.type):
        policy_logits, value = model(state)
        
    policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    root.expand(policy_probs)

    # 2. Add exploration noise
    if add_dirichlet_noise:
        _add_dirichlet_noise(root, epsilon=epsilon)

    # 3. Backpropagate the initial evaluation
    root.backpropagate(value.float().item())

    # 4. Standard MCTS iterations
    for _ in range(num_sims - 1):
        node = root
        
        # Selection
        while not node.is_leaf():
            _, node = node.select_child(c_puct)
            if node.is_terminal(): 
                break
            
        # Leaf evaluation
        if node.is_terminal():
            node.backpropagate(node.terminal_value)
            continue

        state = board_to_tensor(node.game).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast(device.type):
            policy_logits, value = model(state)
            
        node.expand(F.softmax(policy_logits, dim=1).squeeze().cpu().numpy())
        node.backpropagate(value.float().item())

    # ── TACTICAL OVERRIDE ──
    # 1. Immediate Win
    win_move = game.get_winning_move(game.current_player)
    if win_move is not None:
        if return_root: return None
        probs = np.zeros(7, dtype=np.float32)
        probs[win_move] = 1.0
        return probs

    # 2. Must Block (Opponent can win)
    opponent = -game.current_player
    block_move = game.get_winning_move(opponent)
    if block_move is not None:
        if return_root: return None
        probs = np.zeros(7, dtype=np.float32)
        probs[block_move] = 1.0
        return probs

    # 5. Result Extraction
    visits = np.zeros(7, dtype=np.float64)
    for move, child in root.children.items():
        visits[move] = child.visit_count

    if temperature <= 0.01:
        probs = np.zeros(7, dtype=np.float32)
        probs[np.argmax(visits)] = 1.0
    else:
        visits = visits ** (1.0 / temperature)
        probs = (visits / visits.sum()).astype(np.float32)

    return (probs, root) if return_root else probs
