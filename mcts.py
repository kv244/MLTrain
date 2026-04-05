import math
import numpy as np
import torch
import torch.nn.functional as F


# ── Connect 4 game engine ─────────────────────────────────────────────────────

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.current_player = 1          # +1 or -1

    def clone(self):
        g = Connect4()
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g

    def get_valid_moves(self):
        return [c for c in range(7) if self.board[0, c] == 0]

    def play(self, column):
        for row in reversed(range(6)):
            if self.board[row, column] == 0:
                self.board[row, column] = self.current_player
                self.current_player *= -1
                return row, column
        return None                      # column full — shouldn't happen if caller checks

    def check_win(self, r, c):
        p = self.board[r, c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                nr, nc = r + dr * sign, c + dc * sign
                while 0 <= nr < 6 and 0 <= nc < 7 and self.board[nr, nc] == p:
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            if count >= 4:
                return True
        return False


# ── Board printing utility ──────────────────────────────────────────────────

def print_board(game: Connect4):
    """Prints the board with players as X and O."""
    board = np.flipud(game.board) # Flip for intuitive display (row 0 at bottom)
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print("+---------------+")
    for r in range(6):
        row_str = " ".join([symbols[p] for p in board[r]])
        print(f"| {row_str} |")
    print("+---------------+")
    print("| 0 1 2 3 4 5 6 |")
    print()


# ── Board → tensor encoding ───────────────────────────────────────────────────

def board_to_tensor(game: Connect4) -> torch.Tensor:
    """
    Encode the board from the *current player's* perspective into a (3, 6, 7)
    float32 tensor:
        channel 0 — current player's pieces
        channel 1 — opponent's pieces
        channel 2 — constant turn-indicator plane (1.0 if player-1 to move)
    """
    t = np.zeros((3, 6, 7), dtype=np.float32)
    t[0] = (game.board == game.current_player)
    t[1] = (game.board == -game.current_player)
    t[2] = float(game.current_player == 1)
    return torch.from_numpy(t)


# ── MCTS node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ("game", "parent", "prior", "children",
                 "visit_count", "value_sum", "terminal_value")

    def __init__(self, game: Connect4, parent=None,
                 prior: float = 0.0, terminal_value=None):
        self.game           = game
        self.parent         = parent
        self.prior          = prior
        self.children: dict[int, "MCTSNode"] = {}
        self.visit_count    = 0
        self.value_sum      = 0.0
        self.terminal_value = terminal_value  # None → not terminal

    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    # ------------------------------------------------------------------

    def select_child(self, c_puct: float):
        sqrt_n = math.sqrt(self.visit_count)
        best_score = -float("inf")
        best_move, best_child = None, None
        for move, child in self.children.items():
            score = -child.q_value + c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move, best_child = move, child
        return best_move, best_child

    def expand(self, policy_probs: np.ndarray):
        """
        Create child nodes for every valid move.
        policy_probs: length-7 array (softmax output).  Masked and renormalised
        over legal moves before assigning priors.
        """
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return

        priors = policy_probs[valid_moves].astype(np.float64)
        s = priors.sum()
        priors = priors / s if s > 0 else np.ones(len(valid_moves)) / len(valid_moves)

        for prior, move in zip(priors, valid_moves):
            child_game = self.game.clone()
            r, c = child_game.play(move)

            # Did this move end the game?
            terminal_value = None
            if child_game.check_win(r, c):
                # The player who just moved won.
                # From the child's perspective (next player to move), that is a loss.
                terminal_value = -1.0
            elif not child_game.get_valid_moves():
                terminal_value = 0.0  # draw

            self.children[move] = MCTSNode(
                child_game, parent=self,
                prior=float(prior), terminal_value=terminal_value
            )

    def backpropagate(self, value: float):
        """
        Walk up to the root, alternating sign at each level because each parent
        is the *opponent* of its child.
        """
        self.visit_count += 1
        self.value_sum   += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


# ── Dirichlet noise helper ────────────────────────────────────────────────────

def _add_dirichlet_noise(root: MCTSNode, alpha: float = 0.8, epsilon: float = 0.25):
    """
    Add Dirichlet noise to the root's children priors to encourage exploration
    during self-play.

    alpha=0.8: The AlphaZero paper recommends scaling alpha inversely with the
    average number of legal moves (~35 for chess → 0.3, ~7 for Connect 4 → ~0.8).
    A higher alpha produces a flatter noise distribution, giving more meaningful
    exploration across the small action space.
    """
    moves = list(root.children.keys())
    noise = np.random.dirichlet([alpha] * len(moves))
    for move, eta in zip(moves, noise):
        child = root.children[move]
        child.prior = (1.0 - epsilon) * child.prior + epsilon * eta


# ── MCTS search ───────────────────────────────────────────────────────────────

def run_mcts_simulations(
    game:   Connect4,
    model,                   # AlphaNet — assumed to be on `device` and in eval()
    device: torch.device,
    num_sims:            int   = 400,
    c_puct:              float = 1.5,
    temperature:         float = 1.0,
    add_dirichlet_noise: bool  = False,
) -> np.ndarray:
    """
    Run `num_sims` MCTS simulations from `game` and return a length-7
    probability vector over columns.

    Values returned by the network are always from the *current player's*
    perspective.  Negation during backpropagation handles the perspective flip
    between parent and child.
    """
    root = MCTSNode(game.clone())

    # ── Pre-expand root so every simulation starts with a selection step ──
    state = board_to_tensor(root.game).unsqueeze(0).to(device)
    with torch.inference_mode():
        policy_logits, value = model(state)
    policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    root.expand(policy_probs)
    
    if add_dirichlet_noise and root.children: # FIX 16: Move before backprop
        _add_dirichlet_noise(root)

    root.backpropagate(value.item())

    # ── num_sims - 1 further simulations (root eval counted as one) ───────
    for _ in range(num_sims - 1):
        node = root

        # Selection — descend until a leaf
        while not node.is_leaf():
            _, node = node.select_child(c_puct)

        # Leaf is terminal: propagate known outcome
        if node.is_terminal():
            node.backpropagate(node.terminal_value)
            continue

        # Leaf is non-terminal: evaluate with network, then expand
        state = board_to_tensor(node.game).unsqueeze(0).to(device)
        with torch.inference_mode():
            policy_logits, value = model(state)
        policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
        node.expand(policy_probs)
        node.backpropagate(value.item())

    # ── Convert visit counts → probability distribution ───────────────────
    visits = np.zeros(7, dtype=np.float64)
    for move, child in root.children.items():
        visits[move] = child.visit_count

    if temperature == 0:
        probs = np.zeros(7, dtype=np.float32)
        probs[int(np.argmax(visits))] = 1.0
        return probs

    visits **= (1.0 / temperature)
    return (visits / visits.sum()).astype(np.float32)
