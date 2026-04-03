"""
visualize_mcts.py — Visualize the MCTS search tree after one move decision

Two views:
    1. Tree diagram  — graphviz tree of visited nodes, sized by visit count,
                       coloured by Q-value (green=good, red=bad).
    2. Column heatmap — bar chart of visit counts and Q-values per column,
                        overlaid on the board state.

Usage:
    python visualize_mcts.py --model checkpoint_0190.pt
    python visualize_mcts.py --model checkpoint_0190.pt --simulations 200 --output-dir ./out
    python visualize_mcts.py --model checkpoint_0190.pt --moves 3 4 3    # replay moves first
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import AlphaNet
from mcts import Connect4, MCTSNode, board_to_tensor, run_mcts_simulations


# ── Model loading (mirrors visualize.py) ─────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path)
    model = AlphaNet().to(device)
    checkpoint = torch.load(str(path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ── Tree walking helper ───────────────────────────────────────────────────────

def collect_nodes(root: MCTSNode, max_depth: int = 3):
    """
    BFS walk of the MCTS tree up to max_depth.
    Returns list of (node, parent_id, move, depth) tuples.
    """
    result = []
    queue = [(root, None, None, 0)]
    node_id = 0
    id_map = {id(root): node_id}

    while queue:
        node, parent_id, move, depth = queue.pop(0)
        nid = id_map[id(node)]
        result.append((node, parent_id, move, depth, nid))

        if depth < max_depth:
            for child_move, child in sorted(node.children.items(),
                                            key=lambda x: -x[1].visit_count):
                child_id = len(id_map)
                id_map[id(child)] = child_id
                queue.append((child, nid, child_move, depth + 1))

    return result


# ── View 1: tree diagram ──────────────────────────────────────────────────────

def visualize_tree(root: MCTSNode, max_depth: int = 3, output_dir: str = "."):
    """
    Render the MCTS tree as a top-down diagram using matplotlib.
    Node size ∝ visit count. Colour = Q-value (green=winning, red=losing).
    """
    nodes = collect_nodes(root, max_depth)

    # Group nodes by depth for layout
    depth_groups: dict[int, list] = {}
    for entry in nodes:
        d = entry[3]
        depth_groups.setdefault(d, []).append(entry)

    # Assign x positions within each depth level
    pos = {}
    for depth, entries in depth_groups.items():
        n = len(entries)
        for i, entry in enumerate(entries):
            nid = entry[4]
            pos[nid] = ((i - (n - 1) / 2) * 2.0, -depth * 2.5)

    fig, ax = plt.subplots(figsize=(max(14, len(nodes) * 0.6), max_depth * 3 + 2))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("MCTS Search Tree (depth {}, root visits={})".format(
        max_depth, root.visit_count), fontsize=13, fontweight="bold")

    cmap = plt.cm.RdYlGn

    # Draw edges first
    for node, parent_id, move, depth, nid in nodes:
        if parent_id is not None:
            x0, y0 = pos[parent_id]
            x1, y1 = pos[nid]
            ax.plot([x0, x1], [y0, y1], color="#aaaaaa", linewidth=1, zorder=1)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"col{move}", fontsize=7, ha="center", va="center",
                    color="#555555", zorder=2)

    # Draw nodes
    max_visits = max(n[0].visit_count for n in nodes) or 1
    for node, parent_id, move, depth, nid in nodes:
        x, y = pos[nid]
        radius = 0.35 + 0.55 * (node.visit_count / max_visits)
        q = node.q_value
        color = cmap((q + 1) / 2)  # map [-1,1] → [0,1]
        circle = plt.Circle((x, y), radius, color=color, zorder=3, linewidth=1.5,
                             edgecolor="black")
        ax.add_patch(circle)

        label = f"v={node.visit_count}\nq={q:+.2f}"
        if depth == 0:
            label = f"ROOT\nv={node.visit_count}"
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5,
                fontweight="bold", zorder=4)

    # Colour legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, shrink=0.6)
    cbar.set_label("Q-value (current player)", fontsize=9)

    plt.tight_layout()
    out = Path(output_dir) / "mcts_tree.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved tree diagram: {out}")
    plt.close()


# ── View 2: column heatmap ────────────────────────────────────────────────────

def visualize_column_heatmap(root: MCTSNode, game: Connect4, output_dir: str = "."):
    """
    Two-panel figure:
      Left  — board state with per-column visit-count bar overlaid
      Right — Q-value per column (how good is each move for the current player)
    """
    visits = np.zeros(7)
    q_vals = np.full(7, np.nan)
    priors = np.full(7, np.nan)

    for move, child in root.children.items():
        visits[move] = child.visit_count
        q_vals[move] = child.q_value
        priors[move] = child.prior

    total = visits.sum() or 1
    visit_pct = visits / total

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MCTS Decision Breakdown", fontsize=13, fontweight="bold")

    # ── Panel 1: board ────────────────────────────────────────────────────────
    ax = axes[0]
    board = np.flipud(game.board)
    symbols = {1: "X", -1: "O", 0: ""}
    colors_map = {1: "#4a90d9", -1: "#e05252", 0: "#f0f0f0"}

    for r in range(6):
        for c in range(7):
            val = board[r, c]
            rect = mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=colors_map[val], edgecolor="#333333", linewidth=1
            )
            ax.add_patch(rect)
            if symbols[val]:
                ax.text(c, r, symbols[val], ha="center", va="center",
                        fontsize=18, fontweight="bold", color="white")

    # Overlay visit % as bar along bottom
    for c in range(7):
        height = visit_pct[c] * 2.5
        ax.bar(c, height, bottom=-2.8, width=0.7,
               color="#2ecc71" if visits[c] == visits.max() else "#95a5a6",
               alpha=0.85, zorder=5)
        ax.text(c, -2.8 + height + 0.05, f"{visit_pct[c]:.0%}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-3.2, 5.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"col {i}" for i in range(7)], fontsize=8)
    ax.set_yticks([])
    ax.set_title("Board + Visit % per column", fontweight="bold")
    ax.set_aspect("equal")

    # ── Panel 2: Q-values ─────────────────────────────────────────────────────
    ax = axes[1]
    valid = ~np.isnan(q_vals)
    cols = np.where(valid)[0]
    bar_colors = ["#2ecc71" if q > 0 else "#e74c3c" for q in q_vals[valid]]
    ax.bar(cols, q_vals[valid], color=bar_colors, edgecolor="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Column", fontweight="bold")
    ax.set_ylabel("Q-value (>0 = good for current player)", fontweight="bold")
    ax.set_title("Q-value per column", fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    for c, q in zip(cols, q_vals[valid]):
        ax.text(c, q + (0.05 if q >= 0 else -0.1), f"{q:+.2f}",
                ha="center", fontsize=8, fontweight="bold")

    # ── Panel 3: Prior vs visit share ─────────────────────────────────────────
    ax = axes[2]
    x = np.arange(7)
    width = 0.35
    valid_prior = ~np.isnan(priors)
    ax.bar(x[valid_prior] - width / 2, priors[valid_prior], width,
           label="Network prior", color="#3498db", alpha=0.8, edgecolor="black")
    ax.bar(x[valid] + width / 2, visit_pct[valid], width,
           label="MCTS visit %", color="#e67e22", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Column", fontweight="bold")
    ax.set_ylabel("Probability", fontweight="bold")
    ax.set_title("Network prior vs MCTS visit share\n(divergence = search overruled network)",
                 fontweight="bold")
    ax.set_xticks(range(7))
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "mcts_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved column heatmap: {out}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the MCTS search tree for one move decision."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations to run (default 200)")
    parser.add_argument("--tree-depth", type=int, default=3,
                        help="Depth of tree diagram (default 3)")
    parser.add_argument("--moves", type=int, nargs="*", default=[],
                        help="Column moves to replay before visualizing (e.g. --moves 3 4 3)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save PNGs")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model} on {device}...")
    model = load_model(args.model, device)
    print("Model loaded.")

    # Set up board state
    game = Connect4()
    for col in args.moves:
        r, c = game.play(col)
        if game.check_win(r, c):
            print(f"Game already over after replaying moves (win at col {col})")
            return

    print(f"Running {args.simulations} MCTS simulations...")
    probs, root = run_mcts_simulations(
        game, model, device,
        num_sims=args.simulations,
        temperature=1.0,
        add_dirichlet_noise=False,
        return_root=True,
    )

    chosen = int(np.argmax(probs))
    print(f"Best move: column {chosen}  (visit share: {probs[chosen]:.1%})")

    print("\nGenerating tree diagram...")
    visualize_tree(root, max_depth=args.tree_depth, output_dir=args.output_dir)

    print("Generating column heatmap...")
    visualize_column_heatmap(root, game, output_dir=args.output_dir)

    print("\nDone. Files saved to:", args.output_dir)


if __name__ == "__main__":
    main()
