"""
visualize.py — Feature visualization of learned Conv kernels and activation maps

This script provides tools to understand what the AlphaNet model learns:
  1. Visualize raw conv kernels from the start_block
  2. Visualize intermediate feature maps for a given board state
  3. Track which features activate during a game

Usage:
    python visualize.py --checkpoint checkpoint_0190.pt --board-state initial
    python visualize.py --checkpoint checkpoint_0190.pt --show-kernels
"""

import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import networkx as nx

from model import AlphaNet
from mcts import Connect4, board_to_tensor, print_board, MCTSNode

# Optional accelerated CPU ONNX support (OpenVINO preferred, then ONNXRuntime)
try:
    import openvino.runtime as ov
except ImportError:
    ov = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OpenVINOModel:
    def __init__(self, model_path: str, device: str = "CPU"):
        if ov is None:
            raise RuntimeError("OpenVINO runtime is not installed.")
        core = ov.Core()
        ov_model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=ov_model, device_name=device)
        self.policy_output = self.compiled_model.output(0)
        self.value_output = self.compiled_model.output(1)

    def __call__(self, x: torch.Tensor):
        x_np = x.cpu().numpy().astype(np.float32)
        result = self.compiled_model([x_np])
        policy = torch.from_numpy(result[self.policy_output]).to(x.device)
        value = torch.from_numpy(result[self.value_output]).to(x.device)
        return policy, value

    def eval(self):
        pass


class ONNXRuntimeModel:
    def __init__(self, model_path: str, device: str = "CPU"):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        outputs = [o.name for o in self.session.get_outputs()]
        self.policy_output, self.value_output = outputs[0], outputs[1]

    def __call__(self, x: torch.Tensor):
        x_np = x.cpu().numpy().astype(np.float32)
        res = self.session.run([self.policy_output, self.value_output], {self.session.get_inputs()[0].name: x_np})
        policy = torch.from_numpy(np.asarray(res[0])).to(x.device)
        value = torch.from_numpy(np.asarray(res[1])).to(x.device)
        return policy, value

    def eval(self):
        pass


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained AlphaNet checkpoint or ONNX model."""
    path = Path(checkpoint_path)

    if path.suffix.lower() == ".onnx":
        # Load ONNX model via OpenVINO or ONNX Runtime
        if ov is not None:
            model = OpenVINOModel(str(path), device="CPU")
        elif ort is not None:
            model = ONNXRuntimeModel(str(path), device="CPU")
        else:
            raise RuntimeError("No ONNX backend available (openvino or onnxruntime required).")
        model.eval()
        return model

    # Default path: PyTorch checkpoint
    model = AlphaNet().to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except KeyError:
        # Fallback for raw state_dict checkpoints
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model.eval()
    return model


def visualize_kernels(model: AlphaNet, output_dir: str = "."):
    """
    Visualize the first-layer conv kernels.
    Extracts weight tensor from start_block[0] (Conv2d layer).
    Shape: (128, 3, 3, 3) → 128 filters, 3 input channels, 3x3 spatial.
    """
    kernels = model.start_block[0].weight  # (out_channels=128, in_channels=3, H=3, W=3)
    
    # Visualize first 32 filters (arranged 8x4)
    num_to_viz = min(32, kernels.shape[0])
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("Start Block Conv Kernels (128 filters, 3×3)", fontsize=14, fontweight='bold')
    
    for idx in range(num_to_viz):
        ax = axes.flat[idx]
        # kernel shape: (3, 3, 3) for one filter
        k = kernels[idx].detach().cpu().numpy()  # (3, 3, 3)
        
        # Normalize to [0, 1] per filter
        k_min, k_max = k.min(), k.max()
        if k_max > k_min:
            k = (k - k_min) / (k_max - k_min)
        else:
            k = np.zeros_like(k)
        
        # Average across channels for grayscale, or show as RGB
        k_vis = k.mean(axis=0)  # (3, 3)
        ax.imshow(k_vis, cmap='viridis')
        ax.set_title(f"Filter {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_to_viz, len(axes.flat)):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "kernels_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved kernel visualization: {output_path}")
    plt.close()


def visualize_feature_maps(model: AlphaNet, game: Connect4, 
                          layer_idx: int = 3, output_dir: str = "."):
    """
    Visualize feature maps (activations) from a ResBlock for a given board state.
    
    Args:
      layer_idx: which ResBlock to hook (0-7)
      Visualizes the top 16 most activated channels.
    """
    device = next(model.parameters()).device
    state = board_to_tensor(game).unsqueeze(0).to(device)
    
    activations = []
    
    def hook(module, input, output):
        activations.append(output.detach())
    
    # Register hook on the selected ResBlock
    model.res_blocks[layer_idx].register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(state)
    
    # activation shape: (1, 128, 6, 7)
    if not activations:
        print(f"No activations captured from ResBlock {layer_idx}")
        return
    
    feat = activations[0][0]  # (128, 6, 7)
    
    # Find top 16 channels by mean activation
    channel_means = feat.mean(dim=(1, 2))
    top_channels = torch.argsort(channel_means, descending=True)[:16]
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Feature Maps — ResBlock {layer_idx}, Top 16 Channels", 
                 fontsize=14, fontweight='bold')
    
    for plot_idx, channel_idx in enumerate(top_channels):
        ax = axes.flat[plot_idx]
        feat_map = feat[channel_idx].cpu().numpy()  # (6, 7)
        
        # Normalize to [0, 1]
        f_min, f_max = feat_map.min(), feat_map.max()
        if f_max > f_min:
            feat_map = (feat_map - f_min) / (f_max - f_min)
        
        im = ax.imshow(feat_map, cmap='hot', aspect='auto')
        ax.set_title(f"Ch {channel_idx.item()}\nmean={channel_means[channel_idx].item():.3f}", 
                     fontsize=9)
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"feature_maps_resblock_{layer_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature map visualization: {output_path}")
    plt.close()


def visualize_board_and_inference(model: AlphaNet, game: Connect4, output_dir: str = "."):
    """
    Show the board state, network policy output, and value prediction.
    """
    device = next(model.parameters()).device
    state = board_to_tensor(game).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(state)
    
    policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    value_pred = value.item()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: board visualization
    board = np.flipud(game.board)
    symbols = {1: 'X', -1: 'O', 0: ' '}
    
    # Create colored grid
    board_display = np.zeros((6, 7))
    for r in range(6):
        for c in range(7):
            board_display[r, c] = board[r, c]
    
    im1 = ax1.imshow(board_display, cmap='RdBu', vmin=-1, vmax=1, alpha=0.3)
    ax1.set_xticks(range(7))
    ax1.set_yticks(range(6))
    ax1.set_xticklabels(range(7))
    ax1.set_yticklabels(range(6))
    ax1.grid(True, color='black', linewidth=1)
    ax1.set_title("Board State\n(Blue=Player1(X), Red=Player2(O))", fontweight='bold')
    
    # Overlay piece symbols
    for r in range(6):
        for c in range(7):
            sym = symbols[board[r, c]]
            if sym != ' ':
                ax1.text(c, r, sym, ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Right: policy distribution
    columns = np.arange(7)
    colors = ['green' if p == policy_probs.max() else 'skyblue' for p in policy_probs]
    ax2.bar(columns, policy_probs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel("Column", fontweight='bold')
    ax2.set_ylabel("Probability", fontweight='bold')
    ax2.set_title(f"Policy Output (Move Probabilities)\nValue Pred: {value_pred:.3f}", 
                  fontweight='bold')
    ax2.set_ylim(0, max(policy_probs) * 1.1)
    ax2.set_xticks(range(7))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value interpretation
    if value_pred > 0.5:
        val_text = "Strong Win"
    elif value_pred > 0.2:
        val_text = "Likely Win"
    elif value_pred < -0.5:
        val_text = "Strong Loss"
    elif value_pred < -0.2:
        val_text = "Likely Loss"
    else:
        val_text = "Neutral/Draw"
    
    fig.text(0.5, 0.02, f"Value Interpretation: {val_text}", 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = Path(output_dir) / "board_and_policy.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved board + policy visualization: {output_path}")
    plt.close()


def visualize_mcts_tree(model: AlphaNet, game: Connect4, output_dir: str = "."):
    """Visualizes the MCTS Search Tree using NetworkX."""
    device = next(model.parameters()).device
    
    # Locally run MCTS to extract the root
    root = MCTSNode(game.clone())
    state = board_to_tensor(root.game).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(state)
    policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    root.expand(policy_probs)
    root.backpropagate(value.item())
    
    for _ in range(200):
        node = root
        while not node.is_leaf():
            _, node = node.select_child(1.5)
        if node.is_terminal():
            node.backpropagate(node.terminal_value)
            continue
        st = board_to_tensor(node.game).unsqueeze(0).to(device)
        with torch.no_grad():
            p_logits, val = model(st)
        p_probs = F.softmax(p_logits, dim=1).squeeze().cpu().numpy()
        node.expand(p_probs)
        node.backpropagate(val.item())

    G = nx.DiGraph()
    
    def add_edges(n, parent_name="", depth=0):
        if depth > 2: return 
        name = f"{id(n)}\nN:{n.visit_count}\nQ:{n.q_value:.2f}"
        if parent_name:
            G.add_edge(parent_name, name)
        for move, child in n.children.items():
            if child.visit_count > 0:
                add_edges(child, name, depth+1)
            
    add_edges(root)
    
    plt.figure(figsize=(14, 10))
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        # Fallback if graphviz isn't cleanly installed on Windows
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
    node_colors = [float(node.split("Q:")[1]) for node in G.nodes()]
    node_sizes = [500 + int(node.split("N:")[1].split("\n")[0])*10 for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap="RdYlGn", 
            node_size=node_sizes, font_size=8, font_weight="bold", edge_color="gray")
    plt.title("MCTS Search Tree (Depth=2, Color=Q-Value, Size=Visits)", fontsize=14, fontweight='bold')
    
    output_path = Path(output_dir) / "mcts_tree.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved MCTS tree visualization: {output_path}")
    plt.close()


def visualize_policy_divergence(model: AlphaNet, game: Connect4, output_dir: str = "."):
    """Plot Raw Neural Network Policy vs Final MCTS Search Distribution"""
    device = next(model.parameters()).device
    
    state = board_to_tensor(game).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(state)
    priors = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    
    root = MCTSNode(game.clone())
    root.expand(priors)
    root.backpropagate(value.item())
    for _ in range(400):
        node = root
        while not node.is_leaf():
            _, node = node.select_child(1.5)
        if node.is_terminal():
            node.backpropagate(node.terminal_value)
            continue
        st = board_to_tensor(node.game).unsqueeze(0).to(device)
        with torch.no_grad():
            p_logits, val = model(st)
        p_probs = F.softmax(p_logits, dim=1).squeeze().cpu().numpy()
        node.expand(p_probs)
        node.backpropagate(val.item())
        
    visits = np.zeros(7)
    for m, c in root.children.items():
        visits[m] = c.visit_count
    
    mcts_probs = visits / visits.sum() if visits.sum() > 0 else visits
    
    x = np.arange(7)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, priors, width, label='Raw Network Prior', color='skyblue')
    ax.bar(x + width/2, mcts_probs, width, label='MCTS Post-Search', color='salmon')
    
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Policy Divergence: Raw Neural Network vs MCTS Search', fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    output_path = Path(output_dir) / "policy_divergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved policy divergence visualization: {output_path}")
    plt.close()


def visualize_saliency_map(model: AlphaNet, game: Connect4, output_dir: str = "."):
    """Generate Grad-CAM style heatmap for Value Head input sensitivity"""
    device = next(model.parameters()).device
    state = board_to_tensor(game).unsqueeze(0).to(device)
    state.requires_grad_()
    
    policy_logits, value = model(state)
    
    model.zero_grad()
    value.backward()
    
    saliency, _ = torch.max(state.grad.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    if saliency.max() > saliency.min():
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.flipud(saliency), cmap='hot', alpha=0.9)
    plt.colorbar(im, label="Gradient Magnitude (Sensitivity)")
    
    board = np.flipud(game.board)
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for r in range(6):
        for c in range(7):
            sym = symbols[board[r, c]]
            if sym != ' ':
                ax.text(c, r, sym, ha='center', va='center', fontsize=20, 
                        color='white' if np.flipud(saliency)[r, c] < 0.5 else 'black', fontweight='bold')
                        
    ax.set_xticks(range(7))
    ax.set_yticks(range(6))
    ax.set_title(f"Value Head Saliency Map\nPredicted Value: {value.item():.3f}", fontweight='bold')
    
    output_path = Path(output_dir) / "saliency_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Saliency Map visualization: {output_path}")
    plt.close()


def visualize_game_trajectory(model: AlphaNet, output_dir: str = "."):
    """Plot the entire value trajectory over a synthetic automated game"""
    device = next(model.parameters()).device
    game = Connect4()
    values = []
    
    while True:
        state = board_to_tensor(game).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = model(state)
        
        v = value.item() if game.current_player == 1 else -value.item()
        values.append(v)
        
        policy_probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
        valid = game.get_valid_moves()
        if not valid:
            break
            
        masked_probs = np.zeros(7)
        masked_probs[valid] = policy_probs[valid]
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            masked_probs[valid] = 1.0 / len(valid)
            
        best_move = int(np.argmax(masked_probs))
        r, c = game.play(best_move)
        
        if game.check_win(r, c):
            values.append(1.0 if game.current_player == -1 else -1.0)
            break
            
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(values)+1), values, marker='o', color='purple', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Turn Number", fontweight='bold')
    ax.set_ylabel("Advantage (Player 1 Perspective)", fontweight='bold')
    ax.set_title("Value Trajectory (Self-Play Simulation)", fontweight='bold')
    ax.grid(alpha=0.3)
    
    output_path = Path(output_dir) / "value_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Value Trajectory visualization: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned features in AlphaNet."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--show-kernels", action="store_true",
        help="Visualize raw conv kernels from the start block."
    )
    parser.add_argument(
        "--show-features", action="store_true",
        help="Visualize feature maps (activations) from a ResBlock."
    )
    parser.add_argument(
        "--resblock-idx", type=int, default=3,
        help="Which ResBlock to visualize (0-7, default 3)."
    )
    parser.add_argument(
        "--show-board", action="store_true",
        help="Visualize current board state with network inference."
    )
    parser.add_argument(
        "--show-mcts", action="store_true",
        help="Visualize the MCTS Search Tree (needs networkx)."
    )
    parser.add_argument(
        "--show-divergence", action="store_true",
        help="Visualize divergence between Raw Policy and Post-Search MCTS."
    )
    parser.add_argument(
        "--show-saliency", action="store_true",
        help="Visualize Grad-CAM value saliency map."
    )
    parser.add_argument(
        "--show-trajectory", action="store_true",
        help="Visualize the value trajectory over a greedy self-play game."
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save visualization PNGs."
    )
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, device)
    print("✓ Model loaded successfully")
    
    # Run visualizations
    if args.show_kernels:
        print("\nVisualizing conv kernels...")
        visualize_kernels(model, args.output_dir)
    
    if args.show_features:
        print(f"\nVisualizing feature maps from ResBlock {args.resblock_idx}...")
        game = Connect4()  # Empty initial board
        visualize_feature_maps(model, game, args.resblock_idx, args.output_dir)
    
    if args.show_board:
        print("\nVisualizing board + policy inference...")
        game = Connect4()  # Empty initial board
        visualize_board_and_inference(model, game, args.output_dir)
        
    if args.show_mcts:
        print("\nVisualizing MCTS search tree...")
        visualize_mcts_tree(model, Connect4(), args.output_dir)
        
    if args.show_divergence:
        print("\nVisualizing MCTS vs Policy divergence...")
        visualize_policy_divergence(model, Connect4(), args.output_dir)
        
    if args.show_saliency:
        print("\nVisualizing board saliency heatmap...")
        visualize_saliency_map(model, Connect4(), args.output_dir)
        
    if args.show_trajectory:
        print("\nVisualizing automated game value trajectory...")
        visualize_game_trajectory(model, args.output_dir)
    
    # Default: show all if none selected
    if not (args.show_kernels or args.show_features or args.show_board or args.show_mcts or args.show_divergence or args.show_saliency or args.show_trajectory):
        print("\nNo visualization flags selected. Running all...")
        visualize_kernels(model, args.output_dir)
        game = Connect4()
        visualize_feature_maps(model, game, args.resblock_idx, args.output_dir)
        visualize_board_and_inference(model, game, args.output_dir)
        visualize_mcts_tree(model, game, args.output_dir)
        visualize_policy_divergence(model, game, args.output_dir)
        visualize_saliency_map(model, game, args.output_dir)
        visualize_game_trajectory(model, args.output_dir)
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
