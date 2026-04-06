"""
play.py — Play against a trained model

Example:
    python play.py --checkpoint checkpoint_0190.pt --human-first
"""

import argparse
import torch
import numpy as np
import openvino as ov

from model import AlphaNet
from mcts import Connect4, run_mcts_simulations, print_board


class OpenVINOModel:
    """A wrapper to make an OpenVINO model behave like a PyTorch model for inference."""
    def __init__(self, model_path: str, device: str = "AUTO"):
        core = ov.Core()
        # Read and compile the model for the target device
        ov_model = core.read_model(model=model_path)
        
        # Auto-discover the best hardware accelerator to use
        available = core.available_devices
        if "NPU" in available:
            self.ov_device = "NPU"
        elif "GPU" in available:
            self.ov_device = "GPU"
        else:
            self.ov_device = "CPU"
            
        self.compiled_model = core.compile_model(model=ov_model, device_name=self.ov_device)
        # Get output tensors by their friendly names from the ONNX export
        self.policy_output = self.compiled_model.output("policy")
        self.value_output = self.compiled_model.output("value")

    def __call__(self, x: torch.Tensor):
        # Convert torch.Tensor to a numpy array for OpenVINO
        x_np = x.cpu().numpy()
        result = self.compiled_model([x_np])
        # Convert results back to torch.Tensor to match the original PyTorch model output
        return torch.from_numpy(result[self.policy_output]), \
               torch.from_numpy(result[self.value_output])

    def eval(self):
        # This method is required to mimic the PyTorch model interface, but does nothing for OpenVINO.
        pass

def get_human_move(game: Connect4) -> int:
    """Get a valid column choice from the human player."""
    valid_moves = game.get_valid_moves()
    while True:
        try:
            move_str = input(f"Enter column ({', '.join(map(str, valid_moves))}): ")
            move = int(move_str)
            if move in valid_moves:
                return move
            else:
                print("Invalid column. Please choose a valid, non-full column.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    parser = argparse.ArgumentParser(description="Play Connect 4 against a trained AlphaZero model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.pt for PyTorch, .onnx for OpenVINO).")
    parser.add_argument("--simulations", type=int, default=800, help="Number of MCTS simulations per AI move.")
    parser.add_argument("--human-first", action="store_true", help="Set this flag for the human to play first as 'X'.")
    args = parser.parse_args()

    use_openvino = args.model.endswith(".onnx")

    if use_openvino:
        # We instantiate the OpenVINOModel which will auto-detect NPU/GPU/CPU
        model = OpenVINOModel(args.model)
        print(f"Using OpenVINO for inference on {model.ov_device}.")
        # The MCTS logic still uses torch tensors on the CPU side before they are passed to the model.
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using PyTorch on device: {device}")
        model = AlphaNet().to(device)
        try:
            checkpoint = torch.load(args.model, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"Error: Model file not found at '{args.model}'")
            return
        except KeyError:
            # Fallback for raw state_dict checkpoints
            print("Warning: Checkpoint is not in the expected format. Trying to load raw state_dict.")
            model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    model.eval()

    game = Connect4()
    human_player = 1 if args.human_first else -1

    while True:
        print_board(game)

        if game.current_player == human_player:
            # ── Human Move Evaluation ──
            print("Analyzing best options...")
            mcts_probs = run_mcts_simulations(
                game, model, device,
                num_sims=args.simulations,
                temperature=0,
                add_dirichlet_noise=False,
            )
            max_p = np.max(mcts_probs)
            
            move = get_human_move(game)
            
            # Assessment Logic
            p_move = mcts_probs[move]
            if p_move >= 0.95 * max_p:
                score, comment = 5, "Brilliant! (Best Move)"
            elif p_move >= 0.70 * max_p:
                score, comment = 4, "Strong Move"
            elif p_move >= 0.30 * max_p:
                score, comment = 3, "Decent"
            elif p_move >= 0.05 * max_p:
                score, comment = 2, "Inaccurate"
            else:
                score, comment = 1, "Blunder!"
            
            print(f"Assessment: {'★' * score}{'☆' * (5-score)} — {comment}")
        else:
            print("AI is thinking...")
            # For AI moves, use MCTS with temperature=0 to be greedy
            mcts_probs = run_mcts_simulations(
                game, model, device,
                num_sims=args.simulations,
                temperature=0,
                add_dirichlet_noise=False, # No exploration needed for play
            )
            move = int(np.argmax(mcts_probs))
            print(f"AI chooses column {move}")

        r, c = game.play(move)

        if game.check_win(r, c):
            print_board(game)
            winner_char = 'You' if game.current_player != human_player else 'The AI'
            print(f"Game over. {winner_char} won!")
            break
        if not game.get_valid_moves():
            print_board(game)
            print("Game over. It's a draw!")
            break

if __name__ == "__main__":
    main()