"""
export_onnx.py — Convert a PyTorch checkpoint to ONNX format.

Example:
    python export_onnx.py --checkpoint checkpoint_0190.pt --output model.onnx
"""

import argparse
import torch
from model import AlphaNet

def main():
    parser = argparse.ArgumentParser(description="Export a PyTorch model checkpoint to ONNX format.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PyTorch model checkpoint (.pt file).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output ONNX model (.onnx file).")
    args = parser.parse_args()

    print("Loading PyTorch model...")
    model = AlphaNet()

    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        print("Warning: Checkpoint does not contain 'model_state_dict'. Trying to load raw state_dict.")
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    model.eval()

    # Create a dummy input with the correct shape for the model
    dummy_input = torch.randn(1, 3, 6, 7)

    print(f"Exporting model to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=['input'],
        output_names=['policy', 'value'], # These names are important for the OpenVINO wrapper
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        opset_version=11
    )
    print("Export complete.")

if __name__ == "__main__":
    main()