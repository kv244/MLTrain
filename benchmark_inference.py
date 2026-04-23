"""
benchmark_inference.py — Inference latency comparison across backends.

Runs batch-size-1 forward passes through the AlphaNet on every available
backend and prints average latency in milliseconds. Use this to decide which
runtime to deploy on a given machine.

Backends tested:
  PyTorch   — CPU and CUDA (RTX 4070 if available)
  OpenVINO  — CPU / GPU / NPU (reads model.onnx; requires openvino package)
  ONNX Rt.  — CPUExecutionProvider and CUDAExecutionProvider

Usage:
  python benchmark_inference.py
  python benchmark_inference.py --checkpoint checkpoint_best.pt --onnx model.onnx --iterations 400
"""

import time
import torch
import numpy as np
import argparse
import os

try:
    import openvino as ov
except ImportError:
    ov = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from model import AlphaNet


def benchmark_pytorch(device_name, model, dummy_input, iterations=100):
    """Warm up for 10 passes, then time `iterations` forward passes on PyTorch."""
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # ms per call


def benchmark_openvino(device_name, model_path, iterations=100):
    """Compile the ONNX model with OpenVINO and time synchronous inference."""
    if ov is None:
        return None
    try:
        core = ov.Core()
        ov_model = core.read_model(model=model_path)
        compiled_model = core.compile_model(model=ov_model, device_name=device_name)

        dummy_input = np.random.randn(1, 3, 6, 7).astype(np.float32)

        for _ in range(10):
            compiled_model([dummy_input])

        start = time.perf_counter()
        for _ in range(iterations):
            compiled_model([dummy_input])
        end = time.perf_counter()

        return (end - start) / iterations * 1000
    except Exception as e:
        return f"Error: {e}"


def benchmark_onnx_runtime(provider, model_path, iterations=100):
    """Run inference via ONNX Runtime with the specified execution provider."""
    if ort is None:
        return None
    try:
        session = ort.InferenceSession(
            model_path,
            providers=[provider],
            sess_options=ort.SessionOptions()
        )
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, 6, 7).astype(np.float32)

        for _ in range(10):
            session.run(None, {input_name: dummy_input})

        start = time.perf_counter()
        for _ in range(iterations):
            session.run(None, {input_name: dummy_input})
        end = time.perf_counter()

        return (end - start) / iterations * 1000
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    parser.add_argument("--onnx",       type=str, default="model.onnx")
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    print(f"--- Benchmarking Inference Latency (Batch Size 1, {args.iterations} iterations) ---")
    print(f"Model: {args.checkpoint} / {args.onnx}\n")

    dummy_input_torch = torch.randn(1, 3, 6, 7)

    # ── PyTorch ───────────────────────────────────────────────────────────────
    print("PyTorch Backends:")
    print(f"  CPU: {benchmark_pytorch('cpu', AlphaNet(), dummy_input_torch, args.iterations):.3f} ms")
    if torch.cuda.is_available():
        print(f"  CUDA (RTX 4070): {benchmark_pytorch('cuda', AlphaNet(), dummy_input_torch, args.iterations):.3f} ms")
    else:
        print("  CUDA: Not available")

    # ── OpenVINO ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.onnx):
        print("\nOpenVINO / ONNX Runtime: model.onnx not found — run export_onnx.py first.")
        return

    if ov is not None:
        print("\nOpenVINO Backends:")
        available = ov.Core().available_devices
        for dev in ["CPU", "GPU", "NPU"]:
            if dev in available or (dev == "GPU" and any("GPU" in d for d in available)):
                res = benchmark_openvino(dev, args.onnx, args.iterations)
                print(f"  {dev}: {res:.3f} ms" if isinstance(res, float) else f"  {dev}: {res}")

    # ── ONNX Runtime ──────────────────────────────────────────────────────────
    print("\nONNX Runtime Backends:")
    for provider, label in [("CPUExecutionProvider", "CPU"),
                             ("CUDAExecutionProvider", "CUDA (RTX 4070)")]:
        res = benchmark_onnx_runtime(provider, args.onnx, args.iterations)
        print(f"  {label}: {res:.3f} ms" if isinstance(res, float) else f"  {label}: {res}")


if __name__ == "__main__":
    main()
