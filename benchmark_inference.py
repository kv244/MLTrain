import time
import torch
import numpy as np
import argparse
import os

# We will try to import everything. If it fails, we skip that backend.
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
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms

def benchmark_openvino(device_name, model_path, iterations=100):
    if ov is None: return None
    try:
        core = ov.Core()
        ov_model = core.read_model(model=model_path)
        compiled_model = core.compile_model(model=ov_model, device_name=device_name)
        
        # OpenVINO specific output retrieval
        policy_output = compiled_model.output("policy")
        value_output = compiled_model.output("value")
        
        dummy_input = np.random.randn(1, 3, 6, 7).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            compiled_model([dummy_input])
            
        start = time.perf_counter()
        for _ in range(iterations):
            compiled_model([dummy_input])
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000 # ms
    except Exception as e:
        return f"Error: {e}"

def benchmark_onnx_runtime(provider, model_path, iterations=100):
    if ort is None: return None
    try:
        session_options = ort.SessionOptions()
        # provider can be 'CUDAExecutionProvider' or 'CPUExecutionProvider'
        session = ort.InferenceSession(model_path, providers=[provider], sess_options=session_options)
        
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, 6, 7).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
            
        start = time.perf_counter()
        for _ in range(iterations):
            session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000 # ms
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    parser.add_argument("--onnx", type=str, default="model.onnx")
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    print(f"--- Benchmarking Inference Latency (Batch Size 1, {args.iterations} iterations) ---")
    print(f"Model: {args.checkpoint} / {args.onnx}\n")

    input_shape = (1, 3, 6, 7)
    dummy_input_torch = torch.randn(*input_shape)

    # 1. PyTorch
    print("PyTorch Backends:")
    res_pt_cpu = benchmark_pytorch("cpu", AlphaNet(), dummy_input_torch, args.iterations)
    print(f"  CPU: {res_pt_cpu:.3f} ms")
    
    if torch.cuda.is_available():
        res_pt_cuda = benchmark_pytorch("cuda", AlphaNet(), dummy_input_torch, args.iterations)
        print(f"  CUDA (RTX 4070): {res_pt_cuda:.3f} ms")
    else:
        print("  CUDA: Not available")

    # 2. OpenVINO
    if os.path.exists(args.onnx):
        print("\nOpenVINO Backends:")
        core = ov.Core()
        devices = core.available_devices
        for dev in ["CPU", "GPU", "NPU"]:
            if dev in devices or (dev == "GPU" and any("GPU" in d for d in devices)):
                res = benchmark_openvino(dev, args.onnx, args.iterations)
                if isinstance(res, float):
                    print(f"  {dev}: {res:.3f} ms")
                else:
                    print(f"  {dev}: {res}")
    else:
        print("\nOpenVINO: model.onnx not found. Run export_onnx.py first.")

    # 3. ONNX Runtime
    if os.path.exists(args.onnx):
        print("\nONNX Runtime Backends:")
        res_ort_cpu = benchmark_onnx_runtime("CPUExecutionProvider", args.onnx, args.iterations)
        print(f"  CPU: {res_ort_cpu:.3f} ms" if isinstance(res_ort_cpu, float) else f"  CPU: {res_ort_cpu}")
        
        res_ort_cuda = benchmark_onnx_runtime("CUDAExecutionProvider", args.onnx, args.iterations)
        if isinstance(res_ort_cuda, float):
            print(f"  CUDA (RTX 4070): {res_ort_cuda:.3f} ms")
        else:
            print(f"  CUDA (RTX 4070): {res_ort_cuda}")
    else:
        print("\nONNX Runtime: model.onnx not found.")

if __name__ == "__main__":
    main()
