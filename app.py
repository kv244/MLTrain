import os
import glob
import time
import csv
import torch
import numpy as np
import pathlib
import threading
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import openvino as ov
from dotenv import load_dotenv

load_dotenv()

VERSION = "1.0.2"
LAST_COMMIT = "2026-04-04 08:32 UTC"

from mcts import Connect4, run_mcts_simulations

app = Flask(__name__)
# Tell Flask to trust X-Forwarded-For headers from Nginx
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

RATE_LIMITS = os.environ.get("RATE_LIMIT", "500 per day;200 per hour;15 per minute").split(";")
LIMITER_STORAGE = os.environ.get("LIMITER_STORAGE_URI", "memory://")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=RATE_LIMITS,
    storage_uri=LIMITER_STORAGE
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Too Many Requests", description=str(e.description)), 429

MODELS_DIR = pathlib.Path(".").resolve()
_csv_lock = threading.Lock()
_model_lock = threading.Lock()

def _resolve_checkpoint(name: str):
    """Return safe absolute path or None if invalid."""
    try:
        safe = (MODELS_DIR / pathlib.Path(name).name).resolve()
    except Exception:
        return None
    if safe.parent != MODELS_DIR or safe.suffix != ".onnx":
        return None
    return safe


# Cache models to avoid reloading them constantly
# This is a small optimization for web responsiveness.
LOADED_MODELS = {}
device = torch.device("cpu") # PyTorch tensor device for CPU memory handling

def get_ov_device():
    core = ov.Core()
    available = core.available_devices
    if "NPU" in available: return "npu"
    if "GPU" in available: return "gpu"
    return "cpu"

GLOBAL_OV_DEVICE = get_ov_device()

class OpenVINOModel:
    """A wrapper to make an OpenVINO model behave like a PyTorch model for inference."""
    def __init__(self, model_path: str, device: str = "AUTO"):
        core = ov.Core()
        
        # Check available devices to prioritize NPU
        available_devices = core.available_devices
        if "NPU" in available_devices:
            device = "NPU"
        elif "GPU" in available_devices:
            device = "GPU"
        else:
            device = "CPU"
            
        print(f"OpenVINO initializing on device: {device} (Available: {available_devices})")
        
        ov_model = core.read_model(model=model_path)
        # Latency hint is best for interactive applications like games
        self.compiled_model = core.compile_model(
            model=ov_model, 
            device_name=device,
            config={"PERFORMANCE_HINT": "LATENCY"}
        )
        self.policy_output = self.compiled_model.output("policy")
        self.value_output = self.compiled_model.output("value")

    def __call__(self, x: torch.Tensor):
        x_np = x.cpu().numpy()
        result = self.compiled_model([x_np])
        return torch.from_numpy(result[self.policy_output]), \
               torch.from_numpy(result[self.value_output])

    def eval(self):
        pass

def get_model(checkpoint_path):
    """Loads a compiled OpenVINO `.onnx` file or retrieves it from cache."""
    with _model_lock:
        if checkpoint_path not in LOADED_MODELS:
            try:
                model = OpenVINOModel(checkpoint_path)
                LOADED_MODELS[checkpoint_path] = model
            except Exception as e:
                return None, str(e)
    return LOADED_MODELS[checkpoint_path], None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info")
def get_info():
    """Returns runtime hardware info to the frontend."""
    return jsonify({"device": GLOBAL_OV_DEVICE})

@app.route("/api/game_end", methods=["POST"])
def log_game_end():
    """Receives and logs game outcome telemtry."""
    data = request.json
    winner = data.get("winner", "unknown")
    model_version = data.get("model", "unknown")
    
    results_file = os.environ.get("RESULTS_LOG_PATH", "game_results.csv")
    file_exists = os.path.isfile(results_file)
    with _csv_lock:
        with open(results_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["timestamp", "model", "winner"])
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), model_version, winner])
        
    return jsonify({"success": True})

@app.route("/api/models")
def list_models():
    """Lists all available model checkpoints in the main directory."""
    # Find all .onnx files in the directory
    models = glob.glob("*.onnx")
    # Sort descending so highest iteration is first
    models.sort(reverse=True)
    return jsonify({"models": models})

@app.route("/api/move", methods=["POST"])
# Default limit is inherited from default_limits, allowing environment injection globally
def get_move():
    """Calculates the best move using the model + MCTS search."""
    data = request.json
    checkpoint_name = data.get("model", "")
    board_state = data.get("board", []) 
    current_player = data.get("current_player", -1) 
    simulations = max(1, min(int(data.get("simulations", 400)), 2048))
    
    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    # Validate board state
    board_arr = np.array(board_state, dtype=np.int8)
    if board_arr.shape != (6, 7) or not np.all(np.isin(board_arr, [-1, 0, 1])):
        return jsonify({"error": "Invalid board state"}), 400

    model, error = get_model(str(checkpoint))
    if error:
        return jsonify({"error": f"Failed to load model: {error}"}), 500

    # Reconstruct the game state for the Python engine
    game = Connect4()
    
    # 1. Provide the board state
    game.board = board_arr
    # Numpy arrays from lists will be [6, 7], Python expects [r][c].
    game.current_player = current_player

    print(f"Evaluating board using {checkpoint_name} for player {current_player}...")

    start_time = time.time()
    mcts_probs = run_mcts_simulations(
        game, model, device,
        num_sims=simulations,
        temperature=0,             # Greedy, we want the *best* move
        add_dirichlet_noise=False  # No exploration in production play
    )
    inference_time = time.time() - start_time
    
    # Write telemetry to CSV
    telemetry_file = os.environ.get("TELEMETRY_LOG_PATH", "telemetry.csv")
    file_exists = os.path.isfile(telemetry_file)
    with _csv_lock:
        with open(telemetry_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["timestamp", "model", "simulations", "inference_time_seconds"])
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), checkpoint_name, simulations, f"{inference_time:.4f}"])
    
    best_move = int(np.argmax(mcts_probs))
    
    return jsonify({
        "move": best_move,
        "probs": [float(p) for p in mcts_probs],
        "inference_time_ms": inference_time * 1000
    })

@app.route("/api/assess", methods=["POST"])
def assess_move():
    """Evaluates a specific move against the AI's best recommended move."""
    data = request.json
    checkpoint_name = data.get("model", "")
    board_state = data.get("board", []) # This should be the board BEFORE the move
    move = data.get("move", -1)
    current_player = data.get("current_player", 1)
    simulations = max(1, min(int(data.get("simulations", 400)), 2048))

    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    # Validate board state
    board_arr = np.array(board_state, dtype=np.int8)
    if board_arr.shape != (6, 7) or not np.all(np.isin(board_arr, [-1, 0, 1])):
        return jsonify({"error": "Invalid board state"}), 400

    model, error = get_model(str(checkpoint))
    if error:
        return jsonify({"error": f"Failed to load model: {error}"}), 500

    game = Connect4()
    game.board = board_arr
    game.current_player = current_player

    # Get AI's opinion on this state
    mcts_probs = run_mcts_simulations(
        game, model, device,
        num_sims=simulations,
        temperature=0,
        add_dirichlet_noise=False
    )
    
    max_p = np.max(mcts_probs)
    p_move = mcts_probs[move]
    
    # Extended assessment logic for more nuance
    if p_move >= 0.98 * max_p:
        score, comment = 5, "Brilliant! (Best Move)"
    elif p_move >= 0.90 * max_p:
        score, comment = 4, "Great Move"
    elif p_move >= 0.75 * max_p:
        score, comment = 4, "Strong"
    elif p_move >= 0.50 * max_p:
        score, comment = 3, "Solid"
    elif p_move >= 0.30 * max_p:
        score, comment = 3, "Standard"
    elif p_move >= 0.15 * max_p:
        score, comment = 2, "Inaccurate"
    elif p_move >= 0.05 * max_p:
        score, comment = 2, "Mistake"
    else:
        score, comment = 1, "Blunder!"

    return jsonify({
        "score": score,
        "comment": comment,
        "best_move": int(np.argmax(mcts_probs)),
        "probs": [float(p) for p in mcts_probs]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Create templates and static directories if they don't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    print("\nStarting Connect 4 AI Server...")
    print(f"Device: {device}")
    
    app_host = os.environ.get("APP_HOST", "127.0.0.1")
    app_port = int(os.environ.get("APP_PORT", 5000))
    app_debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    # Pre-fetch the latest model to avoid cold-start latency for the first user
    initial_models = sorted(glob.glob("*.onnx"), reverse=True)
    if initial_models:
        print(f"Pre-loading latest checkpoint for optimization: {initial_models[0]}")
        get_model(initial_models[0])

    app.run(debug=app_debug, host=app_host, port=app_port)
