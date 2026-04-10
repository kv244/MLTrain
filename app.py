import os
import glob
import re
import time
import csv
import torch
import numpy as np
import pathlib
import threading
import urllib.request
import json as _json
from collections import OrderedDict # FIX 6
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import openvino as ov
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

VERSION = "1.7.0"
LAST_COMMIT = "2026-04-10 00:15 UTC"

from mcts import Connect4, run_mcts_simulations
import background_manager # NEW: Dynamic Environment manager

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024  # FIX 8: 64 KB limit
# x_for=1: trust exactly 1 proxy hop (nginx).
# Only works correctly because nginx strips client X-Forwarded-For above.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

RATE_LIMITS = os.environ.get("RATE_LIMIT", "500 per day;200 per hour;15 per minute").split(";")
LIMITER_STORAGE = os.environ.get("LIMITER_STORAGE_URI", "memory://")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=RATE_LIMITS,
    storage_uri=LIMITER_STORAGE
)

if LIMITER_STORAGE.startswith("memory://"):
    print(
        "[WARNING] Rate limiter is using in-process memory storage. "
        "This does not work correctly with multi-worker Gunicorn. "
        "Set LIMITER_STORAGE_URI to a Redis URL for production."
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
MAX_LOADED_MODELS = 3 # FIX 6: LRU cap
LOADED_MODELS = OrderedDict() # FIX 6
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
        self._lock = threading.Lock() # FIX 5: OpenVINO thread safety
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
        self.infer_request = self.compiled_model.create_infer_request()
        self.policy_output = self.compiled_model.output("policy")
        self.value_output = self.compiled_model.output("value")

    def __call__(self, x: torch.Tensor):
        x_np = x.cpu().numpy()
        with self._lock: # FIX 5: OpenVINO thread safety
            # Switch to asynchronous pattern for future concurrency
            self.infer_request.start_async(inputs={0: x_np})
            self.infer_request.wait()
            policy = self.infer_request.get_output_tensor(0).data
            value = self.infer_request.get_output_tensor(1).data
        return torch.from_numpy(policy), torch.from_numpy(value)

    def eval(self):
        pass

def get_model(checkpoint_path):
    """Loads a compiled OpenVINO `.onnx` file or retrieves it from cache."""
    with _model_lock:
        if checkpoint_path not in LOADED_MODELS:
            try:
                model = OpenVINOModel(checkpoint_path)
                LOADED_MODELS[checkpoint_path] = model
                # FIX 6: Evict oldest if cap reached
                if len(LOADED_MODELS) > MAX_LOADED_MODELS:
                    LOADED_MODELS.popitem(last=False)
            except Exception as e:
                return None, str(e)
        else:
            # Move to end for LRU behavior
            LOADED_MODELS.move_to_end(checkpoint_path)
    return LOADED_MODELS[checkpoint_path], None

@app.route("/")
def index():
    return render_template("index.html", version=VERSION)

@app.route("/api/info")
def get_info():
    """Returns runtime hardware info to the frontend."""
    return jsonify({"device": GLOBAL_OV_DEVICE})

@app.route("/api/game_end", methods=["POST"])
def log_game_end():
    """Receives and logs game outcome telemtry."""
    data = request.json
    if not data: return jsonify({"error": "Missing or invalid JSON"}), 400 # FIX 1
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
    if not data: return jsonify({"error": "Missing or invalid JSON"}), 400 # FIX 1

    checkpoint_name = data.get("model", "")
    board_state = data.get("board", []) 
    current_player = data.get("current_player", -1) 

    # FIX 4: Validate current_player
    if current_player not in (-1, 1):
        return jsonify({"error": "Invalid current_player"}), 400

    # FIX 3: safe int coercion
    try:
        simulations = max(1, min(int(data.get("simulations", 800)), 5000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid simulations value"}), 400
    
    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    # Validate board state
    board_arr = np.array(board_state, dtype=np.int8)
    if board_arr.shape != (6, 7) or not np.all(np.isin(board_arr, [-1, 0, 1])):
        return jsonify({"error": "Invalid board state"}), 400

    # FIX 7: board piece-count validation
    count_p1 = int(np.sum(board_arr == 1))
    count_p2 = int(np.sum(board_arr == -1))
    if abs(count_p1 - count_p2) > 1:
        return jsonify({"error": "Invalid board state: inconsistent piece counts"}), 400

    model, error = get_model(str(checkpoint))
    if error:
        return jsonify({"error": f"Failed to load model: {error}"}), 503

    # Reconstruct the game state for the Python engine
    game = Connect4()
    
    # 1. Provide the board state
    game.board = board_arr
    # Numpy arrays from lists will be [6, 7], Python expects [r][c].
    game.current_player = current_player

    # ── Step 6: Dynamic Simulation Budget ──
    if simulations >= 200:
        # Run a fast 50-sim pass to estimate position complexity
        _, root = run_mcts_simulations(
            game, model, device,
            num_sims=50,
            temperature=0,
            add_dirichlet_noise=False,
            return_root=True
        )
        # Calculate move probabilities from visits if root is available
        if root is not None:
            probs = [0] * 7
            candidates = []
            for move, child in root.children.items():
                v_count = child.visit_count
                q_val = child.total_value / v_count if v_count > 0 else 0
                probs[move] = v_count
                candidates.append((move, v_count, q_val))
            
            # Sort and log top 3 for debugging
            candidates.sort(key=lambda x: x[1], reverse=True)
            log_msg = f"AI thinking: " + ", ".join([f"Col {m}(V:{v}, Q:{q:.2f})" for m,v,q in candidates[:3]])
            print(log_msg)
            
            root_val = root.q_value
            # If position is contested/complex, boost the budget
            if abs(root_val) < 0.4:
                simulations = min(5000, int(simulations * 1.5))
                print(f"[Adaptive] Contested position (q={root_val:.2f}). Boosting sims to {simulations}")
            # If position is nearly decided, reduce the budget
            elif abs(root_val) > 0.85:
                simulations = max(50, int(simulations * 0.5))
                print(f"[Adaptive] Decided position (q={root_val:.2f}). Reducing sims to {simulations}")
        else:
            print("[Adaptive] Tactical Short-Circuit detected. Bypassing simulation boost.")

    print(f"Evaluating board using {checkpoint_name} for player {current_player} (sims={simulations})...")

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
    
    # ── Step 1: Detect Winning Cells ──
    winning_cells = []
    game_clone = game.clone()
    win_result = game_clone.play(best_move)
    if win_result:
        row, col = win_result
        # The game engine `play` was modified to move internal state,
        # but we need to check the state AFTER the move was made.
        # However, `play` in Connect4 already placed the piece.
        win_cells = game_clone.check_win(row, col)
        if win_cells:
            winning_cells = [[int(r), int(c)] for r, c in win_cells]

    return jsonify({
        "move": best_move,
        "probs": [float(p) for p in mcts_probs],
        "inference_time_ms": inference_time * 1000,
        "winning_cells": winning_cells
    })

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None

@app.route("/api/assess", methods=["POST"])
def assess_move():
    """Evaluates a specific move against the AI's best recommended move."""
    data = request.json
    if not data: return jsonify({"error": "Missing or invalid JSON"}), 400 # FIX 1

    checkpoint_name = data.get("model", "")
    board_state = data.get("board", []) # This should be the board BEFORE the move
    move = data.get("move", -1)
    
    # FIX 2: validate move param
    if not isinstance(move, int) or move < 0 or move >= 7:
        return jsonify({"error": "Invalid move"}), 400

    current_player = data.get("current_player", 1)

    # FIX 4: Validate current_player
    if current_player not in (-1, 1):
        return jsonify({"error": "Invalid current_player"}), 400

    # FIX 3: safe int coercion
    try:
        simulations = max(1, min(int(data.get("simulations", 800)), 5000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid simulations value"}), 400

    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    # Validate board state
    board_arr = np.array(board_state, dtype=np.int8)
    if board_arr.shape != (6, 7) or not np.all(np.isin(board_arr, [-1, 0, 1])):
        return jsonify({"error": "Invalid board state"}), 400

    model, error = get_model(str(checkpoint))
    if error:
        return jsonify({"error": f"Failed to load model: {error}"}), 503

    game = Connect4()
    game.board = board_arr
    game.current_player = current_player

    # FIX 8: Board full guard
    if not game.get_valid_moves():
        return jsonify({"error": "Board is full — no moves available"}), 400

    # FIX 7: Move playable guard
    if move not in game.get_valid_moves():
        return jsonify({"error": "Column is full or invalid"}), 400

    # Get AI's opinion on this state.
    # temperature=1.0 gives visit-count proportions, so p_move/max_p is a
    # continuous ratio. temperature=0 returns a one-hot which collapses every
    # non-best move to p=0 and scores everything as either 1 or 5 stars.
    mcts_probs = run_mcts_simulations(
        game, model, device,
        num_sims=simulations,
        temperature=1.0,
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

    
    best_move = int(np.argmax(mcts_probs))
    quote = "Analysis complete."
    if gemini_model:
        try:
            move_quality_pct = int(round(p_move / max_p * 100)) if max_p > 0 else 0
            same_as_best = (move == best_move)
            move_context = (
                f"played column {move} (the optimal move was column {best_move}), "
                f"capturing {move_quality_pct}% of the AI's confidence"
                if not same_as_best else
                f"played column {move}, which IS the optimal move ({move_quality_pct}% confidence match)"
            )
            prompt = (
                f"You are the 'Grid Overseer', a cynical and sophisticated AI commentator for a high-stakes cyberpunk Connect 4 terminal. "
                f"A player just {move_context}. Rated {score}/5 stars.\n"
                f"Provide a unique, non-generic response in the following format:\n"
                f"LABEL: [catchy 1-2 word label, e.g. 'Neural Spike', 'Logic Leak', 'Ghost Protocol']\n"
                f"QUOTE: [short atmospheric quote referencing the move, max 10 words]\n"
                f"Tone guide: 1-2 stars (mocking/cold), 3-4 stars (neutral/impressed), 5 stars (fascinated/alarmed). "
                f"Use cyberpunk slang. Output ONLY the requested format."
            )
            response = gemini_model.generate_content(
                prompt,
                request_options={"timeout": 8}
            )
            responseText = response.text
            
            # More robust parsing using regex or keyword search
            label_match = re.search(r"LABEL:\s*(.*)", responseText, re.IGNORECASE)
            quote_match = re.search(r"QUOTE:\s*(.*)", responseText, re.IGNORECASE)
            
            if label_match:
                comment = label_match.group(1).split("\n")[0].strip().strip('"*#')
            if quote_match:
                quote = quote_match.group(1).split("\n")[0].strip().strip('"*#')

        except Exception as e:
            print(f"Gemini Assessment Error: {e}")
            # Robust fallback to ensure the UI doesn't look empty/broken
            fallbacks = {
                1: {"label": "Blunder", "quote": "Biological error detected. Tactical decay."},
                2: {"label": "Inaccurate", "quote": "Suboptimal pattern. Recalibrating logic."},
                3: {"label": "Standard", "quote": "Acceptable simulation data. Proceeding."},
                4: {"label": "Strong", "quote": "High-efficiency maneuver. Synergy quantified."},
                5: {"label": "Brilliant", "quote": "Statistical anomaly! Your patterns are evolving."}
            }
            f = fallbacks.get(score, fallbacks[3])
            comment, quote = f["label"], f["quote"]

    # ── Step 1: Detect Winning Cells ──
    winning_cells = []
    game_clone = game.clone()
    win_result = game_clone.play(move)
    if win_result:
        row, col = win_result
        win_cells = game_clone.check_win(row, col)
        if win_cells:
            winning_cells = [[int(r), int(c)] for r, c in win_cells]

    return jsonify({
        "score": score,
        "comment": comment,
        "ai_quote": quote,
        "best_move": best_move,
        "probs": [float(p) for p in mcts_probs],
        "winning_cells": winning_cells
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/geoip")
@limiter.limit("10 per minute")
def geoip():
    """Proxy geo-IP lookup server-side to avoid CSP/CORS issues."""
    try:
        with urllib.request.urlopen("https://geolocation-db.com/json/", timeout=4) as resp:
            data = _json.loads(resp.read().decode())
        return jsonify({"country_name": data.get("country_name", "")})
    except Exception:
        return jsonify({"country_name": ""}), 200

# NEW: Dynamic Environment Background Refresh
@app.route("/api/admin/refresh_background")
@limiter.limit("2 per minute")
def refresh_background():
    """Manually triggers a background update using Gemini + Vertex AI Imagen."""
    auth_token = request.args.get("auth")
    # For simplicity, we check an env-based secret
    expected_token = os.environ.get("ADMIN_TOKEN")
    if not expected_token or auth_token != expected_token:
        return jsonify({"error": "Unauthorized"}), 401
    
    success = background_manager.update_background()
    if success:
        return jsonify({"status": "Success", "message": "Background updated"}), 200
    else:
        return jsonify({"status": "Error", "message": "Failed to update background"}), 505

# FIX 9: security response headers
@app.after_request
def set_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; font-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com; style-src 'self' https://fonts.googleapis.com; script-src 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

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

    # Dynamic Environment: Startup check for stale background
    if background_manager.is_background_stale():
        print("[App] Stale background detected, triggering update...")
        # Run in a separate thread to avoid blocking startup
        def _safe_update():
            try:
                background_manager.update_background()
            except Exception as e:
                print(f"[App] Background update failed: {e}")
        threading.Thread(target=_safe_update, daemon=True).start()

    app.run(debug=app_debug, host=app_host, port=app_port)
