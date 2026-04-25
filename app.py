import os
import glob
import hashlib
import re
import time
import csv
import datetime
import json
import torch
import numpy as np
import pathlib
import threading
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import openvino as ov
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import logging

load_dotenv(pathlib.Path(__file__).parent / ".env")


def _setup_logging():
    """Wire Python's logging to Google Cloud Logging when running on GCP.
    Falls back to stderr basicConfig for local dev / missing package."""
    try:
        import google.cloud.logging as cloud_logging
        cloud_logging.Client().setup_logging()
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )


_setup_logging()
logger = logging.getLogger(__name__)


def _git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

VERSION = _git_rev()

from mcts import Connect4, run_mcts_simulations
import background_manager
import bigquery_tracker

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
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
    logger.warning(
        "Rate limiter is using in-process memory storage. "
        "This does not work correctly with multi-worker Gunicorn. "
        "Set LIMITER_STORAGE_URI to a Redis URL for production."
    )

# Initialise BigQuery tracker at module load time so it runs under gunicorn.
# (Previously inside __main__ only — invisible to gunicorn workers.)
bigquery_tracker.init()

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Too Many Requests", description=str(e.description)), 429

MODELS_DIR = pathlib.Path(".").resolve()
_csv_lock = threading.Lock()
_model_lock = threading.Lock()

# Opening book: {board_hash: {"avoid_move": col, "count": N}}
# Built offline by build_opening_book.py from human-win game trajectories.
_opening_book: dict = {}
_BOOK_PATH = MODELS_DIR / "opening_book.json"
if _BOOK_PATH.exists():
    try:
        with open(_BOOK_PATH) as _f:
            _opening_book = json.load(_f)
        logger.info("Opening book loaded: %d positions", len(_opening_book))
    except Exception as _e:
        logger.error("Opening book load failed: %s", _e)

def _book_hash(board_arr: np.ndarray) -> str:
    return hashlib.sha256(board_arr.astype(np.int8).tobytes()).hexdigest()[:16]

# Single shared OpenVINO Core instance — creating ov.Core() is heavyweight
# (scans hardware, loads plugins).  Re-using one instance avoids redundant
# initialisation on every model load and when querying available devices.
_ov_core = ov.Core()

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
MAX_LOADED_MODELS = 3
LOADED_MODELS = OrderedDict()
device = torch.device("cpu") # PyTorch tensor device for CPU memory handling

def get_ov_device():
    available = _ov_core.available_devices
    if "NPU" in available: return "NPU"
    if "GPU" in available: return "GPU"
    return "CPU"

GLOBAL_OV_DEVICE = get_ov_device()

class OpenVINOModel:
    """A wrapper to make an OpenVINO model behave like a PyTorch model for inference."""
    def __init__(self, model_path: str):
        self._lock = threading.Lock()

        logger.info("OpenVINO initializing on device: %s", GLOBAL_OV_DEVICE)

        ov_model = _ov_core.read_model(model=model_path)
        self.compiled_model = _ov_core.compile_model(
            model=ov_model,
            device_name=GLOBAL_OV_DEVICE,
            config={"PERFORMANCE_HINT": "LATENCY"}
        )
        self.infer_request = self.compiled_model.create_infer_request()

    def __call__(self, x: torch.Tensor):
        x_np = x.cpu().numpy()
        with self._lock:
            self.infer_request.start_async(inputs={0: x_np})
            self.infer_request.wait()
            policy = self.infer_request.get_output_tensor(0).data
            value = self.infer_request.get_output_tensor(1).data
        return torch.from_numpy(policy), torch.from_numpy(value)

def get_model(checkpoint_path):
    """Loads a compiled OpenVINO `.onnx` file or retrieves it from cache."""
    with _model_lock:
        if checkpoint_path not in LOADED_MODELS:
            try:
                model = OpenVINOModel(checkpoint_path)
                LOADED_MODELS[checkpoint_path] = model
              
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
    return render_template("index.html", version=VERSION, kofi_tagline=_kofi_tagline)

@app.route("/api/info")
def get_info():
    """Returns runtime hardware info to the frontend."""
    return jsonify({"device": GLOBAL_OV_DEVICE})

@app.route("/api/session", methods=["POST"])
@limiter.limit("30 per minute")
def log_session():
    """Record a page-load visit in BigQuery.
    Body: { "country": "<string>" }  (country resolved by client-side geo-IP)
    Performs INSERT for new IPs, UPDATE for returning visitors."""
    data    = request.json or {}
    country = data.get("country", "")
    ip      = request.remote_addr
    bigquery_tracker.record_session(ip, country)
    return jsonify({"success": True})


@app.route("/api/game_end", methods=["POST"])
@limiter.limit("60 per hour")
def log_game_end():
    """Receives and logs game outcome telemetry to CSV and BigQuery.
    Body: { "winner": "human"|"ai"|"draw", "model": "<str>", "moves": <int>,
            "country": "<str>" }"""
    data = request.json
    if not data:
        return jsonify({"error": "Missing or invalid JSON"}), 400
    winner        = data.get("winner",     "unknown")
    model_version = data.get("model",      "unknown")
    moves         = data.get("moves",      0)
    country       = data.get("country",    "")
    difficulty    = data.get("difficulty", "hard")
    ip            = request.remote_addr

    if winner not in ("human", "ai", "draw"):
        return jsonify({"error": "Invalid winner"}), 400
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "hard"  # silently normalise unknown values

    # CSV log (existing behaviour)
    results_file = os.environ.get("RESULTS_LOG_PATH", "game_results.csv")
    with _csv_lock:
        file_exists = os.path.isfile(results_file)  # checked inside lock to avoid TOCTOU
        with open(results_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["timestamp", "model", "winner", "moves"])
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"),
                             model_version, winner, moves])

    # BigQuery — fire-and-forget in background thread
    bigquery_tracker.record_game(ip, winner, moves, difficulty)

    # Log full trajectory for offline retraining and opening-book construction.
    # Only worth storing when the human wins or draws (AI losses are the learning signal).
    move_sequence  = data.get("move_sequence") or []
    human_player_v = data.get("human_player", 1)
    if (winner in ("human", "draw")
            and isinstance(move_sequence, list)
            and 6 < len(move_sequence) <= 42
            and all(isinstance(m, int) and 0 <= m <= 6 for m in move_sequence)):
        bigquery_tracker.record_human_game(ip, winner, move_sequence, difficulty, human_player_v)

    return jsonify({"success": True})

@app.route("/api/record_win", methods=["POST"])
@limiter.limit("10 per hour")
def record_win():
    """Save a player win to the hall-of-fame table.
    Body: { "name": "<str>", "difficulty": "<str>", "simulations": <int>, "moves": <int> }"""
    data = request.json
    if not data:
        return jsonify({"error": "Missing or invalid JSON"}), 400

    name        = str(data.get("name", "")).strip()[:50]
    difficulty  = data.get("difficulty", "hard")
    moves       = data.get("moves", 0)
    simulations = data.get("simulations", 400)

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "hard"
    try:
        moves       = max(1, int(moves))
        simulations = max(1, min(int(simulations), 2000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid moves or simulations"}), 400

    bigquery_tracker.record_win(request.remote_addr, name, difficulty, simulations, moves)
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
    if not data: return jsonify({"error": "Missing or invalid JSON"}), 400

    checkpoint_name = data.get("model", "")
    board_state = data.get("board", [])
    current_player = data.get("current_player", -1)
    difficulty = data.get("difficulty", "hard").lower()
    if difficulty not in ("easy", "medium", "hard"):
        return jsonify({"error": "Invalid difficulty"}), 400

  
    if current_player not in (-1, 1):
        return jsonify({"error": "Invalid current_player"}), 400

    # safe int coercion — hard cap at 2000 (matches UI slider max)
    try:
        simulations = max(1, min(int(data.get("simulations", 800)), 2000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid simulations value"}), 400

    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    # Validate board state
    board_arr = np.array(board_state, dtype=np.int8)
    if board_arr.shape != (6, 7) or not np.all(np.isin(board_arr, [-1, 0, 1])):
        return jsonify({"error": "Invalid board state"}), 400

  
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
            logger.debug(log_msg)

            root_val = root.q_value
            # If position is contested/complex, boost the budget
            if abs(root_val) < 0.4:
                simulations = min(2000, int(simulations * 1.5))
                logger.debug("[Adaptive] Contested position (q=%.2f). Boosting sims to %d", root_val, simulations)
            # If position is nearly decided, reduce the budget
            elif abs(root_val) > 0.85:
                simulations = max(50, int(simulations * 0.5))
                logger.debug("[Adaptive] Decided position (q=%.2f). Reducing sims to %d", root_val, simulations)
        else:
            logger.debug("[Adaptive] Tactical Short-Circuit detected. Bypassing simulation boost.")

    # Difficulty: random move chance (easy=2/3, medium=1/3, hard=0)
    random_chance = {"easy": 2/3, "medium": 1/3}.get(difficulty, 0)
    if random_chance > 0 and np.random.random() < random_chance:
        valid_cols = [c for c in range(7) if game.board[0][c] == 0]
        best_move = int(np.random.choice(valid_cols))
        logger.debug("[Difficulty:%s] Random move → col %d", difficulty, best_move)
        return jsonify({"move": best_move, "probs": [0.0]*7, "winning_cells": []})

    logger.debug("Evaluating board using %s for player %d (sims=%d)", checkpoint_name, current_player, simulations)

    start_time = time.time()
    mcts_probs = run_mcts_simulations(
        game, model, device,
        num_sims=simulations,
        temperature=0,             # Greedy, we want the *best* move
        add_dirichlet_noise=False  # No exploration in production play
    )
    inference_time = time.time() - start_time
    
    bigquery_tracker.record_telemetry(checkpoint_name, simulations, inference_time)
    
    best_move = int(np.argmax(mcts_probs))

    # Opening book: if MCTS picked a historically-losing move in early positions,
    # steer toward the second-best option instead.
    piece_count = int(np.count_nonzero(board_arr))
    if _opening_book and piece_count < 12:
        book_entry = _opening_book.get(_book_hash(board_arr))
        if book_entry and book_entry.get("count", 0) >= 5 and best_move == book_entry["avoid_move"]:
            for m in sorted(range(7), key=lambda m: mcts_probs[m], reverse=True):
                if m != book_entry["avoid_move"] and game.board[0][m] == 0:
                    logger.debug("[OpeningBook] col %d→%d (avoid_move seen %d×)", best_move, m, book_entry['count'])
                    best_move = int(m)
                    break

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

# Configure Gemini (google.genai SDK — replaces deprecated google.generativeai)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.route("/api/assess", methods=["POST"])
def assess_move():
    """Evaluates a specific move against the AI's best recommended move."""
    data = request.json
    if not data: return jsonify({"error": "Missing or invalid JSON"}), 400

    checkpoint_name = data.get("model", "")
    board_state = data.get("board", []) # This should be the board BEFORE the move
    move = data.get("move", -1)
    
  
    if not isinstance(move, int) or move < 0 or move >= 7:
        return jsonify({"error": "Invalid move"}), 400

    current_player = data.get("current_player", 1)

  
    if current_player not in (-1, 1):
        return jsonify({"error": "Invalid current_player"}), 400

    # safe int coercion — hard cap at 2000 (matches UI slider max)
    try:
        simulations = max(1, min(int(data.get("simulations", 800)), 2000))
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

  
    if not game.get_valid_moves():
        return jsonify({"error": "Board is full — no moves available"}), 400

  
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
    if gemini_client:
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
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    http_options=genai_types.HttpOptions(timeout=15000)
                )
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
            logger.error("Gemini assessment error: %s", e)
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

@app.route("/api/game_summary", methods=["POST"])
@limiter.limit("10 per hour")
def game_summary():
    """Replay a completed game and identify the AI's weakest move (largest deviation from optimal).

    Algorithm:
      - Replay move_sequence using a fresh Connect4 game.
      - At each AI turn, run a short MCTS search (temperature=1.0 for proportional probs).
      - quality_ratio = p_actual / p_best:  1.0 = AI played the best move, ~0.0 = blunder.
      - Tactical-override plies (immediate win/block) are skipped — those are forced moves.
      - Return the ply with the lowest quality_ratio as the "blunder".

    Body:  { "model": str, "move_sequence": [col,...], "human_player": 1|-1,
             "simulations": int (default 100, max 200) }
    Returns: { "blunder_ply": int, "blunder_col": int, "better_col": int,
               "quality_ratio": float }
          or { "blunder": null } when all AI moves were forced/tactical.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing or invalid JSON"}), 400

    checkpoint_name = data.get("model", "")
    move_sequence   = data.get("move_sequence") or []
    human_player    = data.get("human_player", 1)

    # Validate move_sequence: must be a list of 7–42 ints in [0,6].
    # < 7 moves: game can't have ended legitimately; > 42: board only has 42 cells.
    if (not isinstance(move_sequence, list)
            or not (7 <= len(move_sequence) <= 42)
            or not all(isinstance(m, int) and 0 <= m <= 6 for m in move_sequence)):
        return jsonify({"error": "Invalid move_sequence"}), 400

    if human_player not in (1, -1):
        return jsonify({"error": "Invalid human_player"}), 400

    # Hard cap at 100 sims per AI turn regardless of what the caller sends.
    # Worst case: 21 AI turns × 100 sims = 2100 sims total, safely under
    # Gunicorn's 30s worker timeout. 200 sims × 21 turns risks a timeout kill.
    try:
        simulations = max(50, min(int(data.get("simulations", 100)), 100))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid simulations value"}), 400

    checkpoint = _resolve_checkpoint(checkpoint_name)
    if not checkpoint:
        return jsonify({"error": "Invalid model"}), 400

    model, error = get_model(str(checkpoint))
    if error:
        return jsonify({"error": f"Failed to load model: {error}"}), 503

    game = Connect4()

    # worst_ai_move: tracks the ply where the AI deviated most from optimal.
    worst_ai_move = None  # dict: { ply, col, better_col, quality_ratio }

    for ply, col in enumerate(move_sequence):
        # Only evaluate positions where it was the AI's turn.
        if game.current_player != human_player:
            # temperature=1.0: probs are proportional to visit counts, giving a
            # continuous quality ratio. temperature=0 collapses to one-hot, which
            # would score every non-best move as quality_ratio=0 (all equal "blunders").
            probs, root = run_mcts_simulations(
                game, model, device,
                num_sims=simulations,
                temperature=1.0,
                add_dirichlet_noise=False,
                return_root=True,
            )

            # root is None when tactical override fired (forced win/block move).
            # Those are correct-by-definition; skip quality evaluation.
            if root is not None:
                p_best   = float(np.max(probs))
                p_actual = float(probs[col]) if 0 <= col < 7 else 0.0
                ratio    = p_actual / p_best if p_best > 0.0 else 1.0
                better_col = int(np.argmax(probs))

                if worst_ai_move is None or ratio < worst_ai_move["quality_ratio"]:
                    worst_ai_move = {
                        "ply":           ply,
                        "col":           col,
                        "better_col":    better_col,
                        "quality_ratio": ratio,
                    }

        # Advance game state — called ONCE per ply, AFTER the MCTS block above.
        # game.play() mutates board + current_player; calling it inside the MCTS
        # block would corrupt state for all subsequent plies.
        result = game.play(col)
        if result is None:
            # Column was full — malformed sequence; stop replaying.
            break
        r, c = result
        if game.check_win(r, c):
            break  # Game over — no more plies to evaluate.

    if worst_ai_move is None:
        # All AI turns were tactical overrides, or the sequence had no AI turns.
        return jsonify({"blunder": None})

    return jsonify({
        "blunder_ply":   worst_ai_move["ply"],
        "blunder_col":   worst_ai_move["col"],
        "better_col":    worst_ai_move["better_col"],
        "quality_ratio": round(worst_ai_move["quality_ratio"], 4),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/sitemap.xml")
@limiter.exempt
def sitemap():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        '<url>'
        '<loc>https://c4star.com/</loc>'
        '<changefreq>weekly</changefreq>'
        '<priority>1.0</priority>'
        '</url>'
        '</urlset>'
    )
    return app.response_class(xml, mimetype="application/xml")

@app.route("/robots.txt")
@limiter.exempt
def robots():
    txt = "User-agent: *\nAllow: /\nSitemap: https://c4star.com/sitemap.xml\n"
    return app.response_class(txt, mimetype="text/plain")

# Simple 5-minute in-memory cache so every page load doesn't bill a BQ query
_stats_cache = {"data": None, "expires": 0}
_stats_cache_lock = threading.Lock()

@app.route("/api/stats")
@limiter.limit("30 per minute")
def api_stats():
    """Return global game totals from BigQuery, cached for 5 minutes."""
    now = time.time()
    with _stats_cache_lock:
        if _stats_cache["data"] and now < _stats_cache["expires"]:
            return jsonify(_stats_cache["data"])

    if not bigquery_tracker._enabled:
        return jsonify({"total_games": None})

    try:
        ref = bigquery_tracker._table_ref
        # Aggregates totals across all player IPs. 
        # COALESCE handles cases where the table might be empty.
        row = next(iter(bigquery_tracker._client.query(f"""
            SELECT
                COALESCE(SUM(total_games),  0) AS total_games,
                COALESCE(SUM(player_wins),  0) AS player_wins,
                COALESCE(SUM(ai_wins),      0) AS ai_wins,
                COALESCE(COUNT(*),          0) AS unique_players
            FROM `{ref}`
        """).result()))
        data = {
            "total_games":     int(row.total_games),
            "player_wins":     int(row.player_wins),
            "ai_wins":         int(row.ai_wins),
            "unique_players":  int(row.unique_players),
        }
    except Exception as e:
        logger.error("[api/stats] BQ query failed: %s", e)
        return jsonify({"total_games": None})

    with _stats_cache_lock:
        _stats_cache["data"]    = data
        _stats_cache["expires"] = now + 300  # 5 minutes
    return jsonify(data)

_winner_cache = {"data": None, "expires": 0}
_winner_cache_lock = threading.Lock()

@app.route("/api/recent_winner")
@limiter.limit("30 per minute")
def api_recent_winner():
    """Return the most recent entry from win_records, cached for 60 seconds."""
    now = time.time()
    with _winner_cache_lock:
        if _winner_cache["data"] and now < _winner_cache["expires"]:
            return jsonify(_winner_cache["data"])

    if not bigquery_tracker._enabled or not bigquery_tracker._win_table_ref:
        return jsonify({"winner": None})

    try:
        ref = bigquery_tracker._win_table_ref
        # Fetches the absolute latest entry from the hall-of-fame table.
        rows = list(bigquery_tracker._client.query(f"""
            SELECT name, difficulty, simulations, moves,
                   FORMAT_TIMESTAMP('%Y-%m-%d', recorded_at) AS date
            FROM `{ref}`
            ORDER BY recorded_at DESC
            LIMIT 1
        """).result())
        if not rows:
            data = {"winner": None}
        else:
            r = rows[0]
            data = {
                "winner": {
                    "name":        r.name,
                    "difficulty":  r.difficulty,
                    "simulations": int(r.simulations),
                    "moves":       int(r.moves),
                    "date":        r.date,
                }
            }
    except Exception as e:
        logger.error("[api/recent_winner] BQ query failed: %s", e)
        return jsonify({"winner": None})

    with _winner_cache_lock:
        _winner_cache["data"]    = data
        _winner_cache["expires"] = now + 60  # 60-second cache
    return jsonify(data)


_leaderboard_cache = {"data": None, "expires": 0}
_leaderboard_cache_lock = threading.Lock()

@app.route("/api/leaderboard")
@limiter.limit("30 per minute")
def api_leaderboard():
    """
    Returns the 5 most recent player wins, cached for 60 seconds.
    This powers the 'Hall of Fame' scrolling list in the UI.
    """
    now = time.time()
    with _leaderboard_cache_lock:
        if _leaderboard_cache["data"] and now < _leaderboard_cache["expires"]:
            return jsonify(_leaderboard_cache["data"])

    if not bigquery_tracker._enabled or not bigquery_tracker._win_table_ref:
        return jsonify({"winners": []})

    try:
        ref = bigquery_tracker._win_table_ref
        # Retrieves the 5 most recent victors for the Hall of Fame display.
        rows = list(bigquery_tracker._client.query(f"""
            SELECT name, difficulty, simulations, moves,
                   FORMAT_TIMESTAMP('%b %d %Y', recorded_at) AS date
            FROM `{ref}`
            ORDER BY recorded_at DESC
            LIMIT 5
        """).result())
        data = {"winners": [
            {
                "name":        r.name,
                "difficulty":  r.difficulty,
                "simulations": int(r.simulations),
                "moves":       int(r.moves),
                "date":        r.date,
            }
            for r in rows
        ]}
    except Exception as e:
        logger.error("[api/leaderboard] BQ query failed: %s", e)
        return jsonify({"winners": []})

    with _leaderboard_cache_lock:
        _leaderboard_cache["data"]    = data
        _leaderboard_cache["expires"] = now + 60
    return jsonify(data)


# News banner shown for one week after deployment, then silently disappears.
_NEWS_EXPIRY = datetime.date(2026, 4, 30)
_NEWS_BANNER = (
    "New: stronger model deployed · "
    "the AI now learns from games you win — "
    "your victories are used to retrain and build an opening book"
)

def _with_news(d: dict) -> dict:
    """Inject news_banner into a welcome-strings dict while within the expiry window."""
    if datetime.date.today() <= _NEWS_EXPIRY:
        return {**d, "news_banner": _NEWS_BANNER}
    return d


# Strings shown in the welcome toast — Gemini translates these per country.
_ENGLISH_STRINGS = {
    "greeting":         "Thank you for joining me from {country}!",
    "games_globally":   "{n} games played globally",
    "wallpaper_renews": "wallpaper renews in {n} days",
    "wallpaper_soon":   "wallpaper refreshing soon",
    "last_winner":      "Last winner",
    "thoughts":         "thoughts",
    "moves":            "moves",
    "subtitle":         "Easy/Medium/Hard sets move randomness \u00b7 Think Intensity (100\u20132000 sims) controls search depth \u2014 mix both to tune difficulty",
    "help_title":       "How to Play",
    "help_intro":       "Two players take turns dropping coloured pieces into a 7\u00d76 grid.",
    "help_fall":        "Pieces fall to the lowest empty slot in the chosen column.",
    "help_win":         "First to connect 4 in a line \u2014 horizontal, vertical, or diagonal \u2014 wins.",
    "help_draw":        "If all 42 squares fill up with no winner, the game is a draw.",
    "help_tip":         "Tip: raise Think Intensity for a stronger (but slower) AI.",
    "help_close":       "Got it!",
}

# Countries where English is the right default — skip Gemini entirely.
_ENGLISH_COUNTRIES = {
    "", "singapore", "united states", "united kingdom", "australia",
    "canada", "new zealand", "ireland", "the physical realm",
}

_strings_cache      = {}   # country → translated strings dict
_strings_cache_lock = threading.Lock()

@app.route("/api/welcome_strings")
@limiter.limit("60 per minute")
def welcome_strings():
    """Return welcome-toast UI strings translated into the visitor's language.
    Translations are cached per country for the lifetime of the process."""
    country = request.args.get("country", "").strip()

    if country.lower() in _ENGLISH_COUNTRIES:
        return jsonify(_with_news(_ENGLISH_STRINGS))

    with _strings_cache_lock:
        if country in _strings_cache:
            return jsonify(_with_news(_strings_cache[country]))

    if not gemini_client:
        return jsonify(_ENGLISH_STRINGS)

    try:
        prompt = (
            f"Translate the JSON values below from English into the primary language of {country}. "
            f"Rules: return ONLY valid JSON with exactly the same keys; "
            f"do not translate placeholder tokens like {{country}} or {{n}} — keep them verbatim; "
            f"do not translate numbers or punctuation characters like · or —; "
            f"if {country} primarily uses English, return the original values unchanged.\n\n"
            f"{json.dumps(_ENGLISH_STRINGS, ensure_ascii=False)}"
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=15000)
            )
        )
        text = response.text.strip()
        # Strip markdown fences that Gemini sometimes adds
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())

        translated = json.loads(text)
        if set(translated.keys()) == set(_ENGLISH_STRINGS.keys()):
            with _strings_cache_lock:
                _strings_cache[country] = translated
            return jsonify(_with_news(translated))
    except Exception as e:
        logger.warning("[welcome_strings] Translation failed for %r: %s", country, e)

    return jsonify(_with_news(_ENGLISH_STRINGS))


@app.route("/api/geoip")
@limiter.limit("10 per minute")
def geoip():
    """Return wallpaper renewal countdown only.
    Geolocation is now done browser-side (geolocation-db.com added to CSP
    connect-src) so the lookup uses the client's real IP, not the server's."""
    try:
        mtime = background_manager.BG_PATH.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        days_left = max(0, int(7 - age_days))  # floor: whole days remaining
        bg_mtime = int(mtime)
    except Exception:
        days_left = None
        bg_mtime = None
    return jsonify({"wallpaper_days_left": days_left, "bg_mtime": bg_mtime})

# NEW: Dynamic Environment Background Refresh
@app.route("/admin/<token>")
def admin_dashboard(token):
    """Analytics dashboard — access restricted to holders of ADMIN_TOKEN in URL.
    Returns 404 (not 403) to avoid revealing the endpoint exists."""
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected or token != expected:
        from flask import abort
        abort(404)

    rows, daily, totals, error = [], [], {}, None
    if bigquery_tracker._enabled:
        try:
            client = bigquery_tracker._client
            ref    = bigquery_tracker._table_ref

            # ── Per-IP summary ────────────────────────────────────────────────
            rows = list(client.query(f"""
                SELECT
                    ip_address,
                    COALESCE(country, 'Unknown')                AS country,
                    DATE(first_seen)                            AS first_day,
                    DATE(last_seen)                             AS last_day,
                    total_visits,
                    total_games,
                    player_wins,
                    ai_wins,
                    draws,
                    total_moves,
                    COALESCE(easy_games,   0)                   AS easy_games,
                    COALESCE(medium_games, 0)                   AS medium_games,
                    COALESCE(hard_games,   0)                   AS hard_games,
                    ROUND(SAFE_DIVIDE(player_wins,
                          NULLIF(total_games, 0)) * 100, 1)    AS win_pct
                FROM `{ref}`
                ORDER BY last_seen DESC
            """).result())

            # ── Daily new-visitor + games totals (last 30 days) ───────────────
            daily = list(client.query(f"""
                SELECT
                    DATE(first_seen)  AS day,
                    COUNT(*)          AS new_visitors,
                    SUM(total_games)  AS games_that_day
                FROM `{ref}`
                GROUP BY day
                ORDER BY day DESC
                LIMIT 30
            """).result())

            # ── Grand totals ──────────────────────────────────────────────────
            t = list(client.query(f"""
                SELECT
                    COUNT(*)          AS unique_ips,
                    SUM(total_visits) AS total_visits,
                    SUM(total_games)  AS total_games,
                    SUM(player_wins)  AS player_wins,
                    SUM(ai_wins)      AS ai_wins,
                    SUM(draws)                       AS draws,
                    SUM(total_moves)                 AS total_moves,
                    SUM(COALESCE(easy_games,   0))   AS easy_games,
                    SUM(COALESCE(medium_games, 0))   AS medium_games,
                    SUM(COALESCE(hard_games,   0))   AS hard_games
                FROM `{ref}`
            """).result())
            totals = dict(t[0]) if t else {}
        except Exception as exc:
            error = str(exc)
    else:
        error = "BigQuery not configured (GCP_PROJECT_ID missing)."

    return render_template("admin.html",
                           rows=rows, daily=daily, totals=totals, error=error,
                           admin_token=token)


_bg_update_state = {"running": False, "last_result": None}  # guarded by GIL (single worker)

_book_build_state = {"running": False, "last_result": None}

@app.route("/api/admin/rebuild_opening_book", methods=["POST"])
@limiter.limit("2 per minute")
def rebuild_opening_book():
    """Rebuild opening_book.json from human-win games in BigQuery.
    Token is sent in the request body (not the URL) to keep it out of server logs."""
    expected_token = os.environ.get("ADMIN_TOKEN")
    data = request.get_json(silent=True) or {}
    if not expected_token or data.get("token", "") != expected_token:
        return jsonify({"error": "Unauthorized"}), 401

    if _book_build_state["running"]:
        return jsonify({"status": "running"}), 202

    def _do_build():
        import subprocess
        _book_build_state["running"] = True
        _book_build_state["last_result"] = None
        try:
            result = subprocess.run(
                [".venv/bin/python", "build_opening_book.py", "--min-count", "2"],
                cwd=str(MODELS_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("Opening book rebuilt successfully:\n%s", result.stdout.strip())
                # Reload the book into memory without restarting the server
                global _opening_book
                book_path = MODELS_DIR / "opening_book.json"
                if book_path.exists():
                    with open(book_path) as f:
                        _opening_book = json.load(f)
                    logger.info("Opening book reloaded: %d positions", len(_opening_book))
                _book_build_state["last_result"] = "ok"
            else:
                logger.error("Opening book rebuild failed:\n%s", result.stderr.strip())
                _book_build_state["last_result"] = "error"
        except Exception as e:
            logger.error("Opening book rebuild exception: %s", e)
            _book_build_state["last_result"] = "error"
        finally:
            _book_build_state["running"] = False

    threading.Thread(target=_do_build, daemon=True).start()
    return jsonify({"status": "started"}), 202


@app.route("/api/admin/refresh_background", methods=["POST"])
@limiter.limit("2 per minute")
def refresh_background():
    """Start a background image update in a daemon thread and return immediately.
    Token is sent in the request body (not the URL) to keep it out of server logs."""
    expected_token = os.environ.get("ADMIN_TOKEN")
    data = request.get_json(silent=True) or {}
    if not expected_token or data.get("token", "") != expected_token:
        return jsonify({"error": "Unauthorized"}), 401

    if _bg_update_state["running"]:
        return jsonify({"status": "running"}), 202

    mtime_before = None
    try:
        mtime_before = background_manager.BG_PATH.stat().st_mtime
    except Exception:
        pass

    def _do_update():
        _bg_update_state["running"] = True
        _bg_update_state["last_result"] = None
        ok = background_manager.update_background()
        _bg_update_state["last_result"] = "ok" if ok else "error"
        _bg_update_state["running"] = False

    threading.Thread(target=_do_update, daemon=True).start()
    return jsonify({"status": "started", "mtime_before": mtime_before}), 202


@app.route("/api/admin/bg_status")
@limiter.limit("30 per minute")
def bg_status():
    """Poll endpoint: returns current bg mtime and whether an update is running."""
    expected_token = os.environ.get("ADMIN_TOKEN")
    if request.args.get("token", "") != expected_token:
        return jsonify({"error": "Unauthorized"}), 401
    mtime = None
    try:
        mtime = background_manager.BG_PATH.stat().st_mtime
    except Exception:
        pass
    return jsonify({
        "running":     _bg_update_state["running"],
        "last_result": _bg_update_state["last_result"],
        "mtime":       mtime,
    })

@app.after_request
def set_security_headers(response):
    # Admin page uses inline <style>/<script> and is already gated by a secret
    # token — skip CSP so those blocks are not silently blocked by the browser.
    if request.path.startswith("/admin/"):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        return response

    # connect-src: allow direct browser fetch to geolocation-db.com so the
    # client's real IP is used (server-side proxy always resolves to server IP).
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "font-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com; "
        "style-src 'self' https://fonts.googleapis.com; "
        "script-src 'self'; "
        "connect-src 'self' https://geolocation-db.com"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Pre-load the latest ONNX model at worker startup to eliminate cold-start latency.
# Runs under Gunicorn (module import) and direct python app.py alike.
_initial_models = sorted(glob.glob("*.onnx"), reverse=True)
if _initial_models:
    logger.info("Pre-loading model: %s", _initial_models[0])
    get_model(_initial_models[0])

# Generate a fresh coffee-button tagline on each restart via Gemini.
_KOFI_FALLBACKS = [
    "☕ You beat the AI — fuel the dev!",
    "☕ The AI is sulking. Cheer it up with a coffee?",
    "☕ Victory tastes good. Coffee tastes better.",
    "☕ You won. The dev is still coding. Send help.",
    "☕ Outsmarted an AI — that deserves a coffee.",
    "☕ The AI demands a rematch. The dev demands espresso.",
    "☕ Neural networks don't run on air. Neither do devs.",
    "☕ You cracked the AI. Now crack open a coffee for the dev?",
]

def _pick_fallback():
    import hashlib, datetime
    seed = hashlib.md5(datetime.date.today().isoformat().encode(), usedforsecurity=False).digest()[0]
    return _KOFI_FALLBACKS[seed % len(_KOFI_FALLBACKS)]

_kofi_tagline = _pick_fallback()

def _gen_kofi_tagline():
    global _kofi_tagline
    if not gemini_client:
        return
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                "Output ONLY a single short witty line (max 10 words, no quotes, no preamble) "
                "inviting someone to buy the developer a coffee after beating a Connect 4 AI. "
                "Must include the ☕ emoji. Never say 'buy me a coffee'. "
                "Example output: You crushed it! Fuel the dev ☕"
            ),
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=15000)
            )
        )
        # Find the first line that contains ☕ to skip any preamble Gemini adds
        lines = [l.strip() for l in resp.text.splitlines() if '☕' in l and l.strip()]
        line = lines[0] if lines else resp.text.strip().splitlines()[0].strip()
        if line:
            _kofi_tagline = line
            logger.info("Kofi tagline: %s", _kofi_tagline)
    except Exception as exc:
        logger.warning("Kofi tagline generation failed (using fallback): %s", exc)

threading.Thread(target=_gen_kofi_tagline, daemon=True).start()

# Startup check for stale background image — reuses _bg_update_state so the
# admin "running" guard blocks concurrent manual triggers during boot.
if background_manager.is_background_stale():
    logger.info("Stale background detected, triggering update...")
    def _safe_bg_update():
        _bg_update_state["running"] = True
        _bg_update_state["last_result"] = None
        try:
            ok = background_manager.update_background()
            _bg_update_state["last_result"] = "ok" if ok else "error"
        except Exception as e:
            logger.error("Background update failed: %s", e)
            _bg_update_state["last_result"] = "error"
        finally:
            _bg_update_state["running"] = False
    threading.Thread(target=_safe_bg_update, daemon=True).start()

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    logger.info("Starting Connect 4 AI Server...")
    logger.info("Device: %s", device)
    app_host = os.environ.get("APP_HOST", "127.0.0.1")
    app_port = int(os.environ.get("APP_PORT", 5000))
    app_debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=app_debug, host=app_host, port=app_port)
