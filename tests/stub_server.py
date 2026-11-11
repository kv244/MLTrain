"""Minimal stub server for UI smoke tests.

Serves the repo's real template and static files with fake, instant APIs so the
frontend can be tested without a model. Usage: python tests/stub_server.py <port>
"""
import os
import sys
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"),
            static_folder=os.path.join(ROOT, "static"))
SIMS = []  # simulations value of each /api/move request


@app.route("/")
def index():
    return render_template("index.html", version="smoke", kofi_tagline="tip")


@app.route("/api/models")
def api_models():
    return jsonify({"models": ["model.onnx"]})


@app.route("/api/info")
def api_info():
    return jsonify({"device": "CPU"})


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    SIMS.append(data.get("simulations"))
    cols = [c for c in range(7) if data["board"][0][c] == 0]
    return jsonify({"move": cols[0], "probs": [1 / 7] * 7, "winning_cells": []})


@app.route("/api/assess", methods=["POST"])
def api_assess():
    return jsonify({"score": 4, "comment": "Strong", "ai_quote": "ok", "best_move": 3,
                    "probs": [0.1] * 7, "winning_cells": []})


@app.route("/api/move_sims")
def api_move_sims():
    return jsonify(SIMS)


def _empty():
    return jsonify({})


for _p in ("/api/welcome_strings", "/api/geoip", "/api/stats", "/api/recent_winner",
           "/api/session", "/api/game_end", "/api/game_summary"):
    app.add_url_rule(_p, _p, _empty, methods=["GET", "POST"])

if __name__ == "__main__":
    app.run(port=int(sys.argv[1]), threaded=True)
