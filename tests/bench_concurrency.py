"""Compare gunicorn configurations under overlapping requests (run against a live server).

Usage: python tests/bench_concurrency.py http://127.0.0.1:5001 [model.onnx]

Scenarios (median of 3 runs, seconds):
  move alone            : one AI move request (baseline latency)
  move behind rating    : an old, long rating is still running when the AI move is requested
  move + cancel         : same, but the client cancels the rating first (what the page now does)
  4 moves sequential    : four requests one after another
  4 moves concurrent    : four requests at once (wall time)
Set BENCH_ONLY=behind,cancel to run just those scenarios (keys: alone, behind, cancel, seq, conc)
and BENCH_RUNS=n to change the repeat count.
Not a unit test: it needs a running server with a model, and takes about a minute.
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5001"
RUNS = int(os.environ.get("BENCH_RUNS", "3"))
_counter = [0]


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        data = json.load(r)
    return data, time.time() - t


def model_name():
    if len(sys.argv) > 2:
        return sys.argv[2]
    with urllib.request.urlopen(BASE + "/api/models", timeout=30) as r:  # nosec B310
        return json.load(r)["models"][0]


def position(seed, plies=8):
    """A reproducible mid-game position (avoids the result cache between runs)."""
    rnd = random.Random(seed)
    board = [[0] * 7 for _ in range(6)]
    player = 1
    for _ in range(plies):
        cols = [c for c in range(7) if board[0][c] == 0]
        c = rnd.choice(cols)
        r = max(i for i in range(6) if board[i][c] == 0)
        board[r][c] = player
        player = -player
    return board, player


def fresh_move_body(model, sims, **extra):
    _counter[0] += 1
    board, player = position(_counter[0] * 7919)
    return dict({"model": model, "board": board, "current_player": player,
                 "simulations": sims, "difficulty": "hard"}, **extra)


def rating_body(model, sims, **extra):
    _counter[0] += 1
    board, player = position(_counter[0] * 104729)
    return dict({"model": model, "board": board, "move": 3, "current_player": player,
                 "simulations": sims}, **extra)


def med(fn):
    return statistics.median(fn() for _ in range(RUNS))


def s_alone(model):
    return post("/api/move", fresh_move_body(model, 800))[1]


def _move_behind_rating(model, cancel):
    cid = f"bench-client-{time.time_ns()}"
    old = {}
    th = threading.Thread(target=lambda: old.update(r=post("/api/assess", rating_body(model, 1500, client_id=cid, epoch=1))))
    th.start()
    time.sleep(0.3)  # the rating is running
    t0 = time.time()  # what the player waits: cancel (if any) + the AI move
    if cancel:
        post("/api/cancel", {"client_id": cid, "upto": {"assess": 1}})
    post("/api/move", fresh_move_body(model, 800, client_id=cid, kind="move", epoch=1))
    t = time.time() - t0
    th.join()
    return t


def s_sequential(model):
    t = time.time()
    for _ in range(4):
        post("/api/move", fresh_move_body(model, 400))
    return time.time() - t


def s_concurrent(model):
    bodies = [fresh_move_body(model, 400) for _ in range(4)]
    t = time.time()
    ths = [threading.Thread(target=post, args=("/api/move", b)) for b in bodies]
    [x.start() for x in ths]
    [x.join() for x in ths]
    return time.time() - t


if __name__ == "__main__":
    m = model_name()
    post("/api/move", fresh_move_body(m, 100))  # warm-up
    scenarios = {
        "alone": ("move alone", lambda: s_alone(m)),
        "behind": ("move behind rating", lambda: _move_behind_rating(m, False)),
        "cancel": ("move + cancel", lambda: _move_behind_rating(m, True)),
        "seq": ("4 moves sequential", lambda: s_sequential(m)),
        "conc": ("4 moves concurrent", lambda: s_concurrent(m)),
    }
    only = [k for k in os.environ.get("BENCH_ONLY", "").split(",") if k] or list(scenarios)
    results = {scenarios[k][0]: med(scenarios[k][1]) for k in only}
    for k, v in results.items():
        print(f"{k:22s} {v:6.2f}s")
