"""
build_opening_book.py — Build opening_book.json from human-win trajectories stored in BigQuery.

For each position where the AI moved during a game the human eventually won, we record
which column the AI played.  Positions where the same AI move appears >= min_count times
across distinct human-win games are written to the book as "avoid_move" entries.

At runtime app.py loads opening_book.json and, when MCTS would pick a known-bad move
in the first 12 plies, silently redirects to the second-best MCTS candidate instead.

Limitation: every AI move in a losing game is recorded, including early moves that were
actually fine. The min_count threshold reduces noise, but positions the AI passes through
in most games (e.g. the center-first opening) will accumulate counts just by appearing in
many human wins. For a more precise signal, consider restricting to the last 2–4 AI plies
before the human's winning move (where the actual mistake is most likely to live).

Usage:
    python build_opening_book.py              # defaults: min_count=3, limit=5000
    python build_opening_book.py --min-count 5 --limit 10000
"""

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np


def board_hash(board_arr: np.ndarray) -> str:
    return hashlib.sha256(board_arr.astype(np.int8).tobytes()).hexdigest()[:16]


def build_book(min_count: int = 3, limit: int = 5000) -> None:
    import bigquery_tracker
    from mcts import Connect4

    bigquery_tracker.init()
    if not bigquery_tracker._enabled:
        print("BigQuery not enabled — set GCP_PROJECT_ID in environment.")
        return

    print(f"Querying up to {limit} human-win games from BigQuery...")
    games = bigquery_tracker.get_human_games(winner_filter="human", limit=limit)
    print(f"Retrieved {len(games)} games.")

    # position_losses[board_hash][ai_col] = number of human wins where the AI played
    # ai_col from that position (and eventually lost the game).
    position_losses: dict = defaultdict(lambda: defaultdict(int))

    for g in games:
        moves       = g["move_sequence"]
        human_pl    = g["human_player"]
        game        = Connect4()

        for col in moves:
            if not isinstance(col, int) or col < 0 or col > 6:
                break

            if game.current_player != human_pl:
                # It's the AI's turn — record the position and the move it made
                h = board_hash(game.board)
                position_losses[h][col] += 1

            result = game.play(col)
            if result is None:
                break  # column was full — malformed record
            r, c = result
            if game.check_win(r, c):
                break

    book: dict = {}
    for h, move_counts in position_losses.items():
        worst_move  = max(move_counts, key=move_counts.get)
        worst_count = move_counts[worst_move]
        if worst_count >= min_count:
            book[h] = {"avoid_move": worst_move, "count": worst_count}

    print(f"Book contains {len(book)} positions (min_count={min_count}).")
    with open("opening_book.json", "w") as f:
        json.dump(book, f, indent=2)
    print("Saved opening_book.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build opening_book.json from BigQuery human games.")
    parser.add_argument("--min-count", type=int, default=3,    help="Minimum AI-loss count to include a position")
    parser.add_argument("--limit",     type=int, default=5000, help="Max games to query from BigQuery")
    args = parser.parse_args()
    build_book(min_count=args.min_count, limit=args.limit)
