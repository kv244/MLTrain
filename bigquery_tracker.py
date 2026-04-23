"""
bigquery_tracker.py — Connect-4 player analytics via BigQuery

Table: <GCP_PROJECT_ID>.<BQ_DATASET>.<BQ_TABLE>  (defaults: connect4.player_stats)
Schema (CLUSTER BY ip_address for fast per-IP lookups):

    ip_address   STRING    visitor IP (primary key for MERGE)
    country      STRING    from client-side geo-IP lookup
    first_seen   TIMESTAMP first page visit
    last_seen    TIMESTAMP most recent activity
    total_visits INT64     page loads
    total_games  INT64     completed games
    player_wins  INT64
    ai_wins      INT64
    draws        INT64
    total_moves  INT64     cumulative moves across all games

Public API
----------
    init()                          — call once at app startup
    record_session(ip, country)     — upsert on page load (INSERT new / UPDATE returning)
    record_game(ip, winner, moves)  — update game counters at game end

All BQ operations are fire-and-forget (daemon threads); HTTP responses are never
blocked waiting for BigQuery.
"""

import json
import os
import threading
from google.cloud import bigquery

PROJECT_ID           = os.environ.get("GCP_PROJECT_ID")
DATASET              = os.environ.get("BQ_DATASET",             "connect4")
TABLE                = os.environ.get("BQ_TABLE",               "player_stats")
WIN_TABLE            = os.environ.get("BQ_WIN_TABLE",           "win_records")
TELEMETRY_TABLE      = os.environ.get("BQ_TELEMETRY_TABLE",     "move_telemetry")
HUMAN_GAMES_TABLE    = os.environ.get("BQ_HUMAN_GAMES_TABLE",   "human_games")

_client              = None
_enabled             = False
_table_ref           = None   # set in init()
_win_table_ref       = None   # set in init()
_telemetry_table_ref = None   # set in init()
_human_games_table_ref = None # set in init()


# ── Initialisation ────────────────────────────────────────────────────────────

def init():
    """Initialise the BigQuery client and ensure the table exists.
    Silent no-op if GCP_PROJECT_ID is not set (local dev)."""
    global _client, _enabled, _table_ref, _win_table_ref, _telemetry_table_ref, _human_games_table_ref
    if not PROJECT_ID:
        print("[BQTracker] GCP_PROJECT_ID not set — tracking disabled.")
        return
    try:
        _client                = bigquery.Client(project=PROJECT_ID)
        _table_ref             = f"{PROJECT_ID}.{DATASET}.{TABLE}"
        _win_table_ref         = f"{PROJECT_ID}.{DATASET}.{WIN_TABLE}"
        _telemetry_table_ref   = f"{PROJECT_ID}.{DATASET}.{TELEMETRY_TABLE}"
        _human_games_table_ref = f"{PROJECT_ID}.{DATASET}.{HUMAN_GAMES_TABLE}"
        _enabled               = True
        print(f"[BQTracker] Enabled → {_table_ref}")
        threading.Thread(target=_ensure_table,              daemon=True).start()
        threading.Thread(target=_ensure_win_table,          daemon=True).start()
        threading.Thread(target=_ensure_telemetry_table,    daemon=True).start()
        threading.Thread(target=_ensure_human_games_table,  daemon=True).start()
    except Exception as exc:
        print(f"[BQTracker] Init failed: {exc}")


_CREATE_DDL = """
CREATE TABLE IF NOT EXISTS `{table_ref}` (
    ip_address      STRING    NOT NULL,
    country         STRING,
    first_seen      TIMESTAMP,
    last_seen       TIMESTAMP,
    total_visits    INT64,
    total_games     INT64,
    player_wins     INT64,
    ai_wins         INT64,
    draws           INT64,
    total_moves     INT64,
    easy_games      INT64,
    medium_games    INT64,
    hard_games      INT64
)
CLUSTER BY ip_address
OPTIONS (description = 'Connect-4 AlphaZero — player session statistics');
"""

# Migrate existing tables that predate difficulty tracking
_ALTER_DDL = """
ALTER TABLE `{table_ref}`
ADD COLUMN IF NOT EXISTS easy_games   INT64,
ADD COLUMN IF NOT EXISTS medium_games INT64,
ADD COLUMN IF NOT EXISTS hard_games   INT64;
"""

def _ensure_table():
    try:
        _client.query(_CREATE_DDL.format(table_ref=_table_ref)).result()
        _client.query(_ALTER_DDL.format(table_ref=_table_ref)).result()
        print(f"[BQTracker] Table ready: {_table_ref}")
    except Exception as exc:
        print(f"[BQTracker] ensure_table failed: {exc}")


_CREATE_WIN_DDL = """
CREATE TABLE IF NOT EXISTS `{table_ref}` (
    recorded_at  TIMESTAMP NOT NULL,
    name         STRING    NOT NULL,
    difficulty   STRING,
    simulations  INT64,
    moves        INT64,
    ip_address   STRING
)
OPTIONS (description = 'Connect-4 AlphaZero — player win hall of fame');
"""

def _ensure_win_table():
    try:
        _client.query(_CREATE_WIN_DDL.format(table_ref=_win_table_ref)).result()
        print(f"[BQTracker] Win table ready: {_win_table_ref}")
    except Exception as exc:
        print(f"[BQTracker] ensure_win_table failed: {exc}")


_CREATE_TELEMETRY_DDL = """
CREATE TABLE IF NOT EXISTS `{table_ref}` (
    recorded_at          TIMESTAMP NOT NULL,
    model                STRING,
    simulations          INT64,
    inference_time_ms    FLOAT64
)
OPTIONS (description = 'Connect-4 AlphaZero — per-move inference latency');
"""

def _ensure_telemetry_table():
    try:
        _client.query(_CREATE_TELEMETRY_DDL.format(table_ref=_telemetry_table_ref)).result()
        print(f"[BQTracker] Telemetry table ready: {_telemetry_table_ref}")
    except Exception as exc:
        print(f"[BQTracker] ensure_telemetry_table failed: {exc}")


_CREATE_HUMAN_GAMES_DDL = """
CREATE TABLE IF NOT EXISTS `{table_ref}` (
    recorded_at    TIMESTAMP NOT NULL,
    winner         STRING    NOT NULL,
    move_sequence  STRING,
    num_moves      INT64,
    difficulty     STRING,
    ip_address     STRING,
    human_player   INT64
)
OPTIONS (description = 'Connect-4 AlphaZero — human win game trajectories for retraining');
"""

def _ensure_human_games_table():
    try:
        _client.query(_CREATE_HUMAN_GAMES_DDL.format(table_ref=_human_games_table_ref)).result()
        print(f"[BQTracker] Human games table ready: {_human_games_table_ref}")
    except Exception as exc:
        print(f"[BQTracker] ensure_human_games_table failed: {exc}")


# ── SQL templates ─────────────────────────────────────────────────────────────

# MERGE on ip_address: INSERT new visitors, UPDATE returning ones.
_SESSION_MERGE = """
MERGE `{table_ref}` AS T
USING (SELECT @ip AS ip_address, @country AS country,
              CURRENT_TIMESTAMP() AS ts) AS S
ON T.ip_address = S.ip_address
WHEN MATCHED THEN UPDATE SET
    last_seen    = S.ts,
    country      = COALESCE(NULLIF(S.country, ''), T.country),
    total_visits = T.total_visits + 1
WHEN NOT MATCHED THEN INSERT
    (ip_address, country, first_seen, last_seen,
     total_visits, total_games, player_wins, ai_wins, draws, total_moves)
VALUES
    (S.ip_address, S.country, S.ts, S.ts, 1, 0, 0, 0, 0, 0)
"""

_GAME_MERGE = """
MERGE `{table_ref}` AS T
USING (SELECT @ip AS ip_address, CURRENT_TIMESTAMP() AS ts) AS S
ON T.ip_address = S.ip_address
WHEN MATCHED THEN UPDATE SET
    last_seen    = S.ts,
    total_games  = T.total_games  + 1,
    player_wins  = T.player_wins  + @player_win,
    ai_wins      = T.ai_wins      + @ai_win,
    draws        = T.draws        + @is_draw,
    total_moves  = T.total_moves  + @moves,
    easy_games   = COALESCE(T.easy_games,   0) + @easy,
    medium_games = COALESCE(T.medium_games, 0) + @medium,
    hard_games   = COALESCE(T.hard_games,   0) + @hard
WHEN NOT MATCHED THEN INSERT
    (ip_address, first_seen, last_seen,
     total_visits, total_games, player_wins, ai_wins, draws, total_moves,
     easy_games, medium_games, hard_games)
VALUES
    (S.ip_address, S.ts, S.ts,
     0, 1, @player_win, @ai_win, @is_draw, @moves,
     @easy, @medium, @hard)
"""


# ── Internal query runner ─────────────────────────────────────────────────────

def _run(sql_template, params):
    """Execute a parameterised query. Called from a daemon thread."""
    if not _enabled:
        return
    try:
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        _client.query(
            sql_template.format(table_ref=_table_ref),
            job_config=cfg
        ).result(timeout=15)  # prevent threads hanging indefinitely on slow/unreachable BQ
    except Exception as exc:
        print(f"[BQTracker] Query error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def record_session(ip_address, country=None):
    """Upsert a page-load visit.
    New IP  → INSERT row with visit count = 1 and zero game stats.
    Known IP → UPDATE last_seen and increment total_visits."""
    params = [
        bigquery.ScalarQueryParameter("ip",      "STRING", ip_address or "unknown"),
        bigquery.ScalarQueryParameter("country", "STRING", country or ""),
    ]
    threading.Thread(
        target=_run, args=(_SESSION_MERGE, params), daemon=True
    ).start()


def record_win(ip_address, name, difficulty, simulations, moves):
    """Insert a single win record into the win_records hall-of-fame table."""
    if not _enabled:
        return
    params = [
        bigquery.ScalarQueryParameter("ip",          "STRING", ip_address or "unknown"),
        bigquery.ScalarQueryParameter("name",        "STRING", name),
        bigquery.ScalarQueryParameter("difficulty",  "STRING", difficulty),
        bigquery.ScalarQueryParameter("simulations", "INT64",  simulations),
        bigquery.ScalarQueryParameter("moves",       "INT64",  moves),
    ]
    sql = """
INSERT INTO `{table_ref}` (recorded_at, name, difficulty, simulations, moves, ip_address)
VALUES (CURRENT_TIMESTAMP(), @name, @difficulty, @simulations, @moves, @ip)
""".format(table_ref=_win_table_ref)
    threading.Thread(target=_run_raw, args=(sql, params), daemon=True).start()


def _run_raw(sql, params):
    """Execute a raw parameterised query (no .format substitution)."""
    if not _enabled:
        return
    try:
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        _client.query(sql, job_config=cfg).result(timeout=15)
    except Exception as exc:
        print(f"[BQTracker] Query error: {exc}")


def record_telemetry(model, simulations, inference_time_seconds):
    """Insert one row of inference latency into move_telemetry. Fire-and-forget."""
    if not _enabled:
        return
    sql = """
INSERT INTO `{table_ref}` (recorded_at, model, simulations, inference_time_ms)
VALUES (CURRENT_TIMESTAMP(), @model, @simulations, @inference_time_ms)
""".format(table_ref=_telemetry_table_ref)
    params = [
        bigquery.ScalarQueryParameter("model",             "STRING",  model),
        bigquery.ScalarQueryParameter("simulations",       "INT64",   simulations),
        bigquery.ScalarQueryParameter("inference_time_ms", "FLOAT64", round(inference_time_seconds * 1000, 3)),
    ]
    threading.Thread(target=_run_raw, args=(sql, params), daemon=True).start()


def record_human_game(ip_address, winner, move_sequence, difficulty="hard", human_player=1):
    """Store the full move trajectory of a game the human won (or drew).
    move_sequence: list of column ints in play order (alternating players).
    Enables offline retraining and opening-book construction."""
    if not _enabled or not _human_games_table_ref:
        return
    seq_json = json.dumps([int(m) for m in move_sequence]) if move_sequence else "[]"
    sql = """
INSERT INTO `{table_ref}` (recorded_at, winner, move_sequence, num_moves, difficulty, ip_address, human_player)
VALUES (CURRENT_TIMESTAMP(), @winner, @move_sequence, @num_moves, @difficulty, @ip, @human_player)
""".format(table_ref=_human_games_table_ref)
    params = [
        bigquery.ScalarQueryParameter("winner",        "STRING", winner),
        bigquery.ScalarQueryParameter("move_sequence", "STRING", seq_json),
        bigquery.ScalarQueryParameter("num_moves",     "INT64",  len(move_sequence) if move_sequence else 0),
        bigquery.ScalarQueryParameter("difficulty",    "STRING", difficulty or "hard"),
        bigquery.ScalarQueryParameter("ip",            "STRING", ip_address or "unknown"),
        bigquery.ScalarQueryParameter("human_player",  "INT64",  int(human_player)),
    ]
    threading.Thread(target=_run_raw, args=(sql, params), daemon=True).start()


def get_human_games(winner_filter="human", limit=1000):
    """Return game records as a list of dicts for training or book-building.
    Each dict: {move_sequence: [int, ...], winner: str, human_player: int}
    Blocking call — intended for offline training scripts, not request handlers."""
    if not _enabled or not _human_games_table_ref:
        return []
    try:
        where = "WHERE move_sequence IS NOT NULL AND num_moves > 6"
        if winner_filter:
            if winner_filter not in ("human", "ai", "draw"):
                raise ValueError(f"Invalid winner_filter: {winner_filter!r}")
            where += f" AND winner = '{winner_filter}'"
        rows = list(_client.query(f"""
            SELECT move_sequence, winner, human_player
            FROM `{_human_games_table_ref}`
            {where}
            ORDER BY recorded_at DESC
            LIMIT {int(limit)}
        """).result(timeout=30))
        return [
            {
                "move_sequence": json.loads(r.move_sequence),
                "winner":        r.winner,
                "human_player":  int(r.human_player) if r.human_player is not None else 1,
            }
            for r in rows if r.move_sequence
        ]
    except Exception as exc:
        print(f"[BQTracker] get_human_games failed: {exc}")
        return []


def record_game(ip_address, winner, moves, difficulty="hard"):
    """Increment game outcome counters for this IP after a completed game.
    winner:     'human' | 'ai' | 'draw'
    moves:      total half-moves in the game
    difficulty: 'easy' | 'medium' | 'hard'"""
    diff = difficulty.lower() if difficulty else "hard"
    params = [
        bigquery.ScalarQueryParameter("ip",         "STRING", ip_address or "unknown"),
        bigquery.ScalarQueryParameter("player_win", "INT64",  1 if winner == "human" else 0),
        bigquery.ScalarQueryParameter("ai_win",     "INT64",  1 if winner == "ai"    else 0),
        bigquery.ScalarQueryParameter("is_draw",    "INT64",  1 if winner == "draw"  else 0),
        bigquery.ScalarQueryParameter("moves",      "INT64",  max(0, int(moves))),
        bigquery.ScalarQueryParameter("easy",       "INT64",  1 if diff == "easy"   else 0),
        bigquery.ScalarQueryParameter("medium",     "INT64",  1 if diff == "medium" else 0),
        bigquery.ScalarQueryParameter("hard",       "INT64",  1 if diff == "hard"   else 0),
    ]
    threading.Thread(
        target=_run, args=(_GAME_MERGE, params), daemon=True
    ).start()
