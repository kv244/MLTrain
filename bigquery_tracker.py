"""
bigquery_tracker.py — Connect-4 player analytics and telemetry via Google BigQuery.

This module provides a fire-and-forget interface for logging player sessions, 
game outcomes, AI inference telemetry, and winning game trajectories.

Tables:
  - player_stats:    Aggregated per-IP session and game statistics.
  - win_records:     Hall of fame for human players who beat the AI.
  - move_telemetry:  Performance metrics (latency) for AI model inference.
  - human_games:     Detailed move sequences for retraining and analysis.

Implementation Notes:
  - All write operations are executed in daemon threads to prevent blocking 
    the main application (Flask/Gunicorn) response cycle.
  - MERGE statements are used for idempotent 'upserts' on player statistics.
  - Tables are clustered by ip_address to optimize frequent per-user lookups.
"""

import json
import logging
import os
import threading
from google.cloud import bigquery

logger = logging.getLogger(__name__)

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
        logger.warning("GCP_PROJECT_ID not set — tracking disabled.")
        return
    try:
        _client                = bigquery.Client(project=PROJECT_ID)
        _table_ref             = f"{PROJECT_ID}.{DATASET}.{TABLE}"
        _win_table_ref         = f"{PROJECT_ID}.{DATASET}.{WIN_TABLE}"
        _telemetry_table_ref   = f"{PROJECT_ID}.{DATASET}.{TELEMETRY_TABLE}"
        _human_games_table_ref = f"{PROJECT_ID}.{DATASET}.{HUMAN_GAMES_TABLE}"
        _enabled               = True
        logger.info("Enabled → %s", _table_ref)
        threading.Thread(target=_ensure_table,              daemon=True).start()
        threading.Thread(target=_ensure_win_table,          daemon=True).start()
        threading.Thread(target=_ensure_telemetry_table,    daemon=True).start()
        threading.Thread(target=_ensure_human_games_table,  daemon=True).start()
    except Exception as exc:
        logger.error("Init failed: %s", exc)


# Primary analytics table for user retention and overall win/loss ratios.
# Clustered by ip_address to ensure fast filtering for the dashboard.
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
        logger.info("Table ready: %s", _table_ref)
    except Exception as exc:
        logger.error("ensure_table failed: %s", exc)


# Hall of Fame table for players who successfully defeat the AlphaZero AI.
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
        logger.info("Win table ready: %s", _win_table_ref)
    except Exception as exc:
        logger.error("ensure_win_table failed: %s", exc)


# Performance monitoring table to track inference latency across different models.
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
        logger.info("Telemetry table ready: %s", _telemetry_table_ref)
    except Exception as exc:
        logger.error("ensure_telemetry_table failed: %s", exc)


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
        logger.info("Human games table ready: %s", _human_games_table_ref)
    except Exception as exc:
        logger.error("ensure_human_games_table failed: %s", exc)


# ── SQL templates ─────────────────────────────────────────────────────────────

# Atomic Upsert for visitor sessions. 
# If IP exists: Update last seen and increment visit count.
# If IP is new: Insert new record with initial counters.
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

# Atomic update for game statistics.
# Increments win/loss/draw counters and difficulty-specific game counts.
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
    """
    Executes a parameterized SQL query against the primary analytics table.
    
    Args:
        sql_template (str): SQL string with '{table_ref}' placeholder.
        params (list): List of bigquery.ScalarQueryParameter objects.
        
    Note:
        Runs in a background thread with a 15s timeout to avoid blocking 
        frontend requests if BigQuery is slow or throttled.
    """
    if not _enabled:
        return
    try:
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        _client.query(
            sql_template.format(table_ref=_table_ref),
            job_config=cfg
        ).result(timeout=15)
    except Exception as exc:
        logger.error("Query error: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def record_session(ip_address, country=None):
    """
    Records a user's page visit. Increments visit count if the IP is known,
    otherwise creates a new entry.
    
    Args:
        ip_address (str): The visitor's IP address.
        country (str, optional): The visitor's country code from geo-IP.
    """
    params = [
        bigquery.ScalarQueryParameter("ip",      "STRING", ip_address or "unknown"),
        bigquery.ScalarQueryParameter("country", "STRING", country or ""),
    ]
    threading.Thread(
        target=_run, args=(_SESSION_MERGE, params), daemon=True
    ).start()


def record_win(ip_address, name, difficulty, simulations, moves):
    """
    Logs a human victory into the hall-of-fame (win_records) table.
    
    Args:
        ip_address (str): The winner's IP address.
        name (str): The display name of the winner.
        difficulty (str): The AI difficulty level (easy/medium/hard).
        simulations (int): Number of MCTS simulations used by the AI.
        moves (int): Total moves in the game.
    """
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
    """
    Executes a raw parameterized SQL query without string formatting.
    Used for tables other than the primary player_stats table.
    """
    if not _enabled:
        return
    try:
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        _client.query(sql, job_config=cfg).result(timeout=15)
    except Exception as exc:
        logger.error("Query error: %s", exc)


def record_telemetry(model, simulations, inference_time_seconds):
    """
    Logs AI inference performance metrics.
    
    Args:
        model (str): Name or path of the model used.
        simulations (int): Number of simulations performed for the move.
        inference_time_seconds (float): Time taken for inference in seconds.
    """
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
    """
    Persists the full move sequence of a game for future model training 
    or opening book generation.
    
    Args:
        ip_address (str): The player's IP address.
        winner (str): 'human', 'ai', or 'draw'.
        move_sequence (list): List of column indices representing the moves.
        difficulty (str): The AI difficulty level.
        human_player (int): Whether the human was player 1 or 2.
    """
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
    """
    Retrieves recorded game trajectories for offline analysis or training.
    
    Args:
        winner_filter (str): Filter by winner ('human', 'ai', 'draw').
        limit (int): Maximum number of records to return.
        
    Returns:
        list[dict]: A list of game records with move_sequence and metadata.
        
    Warning:
        This is a synchronous, blocking call. Use only in offline scripts.
    """
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
        logger.error("get_human_games failed: %s", exc)
        return []


def record_game(ip_address, winner, moves, difficulty="hard"):
    """
    Updates the aggregated game statistics for a specific player (IP).
    
    Args:
        ip_address (str): The player's IP address.
        winner (str): 'human', 'ai', or 'draw'.
        moves (int): Total half-moves in the game.
        difficulty (str): The AI difficulty level.
    """
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
