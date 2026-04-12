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

import os
import threading
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DATASET    = os.environ.get("BQ_DATASET", "connect4")
TABLE      = os.environ.get("BQ_TABLE",   "player_stats")

_client    = None
_enabled   = False
_table_ref = None   # set in init()


# ── Initialisation ────────────────────────────────────────────────────────────

def init():
    """Initialise the BigQuery client and ensure the table exists.
    Silent no-op if GCP_PROJECT_ID is not set (local dev)."""
    global _client, _enabled, _table_ref
    if not PROJECT_ID:
        print("[BQTracker] GCP_PROJECT_ID not set — tracking disabled.")
        return
    try:
        _client    = bigquery.Client(project=PROJECT_ID)
        _table_ref = f"{PROJECT_ID}.{DATASET}.{TABLE}"
        _enabled   = True
        print(f"[BQTracker] Enabled → {_table_ref}")
        # Create the table in a background thread; don't block app startup
        threading.Thread(target=_ensure_table, daemon=True).start()
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
USING (SELECT @ip AS ip_address) AS S
ON T.ip_address = S.ip_address
WHEN MATCHED THEN UPDATE SET
    last_seen    = CURRENT_TIMESTAMP(),
    total_games  = T.total_games  + 1,
    player_wins  = T.player_wins  + @player_win,
    ai_wins      = T.ai_wins      + @ai_win,
    draws        = T.draws        + @is_draw,
    total_moves  = T.total_moves  + @moves,
    easy_games   = COALESCE(T.easy_games,   0) + @easy,
    medium_games = COALESCE(T.medium_games, 0) + @medium,
    hard_games   = COALESCE(T.hard_games,   0) + @hard
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
        ).result()
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
