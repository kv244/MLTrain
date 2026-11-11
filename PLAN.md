# c4star.com Improvement Plan

Status: proposed (2026-11-11). Items map to the "Planned" list in the README roadmap.

## Phase 1: Server load and responsiveness (highest value)
| # | Item | Change | Done when |
|---|------|--------|-----------|
| 1 | Auto-hint cap (DONE) | `static/script.js` / `app.py`: limit easy-mode hint to ~50 sims, or only compute on "Get Hint" | Easy mode makes one heavy request per turn, not two |
| 2 | Abort stale work | Client sends a request ID; server skips/aborts `/api/assess` and `/api/move` for superseded IDs (after undo/restart) | Restart mid-rating leaves no queued CPU work |
| 3 | Concurrency | Test `--workers 2` vs `--threads 2` on the VM; check RAM first | A rating no longer delays a move; memory stays under limit |
| 4 | Position cache | LRU cache keyed by (board hash, player, sims bucket) for assess and move probs | Repeat openings answer in <50 ms |

## Phase 2: UI polish
| # | Item | Done when |
|---|------|-----------|
| 5 | Collapse empty ratings gap on phones until a rating exists | No blank block above the board on a 390px viewport |
| 6 | Scroll to board after "AI Plays First" and restart | Board fully visible after any start path |
| 7 | "Rating..." indicator | Spinner shows while a rating is pending, clears on result or stale drop |

## Phase 3: Ops and deploy
| # | Item | Done when |
|---|------|-----------|
| 8 | Disk monitoring | Cron log confirms the Nov 15 run; deploy summary prints free MB |
| 9 | Model health check | Deploy fails (and rolls back) if `/api/models` is empty |
| 10 | Version | Rename README entry v2.2.2 to v2.3.0 if the ratings toggle counts as a feature release (decision pending) |

## Phase 4: Testing
| # | Item | Done when |
|---|------|-----------|
| 11 | CI smoke test (DONE) | GitHub Action runs headless Chromium at phone size, taps Start, asserts the board is in view; runs on every PR |

## Suggested order
1. #1 and #11 (small, high payoff, #11 guards against the mobile bug returning)
2. #2, #4, #3 (measure with the real ONNX model before and after)
3. #5 to #7, then #8 to #10

## Risks
- Workers/threads: memory per model copy; benchmark on the real VM before committing.
- Abort support: gunicorn sync workers cannot cancel a running request, so the server must check the ID before starting, and MCTS needs a periodic cancel check.
- Caching: key must include sims bucket and temperature so cached results match request strength.
