"""Request cancellation and search-result caching, shared by the API endpoints.

Pure Python (no torch/Flask) so it can be unit-tested on its own.

Cancellation model
------------------
The browser tags every /api/move and /api/assess call with a random ``client_id``
and a per-kind, increasing ``epoch`` (kinds: "move", "hint", "assess"). A request
is *stale* when
  * a newer request of the same kind has been seen for that client (newest wins), or
  * the client explicitly cancelled it via /api/cancel (restart, undo, next move).
Stale requests return immediately, and a running MCTS search polls
``RequestRegistry.is_stale`` and stops early.

Note: a running search can only notice a newer request if the server handles
requests concurrently, i.e. gunicorn runs with ``--threads`` > 1.
"""
import re
import threading
from collections import OrderedDict

KINDS = ("move", "hint", "assess")
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def valid_client_id(value) -> bool:
    return isinstance(value, str) and bool(_CLIENT_ID_RE.match(value))


class RequestRegistry:
    """Tracks the newest epoch and the cancel floor per (client_id, kind)."""

    def __init__(self, max_entries: int = 4000):
        self._lock = threading.Lock()
        self._state = OrderedDict()  # (client_id, kind) -> [latest, floor]
        self._max = max_entries

    def _entry(self, client_id, kind):
        key = (client_id, kind)
        entry = self._state.get(key)
        if entry is None:
            entry = [0, 0]
            self._state[key] = entry
            if len(self._state) > self._max:
                self._state.popitem(last=False)
        else:
            self._state.move_to_end(key)
        return entry

    def register(self, client_id, kind, epoch) -> bool:
        """Record a new request. Returns True if it is already stale."""
        with self._lock:
            entry = self._entry(client_id, kind)
            if epoch > entry[0]:
                entry[0] = epoch
            return epoch < entry[0] or epoch <= entry[1]

    def cancel(self, client_id, kind, upto) -> None:
        """Cancel every request of this kind with epoch <= upto."""
        with self._lock:
            entry = self._entry(client_id, kind)
            if upto > entry[1]:
                entry[1] = upto

    def is_stale(self, client_id, kind, epoch) -> bool:
        with self._lock:
            entry = self._state.get((client_id, kind))
            if entry is None:
                return False
            return epoch < entry[0] or epoch <= entry[1]


class SearchCache:
    """Thread-safe LRU cache for search results (the engine is deterministic
    without Dirichlet noise, so a position + strength always gives the same result)."""

    def __init__(self, max_entries: int = 4096):
        self._lock = threading.Lock()
        self._data = OrderedDict()
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "hits": self.hits, "misses": self.misses}
