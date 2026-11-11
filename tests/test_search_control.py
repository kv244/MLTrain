"""Unit tests for request cancellation and the search cache (pure Python)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_control import RequestRegistry, SearchCache, valid_client_id  # noqa: E402


class RequestRegistryTests(unittest.TestCase):
    def test_newest_epoch_wins(self):
        r = RequestRegistry()
        self.assertFalse(r.register("client-aaaa", "assess", 1))
        self.assertFalse(r.register("client-aaaa", "assess", 2))
        self.assertTrue(r.is_stale("client-aaaa", "assess", 1))
        self.assertFalse(r.is_stale("client-aaaa", "assess", 2))

    def test_older_request_arriving_late_is_stale(self):
        r = RequestRegistry()
        r.register("client-aaaa", "move", 5)
        self.assertTrue(r.register("client-aaaa", "move", 3))

    def test_kinds_are_independent(self):
        r = RequestRegistry()
        r.register("client-aaaa", "hint", 7)
        self.assertFalse(r.register("client-aaaa", "move", 1))
        self.assertFalse(r.is_stale("client-aaaa", "move", 1))

    def test_clients_are_independent(self):
        r = RequestRegistry()
        r.register("client-aaaa", "move", 9)
        self.assertFalse(r.register("client-bbbb", "move", 1))

    def test_cancel_floor(self):
        r = RequestRegistry()
        r.register("client-aaaa", "assess", 4)
        r.cancel("client-aaaa", "assess", 4)
        self.assertTrue(r.is_stale("client-aaaa", "assess", 4))
        self.assertFalse(r.register("client-aaaa", "assess", 5))   # later requests still run

    def test_cancel_never_lowers_floor(self):
        r = RequestRegistry()
        r.cancel("client-aaaa", "hint", 6)
        r.cancel("client-aaaa", "hint", 2)
        self.assertTrue(r.is_stale("client-aaaa", "hint", 6))

    def test_unknown_request_is_not_stale(self):
        self.assertFalse(RequestRegistry().is_stale("client-aaaa", "move", 1))

    def test_registry_is_bounded(self):
        r = RequestRegistry(max_entries=10)
        for i in range(100):
            r.register(f"client-{i:04d}", "move", 1)
        self.assertLessEqual(len(r._state), 10)

    def test_client_id_validation(self):
        self.assertTrue(valid_client_id("3f2b8c1e-0a7d-4c55-9a53-0d1b8c0a77aa"))
        self.assertFalse(valid_client_id("short"))
        self.assertFalse(valid_client_id("has spaces in it!!"))
        self.assertFalse(valid_client_id(None))
        self.assertFalse(valid_client_id("x" * 65))


class SearchCacheTests(unittest.TestCase):
    def test_hit_and_miss_counters(self):
        c = SearchCache()
        self.assertIsNone(c.get("k"))
        c.put("k", [1])
        self.assertEqual(c.get("k"), [1])
        self.assertEqual(c.stats(), {"size": 1, "hits": 1, "misses": 1})

    def test_lru_eviction(self):
        c = SearchCache(max_entries=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")           # a is now most recent
        c.put("c", 3)        # evicts b
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("c"), 3)


if __name__ == "__main__":
    unittest.main()
