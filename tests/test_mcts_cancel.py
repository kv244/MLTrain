"""MCTS cancellation test (needs torch; skipped where it is not installed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
    from mcts import Connect4, SearchCancelled, run_mcts_simulations
except ImportError:  # pragma: no cover
    torch = None


class _UniformModel:
    """Stand-in network: uniform policy, neutral value."""
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.zeros(1, 7), torch.zeros(1, 1)


@unittest.skipIf(torch is None, "torch not installed")
class MctsCancelTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_runs_to_completion_without_cancel_check(self):
        model = _UniformModel()
        run_mcts_simulations(Connect4(), model, self.device, num_sims=40, add_dirichlet_noise=False)
        self.assertEqual(model.calls, 40)

    def test_cancel_stops_the_search_early(self):
        model = _UniformModel()
        polls = {"n": 0}

        def cancel_check():
            polls["n"] += 1
            return polls["n"] >= 3      # cancel on the third poll

        with self.assertRaises(SearchCancelled):
            run_mcts_simulations(Connect4(), model, self.device, num_sims=2000,
                                 add_dirichlet_noise=False, cancel_check=cancel_check)
        self.assertLess(model.calls, 100)

    def test_cancel_check_false_does_not_change_result(self):
        a = run_mcts_simulations(Connect4(), _UniformModel(), self.device, num_sims=60,
                                 add_dirichlet_noise=False, temperature=1.0)
        b = run_mcts_simulations(Connect4(), _UniformModel(), self.device, num_sims=60,
                                 add_dirichlet_noise=False, temperature=1.0,
                                 cancel_check=lambda: False)
        self.assertEqual(list(a), list(b))


if __name__ == "__main__":
    unittest.main()
