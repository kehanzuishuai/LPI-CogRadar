"""v3 只验证条件化表示和 test-v5 封存边界；不训练、不读取 test-v5。"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

import torch

from rl_resource.policy import ActorCritic, PolicyConfig


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "preference_ppo_v3",
    os.path.join(ROOT, "tools", "run_preference_ppo_v3.py"),
)
assert SPEC is not None and SPEC.loader is not None
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)


class TestPreferencePpoV3(unittest.TestCase):
    def test_film_keeps_state_and_preference_branches_and_changes_logits(self) -> None:
        config = PolicyConfig(obs_dim=109, state_obs_dim=104, preference_dim=5,
                              hidden_sizes=(128, 128), conditioning="film")
        policy = ActorCritic(config)
        first = torch.zeros((1, 109)); first[0, 104] = 1.0
        second = torch.zeros((1, 109)); second[0, 108] = 1.0
        logits_first, values_first = policy(first)
        logits_second, _values_second = policy(second)
        self.assertEqual(tuple(logits_first.shape), (1, 4, 4))
        self.assertEqual(tuple(values_first.shape), (1,))
        self.assertFalse(torch.allclose(logits_first, logits_second))

    def test_film_checkpoint_round_trip_preserves_conditioning(self) -> None:
        policy = ActorCritic(PolicyConfig(obs_dim=109, state_obs_dim=104,
                                          preference_dim=5, conditioning="film"))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.pt")
            policy.save(path)
            loaded, _extra = ActorCritic.load(path)
        self.assertEqual(loaded.config.conditioning, "film")
        self.assertEqual(loaded.config.state_obs_dim, 104)
        self.assertEqual(loaded.config.preference_dim, 5)

    def test_v3_freeze_refuses_test_release_and_retains_audit_states(self) -> None:
        protocol, split, audit = V3._checked()
        self.assertTrue(split["test_v5_sealed"])
        self.assertIn("test-v5", protocol["test_v5_dependency"])
        self.assertEqual([state["id"] for state in audit["states"]],
                         ["sample_legal", "share_process_legal",
                          "stale_local_fresh_remote", "resource_pressure"])
        self.assertNotIn("test", V3.main.__code__.co_varnames)
