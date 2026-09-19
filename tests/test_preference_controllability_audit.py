"""空样本语义与固定审计状态的回归测试；不加载 checkpoint 或 test-v4。"""
from __future__ import annotations

import importlib.util
import os
import unittest

from rl_resource.actions import ACTION_IDLE
from rl_resource.env import CentralizedResourceSchedulingEnv, EnvConfig
from rl_resource.train import _observed_quality_stats


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_SPLIT = os.path.join(ROOT, "config", "preference_ppo_v2_splits.json")
SPEC = importlib.util.spec_from_file_location(
    "preference_controllability_audit",
    os.path.join(ROOT, "tools", "audit_preference_controllability.py"),
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TestPreferenceControllabilityAudit(unittest.TestCase):
    def test_empty_information_stat_is_not_numeric_best(self) -> None:
        self.assertEqual(_observed_quality_stats([], "mean_track_age_s"), (None, 0))
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=24, arm="main_baseline",
            reward_mode="preference_v2", preference=(0, 0, 1, 0, 0),
            preference_conditioned=True, scenario_registry_path=V2_SPLIT,
            share_candidate_requires_track=False))
        env.reset(seed=101)
        _obs, _reward, _terminated, _truncated, info = env.step([ACTION_IDLE, ACTION_IDLE])
        self.assertEqual(info["reward_components"]["preference_v2_estimate_quality"], 0.0)

    def test_stale_local_fresh_remote_state_is_fixed_and_observable(self) -> None:
        state = next(row for row in AUDIT._load(AUDIT.AUDIT)["states"]
                     if row["id"] == "stale_local_fresh_remote")
        env = AUDIT._make_env(state, AUDIT.BALANCED)
        snapshot = AUDIT._state_snapshot(env, state)
        self.assertIn("share", snapshot["legal_actions"])
        self.assertEqual(snapshot["node_information"]["NODE_A"]["information_age_state"], "observed")
        self.assertGreater(snapshot["node_information"]["NODE_A"]["mean_information_age_s"], 0.0)
        self.assertEqual(snapshot["information_age_state"], "not_applicable")
