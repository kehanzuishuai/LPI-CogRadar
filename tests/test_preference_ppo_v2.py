"""v2 的协议边界：不改变 v1，且只对 v2 缩短 share 候选链。"""
from __future__ import annotations

import os
import unittest

from rl_resource.actions import ACTION_IDLE, ACTION_SAMPLE, ACTION_SHARE
from rl_resource.env import CentralizedResourceSchedulingEnv, EnvConfig


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_SPLIT = os.path.join(ROOT, "config", "preference_ppo_v2_splits.json")


class TestPreferencePpoV2(unittest.TestCase):
    def _env(self, *, reward_mode: str, share_requires_track: bool) -> CentralizedResourceSchedulingEnv:
        return CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_validation_handover", seed=211, steps=24,
            arm="main_baseline", reward_mode=reward_mode,
            preference=(0.0, 0.0, 0.0, 0.0, 1.0), preference_conditioned=True,
            scenario_registry_path=V2_SPLIT,
            share_candidate_requires_track=share_requires_track,
        ))

    def test_v2_outbox_can_offer_share_without_local_track_but_v1_cannot(self) -> None:
        v1 = self._env(reward_mode="balanced", share_requires_track=True)
        v1.reset(seed=211)
        v1.step([ACTION_IDLE, ACTION_SAMPLE])
        self.assertFalse(v1._mask[1][ACTION_SHARE])

        v2 = self._env(reward_mode="preference_v2", share_requires_track=False)
        v2.reset(seed=211)
        v2.step([ACTION_IDLE, ACTION_SAMPLE])
        self.assertTrue(v2._mask[1][ACTION_SHARE])
        self.assertFalse(v2._runtime.centers["NODE_B"].tracks)

    def test_v2_communication_saving_is_smooth_and_real_share_is_accounted(self) -> None:
        env = self._env(reward_mode="preference_v2", share_requires_track=False)
        env.reset(seed=211)
        env.step([ACTION_IDLE, ACTION_SAMPLE])
        _obs, _reward, _terminated, _truncated, info = env.step([ACTION_IDLE, ACTION_SHARE])
        self.assertAlmostEqual(info["reward_components"]["preference_v2_communication_saving"], 0.5)
        self.assertEqual(len(env._world["bus"].log), 1)
        consumed = sum(node.budget.consumed.get(__import__(
            "resource_management.units", fromlist=["ResourceUnit"]).ResourceUnit.COMM_BYTE, 0.0)
                       for node in env._executor.nodes.values())
        self.assertEqual(consumed, 128.0)
        self.assertTrue(env._executor.conservation_report()["all_conserved"])
