"""教学资源调度学习协议的接口与精确案例测试。"""

from __future__ import annotations

import ast
import os
import unittest

from experiment_config import make_env
from resource_management.exact_cases import exact_cases
from resource_management.learning_env import (
    ResourceSchedulingLearningEnv,
    bootstrap_multiplier,
)
from resource_management.learning_protocol import (
    SealedTestSplitError,
    get_split,
    load_registry,
    verify_split_digest,
)


class TestExactQueueingCases(unittest.TestCase):
    def test_all_exact_returns_costs_and_rewards(self) -> None:
        for name, (case, expected) in exact_cases().items():
            with self.subTest(case=name):
                env = ResourceSchedulingLearningEnv(case)
                observation, _ = env.reset(seed=42)
                self.assertTrue(env.observation_space.contains(observation))
                rewards = []
                component_rows = []
                info = {}
                terminated = truncated = False
                for action in expected.actions:
                    observation, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward)
                    component_rows.append(info["reward_components"])
                    self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(tuple(rewards), expected.rewards)
                component_keys = (
                    "completion", "invalid_action", "expiry", "waiting",
                    "terminal_supplement",
                )
                actual_components = tuple(
                    tuple(row[key] for key in component_keys)
                    for row in component_rows
                )
                self.assertEqual(actual_components, expected.reward_components)
                for reward, row in zip(rewards, component_rows):
                    self.assertAlmostEqual(sum(row.values()), reward)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["terminated_reason"], expected.terminated_reason)
                metrics = env.evaluation_metrics()
                self.assertAlmostEqual(
                    metrics["undiscounted_return"], expected.undiscounted_return
                )
                self.assertAlmostEqual(
                    metrics["discounted_return"], expected.discounted_return
                )
                self.assertAlmostEqual(
                    metrics["cumulative_cost"], expected.cumulative_cost
                )
                self.assertAlmostEqual(
                    metrics["resource_consumption"], expected.cumulative_cost
                )

    def test_remaining_time_is_part_of_observation(self) -> None:
        case, _ = exact_cases()["wait_then_complete"]
        env = ResourceSchedulingLearningEnv(case)
        observation, _ = env.reset()
        self.assertEqual(observation[1], 1.0)
        observation, _, _, _, info = env.step(0)
        self.assertAlmostEqual(observation[1], 2.0 / 3.0)
        self.assertAlmostEqual(info["remaining_time_fraction"], 2.0 / 3.0)

    def test_external_limit_is_truncation_and_bootstraps(self) -> None:
        case, _ = exact_cases()["horizon_unresolved"]
        env = ResourceSchedulingLearningEnv(case, external_step_limit_steps=1)
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["truncated_reason"], "external_step_limit")
        self.assertTrue(info["bootstrap_allowed"])
        self.assertEqual(info["reward_components"]["terminal_supplement"], 0.0)
        self.assertEqual(reward, -0.25)
        self.assertEqual(bootstrap_multiplier(terminated, truncated), 1.0)
        self.assertTrue(any(info["next_action_mask"]))

    def test_natural_horizon_is_termination_and_no_bootstrap(self) -> None:
        case, _ = exact_cases()["horizon_unresolved"]
        env = ResourceSchedulingLearningEnv(case)
        env.reset()
        env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminated_reason"], "task_horizon")
        self.assertFalse(info["bootstrap_allowed"])
        self.assertEqual(bootstrap_multiplier(terminated, truncated), 0.0)
        self.assertFalse(any(info["next_action_mask"]))

    def test_resource_exhaustion_is_natural_termination(self) -> None:
        case, _ = exact_cases()["resource_exhaustion"]
        env = ResourceSchedulingLearningEnv(case)
        env.reset()
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminated_reason"], "resource_exhausted")
        self.assertEqual(info["metrics"]["completed"], 1)
        self.assertEqual(info["metrics"]["unresolved"], 1)


class TestSplitProtocol(unittest.TestCase):
    def test_registry_is_disjoint_and_digest_is_frozen(self) -> None:
        registry = load_registry()
        self.assertEqual(set(registry["splits"]), {"train", "validation", "test"})
        referenced = {
            scenario
            for split in registry["splits"].values()
            for scenario in split["scenarios"]
        }
        self.assertEqual(set(registry["scenario_catalog"]), referenced)
        self.assertEqual(len(verify_split_digest()), 64)

    def test_validation_is_available_but_test_is_sealed(self) -> None:
        validation = get_split("validation")
        self.assertTrue(validation.seeds)
        with self.assertRaises(SealedTestSplitError):
            get_split("test")
        test = get_split("test", release_test=True)
        self.assertTrue(test.seeds)

    def test_learning_env_is_independent_of_perception_chain(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "resource_management", "learning_env.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), path)
        forbidden = {"engine", "sensor", "fusion", "communication"}
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & forbidden)


class TestLegacyRadarHorizonSemantics(unittest.TestCase):
    def _run_to_horizon(self, mode: str):
        env = make_env(horizon_semantics=mode)
        env.reset(seed=42)
        final = None
        for _ in range(env.sim.scenario.num_steps):
            final = env.step(0)
            if final[2] or final[3]:
                break
        self.assertIsNotNone(final)
        return final

    def test_configured_finite_horizon_is_natural_termination(self) -> None:
        _, _, terminated, truncated, info = self._run_to_horizon("finite_task")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminated_reason"], "task_horizon")
        self.assertFalse(info["bootstrap_allowed"])

    def test_old_horizon_behavior_requires_explicit_compatibility_mode(self) -> None:
        _, _, terminated, truncated, info = self._run_to_horizon(
            "legacy_truncation"
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["truncated_reason"], "legacy_horizon")
        self.assertTrue(info["bootstrap_allowed"])


if __name__ == "__main__":
    unittest.main()
