"""v4.0 部分可观测（POMDP）环境的单元测试。

用标准库 unittest，不依赖 pytest / numpy / torch，
因此可以在任何 Python 下直接运行：

    python -m unittest discover -s tests -v
    python tests/test_pomdp_env.py            # 也可单独跑

测试的核心不变式（最重要的一条）
--------------------------------
「噪声只加在智能体看到的东西上，绝不影响真值演进与奖励」。

因此：**同一串动作**在 full 与 pomdp 两个模式下，
必须给出完全相同的真实轨迹（Pd / Pint / 干扰 / 暴露 / 能耗）与逐步奖励。
这条不变式如果被破坏，就说明观测模型越界改动了物理层，
那么所有「部分可观测 vs 全可观」的对比实验都失去意义。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.observation_model import ObservationModel, ObservationNoiseConfig  # noqa: E402
from experiment_config import (  # noqa: E402
    DEFAULT_HISTORY_LEN,
    OBSERVATION_PRESETS,
    make_env,
    make_pomdp_env,
    observation_preset,
)


ACTION_SEQUENCE = [5, 2, 6, 10, 0, 1, 7, 3, 9, 4] * 3


def _rollout(env, actions):
    obs, _ = env.reset(seed=42)
    trace = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append(
            (
                round(info["tx_power_w"], 9),
                round(info["pd_min"], 12),
                round(info["intercept_prob"], 12),
                round(info["jam_noise_ratio"], 12),
                round(info["exposure_next"], 12),
                round(info["cumulative_energy_j"], 9),
                round(reward, 12),
            )
        )
        if terminated or truncated:
            break
    return trace, obs


class TestFullObservableControl(unittest.TestCase):
    """full 模式必须保持 v3.1 行为：12 维、零噪声。"""

    def test_dim_is_12(self) -> None:
        env = make_env()
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 12)
        self.assertEqual(info["observation_mode"], "full")
        self.assertEqual(info["observation_space_dim"], 12)

    def test_noise_disabled_in_full_mode(self) -> None:
        """即使显式传噪声配置，full 模式也必须把它关掉。"""
        env = make_env(
            observation_mode="full",
            observation_noise=observation_preset("severe"),
        )
        self.assertFalse(env.observation_model.enabled)
        self.assertEqual(env.observation_quality(), 1.0)
        self.assertEqual(env.observation_uncertainty(), {})

    def test_observation_is_truth(self) -> None:
        """full 模式下观测里的暴露量应等于仿真真值。"""
        env = make_env()
        env.reset(seed=42)
        env.step(6)
        obs = env._observation()
        self.assertAlmostEqual(obs[11], env.sim.exposure.value, places=9)


class TestPomdpInvariant(unittest.TestCase):
    """核心不变式：观测模式不改变物理与奖励。"""

    def test_trajectory_and_reward_identical(self) -> None:
        full_env = make_env(observation_mode="full")
        pomdp_env = make_pomdp_env(preset="severe")

        full_trace, _ = _rollout(full_env, ACTION_SEQUENCE)
        pomdp_trace, _ = _rollout(pomdp_env, ACTION_SEQUENCE)

        self.assertEqual(len(full_trace), len(pomdp_trace))
        for step, (a, b) in enumerate(zip(full_trace, pomdp_trace)):
            self.assertEqual(a, b, msg=f"第 {step} 步真实轨迹/奖励被观测噪声改动")

    def test_observation_model_does_not_touch_simulator(self) -> None:
        """观测模型只读仿真，不写。"""
        env = make_pomdp_env(preset="severe")
        env.reset(seed=42)
        before = (env.sim.exposure.value, env.sim.remaining_energy_j)
        env.step(5)
        after = (env.sim.exposure.value, env.sim.remaining_energy_j)
        self.assertNotEqual(before, after)  # 仿真确实推进了
        # 观测模型自身的统计不应干扰仿真能耗
        self.assertLessEqual(after[1], before[1])
        self.assertGreaterEqual(env.observation_model.stats["steps"], 1)


class TestPomdpObservationShape(unittest.TestCase):
    def test_dim_is_16(self) -> None:
        env = make_pomdp_env(preset="moderate")
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 16)
        self.assertEqual(info["observation_mode"], "pomdp")
        self.assertNotIn("observations", info)  # reset 时还没有估计
        _, _, _, _, info = env.step(5)
        self.assertIn("observations", info)
        self.assertIn("target_range", info["observations"])

    def test_truth_hidden_by_default(self) -> None:
        """真值默认不得出现在 info 里，否则策略代码可能顺手读到。"""
        env = make_pomdp_env(preset="moderate")
        env.reset(seed=42)
        _, _, _, _, info = env.step(5)
        for record in info["observations"].values():
            self.assertNotIn("truth", record)

    def test_history_windowing(self) -> None:
        env = make_pomdp_env(preset="moderate", history_len=DEFAULT_HISTORY_LEN)
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 16 * DEFAULT_HISTORY_LEN)
        self.assertEqual(info["history_len"], DEFAULT_HISTORY_LEN)
        self.assertEqual(len(info["observation_features"]), 16 * DEFAULT_HISTORY_LEN)

    def test_observation_bounds_hold(self) -> None:
        env = make_pomdp_env(preset="severe", history_len=3)
        obs, _ = env.reset(seed=7)
        for action in ACTION_SEQUENCE:
            self.assertTrue(env.observation_space.contains(obs))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_uncertainty_channels_reported(self) -> None:
        env = make_pomdp_env(preset="moderate")
        env.reset(seed=42)
        _, _, _, _, info = env.step(5)
        self.assertGreaterEqual(info["observation_quality"], 0.0)
        self.assertLessEqual(info["observation_quality"], 1.0)
        sigma = env.observation_uncertainty()
        self.assertIn("interceptor_range", sigma)
        self.assertGreater(sigma["interceptor_range"], 0.0)


class TestHiddenTruth(unittest.TestCase):
    """被隐藏的量必须真的带上不可忽略的误差。"""

    def test_hidden_fields_are_noisy(self) -> None:
        env = make_pomdp_env(preset="moderate", expose_observation_truth=True)
        env.reset(seed=42)
        errors = {"interceptor_range": [], "pint_eff": [], "exposure": []}
        for action in ACTION_SEQUENCE:
            _, _, terminated, truncated, info = env.step(action)
            for name in errors:
                record = info["observations"][name]
                errors[name].append(abs(record["value"] - record["truth"]))
            if terminated or truncated:
                break

        for name, values in errors.items():
            mean_error = sum(values) / len(values)
            self.assertGreater(mean_error, 0.0, msg=f"{name} 完全没有观测误差")

        # 侦察机距离估计误差应接近配置的 4 km 量级
        esm_error = sum(errors["interceptor_range"]) / len(errors["interceptor_range"])
        self.assertGreater(esm_error, 500.0)

    def test_unhiding_makes_field_exact(self) -> None:
        """关闭 hide 开关后，该量应等于真值且不确定度为 0（消融对照）。"""
        env = make_pomdp_env(
            preset="severe",
            expose_observation_truth=True,
            hide_interceptor_truth=False,
            hide_exposure_truth=False,
            hide_pint_truth=False,
        )
        env.reset(seed=42)
        _, _, _, _, info = env.step(5)
        for name in ("interceptor_range", "exposure", "pint_eff"):
            record = info["observations"][name]
            # to_dict() 保留 6 位小数，故按 6 位比较
            self.assertAlmostEqual(record["value"], record["truth"], places=6)
            self.assertEqual(record["sigma"], 0.0)


class TestNoiseMechanisms(unittest.TestCase):
    def test_reproducible_with_same_seed(self) -> None:
        traces = []
        for _ in range(2):
            env = make_pomdp_env(preset="moderate")
            env.reset(seed=123)
            frames = []
            for action in ACTION_SEQUENCE[:10]:
                _, _, _, _, info = env.step(action)
                frames.append(tuple(sorted(
                    (k, round(v["value"], 12)) for k, v in info["observations"].items()
                )))
            traces.append(frames)
        self.assertEqual(traces[0], traces[1])

    def test_seed_changes_the_noise(self) -> None:
        def first_frame(seed: int):
            env = make_pomdp_env(preset="moderate")
            env.reset(seed=seed)
            _, _, _, _, info = env.step(5)
            return info["observations"]["target_range"]["value"]

        self.assertNotEqual(first_frame(42), first_frame(43))

    def test_dropout_actually_drops(self) -> None:
        env = make_pomdp_env(preset="severe", dropout_prob=0.9)
        env.reset(seed=5)
        for action in ACTION_SEQUENCE:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self.assertGreater(env.observation_model.stats["dropout_slots"], 0)
        self.assertGreater(env.observation_model.stats["dropped"], 0)

    def test_no_dropout_when_prob_zero(self) -> None:
        env = make_pomdp_env(preset="moderate", dropout_prob=0.0)
        env.reset(seed=5)
        for action in ACTION_SEQUENCE:
            _, _, terminated, truncated, info = env.step(action)
            for record in info["observations"].values():
                self.assertTrue(record["observed"])
            if terminated or truncated:
                break

    def test_delay_shifts_measurement(self) -> None:
        """delay_steps=1 时，本步观测应等于上一步的测量（真值随之滞后）。"""
        model = ObservationModel(
            ObservationNoiseConfig(enabled=True, delay_steps=1, dropout_prob=0.0)
        )
        model.reset(seed=1)
        first = model.observe({"target_range": 1000.0}, 0)
        second = model.observe({"target_range": 5000.0}, 1)
        self.assertAlmostEqual(second["target_range"].value, first["target_range"].value, places=9)
        self.assertTrue(second["target_range"].stale)

    def test_zero_delay_follows_truth(self) -> None:
        model = ObservationModel(
            ObservationNoiseConfig(
                enabled=True, delay_steps=0, dropout_prob=0.0, range_sigma_m=0.0
            )
        )
        model.reset(seed=1)
        first = model.observe({"target_range": 1000.0}, 0)
        second = model.observe({"target_range": 5000.0}, 1)
        self.assertAlmostEqual(first["target_range"].value, 1000.0)
        self.assertAlmostEqual(second["target_range"].value, 5000.0)

    def test_underreporting(self) -> None:
        """report_honest_sigma=False 时上报的不确定度应显著低于真实噪声。"""
        honest = ObservationModel(
            ObservationNoiseConfig(enabled=True, delay_steps=0, pint_sigma=0.1)
        )
        honest.reset(seed=3)
        overconfident = ObservationModel(
            ObservationNoiseConfig(
                enabled=True, delay_steps=0, pint_sigma=0.1,
                report_honest_sigma=False, sigma_underreport_factor=0.5,
            )
        )
        overconfident.reset(seed=3)
        h = honest.observe({"pint_eff": 0.5}, 0)["pint_eff"].sigma
        o = overconfident.observe({"pint_eff": 0.5}, 0)["pint_eff"].sigma
        self.assertAlmostEqual(o, h * 0.5, places=9)


class TestPresetsAndConfig(unittest.TestCase):
    def test_all_presets_build(self) -> None:
        for name in OBSERVATION_PRESETS:
            env = make_pomdp_env(preset=name)
            obs, _ = env.reset(seed=42)
            self.assertEqual(len(obs), 16, msg=f"预设 {name} 维度不对")

    def test_preset_copy_is_independent(self) -> None:
        first = observation_preset("moderate")
        first["range_sigma_m"] = 9999.0
        self.assertEqual(OBSERVATION_PRESETS["moderate"]["range_sigma_m"], 120.0)

    def test_invalid_preset_raises(self) -> None:
        with self.assertRaises(KeyError):
            observation_preset("nonexistent")

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObservationNoiseConfig(range_sigma_m=-1.0).validate()
        with self.assertRaises(ValueError):
            ObservationNoiseConfig(dropout_prob=1.0).validate()
        with self.assertRaises(ValueError):
            ObservationNoiseConfig(delay_steps=-1).validate()

    def test_invalid_observation_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_env(observation_mode="bogus")

    def test_severe_is_worse_than_mild(self) -> None:
        """重度部分可观测的观测质量必须低于轻度。"""

        def mean_quality(preset: str) -> float:
            env = make_pomdp_env(preset=preset)
            env.reset(seed=42)
            values = []
            for action in ACTION_SEQUENCE:
                _, _, terminated, truncated, info = env.step(action)
                values.append(info["observation_quality"])
                if terminated or truncated:
                    break
            return sum(values) / len(values)

        self.assertLess(mean_quality("severe"), mean_quality("mild"))

    def test_energy_constraint_still_hard(self) -> None:
        """部分可观测不得松动能量硬约束。"""
        env = make_pomdp_env(preset="severe")
        env.reset(seed=42)
        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(10)  # 一直用最高档
            self.assertLessEqual(
                info["cumulative_energy_j"],
                env.sim.radar.energy_budget_j + 1e-6,
            )
            if terminated or truncated:
                break
        self.assertTrue(terminated or truncated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
