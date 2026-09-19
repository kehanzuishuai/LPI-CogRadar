"""信念状态桥接的单元测试。

这些测试回答一个关键问题：**「仅观测」的脚本策略拿到的到底是不是观测值，
而不是偷偷拿到的真值？**

如果信念桥接没做对（比如忘记写回能量、或者压根没扰动目标位置），
那么「部分可观测下的规则策略」实际上仍在读真值，
整个 P1 对比就会得出「部分可观测没什么影响」的假结论。

因此这里逐项验证：信念仿真器在 `preview()` 下复现的
距离 / RCS / 干扰比 / 剩余能量，必须与观测估计值一致，
并且**在噪声开启时显著偏离真值**。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_config import (  # noqa: E402
    belief_policy_specs,
    make_env,
    make_pomdp_env,
    run_belief_episode,
)
from strategy.belief_policy import (  # noqa: E402
    BeliefOptions,
    BeliefPolicy,
    build_belief_simulator,
)
from strategy.power_policy import GreedyOraclePolicy, RuleBasedPowerPolicy  # noqa: E402


def _belief_after_steps(preset: str, steps: int = 6, seed: int = 42):
    env = make_pomdp_env(preset=preset)
    env.reset(seed=seed)
    info = None
    for action in [5, 6, 7, 4, 6, 8][:steps]:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    belief = build_belief_simulator(env, info["observations"])
    return env, info, belief


class TestBeliefFidelity(unittest.TestCase):
    """信念仿真器必须复现观测到的估计值。"""

    def test_energy_matches_estimate(self) -> None:
        env, info, belief = _belief_after_steps("moderate")
        self.assertAlmostEqual(
            belief.remaining_energy_j,
            info["observations"]["remaining_energy"]["value"],
            places=6,
        )

    def test_primary_range_matches_estimate(self) -> None:
        env, info, belief = _belief_after_steps("moderate")
        primary = belief._primary_target()
        self.assertIsNotNone(primary)
        self.assertAlmostEqual(
            primary.range_to(belief.radar.x, belief.radar.y),
            info["observations"]["target_range"]["value"],
            places=4,
        )

    def test_min_rcs_matches_estimate(self) -> None:
        env, info, belief = _belief_after_steps("moderate")
        belief_min = min(t.rcs_m2 for t in belief.active_targets())
        self.assertAlmostEqual(
            belief_min, info["observations"]["target_rcs"]["value"], places=6
        )

    def test_jam_ratio_matches_estimate(self) -> None:
        """这条最容易出错：干扰比要先标定再反解。"""
        env, info, belief = _belief_after_steps("moderate")
        probe = belief.preview(belief.power_levels_w[0])
        self.assertAlmostEqual(
            probe.jam_noise_ratio,
            info["observations"]["jam_ratio"]["value"],
            places=4,
        )

    def test_exposure_matches_estimate(self) -> None:
        env, info, belief = _belief_after_steps("moderate")
        self.assertAlmostEqual(
            belief.exposure.value, info["observations"]["exposure"]["value"], places=9
        )

    def test_belief_differs_from_truth_under_noise(self) -> None:
        """噪声开启时，信念必须**明显不是**真值，否则桥接形同虚设。"""
        env, info, belief = _belief_after_steps("severe")
        true_remaining = env.sim.remaining_energy_j
        belief_remaining = belief.remaining_energy_j
        self.assertNotAlmostEqual(true_remaining, belief_remaining, places=6)

        true_primary = env.sim._primary_target().range_to(env.sim.radar.x, env.sim.radar.y)
        belief_primary = belief._primary_target().range_to(belief.radar.x, belief.radar.y)
        self.assertNotAlmostEqual(true_primary, belief_primary, places=6)

    def test_belief_does_not_mutate_truth(self) -> None:
        """构造信念不得改动真值仿真器。"""
        env = make_pomdp_env(preset="moderate")
        env.reset(seed=42)
        _, _, _, _, info = env.step(6)
        before = (
            env.sim.remaining_energy_j,
            env.sim.exposure.value,
            [t.rcs_m2 for t in env.sim.targets],
            env.sim._primary_target().range_to(env.sim.radar.x, env.sim.radar.y),
        )
        build_belief_simulator(env, info["observations"])
        after = (
            env.sim.remaining_energy_j,
            env.sim.exposure.value,
            [t.rcs_m2 for t in env.sim.targets],
            env.sim._primary_target().range_to(env.sim.radar.x, env.sim.radar.y),
        )
        self.assertEqual(before, after)

    def test_full_mode_belief_is_identity(self) -> None:
        """full 模式下没有估计值，信念应等于真值。"""
        env = make_env(observation_mode="full")
        env.reset(seed=42)
        env.step(6)
        belief = build_belief_simulator(env, {})
        self.assertAlmostEqual(belief.remaining_energy_j, env.sim.remaining_energy_j)
        self.assertAlmostEqual(
            belief._primary_target().range_to(belief.radar.x, belief.radar.y),
            env.sim._primary_target().range_to(env.sim.radar.x, env.sim.radar.y),
        )


class TestBeliefPolicy(unittest.TestCase):
    def test_full_mode_delegates_to_truth(self) -> None:
        """full 模式下 BeliefPolicy 直接委托内部策略（与旧行为一致）。"""
        env = make_env(observation_mode="full")
        env.reset(seed=42)
        env.step(5)
        policy = BeliefPolicy(RuleBasedPowerPolicy())
        level = policy.select_level_from_env(env)
        self.assertEqual(level, RuleBasedPowerPolicy().select_level(env.sim))
        self.assertEqual(policy.belief_count, 0)  # 未构造信念

    def test_pomdp_mode_builds_belief(self) -> None:
        env = make_pomdp_env(preset="moderate")
        env.reset(seed=42)
        env.step(5)
        policy = BeliefPolicy(RuleBasedPowerPolicy())
        level = policy.select_level_from_env(env)
        self.assertGreaterEqual(level, 0)
        self.assertLess(level, env.sim.num_levels)
        self.assertEqual(policy.belief_count, 1)
        self.assertIsNotNone(policy.last_belief)

    def test_episode_runs_to_completion(self) -> None:
        for preset in ("mild", "moderate", "severe"):
            env = make_pomdp_env(preset=preset)
            policy = BeliefPolicy(RuleBasedPowerPolicy())
            results = run_belief_episode(env, policy, seed=42)
            self.assertGreater(len(results), 0, msg=f"预设 {preset} 未产生任何步")
            self.assertLessEqual(
                env.sim.cumulative_energy_j, env.sim.radar.energy_budget_j + 1e-6
            )

    def test_observation_only_rule_differs_from_truth_rule(self) -> None:
        """仅观测的规则策略在强噪声下应与全状态规则策略产生不同决策。

        如果两者完全一致，就说明信念桥接没有真正限制信息。
        """
        pomdp_env = make_pomdp_env(preset="severe")
        policy = BeliefPolicy(RuleBasedPowerPolicy())
        belief_results = run_belief_episode(pomdp_env, policy, seed=42)
        belief_powers = [round(r.tx_power_w, 6) for r in belief_results]

        full_env = make_env(observation_mode="full")
        truth_results = run_belief_episode(
            full_env, BeliefPolicy(RuleBasedPowerPolicy()), seed=42
        )
        truth_powers = [round(r.tx_power_w, 6) for r in truth_results]

        self.assertNotEqual(belief_powers, truth_powers)

    def test_specs_build(self) -> None:
        specs = belief_policy_specs(60)
        self.assertEqual(len(specs), 3)
        for spec in specs:
            policy = spec.factory()
            self.assertTrue(hasattr(policy, "select_level_from_env"))
            self.assertIn("仅观测", policy.describe())

    def test_greedy_oracle_usable_on_belief(self) -> None:
        env = make_pomdp_env(preset="moderate")
        policy = BeliefPolicy(GreedyOraclePolicy())
        results = run_belief_episode(env, policy, seed=7)
        self.assertGreater(len(results), 0)


class TestNoFutureLeak(unittest.TestCase):
    """v4.3 真值泄漏审计的回归测试。

    背景：`copy.deepcopy(sim)` 会把**整段预生成的干扰起伏**与
    **观测不到的平台真值位置**一起复制进信念。审计实测这两项都会
    真的改变策略行为，等于让"部分可观测"策略偷看未来与敌方位置：

    * 篡改真值**未来**起伏后，信念推演 J/N 从 1.52 变成 6.91；
    * 把真值 ESM 挪走 100 km 后，信念 Pint 从 0.530 变成 0.329。

    这里的判据是**行为级**的，不是"字段是否相等"：
    篡改真值的未来 / 他平台位置后，信念的推演结果必须**逐位不变**。
    字段级判据太脆弱（改个字段名就失效），行为级才是真正要守的东西。
    """

    @staticmethod
    def _env_with_active_jammer(steps: int = 30):
        """推进到干扰机**开机时刻**。

        干扰机时间窗是 20~45 s。用 t=10 s 做这个测试会得到 J/N 恒为 0，
        对"未来起伏"完全不敏感 —— 第一次跑就踩了这个坑，测出来是**假阴性**。
        """
        env = make_pomdp_env(preset="moderate")
        env.reset(seed=42)
        for _ in range(steps):
            env.step(6)
        return env

    @staticmethod
    def _rollout_jam(env, steps: int = 8):
        belief = build_belief_simulator(env)
        out = []
        for _ in range(steps):
            if belief.is_done:
                break
            belief.step(6)
            out.append(round(belief.jam_state()[3], 9))
        return out

    def test_belief_does_not_carry_future_jammer(self) -> None:
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(env)
        truth_series = env.sim.jammers[0]._fluctuation_series
        belief_series = belief.jammers[0]._fluctuation_series
        index = env.sim.step_index
        self.assertNotEqual(
            truth_series[index:], belief_series[index:],
            "信念仍携带真值的未来起伏序列",
        )
        self.assertEqual(truth_series[: index + 1], belief_series[: index + 1],
                         "可观测的历史必须保留")

    def test_belief_rollout_invariant_to_truth_future(self) -> None:
        """**决定性判据**：篡改真值未来，信念推演必须不变。"""
        env = self._env_with_active_jammer()
        before = self._rollout_jam(env)
        self.assertTrue(any(v > 0.0 for v in before),
                        "干扰机未开机，本测试无效（J/N 恒为 0 时测不出泄漏）")
        jammer = env.sim.jammers[0]
        for i in range(env.sim.step_index + 1, len(jammer._fluctuation_series)):
            jammer._fluctuation_series[i] = 5.0
        after = self._rollout_jam(env)
        self.assertEqual(before, after, "信念推演随真值未来变化 → 未来信息泄漏")

    def test_belief_pint_invariant_to_truth_esm_position(self) -> None:
        """雷达观测不到侦察机，信念的 Pint 不得跟随真值 ESM 位置。"""
        env = self._env_with_active_jammer()
        before = build_belief_simulator(env).preview(18.0).intercept_prob
        env.sim.interceptors[0].x += 100000.0
        after = build_belief_simulator(env).preview(18.0).intercept_prob
        self.assertAlmostEqual(before, after, places=12,
                               msg="信念 Pint 随真值 ESM 位置变化 → 他平台真值泄漏")

    def test_unobserved_interceptor_deactivated(self) -> None:
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(env)
        self.assertFalse(belief.interceptors[0].is_active,
                         "观测不到的侦察机应被置为 inactive")

    def test_belief_uses_own_rng(self) -> None:
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(env)
        self.assertIsNot(belief.rng, env.sim.rng)

    def test_keep_truth_option_is_explicit_leak(self) -> None:
        """`keep_truth` 是显式"先知"上界，必须真的保留真值。"""
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(
            env, options=BeliefOptions(future_jammer_model="keep_truth")
        )
        index = env.sim.step_index
        self.assertEqual(
            env.sim.jammers[0]._fluctuation_series[index:],
            belief.jammers[0]._fluctuation_series[index:],
        )

    def test_resample_option_has_no_truth_but_is_not_constant(self) -> None:
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(
            env, options=BeliefOptions(future_jammer_model="resample")
        )
        index = env.sim.step_index
        self.assertNotEqual(
            env.sim.jammers[0]._fluctuation_series[index:],
            belief.jammers[0]._fluctuation_series[index:],
        )
        self.assertGreater(len(set(belief.jammers[0]._fluctuation_series[index:])), 1)

    def test_invalid_options_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BeliefOptions(future_jammer_model="bogus").validate()
        with self.assertRaises(ValueError):
            BeliefOptions(unobservable_interceptor="bogus").validate()

    def test_belief_metadata_recorded(self) -> None:
        env = self._env_with_active_jammer()
        belief = build_belief_simulator(env)
        self.assertTrue(getattr(belief, "belief_metadata", {}).get("sanitized"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
