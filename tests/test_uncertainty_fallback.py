"""P2 单元测试 + 集成烟雾测试：集成 DQN 与不确定度感知回退。

这些测试要钉住三件事：
1. 集成的 N 个成员**初始化确实不同**（否则分歧恒为 0，整个不确定度信号失效）；
2. OOD 评分在观测远离训练分布时**确实升高**；
3. 回退策略在阈值被故意调松/调紧时**行为随之改变**——
   这既验证了机制有效，也说明回退率是阈值驱动的（不能当成"AI 的自省能力"）。
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_config import make_pomdp_env  # noqa: E402
from rl.dqn_agent import DQNConfig  # noqa: E402
from rl.ensemble_agent import EnsembleConfig, EnsembleDQNAgent  # noqa: E402
from strategy.uncertainty_policy import (  # noqa: E402
    MODE_AI,
    MODE_FALLBACK,
    MODE_SHIELD,
    FallbackConfig,
    UncertaintyAwarePolicy,
)


def _trained_agent(obs_dim: int = 16, members: int = 3, steps: int = 80) -> EnsembleDQNAgent:
    """造一个「训练了一小会儿」的集成智能体（纯随机数据，只为跑通机制）。"""
    config = DQNConfig(
        obs_dim=obs_dim, n_actions=11, seed=0,
        learning_starts=10, batch_size=8, buffer_capacity=500,
    )
    agent = EnsembleDQNAgent(config, EnsembleConfig(ensemble_size=members), device="cpu")
    rng = random.Random(0)
    for _ in range(steps):
        obs = [rng.random() for _ in range(obs_dim)]
        next_obs = [rng.random() for _ in range(obs_dim)]
        agent.store(obs, rng.randrange(11), rng.uniform(-1, 1), next_obs, False, [True] * 11)
        agent.env_steps += 1
        agent.maybe_train()
    return agent


class TestEnsembleMechanics(unittest.TestCase):
    def test_members_are_different(self) -> None:
        agent = _trained_agent()
        self.assertEqual(len(agent.online_networks), 3)
        obs = [0.5] * 16
        q = agent.q_ensemble(obs)
        self.assertEqual(tuple(q.shape), (3, 11))
        # 成员之间必须有非零分歧，否则不确定度信号恒为 0、完全无用
        spread = float(q[:, 0].std(unbiased=False))
        self.assertGreater(spread, 1e-6)
        self.assertGreater(float(q.std(dim=0, unbiased=False).max()), 1e-6)

    def test_q_values_is_ensemble_mean(self) -> None:
        agent = _trained_agent()
        obs = [0.3] * 16
        q_all = agent.q_ensemble(obs)
        self.assertTrue(
            all(
                abs(float(agent.q_values(obs)[i]) - float(q_all.mean(dim=0)[i])) < 1e-6
                for i in range(11)
            )
        )

    def test_uncertainty_respects_mask(self) -> None:
        agent = _trained_agent()
        obs = [0.5] * 16
        mask = [True, True, True] + [False] * 8
        info, _ = agent.uncertainty(obs, mask)
        self.assertEqual(info.n_feasible, 3)

    def test_disagreement_in_range(self) -> None:
        agent = _trained_agent()
        info, _ = agent.uncertainty([0.5] * 16, [True] * 11)
        self.assertGreaterEqual(info.disagreement, 0.0)
        self.assertLessEqual(info.disagreement, 1.0)

    def test_save_load_roundtrip(self) -> None:
        agent = _trained_agent()
        path = os.path.join("output", "_test_ensemble", "agent.pt")
        agent.save(path)
        reloaded = EnsembleDQNAgent.load(path)
        self.assertEqual(len(reloaded.online_networks), 3)
        obs = [0.4] * 16
        q1 = agent.q_ensemble(obs)
        q2 = reloaded.q_ensemble(obs)
        self.assertTrue(float((q1 - q2).abs().max()) < 1e-6)
        self.assertEqual(reloaded.obs_stats["count"], 80)

    def test_single_model_checkpoint_rejected(self) -> None:
        from rl.dqn_agent import DQNAgent

        path = os.path.join("output", "_test_ensemble", "single.pt")
        DQNAgent(DQNConfig(obs_dim=16, n_actions=11), device="cpu").save(path)
        with self.assertRaises(ValueError):
            EnsembleDQNAgent.load(path)

    def test_ood_score_rises_off_distribution(self) -> None:
        """OOD 评分必须在观测远离训练分布时显著升高。"""
        agent = _trained_agent()
        agent.ensemble_config.ood_warmup_steps = 10
        on_dist = agent._ood_score([0.5] * 16)
        off_dist = agent._ood_score([50.0] * 16)
        self.assertGreater(off_dist, on_dist)
        self.assertGreater(off_dist, 5.0)

    def test_bootstrap_prob_one_is_plain_ensemble(self) -> None:
        config = DQNConfig(obs_dim=16, n_actions=11, seed=1, learning_starts=5,
                           batch_size=8, buffer_capacity=200)
        agent = EnsembleDQNAgent(
            config, EnsembleConfig(ensemble_size=3, bootstrap_prob=1.0), device="cpu"
        )
        rng = random.Random(1)
        for _ in range(30):
            obs = [rng.random() for _ in range(16)]
            agent.store(obs, 0, 0.0, obs, False, [True] * 11)
            agent.env_steps += 1
            agent.maybe_train()
        self.assertIsNotNone(agent.last_loss)

    def test_invalid_ensemble_size(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig(ensemble_size=1).validate()
        with self.assertRaises(ValueError):
            EnsembleConfig(bootstrap_prob=0.0).validate()


class TestFallbackPolicy(unittest.TestCase):
    def _policy(self, **cfg_kwargs) -> UncertaintyAwarePolicy:
        agent = _trained_agent()
        config = FallbackConfig(**cfg_kwargs)
        return UncertaintyAwarePolicy(agent, config)

    def test_runs_episode(self) -> None:
        env = make_pomdp_env(preset="moderate")
        policy = self._policy()
        policy.reset()
        obs, _ = env.reset(seed=42)
        steps = 0
        while True:
            action, record = policy.select_action(obs, env)
            self.assertEqual(record.mode in (MODE_AI, MODE_SHIELD, MODE_FALLBACK), True)
            obs, _r, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        self.assertGreater(steps, 0)
        summary = policy.summary()
        self.assertEqual(summary["decision_steps"], steps)
        self.assertAlmostEqual(
            summary["ai_autonomy_rate"] + summary["shield_rate"] + summary["fallback_rate"],
            1.0,
            places=9,
        )

    def test_summary_rates_are_consistent(self) -> None:
        env = make_pomdp_env(preset="severe")
        policy = self._policy()
        policy.reset()
        obs, _ = env.reset(seed=7)
        while True:
            action, _ = policy.select_action(obs, env)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        summary = policy.summary()
        total = summary["decision_steps"]
        self.assertEqual(sum(summary["mode_counts"].values()), total)

    def test_loose_thresholds_give_full_autonomy(self) -> None:
        """阈值放到不可能触发时，必须全部是 AI 自主决策。"""
        policy = self._policy(
            q_std_threshold=1e9,
            disagreement_threshold=1e9,
            q_margin_threshold=0.0,
            ood_threshold=1e9,
            obs_quality_threshold=0.0,
        )
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action, record = policy.select_action(obs, env)
            self.assertEqual(record.mode, MODE_AI)
            self.assertEqual(record.triggered, ())
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        self.assertAlmostEqual(policy.summary()["ai_autonomy_rate"], 1.0)

    def test_tight_thresholds_trigger_high_risk(self) -> None:
        """阈值收紧后必须出现高风险状态，且 reason_code 会落到记录里。"""
        policy = self._policy(
            q_std_threshold=0.0,
            disagreement_threshold=0.0,
            q_margin_threshold=1e9,
            ood_threshold=0.0,
            obs_quality_threshold=1.0,
        )
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=42)
        reasons = set()
        for _ in range(20):
            action, record = policy.select_action(obs, env)
            if record.reason_code:
                reasons.add(record.reason_code)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        self.assertGreater(policy.summary()["high_risk_rate"], 0.0)
        self.assertTrue(reasons)

    def test_turning_off_all_signals_disables_fallback(self) -> None:
        policy = self._policy(
            use_ensemble_signal=False, use_ood_signal=False, use_observation_signal=False
        )
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=42)
        for _ in range(15):
            action, record = policy.select_action(obs, env)
            self.assertEqual(record.mode, MODE_AI)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    def test_fallback_rule_mode_hands_over(self) -> None:
        policy = self._policy(
            fallback_mode="fallback_rule",
            q_std_threshold=0.0,
            disagreement_threshold=0.0,
            q_margin_threshold=1e9,
            ood_threshold=0.0,
            obs_quality_threshold=1.0,
        )
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action, record = policy.select_action(obs, env)
            if record.triggered and record.rule_action is not None:
                self.assertEqual(record.mode, MODE_FALLBACK)
                self.assertEqual(action, record.rule_action)
                break
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        self.assertGreater(policy.summary()["fallback_rate"], 0.0)

    def test_shield_never_lowers_power(self) -> None:
        """护盾只允许抬升档位，绝不能把 AI 的动作调低。"""
        policy = self._policy(
            q_std_threshold=0.0, disagreement_threshold=0.0,
            q_margin_threshold=1e9, ood_threshold=0.0, obs_quality_threshold=1.0,
        )
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=11)
        for _ in range(25):
            action, record = policy.select_action(obs, env)
            self.assertGreaterEqual(action, record.ai_action)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    def test_energy_constraint_respected(self) -> None:
        policy = self._policy()
        env = make_pomdp_env(preset="severe")
        policy.reset()
        obs, _ = env.reset(seed=42)
        while True:
            action, _ = policy.select_action(obs, env)
            obs, _r, terminated, truncated, info = env.step(action)
            self.assertLessEqual(
                info["cumulative_energy_j"], env.sim.radar.energy_budget_j + 1e-6
            )
            if terminated or truncated:
                break

    def test_invalid_fallback_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FallbackConfig(fallback_mode="bogus").validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
