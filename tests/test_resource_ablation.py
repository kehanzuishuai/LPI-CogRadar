"""四组同结构消融的测试（大阶段二第二步）。

需要 torch：

    D:\\anaconda\\envs\\pytorch_env\\python.exe -m unittest tests.test_resource_ablation -v

覆盖用户点名的硬要求
--------------------
* **四组输入维度完全相同**，被消融的特征**置零**而不是删除；
* **冻结清单**逐项断言：改任何一项都会让消融失败；
* 新鲜度只用已到达年龄；不确定度只用协方差 + 来源一致性指示量；
  禁止 `truth_id` / 真实偏差 / 真实关联标签 / 未来消息 / 离线真值；
* **不得把来源一致性称为概率**；
* 四组同训练种子、同场景顺序、同预算；validation 用确定性 masked argmax；
  test 分区继续封存；
* mask 是承重机制：三个非法动作指标都要记录。
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from resource_management.closed_loop import NODE_LAYOUT  # noqa: E402
from resource_management.contract_v1 import verify_frozen  # noqa: E402
from resource_management.information_research import (  # noqa: E402
    ABLATION_ARMS, RESEARCH_HYPOTHESIS,
)
from resource_management.learning_protocol import (  # noqa: E402
    SealedTestSplitError, get_split,
)

from rl_resource.ablation import (  # noqa: E402
    FROZEN_BASELINE, assert_frozen, compare_arms, _arm_table,
    _hypothesis_verdict,
)
from rl_resource.env import (  # noqa: E402
    CentralizedResourceSchedulingEnv, EnvConfig,
)
from rl_resource.research_obs import (  # noqa: E402
    RESEARCH_OBS_SCHEMA, ResearchObservationEncoder, arm_from_value,
)
from rl_resource.train import TrainConfig, _episode_spec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(ROOT, "config", "learning_splits_v1.json")
NODE_IDS = tuple(sorted(NODE_LAYOUT))


def _working_action(mask):
    """会**真的形成航迹**的手工策略：优先 process，其次 sample，最后 share。

    ⚠️ 用"按 SAMPLE→PROCESS→SHARE 顺序取第一个合法动作"不行：
    `expose_all` 下 sample 在 96% 的 tick 上都合法，于是永远只采样、
    从不处理，**一条航迹也不会形成**——四组的非零槽位数就会完全相同，
    消融测试变成空转（这个坑踩过两次）。
    """
    from rl_resource.actions import (
        ACTION_IDLE, ACTION_PROCESS, ACTION_SAMPLE, ACTION_SHARE)

    action = [ACTION_IDLE] * len(mask)
    for index, row in enumerate(mask):
        for candidate in (ACTION_PROCESS, ACTION_SAMPLE, ACTION_SHARE):
            if row[candidate]:
                action[index] = candidate
                break
    return action


# ----------------------------------------------------------------------
# ① 四组同形 + 置零
# ----------------------------------------------------------------------


class TestSameShapeAndZeroing(unittest.TestCase):
    def test_all_arms_have_identical_dimension(self) -> None:
        dims = set()
        for arm in ABLATION_ARMS:
            encoder = ResearchObservationEncoder(NODE_IDS)
            dims.add(encoder.output_dim)
        self.assertEqual(len(dims), 1,
                         f"四组维度必须完全相同：{dims}")

    def test_dimension_matches_environment(self) -> None:
        encoder = ResearchObservationEncoder(NODE_IDS)
        for arm in ABLATION_ARMS:
            env = CentralizedResourceSchedulingEnv(EnvConfig(
                scenario="rm_train_base", seed=101, steps=6,
                arm=arm.value))
            obs, _info = env.reset()
            self.assertEqual(len(obs), encoder.output_dim, arm.value)

    def test_disabled_features_are_zero_not_removed(self) -> None:
        """关掉的特征必须是**置零**，槽位仍在（否则维度会变）。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=8,
            arm="main_baseline"))
        obs, info = env.reset()
        encoder = ResearchObservationEncoder(NODE_IDS)
        zeroed = encoder.zeroed_slot_mask("main_baseline")
        self.assertEqual(len(obs), len(zeroed))
        self.assertTrue(any(zeroed), "主基线组必须有关闭的槽位")
        # 跑几个 tick 产生航迹后，被关闭的槽位仍必须为 0
        for _ in range(6):
            obs, _reward, terminated, truncated, info = env.step(
                [0] * env.config.max_nodes)
            if terminated or truncated:
                break
            for index, is_zeroed in enumerate(zeroed):
                if is_zeroed:
                    self.assertEqual(obs[index], 0.0,
                                     f"槽位 {index} 已关闭却不是 0")

    def test_enabled_arm_actually_populates_the_slots(self) -> None:
        """开了新鲜度/不确定度的组，在**有航迹之后**槽位必须非零。

        否则"消融"只是名义上的：置零逻辑没错，但开着的组也没数据。
        ⚠️ 必须用**会真正采样/处理**的策略：全 idle 时永远没有航迹，
        四组的非零槽位数会完全相同（这个坑踩过）。
        """
        def nonzero(arm: str) -> int:
            env = CentralizedResourceSchedulingEnv(EnvConfig(
                scenario="rm_train_base", seed=101, steps=12, arm=arm))
            _obs, info = env.reset()
            best = 0
            for _ in range(12):
                obs, _r, terminated, truncated, info = env.step(
                    _working_action(info["mask"]))
                if terminated or truncated:
                    break
                best = max(best, sum(1 for value in obs if abs(value) > 1e-12))
            return best

        baseline = nonzero("main_baseline")
        freshness = nonzero("freshness_only")
        uncertainty = nonzero("uncertainty_only")
        both = nonzero("freshness_uncertainty")
        self.assertGreater(baseline, 0, "基线组也应有非零槽位（资源/队列计数）")
        self.assertGreater(freshness, baseline,
                           "开启新鲜度后非零槽位必须更多")
        self.assertGreater(uncertainty, baseline,
                           "开启不确定度后非零槽位必须更多")
        self.assertGreater(both, baseline)

    def test_zeroed_mask_is_arm_consistent(self) -> None:
        encoder = ResearchObservationEncoder(NODE_IDS)
        for arm in ABLATION_ARMS:
            resolved = arm_from_value(arm)
            flags = encoder.zeroed_slot_mask(arm)
            self.assertEqual(len(flags), encoder.output_dim)
            if resolved.uses_freshness and resolved.uses_uncertainty:
                self.assertFalse(any(flags),
                                 "全开组不应有关闭槽位")
        for arm in ABLATION_ARMS:
            self.assertEqual(arm_from_value(arm.value), arm)


# ----------------------------------------------------------------------
# ② 冻结清单
# ----------------------------------------------------------------------


class TestFrozenBaseline(unittest.TestCase):
    def test_frozen_defaults_are_self_consistent(self) -> None:
        frozen = FROZEN_BASELINE
        validation = get_split("validation", path=SPLIT_PATH)
        cfg = TrainConfig(
            scenarios=frozen.train_scenarios, seeds=frozen.train_seeds,
            episodes=frozen.episodes_per_update * frozen.updates,
            steps=frozen.steps, rollout_episodes=frozen.episodes_per_update,
            updates=frozen.updates, max_nodes=frozen.max_nodes,
            ppo=frozen.ppo(), policy=frozen.policy(obs_dim=0), arm="main_baseline",
            eval_scenarios=validation.scenarios,
            eval_seeds=validation.seeds, seed=frozen.train_seed)
        assert_frozen(cfg, frozen)          # 不抛异常即通过

    def test_any_change_is_rejected(self) -> None:
        """改任何一项冻结参数都必须报错——否则消融不可信。"""
        frozen = FROZEN_BASELINE
        mutations = {
            "learning_rate": {"ppo": FROZEN_BASELINE.ppo()},
            "entropy": {"ppo": FROZEN_BASELINE.ppo()},
        }
        for key, kwargs in mutations.items():
            ppo = FROZEN_BASELINE.ppo()
            if key == "learning_rate":
                ppo.learning_rate = 1e-3
            else:
                ppo.entropy_coef = 0.05
            cfg = TrainConfig(
                scenarios=frozen.train_scenarios, seeds=frozen.train_seeds,
                steps=frozen.steps, rollout_episodes=frozen.episodes_per_update,
                updates=frozen.updates, ppo=ppo,
                policy=frozen.policy(obs_dim=0), arm="main_baseline")
            with self.assertRaises(RuntimeError, msg=key):
                assert_frozen(cfg, frozen)

    def test_hidden_sizes_and_optimizer_are_frozen(self) -> None:
        self.assertEqual(tuple(FROZEN_BASELINE.hidden_sizes), (128, 128))
        self.assertEqual(FROZEN_BASELINE.optimizer, "adam")
        self.assertEqual(FROZEN_BASELINE.runtime_mode,
                         "plan_controlled_feedback")
        self.assertEqual(FROZEN_BASELINE.task_gating, "expose_all")
        self.assertTrue(FROZEN_BASELINE.test_sealed)
        self.assertEqual(FROZEN_BASELINE.eval_split, "validation")
        self.assertEqual(FROZEN_BASELINE.obs_schema, RESEARCH_OBS_SCHEMA)

    def test_resource_contract_v1_still_frozen(self) -> None:
        result = verify_frozen(strict=False)
        self.assertTrue(result["ok"], result["hint"])

    def test_test_split_still_sealed(self) -> None:
        with self.assertRaises(SealedTestSplitError):
            get_split("test", path=SPLIT_PATH)


# ----------------------------------------------------------------------
# ③ 四组等价性：同种子 / 同场景顺序 / 同预算
# ----------------------------------------------------------------------


class TestArmEquivalence(unittest.TestCase):
    def test_episode_order_is_a_pure_function_of_index(self) -> None:
        """episode 来源是纯函数 → 四组看到的序列逐项相同。"""
        frozen = FROZEN_BASELINE
        cfgs = {}
        for arm in ABLATION_ARMS:
            cfgs[arm.value] = TrainConfig(
                scenarios=frozen.train_scenarios, seeds=frozen.train_seeds,
                steps=frozen.steps, updates=frozen.updates,
                rollout_episodes=frozen.episodes_per_update, arm=arm.value
            ).resolved()
        sequences = {
            arm: [_episode_spec(cfg, index) for index in range(24)]
            for arm, cfg in cfgs.items()}
        first = next(iter(sequences.values()))
        for arm, sequence in sequences.items():
            self.assertEqual(sequence, first, f"{arm} 的 episode 序列不同")

    def test_arms_differ_only_in_the_arm_field(self) -> None:
        frozen = FROZEN_BASELINE
        obs_dims = set()
        for arm in ABLATION_ARMS:
            cfg = TrainConfig(
                scenarios=frozen.train_scenarios, seeds=frozen.train_seeds,
                steps=frozen.steps, updates=frozen.updates,
                rollout_episodes=frozen.episodes_per_update,
                ppo=frozen.ppo(), arm=arm.value).resolved()
            obs_dims.add(cfg.policy.obs_dim)
        self.assertEqual(len(obs_dims), 1,
                         f"四组的网络输入维度必须相同：{obs_dims}")

    def test_validation_is_deterministic_argmax(self) -> None:
        """评测路径必须用确定性动作：同权重两次评测完全一致。"""
        from rl_resource.policy import ActorCritic, PolicyConfig
        from rl_resource.train import evaluate
        frozen = FROZEN_BASELINE
        cfg = TrainConfig(steps=8, max_nodes=frozen.max_nodes,
                          arm="freshness_only",
                          eval_scenarios=("rm_validation_handover",),
                          eval_seeds=(211,)).resolved()
        torch.manual_seed(0)
        model = ActorCritic(PolicyConfig(obs_dim=cfg.policy.obs_dim,
                                         max_nodes=cfg.max_nodes))
        first = evaluate(model, cfg, torch.device("cpu"),
                         cfg.eval_scenarios, cfg.eval_seeds)
        second = evaluate(model, cfg, torch.device("cpu"),
                          cfg.eval_scenarios, cfg.eval_seeds)
        self.assertAlmostEqual(first["mean_return"], second["mean_return"],
                               places=9)
        self.assertAlmostEqual(first["mean_resource_cost"],
                               second["mean_resource_cost"], places=12)


# ----------------------------------------------------------------------
# ④ 信息纪律
# ----------------------------------------------------------------------


class TestInformationDiscipline(unittest.TestCase):
    def test_research_obs_reads_only_observation_and_queue(self) -> None:
        path = os.path.join(ROOT, "rl_resource", "research_obs.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        for forbidden in ("engine", "sensor", "fusion", "communication",
                          "torch"):
            self.assertEqual(
                [name for name in modules
                 if name == forbidden or name.startswith(forbidden + ".")],
                [], f"研究观测不得依赖 {forbidden}")

    def test_no_forbidden_field_names_in_features(self) -> None:
        encoder = ResearchObservationEncoder(NODE_IDS)
        names = " ".join(encoder.feature_names()).lower()
        for forbidden in ("truth", "bias", "false_alarm", "is_false",
                          "association_label", "future"):
            self.assertNotIn(forbidden, names,
                             f"特征名不得包含 {forbidden}")

    def test_consistency_is_never_called_a_probability(self) -> None:
        path = os.path.join(ROOT, "rl_resource", "research_obs.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        # 允许出现"不得称为概率"这类**否定**表述，但不允许把指示量**命名**成概率
        self.assertNotIn("source_consistency_probability", source)
        self.assertNotIn("出错概率，", source)
        self.assertIn("不得", source)

    def test_offline_truth_is_not_in_the_observation(self) -> None:
        """观测里不得出现离线误差表的内容。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_validation_sensor_bias", seed=211, steps=12,
            arm="freshness_uncertainty"))
        obs, info = env.reset()
        for _ in range(12):
            # 观测必须**归一化**：位置/速度是有符号特征，允许负值，
            # 但幅度必须 ≤1。若把米/秒级的真值量直接塞进来就会超界。
            self.assertTrue(all(abs(float(value)) <= 1.0 + 1e-9
                                for value in obs),
                            "观测必须落在 [-1,1]，不得混入未归一化的真值量")
            obs, _r, terminated, truncated, info = env.step(
                _working_action(info["mask"]))
            if terminated or truncated:
                break
        final = env.finalize()
        # 离线误差表**必须存在**（证明它确实在用真值对齐），
        # 但它只挂在 result 上、不进观测。
        self.assertTrue(final["result"].estimate_error_samples,
                        "全 idle 时不会有航迹也就没有误差样本；本测试用贪心策略")
        self.assertTrue(all(
            sample["provenance"] == "offline_truth_nearest_track"
            for sample in final["result"].estimate_error_samples))


# ----------------------------------------------------------------------
# ⑤ 判据与报告口径
# ----------------------------------------------------------------------


class TestVerdictAndReporting(unittest.TestCase):
    def _table(self, arm: str, worst: float, resource: float,
               ret: float) -> dict:
        return {"arm": arm, "worst_completion_rate": worst,
                "resource_consumption": resource, "validation_return": ret}

    def test_verdict_requires_both_conditions(self) -> None:
        tables = [
            self._table("main_baseline", 0.50, 0.30, -10.0),
            self._table("freshness_only", 0.52, 0.28, -9.0),
            self._table("uncertainty_only", 0.51, 0.29, -9.5),
            self._table("freshness_uncertainty", 0.45, 0.31, -11.0),
        ]
        verdict = _hypothesis_verdict(tables, compare_arms(tables))
        self.assertIn("仍未被支持", verdict["verdict"])
        self.assertIn("不下降=False", verdict["reason"])

    def test_verdict_records_support_when_both_hold(self) -> None:
        tables = [
            self._table("main_baseline", 0.50, 0.30, -10.0),
            self._table("freshness_only", 0.52, 0.29, -9.0),
            self._table("uncertainty_only", 0.51, 0.29, -9.5),
            self._table("freshness_uncertainty", 0.53, 0.29, -8.0),
        ]
        verdict = _hypothesis_verdict(tables, compare_arms(tables))
        self.assertIn("未发现反例", verdict["verdict"])

    def test_comparison_uses_main_baseline_as_reference(self) -> None:
        tables = [
            self._table("main_baseline", 0.50, 0.30, -10.0),
            self._table("freshness_only", 0.52, 0.28, -9.0),
        ]
        comparison = compare_arms(tables)
        self.assertEqual(comparison["reference"], "main_baseline")
        delta = comparison["deltas_vs_reference"]["freshness_only"]
        self.assertAlmostEqual(delta["validation_return"], 1.0)
        self.assertAlmostEqual(delta["resource_consumption"], -0.02)

    def test_arm_table_records_three_invalid_metrics(self) -> None:
        summary = {
            "tag": "main_baseline", "selected_update": 3,
            "validation_of_selected": {
                "mean_return": -5.0, "rows": [], "legal_action_rates": {},
                "illegal_action_rate": 0.0,
                "executor_rejection_rate": 0.0,
                "conservation_all_ok": True,
                "max_cost_reconciliation_error": 0.0, "plan_fatal_ticks": 0},
            "validation_without_mask": {
                "illegal_action_rate": 1.0, "illegal_probability_mass": 0.7},
        }
        table = _arm_table(summary)
        self.assertEqual(table["masked_invalid_rate"], 0.0)
        self.assertEqual(table["unmasked_argmax_invalid_rate"], 1.0)
        self.assertEqual(table["invalid_probability_mass"], 0.7)
        for key in ("masked_invalid_rate_definition",
                    "unmasked_argmax_invalid_rate_definition",
                    "invalid_probability_mass_definition"):
            self.assertIn(key, table)

    def test_hypothesis_text_is_available(self) -> None:
        self.assertIn("最差", RESEARCH_HYPOTHESIS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
