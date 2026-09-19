"""集中式学习资源调度基线（`rl_resource`）的测试。

需要 torch（用 `pytorch_env` 解释器运行）：

    D:\\anaconda\\envs\\pytorch_env\\python.exe -m unittest tests.test_resource_rl -v

覆盖用户点名的硬要求
--------------------
* **不改冻结契约**：`verify_frozen()` 必须仍然通过（resource-contract-v1）；
* **训练跑真闭环**：环境固定 `plan_controlled_feedback` + `expose_all` 门控；
* **智能体不越权**：`rl_resource` 不 import `engine`/`sensor`/`fusion`/`communication`，
  也不直接改传感器/融合/总线（副作只能经 `RuntimeExecutor`）；
* **结构化动作 + 合法 mask**：逐节点分类、mask 判据可测、动作→标准 `ExecutionPlan`；
* **数据纪律**：test 分区封存；train/validation 不重叠；
* **守恒与对账**：`Σ_t c_t == 实测资源消耗`；终止/截断的 bootstrap 语义正确。
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from resource_management.contract_v1 import verify_frozen  # noqa: E402
from resource_management.learning_protocol import (  # noqa: E402
    SealedTestSplitError, get_split, split_digest, verify_split_digest,
)
from resource_management.tasks import QueueTaskKind, TaskStatus  # noqa: E402
from resource_management.units import ResourceUnit, TaskKind  # noqa: E402

from rl_resource.actions import (  # noqa: E402
    ACTION_IDLE, ACTION_NAMES, ACTION_PROCESS, ACTION_SAMPLE, ACTION_SHARE,
    IDLE_ONLY_ROW, N_ACTIONS, build_action_context, build_plan, pad_mask,
)
from rl_resource.env import (  # noqa: E402
    CentralizedResourceSchedulingEnv, EnvConfig, REWARD_VERSION,
)
from rl_resource.obs import (  # noqa: E402
    DEFAULT_MAX_NODES, encode, feature_metadata, observation_dim,
)
from rl_resource.policy import ActorCritic, PolicyConfig  # noqa: E402
from rl_resource.ppo import PPOConfig, RolloutBuffer, ppo_update  # noqa: E402
from rl_resource.scenarios import (  # noqa: E402
    REMOTE_NODE_ID, SCENARIO_MAPPING_VERSION, closed_loop_kwargs,
    deadline_offsets, describe_mapping, load_scenarios, mapping_digest,
    scaled_budgets, scenario_names_for_split, sensor_bias,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(ROOT, "config", "learning_splits_v1.json")


def _module_imports(relative: str) -> list:
    path = os.path.join(ROOT, relative)
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def _greedy_action(mask):
    action = [ACTION_IDLE] * len(mask)
    for index, row in enumerate(mask):
        for candidate in (ACTION_SAMPLE, ACTION_PROCESS, ACTION_SHARE):
            if row[candidate]:
                action[index] = candidate
                break
    return action


# ----------------------------------------------------------------------
# ① 冻结契约与数据纪律
# ----------------------------------------------------------------------


class TestFrozenContractAndSplits(unittest.TestCase):
    def test_resource_contract_v1_still_frozen(self) -> None:
        """用户点名要求：**不得修改**已冻结的 resource-contract-v1。"""
        result = verify_frozen(strict=False)
        self.assertTrue(result["ok"], result["hint"])
        self.assertEqual(result["recorded"], result["current"])

    def test_split_registry_digest_matches(self) -> None:
        self.assertEqual(verify_split_digest(SPLIT_PATH, os.path.join(
            ROOT, "config", "learning_splits_v1.sha256")), split_digest(SPLIT_PATH))

    def test_test_split_is_sealed(self) -> None:
        with self.assertRaises(SealedTestSplitError):
            get_split("test", path=SPLIT_PATH)
        with self.assertRaises(SealedTestSplitError):
            scenario_names_for_split("test", path=SPLIT_PATH)

    def test_train_and_validation_do_not_overlap(self) -> None:
        train = get_split("train", path=SPLIT_PATH)
        validation = get_split("validation", path=SPLIT_PATH)
        self.assertFalse(set(train.seeds) & set(validation.seeds))
        self.assertFalse(set(train.scenarios) & set(validation.scenarios))


# ----------------------------------------------------------------------
# ② 信息边界：智能体不越权
# ----------------------------------------------------------------------


class TestInformationBoundary(unittest.TestCase):
    def test_learning_modules_do_not_import_truth_layers(self) -> None:
        """`rl_resource` 的观测/动作/策略不得依赖真值层。"""
        for relative in ("rl_resource/obs.py", "rl_resource/actions.py"):
            modules = _module_imports(relative)
            for forbidden in ("engine", "sensor", "fusion", "communication",
                              "torch"):
                self.assertEqual(
                    [name for name in modules
                     if name == forbidden
                     or name.startswith(forbidden + ".")],
                    [], f"{relative} 不得依赖 {forbidden}")

    def test_env_only_reaches_the_world_through_the_closed_loop(self) -> None:
        """`env.py` 不进真值层；副作只能经闭环提供的运行时执行器。"""
        modules = _module_imports("rl_resource/env.py")
        for forbidden in ("engine", "sensor", "fusion", "communication"):
            self.assertEqual(
                [name for name in modules
                 if name == forbidden or name.startswith(forbidden + ".")],
                [], f"env.py 不得依赖 {forbidden}")
        self.assertIn("resource_management.closed_loop", modules)

    def test_agent_cannot_act_outside_the_plan(self) -> None:
        """空闲动作**不得**产生任何测量、融合或消息（不越权的运行时可测形式）。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=6))
        env.reset()
        idle = [ACTION_IDLE] * env.config.max_nodes
        for _ in range(6):
            _obs, _reward, terminated, truncated, info = env.step(idle)
            if terminated or truncated:
                break
        final = env.finalize()
        runtime = final["result"].metrics["runtime_feedback"]
        self.assertEqual(runtime["n_sensor_measurements"], 0)
        self.assertEqual(runtime["n_fusion_updates"], 0)
        self.assertEqual(runtime["n_messages_sent"], 0)
        self.assertEqual(runtime["sensor_scans_by_node"],
                         {node_id: 0 for node_id in env.node_ids})
        self.assertEqual(runtime["truth_payload_violations"], 0)

    def test_observation_has_no_truth_channel(self) -> None:
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=6))
        obs, info = env.reset()
        self.assertEqual(len(obs), observation_dim(env.config.max_nodes))
        metadata = feature_metadata(env.config.max_nodes)
        self.assertEqual(len(metadata), len(obs))
        for entry in metadata:
            self.assertNotIn("truth", entry["name"])


# ----------------------------------------------------------------------
# ③ 结构化动作空间与合法 mask
# ----------------------------------------------------------------------


class TestStructuredActionSpace(unittest.TestCase):
    def test_pad_mask_pads_with_idle_only(self) -> None:
        rows = pad_mask([[True, True, False, False]], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], [True, True, False, False])
        for row in rows[1:]:
            self.assertEqual(row, list(IDLE_ONLY_ROW))

    def test_idle_is_always_legal(self) -> None:
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=4))
        _obs, info = env.reset()
        for row in info["mask"]:
            self.assertTrue(row[ACTION_IDLE])
        self.assertEqual(len(info["mask"][0]), N_ACTIONS)

    def test_mask_requires_feasibility_not_just_a_queued_task(self) -> None:
        """只有"真的有数据可处理/可发送"时才允许 process / share。

        这条防的是"队列里有任务就放行"这种偷懒判据——那会让策略选到
        必然空转的动作，白烧资源。
        """
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=8))
        _obs, info = env.reset()
        process_seen = share_seen = False
        for _ in range(8):
            mask = info["mask"]
            context = env.action_context()
            for index, node_id in enumerate(env.node_ids):
                if mask[index][ACTION_PROCESS]:
                    process_seen = True
                    self.assertTrue(context.processable[node_id],
                                    "process 合法时必须真的有数据可处理")
                if mask[index][ACTION_SHARE]:
                    share_seen = True
                    self.assertTrue(context.shareable[node_id],
                                    "share 合法时 outbox 必须非空")
            _obs, _reward, terminated, truncated, info = env.step(
                _greedy_action(mask))
            if terminated or truncated:
                break
        self.assertTrue(process_seen or share_seen,
                        "整个 episode 里 process/share 从未合法，步数可能不足")

    def test_action_translates_to_standard_execution_plan(self) -> None:
        """动作必须翻译成**标准** `ExecutionPlan`（含 start_s == now）。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=6))
        _obs, info = env.reset()
        context = env.action_context()
        now = 1.0
        n_nodes = len(env.node_ids)
        built = build_plan(_greedy_action(info["mask"])[:n_nodes], context, now,
                           plan_id="test-plan", mask=info["mask"][:n_nodes])
        self.assertIsNotNone(built.plan)
        self.assertEqual(built.plan.plan_id, "test-plan")
        self.assertEqual(built.plan.submit_time_s, now)
        # 执行器层用的是 TaskKind（sample/process/share），队列层是 QueueTaskKind
        # （predefined_sample/process/share）——两者是**不同枚举**，断言要用前者。
        allowed = {TaskKind.SAMPLE, TaskKind.PROCESS, TaskKind.SHARE}
        for task in built.plan.tasks:
            self.assertEqual(task.start_s, now)
            self.assertIn(task.kind, allowed,
                          "不得引入 sample/process/share 之外的新动作")
            self.assertGreaterEqual(task.effective_duration_s(), 0.0)

    def test_illegal_action_is_recorded_not_silently_dropped(self) -> None:
        """非法选择必须**被记录**并交给执行器，而不是被悄悄丢弃。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=6))
        _obs, info = env.reset()
        # 构造一个必定非法的动作：非 idle 但 mask 不允许
        illegal = [ACTION_SHARE] * env.config.max_nodes
        mask = info["mask"]
        if mask[0][ACTION_SHARE]:
            self.skipTest("该 tick 的 share 恰好合法，换个场景再测")
        _obs, _reward, _t, _tr, out = env.step(illegal)
        self.assertTrue(out["note"], "非法选择必须留下说明")
        final = env.finalize()
        self.assertGreater(final["n_unmasked_illegal_actions"], 0)
        self.assertGreater(final["unmasked_illegal_action_rate"], 0.0)


# ----------------------------------------------------------------------
# ④ 真闭环、守恒与对账
# ----------------------------------------------------------------------


class TestClosedLoopEnvironment(unittest.TestCase):
    def _run(self, scenario: str, steps: int = 12, seed: int = 101):
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario=scenario, seed=seed, steps=steps))
        _obs, info = env.reset()
        total = 0.0
        done = False
        while not done:
            _obs, reward, terminated, truncated, info = env.step(
                _greedy_action(info["mask"]))
            total += reward
            done = terminated or truncated
        return env, env.finalize(), total, info

    def test_episode_runs_to_task_horizon(self) -> None:
        env, final, _total, info = self._run("rm_train_base", steps=12)
        self.assertEqual(info["termination_reason"], "task_horizon")
        self.assertEqual(info["step"], 12)
        self.assertEqual(final["trace"][-1]["step"], 12)

    def test_cost_reconciles_with_measured_consumption(self) -> None:
        """`Σ_t c_t` 必须等于账本重算的资源消耗（协议 §2.3 的恒等式）。"""
        for scenario in ("rm_train_base", "rm_train_constrained_comm",
                         "rm_validation_node_outage"):
            _env, final, _total, _info = self._run(scenario)
            self.assertLess(final["cost_reconciliation_error"], 1e-6,
                            f"{scenario} 的代价对账失败")
            self.assertGreater(final["cumulative_resource_cost"], 0.0)

    def test_resource_conservation_holds(self) -> None:
        for scenario in ("rm_train_base", "rm_train_high_arrival"):
            _env, final, _total, _info = self._run(scenario)
            self.assertTrue(final["conservation_ok"], scenario)
            self.assertTrue(
                final["result"].metrics["conservation_all"], scenario)

    def test_no_duplicate_execution_and_no_plan_fatal(self) -> None:
        _env, final, _total, _info = self._run("rm_train_base")
        self.assertEqual(final["plan_fatal_ticks"], 0)
        runtime = final["result"].metrics["runtime_feedback"]
        self.assertEqual(runtime["n_runtime_tasks"],
                         runtime["n_unique_task_keys"])
        self.assertEqual(runtime["n_duplicate_runtime_tasks"], 0)

    def test_same_seed_is_reproducible(self) -> None:
        first = self._run("rm_train_base", steps=8, seed=107)[1]
        second = self._run("rm_train_base", steps=8, seed=107)[1]
        self.assertAlmostEqual(first["cumulative_resource_cost"],
                               second["cumulative_resource_cost"], places=12)
        self.assertEqual(first["n_planned"], second["n_planned"])
        self.assertEqual(len(first["trace"]), len(second["trace"]))

    def test_bootstrap_semantics(self) -> None:
        """`info["bootstrap_allowed"]` 必须恒等于 `not terminated`。"""
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=6))
        _obs, info = env.reset()
        done = False
        while not done:
            _obs, _reward, terminated, truncated, info = env.step(
                _greedy_action(info["mask"]))
            self.assertEqual(info["bootstrap_allowed"], not terminated)
            done = terminated or truncated

    def test_external_step_limit_is_truncation_with_bootstrap(self) -> None:
        env = CentralizedResourceSchedulingEnv(EnvConfig(
            scenario="rm_train_base", seed=101, steps=10,
            external_step_limit=4))
        _obs, info = env.reset()
        done = False
        while not done:
            _obs, _reward, terminated, truncated, info = env.step(
                _greedy_action(info["mask"]))
            done = terminated or truncated
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["termination_reason"], "external_step_limit")
        self.assertTrue(info["bootstrap_allowed"])
        # **外部截断不补计终端惩罚**（协议 §2.3）：这是本条的要点。
        # 第一版把断言写反了（要求出现 terminal_supplement），
        # 那恰好是协议禁止的行为。
        self.assertNotIn("terminal_supplement", info["reward_components"],
                         "外部截断不得加终端补计")


class TestScenarioMapping(unittest.TestCase):
    def test_budget_multiplier_scales_capacities(self) -> None:
        base = scaled_budgets(1.0)
        half = scaled_budgets(0.5)
        for node_id, budgets in base.items():
            for unit, value in budgets.items():
                self.assertAlmostEqual(half[node_id][unit], value * 0.5)
        # 不得修改默认值本身
        from resource_management.closed_loop import DEFAULT_NODE_BUDGETS
        self.assertAlmostEqual(
            float(DEFAULT_NODE_BUDGETS["NODE_A"][ResourceUnit.SAMPLE_SLOT]),
            30.0)

    def test_load_multiplier_scales_deadline_offsets(self) -> None:
        base = deadline_offsets(1.0)
        high = deadline_offsets(1.4)
        for kind, value in base.items():
            self.assertAlmostEqual(high[kind], value / 1.4)
        self.assertAlmostEqual(base[QueueTaskKind.PREDEFINED_SAMPLE], 6.0)

    def test_outage_window_targets_the_remote_node(self) -> None:
        scenarios = load_scenarios(SPLIT_PATH)
        kwargs = closed_loop_kwargs(scenarios["rm_validation_node_outage"],
                                    path=SPLIT_PATH)
        windows = kwargs["mechanisms"]["unavailable_windows"]
        self.assertEqual(list(windows), [REMOTE_NODE_ID])

    def test_sensor_bias_only_pollutes_measurements(self) -> None:
        bias = sensor_bias(1.0)
        self.assertEqual(list(bias), [REMOTE_NODE_ID])
        self.assertIn("noise_underreport_factor", bias[REMOTE_NODE_ID])
        self.assertLess(bias[REMOTE_NODE_ID]["noise_underreport_factor"], 1.0)
        self.assertEqual(sensor_bias(0.0), {})

    def test_mapping_is_described_and_digested(self) -> None:
        description = describe_mapping(SPLIT_PATH)
        self.assertEqual(description["mapping_version"],
                         SCENARIO_MAPPING_VERSION)
        self.assertTrue(description["limitations"])
        self.assertEqual(len(mapping_digest(SPLIT_PATH)), 32)


# ----------------------------------------------------------------------
# ⑤ 策略、PPO 与保存/恢复
# ----------------------------------------------------------------------


class TestPolicyAndPPO(unittest.TestCase):
    def test_policy_shapes_and_mask_respected(self) -> None:
        config = PolicyConfig(obs_dim=observation_dim(2), max_nodes=2)
        model = ActorCritic(config)
        obs = torch.zeros(3, config.obs_dim)
        mask = torch.tensor([[[True, True, False, False],
                              [True, False, False, False]]] * 3)
        out = model.act(obs, mask)
        self.assertEqual(tuple(out["action"].shape), (3, 2))
        for index in range(3):
            self.assertIn(int(out["action"][index][0]), (0, 1))
            self.assertEqual(int(out["action"][index][1]), 0)

    def test_deterministic_act_is_repeatable(self) -> None:
        config = PolicyConfig(obs_dim=observation_dim(2), max_nodes=2)
        model = ActorCritic(config)
        obs = torch.zeros(1, config.obs_dim)
        mask = torch.ones(1, 2, N_ACTIONS, dtype=torch.bool)
        first = model.act(obs, mask, deterministic=True)["action"]
        second = model.act(obs, mask, deterministic=True)["action"]
        self.assertTrue(torch.equal(first, second))

    def test_save_and_restore_round_trip(self) -> None:
        config = PolicyConfig(obs_dim=observation_dim(2), max_nodes=2)
        model = ActorCritic(config)
        obs = torch.zeros(1, config.obs_dim)
        mask = torch.ones(1, 2, N_ACTIONS, dtype=torch.bool)
        before = model.act(obs, mask, deterministic=True)["action"].tolist()
        path = os.path.join(ROOT, "output", "_rl_resource_test", "policy.pt")
        model.save(path, extra={"note": "unit-test"})
        restored, extra = ActorCritic.load(path)
        after = restored.act(obs, mask, deterministic=True)["action"].tolist()
        self.assertEqual(before, after)
        self.assertEqual(extra.get("note"), "unit-test")
        os.remove(path)

    def test_gae_cuts_bootstrap_on_termination_but_not_on_truncation(self) -> None:
        """截断保留 bootstrap、终止切断 bootstrap——两者**不能合并成 done**。"""
        def build(terminated: bool, truncated: bool) -> RolloutBuffer:
            buffer = RolloutBuffer()
            buffer.add(obs=[0.0], mask=[[True, False, False, False]],
                       action=[0], log_prob=0.0, value=0.0, reward=1.0,
                       terminated=terminated, truncated=truncated,
                       next_value=5.0)
            return buffer

        gamma, lam = 0.9, 1.0
        terminated_buffer = build(True, False)
        truncated_buffer = build(False, True)
        self.assertAlmostEqual(
            terminated_buffer.compute_gae(gamma, lam)[0][0], 1.0)
        self.assertAlmostEqual(
            truncated_buffer.compute_gae(gamma, lam)[0][0], 1.0 + gamma * 5.0)

    def test_ppo_update_produces_finite_stats(self) -> None:
        config = PolicyConfig(obs_dim=observation_dim(2), max_nodes=2)
        model = ActorCritic(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        buffer = RolloutBuffer()
        for step in range(8):
            buffer.add(obs=[0.1 * step] * config.obs_dim,
                       mask=[[True, True, False, False]] * 2,
                       action=[ACTION_SAMPLE, ACTION_IDLE],
                       log_prob=-1.0, value=0.0, reward=-0.5,
                       terminated=step == 7, truncated=False, next_value=0.0)
        stats = ppo_update(model, optimizer, buffer, PPOConfig(n_epochs=2,
                                                              batch_size=4),
                           torch.device("cpu"))
        self.assertGreater(stats.n_updates, 0)
        for key, value in stats.to_dict().items():
            if isinstance(value, float):
                self.assertTrue(value == value, f"{key} 是 NaN")
                self.assertLess(abs(value), 1e6, f"{key} 爆炸：{value}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
