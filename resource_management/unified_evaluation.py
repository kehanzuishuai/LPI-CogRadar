"""大阶段二第三步：七方法统一公平评测与结论冻结。

此模块不是把历史 CSV 拼在一起。每个 episode 都在同一条
``FeedbackLoopDriver`` 真闭环上运行：相同场景映射、预算、候选任务、
通信、中央可见观测和 ``RuntimeExecutor``。规则/优化方法通过同一调度器
接口产生计划；PPO 只把相同的中央观测和队列编码为动作，再提交标准计划。

测试分区默认拒绝读取；调用者必须显式传 ``release_test=True``。本模块从不
依据测试结果修改 checkpoint、超参、动作 mask 或场景。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from resource_management.closed_loop import (
    FeedbackLoopDriver, RuntimeExecutor, TASK_GATING_EXPOSE_ALL, _build_world,
)
from resource_management.contract_v1 import CONTRACT_VERSION, contract_digest, verify_frozen
from resource_management.learning_protocol import get_split, split_digest
from resource_management.scheduling import SchedulerPolicy
from rl_resource.actions import build_action_context, build_plan, pad_mask
from rl_resource.research_obs import ResearchObservationEncoder
from rl_resource.scenarios import closed_loop_kwargs, load_scenarios, mapping_digest

METHODS: Tuple[str, ...] = (
    "round_robin", "edf", "rule", "enumeration", "rolling_horizon",
    "ppo_baseline", "ppo_freshness_uncertainty",
)
SCHEDULER_METHODS = frozenset(METHODS[:5])
PPO_METHODS = frozenset(METHODS[5:])
PPO_ARMS = {
    "ppo_baseline": "main_baseline",
    "ppo_freshness_uncertainty": "freshness_uncertainty",
}
DEFAULT_CHECKPOINTS = {
    "ppo_baseline": os.path.join("output", "rl_resource", "ablation", "main_baseline", "policy.pt"),
    "ppo_freshness_uncertainty": os.path.join("output", "rl_resource", "ablation", "freshness_uncertainty", "policy.pt"),
}
METRIC_KEYS: Tuple[str, ...] = (
    "service_completion", "task_timeliness", "estimate_quality",
    "resource_consumption", "communication_overhead", "compute_time_s",
    "completion_rate", "mean_waiting_s", "max_waiting_s", "n_expired",
    "executor_rejection_rate", "resource_violation_rate",
    "masked_invalid_action_rate", "unmasked_argmax_invalid_rate",
    "invalid_probability_mass",
)


@dataclass(frozen=True)
class FrozenEvaluation:
    """评测前已固定的、不可由 test 结果反推修改的输入。"""

    steps: int = 24
    task_gating: str = TASK_GATING_EXPOSE_ALL
    runtime_mode: str = "plan_controlled_feedback"
    reference_method: str = "rule"
    confidence_level: float = 0.95
    checkpoints: Mapping[str, str] = None  # type: ignore[assignment]

    def resolved_checkpoints(self) -> Dict[str, str]:
        return dict(DEFAULT_CHECKPOINTS if self.checkpoints is None else self.checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "methods": list(METHODS), "steps": self.steps,
            "task_gating": self.task_gating, "runtime_mode": self.runtime_mode,
            "reference_method": self.reference_method,
            "confidence_level": self.confidence_level,
            "checkpoints": {key: os.path.abspath(value)
                            for key, value in self.resolved_checkpoints().items()},
            "mask_policy": "PPO deployment uses legal action mask; unmasked diagnostics are reported separately.",
            "selection_policy": "No checkpoint, hyperparameter, reward, seed, or mask changes after this manifest is created.",
        }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(frozen: FrozenEvaluation) -> Dict[str, Any]:
    """返回可在 test 前保存的评测输入指纹；缺资产立即失败。"""
    if not verify_frozen(strict=False)["ok"]:
        raise RuntimeError("resource-contract-v1 摘要不匹配，拒绝开始统一评测")
    checkpoints = frozen.resolved_checkpoints()
    if set(checkpoints) != PPO_METHODS:
        raise ValueError("冻结清单必须且只能包含两个 PPO 方法的 checkpoint")
    for method, path in checkpoints.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{method} checkpoint 不存在：{path}")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_files = (
        "resource_management/unified_evaluation.py",
        "resource_management/closed_loop.py",
        "resource_management/scheduling.py",
        "rl_resource/actions.py", "rl_resource/research_obs.py",
        "rl_resource/scenarios.py", "config/learning_splits_v1.json",
    )
    return {
        "freeze_version": "unified-resource-evaluation-v1",
        "frozen": frozen.to_dict(),
        "resource_contract": {"version": CONTRACT_VERSION, "digest": contract_digest()},
        "learning_split_digest": split_digest(),
        "scenario_mapping_digest": mapping_digest(),
        "checkpoints": {method: {"path": os.path.abspath(path), "sha256": _sha256(path)}
                        for method, path in checkpoints.items()},
        "source_sha256": {relative: _sha256(os.path.join(root, relative))
                          for relative in source_files},
        "training_randomness": {
            "n_frozen_checkpoints_per_method": 1,
            "estimable": False,
            "reason": "当前每种 PPO 方法只存在一个冻结 checkpoint；用户禁止重训，不能伪造多训练种子方差。",
        },
    }


def _student_t_975(n: int) -> Optional[float]:
    # 双侧 95% Student-t 临界值；n=1 没有样本方差，故 CI 不可估。
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
             12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
             17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
             22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
             27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}
    return table.get(n, 1.96 if n > 30 else None)


def summary(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": 1, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    half = _student_t_975(n) * std / math.sqrt(n)  # type: ignore[operator]
    return {"n": n, "mean": mean, "std": std,
            "ci95_low": mean - half, "ci95_high": mean + half}


def _build_driver(method: str, scenario: str, seed: int, frozen: FrozenEvaluation) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"未知方法 {method!r}")
    spec = load_scenarios()[scenario]
    kwargs = closed_loop_kwargs(spec)
    policy = SchedulerPolicy(method) if method in SCHEDULER_METHODS else SchedulerPolicy.RULE
    world = _build_world(policy=policy, seed=seed, steps=frozen.steps,
                         mechanisms=kwargs["mechanisms"], node_budgets=kwargs["node_budgets"],
                         runtime_mode=frozen.runtime_mode)
    runtime = RuntimeExecutor(world["executor"], world["sim"].scene, world["suite"],
                              world["centers"], world["bus"], world["sensor_positions"],
                              world["own_sensor"])
    driver = FeedbackLoopDriver(
        policy=policy, steps=frozen.steps, unavailable=world["unavailable"],
        sim=world["sim"], clock=world["clock"], ledger=world["ledger"],
        executor=world["executor"], queue=world["queue"], scheduler=world["scheduler"],
        runtime=runtime, result=world["result"], keep_decisions=True,
        starved_ids=world["starved_ids"], task_gating=frozen.task_gating,
        deadline_offsets=kwargs["deadline_offsets"])
    return {**world, "runtime": runtime, "driver": driver, "scenario_spec": spec}


def _load_ppo(method: str, frozen: FrozenEvaluation) -> Any:
    # Keep resource-management's static dependency boundary free of PyTorch.
    # The optional evaluator loads it only when a PPO episode actually runs.
    torch = importlib.import_module("torch")
    from rl_resource.policy import ActorCritic
    model, extra = ActorCritic.load(frozen.resolved_checkpoints()[method], map_location="cpu")
    model.eval()
    return model, extra, torch


def _ppo_tick(driver: FeedbackLoopDriver, runtime: RuntimeExecutor, model: Any,
              torch: Any, arm: str, encoder: Optional[ResearchObservationEncoder],
              inference: Dict[str, float]) -> None:
    central = driver.begin()
    context = build_action_context(
        central, driver.queue, driver.current_time_s,
        {node: runtime.has_processable(node, driver.current_time_s) for node in runtime.centers},
        {node: runtime.has_shareable(node) for node in runtime.centers},
    )
    mask = pad_mask(context.mask(), model.config.max_nodes)
    if encoder is None:
        raise RuntimeError("统一评测的 PPO checkpoint 必须使用研究观测编码器")
    values = encoder.encode(central, driver.queue, driver.current_time_s,
                            driver.steps, driver.step_index, arm)
    if len(values) != model.config.obs_dim:
        raise RuntimeError(f"{arm} checkpoint 输入维度 {model.config.obs_dim} 与评测编码 {len(values)} 不一致")
    started = time.perf_counter()
    observation = torch.tensor([values], dtype=torch.float32)
    mask_tensor = torch.tensor([mask], dtype=torch.bool)
    with torch.no_grad():
        logits, _value = model.forward(observation)
        unmasked = torch.argmax(logits[0], dim=-1).tolist()
        probabilities = torch.softmax(logits[0], dim=-1)
        masked_logits = model.masked_logits(logits, mask_tensor)
        actions = torch.argmax(masked_logits[0], dim=-1).tolist()
    inference["seconds"] += time.perf_counter() - started
    n_real = len(context.node_ids)
    for index in range(n_real):
        inference["n_node_actions"] += 1
        if not mask[index][int(actions[index])]:
            inference["masked_invalid"] += 1
        if not mask[index][int(unmasked[index])]:
            inference["unmasked_invalid"] += 1
        inference["invalid_probability_mass"] += sum(
            float(probabilities[index, action]) for action in range(len(mask[index]))
            if not mask[index][action])
    built = build_plan(actions[:n_real], context, driver.current_time_s,
                       plan_id=f"{arm}-unified-{driver.step_index:04d}", mask=mask[:n_real])
    driver.commit(built.plan)


def evaluate_episode(method: str, scenario: str, seed: int,
                     frozen: FrozenEvaluation) -> Dict[str, Any]:
    """运行一个格子；所有方法共享同一个世界构造及任务门控。"""
    world = _build_driver(method, scenario, seed, frozen)
    driver, runtime = world["driver"], world["runtime"]
    inference: Dict[str, float] = {"seconds": 0.0, "n_node_actions": 0.0,
                                    "masked_invalid": 0.0, "unmasked_invalid": 0.0,
                                    "invalid_probability_mass": 0.0}
    checkpoint_extra: Dict[str, Any] = {}
    if method in PPO_METHODS:
        model, checkpoint_extra, torch = _load_ppo(method, frozen)
        encoder = ResearchObservationEncoder(tuple(world["node_ids"]))
        for _ in range(frozen.steps):
            _ppo_tick(driver, runtime, model, torch, PPO_ARMS[method], encoder, inference)
    else:
        for _ in range(frozen.steps):
            driver.tick()
        inference["seconds"] = float(getattr(world["scheduler"], "planning_time_s", 0.0))
    result = driver.finalize()
    metrics = result.metrics
    vector = metrics["evaluation_vector"]["values"]
    n_actions = max(1.0, inference["n_node_actions"])
    n_planned = sum(int(row.get("n_planned", 0)) for row in driver.tick_stats)
    n_rejected = sum(int(row.get("n_rejected", 0)) for row in driver.tick_stats)
    row = {
        "method": method, "scenario": scenario, "environment_seed": int(seed),
        "training_seed_count": 1 if method in PPO_METHODS else 0,
        "runtime_mode": frozen.runtime_mode, "task_gating": frozen.task_gating,
        "service_completion": float(vector["service_completion"]),
        "task_timeliness": float(vector["task_timeliness"]),
        "estimate_quality": float(vector["estimate_quality"]),
        "resource_consumption": float(vector["resource_consumption"]),
        "communication_overhead": float(vector["communication_overhead"]),
        "compute_time_s": float(inference["seconds"]),
        "completion_rate": float(metrics["completion_rate"]),
        "mean_waiting_s": float(metrics["mean_waiting_s"]),
        "max_waiting_s": float(metrics["max_waiting_s"]),
        "n_expired": int(metrics["n_expired"]),
        "n_tasks_total": int(metrics["n_tasks_total"]),
        "n_completed": int(metrics["n_completed"]),
        "n_planned": n_planned, "n_rejected_by_executor": n_rejected,
        "executor_rejection_rate": n_rejected / n_planned if n_planned else 0.0,
        "resource_violation_rate": 0.0 if result.conservation["all_conserved"] else 1.0,
        "conservation_ok": bool(result.conservation["all_conserved"]),
        "masked_invalid_action_rate": inference["masked_invalid"] / n_actions,
        "unmasked_argmax_invalid_rate": (inference["unmasked_invalid"] / n_actions
                                          if method in PPO_METHODS else None),
        "invalid_probability_mass": (inference["invalid_probability_mass"] / n_actions
                                      if method in PPO_METHODS else None),
        "truth_payload_violations": int(metrics["runtime_feedback"]["truth_payload_violations"]),
        "duplicate_runtime_tasks": int(metrics["runtime_feedback"]["n_duplicate_runtime_tasks"]),
        "checkpoint_extra": checkpoint_extra if method in PPO_METHODS else {},
    }
    return row


def evaluate_split(split_name: str, frozen: FrozenEvaluation,
                   release_test: bool = False) -> List[Dict[str, Any]]:
    split = get_split(split_name, release_test=release_test)
    records: List[Dict[str, Any]] = []
    for method in METHODS:
        for scenario in split.scenarios:
            for seed in split.seeds:
                records.append(evaluate_episode(method, scenario, int(seed), frozen))
    return records


def aggregate(records: Sequence[Mapping[str, Any]], reference: str) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["method"]), str(row["scenario"])), []).append(row)
    by_method: Dict[str, Dict[str, Any]] = {}
    by_scenario: Dict[str, Dict[str, Any]] = {}
    for (method, scenario), rows in sorted(grouped.items()):
        stats = {key: summary([float(row[key]) for row in rows if row.get(key) is not None])
                 for key in METRIC_KEYS}
        by_scenario.setdefault(scenario, {})[method] = stats
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        by_method[method] = {key: summary([float(row[key]) for row in rows if row.get(key) is not None])
                             for key in METRIC_KEYS}
    paired: Dict[str, Any] = {}
    for scenario, table in by_scenario.items():
        ref_rows = {(row["environment_seed"]): row for row in records
                    if row["method"] == reference and row["scenario"] == scenario}
        paired[scenario] = {}
        for method in METHODS:
            if method == reference:
                continue
            rows = [row for row in records if row["method"] == method and row["scenario"] == scenario]
            paired[scenario][method] = {
                key: summary([float(row[key]) - float(ref_rows[row["environment_seed"]][key])
                              for row in rows if row["environment_seed"] in ref_rows
                              and row.get(key) is not None
                              and ref_rows[row["environment_seed"]].get(key) is not None])
                for key in METRIC_KEYS}
    return {"by_method": by_method, "by_scenario": by_scenario,
            "paired_difference_vs_" + reference: paired}


def validate_records(records: Iterable[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for row in records:
        label = f"{row.get('method')}/{row.get('scenario')}/s{row.get('environment_seed')}"
        if not row.get("conservation_ok"):
            errors.append(label + ": resource conservation failed")
        if float(row.get("resource_violation_rate", 1.0)) != 0.0:
            errors.append(label + ": resource violation")
        if int(row.get("truth_payload_violations", 1)) != 0:
            errors.append(label + ": truth payload violation")
        if int(row.get("duplicate_runtime_tasks", 1)) != 0:
            errors.append(label + ": duplicate runtime task")
        if row.get("runtime_mode") != "plan_controlled_feedback":
            errors.append(label + ": wrong runtime mode")
        if row.get("task_gating") != TASK_GATING_EXPOSE_ALL:
            errors.append(label + ": unequal task gating")
    return errors


__all__ = ["DEFAULT_CHECKPOINTS", "FrozenEvaluation", "METHODS", "METRIC_KEYS",
           "aggregate", "evaluate_episode", "evaluate_split", "freeze_manifest",
           "summary", "validate_records"]
