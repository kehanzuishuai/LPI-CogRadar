"""集中式学习资源调度基线：训练 CLI。

它做什么
--------
1. 在 **train 分区**的场景×种子上训练一个 PPO Actor-Critic；
2. 每轮在 **validation 分区**上评测，按**预先声明的规则**选 checkpoint；
3. **test 分区保持封存**（`get_split("test")` 默认抛 `SealedSplitError`，
   本脚本没有任何路径可以打开它）；
4. 记录：随机种子、训练曲线、非法动作率、资源违反率、守恒、代价对账、
   实验契约摘要（场景映射摘要 + 奖励版本 + 数据划分摘要）。

checkpoint 选择规则（**先声明、后执行**，不得按 validation 结果回调）
--------------------------------------------------------------------
1. 淘汰资源守恒失败或代价对账失败的候选（硬闸门）；
2. 在通过硬闸门的候选里按**平均折扣回报**最大选；
3. 并列时依次比：平均累计代价更小 → 更早的 checkpoint。

冒烟训练与正式训练用同一个脚本，只用规模参数区分：
`--episodes` / `--updates`。**本轮只做冒烟与少量种子**，
目标只是证明"能稳定学习且资源守恒"，**不要求**超过规则或优化参考。

用法
----
    D:\\anaconda\\envs\\pytorch_env\\python.exe -m rl_resource.train --smoke
    D:\\anaconda\\envs\\pytorch_env\\python.exe -m rl_resource.train \\
        --scenarios rm_train_base --seeds 101 103 --updates 6 --out-dir output/rl_resource/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import ensure_utf8_console  # noqa: E402

import torch  # noqa: E402

from resource_management.learning_protocol import (  # noqa: E402
    SealedTestSplitError, get_split, split_digest,
)
from rl_resource.actions import N_ACTIONS, pad_mask  # noqa: E402
from rl_resource.env import (  # noqa: E402
    CentralizedResourceSchedulingEnv, EnvConfig, PREFERENCE_V2_REWARD_VERSION,
    REWARD_VERSION,
)
from rl_resource.obs import DEFAULT_MAX_NODES, observation_dim  # noqa: E402
from rl_resource.policy import (  # noqa: E402
    ActorCritic, PolicyConfig, legal_action_rates, mask_to_tensor,
)
from rl_resource.ppo import PPOConfig, RolloutBuffer, ppo_update  # noqa: E402
from rl_resource.scenarios import (  # noqa: E402
    SCENARIO_MAPPING_VERSION, mapping_digest, scenario_names_for_split,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join("output", "rl_resource")


# ----------------------------------------------------------------------
# 训练配置
# ----------------------------------------------------------------------


@dataclass
class TrainConfig:
    scenarios: Tuple[str, ...] = ()
    seeds: Tuple[int, ...] = ()
    episodes: int = 24
    steps: int = 24
    rollout_episodes: int = 6
    updates: int = 4
    max_nodes: int = DEFAULT_MAX_NODES
    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    out_dir: str = DEFAULT_OUT
    tag: str = "smoke"
    smoke: bool = False
    eval_scenarios: Tuple[str, ...] = ()
    eval_seeds: Tuple[int, ...] = ()
    seed: int = 0
    #: 消融组；空 = §11L 的 49 维基线观测
    arm: str = ""
    reward_mode: str = "baseline"
    preference_conditioned: bool = False
    preference_set: Tuple[Tuple[float, float, float, float, float], ...] = ()
    evaluation_preference: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2)
    scenario_registry_path: str = "config/learning_splits_v1.json"
    share_candidate_requires_track: bool = True

    def resolved(self) -> "TrainConfig":
        if self.smoke:
            if not self.scenarios:
                self.scenarios = ("rm_train_base",)
            if not self.seeds:
                self.seeds = (101,)
            self.updates = min(self.updates, 3)
            self.episodes = min(self.episodes, 12)
            self.rollout_episodes = min(self.rollout_episodes, 4)
        else:
            if not self.scenarios:
                self.scenarios = scenario_names_for_split("train")
            if not self.seeds:
                self.seeds = get_split("train").seeds[:2]
        if not self.eval_scenarios:
            self.eval_scenarios = scenario_names_for_split("validation")
        if not self.eval_seeds:
            self.eval_seeds = get_split("validation").seeds
        self.policy.max_nodes = self.max_nodes
        if self.arm:
            # 研究消融组用的是**同形 104 维**观测（不是 §11L 的 49 维），
            # 网络输入必须按它来，否则前向直接形状不符。
            from resource_management.closed_loop import NODE_LAYOUT
            from rl_resource.research_obs import ResearchObservationEncoder
            self.policy.obs_dim = ResearchObservationEncoder(
                tuple(sorted(NODE_LAYOUT))).output_dim
        else:
            self.policy.obs_dim = observation_dim(self.max_nodes)
        if self.preference_conditioned:
            self.policy.obs_dim += 5
            if self.policy.conditioning == "film":
                # Preference observations are deliberately appended by the
                # environment.  The policy alone splits that frozen layout;
                # concat arms retain their historical behavior unchanged.
                self.policy.state_obs_dim = self.policy.obs_dim - 5
                self.policy.preference_dim = 5
        return self


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _episode_spec(cfg: TrainConfig, index: int) -> Tuple[str, int]:
    """第 `index` 个训练 episode 的（场景, 种子）。

    **纯函数、可无限延伸**。第一版把 episode 计划写成一个长度 `episodes` 的
    固定列表，跑完就没了——结果第 6 次更新起 `trans 0`，后面 15 次"训练"
    全是空转（entropy/kl 恒为 0），报出来的"改善"只是验证噪声。
    这里改成按全局 episode 序号循环取场景与种子：场景轮转、种子每轮一圈后换。
    """
    scenarios = cfg.scenarios or ("rm_train_base",)
    seeds = cfg.seeds or (101,)
    scenario = scenarios[index % len(scenarios)]
    seed = seeds[(index // len(scenarios)) % len(seeds)]
    return scenario, seed


def _preference_for_episode(cfg: TrainConfig, index: int) -> Tuple[float, float, float, float, float]:
    if not cfg.preference_set:
        return cfg.evaluation_preference
    return cfg.preference_set[random.Random(cfg.seed * 100003 + index).randrange(len(cfg.preference_set))]


def collect_rollout(env: CentralizedResourceSchedulingEnv, model: ActorCritic,
                    cfg: TrainConfig, device: torch.device,
                    start_index: int) -> Tuple[RolloutBuffer, List[Dict[str, Any]]]:
    """收集 `rollout_episodes` 个 episode 的转移。"""
    buffer = RolloutBuffer()
    episodes: List[Dict[str, Any]] = []
    for offset in range(cfg.rollout_episodes):
        scenario, seed = _episode_spec(cfg, start_index + offset)
        env.config.scenario = scenario
        env.config.preference = _preference_for_episode(cfg, start_index + offset)
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        masked_actions = 0
        n_nodes_steps = 0
        while not done:
            for row in info["mask"]:
                masked_actions += sum(1 for value in row if not value)
                n_nodes_steps += len(row)
            obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
            mask_t = mask_to_tensor(info["mask"], device).unsqueeze(0)
            with torch.no_grad():
                out = model.act(obs_t, mask_t)
            action = out["action"][0].tolist()
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            if terminated:
                next_value = 0.0
            else:
                next_obs_t = torch.tensor([next_obs], dtype=torch.float32,
                                          device=device)
                next_mask_t = mask_to_tensor(
                    next_info.get("next_mask") or next_info["mask"],
                    device).unsqueeze(0)
                with torch.no_grad():
                    _logits, next_value = model(next_obs_t)
                    next_value = float(next_value.item())
            buffer.add(
                obs=obs, mask=info["mask"], action=action,
                log_prob=float(out["log_prob"][0].item()),
                value=float(out["value"][0].item()),
                reward=float(reward), terminated=terminated,
                truncated=truncated, next_value=next_value)
            total_reward += float(reward)
            obs, info = next_obs, next_info
            done = terminated or truncated

        final = env.finalize()
        episodes.append({
            "scenario": scenario, "seed": seed,
            "return": total_reward,
            "steps": final["trace"][-1]["step"] if final["trace"] else 0,
            "termination_reason":
                final["trace"][-1]["termination_reason"] if final["trace"] else "",
            "cumulative_resource_cost": final["cumulative_resource_cost"],
            "measured_resource_consumption":
                final["measured_resource_consumption"],
            "cost_reconciliation_error": final["cost_reconciliation_error"],
            "conservation_ok": final["conservation_ok"],
            "n_planned": final["n_planned"],
            "n_rejected_by_executor": final["n_rejected_by_executor"],
            "executor_rejection_rate": final["executor_rejection_rate"],
            "unmasked_illegal_action_rate":
                fin_unmasked_illegal(final),
            "plan_fatal_ticks": fin_plan_fatal(final),
            "masked_action_fraction": (masked_actions / (n_nodes_steps * N_ACTIONS)
                                       if n_nodes_steps else 0.0),
        })
    return buffer, episodes


def fin_unmasked_illegal(final: Dict[str, Any]) -> float:
    return float(final["unmasked_illegal_action_rate"])


def fin_plan_fatal(final: Dict[str, Any]) -> int:
    return int(final["plan_fatal_ticks"])


# ----------------------------------------------------------------------
# 评测（validation：checkpoint 选择）
# ----------------------------------------------------------------------


def evaluate(model: ActorCritic, cfg: TrainConfig, device: torch.device,
             scenarios: Sequence[str], seeds: Sequence[int],
             use_mask: bool = True) -> Dict[str, Any]:
    """在给定场景×种子上评测。**默认关闭梯度、默认带 mask**。

    `use_mask=False` 用于测"策略在**没有** mask 时会做多少非法动作"——
    这是非法动作率的**诚实测法**（带 mask 时它结构性为 0）。
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for scenario in scenarios:
            for seed in seeds:
                env = CentralizedResourceSchedulingEnv(EnvConfig(
                    scenario=scenario, seed=seed, steps=cfg.steps,
                    max_nodes=cfg.max_nodes, keep_trace=False,
                    arm=cfg.arm, reward_mode=cfg.reward_mode,
                    preference=cfg.evaluation_preference,
                    preference_conditioned=cfg.preference_conditioned,
                    scenario_registry_path=cfg.scenario_registry_path,
                    share_candidate_requires_track=cfg.share_candidate_requires_track))
                obs, info = env.reset(seed=seed)
                done = False
                total_reward = 0.0
                illegal = 0
                n_node_steps = 0
                illegal_mass = 0.0
                n_node_steps_for_mass = 0
                legal_counts = [0] * N_ACTIONS
                action_counts = [0] * N_ACTIONS
                while not done:
                    mask = info["mask"]
                    if use_mask:
                        effective = mask
                    else:
                        # 关掉 mask：所有动作都"允许"（用来测非法率）
                        effective = [[True] * N_ACTIONS for _ in mask]
                    for index in range(len(env.node_ids)):
                        for action_index in range(N_ACTIONS):
                            if mask[index][action_index]:
                                legal_counts[action_index] += 1
                    obs_t = torch.tensor([obs], dtype=torch.float32,
                                         device=device)
                    mask_t = mask_to_tensor(effective, device).unsqueeze(0)
                    out = model.act(obs_t, mask_t, deterministic=True)
                    action = out["action"][0].tolist()
                    for chosen in action[:len(env.node_ids)]:
                        action_counts[int(chosen)] += 1
                    if not use_mask:
                        # 更公允的"非法倾向"度量：未加 mask 时策略分配给
                        # **非法动作**的概率质量。用 argmax 会得到 100%，
                        # 那只是"被掩掉的 logit 从未被训练"的必然结果，
                        # 不是"策略学坏了"——两者要分开报。
                        with torch.no_grad():
                            logits, _value = model(obs_t)
                            probs = torch.softmax(logits[0], dim=-1)
                        for index in range(len(env.node_ids)):
                            illegal_mass += float(probs[index][
                                ~torch.tensor(mask[index], device=device)
                            ].sum().item())
                            n_node_steps_for_mass += 1
                    for index in range(len(env.node_ids)):
                        n_node_steps += 1
                        if not mask[index][action[index]]:
                            illegal += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated
                final = env.finalize()
                trace = final["trace"]
                metrics = final["result"].metrics
                vector = metrics["evaluation_vector"]["values"]
                rows.append({
                    "scenario": scenario, "seed": seed,
                    "return": total_reward,
                    "termination_reason": (trace[-1]["termination_reason"]
                                           if trace else ""),
                    "cumulative_resource_cost":
                        final["cumulative_resource_cost"],
                    "measured_resource_consumption":
                        final["measured_resource_consumption"],
                    "cost_reconciliation_error":
                        final["cost_reconciliation_error"],
                    "conservation_ok": final["conservation_ok"],
                    "n_planned": final["n_planned"],
                    "n_rejected_by_executor": final["n_rejected_by_executor"],
                    "executor_rejection_rate": final["executor_rejection_rate"],
                    "illegal_action_rate": (illegal / n_node_steps
                                            if n_node_steps else 0.0),
                    "illegal_probability_mass": (
                        illegal_mass / n_node_steps_for_mass
                        if n_node_steps_for_mass else 0.0),
                    "plan_fatal_ticks": final["plan_fatal_ticks"],
                    "legal_action_rates": {
                        name: (legal_counts[index] / n_node_steps
                               if n_node_steps else 0.0)
                        for index, name in enumerate(
                            ("idle", "sample", "process", "share"))},
                    "action_composition": {name: action_counts[index]
                                           for index, name in enumerate(("idle", "sample", "process", "share"))},
                    # --- 完整闭环指标（与规则基线同一口径）---
                    "n_tasks": metrics["n_tasks_total"],
                    "n_completed": metrics["n_completed"],
                    "n_completed_on_time": metrics["completed_on_time"],
                    "n_expired": metrics["n_expired"],
                    "n_abandoned": metrics["n_abandoned"],
                    "n_starved": metrics["n_starved"],
                    "completion_rate": metrics["completion_rate"],
                    "timeliness": vector["task_timeliness"],
                    "mean_waiting_s": metrics["mean_waiting_s"],
                    "max_waiting_s": metrics["max_waiting_s"],
                    "estimate_quality": vector["estimate_quality"],
                    "mean_information_age_s": _observed_quality_stats(
                        final["result"].node_ages, "mean_track_age_s")[0],
                    "information_age_state": (
                        "observed" if _observed_quality_stats(
                            final["result"].node_ages, "mean_track_age_s")[1]
                        else "not_applicable"),
                    "mean_sigma_m": _observed_quality_stats(
                        final["result"].node_ages, "mean_sigma_m")[0],
                    "sigma_state": (
                        "observed" if _observed_quality_stats(
                            final["result"].node_ages, "mean_sigma_m")[1]
                        else "not_applicable"),
                    "resource_consumption": vector["resource_consumption"],
                    "comm_overhead_bytes": vector["communication_overhead"],
                })
    model.train()
    n = len(rows) or 1
    summary = {
        "n_episodes": len(rows),
        "mean_return": sum(row["return"] for row in rows) / n,
        "mean_resource_cost": (sum(row["cumulative_resource_cost"]
                                   for row in rows) / n),
        "mean_measured_consumption": (
            sum(row["measured_resource_consumption"] for row in rows) / n),
        "max_cost_reconciliation_error": max(
            (row["cost_reconciliation_error"] for row in rows), default=0.0),
        "conservation_all_ok": all(row["conservation_ok"] for row in rows),
        "executor_rejection_rate": (sum(row["n_rejected_by_executor"]
                                        for row in rows)
                                    / max(1, sum(row["n_planned"]
                                                 for row in rows))),
        "illegal_action_rate": sum(row["illegal_action_rate"]
                                   for row in rows) / n,
        "illegal_probability_mass": sum(row["illegal_probability_mass"]
                                        for row in rows) / n,
        "plan_fatal_ticks": sum(row["plan_fatal_ticks"] for row in rows),
        "legal_action_rates": {
            name: (sum(row["legal_action_rates"][name] for row in rows) / n)
            for name in ("idle", "sample", "process", "share")},
        "rows": rows,
        # --- 完整闭环指标（逐 episode 均值）---
        "mean_completion_rate": sum(row["completion_rate"] for row in rows) / n,
        "mean_timeliness": sum(row["timeliness"] for row in rows) / n,
        "mean_waiting_s": sum(row["mean_waiting_s"] for row in rows) / n,
        "worst_waiting_s": max((row["max_waiting_s"] for row in rows),
                               default=0.0),
        "mean_expired": sum(row["n_expired"] for row in rows) / n,
        "mean_completed": sum(row["n_completed"] for row in rows) / n,
        "mean_estimate_quality": sum(row["estimate_quality"]
                                     for row in rows) / n,
        "mean_resource_consumption": sum(row["resource_consumption"]
                                         for row in rows) / n,
        "mean_comm_overhead_bytes": sum(row["comm_overhead_bytes"]
                                        for row in rows) / n,
        "worst_completion_rate": min(row["completion_rate"] for row in rows),
    }
    return summary


def _observed_quality_stats(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[Optional[float], int]:
    values = [float(value) for row in rows for value in row.get(key, [])
              if value is not None]
    return ((sum(values) / len(values)) if values else None, len(values))


# ----------------------------------------------------------------------
# checkpoint 选择（**先声明、后执行**）
# ----------------------------------------------------------------------


def select_checkpoint(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按预声明规则选 checkpoint。

    1. 淘汰守恒/对账失败的候选（硬闸门）；
    2. 通过者里按平均折扣回报最大选；
    3. 并列时比平均累计代价更小 → 更早的 checkpoint。
    """
    eligible = [row for row in candidates
                if row["validation"]["conservation_all_ok"]
                and row["validation"]["max_cost_reconciliation_error"] <= 1e-6]
    pool = eligible or list(candidates)
    if not pool:
        raise ValueError("没有任何候选 checkpoint")
    return min(pool, key=lambda row: (
        -row["validation"]["mean_return"],
        row["validation"]["mean_resource_cost"],
        row["update"],
    ))


# ----------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------


def train(cfg: TrainConfig, quiet: bool = False) -> Dict[str, Any]:
    cfg = cfg.resolved()
    cfg.ppo.validate()
    _set_seeds(cfg.seed)
    device = torch.device("cpu")
    model = ActorCritic(cfg.policy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)

    env = CentralizedResourceSchedulingEnv(EnvConfig(
        scenario=cfg.scenarios[0], seed=cfg.seeds[0], steps=cfg.steps,
        max_nodes=cfg.max_nodes, keep_trace=True, arm=cfg.arm, reward_mode=cfg.reward_mode,
        preference=cfg.evaluation_preference, preference_conditioned=cfg.preference_conditioned,
        scenario_registry_path=cfg.scenario_registry_path,
        share_candidate_requires_track=cfg.share_candidate_requires_track))

    run_dir = os.path.join(cfg.out_dir, cfg.tag)
    os.makedirs(run_dir, exist_ok=True)
    curve: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    started = time.perf_counter()
    episode_index = 0
    last_update: Dict[str, Any] = {}

    for update in range(1, cfg.updates + 1):
        buffer, episodes = collect_rollout(env, model, cfg, device, episode_index)
        episode_index += len(episodes)
        stats = ppo_update(model, optimizer, buffer, cfg.ppo, device)
        last_update = stats.to_dict()
        validation = evaluate(model, cfg, device, cfg.eval_scenarios,
                              cfg.eval_seeds)
        record = {
            "update": update,
            "n_transitions": len(buffer),
            "train_mean_return": (sum(row["return"] for row in episodes)
                                  / max(1, len(episodes))),
            "train_mean_cost": (sum(row["cumulative_resource_cost"]
                                    for row in episodes)
                                / max(1, len(episodes))),
            "train_conservation_ok": all(row["conservation_ok"]
                                         for row in episodes),
            "train_cost_reconciliation_error": max(
                (row["cost_reconciliation_error"] for row in episodes),
                default=0.0),
            "train_illegal_action_rate": (sum(
                row["unmasked_illegal_action_rate"] for row in episodes)
                / max(1, len(episodes))),
            "train_masked_action_fraction": (sum(
                row["masked_action_fraction"] for row in episodes)
                / max(1, len(episodes))),
            "validation": validation,
            **{f"ppo_{key}": value for key, value in last_update.items()},
        }
        curve.append(record)
        candidates.append(record)
        if not quiet:
            print("update %d/%d | trans %d | train R %+.3f | val R %+.3f "
                  "| val cost %.4f | conserv %s | kl %.4f | entropy %.3f"
                  % (update, cfg.updates, len(buffer),
                     record["train_mean_return"],
                     validation["mean_return"],
                     validation["mean_resource_cost"],
                     validation["conservation_all_ok"],
                     last_update.get("approx_kl", 0.0),
                     last_update.get("entropy", 0.0)))
        buffer.clear()

    best = select_checkpoint(candidates)
    model_path = os.path.join(run_dir, "policy.pt")
    model.save(model_path, extra={
        "tag": cfg.tag,
        "update": best["update"],
        "selection_rule": ("conservation gate -> max validation mean_return "
                           "-> lower mean cost -> earlier update"),
        "scenario_mapping_version": SCENARIO_MAPPING_VERSION,
        "scenario_mapping_digest": mapping_digest(cfg.scenario_registry_path),
        "reward_version": (PREFERENCE_V2_REWARD_VERSION
                           if cfg.reward_mode == "preference_v2" else REWARD_VERSION),
        "split_digest": split_digest(cfg.scenario_registry_path),
        "scenarios": list(cfg.scenarios), "seeds": list(cfg.seeds),
        "eval_scenarios": list(cfg.eval_scenarios),
        "eval_seeds": list(cfg.eval_seeds),
        "steps": cfg.steps, "max_nodes": cfg.max_nodes,
    })

    # 非法动作用**不带 mask** 的方式单独测一次（诚实测法）
    unmasked = evaluate(model, cfg, device, cfg.eval_scenarios,
                        cfg.eval_seeds, use_mask=False)

    summary = {
        "tag": cfg.tag,
        "smoke": cfg.smoke,
        "train_config": {
            "scenarios": list(cfg.scenarios), "seeds": list(cfg.seeds),
            "episodes": cfg.episodes, "steps": cfg.steps,
            "rollout_episodes": cfg.rollout_episodes, "updates": cfg.updates,
            "max_nodes": cfg.max_nodes, "seed": cfg.seed, "reward_mode": cfg.reward_mode,
            "scenario_registry_path": cfg.scenario_registry_path,
            "share_candidate_requires_track": cfg.share_candidate_requires_track,
            "eval_scenarios": list(cfg.eval_scenarios),
            "eval_seeds": list(cfg.eval_seeds),
        },
        "ppo": asdict(cfg.ppo),
        "policy": {
            "obs_dim": cfg.policy.obs_dim,
            "max_nodes": cfg.policy.max_nodes,
            "hidden_sizes": list(cfg.policy.hidden_sizes),
            "activation": cfg.policy.activation,
            "conditioning": cfg.policy.conditioning,
            "state_obs_dim": cfg.policy.state_obs_dim,
            "preference_dim": cfg.policy.preference_dim,
            "preference_hidden_size": cfg.policy.preference_hidden_size,
            "n_actions_per_node": N_ACTIONS,
        },
        "provenance": {
            "scenario_mapping_version": SCENARIO_MAPPING_VERSION,
            "scenario_mapping_digest": mapping_digest(cfg.scenario_registry_path),
            "reward_version": (PREFERENCE_V2_REWARD_VERSION
                               if cfg.reward_mode == "preference_v2" else REWARD_VERSION),
            "split_digest": split_digest(cfg.scenario_registry_path),
            "runtime_mode": "plan_controlled_feedback",
            "task_gating": "expose_all",
            "test_split_sealed": True,
        },
        "curve": curve,
        "selected_update": best["update"],
        "validation_of_selected": best["validation"],
        "validation_without_mask": unmasked,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "policy_path": model_path,
        "device": str(device),
    }

    _write_outputs(run_dir, summary)
    return summary


def _write_outputs(run_dir: str, summary: Dict[str, Any]) -> None:
    with open(os.path.join(run_dir, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

    rows = []
    for record in summary["curve"]:
        validation = record["validation"]
        rows.append({
            "update": record["update"],
            "n_transitions": record["n_transitions"],
            "train_mean_return": record["train_mean_return"],
            "train_mean_cost": record["train_mean_cost"],
            "train_conservation_ok": record["train_conservation_ok"],
            "train_illegal_action_rate": record["train_illegal_action_rate"],
            "train_masked_action_fraction":
                record["train_masked_action_fraction"],
            "val_mean_return": validation["mean_return"],
            "val_mean_resource_cost": validation["mean_resource_cost"],
            "val_conservation_all_ok": validation["conservation_all_ok"],
            "val_executor_rejection_rate":
                validation["executor_rejection_rate"],
            "val_max_cost_reconciliation_error":
                validation["max_cost_reconciliation_error"],
            "val_plan_fatal_ticks": validation["plan_fatal_ticks"],
            "ppo_policy_loss": record.get("ppo_policy_loss"),
            "ppo_value_loss": record.get("ppo_value_loss"),
            "ppo_entropy": record.get("ppo_entropy"),
            "ppo_approx_kl": record.get("ppo_approx_kl"),
            "ppo_clip_fraction": record.get("ppo_clip_fraction"),
            "ppo_explained_variance": record.get("ppo_explained_variance"),
        })
    path = os.path.join(run_dir, "training_curve.csv")
    if rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(run_dir, "selection.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "selected_update": summary["selected_update"],
            "selection_rule": ("conservation gate -> max validation mean_return "
                               "-> lower mean cost -> earlier update"),
            "candidates": [
                {"update": record["update"],
                 "val_mean_return": record["validation"]["mean_return"],
                 "val_mean_cost": record["validation"]["mean_resource_cost"],
                 "conservation_ok":
                     record["validation"]["conservation_all_ok"]}
                for record in summary["curve"]],
        }, handle, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(
        description="集中式学习资源调度基线（PPO）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟训练：1 场景 1 种子、少量 update")
    parser.add_argument("--scenarios", nargs="+", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[])
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--rollout-episodes", type=int, default=6)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--eval-scenarios", nargs="+", default=[])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[])
    args = parser.parse_args(argv)

    ppo = PPOConfig(learning_rate=args.learning_rate,
                    entropy_coef=args.entropy_coef,
                    n_epochs=args.epochs, batch_size=args.batch_size)
    cfg = TrainConfig(
        scenarios=tuple(args.scenarios), seeds=tuple(args.seeds),
        episodes=args.episodes, steps=args.steps,
        rollout_episodes=args.rollout_episodes, updates=args.updates,
        ppo=ppo, out_dir=args.out_dir, tag=args.tag, smoke=args.smoke,
        eval_scenarios=tuple(args.eval_scenarios),
        eval_seeds=tuple(args.eval_seeds), seed=args.seed)
    summary = train(cfg)
    print()
    print("产物：%s" % os.path.join(args.out_dir, args.tag))
    print("选中 checkpoint：update %d" % summary["selected_update"])
    validation = summary["validation_of_selected"]
    print("validation：R %+.3f | 资源消耗 %.4f | 守恒 %s | 执行器拒绝率 %.4f"
          % (validation["mean_return"], validation["mean_resource_cost"],
             validation["conservation_all_ok"],
             validation["executor_rejection_rate"]))
    print("无 mask 时的非法动作率（argmax）：%.4f"
          % summary["validation_without_mask"]["illegal_action_rate"])
    print("无 mask 时非法动作的**概率质量**：%.4f（比 argmax 更公允，见 docstring）"
          % summary["validation_without_mask"]["illegal_probability_mass"])
    print("对账最大误差：%.2e（必须 ~0）"
          % validation["max_cost_reconciliation_error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
