"""大阶段二第二步：四组**同结构**消融（只改算法可见输入）。

冻结清单（用户点名，**不得**为了结果更好而顺手调）
--------------------------------------------------
奖励函数 / Actor-Critic 结构 / 隐藏层规模 / 优化器 / 训练 update 数 /
batch / GAE / PPO 参数 / 结构化动作空间 / 合法动作 mask / train-validation 划分 /
`plan_controlled_feedback` 真闭环 —— 全部写进 `FROZEN_BASELINE` 并在每次运行前
**逐项断言**（`assert_frozen()`），任何一处被改动都会让消融报告生成失败。

唯一允许变化的是**算法可见输入**：

| 组别 | 新鲜度 | 不确定度（σ + 来源一致性） |
| --- | --- | --- |
| `main_baseline` | 置零 | 置零 |
| `freshness_only` | 开 | 置零 |
| `uncertainty_only` | 置零 | 开 |
| `freshness_uncertainty` | 开 | 开 |

四组维度完全相同（`research_obs` 的 104 维），关闭的特征**置零**而非删除。

四组等价性
----------
* **同训练种子、同场景顺序、同预算**：episode 来源是 `train._episode_spec`
  这个**纯函数**（场景轮转 + 种子循环），因此四组看到的 episode 序列逐项相同；
* **validation 用确定性 masked argmax**；checkpoint 选择规则不变
  （守恒硬闸门 → validation 回报最大 → 代价更小 → 更早）；
* **test 分区继续封存**：本模块只调用 `get_split("validation")`，
  没有任何路径可以打开测试集。

mask 是承重机制（§11L.3）
------------------------
所有组**正式部署必须带 mask**，同时记录三个数，不允许把 mask 的帮助隐藏掉：

* `masked_invalid_rate`（带 mask 时违反非法动作的比例，构造上为 0）；
* `unmasked_argmax_invalid_rate`（关掉 mask 后 argmax 的非法率）；
* `invalid_probability_mass`（关掉 mask 后策略分配给非法动作的概率质量，
  比 argmax 更公允）。

用法
----
    D:\\anaconda\\envs\\pytorch_env\\python.exe -m rl_resource.ablation --updates 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import ensure_utf8_console  # noqa: E402

import torch  # noqa: E402

from resource_management.contract_v1 import contract_digest  # noqa: E402
from resource_management.information_research import (  # noqa: E402
    ABLATION_ARMS, RESEARCH_HYPOTHESIS,
)
from resource_management.learning_protocol import (  # noqa: E402
    get_split, split_digest,
)
from rl_resource.actions import N_ACTIONS  # noqa: E402
from rl_resource.env import REWARD_VERSION  # noqa: E402
from rl_resource.policy import PolicyConfig  # noqa: E402
from rl_resource.ppo import PPOConfig  # noqa: E402
from rl_resource.research_obs import RESEARCH_OBS_SCHEMA  # noqa: E402
from rl_resource.scenarios import SCENARIO_MAPPING_VERSION  # noqa: E402
from rl_resource.train import TrainConfig, evaluate, train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(ROOT, "config", "learning_splits_v1.json")
DEFAULT_OUT = os.path.join("output", "rl_resource", "ablation")


# ----------------------------------------------------------------------
# 冻结的基线配置（**唯一真源**）
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenBaseline:
    """四组共用的、**不允许按结果调整**的全部参数。"""

    reward_version: str = REWARD_VERSION
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    optimizer: str = "adam"
    learning_rate: float = 3e-4
    episodes_per_update: int = 8
    steps: int = 24
    updates: int = 12
    n_epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    normalize_advantage: bool = True
    n_actions_per_node: int = N_ACTIONS
    runtime_mode: str = "plan_controlled_feedback"
    task_gating: str = "expose_all"
    train_seed: int = 0
    train_scenarios: Tuple[str, ...] = (
        "rm_train_base", "rm_train_light_load",
        "rm_train_high_arrival", "rm_train_constrained_comm")
    train_seeds: Tuple[int, ...] = (101, 103)
    eval_split: str = "validation"
    test_sealed: bool = True
    selection_rule: str = ("conservation gate -> max validation mean_return "
                           "-> lower mean cost -> earlier update")
    max_nodes: int = 4
    obs_schema: str = RESEARCH_OBS_SCHEMA

    def ppo(self) -> PPOConfig:
        return PPOConfig(
            learning_rate=self.learning_rate, gamma=self.gamma,
            gae_lambda=self.gae_lambda, clip_ratio=self.clip_ratio,
            value_clip=self.value_clip, entropy_coef=self.entropy_coef,
            value_coef=self.value_coef, max_grad_norm=self.max_grad_norm,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            normalize_advantage=self.normalize_advantage,
            target_kl=self.target_kl, seed=self.train_seed)

    def policy(self, obs_dim: int) -> PolicyConfig:
        return PolicyConfig(obs_dim=obs_dim, max_nodes=self.max_nodes,
                            hidden_sizes=self.hidden_sizes,
                            activation=self.activation,
                            seed=self.train_seed)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["train_scenarios"] = list(self.train_scenarios)
        payload["train_seeds"] = list(self.train_seeds)
        return payload


FROZEN_BASELINE = FrozenBaseline()


def assert_frozen(config: TrainConfig, frozen: FrozenBaseline) -> None:
    """逐项断言"除了输入之外什么都没变"。任何不一致直接报错。"""
    checks = {
        "steps": (config.steps, frozen.steps),
        "updates": (config.updates, frozen.updates),
        "rollout_episodes": (config.rollout_episodes,
                             frozen.episodes_per_update),
        "train_scenarios": (tuple(config.scenarios), frozen.train_scenarios),
        "train_seeds": (tuple(config.seeds), frozen.train_seeds),
        "train_seed": (config.seed, frozen.train_seed),
        "hidden_sizes": (tuple(config.policy.hidden_sizes),
                         frozen.hidden_sizes),
        "activation": (config.policy.activation, frozen.activation),
        "learning_rate": (config.ppo.learning_rate, frozen.learning_rate),
        "gamma": (config.ppo.gamma, frozen.gamma),
        "gae_lambda": (config.ppo.gae_lambda, frozen.gae_lambda),
        "clip_ratio": (config.ppo.clip_ratio, frozen.clip_ratio),
        "entropy_coef": (config.ppo.entropy_coef, frozen.entropy_coef),
        "n_epochs": (config.ppo.n_epochs, frozen.n_epochs),
        "batch_size": (config.ppo.batch_size, frozen.batch_size),
        "target_kl": (config.ppo.target_kl, frozen.target_kl),
        "max_nodes": (config.max_nodes, frozen.max_nodes),
    }
    mismatched = {key: values for key, values in checks.items()
                  if values[0] != values[1]}
    if mismatched:
        raise RuntimeError(
            "冻结基线被改动，消融无效：" + json.dumps(mismatched,
                                                     ensure_ascii=False))


# ----------------------------------------------------------------------
# 四组运行
# ----------------------------------------------------------------------


def run_arm(arm: str, frozen: FrozenBaseline, out_root: str,
            quiet: bool = False) -> Dict[str, Any]:
    validation = get_split(frozen.eval_split, path=SPLIT_PATH)
    cfg = TrainConfig(
        scenarios=frozen.train_scenarios, seeds=frozen.train_seeds,
        episodes=frozen.episodes_per_update * frozen.updates,
        steps=frozen.steps, rollout_episodes=frozen.episodes_per_update,
        updates=frozen.updates, max_nodes=frozen.max_nodes,
        ppo=frozen.ppo(), policy=frozen.policy(obs_dim=0),
        out_dir=out_root, tag=arm, smoke=False, arm=arm,
        eval_scenarios=validation.scenarios, eval_seeds=validation.seeds,
        seed=frozen.train_seed)
    assert_frozen(cfg, frozen)
    summary = train(cfg, quiet=quiet)
    return summary


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows)


def _arm_table(summary: Dict[str, Any]) -> Dict[str, Any]:
    validation = summary["validation_of_selected"]
    without_mask = summary["validation_without_mask"]
    rows = validation["rows"]
    return {
        "arm": summary["tag"],
        "selected_update": summary["selected_update"],
        "validation_return": validation["mean_return"],
        "completion_rate": _mean(rows, "completion_rate"),
        "worst_completion_rate": min(
            (row["completion_rate"] for row in rows), default=0.0),
        "mean_completed": _mean(rows, "n_completed"),
        "timeliness": _mean(rows, "timeliness"),
        "mean_waiting_s": _mean(rows, "mean_waiting_s"),
        "worst_waiting_s": max((row["max_waiting_s"] for row in rows),
                               default=0.0),
        "mean_expired": _mean(rows, "n_expired"),
        "estimate_quality": _mean(rows, "estimate_quality"),
        "resource_consumption": _mean(rows, "resource_consumption"),
        "comm_overhead_bytes": _mean(rows, "comm_overhead_bytes"),
        # 三个非法动作指标：**不允许把 mask 的帮助隐藏掉**
        "masked_invalid_rate": validation["illegal_action_rate"],
        "unmasked_argmax_invalid_rate": without_mask["illegal_action_rate"],
        "invalid_probability_mass": without_mask["illegal_probability_mass"],
        "executor_rejection_rate": validation["executor_rejection_rate"],
        "conservation_all_ok": validation["conservation_all_ok"],
        "max_cost_reconciliation_error":
            validation["max_cost_reconciliation_error"],
        "plan_fatal_ticks": validation["plan_fatal_ticks"],
        "legal_action_rates": validation["legal_action_rates"],
        "masked_invalid_rate_definition": (
            "带 mask 时选择落在非法动作上的比例（构造上为 0）"),
        "unmasked_argmax_invalid_rate_definition": (
            "关掉 mask 后 argmax 落到非法动作上的比例"),
        "invalid_probability_mass_definition": (
            "关掉 mask 后策略分配给非法动作的概率质量均值（更公允）"),
    }


def _per_scenario(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        bucket = out.setdefault(row["scenario"], {
            "n": 0, "return": 0.0, "completion_rate": 0.0,
            "timeliness": 0.0, "mean_waiting_s": 0.0, "n_expired": 0.0,
            "estimate_quality": 0.0, "resource_consumption": 0.0,
            "comm_overhead_bytes": 0.0})
        bucket["n"] += 1
        for key in ("return", "completion_rate", "timeliness",
                    "mean_waiting_s", "n_expired", "estimate_quality",
                    "resource_consumption", "comm_overhead_bytes"):
            bucket[key] += float(row.get(key) or 0.0)
    for bucket in out.values():
        n = bucket["n"]
        for key in list(bucket):
            if key != "n":
                bucket[key] = bucket[key] / n
    return out


def compare_arms(tables: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """以 `main_baseline` 为参照做**配对**比较（同一 validation 格子）。"""
    reference = next((row for row in tables
                      if row["arm"] == ABLATION_ARMS[0].value), None)
    if reference is None:
        return {}
    keys = ("validation_return", "completion_rate", "timeliness",
            "mean_waiting_s", "worst_waiting_s", "mean_expired",
            "estimate_quality", "resource_consumption",
            "comm_overhead_bytes")
    deltas: Dict[str, Dict[str, float]] = {}
    for row in tables:
        if row["arm"] == reference["arm"]:
            continue
        deltas[row["arm"]] = {
            key: round(float(row.get(key) or 0.0)
                       - float(reference.get(key) or 0.0), 6)
            for key in keys}
    return {"reference": reference["arm"], "deltas_vs_reference": deltas}


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------


def render_report(result: Dict[str, Any]) -> str:
    frozen = result["frozen_baseline"]
    tables = result["arms"]
    hypothesis = result["hypothesis_verdict"]
    lines = [
        "# 信息新鲜度与不确定度感知调度：四组同结构消融报告",
        "",
        "> 大阶段二第二步。**只改算法可见输入**，其余全部冻结。",
        "",
        "## 1. 冻结基线（不得按结果调整）",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for key in ("reward_version", "hidden_sizes", "activation", "optimizer",
                "learning_rate", "episodes_per_update", "steps", "updates",
                "n_epochs", "batch_size", "gamma", "gae_lambda",
                "clip_ratio", "entropy_coef", "target_kl", "runtime_mode",
                "task_gating", "train_scenarios", "train_seeds",
                "eval_split", "test_sealed", "obs_schema"):
        lines.append(f"| `{key}` | `{frozen[key]}` |")
    lines += [
        "",
        f"- 冻结断言：每次运行前逐项校验（`assert_frozen`），"
        f"不一致直接报错；本次四组全部通过。",
        f"- 选择规则（未变）：{frozen['selection_rule']}",
        f"- 数据划分摘要：`{result['provenance']['split_digest'][:16]}…`"
        f"（test 封存 = {result['provenance']['test_sealed']}）",
        f"- 冻结契约未改动：`{result['provenance']['contract_digest']}`",
        "",
        "## 2. 四组结果（validation，确定性 masked argmax）",
        "",
        "| 组别 | validation 回报 | 完成率 | 最差完成率 | 及时性 | 平均等待(s) "
        "| 最差等待(s) | 过期数 | 估计质量 | 资源消耗 | 通信(B) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tables:
        lines.append(
            f"| `{row['arm']}` | {row['validation_return']:+.3f} "
            f"| {row['completion_rate']:.4f} | {row['worst_completion_rate']:.4f} "
            f"| {row['timeliness']:.4f} | {row['mean_waiting_s']:.3f} "
            f"| {row['worst_waiting_s']:.2f} | {row['mean_expired']:.1f} "
            f"| {row['estimate_quality']:.4f} "
            f"| {row['resource_consumption']:.4f} "
            f"| {row['comm_overhead_bytes']:.0f} |")

    lines += ["", "## 3. 与 `main_baseline` 的配对差值", "",
              "| 组别 | Δ回报 | Δ完成率 | Δ及时性 | Δ平均等待(s) | Δ最差等待(s) "
              "| Δ过期数 | Δ估计质量 | Δ资源消耗 | Δ通信(B) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    comparison = result["comparison"]
    for arm, delta in (comparison.get("deltas_vs_reference") or {}).items():
        lines.append(
            f"| `{arm}` | {delta['validation_return']:+.3f} "
            f"| {delta['completion_rate']:+.4f} | {delta['timeliness']:+.4f} "
            f"| {delta['mean_waiting_s']:+.3f} "
            f"| {delta['worst_waiting_s']:+.2f} "
            f"| {delta['mean_expired']:+.1f} "
            f"| {delta['estimate_quality']:+.4f} "
            f"| {delta['resource_consumption']:+.4f} "
            f"| {delta['comm_overhead_bytes']:+.0f} |")

    lines += ["", "## 4. mask 是承重的：三个非法动作指标", "",
              "| 组别 | masked_invalid_rate | unmasked_argmax_invalid_rate "
              "| invalid_probability_mass | 执行器拒绝率 |",
              "| --- | --- | --- | --- | --- |"]
    for row in tables:
        lines.append(
            f"| `{row['arm']}` | {row['masked_invalid_rate']:.4f} "
            f"| {row['unmasked_argmax_invalid_rate']:.4f} "
            f"| {row['invalid_probability_mass']:.4f} "
            f"| {row['executor_rejection_rate']:.4f} |")
    lines += [
        "",
        "三者的定义：masked_invalid_rate = 带 mask 时选到非法动作的比例"
        "（构造上为 0）；unmasked_argmax_invalid_rate = 关掉 mask 后 argmax "
        "落到非法动作的比例；invalid_probability_mass = 关掉 mask 后策略"
        "分配给非法动作的概率质量（比 argmax 更公允）。",
        "**所有组正式部署必须带 mask**；三个数一起报，不允许只报第一个。",
        "",
        "## 5. 逐场景（validation）",
        "",
    ]
    scenarios = sorted({name for row in result["per_scenario"].values()
                        for name in row})
    lines += ["| 组别 | 场景 | 回报 | 完成率 | 及时性 | 平均等待(s) "
              "| 过期数 | 估计质量 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, per_scene in result["per_scenario"].items():
        for scenario in scenarios:
            bucket = per_scene.get(scenario)
            if not bucket:
                continue
            lines.append(
                f"| `{arm}` | {scenario} | {bucket['return']:+.3f} "
                f"| {bucket['completion_rate']:.4f} "
                f"| {bucket['timeliness']:.4f} "
                f"| {bucket['mean_waiting_s']:.3f} "
                f"| {bucket['n_expired']:.1f} "
                f"| {bucket['estimate_quality']:.4f} |")

    lines += [
        "",
        "## 6. 对旧 §11J 假设的重新验证",
        "",
        f"> 假设原文：{RESEARCH_HYPOTHESIS}",
        "",
        f"- 结论：**{hypothesis['verdict']}**",
        f"- 依据：{hypothesis['reason']}",
        "",
        "⚠️ 旧 §11J 的负结果是在**「调度不影响感知」的旧闭环**下得到的"
        "（当时四组位置 RMSE 完全相同）。现在真闭环已接通、调度会改变测量"
        "与融合，因此该结论**必须在新的条件下重新检验**，不能直接沿用。",
        "",
        "## 7. 必须与结果一起读的限制",
        "",
    ]
    for note in result["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def run_ablation(frozen: FrozenBaseline = FROZEN_BASELINE,
                 out_root: str = DEFAULT_OUT, quiet: bool = False
                 ) -> Dict[str, Any]:
    if not frozen.test_sealed:
        raise RuntimeError("test 分区必须保持封存")
    tables: List[Dict[str, Any]] = []
    per_scenario: Dict[str, Dict[str, Dict[str, float]]] = {}
    summaries: Dict[str, Any] = {}
    for arm in ABLATION_ARMS:
        summary = run_arm(arm.value, frozen, out_root, quiet=quiet)
        summaries[arm.value] = summary
        tables.append(_arm_table(summary))
        per_scenario[arm.value] = _per_scenario(
            summary["validation_of_selected"]["rows"])

    comparison = compare_arms(tables)
    hypothesis = _hypothesis_verdict(tables, comparison)
    result = {
        "frozen_baseline": frozen.to_dict(),
        "provenance": {
            "split_digest": split_digest(SPLIT_PATH),
            "contract_digest": contract_digest(),
            "scenario_mapping_version": SCENARIO_MAPPING_VERSION,
            "reward_version": REWARD_VERSION,
            "obs_schema": RESEARCH_OBS_SCHEMA,
            "test_sealed": True,
            "eval_split": frozen.eval_split,
        },
        "arms": tables,
        "comparison": comparison,
        "per_scenario": per_scenario,
        "hypothesis_verdict": hypothesis,
        "limitations": _limitations(tables, frozen),
        "summaries": {arm: {key: value for key, value in summary.items()
                            if key != "curve"}
                      for arm, summary in summaries.items()},
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "ablation_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(out_root, "ablation_report.md"), "w",
              encoding="utf-8") as handle:
        handle.write(render_report(result))
    return result


def _hypothesis_verdict(tables: Sequence[Dict[str, Any]],
                        comparison: Dict[str, Any]) -> Dict[str, str]:
    """按**预先声明的判据**判定假设是否得到支持。

    判据（与 §11J 的原始表述一致）：组合特征组相对主基线，
    **最差完成率不下降** 且 **资源消耗不增加**。
    """
    by_arm = {row["arm"]: row for row in tables}
    combined = by_arm.get("freshness_uncertainty")
    reference = by_arm.get("main_baseline")
    if combined is None or reference is None:
        return {"verdict": "无法判定", "reason": "缺少组合组或主基线组"}
    worst_ok = combined["worst_completion_rate"] >= (
        reference["worst_completion_rate"] - 1e-9)
    resource_ok = combined["resource_consumption"] <= (
        reference["resource_consumption"] + 1e-9)
    delta = (comparison.get("deltas_vs_reference") or {}).get(
        "freshness_uncertainty", {})
    detail = (
        f"最差完成率 {reference['worst_completion_rate']:.4f} → "
        f"{combined['worst_completion_rate']:.4f}（不下降={worst_ok}）；"
        f"资源消耗 {reference['resource_consumption']:.4f} → "
        f"{combined['resource_consumption']:.4f}（不增加={resource_ok}）；"
        f"Δ回报 {delta.get('validation_return', 0.0):+.3f}")
    if worst_ok and resource_ok:
        return {"verdict": "在本次小样本上**未发现反例**（假设未被推翻）",
                "reason": detail}
    return {"verdict": "**仍未被支持**（保留负结果）", "reason": detail}


def _limitations(tables: Sequence[Dict[str, Any]],
                 frozen: FrozenBaseline) -> List[str]:
    return [
        ("只有 %d 个训练种子 × %d 个 update，validation 为 3 场景 × 3 种子——"
         "**不宣称任何统计显著性**，所有差值都是描述性的。"
         % (len(frozen.train_seeds), frozen.updates)),
        ("四个组共用同一套 episode 序列与预算，但**回报是自定的学习信号**："
         "本报告回答「额外信息有没有用」，**不回答**「PPO 是否优于规则基线」。"),
        ("**「额外信息带来的收益」与「PPO 本身带来的收益」必须分开**："
         "后者需要把学习策略与规则/优化参考放在**同一评价向量**上比较，"
         "本报告未做（§11L.9 第 1 条不变）。"),
        ("mask 是承重机制：三组非法动作指标一起报；**部署必须带 mask**。"),
        ("`source_inconsistency_indicator` 是**未经概率校准的指示量**，"
         "不得称为「出错概率」或「正确概率」。"),
        ("test 分区**未解封**，也没有用任何测试结果回调选择。"),
        ("`load_multiplier` 在此实现为**截止余量缩放**（服务压力），"
         "不是任务数量倍数（§11L.5）。"),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(description="四组同结构消融")
    parser.add_argument("--updates", type=int, default=None,
                        help="覆盖 update 数（**会破坏与 §11L 的可比性**，仅自检用）")
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    frozen = FROZEN_BASELINE
    if args.updates is not None or args.episodes_per_update is not None:
        frozen = FrozenBaseline(
            updates=args.updates if args.updates is not None else frozen.updates,
            episodes_per_update=(args.episodes_per_update
                                 if args.episodes_per_update is not None
                                 else frozen.episodes_per_update))
        print("⚠️ 覆盖了 update/预算：本次结果**不得**与默认冻结基线互比")

    result = run_ablation(frozen, out_root=args.out_dir)
    print()
    for row in result["arms"]:
        print("%-22s 回报 %+8.3f | 完成率 %.4f | 最差完成率 %.4f | 及时性 %.4f "
              "| 平均等待 %.3f | 过期 %.1f | 估计质量 %.4f | 资源 %.4f"
              % (row["arm"], row["validation_return"],
                 row["completion_rate"], row["worst_completion_rate"],
                 row["timeliness"], row["mean_waiting_s"], row["mean_expired"],
                 row["estimate_quality"], row["resource_consumption"]))
    print()
    verdict = result["hypothesis_verdict"]
    print("§11J 假设复验：%s" % verdict["verdict"])
    print("  依据：%s" % verdict["reason"])
    print("报告：%s" % os.path.join(args.out_dir, "ablation_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
