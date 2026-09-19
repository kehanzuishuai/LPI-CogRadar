"""P2 实验：不确定度感知的可信决策与安全回退。

回答 v4.0 的第二个研究问题：

    **模型知不知道自己什么时候不可靠？**

四个必须汇报的指标（v4.0 明确要求）
-----------------------------------
1. **AI 自主率** `ai_autonomy_rate`   —— 有多少步真的由 AI 的 argmax 决定；
2. **回退率**   `fallback_rate` / `shield_rate` —— 有多少步交出去或被打补丁；
3. **错误决策率** `wrong_decision_rate_*` —— 决策后本步探测未达标的比例。
   分别统计「AI 自主决策的步」与「被干预的步」——**这是判断回退有没有用的关键**：
   如果被干预步的错误率并不低于自主步，回退就是无效的装饰。
4. **高风险状态下的任务满足率** `high_risk_satisfaction_rate`
   —— 「高风险」定义为任一回退触发条件命中（无论最终是否干预）。

对比臂
------
* 普通 DQN（部分可观测）            —— 无不确定度机制
* 集成 DQN（纯 argmax，无回退）      —— 有不确定度信号但不使用
* 集成 DQN + 安全护盾               —— 触发时抬升到满足要求的最低档
* 集成 DQN + 完全回退规则            —— 触发时整体交给规则
* 规则(仅观测)                      —— 可解释基准

阈值灵敏度与信号消融
--------------------
因为回退阈值是**人工设定**的，「回退率」很大程度上反映阈值而非智能体的内省能力。
所以本脚本默认额外跑两组：
* `--threshold-sweep`：松/中/紧三组阈值下的自主率与错误率；
* `--signal-ablation`：只用集成分歧 / 只用 OOD / 只用观测质量 / 全用。

只有同时看到这两组结果，「回退机制有效」这句话才站得住。
"""

from __future__ import annotations

import argparse
import csv
import copy
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()
import experiment_config as ec
from rl.dqn_agent import DQNAgent, silence_numpy_bridge_warning
from rl.ensemble_agent import EnsembleDQNAgent
from strategy.power_policy import RuleBasedPowerPolicy
from strategy.uncertainty_policy import (
    MODE_AI,
    FallbackConfig,
    UncertaintyAwarePolicy,
)
from strategy.belief_policy import BeliefPolicy

silence_numpy_bridge_warning()

DEFAULT_OUT_DIR = "output/uncertainty"
DEFAULT_ENSEMBLE_MODEL = os.path.join("output", "rl_ensemble", "ensemble_best.pt")
DEFAULT_POMDP_MODEL = os.path.join("output", "rl_pomdp_1200", "dqn_agent_best.pt")

#: 阈值灵敏度扫描：从「几乎不触发」到「极其敏感」。
#: 观测质量阈值按实测分布校准（mild 0.66~0.69 / moderate 0.58~0.69 / severe 0.50~0.67）。
THRESHOLD_LEVELS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("松（几乎不回退）", {
        "q_std_threshold": 0.80, "disagreement_threshold": 0.90,
        "q_margin_threshold": 0.01, "ood_threshold": 6.0,
        "obs_quality_threshold": 0.45,
    }),
    ("中（默认）", {
        "q_std_threshold": 0.35, "disagreement_threshold": 0.60,
        "q_margin_threshold": 0.05, "ood_threshold": 3.0,
        "obs_quality_threshold": 0.60,
    }),
    ("紧（频繁回退）", {
        "q_std_threshold": 0.15, "disagreement_threshold": 0.30,
        "q_margin_threshold": 0.20, "ood_threshold": 1.5,
        "obs_quality_threshold": 0.67,
    }),
)

#: 信号消融：只保留一路信号
SIGNAL_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("只用集成分歧", {"use_ensemble_signal": True, "use_ood_signal": False,
                  "use_observation_signal": False}),
    ("只用OOD评分", {"use_ensemble_signal": False, "use_ood_signal": True,
                 "use_observation_signal": False}),
    ("只用观测质量", {"use_ensemble_signal": False, "use_ood_signal": False,
                  "use_observation_signal": True}),
    ("三路全用（默认）", {"use_ensemble_signal": True, "use_ood_signal": True,
                    "use_observation_signal": True}),
    ("全部关闭（对照）", {"use_ensemble_signal": False, "use_ood_signal": False,
                    "use_observation_signal": False}),
)


# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P2：不确定度感知的可信决策评测（v4.0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ec.DEFAULT_SEEDS))
    parser.add_argument("--no-jitter", action="store_true",
                        help="关闭初始条件域随机化（默认开启）")
    parser.add_argument("--energy-budget", type=float, default=None)
    parser.add_argument("--device", default="cpu")

    ec.add_observation_arguments(parser)
    parser.add_argument("--ensemble-model", default=DEFAULT_ENSEMBLE_MODEL)
    parser.add_argument("--pomdp-model", default=DEFAULT_POMDP_MODEL)

    parser.add_argument("--fallback-mode", choices=["shield", "fallback_rule"],
                        default="shield", help="主对比使用的回退模式")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="额外跑阈值灵敏度扫描")
    parser.add_argument("--signal-ablation", action="store_true",
                        help="额外跑不确定度信号消融")
    parser.add_argument("--no-plain-dqn", action="store_true",
                        help="跳过普通 DQN 对照臂")
    return parser


# ----------------------------------------------------------------------
# episode 运行 + 逐步指标
# ----------------------------------------------------------------------

def run_policy_episode(
    env: Any, policy: UncertaintyAwarePolicy, seed: int, label: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """跑一个 episode，返回 (episode 汇总, 逐步记录)。

    逐步记录里同时保留：
    * 决策侧：模式、原因码、不确定度（来自策略）
    * 真实侧：`task_satisfied`、`pd_min`（来自**真值**仿真）
    两者必须分开存，否则「决策看起来对」和「实际确实对」会被混为一谈。
    """
    obs, _ = env.reset(seed=seed)
    policy.reset()
    steps: List[Dict[str, Any]] = []

    while True:
        action, record = policy.select_action(obs, env)

        # --- 单步反事实：在**决策时刻的真实状态**上试算两个候选动作 ---
        # 说明：这里用真值状态做试算，是**评测**行为，不是控制行为——
        # 策略本身看不到这些试算结果。目的是回答
        # 「同样状态下，如果执行 AI 原本的动作会不会更好」。
        levels = env.sim.power_levels_w
        cf_ai_satisfied: Optional[bool] = None
        cf_shield_satisfied: Optional[bool] = None
        if record.mode != MODE_AI and 0 <= record.ai_action < len(levels):
            try:
                cf_ai_satisfied = bool(env.sim.preview(levels[record.ai_action]).task_satisfied)
                cf_shield_satisfied = bool(env.sim.preview(levels[action]).task_satisfied)
            except Exception:
                cf_ai_satisfied = cf_shield_satisfied = None

        obs, _reward, terminated, truncated, info = env.step(action)
        steps.append({
            "mode": record.mode,
            "reason_code": record.reason_code,
            "triggered": len(record.triggered),
            "obs_quality": record.obs_quality,
            "q_std_max": float(record.uncertainty.get("q_std_max", 0.0)),
            "ood_score": float(record.uncertainty.get("ood_score", 0.0)),
            "q_margin": float(record.uncertainty.get("q_margin", 0.0)),
            "disagreement": float(record.uncertainty.get("disagreement", 0.0)),
            "action": action,
            "ai_action": record.ai_action,
            # --- 真值侧 ---
            "pd_min": float(info["pd_min"]),
            "task_satisfied": bool(info["task_satisfied"]),
            "tx_power_w": float(info["tx_power_w"]),
            "intercept_prob": float(info["intercept_prob"]),
            "cumulative_energy_j": float(info["cumulative_energy_j"]),
            # --- 单步反事实（仅被干预步有值）---
            "cf_ai_satisfied": cf_ai_satisfied,
            "cf_shield_satisfied": cf_shield_satisfied,
        })
        if terminated or truncated:
            break

    summary = ec.summarize_episode(env, list(env.sim.results), label=label, policy_name=label)
    summary["decision_summary"] = policy.summary()
    return summary, steps


def _rate(flags: Sequence[bool]) -> Optional[float]:
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)


def step_level_metrics(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """从逐步记录里算出 P2 的四个指标。"""
    total = len(steps)
    if total == 0:
        return {}

    autonomous = [s for s in steps if s["mode"] == MODE_AI]
    intervened = [s for s in steps if s["mode"] != MODE_AI]
    high_risk = [s for s in steps if s["triggered"]]
    low_risk = [s for s in steps if not s["triggered"]]
    shielded = [s for s in steps if s["mode"] == "shield"]
    fell_back = [s for s in steps if s["mode"] == "fallback_rule"]

    def wrong(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
        """错误决策率：该步探测未达标（真值口径）的比例。"""
        return _rate([not s["task_satisfied"] for s in rows])

    def satisfied(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
        return _rate([s["task_satisfied"] for s in rows])

    def mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        return sum(float(s[key]) for s in rows) / len(rows) if rows else None

    # --- 单步反事实：干预是「挽回」还是「变差」---
    # 只统计两边都有试算结果的被干预步。
    evaluable = [
        s for s in intervened
        if s.get("cf_ai_satisfied") is not None and s.get("cf_shield_satisfied") is not None
    ]
    rescued = [s for s in evaluable if not s["cf_ai_satisfied"] and s["cf_shield_satisfied"]]
    harmed = [s for s in evaluable if s["cf_ai_satisfied"] and not s["cf_shield_satisfied"]]
    neutral = [s for s in evaluable if s["cf_ai_satisfied"] == s["cf_shield_satisfied"]]

    return {
        "decision_steps": total,
        "ai_autonomy_rate": len(autonomous) / total,
        "shield_rate": len(shielded) / total,
        "fallback_rate": len(fell_back) / total,
        "intervention_rate": len(intervened) / total,
        "high_risk_rate": len(high_risk) / total,
        # --- 错误决策率（分来源）---
        "wrong_decision_rate_ai": wrong(autonomous),
        "wrong_decision_rate_intervened": wrong(intervened),
        "wrong_decision_rate_overall": wrong(steps),
        # --- 高风险 / 低风险下的任务满足率 ---
        "high_risk_satisfaction_rate": satisfied(high_risk),
        "low_risk_satisfaction_rate": satisfied(low_risk),
        # --- 单步反事实 ---
        "cf_evaluable_steps": float(len(evaluable)),
        "shield_rescue_rate": len(rescued) / len(evaluable) if evaluable else None,
        "shield_harm_rate": len(harmed) / len(evaluable) if evaluable else None,
        "shield_neutral_rate": len(neutral) / len(evaluable) if evaluable else None,
        # --- 不确定度画像 ---
        "mean_q_std_max": mean(steps, "q_std_max"),
        "mean_ood_score": mean(steps, "ood_score"),
        "mean_obs_quality": mean(steps, "obs_quality"),
        "mean_q_margin": mean(steps, "q_margin"),
        "mean_disagreement": mean(steps, "disagreement"),
    }


def aggregate_over_seeds(
    per_seed: Sequence[Dict[str, Any]], label: str, group: str
) -> Dict[str, Any]:
    """把多种子的逐步指标与 episode 汇总聚合成均值±标准差。

    命名约定与 `experiment_config.aggregate_summaries` **保持一致**：
        <key>        = 均值
        <key>__std   = 样本标准差（ddof=1）
    之前误用 `<key>_mean` / `<key>_std` 导致整张表读出来全是 n/a，
    这类命名不一致必须靠测试钉住，不能靠肉眼。
    """
    result = ec.aggregate_summaries(
        [row["episode"] for row in per_seed], label=label
    )
    result["label"] = label
    result["group"] = group

    keys = set()
    for row in per_seed:
        keys.update(row["steps"].keys())
    for key in sorted(keys):
        values = [
            row["steps"][key] for row in per_seed
            if row["steps"].get(key) is not None
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        result[key] = mean
        result[f"{key}__std"] = std
    return result


# ----------------------------------------------------------------------
# 实验臂
# ----------------------------------------------------------------------

def _make_env(args: argparse.Namespace, seed: int) -> Any:
    return ec.make_env_for_seed(
        seed,
        config_path=args.config,
        energy_budget_j=args.energy_budget,
        jitter=not args.no_jitter,
        **ec.observation_kwargs_from_args(args),
    )


def eval_uncertainty_policy(
    args: argparse.Namespace,
    agent: Any,
    label: str,
    group: str,
    fallback_config: Optional[FallbackConfig],
) -> Optional[Dict[str, Any]]:
    """评测一个「集成 DQN + 回退」配置。

    fallback_config=None 表示纯 argmax（不做任何回退），用于隔离
    「不确定度信号本身」与「回退动作」各自的贡献。
    """
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        env = _make_env(args, seed)
        obs_dim = int(env.observation_space.shape[0])
        if obs_dim != agent.config.obs_dim:
            print(f"  [跳过] {label}：观测维度不匹配（{obs_dim} vs {agent.config.obs_dim}）")
            return None

        if fallback_config is None:
            config = FallbackConfig(enabled=False)
        else:
            config = copy.deepcopy(fallback_config)
        policy = UncertaintyAwarePolicy(agent, config)
        episode, steps = run_policy_episode(env, policy, seed, label)
        per_seed.append({"episode": episode, "steps": step_level_metrics(steps)})

    aggregate = aggregate_over_seeds(per_seed, label, group)
    print(
        f"  {label:34s} 自主率={aggregate.get('ai_autonomy_rate', 0.0):.3f}  "
        f"干预率={aggregate.get('intervention_rate', 0.0):.3f}  "
        f"错误率(AI)={_fmt(aggregate.get('wrong_decision_rate_ai'))}  "
        f"错误率(干预)={_fmt(aggregate.get('wrong_decision_rate_intervened'))}  "
        f"综合收益={aggregate.get('composite_reward', 0.0):+.4f}"
    )
    return aggregate


def _fmt(value: Optional[float]) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def eval_plain_dqn(
    args: argparse.Namespace, label: str, model_path: str
) -> Optional[Dict[str, Any]]:
    """普通 DQN 对照臂：没有任何不确定度机制。"""
    if not os.path.exists(model_path):
        print(f"  [跳过] {label}：找不到 {model_path}")
        return None
    agent = DQNAgent.load(model_path, device=args.device)

    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        env = _make_env(args, seed)
        if int(env.observation_space.shape[0]) != agent.config.obs_dim:
            print(f"  [跳过] {label}：观测维度不匹配")
            return None
        obs, _ = env.reset(seed=seed)
        steps: List[Dict[str, Any]] = []
        while True:
            action = agent.select_action(
                obs, greedy=True, action_mask=env.action_masks()
            )
            obs, _r, terminated, truncated, info = env.step(action)
            steps.append({
                "mode": MODE_AI,
                "reason_code": "",
                "triggered": 0,
                "obs_quality": float(env.observation_quality()),
                "q_std_max": 0.0,
                "ood_score": 0.0,
                "q_margin": 0.0,
                "disagreement": 0.0,
                "action": action,
                "ai_action": action,
                "pd_min": float(info["pd_min"]),
                "task_satisfied": bool(info["task_satisfied"]),
                "tx_power_w": float(info["tx_power_w"]),
                "intercept_prob": float(info["intercept_prob"]),
                "cumulative_energy_j": float(info["cumulative_energy_j"]),
            })
            if terminated or truncated:
                break
        episode = ec.summarize_episode(
            env, list(env.sim.results), label=label, policy_name=label
        )
        per_seed.append({"episode": episode, "steps": step_level_metrics(steps)})

    aggregate = aggregate_over_seeds(per_seed, label, "A_无不确定度机制")
    print(
        f"  {label:34s} 自主率={aggregate.get('ai_autonomy_rate', 0.0):.3f}  "
        f"（无回退机制）  错误率={_fmt(aggregate.get('wrong_decision_rate_overall'))}  "
        f"综合收益={aggregate.get('composite_reward', 0.0):+.4f}"
    )
    return aggregate


def eval_observed_rule(args: argparse.Namespace, label: str) -> Optional[Dict[str, Any]]:
    """规则(仅观测) 基准，作为可解释性的下限参考。"""
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        env = _make_env(args, seed)
        policy = BeliefPolicy(RuleBasedPowerPolicy())
        results = ec.run_belief_episode(env, policy, seed)
        episode = ec.summarize_episode(env, results, label=label, policy_name=label)
        steps = [{
            "mode": MODE_AI, "reason_code": "", "triggered": 0,
            "obs_quality": 1.0, "q_std_max": 0.0, "ood_score": 0.0,
            "q_margin": 0.0, "disagreement": 0.0, "action": -1, "ai_action": -1,
            "pd_min": float(r.pd_min), "task_satisfied": bool(r.task_satisfied),
            "tx_power_w": float(r.tx_power_w),
            "intercept_prob": float(r.intercept_prob),
            "cumulative_energy_j": float(r.cumulative_energy_j),
        } for r in results]
        per_seed.append({"episode": episode, "steps": step_level_metrics(steps)})
    aggregate = aggregate_over_seeds(per_seed, label, "B_可解释基准")
    print(
        f"  {label:34s} 错误率={_fmt(aggregate.get('wrong_decision_rate_overall'))}  "
        f"综合收益={aggregate.get('composite_reward', 0.0):+.4f}"
    )
    return aggregate


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------

P2_METRIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("ai_autonomy_rate", "AI自主率"),
    ("intervention_rate", "干预率"),
    ("high_risk_rate", "高风险步占比"),
    ("wrong_decision_rate_ai", "错误率(AI自主)"),
    ("wrong_decision_rate_intervened", "错误率(被干预)"),
    ("shield_rescue_rate", "护盾挽回率"),
    ("high_risk_satisfaction_rate", "高风险满足率"),
    ("low_risk_satisfaction_rate", "低风险满足率"),
    ("mean_q_std_max", "平均集成分歧"),
    ("mean_ood_score", "平均OOD评分"),
    ("mean_obs_quality", "平均观测质量"),
    ("horizon_satisfaction_rate", "任务满足率"),
    ("composite_reward", "综合收益"),
)


def format_p2_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = f"{'配置':<34s}" + "".join(f"{title:>16s}" for _, title in P2_METRIC_COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _ in P2_METRIC_COLUMNS:
            mean = row.get(key)
            std = row.get(f"{key}__std")
            if mean is None:
                cells.append(f"{'n/a':>16s}")
            elif std is None:
                cells.append(f"{float(mean):>16.4f}")
            else:
                cells.append(f"{float(mean):>8.3f}±{float(std):<7.3f}")
        lines.append(f"{row['label']:<34s}" + "".join(cells))
    return "\n".join(lines)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = ["group", "label"]
    seen = set(keys)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _write_text_html(path: str, title: str, sections: Sequence[Tuple[str, Sequence[str]]]) -> str:
    import html as html_mod

    parts = [
        "<!DOCTYPE html>", '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{html_mod.escape(title)}</title>",
        "<style>body{font-family:Consolas,'Microsoft YaHei',monospace;margin:24px;"
        "background:#fafafa;color:#222;line-height:1.55}"
        "h1{font-size:20px}h2{font-size:16px;margin-top:28px;"
        "border-left:4px solid #4472c4;padding-left:8px}"
        "pre{background:#fff;border:1px solid #ddd;border-radius:4px;padding:12px;"
        "overflow-x:auto;font-size:12.5px}</style></head><body>",
        f"<h1>{html_mod.escape(title)}</h1>",
    ]
    for heading, blocks in sections:
        parts.append(f"<h2>{html_mod.escape(heading)}</h2>")
        for block in blocks:
            parts.append(f"<pre>{html_mod.escape(str(block))}</pre>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path


def build_conclusions(rows: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> List[str]:
    by_label = {r["label"]: r for r in rows}
    lines = [f"评测种子数：{len(seeds)}", ""]

    def g(row: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if row is None:
            return None
        value = row.get(key)
        return None if value is None else float(value)

    plain = by_label.get("普通DQN(部分可观测)")
    ens = by_label.get("集成DQN(纯argmax)")
    shield = by_label.get("集成DQN+安全护盾")
    fb = by_label.get("集成DQN+完全回退")

    lines.append("【1】集成带来的信息（不改变动作，只增加不确定度信号）")
    if plain and ens:
        p = g(plain, "composite_reward") or 0.0
        e = g(ens, "composite_reward") or 0.0
        lines.append(
            f"  普通DQN 综合收益 {p:+.4f}；集成DQN(纯argmax) {e:+.4f}（Δ {e - p:+.4f}）"
        )
        lines.append(
            "  说明：两者都不使用回退，差别只在集成平均。若 Δ 很小，"
            "说明集成本身没有改变策略质量，它的价值在于**提供不确定度信号**。"
        )
    else:
        lines.append("  缺少普通 DQN 或集成 DQN 的 checkpoint，无法比较。")
    lines.append("")

    lines.append("【2】触发条件是否定位到了困难状态（用高风险/低风险满足率判断）")
    for name, row in (("安全护盾", shield), ("完全回退", fb)):
        if not row:
            continue
        hr = g(row, "high_risk_satisfaction_rate")
        lr = g(row, "low_risk_satisfaction_rate")
        lines.append(
            f"  {name}：高风险步占比 {g(row, 'high_risk_rate') or 0:.3f}，"
            f"自主率 {g(row, 'ai_autonomy_rate') or 0:.3f}"
        )
        lines.append(f"    任务满足率：高风险步 {_fmt(hr)} vs 低风险步 {_fmt(lr)}")
        high_risk_rate = g(row, "high_risk_rate") or 0.0
        if high_risk_rate >= 0.90 or high_risk_rate <= 0.10:
            lines.append(
                f"    ⚠ 高风险步占比 {high_risk_rate:.3f} 过于极端，"
                "高风险/低风险的分组对比**在本阈值下不可解读**"
                "（其中一组样本太少，均值没有意义）。"
                "要看有意义的对比，请改用阈值灵敏度扫描里更合适的档位，或先校准阈值。"
            )
        elif hr is not None and lr is not None:
            if hr < lr:
                lines.append(
                    "    → 高风险步的满足率确实更低：触发条件**定位到了困难状态**。"
                )
            else:
                lines.append(
                    "    → 高风险步的满足率并不更低：触发条件没有定位到困难状态，"
                    "该阈值设定在本场景下无效，应调整或弃用。"
                )
        lines.append("")

    lines.append("【3】回退动作本身是否起了作用（单步反事实）")
    lines.append(
        "  口径：对被干预的步，在**决策时刻的真实状态**上分别试算"
        "「AI 原本要选的功率」与「实际执行的功率」，比较两者的 Pd 是否达标。"
        "这是单步反事实，不是完整重仿真——它回答「同样状态下换动作会不会更好」，"
        "不回答「换了动作后续会不会连锁变化」（那需要轨迹级反事实）。"
    )
    for name, row in (("安全护盾", shield), ("完全回退", fb)):
        if not row:
            continue
        rescue = g(row, "shield_rescue_rate")
        harm = g(row, "shield_harm_rate")
        lines.append(
            f"  {name}：护盾挽回率 {_fmt(rescue)}（AI 动作本会失败、干预后达标的比例）；"
            f"干预反而变差率 {_fmt(harm)}"
        )
    if shield and g(shield, "shield_rescue_rate") is not None:
        rescue = g(shield, "shield_rescue_rate") or 0.0
        if rescue > 0.0:
            lines.append(
                "  → 存在被挽回的步，说明**护盾确实阻止了一部分本会发生的探测失败**。"
            )
        else:
            lines.append(
                "  → **没有任何一步被挽回**。触发时 AI 的动作与规则给的动作"
                "在达标性上没有差别，护盾在本场景下没有产生实际收益。"
                "这必须如实报告。"
            )
    lines.append("")

    lines.append("【4】错误决策率的分组对照")
    lines.append(
        "  注意：被干预步的错误率通常**高于** AI 自主步，这不是回退失败，"
        "而是因为触发条件本就倾向于在困难步上命中（选择性偏差）。"
        "判断回退有没有用请看第【3】节的单步反事实，而不是直接比这两个错误率。"
    )
    for row in rows:
        lines.append(
            f"  {row['label']:<34s} "
            f"错误率(AI)={_fmt(g(row, 'wrong_decision_rate_ai'))}  "
            f"错误率(干预)={_fmt(g(row, 'wrong_decision_rate_intervened'))}  "
            f"总错误率={_fmt(g(row, 'wrong_decision_rate_overall'))}  "
            f"综合收益={_fmt(g(row, 'composite_reward'))}"
        )
    return lines


# ----------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("======== P2：不确定度感知的可信决策评测 ========")
    print(f"观测模式  : {ec.observation_label(args)}")
    print(f"种子      : {len(args.seeds)} 个")
    print(f"扰动      : {'关' if args.no_jitter else '开'}")
    print(f"回退模式  : {args.fallback_mode}")
    print(f"输出目录  : {args.out_dir}\n")

    rows: List[Dict[str, Any]] = []

    # ---------------- B 组：可解释基准 ----------------
    print("---- B 组：可解释基准 ----")
    rule_row = eval_observed_rule(args, "规则(仅观测)")
    if rule_row:
        rows.append(rule_row)

    # ---------------- A 组：无不确定度机制 ----------------
    if not args.no_plain_dqn:
        print("\n---- A 组：无不确定度机制 ----")
        plain = eval_plain_dqn(args, "普通DQN(部分可观测)", args.pomdp_model)
        if plain:
            rows.append(plain)

    # ---------------- C 组：集成 + 回退 ----------------
    print("\n---- C 组：集成与安全回退 ----")
    if not os.path.exists(args.ensemble_model):
        print(f"  找不到集成 checkpoint：{args.ensemble_model}")
        print("  请先运行：python train_ensemble.py --jitter --observation-mode pomdp")
    else:
        agent = EnsembleDQNAgent.load(args.ensemble_model, device=args.device)
        print(f"  集成模型：{agent.describe()}\n")

        base = FallbackConfig(fallback_mode="shield")
        for label, config, group in (
            ("集成DQN(纯argmax)", None, "C_集成无回退"),
            ("集成DQN+安全护盾",
             FallbackConfig(fallback_mode="shield"), "D_集成+回退"),
            ("集成DQN+完全回退",
             FallbackConfig(fallback_mode="fallback_rule"), "D_集成+回退"),
        ):
            row = eval_uncertainty_policy(args, agent, label, group, config)
            if row:
                rows.append(row)
        del base

    # ---------------- 汇总 ----------------
    print("\n======== 主对比表（多种子 均值±标准差）========")
    print(format_p2_table(rows))
    print("\n======== 结论 ========")
    conclusions = build_conclusions(rows, args.seeds)
    for line in conclusions:
        print(line)

    _write_csv(os.path.join(args.out_dir, "uncertainty_comparison.csv"), rows)
    print(f"\n→ {os.path.join(args.out_dir, 'uncertainty_comparison.csv')}")

    # ---------------- 阈值灵敏度 ----------------
    sweep_rows: List[Dict[str, Any]] = []
    if args.threshold_sweep and os.path.exists(args.ensemble_model):
        print("\n======== 阈值灵敏度（回退率是阈值驱动的，必须一并汇报）========")
        sweep_agent = EnsembleDQNAgent.load(args.ensemble_model, device=args.device)
        for name, kwargs in THRESHOLD_LEVELS:
            config = FallbackConfig(fallback_mode=args.fallback_mode, **kwargs)
            row = eval_uncertainty_policy(
                args, sweep_agent, f"阈值: {name}", "E_阈值灵敏度", config
            )
            if row:
                sweep_rows.append(row)
        if sweep_rows:
            print()
            print(format_p2_table(sweep_rows))
            _write_csv(os.path.join(args.out_dir, "uncertainty_threshold_sweep.csv"), sweep_rows)
            print(f"→ {os.path.join(args.out_dir, 'uncertainty_threshold_sweep.csv')}")

    # ---------------- 信号消融 ----------------
    ablation_rows: List[Dict[str, Any]] = []
    if args.signal_ablation and os.path.exists(args.ensemble_model):
        print("\n======== 不确定度信号消融（哪一路信号在起作用）========")
        abl_agent = EnsembleDQNAgent.load(args.ensemble_model, device=args.device)
        for name, kwargs in SIGNAL_CASES:
            config = FallbackConfig(fallback_mode=args.fallback_mode, **kwargs)
            row = eval_uncertainty_policy(
                args, abl_agent, f"信号: {name}", "F_信号消融", config
            )
            if row:
                ablation_rows.append(row)
        if ablation_rows:
            print()
            print(format_p2_table(ablation_rows))
            _write_csv(os.path.join(args.out_dir, "uncertainty_signal_ablation.csv"), ablation_rows)
            print(f"→ {os.path.join(args.out_dir, 'uncertainty_signal_ablation.csv')}")

    # ---------------- HTML ----------------
    sections: List[Tuple[str, Sequence[str]]] = [
        ("1. 说明", [
            "本报告评测「不确定度感知的可信决策」：集成 DQN 提供集成分歧与 OOD 评分，",
            "策略在不确定度过高时抬升功率（安全护盾）或整体回退到规则策略。",
            "",
            "关键口径：",
            "  * AI自主率 = 最终执行 AI argmax 的步数占比；",
            "  * 干预率   = 动作被护盾抬升或整体回退的步数占比；",
            "  * 错误决策率 = 该步探测未达标（真值口径）的比例，分 AI 自主步与被干预步；",
            "  * 高风险步 = 任一回退触发条件命中的步（无论最终是否干预）。",
            "",
            f"观测设置：{ec.observation_label(args)}；种子 {len(args.seeds)} 个；"
            f"扰动 {'关' if args.no_jitter else '开'}。",
        ]),
        ("2. 主对比表", [format_p2_table(rows)]),
        ("3. 结论", conclusions),
    ]
    if sweep_rows:
        sections.append(("4. 阈值灵敏度", [
            "回退阈值是人工设定的，因此回退率主要反映阈值选择而非智能体的内省能力。",
            format_p2_table(sweep_rows),
        ]))
    if ablation_rows:
        sections.append(("5. 信号消融", [
            "分别只使用集成分歧 / OOD 评分 / 观测质量，检验哪一路信号真正有效。",
            format_p2_table(ablation_rows),
        ]))

    html = _write_text_html(
        os.path.join(args.out_dir, "uncertainty_report.html"),
        "LPI-CogRadar v4.0 — P2 不确定度感知可信决策报告",
        sections,
    )
    print(f"→ {html}")


if __name__ == "__main__":
    main()
