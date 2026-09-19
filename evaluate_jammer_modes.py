"""固定干扰 vs 自适应智能干扰机：分别评测。

    python evaluate_jammer_modes.py
    python evaluate_jammer_modes.py --policies rule dqn myopic lookahead
    python evaluate_jammer_modes.py --model output/rl/dqn_agent.pt

对**同一个策略集合**分别在两种对手下跑一遍：

    固定干扰       预置时间窗 + 随机强度起伏（开环，第三版默认）
    自适应智能干扰 依据 ESM 累计暴露、雷达辐射强度与干扰有效性切换四种动作（闭环）

刻意把两者**分开评测**并各自出表，而不是混在一张表里比高低：

* 自适应干扰机改变了环境本身，跨对手比较策略性能没有意义；
* 更要紧的是**不要**把「规则自适应对手」宣传成「学习型对手」——
  它只是按固定判据切换动作的状态机，本脚本会在报告里明确标注这一点。

产物（默认 output/jammer_modes/）
    summary.csv               两种对手 × 各策略的完整指标
    jammer_trace_*.csv        自适应干扰机的逐步决策轨迹（敌我双方动作记录）
    jammer_modes_report.html  报告页（表格 + 曲线）
    jammer_modes_curves.png   曲线 PNG（需 matplotlib）
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence

import experiment_config as ec
from metrics import write_curves_html, write_step_metrics_csv
from models.adaptive_jammer import MODE_CN
from rl import DQNAgent, silence_numpy_bridge_warning

silence_numpy_bridge_warning()

DEFAULT_OUT_DIR = os.path.join("output", "jammer_modes")

#: 对比的策略（按此顺序输出）
POLICY_ORDER = ["固定功率基线(80W)", "规则功率控制", "随机策略",
                "DQN(贪心)", "逐档贪心(短视)", "前瞻规划(非短视)"]

SUMMARY_FIELDS = [
    "jammer_mode",
    "label",
    "steps",
    "horizon_satisfaction_rate",
    "violation_rate",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "avg_intercept_prob",
    "avg_exposure",
    "cumulative_exposure",
    "composite_reward",
    "jammed_steps",
    "avg_jam_noise_ratio",
    "jammer_action_counts",
]

PLOT_METRICS = [
    ("horizon_satisfaction_rate", "探测任务满足率", "满足率", False),
    ("violation_rate", "约束违反率", "违反率", True),
    ("avg_tx_power_w", "平均发射功率", "Pt (W)", True),
    ("cumulative_energy_j", "累计能耗", "E (J)", True),
    ("avg_exposure", "平均累计暴露", "暴露量", True),
    ("composite_reward", "综合收益", "reward", False),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定干扰 vs 自适应智能干扰机 分别评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=os.path.join("output", "rl", "dqn_agent.pt"),
                        help="DQN 模型路径")
    parser.add_argument("--seed", type=int, default=ec.DEFAULT_SEED)
    parser.add_argument("--energy-budget", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"未找到 DQN 模型 {args.model}，请先训练 train_dqn.py")
    agent = DQNAgent.load(args.model, device=args.device)

    num_steps = ec.scenario_horizon(args.config, args.energy_budget)
    specs = {
        s.label: s
        for s in ec.scripted_policy_specs(num_steps, include_myopic=True, include_lookahead=True)
    }

    print(f"======== {ec.PROJECT_NAME}：固定干扰 vs 自适应智能干扰机 ========")
    print(f"场景 {args.config} | 种子 {args.seed} | 能量预算 "
          f"{ec.make_env(args.config, args.energy_budget).sim.radar.energy_budget_j:.0f} J")
    print("注意：自适应干扰机是**规则型**对手（按态势判据切换四种动作的状态机），")
    print("      不是学习型对手。请勿在报告里把它宣传成 RL 对手。\n")

    rows: List[Dict[str, Any]] = []
    run_store: Dict[str, Dict[str, List[Any]]] = {}

    for jammer_mode, use_adaptive in (("固定干扰", False), ("自适应智能干扰", True)):
        env = ec.make_env(args.config, energy_budget_j=args.energy_budget)
        if use_adaptive:
            env.sim.apply_overrides(adaptive_jammer=True)

        print(f"---- 对手：{jammer_mode} ----")
        run_store[jammer_mode] = {}

        for label in POLICY_ORDER:
            if label == "DQN(贪心)":
                results = ec.run_dqn_episode(env, agent, args.seed)
                policy_name = "DQN(greedy, 动作掩码)"
            else:
                spec = specs.get(label)
                if spec is None:
                    continue
                policy = spec.factory()
                results = ec.run_scripted_episode(env, policy, args.seed)
                policy_name = policy.describe()

            summary = ec.summarize_episode(env, results, label, policy_name)
            run_store[jammer_mode][label] = results

            traces = env.sim.jammer_action_traces()
            action_counts = ""
            for jammer_id, trace in traces.items():
                counts: Dict[str, int] = {}
                for item in trace:
                    counts[item["mode_cn"]] = counts.get(item["mode_cn"], 0) + 1
                action_counts = "、".join(f"{k}×{v}" for k, v in sorted(counts.items()))

            rows.append(
                {
                    "jammer_mode": jammer_mode,
                    "label": label,
                    **{k: summary.get(k) for k in SUMMARY_FIELDS[2:]},
                    "jammer_action_counts": action_counts,
                }
            )
            print(
                f"  {label:<18} 满足率={summary['horizon_satisfaction_rate']:.4f}  "
                f"违反率={summary['violation_rate']:.4f}  "
                f"功率={summary['avg_tx_power_w']:6.2f}W  "
                f"能耗={summary['cumulative_energy_j']:7.1f}J  "
                f"Pint={summary['avg_intercept_prob']:.4f}  "
                f"暴露={summary['avg_exposure']:.4f}  "
                f"收益={summary['composite_reward']:+.4f}"
                + (f"  | 干扰动作：{action_counts}" if action_counts else "")
            )
        print()

    # --- CSV ---
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    # --- 干扰机决策轨迹 ---
    trace_paths: List[str] = []
    env_trace = ec.make_env(args.config, energy_budget_j=args.energy_budget)
    env_trace.sim.apply_overrides(adaptive_jammer=True)
    ec.run_scripted_episode(env_trace, ec.scripted_policy_specs(num_steps)[1].factory(),
                            args.seed)
    for jammer_id, trace in env_trace.sim.jammer_action_traces().items():
        path = os.path.join(args.out_dir, f"jammer_trace_{jammer_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            fields = ["step_index", "applies_to_step", "time", "jammer_id", "mode",
                      "mode_cn", "threat", "power_scale", "exposure",
                      "radar_power_w", "radar_satisfied", "radiation_ratio",
                      "recent_satisfaction_rate"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in trace:
                writer.writerow({k: item.get(k, "") for k in fields})
        trace_paths.append(path)

    # --- 逐步明细 ---
    step_paths: List[str] = []
    for jammer_mode, per_policy in run_store.items():
        for label, results in per_policy.items():
            safe = f"{jammer_mode}_{label}".replace("(", "_").replace(")", "").replace(" ", "")
            path = os.path.join(args.out_dir, f"step_{safe}.csv")
            write_step_metrics_csv(results, path)
            step_paths.append(path)

    # --- HTML / PNG ---
    curves = []
    for key, title, ylabel, lower_better in PLOT_METRICS:
        series: Dict[str, Dict[str, list]] = {}
        for label in POLICY_ORDER:
            xs, ys = [], []
            for index, jammer_mode in enumerate(("固定干扰", "自适应智能干扰")):
                match = [r for r in rows if r["jammer_mode"] == jammer_mode
                         and r["label"] == label]
                if not match or match[0].get(key) is None:
                    continue
                xs.append(index)
                ys.append(float(match[0][key]))
            if xs:
                series[label] = {"x": xs, "y": ys}
        if series:
            curves.append(
                {
                    "title": f"{title}（x=0 固定干扰，x=1 自适应干扰）",
                    "x_label": "对手类型",
                    "y_label": ylabel,
                    "series": series,
                }
            )

    report_path = os.path.join(args.out_dir, "jammer_modes_report.html")
    write_curves_html(
        curves,
        report_path,
        page_title=f"{ec.PROJECT_NAME}：固定干扰 vs 自适应智能干扰机",
        subtitle=("同一策略集合分别在两种对手下评测；自适应干扰机为**规则型**对手"
                  "（按 ESM 累计暴露、雷达辐射强度、干扰有效性切换"
                  "不干扰/低功率压制/高功率压制/间歇干扰四种动作）"),
        metadata={
            "场景": os.path.basename(args.config),
            "种子": args.seed,
            "能量预算 (J)": ec.make_env(args.config, args.energy_budget).sim.radar.energy_budget_j,
            "自适应干扰动作": "、".join(MODE_CN.values()),
            "对手性质": "规则型状态机（非学习型对手）",
        },
    )

    png_path = os.path.join(args.out_dir, "jammer_modes_curves.png")
    _write_png(curves, png_path)

    # --- 结论 ---
    print("======== 两种对手下的表现差异（以规则功率控制为例）========")
    for label in ("规则功率控制", "DQN(贪心)"):
        fixed = next((r for r in rows if r["jammer_mode"] == "固定干扰" and r["label"] == label), None)
        adapt = next((r for r in rows if r["jammer_mode"] == "自适应智能干扰" and r["label"] == label), None)
        if fixed and adapt:
            print(
                f"  {label}：满足率 {fixed['horizon_satisfaction_rate']:.4f} → "
                f"{adapt['horizon_satisfaction_rate']:.4f}；"
                f"能耗 {fixed['cumulative_energy_j']:.1f} → {adapt['cumulative_energy_j']:.1f} J；"
                f"收益 {fixed['composite_reward']:+.4f} → {adapt['composite_reward']:+.4f}"
            )

    print("\n======== 输出文件 ========")
    for path in [summary_csv, report_path, png_path] + trace_paths + step_paths:
        if os.path.exists(path):
            print(f"  {path}")


def _write_png(curves: Sequence[Dict[str, Any]], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        return

    use_cjk = False
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate), fallback_to_default=False
            )
        except Exception:
            continue
        matplotlib.rcParams["font.sans-serif"] = [candidate]
        matplotlib.rcParams["axes.unicode_minus"] = False
        use_cjk = True
        break

    if not curves:
        return
    cols = 3
    rows_n = (len(curves) + cols - 1) // cols
    figure, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.4 * rows_n))
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for axis, curve in zip(axes_list, curves):
        for name, series in curve["series"].items():
            axis.plot(series["x"], series["y"], marker="o", ms=5, lw=1.5, label=name)
        axis.set_title(curve["title"], fontsize=10)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["fixed", "adaptive"], fontsize=8)
        axis.set_ylabel(curve["y_label"], fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    for axis in axes_list[len(curves):]:
        axis.axis("off")

    figure.suptitle(
        "固定干扰 vs 自适应智能干扰机（规则型对手）" if use_cjk
        else "Fixed vs adaptive jammer",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
