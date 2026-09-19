"""多随机种子评测：避免结论只依赖 seed=42。

    python evaluate_multiseed.py
    python evaluate_multiseed.py --seeds 42 7 13 21 33 --model output/rl/dqn_agent.pt

在**同一份场景、同一套策略定义**下，对每个种子各跑一遍全部策略，
然后报告「均值 ± 样本标准差」，并给出 DQN 相对基线的**逐种子配对差异**：

    配对差值 d_i = DQN_i − 基线_i（i = 1..n_seeds）
    报告 mean(d) ± std(d)、DQN 胜出的种子数，以及配对 t 统计量
    t = mean(d) / (std(d)/√n)。n=10 时双侧 α=0.05 的临界值为 2.262，
    因此 |t| > 2.262 即可认为差异在 5% 水平上显著。

这样就不必依赖单个种子的偶然结果。

产物（默认 output/rl_multiseed/）
    per_seed.csv          每个 seed × 每个策略 的完整指标
    summary_mean_std.csv  各策略的 均值 / 标准差 / 最小值 / 最大值
    paired_vs_baseline.csv DQN 相对每个基线的逐种子配对差异统计
    multiseed_report.html 报告页（表格 + 跨种子折线图）
    multiseed_curves.png  折线图 PNG（需 matplotlib）
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence, Tuple

import experiment_config as ec
from metrics import write_curves_html
from rl import DQNAgent, silence_numpy_bridge_warning

silence_numpy_bridge_warning()

DEFAULT_MODEL_PATH = os.path.join("output", "rl", "dqn_agent.pt")
DEFAULT_OUT_DIR = os.path.join("output", "rl_multiseed")

# 需要跨种子统计的指标：(键, 中文标题, y 轴标签, 是否越小越好)
METRICS: List[Tuple[str, str, str, bool]] = [
    ("horizon_satisfaction_rate", "完整视野满足率", "满足率", False),
    ("avg_tx_power_w", "平均发射功率", "Pt (W)", True),
    ("cumulative_energy_j", "累计能耗", "E (J)", True),
    ("avg_intercept_prob", "平均截获概率 Pint_eff", "Pint", True),
    ("avg_exposure", "平均累计暴露量", "暴露量", True),
    ("cumulative_exposure", "累计暴露", "Σ Pint_inst", True),
    ("composite_reward", "综合收益", "reward", False),
]

PAIRED_BASELINES = ("规则功率控制", "逐档贪心(短视)", "前瞻规划(非短视)", "随机策略")

#: n=10、df=9、双侧 α=0.05 的配对 t 临界值
T_CRITICAL_DF9 = 2.262


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多随机种子评测（报告均值 ± 标准差与配对差异）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="DQN checkpoint 路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ec.DEFAULT_SEEDS),
                        help="用于评测的随机种子列表")
    parser.add_argument("--energy-budget", type=float, default=None,
                        help="覆盖能量预算（J）；默认用配置值")
    parser.add_argument("--no-jitter", action="store_true",
                        help="不做初始条件扰动：此时场景只随种子改变干扰强度起伏。"
                             "注意该起伏摆幅（~0.83 dB）小于档位间隔（~1.46 dB），"
                             "确定性策略的标准差会恒为 0，多种子统计失去意义——"
                             "所以默认开启扰动（只扰动目标/干扰机初始位置与干扰时间窗）")
    parser.add_argument("--device", default="cpu", help="torch 设备")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"未找到 DQN 模型 {args.model}。请先训练：\n"
            f"  D:\\anaconda\\envs\\pytorch_env\\python.exe train_dqn.py"
        )

    env = ec.make_env(args.config, energy_budget_j=args.energy_budget)
    num_steps = int(env.sim.scenario.num_steps)
    budget = float(env.sim.radar.energy_budget_j)
    agent = DQNAgent.load(args.model, device=args.device)
    use_jitter = not args.no_jitter

    print("======== 多随机种子评测 ========")
    print(f"场景配置 : {args.config}")
    print(f"能量预算 : {budget:.0f} J（硬约束）")
    print(f"种子集合 : {list(args.seeds)}（共 {len(args.seeds)} 个）")
    print(f"初始条件扰动 : {'开启' if use_jitter else '关闭'}"
          + ("（目标/干扰机初始位置、干扰时间窗按种子随机化；"
             "不触碰任何物理参数）" if use_jitter else
             "（仅干扰强度起伏随种子变化，确定性策略标准差将为 0）"))
    print(f"模型     : {args.model}")
    print(f"策略网络 : {agent.describe()}\n")

    labels = ["固定功率基线(80W)", "规则功率控制", "随机策略", "DQN(贪心)",
              "逐档贪心(短视)", "前瞻规划(非短视)"]
    specs = {s.label: s for s in ec.scripted_policy_specs(
        num_steps,
        random_seed=ec.RANDOM_POLICY_SEED,
        include_fixed=True,
        include_rule=True,
        include_random=True,
        include_myopic=True,
        include_lookahead=True,
    )}

    per_seed: Dict[str, Dict[int, Dict[str, Any]]] = {label: {} for label in labels}

    for seed in args.seeds:
        # 每个种子用**独立构造**的环境：扰动模式下初始条件随种子变化
        seed_env = ec.make_env_for_seed(
            seed, args.config, energy_budget_j=args.energy_budget, jitter=use_jitter
        )
        print(f"---- seed = {seed} ----")
        line_parts = []
        for label in labels:
            if label == "DQN(贪心)":
                results = ec.run_dqn_episode(seed_env, agent, seed)
                summary = ec.summarize_episode(
                    seed_env, results, label, "DQN(greedy, 动作掩码)"
                )
            else:
                policy = specs[label].factory()
                results = ec.run_scripted_episode(seed_env, policy, seed)
                summary = ec.summarize_episode(seed_env, results, label, policy.describe())

            per_seed[label][seed] = summary
            line_parts.append(f"{label.split('(')[0]}={summary['horizon_satisfaction_rate']:.3f}")
        print("   满足率：" + "  ".join(line_parts))

    # --- 逐种子明细 CSV ---
    per_seed_csv = os.path.join(args.out_dir, "per_seed.csv")
    metric_keys = [m[0] for m in METRICS]
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "label", "steps", "violation_steps"] + metric_keys)
        for seed in args.seeds:
            for label in labels:
                s = per_seed[label][seed]
                writer.writerow(
                    [seed, label, s["steps"], s["violation_steps"]]
                    + [f"{s[k]:.6f}" for k in metric_keys]
                )

    # --- 聚合：均值 ± 标准差 ---
    aggregates = [
        ec.aggregate_summaries(
            [per_seed[label][seed] for seed in args.seeds], label,
            per_seed[label][args.seeds[0]]["policy"],
        )
        for label in labels
    ]

    agg_csv = os.path.join(args.out_dir, "summary_mean_std.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["label", "n_seeds"]
        for key in metric_keys:
            fields += [key, f"{key}__std", f"{key}__min", f"{key}__max"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in aggregates:
            writer.writerow({k: row.get(k, "") for k in fields})

    # --- 配对差异：DQN vs 各基线 ---
    paired_rows = _paired_statistics(per_seed, args.seeds, "DQN(贪心)", PAIRED_BASELINES)
    paired_csv = os.path.join(args.out_dir, "paired_vs_baseline.csv")
    with open(paired_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["baseline", "metric", "n", "wins", "mean_diff", "std_diff", "t_stat"]
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    # --- 控制台报告 ---
    print("\n======== 各策略跨种子统计（均值 ± 标准差，n=%d）========" % len(args.seeds))
    print(ec.format_aggregate_table(aggregates))

    print("\n======== DQN 相对各基线的逐种子配对差异（综合收益）========")
    print(f"  {'基线':<18}{'胜出种子数':>12}{'均值差':>12}{'标准差':>12}{'t 统计量':>12}{'显著':>8}")
    for row in paired_rows:
        if row["metric"] != "composite_reward":
            continue
        significant = "是" if abs(row["t_stat"]) > T_CRITICAL_DF9 else "否"
        print(f"  {row['baseline']:<18}{row['wins']:>7}/{row['n']:<4}"
              f"{row['mean_diff']:>+12.4f}{row['std_diff']:>12.4f}"
              f"{row['t_stat']:>+12.3f}{significant:>8}")
    print(f"  （n={len(args.seeds)}、双侧 α=0.05 的配对 t 临界值 ≈ {T_CRITICAL_DF9}；"
          f"超过即认为差异显著）")

    print("\n======== 各策略满足率跨种子稳定性 ========")
    for row in aggregates:
        std = row.get("horizon_satisfaction_rate__std", 0.0)
        rng = (row.get("horizon_satisfaction_rate__max", 0.0)
               - row.get("horizon_satisfaction_rate__min", 0.0))
        print(f"  {row['label']:<18} 均值={row['horizon_satisfaction_rate']:.4f}  "
              f"标准差={std:.4f}  极差={rng:.4f}")

    # --- HTML / PNG ---
    curves = []
    for key, title, ylabel, lower_better in METRICS:
        series: Dict[str, Dict[str, list]] = {}
        for label in labels:
            series[label] = {
                "x": [i + 1 for i in range(len(args.seeds))],
                "y": [float(per_seed[label][seed][key]) for seed in args.seeds],
            }
        curves.append(
            {
                "title": f"{title}（{'越小越好' if lower_better else '越大越好'}）",
                "x_label": "种子序号（对应 --seeds 顺序）",
                "y_label": ylabel,
                "series": series,
            }
        )

    report_path = os.path.join(args.out_dir, "multiseed_report.html")
    write_curves_html(
        curves,
        report_path,
        page_title="多随机种子评测 —— 低截获雷达功率调控",
        subtitle=f"共 {len(args.seeds)} 个种子：{list(args.seeds)}；"
        f"同一场景模板、同一能量预算 {budget:.0f} J、同一套策略定义；"
        + ("按种子随机化目标/干扰机初始位置与干扰时间窗（不触碰物理参数）"
           if use_jitter else "仅干扰强度起伏随种子变化"),
        metadata={
            "种子列表": list(args.seeds),
            "初始条件扰动": "开启" if use_jitter else "关闭",
            "能量预算 (J)": budget,
            "DQN 模型": os.path.basename(args.model),
            "DQN 网络": f"hidden={tuple(agent.config.hidden_sizes)}, γ={agent.config.gamma}",
            "前瞻视野": f"{ec.lookahead_horizon(num_steps)} 步（任务步数+1）",
        },
    )

    png_path = os.path.join(args.out_dir, "multiseed_curves.png")
    _write_png(curves, png_path)

    print("\n======== 输出文件 ========")
    for path in [per_seed_csv, agg_csv, paired_csv, report_path, png_path]:
        if os.path.exists(path):
            print(f"  {path}")


# ----------------------------------------------------------------------

def _paired_statistics(
    per_seed: Dict[str, Dict[int, Dict[str, Any]]],
    seeds: Sequence[int],
    treatment: str,
    baselines: Sequence[str],
) -> List[Dict[str, Any]]:
    """计算 treatment 相对每个基线、每个指标的配对差异统计。"""
    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        if baseline not in per_seed:
            continue
        for key, _, _, lower_is_better in METRICS:
            diffs = [
                float(per_seed[treatment][s][key]) - float(per_seed[baseline][s][key])
                for s in seeds
            ]
            n = len(diffs)
            mean = sum(diffs) / n if n else 0.0
            if n > 1:
                var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
                std = var ** 0.5
            else:
                std = 0.0
            t_stat = (mean / (std / (n ** 0.5))) if std > 1e-12 and n > 1 else 0.0
            wins = sum(
                1
                for d in diffs
                if (d < 0 if lower_is_better else d > 0)
            )
            rows.append(
                {
                    "baseline": baseline,
                    "metric": key,
                    "n": n,
                    "wins": wins,
                    "mean_diff": round(mean, 6),
                    "std_diff": round(std, 6),
                    "t_stat": round(t_stat, 4),
                }
            )
    return rows


def _write_png(curves: Sequence[Dict[str, Any]], path: str) -> None:
    """可选：matplotlib 输出跨种子折线图。"""
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
    rows = (len(curves) + cols - 1) // cols
    figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for axis, curve in zip(axes_list, curves):
        for name, series in curve["series"].items():
            axis.plot(series["x"], series["y"], marker="o", ms=4, lw=1.4, label=name)
        axis.set_title(curve["title"], fontsize=10)
        axis.set_xlabel(curve["x_label"], fontsize=9)
        axis.set_ylabel(curve["y_label"], fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    for axis in axes_list[len(curves):]:
        axis.axis("off")

    figure.suptitle(
        "多随机种子评测 —— 低截获雷达功率调控" if use_cjk
        else "Multi-seed evaluation - LPI radar power control",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
