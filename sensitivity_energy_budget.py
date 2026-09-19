"""能量预算敏感性实验。

    python sensitivity_energy_budget.py                       # 训练缺失的模型并跑全部预算
    python sensitivity_energy_budget.py --budgets 1400 1500   # 只跑指定预算
    python sensitivity_energy_budget.py --reuse               # 只评测，不训练

在**相同场景与相同随机种子**下，逐个能量预算比较四类策略：

    规则功率控制     每步取满足探测要求的最低**可行**档
    DQN(贪心)        在该预算下独立训练的同架构 DQN（带动作掩码）
    逐档贪心(短视)   每步在可行档内穷举单步收益最大者
    前瞻规划(非短视) 完整任务视野的滚动前瞻

报告指标：完整视野满足率、平均功率、累计能耗、平均 Pint、平均/累计暴露、综合收益。

为什么同一预算下要**独立训练**一个 DQN
--------------------------------------
能量预算属于任务约束，各预算下的最优策略不同（例如预算 1800 J 时无需牺牲任何步，
预算 1200 J 时必须裁掉更多步）。若拿预算 1400 J 训好的模型去跑所有预算，
测到的是「迁移能力」而不是「该预算下策略的真实上限」，会低估 DQN。
因此这里默认对每个预算用**同一套超参与随机种子**独立训练一个模型，
并把训练产物放在 `output/rl_sensitivity/budget_<B>/` 下。

产物（默认 output/rl_sensitivity/）
    summary.csv               预算 × 策略 的完整指标（宽表，便于画图）
    sensitivity_report.html   报告页（表格 + 折线图）
    sensitivity_curves.png    折线图 PNG（需 matplotlib）
    step_<策略>.csv           最后一个预算的逐步明细
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence

import experiment_config as ec
from metrics import write_curves_html, write_step_metrics_csv
from rl import DQNAgent, silence_numpy_bridge_warning

silence_numpy_bridge_warning()

DEFAULT_OUT_DIR = os.path.join("output", "rl_sensitivity")

# 画折线图用的指标：(键, 中文标题, y 轴标签, 是否越小越好)
PLOT_METRICS = [
    ("horizon_satisfaction_rate", "完整视野满足率", "满足率", False),
    ("avg_tx_power_w", "平均发射功率", "Pt (W)", True),
    ("cumulative_energy_j", "累计能耗", "E (J)", True),
    ("avg_intercept_prob", "平均截获概率 Pint_eff", "Pint", True),
    ("avg_exposure", "平均累计暴露量", "暴露量", True),
    ("cumulative_exposure", "累计暴露", "Σ Pint_inst", True),
    ("composite_reward", "综合收益", "reward", False),
]

# summary.csv 的列
SUMMARY_FIELDS = [
    "budget_j",
    "label",
    "steps",
    "horizon_steps",
    "horizon_satisfaction_rate",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "remaining_energy_j",
    "energy_utilization",
    "avg_intercept_prob",
    "avg_exposure",
    "cumulative_exposure",
    "composite_reward",
    "violation_steps",
    "power_switch_count",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="能量预算敏感性实验（预算 ∈ 任务约束，不改任何物理参数）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=list(ec.DEFAULT_ENERGY_BUDGETS),
                        help="要测试的能量预算（J）")
    parser.add_argument("--seed", type=int, default=ec.DEFAULT_SEED, help="统一随机种子")
    parser.add_argument("--episodes", type=int, default=800,
                        help="每个预算下 DQN 的训练 episode 数")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="训练期间的评测间隔")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                        help="Q 网络隐藏层宽度")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--double-dqn", action="store_true", help="启用 Double DQN")
    parser.add_argument("--reuse", action="store_true",
                        help="复用已有 checkpoint，不重新训练")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="跳过前瞻规划（较慢）")
    parser.add_argument("--device", default="cpu", help="torch 设备")
    return parser


# ----------------------------------------------------------------------

def train_for_budget(args: argparse.Namespace, budget: float, model_path: str) -> None:
    """用 train_dqn.train() 在指定预算下训练一个 DQN（超参完全一致）。"""
    if args.reuse and os.path.exists(model_path):
        print(f"  [复用] {model_path}")
        return
    if os.path.exists(model_path) and args.reuse:
        return

    import train_dqn

    print(f"  [训练] 预算 {budget:.0f} J -> {model_path}")
    train_args = train_dqn.build_arg_parser().parse_args([])
    train_args.config = args.config
    train_args.out_dir = os.path.dirname(model_path)
    train_args.episodes = args.episodes
    train_args.seed = args.seed
    train_args.eval_seed = args.seed
    train_args.eval_every = args.eval_every
    train_args.hidden = args.hidden
    train_args.lr = args.lr
    train_args.gamma = args.gamma
    train_args.double_dqn = args.double_dqn
    train_args.quiet = True
    # 能量预算属于环境构造参数：通过 LpiPowerEnv 的 energy_budget_j 传入
    train_dqn.train(train_args, energy_budget_j=budget)


def evaluate_budget(
    args: argparse.Namespace, budget: float, model_path: str
) -> tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """在给定预算下评测全部策略，返回 (汇总行, 逐步结果)。"""
    env = ec.make_env(args.config, energy_budget_j=budget)
    num_steps = int(env.sim.scenario.num_steps)

    rows: List[Dict[str, Any]] = []
    runs: Dict[str, List[Any]] = {}

    # --- DQN ---
    agent = DQNAgent.load(model_path, device=args.device)
    results = ec.run_dqn_episode(env, agent, args.seed)
    summary = ec.summarize_episode(env, results, "DQN(贪心)", "DQN(greedy, 动作掩码)")
    rows.append(summary)
    runs["DQN(贪心)"] = results

    # --- 脚本策略 ---
    specs = ec.scripted_policy_specs(
        num_steps,
        random_seed=ec.RANDOM_POLICY_SEED,
        include_fixed=False,
        include_rule=True,
        include_random=False,
        include_myopic=True,
        include_lookahead=not args.no_lookahead,
    )
    for spec in specs:
        policy = spec.factory()
        results = ec.run_scripted_episode(env, policy, args.seed)
        rows.append(ec.summarize_episode(env, results, spec.label, policy.describe()))
        runs[spec.label] = results

    # 按固定顺序输出（规则 / DQN / 短视 / 前瞻）
    order = ["规则功率控制", "DQN(贪心)", "逐档贪心(短视)", "前瞻规划(非短视)"]
    rank = {name: i for i, name in enumerate(order)}
    rows.sort(key=lambda r: rank.get(r["label"], 99))
    return rows, runs


# ----------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    last_runs: Dict[str, List[Any]] = {}
    last_budget = None

    print("======== 能量预算敏感性实验 ========")
    print(f"场景      : {args.config}")
    print(f"预算档位  : {[int(b) for b in args.budgets]} J")
    print(f"统一种子  : {args.seed}")
    print(f"训练规模  : 每个预算 {args.episodes} episode（相同超参与种子）\n")

    for budget in args.budgets:
        budget_dir = os.path.join(args.out_dir, f"budget_{int(budget)}")
        os.makedirs(budget_dir, exist_ok=True)
        model_path = os.path.join(budget_dir, "dqn_agent.pt")

        print(f"---- 能量预算 {budget:.0f} J ----")
        train_for_budget(args, budget, model_path)

        rows, runs = evaluate_budget(args, budget, model_path)
        for row in rows:
            all_rows.append(row)
            long_rows.append({"budget_j": budget, **{k: row.get(k, "") for k in SUMMARY_FIELDS[1:]}})
            print(f"    {row['label']:<18} 满足率={row['horizon_satisfaction_rate']:.4f} "
                  f"功率={row['avg_tx_power_w']:6.2f}W 能耗={row['cumulative_energy_j']:7.1f}J "
                  f"Pint={row['avg_intercept_prob']:.4f} 暴露={row['avg_exposure']:.4f} "
                  f"收益={row['composite_reward']:+.4f}")

        last_runs = runs
        last_budget = budget
        print()

    # --- CSV ---
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in long_rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    step_csv_paths = []
    if last_budget is not None:
        for label, results in last_runs.items():
            safe = label.replace("(", "_").replace(")", "").replace(" ", "")
            path = os.path.join(args.out_dir, f"step_{safe}.csv")
            write_step_metrics_csv(results, path)
            step_csv_paths.append(path)

    # --- HTML 报告（预算为 x 轴，每个指标一张折线图）---
    curves = []
    budgets = sorted({float(r["budget_j"]) for r in long_rows})
    for key, title, ylabel, lower_better in PLOT_METRICS:
        series: Dict[str, Dict[str, list]] = {}
        for label in ["规则功率控制", "DQN(贪心)", "逐档贪心(短视)", "前瞻规划(非短视)"]:
            xs, ys = [], []
            for b in budgets:
                match = [r for r in long_rows
                         if float(r["budget_j"]) == b and r["label"] == label]
                if not match or match[0].get(key, "") == "":
                    continue
                xs.append(b)
                ys.append(float(match[0][key]))
            if xs:
                series[label] = {"x": xs, "y": ys}
        if series:
            curves.append(
                {
                    "title": f"{title}（{'越小越好' if lower_better else '越大越好'}）",
                    "x_label": "能量预算 (J)",
                    "y_label": ylabel,
                    "series": series,
                }
            )

    report_path = os.path.join(args.out_dir, "sensitivity_report.html")
    write_curves_html(
        curves,
        report_path,
        page_title="能量预算敏感性实验 —— 低截获雷达功率调控",
        subtitle=f"相同场景与种子（seed={args.seed}）；"
        f"每个预算下的 DQN 均为该预算独立训练（{args.episodes} episode，同超参）",
        metadata={
            "预算档位 (J)": [int(b) for b in budgets],
            "统一随机种子": args.seed,
            "DQN 训练 episode/预算": args.episodes,
            "Q 网络": args.hidden,
            "学习率": args.lr,
            "折扣因子 γ": args.gamma,
            "Double DQN": args.double_dqn,
            "前瞻视野": f"{ec.lookahead_horizon(ec.scenario_horizon(args.config))} 步"
            f"（任务步数+1）",
            "说明": "能量预算属于任务约束；本实验未改动任何雷达/目标/侦察机/干扰机物理参数",
        },
    )

    png_path = os.path.join(args.out_dir, "sensitivity_curves.png")
    _write_png(curves, png_path)

    print("======== 输出文件 ========")
    for path in [summary_csv, report_path, png_path] + step_csv_paths:
        if os.path.exists(path):
            print(f"  {path}")

    # 结论速览
    print("\n======== 敏感性结论速览 ========")
    for key, title, _, lower_better in PLOT_METRICS[:1]:
        print(f"  {title}随预算的变化：")
        for label in ["规则功率控制", "DQN(贪心)", "逐档贪心(短视)", "前瞻规划(非短视)"]:
            values = []
            for b in budgets:
                match = [r for r in long_rows
                         if float(r["budget_j"]) == b and r["label"] == label]
                if match:
                    values.append(f"{b:.0f}J:{float(match[0][key]):.4f}")
            if values:
                print(f"    {label:<18} " + "  ".join(values))


def _write_png(curves: Sequence[Dict[str, Any]], path: str) -> None:
    """可选：matplotlib 输出折线图 PNG（缺字体时自动退回英文标题）。"""
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

    n = len(curves)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for axis, curve in zip(axes_list, curves):
        for name, series in curve["series"].items():
            axis.plot(series["x"], series["y"], marker="o", ms=4, lw=1.6, label=name)
        axis.set_title(curve["title"], fontsize=10)
        axis.set_xlabel(curve["x_label"], fontsize=9)
        axis.set_ylabel(curve["y_label"], fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    for axis in axes_list[len(curves):]:
        axis.axis("off")

    figure.suptitle(
        "能量预算敏感性 —— 低截获雷达功率调控" if use_cjk
        else "Energy budget sensitivity - LPI radar power control",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
