"""统一测试入口：在**完全相同的场景与随机种子**下对比所有策略。

    python evaluate_dqn.py
    python evaluate_dqn.py --model output/rl/dqn_agent.pt --seed 42
    python evaluate_dqn.py --energy-budget 1200          # 换个能量预算看可行域

对比对象
--------
    固定功率基线      80 W 满功率（能量不足时被裁剪到可行最高档）
    规则功率控制      每步取满足探测要求的最低**可行**档
    随机策略          在可行档位内随机
    DQN（贪心）        --model 指定的训练产物（带动作掩码）
    逐档贪心(短视)    每步在可行档内穷举单步收益最大者（考虑不到未来）
    前瞻规划(非短视)  完整任务视野的滚动前瞻（非短视参考，需要完整模型知识）

统一性保证
----------
1. 所有策略都在同一份配置、同一个 `--seed`、同一个能量预算下运行；
2. 策略定义、前瞻视野规则、episode 运行方式全部来自 `experiment_config`，
   与 `diagnose_temporal_coupling.py` / `sensitivity_energy_budget.py` /
   `evaluate_multiseed.py` 共用同一套工厂，从结构上杜绝脚本间结论漂移；
3. 指标一律由 `metrics.collector.summarize_run` 计算，与 `main.py` 口径一致；
4. 启动时**自动复核**固定/规则/随机三组是否复现 `main.py` 的历史值，不一致立即告警。

产物（默认 output/rl/）
    comparison.csv                  各策略核心指标对照
    step_<策略>.csv                 各策略逐步指标
    dqn_comparison_report.html      对比报告（表格 + 量化结论 + 曲线图）
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import experiment_config as ec
from metrics import (
    format_summary_table,
    write_html_report,
    write_step_metrics_csv,
    write_summary_csv,
)
from rl import (
    DQNAgent,
    LagrangianDQNAgent,
    silence_numpy_bridge_warning,
)

silence_numpy_bridge_warning()


@dataclass
class EvalItem:
    """评测清单里的一项：脚本策略、主 DQN、或约束版 DQN。"""

    label: str
    csv_name: str
    is_dqn: bool = False
    spec: Optional[ec.PolicySpec] = None
    agent: Any = None

DEFAULT_MODEL_PATH = os.path.join("output", "rl", "dqn_agent.pt")
DEFAULT_OUT_DIR = os.path.join("output", "rl")

# main.py 在当前场景（第二版：能量硬约束 + 累计暴露 + 未达标惩罚）下的结果，
# 用于回归核对——防止场景被改动后还拿旧结论做对比。
# 数值来自 `python main.py`，改动环境后必须同步更新。
EXPECTED_BASELINES = {
    "固定功率基线(80W)": -3.1912,
    "规则功率控制": 0.3510,
    "随机策略": -0.6289,
}
REGRESSION_METRIC = "composite_reward"
TOLERANCE = 5e-4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DQN 与基线策略的统一测试（同一场景、同一种子、同一能量预算）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="DQN checkpoint 路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--seed", type=int, default=ec.DEFAULT_SEED,
                        help="测试场景种子（默认 42，即场景默认种子）")
    parser.add_argument("--episodes", type=int, default=1,
                        help="每个策略重复的 episode 数（>1 时指标取平均）")
    parser.add_argument("--random-seed", type=int, default=ec.RANDOM_POLICY_SEED,
                        help="随机策略自身的种子（与场景种子区分开）")
    parser.add_argument("--energy-budget", type=float, default=None,
                        help="覆盖能量预算（J）；默认用配置里的值。只改任务约束，不碰物理参数")
    parser.add_argument("--adaptive-jammer", action="store_true",
                        help="把干扰机切换为规则自适应智能干扰机（默认固定时间窗）")
    parser.add_argument("--safe-model", default=None,
                        help="可选：拉格朗日约束 DQN 的 checkpoint，"
                             "会额外加入一行「DQN(约束版)」对照")
    parser.add_argument("--no-oracle", action="store_true",
                        help="不运行逐档贪心(短视)基线")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="不运行前瞻规划参考（完整视野，较慢）")
    parser.add_argument("--device", default="cpu", help="torch 设备")
    return parser


def evaluate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    os.makedirs(args.out_dir, exist_ok=True)

    env = ec.make_env(args.config, energy_budget_j=args.energy_budget)
    if args.adaptive_jammer:
        env.sim.apply_overrides(adaptive_jammer=True)
    num_steps = int(env.sim.scenario.num_steps)
    budget = float(env.sim.radar.energy_budget_j)

    print("======== DQN 统一测试 ========")
    print(f"项目     : {ec.PROJECT_NAME}")
    print(f"场景配置 : {args.config}")
    print(f"干扰机   : {'规则自适应智能干扰机' if args.adaptive_jammer else '固定时间窗（默认）'}")
    print(f"测试种子 : {args.seed}（所有策略完全一致）")
    print(f"能量预算 : {budget:.0f} J（硬约束）")
    print(f"重复次数 : {args.episodes} episode / 策略")

    # --- 载入 DQN ---
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"未找到 DQN 模型 {args.model}。请先训练：\n"
            f"  D:\\anaconda\\envs\\pytorch_env\\python.exe train_dqn.py"
        )
    agent = DQNAgent.load(args.model, device=args.device)
    metadata = DQNAgent.read_metadata(args.model)
    print(f"模型     : {args.model}")
    print(f"策略网络 : {agent.describe()}")
    if metadata:
        print(f"训练信息 : seed={metadata.get('train_seed')}, "
              f"episode={metadata.get('episode')}, "
              f"env_steps={metadata.get('env_steps')}, "
              f"ε={metadata.get('epsilon')}")

    # --- 策略清单：脚本策略统一来自 experiment_config（与诊断脚本同一工厂）---
    scripted = ec.scripted_policy_specs(
        num_steps,
        random_seed=args.random_seed,
        include_fixed=True,
        include_rule=True,
        include_random=True,
        include_myopic=not args.no_oracle,
        include_lookahead=not args.no_lookahead,
    )

    # 组装评测清单：固定 / 规则 / 随机 / DQN / 逐档贪心 / 前瞻
    items: List[EvalItem] = []
    dqn_inserted = False
    for spec in scripted:
        if not dqn_inserted and spec.label.startswith("逐档贪心"):
            items.append(EvalItem("DQN(贪心)", "step_dqn.csv", is_dqn=True))
            dqn_inserted = True
        items.append(EvalItem(spec.label, spec.csv_name, is_dqn=False, spec=spec))
    if not dqn_inserted:
        items.insert(min(3, len(items)),
                     EvalItem("DQN(贪心)", "step_dqn.csv", is_dqn=True))

    # 可选：安全 RL（拉格朗日约束）分支的对照行
    safe_agent = None
    if args.safe_model:
        if not os.path.exists(args.safe_model):
            raise FileNotFoundError(f"未找到约束版模型 {args.safe_model}")
        safe_agent = LagrangianDQNAgent.load(args.safe_model, device=args.device)
        print(f"约束版模型 : {args.safe_model}")
        print(f"            {safe_agent.describe()}")
        items.append(EvalItem("DQN(约束版)", "step_dqn_safe.csv", is_dqn=True,
                              agent=safe_agent))

    runs: Dict[str, List[Any]] = {}
    summaries: List[Dict[str, Any]] = []
    csv_paths: List[str] = []

    print()
    for item in items:
        label = item.label
        per_episode: List[Dict[str, Any]] = []
        last_results: List[Any] = []

        for episode in range(max(1, args.episodes)):
            episode_seed = args.seed + episode
            if item.is_dqn:
                active_agent = item.agent if item.agent is not None else agent
                last_results = ec.run_dqn_episode(env, active_agent, episode_seed)
                policy_name = (
                    "DQN(约束版, greedy, 动作掩码)"
                    if item.agent is not None
                    else "DQN(greedy, 动作掩码)"
                )
            else:
                assert item.spec is not None
                policy = item.spec.factory()
                last_results = ec.run_scripted_episode(env, policy, episode_seed)
                policy_name = policy.describe()

            per_episode.append(ec.summarize_episode(env, last_results, label, policy_name))

        runs[label] = last_results
        summaries.append(ec.aggregate_summaries(per_episode, label, per_episode[0]["policy"]))

        csv_path = os.path.join(args.out_dir, item.csv_name)
        write_step_metrics_csv(last_results, csv_path)
        csv_paths.append(csv_path)

        s = summaries[-1]
        print(
            f"  {label:<18} 满足率={s['horizon_satisfaction_rate']:.4f}  "
            f"违反率={s['violation_rate']:.4f}  "
            f"执行步数={s['steps']:>4.0f}  平均功率={s['avg_tx_power_w']:6.2f}W  "
            f"能耗={s['cumulative_energy_j']:7.1f}J  "
            f"平均Pint={s['avg_intercept_prob']:.4f}  "
            f"平均暴露={s['avg_exposure']:.4f}  "
            f"综合收益={s['composite_reward']:+.4f}"
        )

    _verify_baselines(summaries)

    comparison_csv = os.path.join(args.out_dir, "comparison.csv")
    write_summary_csv(summaries, comparison_csv)

    report_path = os.path.join(args.out_dir, "dqn_comparison_report.html")
    write_html_report(summaries, runs, report_path, scenario_info=_scenario_info(env, args, agent))

    print("\n======== 核心指标对照 ========")
    print(format_summary_table(summaries))

    _print_dqn_vs_baseline(summaries, "规则功率控制")

    print("\n======== 输出文件 ========")
    for path in csv_paths + [comparison_csv, report_path]:
        print(f"  {path}")

    return summaries


# ----------------------------------------------------------------------

def _verify_baselines(summaries: Sequence[Dict[str, Any]]) -> None:
    """核对脚本策略是否复现 main.py 的历史数值。"""
    print("\n======== 基线回归核对（应与 main.py 一致）========")
    by_label = {s["label"]: s for s in summaries}
    checked = 0
    all_ok = True

    for label, expected in EXPECTED_BASELINES.items():
        if expected is None:
            continue
        summary = by_label.get(label)
        if summary is None:
            continue
        actual = float(summary[REGRESSION_METRIC])
        ok = abs(actual - expected) <= TOLERANCE
        all_ok = all_ok and ok
        checked += 1
        print(f"  {'OK ' if ok else '!! '}{label:<20} "
              f"{REGRESSION_METRIC} 实测={actual:.4f}  期望={expected:.4f}")

    if checked == 0:
        print("  （未设置期望值，跳过）")
    elif all_ok:
        print("  → 场景与基线完全复现，对比结果可信。")
    else:
        print("  → 警告：基线与历史值不一致！场景参数可能已被改动，"
              "本次对比结论需重新审视。")


def _print_dqn_vs_baseline(summaries: Sequence[Dict[str, Any]], baseline: str) -> None:
    by_label = {s["label"]: s for s in summaries}
    dqn = by_label.get("DQN(贪心)")
    base = by_label.get(baseline)
    if dqn is None or base is None:
        return

    print(f"\n======== DQN 相对「{baseline}」基线 ========")
    items = [
        ("探测任务满足率", "horizon_satisfaction_rate", False),
        ("平均发射功率", "avg_tx_power_w", True),
        ("累计能耗", "cumulative_energy_j", True),
        ("平均截获概率", "avg_intercept_prob", True),
        ("平均累计暴露", "avg_exposure", True),
        ("累计暴露", "cumulative_exposure", True),
        ("综合收益", "composite_reward", False),
    ]
    for title, key, lower_is_better in items:
        b, v = float(base[key]), float(dqn[key])
        delta = v - b
        rel = (delta / abs(b) * 100.0) if abs(b) > 1e-12 else float("nan")
        if abs(delta) < 1e-9:
            verdict = "持平"
        else:
            improved = (delta < 0) if lower_is_better else (delta > 0)
            verdict = "改善" if improved else "变差"
        print(f"  {title:<14} 基线={b:10.4f}   DQN={v:10.4f}   "
              f"变化={delta:+9.4f} ({rel:+7.2f}%)  {verdict}")

    b_reward, v_reward = float(base["composite_reward"]), float(dqn["composite_reward"])
    print()
    if v_reward > b_reward:
        print(f"  → DQN 综合收益 {v_reward:.4f} 高于基线 {b_reward:.4f}，"
              f"提升 {(v_reward - b_reward) / abs(b_reward) * 100:+.2f}%。")
    elif abs(v_reward - b_reward) <= 0.01:
        print(f"  → DQN 综合收益 {v_reward:.4f} 与基线 {b_reward:.4f} 基本持平。")
    else:
        print(f"  → DQN 综合收益 {v_reward:.4f} 低于基线 {b_reward:.4f}，"
              f"差距 {(v_reward - b_reward) / abs(b_reward) * 100:+.2f}%，"
              f"建议增加训练 episode 或调整超参后重训。")


def _scenario_info(env: Any, args: argparse.Namespace, agent: DQNAgent) -> Dict[str, Any]:
    sim = env.sim
    radar, scenario = sim.radar, sim.scenario
    return {
        "场景": scenario.scenario_name,
        "版本": "第二版（时序决策：能量硬约束 + 累计暴露 + 未达标惩罚）",
        "测试种子": args.seed,
        "每策略 episode 数": args.episodes,
        "仿真时长/步数": f"{scenario.sim_duration} s / {scenario.num_steps} 步",
        "功率档位": f"{scenario.num_levels} 档：{scenario.power_levels_w}",
        "固定功率基线": f"档位 {scenario.resolve_fixed_power_level()}"
        f"（{scenario.power_levels_w[scenario.resolve_fixed_power_level()]} W，"
        f"能量不足时裁剪到可行最高档）",
        "任务要求 Pd": radar.required_pd,
        "能量预算 (J)": f"{radar.energy_budget_j}（硬约束：动作执行前检查 Pt·Δt，"
        f"累计能耗不越预算）",
        "累计暴露模型": f"decay={scenario.exposure_decay}, gain={scenario.exposure_gain}"
        f"（稳态暴露量≈瞬时截获概率）",
        "奖励权重": scenario.reward_weights,
        "低截获判据": f"Pint_eff <= {scenario.lpi_pint_threshold}",
        "DQN 模型": os.path.basename(args.model),
        "DQN 网络": f"hidden={tuple(agent.config.hidden_sizes)}, "
        f"params={agent.parameter_count()}, γ={agent.config.gamma}, masked=True",
        "前瞻视野": f"{ec.lookahead_horizon(scenario.num_steps)} 步（任务步数+1，"
        f"γ={ec.LOOKAHEAD_DISCOUNT}）",
    }


def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
