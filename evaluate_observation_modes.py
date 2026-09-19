"""三组观测对照实验（v4.2）。

回答的核心问题
--------------
**把"真值观测"换成"真实传感器测量"之后，现有策略的性能掉多少？**

三组构成一条**信息阶梯**（这是本实验的设计要点）：

| 组 | 观测来源 | 可见性约束 | 噪声/漏检/虚警 | 信息量 |
| --- | --- | --- | --- | --- |
| `full-truth` | 仿真真值 | 无 | 无 | 最高（不现实） |
| `ideal-measurement` | 传感器测量 | **有**（作用距离/视场/遮挡/周期） | **无** | 只看得到"看得见的东西"，但看得准 |
| `realistic-measurement` | 传感器测量 | 有（同上） | **有** | 最接近真实系统 |

阶梯的意义：`full → ideal` 的落差是**信息可得性**的代价（很多东西本来就不该知道），
`ideal → realistic` 的落差才是**测量不完美**的代价。
把两者分开报告，才能说清"性能下降到底该怪谁"。

⚠️ 三组的观测维度不同（12 / 53 / 53），因此：

* 脚本策略（规则、前瞻）在三组里都用**同一份信念桥接**做决策，
  只把"信念从哪来"换掉——这是可以跨组比较的；
* DQN 必须在**每组各自训练**（网络输入维度不同），
  因此 DQN 的跨组对比需要三份 checkpoint，脚本会分别加载并明确标注。

用法
----
    python evaluate_observation_modes.py --seeds 42 7 13 21 33
    python evaluate_observation_modes.py --full-model output/rl/dqn_agent_best.pt \\
        --ideal-model output/rl_ideal/dqn_agent_best.pt \\
        --realistic-model output/rl_realistic/dqn_agent_best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()
import experiment_config as ec
from strategy.belief_policy import BeliefPolicy
from strategy.power_policy import (
    FixedPowerPolicy,
    GreedyOraclePolicy,
    RandomPowerPolicy,
    RuleBasedPowerPolicy,
)

DEFAULT_OUT_DIR = "output/observation_modes"

#: 三组对照：(键, 显示名, 环境 observation_mode, 该组的说明)
GROUPS: Tuple[Tuple[str, str, str, str], ...] = (
    ("full", "① 全真值", "full",
     "直接给真值：距离/干扰/能量/侦察机位置/暴露全部精确已知（不现实的上界）"),
    ("ideal", "② 理想测量", "ideal",
     "走传感器测量层：作用距离/视场/遮挡/更新周期生效，但测量无噪声、无漏检、无虚警"),
    ("realistic", "③ 真实测量", "realistic",
     "同②的可见性约束，再叠加量测噪声、概率漏检与虚警（最接近真实系统）"),
)

#: 要在三组里比较的策略
POLICY_KEYS: Tuple[Tuple[str, str], ...] = (
    ("fixed", "固定功率基线(80W)"),
    ("rule", "规则功率控制"),
    ("random", "随机策略"),
    ("myopic", "逐档贪心(短视)"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="三组观测对照：full-truth / ideal-measurement / realistic-measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ec.DEFAULT_SEEDS))
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--energy-budget", type=float, default=None)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--full-model", default=os.path.join("output", "rl", "dqn_agent_best.pt"))
    parser.add_argument("--ideal-model", default=os.path.join("output", "rl_ideal", "dqn_agent_best.pt"))
    parser.add_argument("--realistic-model",
                        default=os.path.join("output", "rl_realistic", "dqn_agent_best.pt"))
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument("--no-lookahead", action="store_true", default=True)
    parser.add_argument("--with-lookahead", action="store_true",
                        help="额外跑前瞻规划（计算量大，默认关闭）")
    parser.add_argument("--measurement-stats", action="store_true",
                        help="额外输出 realistic 组的测量级统计（缺失原因分解）")
    return parser


# ----------------------------------------------------------------------


def make_group_env(
    args: argparse.Namespace, seed: int, mode: str
) -> Any:
    """按组构造环境。`ideal` / `realistic` 会自动启用传感器测量层。"""
    env = ec.make_env_for_seed(
        seed,
        config_path=args.config,
        energy_budget_j=args.energy_budget,
        jitter=not args.no_jitter,
        observation_mode=mode,
        measurement_max_tracks=args.max_tracks,
    )
    return env


def run_policy(env: Any, key: str, seed: int) -> List[Any]:
    """在三组里用**同一套决策逻辑**跑一个 episode。

    关键：脚本策略在 `ideal` / `realistic` 下必须走**信念桥接**
    （只用观测），否则它们会直接读真值仿真器，三组对比就失去意义。
    `BeliefPolicy` 已经在 `env.observation_mode != "full"` 时自动构造信念，
    因此这里只需统一用 `run_belief_episode`。
    """
    num_steps = env.sim.scenario.num_steps
    if key == "fixed":
        inner: Any = FixedPowerPolicy()
    elif key == "rule":
        inner = RuleBasedPowerPolicy()
    elif key == "random":
        inner = RandomPowerPolicy(seed=ec.RANDOM_POLICY_SEED)
    elif key == "myopic":
        inner = GreedyOraclePolicy()
    else:
        raise KeyError(key)

    if env.observation_mode == "full":
        # 全真值组：脚本策略直接读真值（与 v3.1 完全一致的口径）
        return ec.run_scripted_episode(env, inner, seed)
    return ec.run_belief_episode(env, BeliefPolicy(inner), seed)


def run_dqn(env: Any, agent: Any, seed: int) -> List[Any]:
    return ec.run_dqn_episode(env, agent, seed=seed, greedy=True)


# ----------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("======== 三组观测对照（full-truth / ideal / realistic）========")
    print(f"配置      : {args.config}")
    print(f"种子      : {len(args.seeds)} 个 {list(args.seeds)}")
    print(f"扰动      : {'关' if args.no_jitter else '开'}")
    print(f"输出目录  : {args.out_dir}\n")
    for _key, name, _mode, note in GROUPS:
        print(f"  {name}：{note}")
    print()

    rows: List[Dict[str, Any]] = []
    dqn_agents: Dict[str, Any] = {}
    if not args.no_dqn:
        try:
            from rl.dqn_agent import DQNAgent

            for key, _name, _mode, _note in GROUPS:
                path = {"full": args.full_model, "ideal": args.ideal_model,
                        "realistic": args.realistic_model}[key]
                if os.path.exists(path):
                    dqn_agents[key] = DQNAgent.load(path, device=args.device)
                    print(f"  已加载 {key} 组 DQN：{path}")
                else:
                    print(f"  [跳过 DQN] {key} 组缺少 checkpoint：{path}")
        except ImportError:
            print("  [跳过 DQN] 未安装 torch")

    for key, name, mode, _note in GROUPS:
        print(f"\n---- {name}（observation_mode={mode}）----")
        env_probe = make_group_env(args, args.seeds[0], mode)
        print(f"  观测维度 = {env_probe.observation_space.shape[0]}")

        for policy_key, policy_name in POLICY_KEYS:
            per_seed: List[Dict[str, Any]] = []
            for seed in args.seeds:
                env = make_group_env(args, seed, mode)
                results = run_policy(env, policy_key, seed)
                per_seed.append(ec.summarize_episode(env, results, name, policy_name))
            aggregate = ec.aggregate_summaries(per_seed, label=f"{name}｜{policy_name}")
            aggregate["group"] = name
            aggregate["group_key"] = key
            aggregate["policy"] = policy_name
            aggregate["obs_dim"] = int(env_probe.observation_space.shape[0])
            rows.append(aggregate)
            print(f"  {policy_name:<20s} 满足率={aggregate['horizon_satisfaction_rate']:.4f}"
                  f"±{aggregate['horizon_satisfaction_rate__std']:.4f}  "
                  f"综合收益={aggregate['composite_reward']:+.4f}"
                  f"±{aggregate['composite_reward__std']:+.4f}  "
                  f"平均功率={aggregate['avg_tx_power_w']:.2f}W")

        agent = dqn_agents.get(key)
        if agent is not None:
            per_seed = []
            mismatch = None
            for seed in args.seeds:
                env = make_group_env(args, seed, mode)
                if int(env.observation_space.shape[0]) != agent.config.obs_dim:
                    mismatch = (env.observation_space.shape[0], agent.config.obs_dim)
                    break
                results = run_dqn(env, agent, seed)
                per_seed.append(ec.summarize_episode(env, results, name, "DQN"))
            if mismatch:
                print(f"  DQN 观测维度不匹配（环境 {mismatch[0]} vs 模型 {mismatch[1]}），跳过")
            else:
                aggregate = ec.aggregate_summaries(per_seed, label=f"{name}｜DQN")
                aggregate["group"] = name
                aggregate["group_key"] = key
                aggregate["policy"] = "DQN"
                aggregate["obs_dim"] = agent.config.obs_dim
                rows.append(aggregate)
                print(f"  {'DQN':<20s} 满足率={aggregate['horizon_satisfaction_rate']:.4f}"
                      f"±{aggregate['horizon_satisfaction_rate__std']:.4f}  "
                      f"综合收益={aggregate['composite_reward']:+.4f}"
                      f"±{aggregate['composite_reward__std']:+.4f}  "
                      f"平均功率={aggregate['avg_tx_power_w']:.2f}W")

    # ---------------- 汇总表 ----------------
    print("\n======== 三组对照汇总 ========")
    header = (f"{'组':<14s}{'策略':<20s}{'观测维':>7s}"
              f"{'满足率':>18s}{'综合收益':>20s}{'平均功率W':>11s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['group']:<14s}{row['policy']:<20s}{row['obs_dim']:>7d}"
              f"{row['horizon_satisfaction_rate']:>10.4f}±{row['horizon_satisfaction_rate__std']:<7.4f}"
              f"{row['composite_reward']:>+11.4f}±{row['composite_reward__std']:<8.4f}"
              f"{row['avg_tx_power_w']:>11.2f}")

    # ---------------- 落差分解 ----------------
    print("\n======== 信息阶梯落差分解 ========")
    by_key = {(r["group_key"], r["policy"]): r for r in rows}
    notes: List[str] = []
    for _key, policy_name in [("rule", "规则功率控制"), ("myopic", "逐档贪心(短视)"),
                              ("fixed", "固定功率基线(80W)"), ("dqn", "DQN")]:
        full = by_key.get(("full", policy_name))
        ideal = by_key.get(("ideal", policy_name))
        real = by_key.get(("realistic", policy_name))
        if not (full and ideal and real):
            continue
        step1 = ideal["composite_reward"] - full["composite_reward"]
        step2 = real["composite_reward"] - ideal["composite_reward"]
        notes.append(
            f"  {policy_name}：全真值 {full['composite_reward']:+.4f}"
            f" → 理想测量 {ideal['composite_reward']:+.4f}（Δ {step1:+.4f}，信息可得性代价）"
            f" → 真实测量 {real['composite_reward']:+.4f}（Δ {step2:+.4f}，测量不完美代价）"
        )
    for line in notes:
        print(line)
    if not notes:
        print("  （缺少数值，无法分解；请确认三组都跑出了结果）")
    print("\n  读法：第一段落差来自「很多东西本来就不该知道」（视场/作用距离/遮挡/周期），")
    print("        第二段落差才是「测量有噪声、会漏检、有虚警」的代价。两者必须分开说。")

    # ---------------- DQN 跨组比较的额外警告 ----------------
    dqn_rows = [r for r in rows if r["policy"] == "DQN"]
    if len(dqn_rows) >= 2:
        print("\n  ⚠️ DQN 的跨组差值与脚本策略**不可同等解读**：")
        print("     脚本策略三组用的是同一套决策逻辑（只换了信念来源），落差可干净归因于观测；")
        print("     但 DQN 每组必须各自训练（观测维度 12 vs 53），")
        print("     跨组落差里混着「观测变化」与「两次独立训练」两个因素，")
        print("     不能直接说成「信息少了所以变差了」。")
        dqn_by_group = {r["group_key"]: r for r in dqn_rows}
        ideal_row = dqn_by_group.get("ideal")
        real_row = dqn_by_group.get("realistic")
        if ideal_row and real_row:
            diff = real_row["composite_reward"] - ideal_row["composite_reward"]
            n = max(int(ideal_row.get("n_seeds", 1) or 1), 1)
            se = (
                (ideal_row["composite_reward__std"] ** 2
                 + real_row["composite_reward__std"] ** 2) / n
            ) ** 0.5
            t_like = diff / se if se > 0 else 0.0
            print(f"     实测 ideal→realistic：{ideal_row['composite_reward']:+.4f} → "
                  f"{real_row['composite_reward']:+.4f}（Δ {diff:+.4f}）")
            print(f"     差值标准误≈{se:.4f}，t≈{t_like:.2f}"
                  + ("（**不显著**，两组不可分辨）" if abs(t_like) < 2.0 else "（显著）"))
            if abs(t_like) < 2.0:
                print("     → 只能得出：在理想测量下训练并不比在真实测量下训练更好，")
                print("       也没有更差；训练期噪声没有带来可分辨的退化。")

    # ---------------- 导出 ----------------
    csv_path = os.path.join(args.out_dir, "observation_mode_comparison.csv")
    fields = ["group", "group_key", "policy", "obs_dim",
              "horizon_satisfaction_rate", "horizon_satisfaction_rate__std",
              "composite_reward", "composite_reward__std",
              "avg_tx_power_w", "avg_tx_power_w__std",
              "avg_intercept_prob", "avg_exposure", "cumulative_energy_j",
              "violation_rate"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\n→ {csv_path}")

    # ---------------- 测量级统计（realistic 组）----------------
    if args.measurement_stats:
        print("\n======== realistic 组的测量级统计 ========")
        from sensor.reporting import (
            MeasurementLog, format_statistics, measurement_statistics,
        )
        env = make_group_env(args, args.seeds[0], "realistic")
        log = MeasurementLog()
        obs, _ = env.reset(seed=args.seeds[0])
        steps = 0
        from strategy.power_policy import RuleBasedPowerPolicy
        policy = RuleBasedPowerPolicy()
        policy.reset()
        while True:
            obs, _r, terminated, truncated, _i = env.step(policy.select_level(env.sim))
            report = env.suite_report()
            if report is not None:
                log.add(report, include_truth=True)
            steps += 1
            if terminated or truncated:
                break
        stats = measurement_statistics(
            log.measurements, log.outcome_rows, scan_times_by_sensor=log.scan_times
        )
        print(format_statistics(stats))
        stats_dir = os.path.join(args.out_dir, "measurement_stats")
        os.makedirs(stats_dir, exist_ok=True)
        log.write_measurements_csv(os.path.join(stats_dir, "measurements.csv"), True)
        log.write_outcomes_csv(os.path.join(stats_dir, "outcomes.csv"), True)
        log.write_json(os.path.join(stats_dir, "measurements.json"), True)
        print(f"\n→ {stats_dir}")


if __name__ == "__main__":
    main()
