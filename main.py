"""低截获雷达智能功率调控仿真 第一版 —— 运行入口。

    python main.py

做的事情：
  1. 打印场景与链路预算（把模型标定是否合理摆在最前面，便于核对）
  2. 跑三组实验：固定功率基线 / 规则功率控制基线 / 随机策略（动作空间演示）
  3. 打印核心指标对照表
  4. 输出 CSV（逐步 + 汇总）与 HTML 实验结果报告
  5. 演示 Gymnasium 风格 reset()/step(action) 接口（后续接 DQN 的入口）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from engine import LpiPowerEnv, Simulator
from metrics import (
    format_summary_table,
    summarize_run,
    write_html_report,
    write_step_metrics_csv,
    write_summary_csv,
)
from strategy.power_policy import (
    FixedPowerPolicy,
    RandomPowerPolicy,
    RuleBasedPowerPolicy,
)

CONFIG_PATH = "config/radar_scenario_v1.json"
OUTPUT_DIR = "output"
REPORT_PATH = os.path.join(OUTPUT_DIR, "lpi_power_report.html")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "summary.csv")


# ----------------------------------------------------------------------
# 1. 链路预算
# ----------------------------------------------------------------------

def print_budget(sim: Simulator) -> Dict[str, Any]:
    info = sim.describe()

    print("======== 场景与链路预算 ========")
    print(f"场景            : {info['scenario']}")
    print(f"仿真步数        : {info['num_steps']}（步长 {sim.scenario.time_step} s）")
    print(f"功率档位        : {info['num_levels']} 档 {info['power_levels_w']}")
    print(f"固定功率基线档位: {info['fixed_power_level']} "
          f"（{sim.power_levels_w[info['fixed_power_level']]} W）")
    print(f"雷达噪声功率    : {info['radar_noise_w']:.4e} W")
    print(f"任务要求        : Pd >= {info['required_pd']} "
          f"→ 需 SNR >= {info['required_snr_db']:.3f} dB")
    print(f"能量预算        : {info['energy_budget_j']} J"
          f"（硬约束：耗尽即 terminated）")
    print(f"累计暴露模型    : decay={info['exposure_decay']}, gain={info['exposure_gain']}")
    print(f"奖励权重        : {info['reward_weights']}")
    print(f"低截获判据      : Pint_eff <= {sim.scenario.lpi_pint_threshold} 视为低截获状态")  # type: ignore[union-attr]

    if "binding_target" in info:
        print(f"最近目标        : {info['primary_target']} "
              f"R={info['primary_range_m']:.0f} m, RCS={info['primary_rcs_m2']} m²")
        print(f"雷达 SNR 灵敏度 : {info['radar_snr_db_per_watt']:.3f} dB/W（该目标，1 W 发射时）")
        print(f"卡脖子目标      : {info['binding_target']}"
              + "".join(
                  f"\n                  {tid}: R={b['range_m']:.0f} m, RCS={b['rcs_m2']} m², "
                  f"SNR={b['snr_db_per_watt']:.2f} dB/W, 无干扰需 {b['needed_power_w']:.2f} W"
                  for tid, b in info["per_target_budget"].items()
              ))
        level = info.get("min_level_for_all_targets")
        if level is None:
            print("无干扰最低档位  : 满功率也无法同时满足全部目标（需调低 required_pd 或拉近距离）")
        else:
            print(f"无干扰最低档位  : {level}"
                  f"（{sim.power_levels_w[level]} W，需同时满足全部目标）")
    print(f"侦察接收机      : {info['interceptor']} "
          f"R={info['interceptor_range_m'] / 1000:.0f} km, "
          f"照射方式={info['interceptor_beam']}, "
          f"截获门限 SNR50={info['interceptor_snr50_db']} dB")
    return info


# ----------------------------------------------------------------------
# 2. 三组实验
# ----------------------------------------------------------------------

def run_experiments(
    sim: Simulator,
) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]], List[str]]:
    experiments = [
        ("固定功率基线", "step_fixed_power.csv", FixedPowerPolicy()),
        ("规则功率控制", "step_rule_based.csv", RuleBasedPowerPolicy()),
        ("随机策略", "step_random.csv", RandomPowerPolicy(seed=2026)),
    ]

    runs: Dict[str, List[Any]] = {}
    summaries: List[Dict[str, Any]] = []
    csv_paths: List[str] = []

    for label, csv_name, policy in experiments:
        print(f"\n======== {label} ========")
        print(f"策略: {policy.describe()}")

        results = sim.run(policy)
        runs[label] = results

        summary = summarize_run(
            results,
            label=label,
            policy=policy.describe(),
            energy_budget_j=sim.radar.energy_budget_j,  # type: ignore[union-attr]
            lpi_pint_threshold=sim.scenario.lpi_pint_threshold,  # type: ignore[union-attr]
            horizon_steps=sim.scenario.num_steps,  # type: ignore[union-attr]
        )
        summaries.append(summary)

        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        write_step_metrics_csv(results, csv_path)
        csv_paths.append(csv_path)

        for step in _sample_steps(results):
            print(
                f"  t={step.time:5.1f}s  Pt={step.tx_power_w:6.2f}W  "
                f"Pd={step.pd_min:5.3f}{'✓' if step.task_satisfied else '✗'}  "
                f"Pint={step.intercept_prob:5.3f}  "
                f"暴露={step.exposure:5.3f}  "
                f"剩余={step.remaining_energy_j:7.1f}J  "
                f"r={step.reward:+7.3f}"
            )
        print(f"  → {csv_path}")

    return runs, summaries, csv_paths


def _sample_steps(results: List[Any]) -> List[Any]:
    """抽取有代表性的若干步打印（含首步、末步与干扰期采样点）。"""
    indices = {0, len(results) // 3, len(results) // 2, 2 * len(results) // 3, len(results) - 1}
    return [results[i] for i in sorted(indices)]


# ----------------------------------------------------------------------
# 3. Gymnasium 风格接口演示
# ----------------------------------------------------------------------

def demo_env(seed: int = 7, steps: int = 6) -> None:
    print("\n======== Gymnasium 风格接口演示（reset / step）========")
    env = LpiPowerEnv(CONFIG_PATH, render_mode="ansi")
    obs, info = env.reset(seed=seed)

    print(f"action_space      = {env.action_space}")
    print(f"observation_space = {env.observation_space.shape} 维")
    print(f"obs 特征顺序      : {', '.join(env.metadata['observation_features'])}")
    print(f"初始 obs          : {[round(v, 4) for v in obs]}")

    total_reward = 0.0
    for i in range(steps):
        action = env.action_space.sample()  # 后续替换为 DQN 的策略网络
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  step {i}: action={action} -> reward={reward:+.4f}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"Pt={info['tx_power_w']:.2f}W, Pd={info['pd_min']:.3f}, "
            f"Pint={info['intercept_prob']:.3f}"
        )
        if terminated or truncated:
            print("  episode 结束")
            break

    print(f"演示 {steps} 步累计奖励 = {total_reward:+.4f}")
    print("说明：本例用随机动作，仅验证接口；DQN 只需替换上面的 action 选择即可。")
    env.close()


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim = Simulator(CONFIG_PATH)
    sim.load_config()

    budget_info = print_budget(sim)
    runs, summaries, step_csv_paths = run_experiments(sim)

    print("\n======== 核心指标对照 ========")
    print(format_summary_table(summaries))

    write_summary_csv(summaries, SUMMARY_CSV_PATH)
    write_html_report(summaries, runs, REPORT_PATH, scenario_info=_scenario_info(budget_info, sim))

    print("\n======== 输出文件 ========")
    for path in step_csv_paths + [SUMMARY_CSV_PATH, REPORT_PATH]:
        print(f"  {path}")

    demo_env()


def _scenario_info(budget: Dict[str, Any], sim: Simulator) -> Dict[str, Any]:
    """整理给 HTML 报告用的场景摘要。"""
    info = {
        "场景": budget.get("scenario"),
        "仿真时长/步数": f"{sim.scenario.sim_duration} s / {budget.get('num_steps')} 步",  # type: ignore[union-attr]
        "功率档位": f"{budget.get('num_levels')} 档：{budget.get('power_levels_w')}",
        "固定功率基线": f"档位 {budget.get('fixed_power_level')}",
        "任务要求 Pd": budget.get("required_pd"),
        "所需 SNR (dB)": round(float(budget.get("required_snr_db", 0.0)), 3),
        "雷达噪声功率 (W)": f"{float(budget.get('radar_noise_w', 0.0)):.4e}",
        "能量预算 (J)": budget.get("energy_budget_j"),
        "侦察机距离 (km)": round(float(budget.get("interceptor_range_m", 0.0)) / 1000.0, 1),
        "侦察机照射方式": budget.get("interceptor_beam"),
    }
    return info


if __name__ == "__main__":
    main()
