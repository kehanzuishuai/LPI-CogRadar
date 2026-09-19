"""指标采集与汇总（低截获雷达智能功率调控仿真 第一版）。

取代原通信抗干扰阶段的链路数/可达率指标。输出的核心指标：

    探测任务满足率    detection_task_satisfaction_rate  满足 Pd >= required_pd 的步数占比
    平均发射功率      avg_tx_power_w
    累计能耗          cumulative_energy_j
    平均截获概率      avg_intercept_prob
    综合收益          composite_reward                  与 RL 奖励同源（models.reward）

附加指标用于解释结果：累计被截获概率、平均 Pd、平均截获 SNR、
档位切换次数、平均 J/N、能量预算占用率等。
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Sequence

# 逐步 CSV 的列（与 StepResult.to_row() 保持一致）
STEP_FIELDS = [
    "time",
    "step_index",
    "power_level",
    "tx_power_w",
    "pd_min",
    "snr_radar_db_min",
    "task_satisfied",
    "task_violated",
    "intercept_prob",
    "intercept_prob_instant",
    "intercept_snr_db",
    "exposure",
    "exposure_next",
    "jammer_active",
    "jammer_mode",
    "jam_noise_ratio",
    "step_energy_j",
    "cumulative_energy_j",
    "remaining_energy_j",
    "energy_fraction",
    "energy_exhausted",
    "reward",
    "terminal_penalty",
]

# 汇总 CSV 的列
SUMMARY_FIELDS = [
    "label",
    "policy",
    "steps",
    "horizon_steps",
    "horizon_satisfaction_rate",
    "detection_task_satisfaction_rate",
    "violation_steps",
    "violation_rate",
    "explicit_violation_steps",
    "avg_tx_power_w",
    "peak_tx_power_w",
    "cumulative_energy_j",
    "remaining_energy_j",
    "energy_budget_j",
    "energy_utilization",
    "energy_exhausted",
    "terminated_early",
    "avg_intercept_prob",
    "avg_instant_intercept_prob",
    "cumulative_intercept_prob",
    "avg_exposure",
    "final_exposure",
    "cumulative_exposure",
    "lpi_compliant_rate",
    "first_intercept_time",
    "intercept_steps",
    "avg_pd",
    "min_pd",
    "avg_intercept_snr_db",
    "composite_reward",
    "power_switch_count",
    "jammed_steps",
    "avg_jam_noise_ratio",
    "jammer_modes",
]


def extract_step_metrics(step: Any) -> Dict[str, object]:
    """把一个 StepResult 展平成一行逐步指标。"""
    return step.to_row()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_run(
    results: List[Any],
    label: str,
    policy: str = "",
    energy_budget_j: float = 0.0,
    lpi_pint_threshold: float = 0.5,
    horizon_steps: int | None = None,
) -> Dict[str, Any]:
    """把一段 episode 的逐步结果汇总为核心指标。

    参数
    ----
    energy_budget_j   : 从场景配置传入（指标层不持有场景对象）
    lpi_pint_threshold: 低截获达标判据
    horizon_steps     : **完整任务步数**。能量成为硬约束后 episode 可能提前终止，
        此时若只用「实际执行步数」作分母，会出现
        「早早烧光能量 -> 分母变小 -> 满足率虚高」的指标漏洞。
        传入 horizon_steps 后额外给出 `horizon_satisfaction_rate`
        （满足步数 / 完整任务步数），未执行的任务步按**未满足**计入。
        这是第二版的主指标。
    """
    if not results:
        raise ValueError(f"[{label}] 结果为空，无法汇总")

    steps = len(results)
    tx_powers = [r.tx_power_w for r in results]
    pds = [r.pd_min for r in results]
    pints = [r.intercept_prob for r in results]
    pints_instant = [r.intercept_prob_instant for r in results]
    exposures = [r.exposure for r in results]
    intercept_snrs = [
        r.intercept_snr_db for r in results if r.intercept_snr_db != float("-inf")
    ]

    satisfied_steps = sum(1 for r in results if r.task_satisfied)
    violation_steps = sum(1 for r in results if r.task_violated)

    horizon = int(horizon_steps) if horizon_steps else steps

    # 累计被截获概率：整段 episode 中「至少被截获一次」的概率。
    # 注意：在长驻留（61 步）场景下各策略都会趋近 1.0，区分度低，
    # 真正有区分力的是 avg_intercept_prob / lpi_compliant_rate /
    # first_intercept_time / avg_exposure。
    survive = 1.0
    for pint in pints:
        survive *= max(0.0, 1.0 - pint)
    cumulative_intercept_prob = 1.0 - survive

    # 低截获达标：Pint_eff 不超过阈值视为处于「低截获状态」
    compliant_steps = sum(1 for pint in pints if pint <= lpi_pint_threshold)
    intercept_steps = steps - compliant_steps

    # 首次被截获时间：第一个 Pint_eff 超过阈值的时刻；全程未超过则为 -1
    first_intercept_time = -1.0
    for r in results:
        if r.intercept_prob > lpi_pint_threshold:
            first_intercept_time = r.time
            break

    switch_count = sum(
        1
        for prev, cur in zip(results, results[1:])
        if prev.power_level != cur.power_level
    )

    cumulative_energy = results[-1].cumulative_energy_j
    energy_exhausted = bool(getattr(results[-1], "energy_exhausted", False)) or (
        steps < horizon
    )

    return {
        "label": label,
        "policy": policy,
        "steps": steps,
        "horizon_steps": horizon,
        # 主指标：以完整任务步数为分母，未执行的步按未满足计
        "horizon_satisfaction_rate": satisfied_steps / horizon if horizon else 0.0,
        # 辅助：只统计实际执行的步（提前终止时会被抬高，勿单独使用）
        "detection_task_satisfaction_rate": satisfied_steps / steps,
        # 约束违反率（安全 RL 的 cost 监控口径）：
        #   = 未满足的任务步 / 完整任务步数 = 1 − horizon_satisfaction_rate
        # **必须**用完整任务步数当分母：能量耗尽导致没执行到的步同样算"没完成探测任务"。
        # 若只用「显式 Pd 未达标的步数 / 完整步数」，会出现
        # 「早早烧光能量 -> 一步都没执行 -> 违反率为 0」的指标漏洞
        # （固定 80 W 基线就是这样拿到"最低违反率"的假象）。
        "violation_rate": (horizon - satisfied_steps) / horizon if horizon else 0.0,
        # 显式未达标步数（Pd < required_pd 且确实执行了）——与上面的口径区分开
        "explicit_violation_steps": violation_steps,
        "avg_tx_power_w": _mean(tx_powers),
        "peak_tx_power_w": max(tx_powers),
        "cumulative_energy_j": cumulative_energy,
        "remaining_energy_j": max(0.0, energy_budget_j - cumulative_energy),
        "energy_budget_j": energy_budget_j,
        "energy_utilization": (
            cumulative_energy / energy_budget_j if energy_budget_j > 0 else 0.0
        ),
        "energy_exhausted": int(energy_exhausted),
        "terminated_early": int(steps < horizon),
        "avg_intercept_prob": _mean(pints),
        "avg_instant_intercept_prob": _mean(pints_instant),
        "cumulative_intercept_prob": cumulative_intercept_prob,
        "avg_exposure": _mean(exposures),
        "final_exposure": results[-1].exposure_next,
        "cumulative_exposure": sum(pints_instant),  # Σ Pint_inst，期望截获次数
        "lpi_compliant_rate": compliant_steps / steps,
        "first_intercept_time": first_intercept_time,
        "intercept_steps": intercept_steps,
        "violation_steps": violation_steps,
        "avg_pd": _mean(pds),
        "min_pd": min(pds),
        "avg_intercept_snr_db": _mean(intercept_snrs),
        "composite_reward": _mean([r.reward for r in results]),
        "power_switch_count": switch_count,
        "jammed_steps": sum(1 for r in results if r.jammer_active),
        "avg_jam_noise_ratio": _mean([r.jam_noise_ratio for r in results]),
        # 自适应干扰机模式轨迹（固定干扰模式下为空）
        "jammer_modes": "|".join(
            sorted({r.jammer_mode for r in results if getattr(r, "jammer_mode", "")})
        ),
    }


def write_step_metrics_csv(results: List[Any], csv_path: str) -> None:
    """写逐步指标 CSV（保留原工程 output/*.csv 的输出形态）。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for step in results:
            writer.writerow(extract_step_metrics(step))


def write_summary_csv(summaries: Sequence[Dict[str, Any]], csv_path: str) -> None:
    """写多策略对照汇总 CSV。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary.get(k, "") for k in SUMMARY_FIELDS})


# ----------------------------------------------------------------------
# 控制台表格
# ----------------------------------------------------------------------

_TABLE_COLUMNS = [
    ("label", "策略", 18),
    ("horizon_satisfaction_rate", "满足率*", 9),
    ("avg_tx_power_w", "平均功率W", 10),
    ("cumulative_energy_j", "累计能耗J", 11),
    ("avg_intercept_prob", "平均Pint", 9),
    ("avg_exposure", "平均暴露", 9),
    ("composite_reward", "综合收益", 9),
]


def format_summary_table(summaries: Sequence[Dict[str, Any]]) -> str:
    """把汇总结果排版成等宽文本表格，便于直接打印到控制台。

    满足率* = horizon_satisfaction_rate：以**完整任务步数**为分母，
    episode 因能量耗尽提前终止时，未执行的步按未满足计入。
    """
    header = "  ".join(f"{title:^{width}}" for _, title, width in _TABLE_COLUMNS)
    lines = [header, "-" * len(header)]

    for summary in summaries:
        cells = []
        for key, _, width in _TABLE_COLUMNS:
            value = summary.get(key, "")
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(f"{text:^{width}}")
        lines.append("  ".join(cells))

    return "\n".join(lines)
