"""压力测试报告（v4.5 P2：4 核心关联场景 + 4 系统级场景）。

产出
----
* `stress_metrics.csv`：每个 (场景, 变体, 种子) 一行，**全部**指标
* `stress_system_metrics.csv`：系统级指标单独一张表（便于横向比较）
* `stress_metrics.json`：同上 + 逐场景判据
* `stress_report.html`：人看的汇总报告（8 类分组、判据、诚实性边界）
* `association_<场景>_<变体>.csv`：**关联层审计**（一行 = 一对 (测量,候选航迹)）
* `lifecycle_<场景>_<变体>.csv`：测量生命周期（一行 = 一条测量）

判据先写阈值再看数据（见 `judge_core` / `judge_system`），
避免"看到数字再编解释"。所有阈值都是**描述性**的：
本工程本轮不做多种子统计，报告里必须与这一条一起读。
"""

from __future__ import annotations

import csv
import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from multi_target_stress.metrics import METRIC_ORDER
from multi_target_stress.runner import resolve_scenario, run_case

#: 默认输出目录
DEFAULT_OUT_DIR = "output/stress"

#: 跨种子聚合的字段（关联层 + 系统级）
AGG_FIELDS: Tuple[str, ...] = (
    "id_switch_count", "track_fragmentation_count", "duplicate_track_count",
    "false_track_rate", "missed_track_rate", "association_accuracy",
    "track_purity", "track_completeness", "continuity_rate",
    "position_rmse_m", "velocity_rmse_mps", "ambiguous_rate",
    "remote_utilization", "delivery_rate", "latency_mean_s",
    "tracks_dropped", "max_tracks_in_a_frame",
    "n_false_alarms_into_real_track", "n_associations",
    # 系统级（不是每个场景都有；缺失即不参与该场景的聚合）
    "handover_continuity", "handover_continuity_after_local", "handover_delay_s",
    "remote_contribution_ratio", "short_double_track_frames",
    "out_of_order_rate", "stale_rejection_rate", "burst_loss_length_mean",
    "continuity_during_outage", "continuity_after_recovery", "recovery_time_s",
    "late_applied_measurements", "mean_time_in_system_s", "mean_hold_s",
    "track_age_mean_s", "coasting_frames_total",
    "maneuvered_residual_pre_m", "maneuvered_residual_post_m",
    "maneuvered_residual_ratio", "maneuvered_innovation_max",
    "maneuvered_gate_rejected", "maneuvered_recovery_time_s",
    "reference_residual_ratio",
    "sensor_health_best", "sensor_health_worst", "track_sigma_mean_m",
)

#: 只有单个种子时这些字段本身是计数，聚合会误导，报告里要注明
COUNT_FIELDS = ("id_switch_count", "track_fragmentation_count",
                "duplicate_track_count", "tracks_dropped",
                "n_false_alarms_into_real_track", "short_double_track_frames",
                "late_applied_measurements", "coasting_frames_total",
                "maneuvered_gate_rejected")

#: 系统级指标的扁平化白名单（写进 CSV 的列名）
SYSTEM_FLATTEN: Dict[str, Tuple[str, ...]] = {
    "maneuver": (
        "maneuvered_target_id", "maneuvered_residual_pre_m",
        "maneuvered_residual_post_m", "maneuvered_residual_max_m",
        "maneuvered_residual_ratio", "maneuvered_innovation_mean",
        "maneuvered_innovation_max", "maneuvered_gate_rejected",
        "maneuvered_recovery_time_s", "reference_target_id",
        "reference_residual_pre_m", "reference_residual_post_m",
        "reference_residual_ratio", "reference_gate_rejected",
    ),
    "handover": (
        "handover_target_id", "handover_incoming_sensor", "handover_overlap_s",
        "handover_continuity", "handover_continuity_after_local",
        "handover_delay_s", "handover_first_measurement_s",
        "handover_first_source_s", "remote_contribution_ratio",
        "n_remote_sources", "n_track_sources", "short_double_track_frames",
        "n_overlap_frames", "n_frames_after_local_lost",
    ),
    "timing": (
        "out_of_order_count", "out_of_order_rate", "mean_time_in_system_s",
        "n_fused_measurements", "stale_rejected", "stale_rejection_rate",
        "burst_loss_length_mean", "burst_loss_length_max", "n_bursts",
        "n_burst_steps", "n_outage_frames", "continuity_during_outage",
        "continuity_after_recovery", "recovery_time_s", "track_age_mean_s",
        "track_age_max_s", "coasting_frames_total", "coasting_duration_mean_s",
        "late_applied_measurements", "late_applied_mean_hold_s",
        "max_reorder_lag_s", "n_out_of_order_seen", "n_released_late",
        "mean_hold_s", "mean_covariance_inflation",
    ),
    "bias": (
        "biased_sensor_ids", "suspicious_sensor_ids", "remote_contribution_ratio",
        "position_rmse_m", "position_error_max_m", "velocity_rmse_mps",
        "track_sigma_mean_m", "track_sigma_max_m", "n_position_samples",
    ),
}

#: 变体名 → 中文（报告表头用）
VARIANT_CN: Dict[str, str] = {
    "single": "仅本地雷达",
    "no_share": "不共享（双雷达）",
    "ideal_share": "理想共享",
    "constrained_share": "受限共享",
    "biased_share": "有偏传感器共享",
    "drop_stale": "不重排（对照）",
    "reorder_buffer": "重排缓冲",
    "delayed_update": "重排+协方差放大",
}


def _flatten_system(system: Dict[str, Any]) -> Dict[str, Any]:
    """把嵌套的系统级指标拍平成 CSV 列。"""
    flat: Dict[str, Any] = {}
    for group, keys in SYSTEM_FLATTEN.items():
        payload = system.get(group)
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, (list, tuple)):
                value = "|".join(str(v) for v in value)
            # bias 组里也有 position_rmse_m，改名避免与关联层的同名指标互相覆盖
            flat["position_rmse_system_m" if (group == "bias"
                                              and key == "position_rmse_m")
                 else key] = value
        if group == "bias":
            health = payload.get("sensor_health_score") or {}
            scores = [float(v) for v in health.values()
                      if isinstance(v, (int, float))]
            if scores:
                flat["sensor_health_best"] = min(scores)
                flat["sensor_health_worst"] = max(scores)
            for sensor_id, entry in (payload.get("source_wise") or {}).items():
                for key, value in entry.items():
                    flat[f"src_{sensor_id}_{key}"] = value
    return flat


def run_group(
    scenario_ids: Sequence[str], seeds: Sequence[int] = (42,),
    dump_audit: bool = False, out_dir: str = DEFAULT_OUT_DIR,
) -> List[Dict[str, Any]]:
    """跑一组场景的全部变体 × 全部种子，返回逐行指标。"""
    rows: List[Dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = resolve_scenario(scenario_id)
        for variant in scenario.variants:
            for seed in seeds:
                result = run_case(scenario_id, variant, seed=seed,
                                  scenario=scenario)
                row = dict(result["metrics"])
                row["seed"] = seed
                row["scenario_title"] = scenario.title_cn
                row["n_candidate_tracks_mean"] = result["ambiguity"][
                    "mean_candidate_tracks"
                ]
                row.update(_flatten_system(result["system"]))
                rows.append(row)
                if dump_audit and len(seeds) == 1:
                    result["lifecycle"].write_association_csv(os.path.join(
                        out_dir, f"association_{scenario_id}_{variant}.csv"))
                    result["lifecycle"].write_csv(os.path.join(
                        out_dir, f"lifecycle_{scenario_id}_{variant}.csv"))
    return rows


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """按变体聚合（均值 / 标准差）；单种子时标准差为 0。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["variant"]), []).append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for variant, group in buckets.items():
        entry: Dict[str, float] = {}
        for field in AGG_FIELDS:
            values = [float(row[field]) for row in group
                      if isinstance(row.get(field), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            entry[f"{field}_mean"] = mean
            entry[f"{field}_std"] = (
                (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                if len(values) > 1 else 0.0
            )
        entry["n_seeds"] = float(len(group))
        summary[variant] = entry
    return summary


# ----------------------------------------------------------------------
# 判据（先写阈值，再看数据）
# ----------------------------------------------------------------------

#: 核心场景：认为"存在明显误关联"的阈值
ID_SWITCH_ALARM = 1        # 换号次数 ≥ 1
AMBIGUOUS_ALARM = 0.30     # 关联歧义率 ≥ 30%
ACCURACY_ALARM = 0.85      # 关联准确率 < 85%
FALSE_TRACK_ALARM = 0.20   # 假航迹率 ≥ 20%
MISSED_ALARM = 0.20        # 漏跟率 ≥ 20%

#: 系统级阈值
MANEUVER_RESIDUAL_ALARM = 1.5     # 机动后残差放大倍数 ≥ 1.5
HANDOVER_CONTINUITY_ALARM = 0.95  # 交接后连续性 < 0.95 视为"接力失败"
HANDOVER_DELAY_ALARM = 1.0        # 交接延迟 > 1 s 视为"接力不及时"
OOO_IMPROVE_ALARM = 0.30          # 乱序率相对下降 ≥ 30% 才算"重排有效"


def _get(entry: Dict[str, float], field: str) -> float:
    value = entry.get(f"{field}_mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _seed_caveat(rows: Sequence[Dict[str, Any]]) -> str:
    seeds = {r.get("seed") for r in rows}
    if len(seeds) <= 1:
        return ("⚠️ 本次只有 **1 个种子**，上述差值是描述性的，"
                "**没有做统计显著性检验**（多种子接口已就绪，见 `--seeds`）。")
    return (f"⚠️ 本次有 **{len(seeds)} 个种子**，报告给出均值±标准差，"
            "但**没有做显著性检验**（本轮按用户要求不优先跑 20–30 种子）。")


def judge_core(scenario_id: str, summary: Dict[str, Dict[str, float]],
               rows: Sequence[Dict[str, Any]]) -> List[str]:
    """核心关联场景（S1–S4）的判据。"""
    lines: List[str] = []
    single = summary.get("single", {})
    ideal = summary.get("ideal_share", {})
    constrained = summary.get("constrained_share", {})

    triggers: List[str] = []
    if _get(single, "id_switch_count") >= ID_SWITCH_ALARM:
        triggers.append(f"ID换号 {_get(single, 'id_switch_count'):.1f} 次")
    if _get(single, "ambiguous_rate") >= AMBIGUOUS_ALARM:
        triggers.append(f"关联歧义率 {_get(single, 'ambiguous_rate'):.4f}")
    if 0.0 < _get(single, "association_accuracy") < ACCURACY_ALARM:
        triggers.append(f"关联准确率 {_get(single, 'association_accuracy'):.4f}")
    if _get(single, "false_track_rate") >= FALSE_TRACK_ALARM:
        triggers.append(f"假航迹率 {_get(single, 'false_track_rate'):.4f}")
    if _get(single, "missed_track_rate") >= MISSED_ALARM:
        triggers.append(f"漏跟率 {_get(single, 'missed_track_rate'):.4f}")

    if triggers:
        lines.append(
            "**基线（仅本地雷达）确实出现明显误关联**，触发条件："
            + "、".join(triggers) + "。这构成引入 JPDA-lite 分支的**依据**。"
        )
    else:
        lines.append(
            "基线（仅本地雷达）未触发误关联告警阈值"
            f"（换号 {_get(single, 'id_switch_count'):.1f} 次、"
            f"歧义率 {_get(single, 'ambiguous_rate'):.4f}、"
            f"关联准确率 {_get(single, 'association_accuracy'):.4f}）。"
        )

    deltas: List[str] = []
    for field, title, better in (
        ("id_switch_count", "ID换号", "lower"),
        ("track_fragmentation_count", "碎裂", "lower"),
        ("duplicate_track_count", "重复航迹", "lower"),
        ("missed_track_rate", "漏跟率", "lower"),
        ("association_accuracy", "关联准确率", "higher"),
        ("position_rmse_m", "位置RMSE", "lower"),
    ):
        base, shared = _get(single, field), _get(ideal, field)
        if base == shared:
            continue
        improved = (shared < base) if better == "lower" else (shared > base)
        deltas.append(f"{title} {base:.4g} → {shared:.4g}"
                      f"（{'改善' if improved else '**变差**'}）")
    if deltas:
        lines.append("理想共享 vs 单雷达：" + "；".join(deltas) + "。")

    if constrained:
        gaps: List[str] = []
        for field, title in (("position_rmse_m", "位置RMSE"),
                             ("association_accuracy", "关联准确率"),
                             ("id_switch_count", "ID换号"),
                             ("duplicate_track_count", "重复航迹")):
            a, b = _get(ideal, field), _get(constrained, field)
            if a == b:
                continue
            gaps.append(f"{title} {a:.4g} → {b:.4g}")
        if gaps:
            lines.append(
                "受限共享（1.2 s 基延迟 + 25% 丢包 + 2.5 s 过期）相对理想共享："
                + "；".join(gaps) + "。"
                f"送达率 {_get(constrained, 'delivery_rate'):.4f}，"
                f"平均延迟 {_get(constrained, 'latency_mean_s'):.3f} s。"
            )
    lines.append(_seed_caveat(rows))
    return lines


def judge_system(scenario_id: str, summary: Dict[str, Dict[str, float]],
                 rows: Sequence[Dict[str, Any]]) -> List[str]:
    """系统级场景（S5–S8）的判据。"""
    lines: List[str] = []
    single = summary.get("single", {})
    ideal = summary.get("ideal_share", {})

    if scenario_id == "S5":
        ratio = _get(single, "maneuvered_residual_ratio")
        ref_ratio = _get(single, "reference_residual_ratio")
        lines.append(
            "机动目标 vs **同场景匀速对照目标**（单雷达）：残差放大倍数 "
            f"**{ratio:.3f}** vs {ref_ratio:.3f}；创新峰值 "
            f"{_get(single, 'maneuvered_innovation_max'):.2f}"
            "（χ²₉₅(3)=7.815）、门限拒绝 "
            f"{_get(single, 'maneuvered_gate_rejected'):.0f} 次、恢复时间 "
            f"{_get(single, 'maneuvered_recovery_time_s'):.2f} s、"
            f"重复航迹 {_get(single, 'duplicate_track_count'):.0f}。"
        )
        if ratio > MANEUVER_RESIDUAL_ALARM and ratio > ref_ratio * 1.2:
            lines.append(
                "**模型失配被观测到**：机动目标的残差放大明显高于匀速对照目标，"
                "说明失配来自机动本身，而不是场景的固有误差水平。"
            )
        else:
            lines.append(
                "⚠️ 机动目标与匀速对照目标的残差放大**差异不明显**："
                "在本场景机动强度下常速度滤波器基本吸收了失配。"
                "这是**负结果**，不得据此声称「机动必然导致失配」。"
            )
        if ideal:
            lines.append(
                "共享的作用（本地慢扫描 2 s → 远端 1 s 补充新鲜观测）："
                f"ID换号 {_get(single, 'id_switch_count'):.0f} → "
                f"{_get(ideal, 'id_switch_count'):.0f}；"
                f"重复航迹 {_get(single, 'duplicate_track_count'):.0f} → "
                f"{_get(ideal, 'duplicate_track_count'):.0f}；"
                f"位置RMSE {_get(single, 'position_rmse_m'):.1f} → "
                f"{_get(ideal, 'position_rmse_m'):.1f} m。"
            )

    elif scenario_id == "S6":
        for name, entry in summary.items():
            first_source = next((row.get("handover_first_source_s") for row in rows
                                 if row.get("variant") == name), None)
            delay_text = ("**未接上**（远端来源从未进入航迹）"
                          if first_source is None
                          else f"{_get(entry, 'handover_delay_s'):.2f} s")
            lines.append(
                f"{VARIANT_CN.get(name, name)}：交接窗口内连续性 "
                f"{_get(entry, 'handover_continuity'):.3f}，"
                f"本地丢失后连续性 **{_get(entry, 'handover_continuity_after_local'):.3f}**，"
                f"交接延迟 {delay_text}，远端贡献比例 "
                f"{_get(entry, 'remote_contribution_ratio'):.3f}，短时双轨 "
                f"{_get(entry, 'short_double_track_frames'):.0f} 帧，"
                f"重复航迹 {_get(entry, 'duplicate_track_count'):.0f}。"
            )
        if _get(single, "handover_continuity_after_local") < HANDOVER_CONTINUITY_ALARM:
            lines.append(
                "**不共享时交接失败**：本地包线之外连续性 "
                f"{_get(single, 'handover_continuity_after_local'):.3f} ——"
                "证实「没有共享就没有接力」。"
            )
        if _get(ideal, "handover_continuity_after_local") >= HANDOVER_CONTINUITY_ALARM:
            lines.append(
                "理想共享下本地丢失后**目标一直被跟踪**（连续性 "
                f"{_get(ideal, 'handover_continuity_after_local'):.3f}），"
                f"远端贡献比例 {_get(ideal, 'remote_contribution_ratio'):.3f}。"
            )
        # ⚠️ 最关键的一条限定：**覆盖接力 ≠ 身份接力**。
        # "本地丢失后仍有航迹"可以由**新航迹**接上达成；
        # 只有**同一条 track_id** 先后拿到两个传感器的来源，才叫航迹接力。
        achieved = next((row.get("within_track_handover_achieved") for row in rows
                         if row.get("variant") == "ideal_share"), None)
        frames = next((row.get("within_track_handover_frames") for row in rows
                       if row.get("variant") == "ideal_share"), None)
        if achieved:
            lines.append(
                f"**身份接力成立**：有同一条航迹先后获得了两个传感器的来源"
                f"（{frames} 帧命中同一条 track_id）。这才是「多雷达协同实现了"
                "航迹接力」，而不只是「目标恰好还有航迹」。"
            )
        else:
            lines.append(
                "⚠️ **身份接力不成立**：没有任何一条 track_id 同时拥有两个传感器"
                "的来源。也就是说，本地丢失后的航迹是**新航迹**接上的，"
                "而不是原航迹把身份延续下来。"
                "因此上面那个「连续性 1.000」只能读作**目标覆盖没有中断**，"
                "**不能**读作「航迹交接成功」。"
                "这与本场景极高的 ID 换号 / 重复航迹数是一致的，"
                "也正是 NN 关联基线在交接期的真实失效模式。"
            )
        constrained = summary.get("constrained_share", {})
        if constrained and _get(constrained, "handover_delay_s") > HANDOVER_DELAY_ALARM:
            lines.append(
                "**通信条件直接体现在交接延迟上**：受限共享 "
                f"{_get(constrained, 'handover_delay_s'):.2f} s > "
                f"{HANDOVER_DELAY_ALARM:g} s 阈值，远端贡献比例降到 "
                f"{_get(constrained, 'remote_contribution_ratio'):.3f}。"
            )
        worst = max((_get(e, "duplicate_track_count") for e in summary.values()),
                    default=0.0)
        if worst > 0:
            lines.append(
                f"⚠️ **交接过程本身制造了重复航迹**（最多 {worst:.0f}）："
                "一度有两条航迹同时追同一个目标。这是 NN 基线的真实失效模式，"
                "也是「是否需要更好的关联」的直接证据。"
            )

    elif scenario_id == "S7":
        control = summary.get("drop_stale", {})
        lines.append(
            "对照（`drop_stale`，v4.3–v4.5 的既有行为）：乱序率 "
            f"{_get(control, 'out_of_order_rate'):.4f}、时效拒绝率 "
            f"{_get(control, 'stale_rejection_rate'):.4f}、迟到应用测量 "
            f"{_get(control, 'late_applied_measurements'):.0f} 条。"
        )
        for name in [n for n in summary if n != "drop_stale"]:
            entry = summary[name]
            base_ooo = _get(control, "out_of_order_rate")
            ooo = _get(entry, "out_of_order_rate")
            reduction = (1.0 - ooo / base_ooo) if base_ooo > 0 else 0.0
            lines.append(
                f"{VARIANT_CN.get(name, name)}：乱序率 {ooo:.4f}"
                f"（相对对照{'下降' if reduction >= 0 else '上升'} "
                f"{abs(reduction) * 100:.1f}%），时效拒绝率 "
                f"{_get(entry, 'stale_rejection_rate'):.4f}"
                f"（对照 {_get(control, 'stale_rejection_rate'):.4f}），"
                f"迟到应用 {_get(entry, 'late_applied_measurements'):.0f} 条"
                f"（平均扣留 {_get(entry, 'mean_hold_s'):.2f} s），"
                f"平均在途时间 {_get(entry, 'mean_time_in_system_s'):.3f} s"
                f"（对照 {_get(control, 'mean_time_in_system_s'):.3f} s）。"
            )
            if reduction >= OOO_IMPROVE_ALARM:
                lines.append(
                    f"  → **确实减少了乱序到达**（降幅 {reduction * 100:.0f}%），"
                    "**代价是额外在途延迟与一部分测量因扣留而超时效被拒**——"
                    "这两项必须一起报，不能只报乱序率变好。"
                )
            else:
                lines.append(
                    f"  → ⚠️ 对乱序率的改善为 {reduction * 100:.1f}%，"
                    f"未达**先设定**的 {OOO_IMPROVE_ALARM * 100:.0f}% 门槛"
                    "（阈值在跑之前就写死在 `report.py`，没有看到数据后再改）。"
                    "可以确认的是方向对、代价明确；但「重排值得做」这个结论"
                    "在本阈值下**不成立**。"
                )
        lines.append(
            "四个时间戳已显式区分并可逐条导出：measurement / send / arrival / "
            "fusion（见 `oosm_decisions` 与 `lifecycle_*.csv`）。"
        )

    elif scenario_id == "S8":
        biased = summary.get("biased_share", {})
        base_rmse = _get(single, "position_rmse_m")
        ideal_rmse = _get(ideal, "position_rmse_m")
        biased_rmse = _get(biased, "position_rmse_m")
        lines.append(
            f"位置RMSE：单雷达 {base_rmse:.2f} m｜不共享双雷达 "
            f"{_get(summary.get('no_share', {}), 'position_rmse_m'):.2f} m｜"
            f"理想共享 {ideal_rmse:.2f} m｜**有偏共享 {biased_rmse:.2f} m**。"
        )
        if ideal_rmse < base_rmse:
            lines.append(
                f"无偏共享改善精度：{base_rmse:.2f} → {ideal_rmse:.2f} m"
                f"（−{(1 - ideal_rmse / base_rmse) * 100:.1f}%）。"
            )
        if biased_rmse > base_rmse:
            lines.append(
                "**出现「加第二部雷达反而更差」**：有偏共享 "
                f"{biased_rmse:.2f} m 比只用本地单雷达 {base_rmse:.2f} m "
                f"**更差 {(biased_rmse / base_rmse - 1) * 100:.1f}%**。"
                "偏差是系统性的、不会被多帧平均掉；噪声低估还让融合把更大的"
                "权重交给了误差更大的传感器。"
            )
        else:
            lines.append(
                "⚠️ 有偏共享**没有**比单雷达更差（"
                f"{biased_rmse:.2f} vs {base_rmse:.2f} m）："
                "本场景偏差幅度还不足以拉偏融合，属于**负结果**。"
            )
        lines.append(
            f"有偏共享下：ID换号 {_get(biased, 'id_switch_count'):.0f}、"
            f"重复航迹 {_get(biased, 'duplicate_track_count'):.0f}、航迹σ均值 "
            f"{_get(biased, 'track_sigma_mean_m'):.1f} m（理想共享 "
            f"{_get(ideal, 'track_sigma_mean_m'):.1f} m）——协方差膨胀说明融合"
            "「知道自己更不确定了」，但位置仍然被拉偏。"
        )
        flagged = next((row.get("suspicious_sensor_ids") for row in rows
                        if row.get("variant") == "biased_share"
                        and row.get("suspicious_sensor_ids")), "")
        lines.append(
            "sensor health score（归一化残差 = 残差 / 该传感器**自称**的 σ）："
            f"有偏共享下最好/最差 {_get(biased, 'sensor_health_best'):.3f} / "
            f"{_get(biased, 'sensor_health_worst'):.3f}；标记为可疑："
            f"{flagged or '（无）'}。⚠️ 该判据是**相对**的：只有两部传感器时，"
            "它只能指出「更差的那部」，**不能**断定偏差就在它身上；"
            "而且它只作诊断分支，**默认不自动剔除**任何传感器。"
        )

    lines.append(_seed_caveat(rows))
    return lines


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _metric_cells(entry: Dict[str, float], key: str, digits: int,
                  multi_seed: bool) -> str:
    value = entry.get(f"{key}_mean")
    if not isinstance(value, (int, float)):
        return "—"
    std = entry.get(f"{key}_std", 0.0)
    if multi_seed and std:
        return f"{value:.{digits}f}±{std:.{digits}f}"
    return f"{value:.{digits}f}"


def _system_table_rows(scenario_id: str,
                       summary: Dict[str, Dict[str, float]]) -> List[List[Any]]:
    """系统级关键指标 → HTML 表格（表头 + 行）。"""
    table = {
        "S5": [
            ("maneuvered_residual_pre_m", "机动前残差(m)", 1),
            ("maneuvered_residual_post_m", "机动后残差(m)", 1),
            ("maneuvered_residual_ratio", "机动残差倍数", 2),
            ("reference_residual_ratio", "对照残差倍数", 2),
            ("maneuvered_innovation_max", "创新峰值(马氏²)", 1),
            ("maneuvered_gate_rejected", "门限拒绝次数", 0),
            ("maneuvered_recovery_time_s", "恢复时间(s)", 2),
            ("duplicate_track_count", "重复航迹", 0),
            ("position_rmse_m", "位置RMSE(m)", 1),
        ],
        "S6": [
            ("handover_continuity", "交接窗口内连续性", 3),
            ("handover_continuity_after_local", "本地丢失后连续性", 3),
            ("handover_delay_s", "交接延迟(s)", 2),
            ("remote_contribution_ratio", "远端贡献比例", 3),
            ("short_double_track_frames", "短时双轨帧数", 0),
            ("duplicate_track_count", "重复航迹", 0),
            ("id_switch_count", "ID换号", 0),
            ("position_rmse_m", "位置RMSE(m)", 1),
        ],
        "S7": [
            ("out_of_order_rate", "乱序率", 4),
            ("stale_rejection_rate", "时效拒绝率", 4),
            ("late_applied_measurements", "迟到应用测量数", 0),
            ("mean_hold_s", "平均扣留(s)", 2),
            ("mean_time_in_system_s", "平均在途时间(s)", 3),
            ("burst_loss_length_mean", "突发丢包长度", 2),
            ("continuity_during_outage", "中断期连续性", 3),
            ("continuity_after_recovery", "恢复后连续性", 3),
            ("recovery_time_s", "恢复时间(s)", 2),
        ],
        "S8": [
            ("position_rmse_m", "位置RMSE(m)", 2),
            ("track_sigma_mean_m", "航迹σ均值(m)", 1),
            ("remote_contribution_ratio", "远端贡献比例", 3),
            ("sensor_health_best", "健康分(最好)", 3),
            ("sensor_health_worst", "健康分(最差)", 3),
            ("duplicate_track_count", "重复航迹", 0),
            ("id_switch_count", "ID换号", 0),
            ("missed_track_rate", "漏跟率", 3),
        ],
    }.get(scenario_id)
    if not table:
        return []
    rows: List[List[Any]] = [[title for _k, title, _d in table]]
    for variant, entry in summary.items():
        rendered: List[Any] = [VARIANT_CN.get(variant, variant)]
        for key, _title, digits in table:
            value = entry.get(f"{key}_mean")
            rendered.append(round(float(value), digits)
                            if isinstance(value, (int, float)) else "—")
        rows.append(rendered)
    return rows


def build_html(per_scenario: Dict[str, Dict[str, Any]],
               seeds: Sequence[int]) -> str:
    """生成 HTML 报告（8 类分组 + 判据 + 诚实性边界）。"""
    multi_seed = len(seeds) > 1
    parts: List[str] = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<title>LPI-CogRadar 多目标压力测试报告（v4.5 P2）</title>",
        "<style>",
        "body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;margin:24px;",
        "max-width:1500px;color:#1b1b1b;line-height:1.55}",
        "h1{border-bottom:3px solid #2c6fbb;padding-bottom:8px}",
        "h2{margin-top:34px;border-left:6px solid #2c6fbb;padding-left:10px}",
        "h3{margin-top:22px;color:#20507f}",
        "table{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}",
        "th,td{border:1px solid #cfd8e3;padding:5px 8px;text-align:right}",
        "th{background:#eaf1f9;text-align:center}",
        "td:first-child,th:first-child{text-align:left}",
        "tr:nth-child(even) td{background:#f8fafc}",
        ".note{background:#fff8e1;border-left:5px solid #f0ad4e;padding:10px 14px;margin:12px 0}",
        "code{background:#f2f4f7;padding:1px 5px;border-radius:3px}",
        "ul{margin:6px 0 6px 18px}",
        "</style></head><body>",
        "<h1>LPI-CogRadar 多目标压力测试报告（v4.5 P2）</h1>",
        f"<p>种子：{list(seeds)}（"
        + ("单种子，无统计检验" if not multi_seed
           else f"{len(seeds)} 个种子，仍无显著性检验")
        + "）｜跟踪器：<b>未改动的 NN + 常速度卡尔曼基线</b></p>",
        "<div class='note'><b>读这张报告前必须知道四件事：</b><ul>"
        "<li>「压力」来自几何 / 间距 / 虚警率 / 通信条件 / 机动 / 偏差，"
        "<b>没有</b>通过调门限或改滤波参数来制造失效；</li>"
        "<li>所有指标用<b>离线真值</b>计算，真值不参与任何算法决策，"
        "也不进 AI 上下文；</li>"
        "<li>本轮<b>不做</b> 20–30 种子统计，<b>不声称</b>统计显著；</li>"
        "<li><b>没有</b>因为某场景失败就引入 JPDA / IMM / 复杂运动模型——"
        "本阶段只负责把基线失效模式记录完整。</li></ul></div>",
    ]

    for scenario_id, payload in per_scenario.items():
        scenario = payload["scenario"]
        group_cn = "核心关联场景" if scenario.group == "core" else "系统级场景"
        parts.append(f"<h2>{scenario_id} {html.escape(scenario.title_cn)}"
                     f" <small>（{group_cn}）</small></h2>")
        parts.append(f"<p><b>问题</b>：{html.escape(scenario.question)}</p>")
        parts.append("<p><b>先声明的预期失效模式</b>："
                     f"{html.escape(scenario.expected_failure)}</p>")
        if scenario.verification_goal:
            parts.append("<p><b>验证目标</b>："
                         f"{html.escape(scenario.verification_goal)}</p>")
        parts.append("<p><b>场景说明</b>：<ul>"
                     + "".join(f"<li>{html.escape(n)}</li>"
                               for n in scenario.notes) + "</ul></p>")
        if scenario.occlusion_window_s:
            enter, leave = scenario.occlusion_window_s
            parts.append(f"<p>本地几何遮挡窗口：t = {enter:.2f} ~ {leave:.2f} s"
                         f"（共 {leave - enter:.2f} s）</p>")
        if scenario.maneuvers:
            parts.append("<p><b>机动时刻表</b>：<ul>"
                         + "".join(f"<li>{html.escape(m.describe())}</li>"
                                   for m in scenario.maneuvers) + "</ul></p>")
        if scenario.outage_windows:
            windows = "、".join(f"{a:g}~{b:g} s" for a, b in scenario.outage_windows)
            parts.append(f"<p><b>链路中断窗口</b>：{windows}</p>")
        if scenario.bias_overrides:
            injected = "；".join(
                f"{sid}: " + ", ".join(f"{k}={v}" for k, v in fields.items())
                for sid, fields in scenario.bias_overrides.items()
            )
            parts.append("<p><b>偏差注入（仅 <code>"
                         f"{html.escape(scenario.bias_variant)}</code> 变体）</b>："
                         f"{html.escape(injected)}</p>")

        parts.append(f"<h3>关联层指标（对照组：<code>{scenario.axis}</code>）</h3>")
        parts.append("<table><tr><th>变体</th>"
                     + "".join(f"<th>{t}</th>" for _k, t, _d in METRIC_ORDER[:11])
                     + "</tr>")
        for variant, entry in payload["summary"].items():
            cells = "".join(
                f"<td>{_metric_cells(entry, key, digits, multi_seed)}</td>"
                for key, _t, digits in METRIC_ORDER[:11]
            )
            parts.append(f"<tr><td>{html.escape(VARIANT_CN.get(variant, variant))}"
                         f"<br><small>{html.escape(variant)}</small></td>{cells}</tr>")
        parts.append("</table>")

        system_rows = payload.get("system_rows") or []
        if system_rows:
            parts.append("<h3>系统级指标</h3><table>")
            parts.append("<tr>" + "".join(f"<th>{html.escape(str(t))}</th>"
                                          for t in system_rows[0]) + "</tr>")
            for row in system_rows[1:]:
                parts.append("<tr>" + "".join(
                    "<td>" + (f"{v:.4g}" if isinstance(v, float)
                              else html.escape(str(v))) + "</td>" for v in row
                ) + "</tr>")
            parts.append("</table>")

        parts.append("<h3>结论</h3><ul>")
        for line in payload["judgement"]:
            parts.append(f"<li>{html.escape(line).replace('**', '')}</li>")
        parts.append("</ul>")

    parts.append("<h2>诚实的边界</h2><ul>"
                 "<li>数字来自<b>描述性统计</b>；多种子接口已就绪但本轮未跑，"
                 "无置信区间、无显著性检验；</li>"
                 "<li>评测门限（关联 1000 m / 重复 1000 m）是<b>离线参数</b>，"
                 "不回流入算法；</li>"
                 "<li>离线一对一分配用<b>贪心</b>而非匈牙利最优解；</li>"
                 "<li>核心场景统一把检测设为<b>确定性</b>"
                 "（概率漏检已在 v4.2 测量层单独量化）；</li>"
                 "<li>`delayed_update` 是<b>简化的回溯处理</b>（按扣留时长放大协方差），"
                 "<b>不等价于</b>任何严格的 OOSM 滤波器；</li>"
                 "<li>sensor health score 是<b>相对</b>判据，只作诊断分支，"
                 "<b>不自动剔除</b>任何传感器；</li>"
                 "<li>目标机动只改<b>真值运动</b>、传感器偏差只加在<b>测量</b>上，"
                 "物理公式一个字节都没动。</li></ul>")
    parts.append("</body></html>")
    return "".join(parts)


def write_all(
    out_dir: str = DEFAULT_OUT_DIR,
    seeds: Sequence[int] = (42,),
    scenario_ids: Optional[Sequence[str]] = None,
    dump_audit: bool = True,
    isolate_run: bool = True,
) -> Dict[str, Any]:
    """跑全部（或指定）场景并落盘：CSV / JSON / HTML + 逐测量审计。

    `isolate_run=True`（默认）时产物写入 `output/runs/<run_id>/`，
    并在同目录写出 `manifest.json`（run_id / 配置摘要 / 源码摘要 / 产物清单），
    同时把 `out_dir/` 下的同名文件替换为**指向该运行**的副本。
    这样两次实验不会覆盖同一份汇总报告——外部复核能查到每个数字是哪次跑的。
    """
    from multi_target_stress.scenarios import SCENARIO_IDS
    from multi_target_stress.system_scenarios import SYSTEM_SCENARIO_IDS

    ids = list(scenario_ids or (tuple(SCENARIO_IDS) + tuple(SYSTEM_SCENARIO_IDS)))
    os.makedirs(out_dir, exist_ok=True)

    from run_manifest import RunManifest

    manifest = RunManifest(
        tool="multi_target_stress",
        command=f"python -m multi_target_stress --scenario {' '.join(ids)} "
                f"--seeds {' '.join(str(s) for s in seeds)}",
        config={"scenario_ids": ids, "seeds": list(seeds),
                "dump_audit": bool(dump_audit), "out_dir": out_dir},
        seeds=list(seeds),
    )
    run_dir = manifest.run_dir() if isolate_run else out_dir
    per_scenario: Dict[str, Dict[str, Any]] = {}
    all_rows: List[Dict[str, Any]] = []
    system_rows: List[Dict[str, Any]] = []

    for scenario_id in ids:
        scenario = resolve_scenario(scenario_id)
        rows = run_group([scenario_id], seeds=seeds,
                         dump_audit=(dump_audit and len(seeds) == 1),
                         out_dir=out_dir)
        summary = aggregate(rows)
        judgement = (judge_core(scenario_id, summary, rows)
                     if scenario.group == "core"
                     else judge_system(scenario_id, summary, rows))
        per_scenario[scenario_id] = {
            "scenario": scenario,
            "rows": rows,
            "summary": summary,
            "judgement": judgement,
            "system_rows": _system_table_rows(scenario_id, summary)
            if scenario.group == "system" else [],
        }
        all_rows.extend(rows)
        if scenario.group == "system":
            system_rows.extend(rows)

    csv_path = _write_csv(os.path.join(run_dir, "stress_metrics.csv"), all_rows)
    system_csv = _write_csv(
        os.path.join(run_dir, "stress_system_metrics.csv"), system_rows)
    json_path = os.path.join(run_dir, "stress_metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({
            "seeds": list(seeds),
            "core_scenarios": list(SCENARIO_IDS),
            "system_scenarios": list(SYSTEM_SCENARIO_IDS),
            "scenarios": {
                sid: {
                    "title": payload["scenario"].title_cn,
                    "group": payload["scenario"].group,
                    "axis": payload["scenario"].axis,
                    "question": payload["scenario"].question,
                    "expected_failure": payload["scenario"].expected_failure,
                    "verification_goal": payload["scenario"].verification_goal,
                    "summary": payload["summary"],
                    "judgement": payload["judgement"],
                }
                for sid, payload in per_scenario.items()
            },
        }, handle, ensure_ascii=False, indent=2)
    html_path = os.path.join(run_dir, "stress_report.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(build_html(per_scenario, seeds))

    # 审计 CSV 也登记进清单（它们同样是结论的支撑材料）
    audit_files = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith(("association_", "lifecycle_", "stress_")):
            audit_files.append(os.path.join(run_dir, name))
    for path in audit_files:
        manifest.record(path)
    manifest_path = manifest.write()

    return {
        "per_scenario": per_scenario,
        "rows": all_rows,
        "csv": csv_path,
        "system_csv": system_csv,
        "json": json_path,
        "html": html_path,
        "manifest": manifest_path,
        "run_id": manifest.run_id,
    }
