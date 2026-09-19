"""测量记录的导出与测量级统计验证（v4.2）。

两件事
------
1. **导出**：把测量记录、逐 (传感器,目标) 结果、套件汇总写成 CSV / JSON。
   两种口径：`include_truth=False`（默认，可直接给算法/外部工具，不含真值）
   与 `include_truth=True`（评测用，含真值与误差列）。
2. **统计验证**：回答"这套传感器是不是按配置在工作"——
   * 距离误差随距离怎么变（相对误差模型下 σ 应随 R 线性增长）；
   * 方位误差均值应接近 0、标准差应接近配置的 σ；
   * **逐原因**的漏检率（而不是笼统一个"丢测率"）；
   * 虚警率是否接近配置值；
   * 不同更新周期下的观测序列是否真的按周期出现。

为什么统计要按"原因"分开
------------------------
如果只报一个总的"没数据比例"，就无法判断传感器是被**指向**限制了、
被**距离**限制了、被**遮挡**限制了，还是**概率**上丢了一帧。
这四种的工程对策完全不同（转雷达 / 提功率 / 换位置 / 再来一帧），
所以统计必须分组。

⚠️ 统计本身**必须读真值**（否则算不出误差）。因此本模块属于**评测通道**，
其输出不得进入算法输入链路。
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sensor.record import (
    NO_DATA_REASONS,
    REASON_CN,
    MeasurementRecord,
    SuiteReport,
)

#: 测量记录 CSV 的列（`include_truth` 决定是否追加真值与误差列）
MEASUREMENT_FIELDS: Tuple[str, ...] = (
    "time_s", "sensor_id", "sensor_kind", "candidate_id",
    "is_fresh", "age_s", "is_false_alarm",
    "range_m", "azimuth_deg", "elevation_deg", "range_rate_mps",
    "std_range_m", "std_az_deg", "std_el_deg", "std_range_rate_mps",
    "confidence", "snr_db", "rcs_est_m2", "jam_ratio_est", "cov_xx", "cov_yy", "cov_zz",
)
TRUTH_MEASUREMENT_FIELDS: Tuple[str, ...] = (
    "truth_id", "truth_range_m", "truth_azimuth_deg",
    "truth_elevation_deg", "truth_range_rate_mps",
    "err_range_m", "err_azimuth_deg", "err_range_rate_mps",
)

#: 逐 (传感器,目标) 结果 CSV 的列
OUTCOME_FIELDS: Tuple[str, ...] = (
    "time_s", "sensor_id", "status", "reason", "reason_cn", "reason_dimension",
    "candidate_id", "detected", "range_m", "azimuth_deg", "elevation_deg",
    "range_rate_mps", "std_range_m", "confidence",
    "is_false_alarm", "is_fresh", "age_s",
)
TRUTH_OUTCOME_FIELDS: Tuple[str, ...] = ("truth_id", "truth_range_m")


# ----------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------


def _write_rows(
    path: str, fields: Sequence[str], rows: Sequence[Dict[str, Any]], append: bool
) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


class MeasurementLog:
    """累积多步测量记录，最后统一导出（也可逐步 append 到磁盘）。"""

    def __init__(self) -> None:
        self.measurements: List[MeasurementRecord] = []
        self.outcome_rows: List[Dict[str, Any]] = []
        self.suite_rows: List[Dict[str, Any]] = []
        #: 逐传感器的**扫描时刻**（与是否检测到目标无关）
        self.scan_times: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------

    def add(self, report: SuiteReport, include_truth: bool = False) -> None:
        """累积一次套件报告。"""
        for measurement in report.measurements:
            self.measurements.append(measurement)
        for sensor_report in report.reports:
            for outcome in sensor_report.outcomes:
                self.outcome_rows.append(outcome.to_dict(include_truth=include_truth))
            if sensor_report.updated:
                self.scan_times.setdefault(sensor_report.sensor_id, []).append(
                    float(sensor_report.time_s)
                )
        self.suite_rows.append(report.to_dict(include_truth=False))

    def clear(self) -> None:
        self.measurements.clear()
        self.outcome_rows.clear()
        self.suite_rows.clear()
        self.scan_times.clear()

    # ------------------------------------------------------------------

    def measurement_rows(self, include_truth: bool = False) -> List[Dict[str, Any]]:
        return [m.to_dict(include_truth=include_truth) for m in self.measurements]

    def write_measurements_csv(self, path: str, include_truth: bool = False) -> str:
        fields = list(MEASUREMENT_FIELDS)
        if include_truth:
            fields += list(TRUTH_MEASUREMENT_FIELDS)
        return _write_rows(path, fields, self.measurement_rows(include_truth), False)

    def write_outcomes_csv(self, path: str, include_truth: bool = False) -> str:
        fields = list(OUTCOME_FIELDS)
        if include_truth:
            fields += list(TRUTH_OUTCOME_FIELDS)
        return _write_rows(path, fields, self.outcome_rows, False)

    def write_json(self, path: str, include_truth: bool = False) -> str:
        payload = {
            "n_measurements": len(self.measurements),
            "n_outcomes": len(self.outcome_rows),
            "n_steps": len(self.suite_rows),
            "include_truth": include_truth,
            "measurements": self.measurement_rows(include_truth),
            "outcomes": self.outcome_rows,
            "suite_timeline": self.suite_rows,
            "statistics": measurement_statistics(
                self.measurements, self.outcome_rows,
                scan_times_by_sensor=self.scan_times,
            ),
        }
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def error_vs_range(
    measurements: Sequence[MeasurementRecord],
    n_bins: int = 5,
    max_range_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """按距离分箱统计误差 —— 验证"相对误差模型"是否按预期工作。

    对 `std_range = R·rel + abs` 的模型，距离误差标准差应随 R **线性增长**；
    方位误差标准差应与 R **无关**（角度误差来自波束/相位，不随距离变化）。
    这两条是很好的"传感器是否真按配置工作"的判据，因此分别统计。
    """
    samples = [
        m for m in measurements
        if m.range_m is not None and m.truth_range_m is not None
        and not m.is_false_alarm
    ]
    if not samples:
        return []

    upper = max_range_m or max(m.truth_range_m for m in samples)
    if upper <= 0:
        return []
    width = upper / n_bins
    bins: List[Dict[str, Any]] = []
    for index in range(n_bins):
        low = index * width
        high = upper if index == n_bins - 1 else (index + 1) * width
        group = [m for m in samples if low <= m.truth_range_m < high or
                 (index == n_bins - 1 and m.truth_range_m == high)]
        range_errors = [m.range_error_m() for m in group]
        az_errors = [m.azimuth_error_deg() for m in group]
        vr_errors = [m.range_rate_error_mps() for m in group]
        range_errors = [e for e in range_errors if e is not None]
        az_errors = [e for e in az_errors if e is not None]
        vr_errors = [e for e in vr_errors if e is not None]
        mean_r, std_r = _mean_std(range_errors)
        mean_a, std_a = _mean_std(az_errors)
        mean_v, std_v = _mean_std(vr_errors)
        bins.append({
            "bin": index,
            "range_low_m": low,
            "range_high_m": high,
            "n": len(group),
            "err_range_mean_m": mean_r,
            "err_range_std_m": std_r,
            # 相对误差：std/中心距离，用于检查是否近似常数
            "err_range_rel_std": (std_r / ((low + high) / 2.0)) if (low + high) > 0 else 0.0,
            "err_az_mean_deg": mean_a,
            "err_az_std_deg": std_a,
            "err_vr_mean_mps": mean_v,
            "err_vr_std_mps": std_v,
        })
    return bins


def no_data_breakdown(outcome_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """逐原因统计"没有数据"的占比（**不是**笼统一个丢测率）。"""
    by_sensor: Dict[str, Dict[str, int]] = {}
    by_reason_total: Dict[str, int] = {r: 0 for r in NO_DATA_REASONS}
    by_dimension_total: Dict[str, int] = {}
    totals = 0
    detected = 0

    for row in outcome_rows:
        sensor = str(row.get("sensor_id", ""))
        reason = str(row.get("reason", ""))
        bucket = by_sensor.setdefault(sensor, {"total": 0, "detected": 0})
        bucket["total"] += 1
        totals += 1
        if row.get("detected"):
            bucket["detected"] += 1
            detected += 1
            continue
        if reason in by_reason_total:
            by_reason_total[reason] += 1
            bucket[reason] = bucket.get(reason, 0) + 1
        dimension = str(row.get("reason_dimension", ""))
        if dimension:
            by_dimension_total[dimension] = by_dimension_total.get(dimension, 0) + 1

    per_sensor: Dict[str, Any] = {}
    for sensor, bucket in by_sensor.items():
        total = bucket["total"]
        per_sensor[sensor] = {
            "total": total,
            "detected": bucket["detected"],
            "detection_rate": (bucket["detected"] / total) if total else 0.0,
            "reasons": {
                reason: {
                    "count": bucket.get(reason, 0),
                    "rate": (bucket.get(reason, 0) / total) if total else 0.0,
                    "cn": REASON_CN.get(reason, reason),
                }
                for reason in NO_DATA_REASONS
            },
        }

    return {
        "total_outcomes": totals,
        "detected": detected,
        "overall_detection_rate": (detected / totals) if totals else 0.0,
        "by_reason": {
            reason: {
                "count": by_reason_total[reason],
                "rate": (by_reason_total[reason] / totals) if totals else 0.0,
                "cn": REASON_CN.get(reason, reason),
            }
            for reason in NO_DATA_REASONS
        },
        "by_dimension": by_dimension_total,
        "by_sensor": per_sensor,
    }


def false_alarm_statistics(measurements: Sequence[MeasurementRecord]) -> Dict[str, Any]:
    """虚警统计：按传感器给出虚警数与占比。"""
    by_sensor: Dict[str, Dict[str, int]] = {}
    total = 0
    alarms = 0
    for measurement in measurements:
        bucket = by_sensor.setdefault(measurement.sensor_id, {"total": 0, "false_alarm": 0})
        bucket["total"] += 1
        total += 1
        if measurement.is_false_alarm:
            bucket["false_alarm"] += 1
            alarms += 1
    return {
        "total_measurements": total,
        "false_alarms": alarms,
        "false_alarm_fraction": (alarms / total) if total else 0.0,
        "by_sensor": by_sensor,
    }


def update_sequence(
    measurements: Sequence[MeasurementRecord],
    sensor_id: Optional[str] = None,
    scan_times: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """更新序列统计：验证传感器是否真的按 `update_period_s` 出数。

    这里刻意区分两个概念，**混为一谈会得出错误结论**：

    * **扫描节奏**（`scan_times`）：传感器**实际开机扫描**的时刻。
      周期 2 s 的传感器在 dt=1 s 的仿真里，扫描间隔应**恒为 2 s**（标准差 0）。
      标准差不为 0 ⇒ "是否到更新时刻"的判定有 bug。
      它**与是否检测到目标无关**。
    * **输出节奏**（测量时刻）：真正产生测量的时刻。间隔会因**漏检**而变成 2 倍、
      3 倍周期，这是正确行为，**不是**周期抖动。

    早期版本只统计测量时刻，于是把一次漏检误报成"周期 1.21±0.43 s"，
    看起来像扫描节奏乱了，实际是统计口径错了。因此两者都给。
    """
    fresh = [
        m for m in measurements
        if m.is_fresh and not m.is_false_alarm
        and (sensor_id is None or m.sensor_id == sensor_id)
    ]
    measure_times = sorted({round(m.time_s, 9) for m in fresh})
    scans = sorted({round(t, 9) for t in (scan_times or [])})

    def _intervals(times: Sequence[float]) -> Tuple[float, float, float, float]:
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        mean, std = _mean_std(gaps)
        return mean, std, (min(gaps) if gaps else 0.0), (max(gaps) if gaps else 0.0)

    scan_mean, scan_std, scan_min, scan_max = _intervals(scans)
    out_mean, out_std, out_min, out_max = _intervals(measure_times)

    return {
        "sensor_id": sensor_id or "(全部)",
        # --- 扫描节奏（与检测无关，用于验证 update_period_s）---
        "scan_times": scans,
        "n_scans": len(scans),
        "scan_interval_mean_s": scan_mean,
        "scan_interval_std_s": scan_std,
        "scan_interval_min_s": scan_min,
        "scan_interval_max_s": scan_max,
        # --- 输出节奏（受漏检影响，属正常现象）---
        "update_times": measure_times,
        "n_updates": len(measure_times),
        "interval_mean_s": out_mean,
        "interval_std_s": out_std,
        "interval_min_s": out_min,
        "interval_max_s": out_max,
    }


def measurement_statistics(
    measurements: Sequence[MeasurementRecord],
    outcome_rows: Sequence[Dict[str, Any]],
    n_bins: int = 5,
    scan_times_by_sensor: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """汇总一份完整的测量级统计报告。"""
    sensors = sorted({m.sensor_id for m in measurements})
    for sensor_id in (scan_times_by_sensor or {}):
        if sensor_id not in sensors:
            sensors.append(sensor_id)
    return {
        "n_measurements": len(measurements),
        "n_false_alarms": sum(1 for m in measurements if m.is_false_alarm),
        "error_vs_range": error_vs_range(measurements, n_bins=n_bins),
        "no_data": no_data_breakdown(outcome_rows),
        "false_alarm": false_alarm_statistics(measurements),
        "update_sequences": [
            update_sequence(
                measurements, sensor,
                scan_times=(scan_times_by_sensor or {}).get(sensor),
            )
            for sensor in sensors
        ],
    }


# ----------------------------------------------------------------------
# 人类可读报告
# ----------------------------------------------------------------------


def format_statistics(stats: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("======== 测量级统计 ========")
    lines.append(f"测量总数 {stats.get('n_measurements', 0)}"
                 f"（其中虚警 {stats.get('n_false_alarms', 0)}）")

    lines.append("")
    lines.append("---- 距离分箱误差（验证相对误差模型）----")
    lines.append(f"{'距离区间 m':>22s} {'样本':>6s} "
                 f"{'距离误差均值':>14s} {'距离误差σ':>12s} {'相对σ':>10s} "
                 f"{'方位误差均值':>14s} {'方位误差σ':>12s}")
    for item in stats.get("error_vs_range", []):
        lines.append(
            f"{item['range_low_m']:>10.0f}~{item['range_high_m']:<10.0f} {item['n']:>6d} "
            f"{item['err_range_mean_m']:>14.3f} {item['err_range_std_m']:>12.3f} "
            f"{item['err_range_rel_std']:>10.5f} "
            f"{item['err_az_mean_deg']:>14.4f} {item['err_az_std_deg']:>12.4f}"
        )

    no_data = stats.get("no_data", {})
    lines.append("")
    lines.append("---- 「没有数据」逐原因分解（不是笼统一个丢测率）----")
    lines.append(f"总判定 {no_data.get('total_outcomes', 0)} 次，"
                 f"检测成功 {no_data.get('detected', 0)} 次，"
                 f"整体检测率 {no_data.get('overall_detection_rate', 0):.4f}")
    for reason, item in (no_data.get("by_reason") or {}).items():
        if item["count"] == 0:
            continue
        lines.append(f"  {item['cn']:<24s} {reason:<22s} "
                     f"{item['count']:>6d} 次  {item['rate']:.4f}")
    if no_data.get("by_dimension"):
        lines.append("  按维度：" + "，".join(
            f"{k}={v}" for k, v in sorted(no_data["by_dimension"].items())
        ))

    lines.append("")
    lines.append("---- 逐传感器 ----")
    for sensor, item in (no_data.get("by_sensor") or {}).items():
        lines.append(f"  {sensor}：检测率 {item['detection_rate']:.4f}"
                     f"（{item['detected']}/{item['total']}）")

    fa = stats.get("false_alarm", {})
    lines.append("")
    lines.append("---- 虚警 ----")
    lines.append(f"  虚警 {fa.get('false_alarms', 0)} / {fa.get('total_measurements', 0)}"
                 f" = {fa.get('false_alarm_fraction', 0):.4f}")

    lines.append("")
    lines.append("---- 更新周期（验证扫描节奏）----")
    for item in stats.get("update_sequences", []):
        lines.append(
            f"  {item['sensor_id']:<16s} 扫描 {item['n_scans']:>4d} 次，"
            f"扫描间隔 {item['scan_interval_mean_s']:.4f}s"
            f"±{item['scan_interval_std_s']:.6f}s"
            f"  ← 这一列才是周期是否正确的判据"
        )
        lines.append(
            f"  {'':<16s} 出数 {item['n_updates']:>4d} 次，"
            f"出数间隔 {item['interval_mean_s']:.4f}s"
            f"±{item['interval_std_s']:.6f}s"
            f"（含漏检导致的跳帧，属正常）"
        )
    return "\n".join(lines)
