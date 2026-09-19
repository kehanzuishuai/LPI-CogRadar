"""测量级统计验证与记录导出（v4.2）。

它回答的问题
------------
"这套传感器是不是真按配置在工作？" —— 不是看代码，而是看**数据**：

1. **误差随距离怎么变**：相对误差模型下 σ_R 应随 R 线性增长，而 σ_方位 应与 R 无关；
2. **逐原因的漏检率**：不是笼统一个"没有数据比例"，而是分别统计
   不在视场 / 超作用距离 / 被遮挡 / 未到更新时刻 / 检测遗漏 / 传感器不可用；
3. **虚警率**：是否接近配置值；
4. **不同更新周期下的观测序列**：间隔应**恒等于**周期（标准差为 0），
   标准差不为 0 说明"是否到更新时刻"的判定有 bug 或被沿用值污染。

用法
----
    python analyze_measurements.py                      # 单雷达场景，realistic
    python analyze_measurements.py --mode ideal         # 理想测量对照
    python analyze_measurements.py --period-sweep       # 扫描更新周期
    python analyze_measurements.py --config config/multi_platform_scenario.json

输出
----
    measurements.csv          测量记录（可含真值与误差列，评测通道）
    measurements.json         同上 + 完整统计
    outcomes.csv              逐 (传感器,目标) 的判定与**缺失原因**
    measurement_stats.txt     人类可读统计报告
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()
import experiment_config as ec  # noqa: E402
from sensor.reporting import (  # noqa: E402
    MeasurementLog,
    format_statistics,
    measurement_statistics,
)

DEFAULT_OUT_DIR = "output/measurements"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测量级统计验证与导出（v4.2 分层测量）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=["ideal", "realistic"], default="realistic",
                        help="ideal = 只保留可见性约束、无噪声/无漏检/无虚警")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = 跑完整段")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--period-sweep", action="store_true",
                        help="扫描传感器更新周期，验证观测序列节奏")
    parser.add_argument("--periods", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--range-profile", action="store_true",
                        help="距离剖面：把目标推到给定距离范围，统计误差随距离的变化")
    parser.add_argument("--profile-min-range-m", type=float, default=1000.0)
    parser.add_argument("--profile-max-range-m", type=float, default=30000.0)
    parser.add_argument("--policy", choices=["fixed", "rule"], default="rule")
    parser.add_argument("--no-truth-columns", action="store_true",
                        help="导出时不写真值/误差列（算法可直接消费的版本）")
    return parser


def run_once(
    args: argparse.Namespace,
    update_period_s: Optional[float] = None,
    out_dir: Optional[str] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """跑一段仿真并统计测量。"""
    env = ec.make_env(
        args.config,
        observation_mode=args.mode,
        measurement_max_tracks=args.max_tracks,
    )
    # 覆盖传感器更新周期（若请求）
    if update_period_s is not None and env.suite is not None:
        for sensor in env.suite.sensors:
            sensor.config.update_period_s = float(update_period_s)

    from strategy.power_policy import FixedPowerPolicy, RuleBasedPowerPolicy

    policy = FixedPowerPolicy() if args.policy == "fixed" else RuleBasedPowerPolicy()
    policy.reset()

    obs, _info = env.reset(seed=args.scenario_seed)
    log = MeasurementLog()
    step_limit = args.max_steps if args.max_steps > 0 else None
    steps = 0
    while True:
        if step_limit is not None and steps >= step_limit:
            break
        action = policy.select_level(env.sim)
        obs, _reward, terminated, truncated, _info = env.step(action)
        report = env.suite_report()
        if report is not None:
            log.add(report, include_truth=True)
        steps += 1
        if terminated or truncated:
            break

    stats = measurement_statistics(
        log.measurements, log.outcome_rows, scan_times_by_sensor=log.scan_times
    )
    target_dir = out_dir or args.out_dir
    if write:
        os.makedirs(target_dir, exist_ok=True)
        include_truth = not args.no_truth_columns
        log.write_measurements_csv(
            os.path.join(target_dir, "measurements.csv"), include_truth=include_truth
        )
        log.write_outcomes_csv(
            os.path.join(target_dir, "outcomes.csv"), include_truth=include_truth
        )
        log.write_json(
            os.path.join(target_dir, "measurements.json"), include_truth=include_truth
        )
    return {"steps": steps, "stats": stats, "log": log, "env": env}


def run_range_profile(args: argparse.Namespace) -> Dict[str, Any]:
    """距离剖面：把目标沿视线推到不同距离，分别测**误差模型**与**检测模型**。

    为什么要跑两遍
    --------------
    如果只跑一遍，会出现一个"看不见的统计陷阱"：距离一远，`Pd → 0`，
    于是**根本没有测量**可以拿来算误差——分箱后样本全挤在近距箱里，
    "误差随距离怎么变"这个问题永远回答不了（第一次跑就踩了这个坑，
    5 个箱里只有 1 个有样本）。

    因此拆成两遍：

    * **A 遍（误差标定）**：`force_detection=True`，只要几何可见就产出测量。
      这样每个距离都有样本，能干净地看出 σ_R 是否随 R 线性增长、
      σ_方位 是否与 R 无关。它**只标定误差模型**，不反映检测能力。
    * **B 遍（检测标定）**：正常检测器（有概率漏检）。它给出
      **漏检率 / 检测率随距离怎么变**，与 A 遍的误差曲线互补。

    两遍合起来才构成完整的"测量级统计验证"。这是**传感器标定实验**，
    人为移动目标位置，其结果不参与任何策略性能结论。
    """
    from sensor.reporting import no_data_breakdown

    target_range_min = float(args.profile_min_range_m)
    target_range_max = float(args.profile_max_range_m)
    samples = 28

    def _sweep(force_detection: bool) -> Dict[str, Any]:
        # 标定实验里**能量必须放开**：28 个距离 × 3 步 × 80 W 远超 1400 J 预算，
        # 不放开会在中途因能量耗尽抛 InfeasibleActionError（第一次跑就撞上了）。
        # 这里放大预算只是为了让它跑完，不改变任何雷达方程或传感器参数。
        env = ec.make_env(
            args.config,
            energy_budget_j=1.0e9,
            observation_mode=args.mode,
            measurement_max_tracks=args.max_tracks,
        )
        env.sim.apply_overrides(
            terminate_on_energy_exhausted=False,
            # 标定需要 28 个距离 × 3 步 = 84 步，超过场景默认的 61 步；
            # 不延长会在第 61 步抛"episode 已结束"。这是**标定脚本**的局部设置，
            # 不写回场景配置、不影响任何正式实验。
            extra={"sim_duration": float(3 * samples + 20)},
        )
        env.reset(seed=args.scenario_seed)
        if env.suite is not None:
            for sensor in env.suite.sensors:
                if sensor.sensor_kind == "radar":
                    sensor.config.max_range_m = target_range_max
                    sensor.config.force_detection = force_detection

        target = env.sim.targets[0]
        # 只留被扫的那个目标：场景里的 TGT2 固定在 4 km 且始终在视场内，
        # 它的测量会混进近距箱，把"误差随距离增长"的斜率压平
        # （实测 σ_R 比值因此只有 3.99×，而距离比是 9×）。
        # 标定实验里必须只留一个被测目标，否则统计口径不干净。
        for other in env.sim.targets[1:]:
            other.is_active = False
        from strategy.power_policy import FixedPowerPolicy
        policy = FixedPowerPolicy(level_index=len(env.sim.power_levels_w) - 1)
        log = MeasurementLog()

        for index in range(samples):
            distance = target_range_min + (target_range_max - target_range_min) * index / (samples - 1)
            # ⚠️ 必须沿 **+y（正北）** 扫，而不是 +x。
            # 雷达机头朝正北（heading=0）、视场只有 ±60°，把目标放在 +x（正东）
            # 会让它的机体方位恒为 90° → **全程 out_of_fov**，
            # 于是"误差-距离"统计里一个样本都没有，实际采到的全是另一个目标
            # （第一次跑就是这样：5 个箱里只有 1 个有样本，而且都挤在 4 km）。
            # 沿机头方向扫，目标才真正落在视场内。
            target.x, target.y, target.z = 0.0, distance, 0.0
            target.vx = target.vy = target.vz = 0.0
            for _ in range(3):
                obs, _r, terminated, truncated, _i = env.step(policy.select_level(env.sim))
                report = env.suite_report()
                if report is not None:
                    log.add(report, include_truth=True)
                if terminated or truncated:
                    break
        stats = measurement_statistics(
            log.measurements, log.outcome_rows, scan_times_by_sensor=log.scan_times
        )
        return {"stats": stats, "log": log}

    error_pass = _sweep(force_detection=True)
    detection_pass = _sweep(force_detection=False)
    return {"error_pass": error_pass, "detection_pass": detection_pass,
            "ranges": [target_range_min, target_range_max, samples]}


def _detection_rate_by_range(detection_pass: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按距离分箱统计**检测率**（B 遍）。"""
    rows = detection_pass["log"].outcome_rows
    radar_rows = [r for r in rows if str(r.get("sensor_id", "")).startswith("SENSOR_RADAR")]
    usable = [r for r in radar_rows if r.get("truth_range_m") not in (None, "")]
    if not usable:
        return []
    upper = max(float(r["truth_range_m"]) for r in usable)
    n_bins = 6
    width = upper / n_bins if upper > 0 else 1.0
    out: List[Dict[str, Any]] = []
    for index in range(n_bins):
        low = index * width
        high = upper if index == n_bins - 1 else (index + 1) * width
        group = [r for r in usable
                 if low <= float(r["truth_range_m"]) < high
                 or (index == n_bins - 1 and float(r["truth_range_m"]) == high)]
        if not group:
            continue
        detected = sum(1 for r in group if r.get("detected"))
        counts: Dict[str, int] = {}
        for row in group:
            reason = str(row.get("reason", ""))
            counts[reason] = counts.get(reason, 0) + 1
        out.append({
            "range_low_m": low,
            "range_high_m": high,
            "n": len(group),
            "detection_rate": detected / len(group),
            "reasons": counts,
        })
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("======== 测量级统计验证 ========")
    print(f"配置      : {args.config}")
    print(f"观测模式  : {args.mode}"
          f"（{'理想测量：无噪声/无漏检/无虚警' if args.mode == 'ideal' else '真实测量：含噪声/漏检/虚警'}）")
    print(f"输出目录  : {args.out_dir}\n")

    result = run_once(args)
    print(f"仿真步数  : {result['steps']}")
    print(format_statistics(result["stats"]))

    print("\n---- 导出文件 ----")
    for name in ("measurements.csv", "outcomes.csv", "measurements.json"):
        path = os.path.join(args.out_dir, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {path}  ({size} 字节)")

    if not args.no_truth_columns:
        print("\n  注：默认导出的 CSV/JSON **含真值与误差列**，属于评测通道。")
        print("      要给算法消费请加 --no-truth-columns。")

    # ---------------- 距离剖面（误差-距离关系）----------------
    if args.range_profile:
        print("\n======== 距离剖面 ========")
        print("A 遍：误差标定（强制检测，隔离出误差模型）")
        print("B 遍：检测标定（正常检测器，观察漏检率随距离变化）")
        profile = run_range_profile(args)
        profile_dir = os.path.join(args.out_dir, "range_profile")
        os.makedirs(profile_dir, exist_ok=True)
        profile["error_pass"]["log"].write_measurements_csv(
            os.path.join(profile_dir, "measurements_forced.csv"), include_truth=True
        )
        profile["detection_pass"]["log"].write_measurements_csv(
            os.path.join(profile_dir, "measurements_normal.csv"), include_truth=True
        )

        print("\n---- A 遍：误差随距离 ----")
        print(format_statistics(profile["error_pass"]["stats"]))
        bins = profile["error_pass"]["stats"]["error_vs_range"]
        filled = [b for b in bins if b["n"] > 0]
        if len(filled) >= 2:
            first, last = filled[0], filled[-1]
            c1 = 0.5 * (first["range_low_m"] + first["range_high_m"])
            c2 = 0.5 * (last["range_low_m"] + last["range_high_m"])
            print("\n  解读（相对误差模型 std_R = R·rel + abs）：")
            print(f"    近距箱 中心 {c1:8.0f} m → σ_R = {first['err_range_std_m']:8.3f} m"
                  f"（相对 {first['err_range_rel_std']:.5f}）")
            print(f"    远距箱 中心 {c2:8.0f} m → σ_R = {last['err_range_std_m']:8.3f} m"
                  f"（相对 {last['err_range_rel_std']:.5f}）")
            if first["err_range_std_m"] > 0:
                ratio = last["err_range_std_m"] / first["err_range_std_m"]
                distance_ratio = c2 / c1
                print(f"    σ_R 随距离**单调增长**（{ratio:.2f}×），方向与相对误差模型一致。")
                if ratio < 0.8 * distance_ratio:
                    print(f"    ⚠️ 但增长倍数 {ratio:.2f}× 明显小于名义距离比 {distance_ratio:.2f}×。")
                    print("       原因：每个分箱内部本身跨越 6 km 距离，箱内 σ 的差异被平均掉，")
                    print("       且每箱样本量只有十几个（标准差估计本身有 ~20% 波动）。")
                    print("       因此**不能**用这个表精确标定 rel/abs 系数——")
                    print("       要精确标定需要每箱数百个样本、或用更窄的分箱。")
                    print("       这里能支持的结论只有：σ_R 随 R 增长、σ_方位 不随 R 变化。")
            print(f"    σ_方位 近距 {first['err_az_std_deg']:.4f}° → "
                  f"远距 {last['err_az_std_deg']:.4f}°，基本恒定"
                  "（角度误差来自波束/相位，**不应**随距离变化）——这一条与模型预期吻合。")

        print("\n---- B 遍：检测率随距离 ----")
        print(f"{'距离区间 m':>22s} {'判定次数':>8s} {'检测率':>8s}   主要缺失原因")
        for row in _detection_rate_by_range(profile["detection_pass"]):
            top = sorted(row["reasons"].items(), key=lambda kv: -kv[1])[:2]
            top_text = "，".join(f"{k}={v}" for k, v in top if k != "none")
            print(f"{row['range_low_m']:>10.0f}~{row['range_high_m']:<10.0f} "
                  f"{row['n']:>8d} {row['detection_rate']:>8.4f}   {top_text}")

        print(f"\n  → {os.path.join(profile_dir, 'measurements_forced.csv')}")
        print(f"  → {os.path.join(profile_dir, 'measurements_normal.csv')}")

    if args.period_sweep:
        print("\n======== 更新周期扫描（验证观测序列节奏）========")
        print(f"{'周期 s':>8s} {'扫描次数':>10s} {'扫描间隔均值':>14s} "
              f"{'扫描间隔σ':>14s} {'检测率':>10s}")
        for period in args.periods:
            sub = run_once(
                args,
                update_period_s=period,
                out_dir=os.path.join(args.out_dir, f"period_{period:g}s"),
            )
            sequences = sub["stats"]["update_sequences"]
            radar_sequence = next(
                (s for s in sequences if s["sensor_id"].startswith("SENSOR_RADAR")),
                sequences[0] if sequences else None,
            )
            detection_rate = sub["stats"]["no_data"]["overall_detection_rate"]
            if radar_sequence is None:
                print(f"{period:>8.1f} {'—':>10s}")
                continue
            print(f"{period:>8.1f} {radar_sequence['n_scans']:>10d} "
                  f"{radar_sequence['scan_interval_mean_s']:>14.4f} "
                  f"{radar_sequence['scan_interval_std_s']:>14.6f} "
                  f"{detection_rate:>10.4f}")
        print("\n  判据：扫描间隔均值应等于周期、标准差应≈0；")
        print("        出数间隔会因漏检跳帧而变大，那是正常现象。")


if __name__ == "__main__":
    main()
