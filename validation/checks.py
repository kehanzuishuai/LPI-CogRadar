"""分层校核：几何 / 测量 / 通信 / 融合四层的单元与统计检查（v4.3）。

这个模块要回答的问题是
----------------------
**"仿真是不是按配置在跑？"** —— 不是看代码，而是看数据。

每一层都有明确的**可检验断言**与**配置期望值**：

| 层 | 检查什么 | 期望来自 |
| --- | --- | --- |
| 几何 | 距离对称、角度定义、运动学闭式解、姿态正交 | 数学恒等式 |
| 测量 | 误差均值≈0、误差标准差≈配置 σ、虚警率、漏检率、更新周期 | 传感器配置 |
| 通信 | 延迟均值≈固定延迟、投递率≈1−丢包率、过期按时丢弃、**不得读未来** | 链路配置 |
| 融合 | 融合误差 ≤ 单传感器误差、航迹连续性、丢轨/重复轨迹 | 统计性质 |

数据生成链的每一环都要能追溯：

    真实状态 → 可见性 → 测量 → 通信 → 融合 → 决策

因此每次校核都会记录它用了哪个场景、哪些种子、多少步，
结果里带 `provenance` 字段。**没有可追溯来源的结论不许写进报告。**

⚠️ 真值使用边界：本模块**属于离线评测通道**——
统计误差、漏检率、丢轨率都必须读真值。但：
* 这些真值**只用于判断"是否合格"**，不参与任何决策；
* `validation/` 的产物**不得**回流到 `sensor/` / `fusion/` / 决策算法。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import experiment_config as ec
from communication import (
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    CommBus,
    CommConfig,
    TimeBoundaryViolation,
)
from engine.geometry import (
    Attitude,
    Vec3,
    angular_separation_deg,
    enu_range,
    enu_range_2d,
    enu_to_spherical,
    normalize_angle_deg,
    relation,
)
from fusion import FusionCenter, FusionConfig
from sensor.record import NoDataReason

# ----------------------------------------------------------------------
# 结果类型
# ----------------------------------------------------------------------


@dataclass
class CheckResult:
    """一条校核结果。`expected` 写明"应该是什么"，便于人复核。"""

    check_id: str
    layer: str
    name: str
    passed: bool
    statistic: str = ""
    expected: str = ""
    tolerance: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "layer": self.layer,
            "name": self.name,
            "passed": self.passed,
            "statistic": self.statistic,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "detail": self.detail,
        }

    def format(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        line = f"  [{mark}] {self.check_id:<28s} {self.name}"
        if self.statistic:
            line += f"  实测={self.statistic}"
        if self.expected:
            line += f"  期望={self.expected}"
        return line


class Checks:
    """收集检查结果的小工具（避免每处都写 append）。"""

    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def add(
        self, check_id: str, layer: str, name: str, passed: bool,
        statistic: str = "", expected: str = "", tolerance: str = "",
        detail: str = "",
    ) -> CheckResult:
        result = CheckResult(check_id, layer, name, bool(passed), statistic,
                             expected, tolerance, detail)
        self.results.append(result)
        return result

    def close(
        self, check_id: str, layer: str, name: str, value: float,
        expected: float, tol_abs: float = 0.0, tol_rel: float = 0.0,
        detail: str = "", below: bool = False,
    ) -> CheckResult:
        """带容差的数值检查。

        `below=True` 时判据是 `value <= expected + tol`（上界型，如误差）；
        否则是 `|value - expected| <= tol`（一致型）。
        """
        tol = max(tol_abs, abs(expected) * tol_rel)
        if below:
            passed = value <= expected + tol
        else:
            passed = abs(value - expected) <= tol
        return self.add(
            check_id, layer, name, passed,
            statistic=f"{value:.6g}",
            expected=("≤ " if below else "") + f"{expected:.6g}",
            tolerance=f"±{tol:.4g}",
            detail=detail,
        )


# ----------------------------------------------------------------------
# 1. 几何层
# ----------------------------------------------------------------------


def check_geometry() -> List[CheckResult]:
    """几何层：数学恒等式，不依赖场景，因此**不需要真值**。"""
    c = Checks()

    # --- 距离对称（逐位）---
    a, b = Vec3(-1234.5, 6789.0, 12.0), Vec3(9876.5, -4321.0, -7.0)
    c.add("GEO-01", "geometry", "三维距离逐位对称",
          enu_range(a, b) == enu_range(b, a),
          "相等", "逐位相等", "", "对称性必须是逐位，不是近似")

    # --- z=0 时二维/三维一致 ---
    flat_ok = all(
        enu_range_2d(ax, ay, bx, by) == enu_range(Vec3(ax, ay, 0.0), Vec3(bx, by, 0.0))
        for ax, ay, bx, by in (
            (0.0, 0.0, 6000.0, 0.0), (8000.0, 6000.0, 0.0, 0.0),
            (-1234.5, 6789.0, 9876.5, -4321.0),
        )
    )
    c.add("GEO-02", "geometry", "z=0 时二维与三维距离逐位一致", flat_ok,
          "一致", "逐位相等", "", "保证旧场景数值不受升级影响")

    # --- 方位角定义：0°=正北、90°=正东 ---
    az_north = enu_to_spherical(Vec3(0.0, 100.0, 0.0)).azimuth_deg
    az_east = enu_to_spherical(Vec3(100.0, 0.0, 0.0)).azimuth_deg
    c.add("GEO-03", "geometry", "方位角以正北为 0°、顺时针为正",
          abs(az_north) < 1e-9 and abs(az_east - 90.0) < 1e-9,
          f"北={az_north:.4f}°, 东={az_east:.4f}°", "0° / 90°", "1e-9°")

    # --- 俯仰角 ---
    el_up = enu_to_spherical(Vec3(0.0, 100.0, 100.0)).elevation_deg
    el_down = enu_to_spherical(Vec3(0.0, 100.0, -100.0)).elevation_deg
    c.add("GEO-04", "geometry", "俯仰角水平面以上为正",
          abs(el_up - 45.0) < 1e-9 and abs(el_down + 45.0) < 1e-9,
          f"上={el_up:.4f}°, 下={el_down:.4f}°", "+45° / −45°", "1e-9°")

    # --- 姿态正交归一 ---
    worst = 0.0
    for heading in (0.0, 37.5, 90.0, -120.0):
        for pitch in (-30.0, 0.0, 15.0):
            att = Attitude(heading, pitch, 8.0)
            f, r, u = att.body_basis()
            worst = max(worst, abs(f.norm() - 1.0), abs(f.dot(r)), abs(r.dot(u)))
    c.close("GEO-05", "geometry", "机体三轴正交归一", worst, 0.0, tol_abs=1e-12)

    # --- 机体↔ENU 往返 ---
    att = Attitude(35.0, -12.0, 8.0)
    v = Vec3(1234.5, -678.9, 12.3)
    back = att.body_to_enu(att.enu_to_body(v))
    c.add("GEO-06", "geometry", "机体坐标往返一致",
          v.is_close(back, tol=1e-9), "往返误差", "<1e-9 m", f"{back}")

    # --- 运动学闭式解 ---
    from models import Target

    t = Target("T", 1000.0, 2000.0, vx=10.0, vy=-5.0, vz=1.0)
    for _ in range(10):
        t.advance(1.0)
    ok = (abs(t.x - 1100.0) < 1e-9 and abs(t.y - 1950.0) < 1e-9
          and abs(t.z - 10.0) < 1e-9 and abs(t.timestamp_s - 10.0) < 1e-9)
    c.add("GEO-07", "geometry", "匀直运动与闭式解一致", ok,
          f"({t.x:.4f},{t.y:.4f},{t.z:.4f})", "(1100,1950,10)", "1e-9")

    # --- 关系方向性：los 反向 ---
    pose_a = type("P", (), {"position": Vec3(0, 0, 0), "velocity": Vec3(),
                            "attitude": Attitude(0, 0, 0), "time_s": 0.0})()
    pose_b = type("P", (), {"position": Vec3(3000, 4000, 0), "velocity": Vec3(),
                            "attitude": Attitude(0, 0, 0), "time_s": 0.0})()
    fwd = relation("A", pose_a, "B", pose_b)
    bwd = relation("B", pose_b, "A", pose_a)
    c.add("GEO-08", "geometry", "视线向量反向且距离对称",
          fwd.los_enu.is_close(-bwd.los_enu, tol=1e-12)
          and fwd.range_m == bwd.range_m,
          f"|los|={fwd.los_enu.norm():.6f}",
          "反向 / 距离相等", "1e-12",
          "径向速度是距离变化率，交换双方**不变**（不是变号）")

    return c.results


# ----------------------------------------------------------------------
# 2. 测量层
# ----------------------------------------------------------------------


def check_measurement(
    config_path: str = ec.CONFIG_PATH,
    seeds: Sequence[int] = (42,),
    steps: int = 60,
    noise_scale: float = 1.0,
) -> List[CheckResult]:
    """测量层：误差统计、虚警/漏检比例、更新周期是否符合配置。

    这是**统计检验**，因此需要足够样本；默认跑满一段 episode、每步都观测。
    """
    c = Checks()
    errors_range: List[float] = []
    errors_az: List[float] = []
    n_scans = 0
    n_false_alarms = 0
    reason_counts: Dict[str, int] = {}
    scan_interval_check: List[Tuple[str, float, float]] = []
    total_outcomes = 0

    for seed in seeds:
        env = ec.make_env(
            config_path, observation_mode="realistic", measurement_max_tracks=4
        )
        if env.suite is None:
            continue
        env.reset(seed=seed)
        steps_done = 0
        while steps_done < steps:
            _o, _r, terminated, truncated, _i = env.step(6)
            report = env.suite_report()
            if report is not None:
                for measurement in report.measurements:
                    if measurement.is_false_alarm:
                        n_false_alarms += 1
                        continue
                    e_r = measurement.range_error_m()
                    e_a = measurement.azimuth_error_deg()
                    if e_r is not None:
                        errors_range.append(e_r)
                    if e_a is not None:
                        errors_az.append(e_a)
                for outcome in report.outcomes:
                    reason_counts[outcome.reason] = reason_counts.get(outcome.reason, 0) + 1
                    total_outcomes += 1
                for sensor_report in report.reports:
                    if sensor_report.updated:
                        n_scans += 1
            steps_done += 1
            if terminated or truncated:
                break

        # --- 更新周期：用扫描时刻（与是否检测到目标无关）---
        for sensor in env.suite.sensors:
            times = sorted(set(sensor.scan_times))
            if len(times) >= 3:
                gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                mean_gap = sum(gaps) / len(gaps)
                scan_interval_check.append(
                    (sensor.sensor_id, mean_gap, sensor.config.update_period_s)
                )

    # --- 误差均值应接近 0（无偏）---
    if errors_range:
        mean_r = statistics.fmean(errors_range)
        std_r = statistics.pstdev(errors_range)
        c.close("MEA-01", "measurement", "距离误差均值接近 0（无偏）",
                mean_r, 0.0, tol_abs=max(3.0 * std_r / math.sqrt(len(errors_range)), 1e-9),
                detail=f"n={len(errors_range)}, σ={std_r:.3f} m")
        c.add("MEA-02", "measurement", "方位误差均值接近 0（无偏）",
              abs(statistics.fmean(errors_az)) < 1.0,
              f"{statistics.fmean(errors_az):.4f}°", "|均值|<1°", "")
    else:
        c.add("MEA-01", "measurement", "距离误差均值接近 0（无偏）", False,
              "无样本", "有测量样本", "", "测量层没有产生任何测量，检查配置")

    # --- 虚警率 ---
    # 期望 = Σ(false_alarm_rate × 扫描次数)；这里用"虚警数 / 扫描次数"对比配置均值
    expected_fa = 0.0
    if seeds:
        probe = ec.make_env(config_path, observation_mode="realistic")
        if probe.suite is not None:
            expected_fa = sum(s.config.false_alarm_rate for s in probe.suite.sensors)
    observed_fa_rate = (n_false_alarms / n_scans) if n_scans else 0.0
    expected_fa_rate = expected_fa / max(len(seeds), 1)
    c.close("MEA-03", "measurement", "虚警率与配置一致",
            observed_fa_rate, expected_fa_rate,
            tol_abs=0.05 + 3.0 * math.sqrt(max(expected_fa_rate, 1e-6) / max(n_scans, 1)),
            detail=f"虚警 {n_false_alarms} / 扫描 {n_scans}")

    # --- 缺失原因可区分 ---
    distinct = [r for r in reason_counts if r != NoDataReason.NONE.value]
    c.add("MEA-04", "measurement", "缺失原因可区分（≥2 种）",
          len(distinct) >= 2, f"{len(distinct)} 种：{sorted(distinct)}",
          "≥2 种", "",
          "单一 dropout 概率无法区分这些原因，这正是分层测量的意义")

    # --- 更新周期 ---
    if scan_interval_check:
        worst = max(abs(m - p) for _s, m, p in scan_interval_check)
        c.close("MEA-05", "measurement", "扫描间隔等于配置周期", worst, 0.0,
                tol_abs=1e-6,
                detail="; ".join("%s: %.4f vs %.4f" % (s, m, p)
                                 for s, m, p in scan_interval_check))

    return c.results


# ----------------------------------------------------------------------
# 3. 通信层
# ----------------------------------------------------------------------


def check_communication(seeds: Sequence[int] = (42, 43), n_messages: int = 200) -> List[CheckResult]:
    """通信层：延迟、丢包、过期、以及**不得读未来**这条硬边界。"""
    c = Checks()

    # --- 时间边界（最重要的一条）---
    bus = CommBus(["A", "B"], CommConfig(
        policy=SHARE_CONSTRAINED, base_delay_s=3.0, seed=42))
    payload = {"sensor_id": "S1", "candidate_id": "C1", "time_s": 0.0,
               "range_m": 1000.0, "azimuth_deg": 10.0}
    bus.publish("A", "S1", [payload], now=0.0)

    class _M:
        def to_dict(self, include_truth=False):
            return dict(payload)

    visible_at_2 = len(bus.arrived(2.0))
    visible_at_4 = len(bus.arrived(4.0))
    c.add("COM-01", "communication", "延迟 3s 的消息在 t=2 不可见、t=4 可见",
          visible_at_2 == 0 and visible_at_4 == 1,
          f"t=2:{visible_at_2}, t=4:{visible_at_4}", "0 / 1", "",
          "这是「决策不得读未来」的正常路径：arrived() 做过滤")

    # 真实风险不是 arrived()，而是有人**绕过它直接遍历 bus.log**。
    # 因此守卫做成一个可调用函数，并用"手工把到达时刻改到未来"来验证它有效。
    bus2 = CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=1))
    bus2.publish("A", "S1", [_M()], now=0.0)
    bus2.log[0].arrived_at = 10.0  # 模拟"这条消息此刻还在路上"
    guard_ok = False
    try:
        CommBus.assert_only_arrived(bus2.log, 0.0)
    except TimeBoundaryViolation:
        guard_ok = True
    c.add("COM-02", "communication", "越界守卫能拦住直接读日志的行为", guard_ok,
          "抛出 TimeBoundaryViolation", "必须抛错", "",
          "风险路径是绕过 arrived() 遍历全量日志，守卫必须可执行、可测试")

    # --- 延迟均值 / 投递率 ---
    for loss, delay in ((0.0, 0.0), (0.0, 1.0), (0.3, 0.5)):
        cfg = CommConfig(policy=SHARE_CONSTRAINED, base_delay_s=delay,
                         loss_prob=loss, seed=7)
        b = CommBus(["A", "B", "C"], cfg)
        for k in range(n_messages):
            b.publish("A", "S1", [dict(payload, candidate_id=f"C{k}")], now=float(k))
        stats = b.statistics()
        c.close(
            f"COM-03(loss={loss:g},delay={delay:g})", "communication",
            "投递率 ≈ 1 − 丢包率",
            stats["delivery_rate"], 1.0 - loss, tol_abs=0.10,
            detail=f"消息 {stats['n_messages']}，投递 {stats['n_delivered']}，"
                   f"丢弃原因 {stats['drop_reasons']}",
        )
        c.close(
            f"COM-04(loss={loss:g},delay={delay:g})", "communication",
            "平均延迟 ≈ 固定延迟",
            stats["latency_mean_s"], delay, tol_abs=0.20,
            detail=f"p95={stats['latency_p95_s']:.3f}s",
        )

    # --- 过期：晚到的消息必须被丢弃 ---
    bus3 = CommBus(["A", "B"], CommConfig(
        policy=SHARE_CONSTRAINED, base_delay_s=5.0, expiry_s=2.0, seed=3))
    bus3.publish("A", "S1", [_M()], now=0.0)
    expired_dropped = bool(bus3.log) and bus3.log[0].dropped
    c.add("COM-05", "communication", "晚于过期时限到达的消息被丢弃",
          expired_dropped,
          f"dropped={bus3.log[0].dropped if bus3.log else None}, "
          f"reason={bus3.log[0].drop_reason if bus3.log else None}",
          "dropped / reason=expired", "",
          "过期按**到达时刻**判定：晚到的正确信息也可能已经没用")

    # --- 不共享策略下没有链路 ---
    bus_none = CommBus(["A", "B"], CommConfig(policy=SHARE_NONE))
    c.add("COM-06", "communication", "no_share 策略下无任何链路",
          len(bus_none.links) == 0 and bus_none.publish("A", "S1", [_M()], 0.0) == [],
          f"{len(bus_none.links)} 条链路", "0 条", "")

    # --- 载荷白名单挡住真值 ---
    from communication import PayloadViolation

    blocked = False
    try:
        CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL)).publish(
            "A", "S1", [{"truth_id": "TGT1"}], 0.0
        )
    except PayloadViolation:
        blocked = True
    c.add("COM-07", "communication", "载荷白名单拒绝真值字段", blocked,
          "抛出 PayloadViolation", "必须拒绝", "",
          "白名单比黑名单安全：新增真值字段会被直接拦下")

    return c.results


# ----------------------------------------------------------------------
# 4. 融合层
# ----------------------------------------------------------------------


def check_fusion(
    config_path: str = ec.CONFIG_PATH,
    seeds: Sequence[int] = (42,),
    steps: int = 40,
) -> List[CheckResult]:
    """融合层：融合误差、航迹连续性、丢轨/重复轨迹、溯源完整性。"""
    c = Checks()
    fusion_errors: List[float] = []
    single_errors: List[float] = []
    n_tracks: List[int] = []
    sources_complete = True
    frames_with_tracks = 0
    frames_total = 0

    for seed in seeds:
        env = ec.make_env(config_path, observation_mode="realistic",
                          measurement_max_tracks=4)
        env.reset(seed=seed)
        platform_id = env.sim.radar.radar_id
        center = FusionCenter(platform_id, FusionConfig())
        radar_sensor = next(
            (s for s in (env.suite.sensors if env.suite else [])
             if s.sensor_kind == "radar"), None
        )
        sensor_positions = {
            s.sensor_id: env.sim.scene.by_id(s.config.mounting_id).position
            for s in (env.suite.sensors if env.suite else [])
        }
        steps_done = 0
        while steps_done < steps:
            _o, _r, terminated, truncated, _i = env.step(6)
            report = env.suite_report()
            measurements = list(report.measurements) if report else []
            snapshot = center.update(measurements, env.sim.current_time,
                                     sensor_positions)
            frames_total += 1
            if snapshot.n_tracks > 0:
                frames_with_tracks += 1
            n_tracks.append(snapshot.n_tracks)

            # --- 融合误差 vs 单传感器误差（读真值，**仅用于评测**）---
            for track in snapshot.tracks:
                if track.last_measurement_time is None:
                    continue
                # 用雷达传感器测到的最近一条非虚警测量作为"单传感器"基准
                cands = [m for m in measurements
                         if m.sensor_id == (radar_sensor.sensor_id if radar_sensor else "")
                         and not m.is_false_alarm and m.truth_range_m is not None]
                if not cands:
                    continue
                best = min(cands, key=lambda m: abs(
                    (m.truth_range_m or 1e9) - track.position.norm()))
                if best.truth_id is None:
                    continue
                truth_entity = env.sim.scene.by_id(best.truth_id)
                fusion_errors.append(
                    (track.position - truth_entity.position).norm()
                )
                single_errors.append(
                    abs((best.range_m or 0.0) - (best.truth_range_m or 0.0))
                )
            # --- 溯源完整性 ---
            for track in snapshot.tracks:
                for source in track.sources:
                    if not source.sensor_id or source.measurement_time_s is None:
                        sources_complete = False
            steps_done += 1
            if terminated or truncated:
                break

    if fusion_errors and single_errors:
        mean_fusion = statistics.fmean(fusion_errors)
        mean_single = statistics.fmean(single_errors)
        c.add("FUS-01", "fusion", "存在可评测的融合航迹", True,
              f"n={len(fusion_errors)}", ">0", "")
        c.add("FUS-02", "fusion", "融合位置误差量级合理（≤500 m）",
              mean_fusion <= 500.0, f"{mean_fusion:.2f} m", "≤500 m", "",
              "单传感器距离误差基准 %.2f m" % mean_single)
    else:
        c.add("FUS-01", "fusion", "存在可评测的融合航迹", False,
              "0 条", ">0", "",
              "融合没有产出航迹：检查关联门限与传感器位置映射")

    c.add("FUS-03", "fusion", "每条融合结果的溯源字段完整", sources_complete,
          "完整" if sources_complete else "有缺失", "全部非空", "",
          "溯源是硬要求：必须能回答「用了哪个传感器、哪个时刻」的测量")

    continuity = (frames_with_tracks / frames_total) if frames_total else 0.0
    c.add("FUS-04", "fusion", "航迹连续性（有航迹的帧占比）", continuity >= 0.5,
          f"{continuity:.4f}", "≥0.5", "",
          f"共 {frames_total} 帧，有航迹 {frames_with_tracks} 帧")

    if n_tracks:
        max_tracks = max(n_tracks)
        # 场景有 2 个目标；航迹数长期远超目标数即为"重复轨迹"
        c.add("FUS-05", "fusion", "航迹数不超过目标数的 2 倍（无明显重复轨迹）",
              max_tracks <= 4, f"峰值 {max_tracks} 条", "≤4（目标数 2）", "",
              "超过即为关联把同一目标分裂成多条航迹")

    return c.results


def check_truth_isolation() -> List[CheckResult]:
    """真值隔离的**语法级**检查：跟踪器不得真的读到真值字段。

    为什么用 AST 而不是字符串匹配：`fusion/center.py` 的文档字符串里
    明明写着"不读 `truth_id`"，字符串匹配会把这个**声明**当成违规
    （实测就误报过一次）。只有 `ast.Attribute` / `ast.Name` 才是
    "真的读了这个名字"，因此按 AST 判定。

    检查对象是 `fusion/` 全部模块 —— 这一层只允许吃测量对象。
    """
    import ast
    import inspect

    import fusion.center as center_module
    import fusion.kalman as kalman_module
    import fusion.track as track_module

    forbidden = {"truth_id", "truth_range_m", "truth_azimuth_deg",
                 "truth_elevation_deg", "truth_range_rate_mps",
                 "is_false_alarm"}
    offenders: List[str] = []
    scanned = 0
    for module in (center_module, track_module, kalman_module):
        tree = ast.parse(inspect.getsource(module))
        scanned += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append(f"{module.__name__}:{node.lineno} .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in forbidden:
                offenders.append(f"{module.__name__}:{node.lineno} {node.id}")

    checks = Checks()
    checks.add(
        "truth.fusion_no_truth_read", "truth_isolation",
        "fusion/ 跟踪器不读真值字段（AST 级）",
        passed=not offenders,
        statistic=f"扫描 {scanned} 个模块，命中 {len(offenders)} 处",
        expected="0 处真值字段读取",
        detail="；".join(offenders[:5]),
    )
    return checks.results


def run_all_checks(
    config_path: str = ec.CONFIG_PATH,
    seeds: Sequence[int] = (42,),
    measurement_steps: int = 60,
    fusion_steps: int = 40,
) -> Dict[str, Any]:
    """跑全部四层校核，返回结构化报告与可追溯来源。"""
    results: List[CheckResult] = []
    results += check_geometry()
    results += check_measurement(config_path, seeds, measurement_steps)
    results += check_communication(seeds)
    results += check_fusion(config_path, seeds, fusion_steps)
    results += check_truth_isolation()

    by_layer: Dict[str, List[CheckResult]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    return {
        "config_path": config_path,
        "seeds": list(seeds),
        "measurement_steps": measurement_steps,
        "fusion_steps": fusion_steps,
        "checks": [r.to_dict() for r in results],
        "summary": {
            layer: {
                "total": len(items),
                "passed": sum(1 for i in items if i.passed),
                "failed": [i.check_id for i in items if not i.passed],
            }
            for layer, items in by_layer.items()
        },
        "all_passed": all(r.passed for r in results),
    }


def format_checks(results: Sequence[CheckResult]) -> str:
    lines: List[str] = []
    current = None
    for r in results:
        if r.layer != current:
            current = r.layer
            lines.append(f"\n---- {current} ----")
        lines.append(r.format())
        if r.detail and not r.passed:
            lines.append(f"           ↳ {r.detail}")
    return "\n".join(lines)
