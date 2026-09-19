"""传感器测量层单元测试（v4.2）。

重点验证用户明确要求的几条：

1. **五种以上「没有数据」的原因可以区分**，且判定顺序正确
   （不在视场 / 超作用距离 / 被遮挡 / 未到更新时刻 / 检测遗漏 / 传感器不可用）；
2. **真值 ID 不进入算法可见输入**——`fusion` 打包时不得读 `truth_*`，
   也不得调用 `Sensor.truth_of_candidate()`；
3. **噪声只改变"看到什么"，不改变真实轨迹与奖励**（逐位断言）；
4. 测量记录**带时间戳/传感器 ID/候选 ID/估计值/协方差/置信度**；
5. 虚警可配置、且**没有真值 ID**；
6. 遮挡模型（AABB / 球）的几何正确性与"只取线段内部交点"这条纪律。

不需要 torch，base 环境即可运行。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec  # noqa: E402
from engine.geometry import Vec3  # noqa: E402
from sensor import (  # noqa: E402
    BoxOccluder,
    EsmSensor,
    OcclusionModel,
    RadarSensor,
    SensorConfig,
    SensorSuite,
    SphereOccluder,
    TrackTableConfig,
    assert_no_truth_leak,
    fuse_measurements,
)
from sensor.record import NO_DATA_REASONS, NoDataReason  # noqa: E402

LEGACY_CONFIG = "config/radar_scenario_v1.json"


def _radar_sensor(**overrides) -> RadarSensor:
    params = dict(
        sensor_id="RS1", mounting_id="RADAR1", sensor_kind="radar",
        max_range_m=20000.0, min_range_m=0.0,
        az_fov_deg=180.0, el_fov_deg=90.0, update_period_s=1.0,
        range_sigma_rel=0.0, range_sigma_abs_m=0.0,
        az_sigma_deg=0.0, el_sigma_deg=0.0, range_rate_sigma_mps=0.0,
        snr50_db=6.0, pd_slope_db=2.0, false_alarm_rate=0.0,
        tx_power_w=18.0, peak_gain_db=30.0, wavelength_m=0.1,
        bandwidth_hz=1.0e6, noise_figure_db=3.0, system_loss_db=3.0,
        temperature_k=290.0, observes_kind="target", provides_range=True, seed=42,
    )
    params.update(overrides)
    return RadarSensor(SensorConfig(**params))


def _scene():
    from engine.simulator import Simulator

    sim = Simulator(LEGACY_CONFIG)
    sim.load_config()
    sim.reset(seed=42)
    return sim


def _suite_with(sensor):
    return SensorSuite([sensor])


# ----------------------------------------------------------------------


class TestNoDataReasons(unittest.TestCase):
    """五种以上缺失原因必须可区分，且判定顺序固定。"""

    def test_reason_enum_has_at_least_five(self) -> None:
        self.assertGreaterEqual(len(NO_DATA_REASONS), 5)
        for name in ("out_of_fov", "beyond_range", "occluded",
                     "not_updated", "missed_detection", "sensor_unavailable"):
            self.assertIn(name, NO_DATA_REASONS)

    def test_out_of_fov(self) -> None:
        """雷达视场 ±10°，目标在 90° 方向 → 不在视场。"""
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=10.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        reasons = {o.reason for o in report.outcomes}
        self.assertIn(NoDataReason.OUT_OF_FOV.value, reasons)

    def test_beyond_range(self) -> None:
        """作用距离 1 km，目标在 4~6 km → 超作用距离。"""
        sim = _scene()
        sensor = _radar_sensor(max_range_m=1000.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        reasons = {o.reason for o in report.outcomes}
        self.assertEqual(reasons, {NoDataReason.BEYOND_RANGE.value})

    def test_min_range_blind_zone(self) -> None:
        """近界盲区也算不可见（min_range_m）。"""
        sim = _scene()
        sensor = _radar_sensor(min_range_m=50000.0, max_range_m=100000.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.BEYOND_RANGE.value})

    def test_occluded(self) -> None:
        """在视场、在距离内，但视线被遮挡体切断。"""
        sim = _scene()
        target = sim.targets[0]  # (6000, 0)
        sensor = _radar_sensor(az_fov_deg=180.0)
        blocker = SphereOccluder(
            occluder_id="HILL", center_x=3000.0, center_y=0.0, center_z=0.0,
            radius_m=500.0,
        )
        suite = SensorSuite([sensor], OcclusionModel([blocker]))
        report = suite.observe(sim.scene, 0.0)
        by_truth = {o.truth_id: o for o in report.outcomes}
        self.assertEqual(by_truth[target.target_id].reason,
                         NoDataReason.OCCLUDED.value)

    def test_not_updated(self) -> None:
        """周期 3 s 的传感器在第 1、2 步不应产生新测量。"""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=3.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        suite.observe(sim.scene, 0.0)  # 首次扫描
        report = suite.observe(sim.scene, 1.0)
        self.assertFalse(report.updated)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.NOT_UPDATED.value})

    def test_missed_detection(self) -> None:
        """低功率 + 远目标 → 通过几何检查但检测概率不足。"""
        sim = _scene()
        sensor = _radar_sensor(
            tx_power_w=1e-9, az_fov_deg=180.0, snr50_db=1000.0,
        )
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.MISSED_DETECTION.value})

    def test_sensor_unavailable(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(available=False, az_fov_deg=180.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.SENSOR_UNAVAILABLE.value})
        self.assertEqual(report.measurements, [])

    def test_precedence_availability_before_geometry(self) -> None:
        """不可用的传感器不该给出任何几何判定（顺序纪律）。"""
        sim = _scene()
        # 距离极远（本会报 beyond_range）+ 不可用 → 必须报不可用
        sensor = _radar_sensor(available=False, max_range_m=1.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.SENSOR_UNAVAILABLE.value})

    def test_precedence_range_before_fov(self) -> None:
        """距离与视场都不满足时，必须报距离（能量问题优先于指向问题）。"""
        sim = _scene()
        sensor = _radar_sensor(max_range_m=1.0, az_fov_deg=0.001)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.BEYOND_RANGE.value})

    def test_precedence_not_updated_skips_geometry(self) -> None:
        """未到更新时刻时不应拿当前真值去算几何。"""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=10.0, max_range_m=1.0)
        suite = _suite_with(sensor)
        suite.observe(sim.scene, 0.0)
        report = suite.observe(sim.scene, 1.0)
        self.assertEqual({o.reason for o in report.outcomes},
                         {NoDataReason.NOT_UPDATED.value})


class TestMeasurementRecord(unittest.TestCase):
    def test_record_has_required_fields(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0, range_sigma_rel=0.01)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertTrue(report.measurements)
        for record in report.measurements:
            self.assertTrue(record.sensor_id)
            self.assertTrue(record.candidate_id)
            self.assertIsInstance(record.time_s, float)
            self.assertIsNotNone(record.range_m)
            self.assertIsNotNone(record.azimuth_deg)
            self.assertIsNotNone(record.elevation_deg)
            self.assertIsNotNone(record.range_rate_mps)
            self.assertIsNotNone(record.std_range_m)
            self.assertIsNotNone(record.covariance)
            self.assertIsInstance(record.confidence, float)

    def test_candidate_id_is_not_truth_id(self) -> None:
        """算法看到的是候选编号，不是真值 ID。"""
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        truth_ids = {t.target_id for t in sim.targets}
        for record in report.measurements:
            self.assertNotIn(record.candidate_id, truth_ids)
            self.assertTrue(record.candidate_id.startswith("RS1-"))

    def test_candidate_mapping_is_eval_only(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        record = report.measurements[0]
        self.assertEqual(sensor.truth_of_candidate(record.candidate_id),
                         record.truth_id)

    def test_to_dict_hides_truth_by_default(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        from sensor.fusion import FORBIDDEN_KEY_PREFIXES
        payload = report.measurements[0].to_dict(include_truth=False)
        # 必须用**完整的**禁止前缀集（含 is_false_alarm）。
        # 早期这里只查了 truth / err_，因此漏掉了 is_false_alarm 被无条件导出 ——
        # 那是 oracle 提示（“这条别信”），真实接收机不可能标注它。
        # 该缺陷最终是被 communication 的字段白名单抓出来的。
        for key in payload:
            for prefix in FORBIDDEN_KEY_PREFIXES:
                self.assertFalse(
                    key.startswith(prefix), f"导出结果出现禁止字段 {key!r}"
                )

    def test_to_dict_with_truth_for_evaluation(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0, range_sigma_abs_m=1.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        payload = report.measurements[0].to_dict(include_truth=True)
        self.assertIn("truth_id", payload)
        self.assertIn("err_range_m", payload)

    def test_covariance_is_diagonal_and_positive(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0, range_sigma_abs_m=10.0,
                               az_sigma_deg=0.5)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        cov = report.measurements[0].covariance
        self.assertAlmostEqual(cov[0][0], 100.0)
        self.assertAlmostEqual(cov[1][1], 0.25)
        self.assertEqual(cov[0][1], 0.0)
        self.assertEqual(cov[1][0], 0.0)

    def test_passive_sensor_has_no_range(self) -> None:
        """单站被动传感器没有距离量测 —— 距离维必须是 None / inf。"""
        sim = _scene()
        sensor = EsmSensor(SensorConfig(
            sensor_id="ES1", mounting_id="ESM1", sensor_kind="esm",
            max_range_m=1.0e6, az_fov_deg=180.0, el_fov_deg=90.0,
            update_period_s=1.0, az_sigma_deg=1.0, el_sigma_deg=1.0,
            peak_gain_db=12.0, provides_range=False, observes_kind="radar",
            snr50_db=-50.0, pd_slope_db=2.0, bandwidth_hz=2e6,
            noise_figure_db=6.0, system_loss_db=2.0, seed=42,
        ))
        report = _suite_with(sensor).observe(
            sim.scene, 0.0,
            context_by_sensor={"ES1": {"emitter_power_w": 18.0, "beam_gain_db": 30.0}},
        )
        self.assertTrue(report.measurements)
        record = report.measurements[0]
        self.assertIsNone(record.range_m)
        self.assertIsNone(record.range_rate_mps)
        self.assertFalse(record.has_range)
        self.assertEqual(record.covariance[0][0], float("inf"))

    def test_false_alarm_configurable_and_has_no_truth(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(false_alarm_rate=1.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        self.assertTrue(report.false_alarms)
        for alarm in report.false_alarms:
            self.assertTrue(alarm.is_false_alarm)
            self.assertIsNone(alarm.truth_id)
            self.assertIn("-FA", alarm.candidate_id)

    def test_no_false_alarm_when_rate_zero(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(false_alarm_rate=0.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        for step in range(5):
            report = suite.observe(sim.scene, float(step))
            self.assertEqual(report.false_alarms, [])


class TestNoTruthLeak(unittest.TestCase):
    """算法可见输入里绝不能出现真值。"""

    def test_fused_observation_has_no_truth(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0, range_sigma_abs_m=5.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        fused = fuse_measurements(report.measurements)
        assert_no_truth_leak(fused.to_dict())  # 不抛错即通过

    def test_assert_no_truth_leak_detects_violation(self) -> None:
        """检查器本身要有效——否则它只是装饰。"""
        with self.assertRaises(AssertionError):
            assert_no_truth_leak({"a": {"truth_id": "TGT1"}})
        with self.assertRaises(AssertionError):
            assert_no_truth_leak({"nested": [{"is_false_alarm": True}]})

    def test_fusion_does_not_call_truth_lookup(self) -> None:
        """把 `truth_of_candidate` 换成会抛错的桩，确认打包路径不调用它。"""
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)

        def _boom(_candidate_id):
            raise AssertionError("fusion 不得调用 truth_of_candidate()")

        sensor.truth_of_candidate = _boom  # type: ignore[assignment]
        fused = fuse_measurements(report.measurements)
        self.assertTrue(fused.vector)

    def test_env_info_measurements_hide_truth_by_default(self) -> None:
        env = ec.make_env(observation_mode="realistic")
        obs, _ = env.reset(seed=42)
        obs, _r, _t, _tr, _i = env.step(6)
        for record in env.measurement_records():
            for key in record:
                self.assertFalse(key.startswith("truth"), key)

    def test_env_info_measurements_can_expose_truth_for_eval(self) -> None:
        env = ec.make_env(observation_mode="realistic", expose_measurement_truth=True)
        env.reset(seed=42)
        env.step(6)
        records = env.measurement_records()
        self.assertTrue(any("truth_id" in r for r in records))


class TestTruthInvariance(unittest.TestCase):
    """噪声只改变"看到什么"，不改变真实轨迹与奖励。"""

    ACTION_SEQUENCE = [5, 2, 6, 10, 0, 1, 7, 3, 9, 4] * 4

    def _rollout(self, mode: str):
        env = ec.make_env(observation_mode=mode)
        obs, _ = env.reset(seed=42)
        trace = []
        for action in self.ACTION_SEQUENCE:
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append((
                round(info["tx_power_w"], 12),
                round(info["pd_min"], 12),
                round(info["intercept_prob"], 12),
                round(info["jam_noise_ratio"], 12),
                round(info["exposure_next"], 12),
                round(info["cumulative_energy_j"], 9),
                round(reward, 12),
            ))
            if terminated or truncated:
                break
        return trace

    def test_full_and_realistic_identical_truth(self) -> None:
        """真值与奖励必须逐位相同——传感器层只读不写。"""
        self.assertEqual(self._rollout("full"), self._rollout("realistic"))

    def test_full_and_ideal_identical_truth(self) -> None:
        self.assertEqual(self._rollout("full"), self._rollout("ideal"))

    def test_sensor_observe_does_not_mutate_scene(self) -> None:
        sim = _scene()
        before = [
            (e.entity_id, e.x, e.y, e.z, e.timestamp_s) for e in sim.scene.entities
        ]
        suite = _suite_with(_radar_sensor(az_fov_deg=180.0, false_alarm_rate=0.5))
        for step in range(5):
            suite.observe(sim.scene, float(step))
        after = [
            (e.entity_id, e.x, e.y, e.z, e.timestamp_s) for e in sim.scene.entities
        ]
        self.assertEqual(before, after)


class TestOcclusionModel(unittest.TestCase):
    def test_box_blocks_segment(self) -> None:
        box = BoxOccluder("B", -100.0, -100.0, -100.0, 100.0, 100.0, 100.0)
        self.assertTrue(box.intersects_segment(Vec3(-500.0, 0.0, 0.0),
                                               Vec3(500.0, 0.0, 0.0)))

    def test_box_behind_observer_does_not_block(self) -> None:
        """关键纪律：只取**线段内部**的交点。

        障碍物在观察者**背后**时，这条无限直线仍然与盒子相交，
        但那不构成遮挡。判据必须限定 t ∈ [0, 1]。
        """
        box = BoxOccluder("B", -400.0, -100.0, -100.0, -200.0, 100.0, 100.0)
        self.assertFalse(box.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                                Vec3(500.0, 0.0, 0.0)))

    def test_box_between_observer_and_target_blocks(self) -> None:
        """对照组：盒子真的夹在中间时必须判为遮挡。"""
        box = BoxOccluder("B", 200.0, -100.0, -100.0, 400.0, 100.0, 100.0)
        self.assertTrue(box.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                               Vec3(500.0, 0.0, 0.0)))

    def test_box_parallel_miss(self) -> None:
        box = BoxOccluder("B", -100.0, 500.0, -100.0, 100.0, 700.0, 100.0)
        self.assertFalse(box.intersects_segment(Vec3(-500.0, 0.0, 0.0),
                                                Vec3(500.0, 0.0, 0.0)))

    def test_endpoint_inside_counts_as_blocked(self) -> None:
        box = BoxOccluder("B", -100.0, -100.0, -100.0, 100.0, 100.0, 100.0)
        self.assertTrue(box.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                               Vec3(0.0, 0.0, 0.0)))

    def test_sphere_blocks_and_misses(self) -> None:
        sphere = SphereOccluder("S", 300.0, 0.0, 0.0, 100.0)
        self.assertTrue(sphere.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                                  Vec3(600.0, 0.0, 0.0)))
        self.assertFalse(sphere.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                                   Vec3(100.0, 0.0, 0.0)))
        self.assertFalse(sphere.intersects_segment(Vec3(0.0, 0.0, 0.0),
                                                   Vec3(0.0, 1000.0, 0.0)))

    def test_invalid_box_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoxOccluder("B", 100.0, 0.0, 0.0, -100.0, 1.0, 1.0)

    def test_invalid_sphere_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SphereOccluder("S", 0.0, 0.0, 0.0, -1.0)

    def test_first_blocker_reported(self) -> None:
        model = OcclusionModel([
            SphereOccluder("NEAR", 200.0, 0.0, 0.0, 50.0),
            SphereOccluder("FAR", 400.0, 0.0, 0.0, 50.0),
        ])
        self.assertEqual(model.first_blocker(Vec3(0.0, 0.0, 0.0),
                                             Vec3(600.0, 0.0, 0.0)), "NEAR")


class TestFusion(unittest.TestCase):
    def test_fixed_dimension_regardless_of_track_count(self) -> None:
        sim = _scene()
        for fov in (0.001, 180.0):
            sensor = _radar_sensor(az_fov_deg=fov)
            report = _suite_with(sensor).observe(sim.scene, 0.0)
            fused = fuse_measurements(report.measurements)
            self.assertEqual(len(fused.vector), TrackTableConfig().table_dim)

    def test_extra_scalars_appended(self) -> None:
        config = TrackTableConfig(max_tracks=2)
        fused = fuse_measurements([], config=config, extra_scalars=[1.0, 2.0])
        self.assertEqual(len(fused.vector), config.table_dim + 2)
        self.assertEqual(fused.vector[-2:], [1.0, 2.0])

    def test_empty_measurements_give_zero_quality(self) -> None:
        fused = fuse_measurements([])
        self.assertEqual(fused.observation_quality, 0.0)
        self.assertEqual(fused.n_measurements, 0)

    def test_sorting_is_deterministic(self) -> None:
        sim = _scene()
        sensor = _radar_sensor(az_fov_deg=180.0, false_alarm_rate=1.0)
        report = _suite_with(sensor).observe(sim.scene, 0.0)
        first = fuse_measurements(report.measurements).vector
        second = fuse_measurements(list(reversed(report.measurements))).vector
        self.assertEqual(first, second)

    def test_invalid_track_config(self) -> None:
        with self.assertRaises(ValueError):
            TrackTableConfig(max_tracks=0).validate()
        with self.assertRaises(ValueError):
            TrackTableConfig(range_scale_m=0.0).validate()


class TestSensorConfigValidation(unittest.TestCase):
    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _radar_sensor(max_range_m=0.0)
        with self.assertRaises(ValueError):
            _radar_sensor(min_range_m=100.0, max_range_m=100.0)
        with self.assertRaises(ValueError):
            _radar_sensor(az_fov_deg=0.0)
        with self.assertRaises(ValueError):
            _radar_sensor(update_period_s=0.0)
        with self.assertRaises(ValueError):
            _radar_sensor(false_alarm_rate=2.0)

    def test_unknown_field_reported(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SensorConfig.from_dict({"sensor_id": "X", "mounting_id": "R",
                                    "bogus_field": 1})
        self.assertIn("bogus_field", str(ctx.exception))

    def test_duplicate_sensor_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SensorSuite([_radar_sensor(), _radar_sensor()])


class TestUpdatePeriodSemantics(unittest.TestCase):
    def test_scan_times_follow_period_exactly(self) -> None:
        """周期 2 s、步长 1 s 时扫描时刻应为 0,2,4,..."""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=2.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        for step in range(8):
            suite.observe(sim.scene, float(step))
        self.assertEqual(sensor.scan_times, [0.0, 2.0, 4.0, 6.0])

    def test_non_integer_period_on_discrete_time_grid(self) -> None:
        """周期 2.5 s、仿真步长 1 s：更新点落在 **0, 3, 5, 8**。

        这里刻意记录一个容易误判的现象：仿真只在整数秒求值，
        所以"2.5 s 周期"实际表现为 2~3 秒交替的节奏，
        **不是** 0, 2.5, 5.0, 7.5。这是离散采样与连续周期的必然结果，
        不是 bug。判据用的是 `floor(t/period)` 是否增加，
        因此它**与步长解耦**：换步长时节奏自动跟着变，
        不会像"按步数取模"那样写死。
        """
        sim = _scene()
        sensor = _radar_sensor(update_period_s=2.5, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        for step in range(10):
            suite.observe(sim.scene, float(step))
        self.assertEqual(sensor.scan_times, [0.0, 3.0, 5.0, 8.0])

    def test_period_cadence_is_decoupled_from_step_size(self) -> None:
        """周期 2.5 s 在 dt=0.5 s 的网格上应落在 0, 2.5, 5.0, 7.5。"""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=2.5, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        t = 0.0
        for _ in range(16):
            suite.observe(sim.scene, t)
            t += 0.5
        self.assertEqual(sensor.scan_times, [0.0, 2.5, 5.0, 7.5])

    def test_held_measurements_between_scans(self) -> None:
        """未更新时沿用上一次测量，且 `age_s` 正确递增、`is_fresh=False`。"""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=3.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        first = suite.observe(sim.scene, 0.0)
        self.assertTrue(first.detections)
        self.assertEqual(first.held, [])

        second = suite.observe(sim.scene, 1.0)
        self.assertFalse(second.updated)
        self.assertTrue(second.held)
        for record in second.held:
            self.assertFalse(record.is_fresh)
            self.assertAlmostEqual(record.age_s, 1.0)

        third = suite.observe(sim.scene, 2.0)
        for record in third.held:
            self.assertAlmostEqual(record.age_s, 2.0)

    def test_held_records_do_not_leak_truth(self) -> None:
        """沿用值来自传感器自己的候选登记表，不含真值 ID。"""
        sim = _scene()
        sensor = _radar_sensor(update_period_s=3.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        suite.observe(sim.scene, 0.0)
        held_report = suite.observe(sim.scene, 1.0)
        truth_ids = {t.target_id for t in sim.targets}
        for record in held_report.held:
            self.assertNotIn(record.candidate_id, truth_ids)

    def test_unavailable_sensor_stops_providing_data(self) -> None:
        """传感器变为不可用后**必须停止输出**，不能继续沿用旧测量。

        这是安全语义：一个坏掉的传感器还在"给数据"，会让算法以为它仍在工作。
        因此不可用分支既不发 held，也清空沿用记忆。
        """
        sim = _scene()
        sensor = _radar_sensor(update_period_s=2.0, az_fov_deg=180.0)
        suite = _suite_with(sensor)
        first = suite.observe(sim.scene, 0.0)
        self.assertTrue(first.detections)
        held = suite.observe(sim.scene, 1.0)
        self.assertTrue(held.held)

        # 第 2 步关机
        sensor.config.available = False
        off = suite.observe(sim.scene, 2.0)
        self.assertEqual(off.measurements, [])
        self.assertEqual(off.held, [])
        self.assertEqual({o.reason for o in off.outcomes},
                         {NoDataReason.SENSOR_UNAVAILABLE.value})

        # 第 3 步仍未更新；记忆已被清空 → 不应有沿用值
        later = suite.observe(sim.scene, 3.0)
        self.assertEqual(later.held, [])

    def test_scan_invalidates_memory(self) -> None:
        """扫描帧作废旧记忆：本帧没测到就不该继续沿用（避免幻影航迹）。

        这里用**作用距离缩到 0** 的方式让本帧必然什么都测不到
        （早先用"把视场缩到 0.001°"是不行的：TGT2 正好在机头正前方，
        机体方位恰好是 0° ≤ 0.001°，照样落在视场内）。
        """
        sim = _scene()
        sensor = _radar_sensor(update_period_s=2.0, az_fov_deg=180.0,
                               false_alarm_rate=0.0)
        suite = _suite_with(sensor)
        suite.observe(sim.scene, 0.0)
        held = suite.observe(sim.scene, 1.0)
        self.assertTrue(held.held)

        # 第 2 步重新扫描，但本帧因距离门限什么都测不到
        sensor.config.min_range_m = 0.0
        sensor.config.max_range_m = 0.001
        rescanned = suite.observe(sim.scene, 2.0)
        self.assertTrue(rescanned.updated)
        self.assertEqual(rescanned.detections, [])

        # 第 3 步未更新，但记忆已被扫描帧清空 → 不应有沿用值
        after = suite.observe(sim.scene, 3.0)
        self.assertEqual(after.held, [])


class TestObservationModes(unittest.TestCase):
    def test_all_four_modes_build(self) -> None:
        expected = {"full": 12, "pomdp": 16, "ideal": 53, "realistic": 53}
        for mode, dim in expected.items():
            with self.subTest(mode=mode):
                kwargs = {}
                if mode == "pomdp":
                    kwargs["observation_noise"] = ec.observation_preset("moderate")
                env = ec.make_env(observation_mode=mode, **kwargs)
                obs, info = env.reset(seed=42)
                self.assertEqual(len(obs), dim)
                self.assertEqual(info["observation_space_dim"], dim)

    def test_measurement_modes_only_build_suite_there(self) -> None:
        for mode in ("full", "pomdp"):
            env = ec.make_env(observation_mode=mode)
            self.assertIsNone(env.suite)
        for mode in ("ideal", "realistic"):
            env = ec.make_env(observation_mode=mode)
            self.assertIsNotNone(env.suite)

    def test_ideal_has_no_stochastic_miss(self) -> None:
        """ideal 组只保留几何可见性，不应出现概率漏检。"""
        env = ec.make_env(observation_mode="ideal")
        env.reset(seed=42)
        for _ in range(10):
            _obs, _r, terminated, truncated, _i = env.step(6)
            report = env.suite_report()
            for sensor_report in report.reports:
                for outcome in sensor_report.outcomes:
                    self.assertNotEqual(
                        outcome.reason, NoDataReason.MISSED_DETECTION.value,
                        "ideal 模式不应出现 missed_detection",
                    )
            if terminated or truncated:
                break

    def test_ideal_has_no_false_alarm_and_no_noise(self) -> None:
        env = ec.make_env(observation_mode="ideal")
        env.reset(seed=42)
        env.step(6)
        for record in env.suite_report().measurements:
            self.assertFalse(record.is_false_alarm)
            self.assertEqual(record.std_az_deg, 0.0)

    def test_realistic_differs_from_ideal(self) -> None:
        """realistic 组必须有噪声与漏检的可能性。"""
        ideal = ec.make_env(observation_mode="ideal")
        real = ec.make_env(observation_mode="realistic")
        for env in (ideal, real):
            env.reset(seed=42)
        ideal_stds, real_stds = [], []
        for _ in range(10):
            for env, bucket in ((ideal, ideal_stds), (real, real_stds)):
                _o, _r, terminated, truncated, _i = env.step(6)
                for record in env.suite_report().measurements:
                    if record.std_az_deg is not None:
                        bucket.append(record.std_az_deg)
                if terminated or truncated:
                    break
        self.assertTrue(ideal_stds)
        self.assertTrue(real_stds)
        self.assertEqual(set(ideal_stds), {0.0})
        self.assertTrue(any(s > 0.0 for s in real_stds))

    def test_missing_reasons_surface_in_observation(self) -> None:
        """缺失原因会进入观测的统计通道（按原因分别计数）。"""
        env = ec.make_env(observation_mode="realistic")
        env.reset(seed=42)
        env.step(6)
        fused = env.fused_observation
        self.assertIsNotNone(fused)
        self.assertIn(NoDataReason.OUT_OF_FOV.value, fused.reason_counts)

    def test_legacy_full_mode_unchanged(self) -> None:
        """full 模式仍必须是 12 维真值观测（旧实验逐位复现的前提）。"""
        env = ec.make_env(observation_mode="full")
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 12)
        env.step(6)
        self.assertAlmostEqual(env._observation()[11], env.sim.exposure.value, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
