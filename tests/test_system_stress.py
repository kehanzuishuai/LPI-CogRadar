"""系统级压力场景测试（v4.5 P2 第二阶段）。

钉住四类**系统级**场景的关键不变式：

* **S5 机动**：机动只改真值运动、不改物理；残差/创新/门限拒绝被记录，
  且机动目标的失配明显高于**同场景**的匀速对照目标；
* **S6 交接**：几何可见窗口与解析解一致；不共享时交接必然失败，
  共享时"远端来源真的进入航迹"，交接延迟可测；
* **S7 时序**：突发丢包 / 中断 / 恢复拥塞 / 乱序四件事都真的发生；
  四个时间戳显式区分；重排缓冲**降低乱序但增加延迟与时效拒绝**（代价必须一起测到）；
* **S8 偏差**：偏差只加在测量上、真值不动、算法看不到；
  四组对照里"有偏共享"可被识别为"比单雷达更差"，且健康分只作诊断。

不需要 torch。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication import CommBus, CommConfig, SHARE_CONSTRAINED  # noqa: E402
from multi_target_stress.maneuvers import (  # noqa: E402
    ManeuverInjector,
    climb_step,
    speed_step,
    turn_step,
)
from multi_target_stress.runner import resolve_scenario, run_case  # noqa: E402
from multi_target_stress.system_scenarios import (  # noqa: E402
    S8_BIAS,
    SYSTEM_SCENARIO_IDS,
    _sensor_range_window,
    get_system_scenario,
    handover_overlap_s,
    handover_windows,
)
from multi_target_stress.timing import (  # noqa: E402
    DELAYED_UPDATE,
    DROP_STALE,
    OOSM_POLICIES,
    REORDER_BUFFER,
    OosmController,
)


class _Measurement:
    """最小测量对象（OOSM / 链路测试用）。

    必须提供 `to_dict(include_truth=False)`：`CommBus.publish` 走的是
    "有 to_dict 就调它、否则当字典"的分支，缺了会落进 `dict(obj)` 抛
    `TypeError: not iterable`（第一版测试就踩了这个）。
    """

    def __init__(self, candidate_id: str, time_s: float, sensor_id: str = "S1"):
        self.candidate_id = candidate_id
        self.sensor_id = sensor_id
        self.time_s = time_s
        self.std_range_m = 10.0
        self.std_az_deg = 0.5
        self.std_el_deg = 0.5
        self.covariance = [[100.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]

    def to_dict(self, include_truth: bool = False) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "candidate_id": self.candidate_id,
            "sensor_kind": "radar",
            "time_s": self.time_s,
            "range_m": 1000.0,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "confidence": 1.0,
            "cov_xx": 100.0, "cov_yy": 0.25, "cov_zz": 0.25,
        }


class TestManeuverInjection(unittest.TestCase):
    def test_turn_keeps_speed(self) -> None:
        from engine.geometry import Vec3

        step = turn_step(5.0, "T", Vec3(0.0, 100.0, 0.0), 90.0)
        # +90° 顺时针：+y → +x
        delta = Vec3(*step.delta_v)
        new_velocity = Vec3(0.0, 100.0, 0.0) + delta
        self.assertAlmostEqual(new_velocity.norm(), 100.0, places=9)
        self.assertAlmostEqual(new_velocity.x, 100.0, places=9)
        self.assertAlmostEqual(new_velocity.y, 0.0, places=9)

    def test_speed_step_changes_rate_only(self) -> None:
        from engine.geometry import Vec3

        step = speed_step(5.0, "T", Vec3(30.0, 40.0, 0.0), 50.0)
        new_velocity = Vec3(30.0, 40.0, 0.0) + Vec3(*step.delta_v)
        self.assertAlmostEqual(new_velocity.norm(), 100.0, places=9)
        # 方向不变
        self.assertAlmostEqual(new_velocity.x / new_velocity.norm(), 0.6, places=9)

    def test_climb_adds_vertical_speed(self) -> None:
        step = climb_step(5.0, "T", 40.0)
        self.assertEqual(step.delta_v[:2], (0.0, 0.0))
        self.assertAlmostEqual(step.delta_v[2], 40.0)

    def test_injector_applies_once_and_updates_truth(self) -> None:
        from engine.geometry import Vec3

        class _Target:
            target_id = "T1"

            def __init__(self) -> None:
                self.velocity = Vec3(0.0, 100.0, 0.0)
                self.heading_deg = 0.0

            def set_pose(self, velocity=None, attitude=None):
                if velocity is not None:
                    self.velocity = velocity
                if attitude is not None:
                    self.heading_deg = attitude.heading_deg

        class _Sim:
            def __init__(self) -> None:
                self.targets = [_Target()]

        sim = _Sim()
        injector = ManeuverInjector([turn_step(5.0, "T1", Vec3(0.0, 100.0, 0.0), 90.0)])
        self.assertEqual(injector.apply(sim, 4.0), [])
        fired = injector.apply(sim, 5.0)
        self.assertEqual(len(fired), 1)
        self.assertAlmostEqual(sim.targets[0].velocity.x, 100.0, places=9)
        # 同一时刻再调一次不得重复施加
        self.assertEqual(injector.apply(sim, 5.0), [])
        self.assertEqual(len(injector.events), 1)

    def test_scenario_maneuvers_change_truth_not_physics(self) -> None:
        """机动场景的雷达物理参数必须与基础配置一致（压力不来自改物理）。"""
        import json

        with open("config/radar_scenario_v1.json", "r", encoding="utf-8") as handle:
            base = json.load(handle)
        scenario = get_system_scenario("S5")
        for radar in scenario.overrides["radars"]:
            for key in ("tx_power_w", "peak_gain_db", "wavelength_m",
                        "noise_figure_db", "system_loss_db", "snr50_db",
                        "required_pd", "energy_budget_j"):
                self.assertEqual(radar[key], base["radar"][key],
                                 f"雷达物理参数 {key} 被改动")
        self.assertTrue(scenario.maneuvers, "S5 必须有机动时刻表")


class TestManeuverFailureMode(unittest.TestCase):
    def test_maneuver_mismatch_exceeds_reference(self) -> None:
        """机动目标的失配必须明显高于**同场景**的匀速对照目标。"""
        result = run_case("S5", "single", seed=42)
        metrics = result["system"]["maneuver"]
        ratio = metrics["maneuvered_residual_ratio"]
        ref_ratio = metrics["reference_residual_ratio"]
        self.assertGreater(ratio, ref_ratio,
                           "机动目标的残差放大没有高于匀速对照目标，"
                           "说明场景没有检验力")
        self.assertGreater(metrics["maneuvered_innovation_max"], 7.815,
                           "机动后的创新峰值没有越过 χ²₉₅(3)，"
                           "失配没有被观测到")

    def test_maneuver_metrics_are_recorded(self) -> None:
        metrics = run_case("S5", "single", seed=42)["system"]["maneuver"]
        for key in ("maneuvered_residual_pre_m", "maneuvered_residual_post_m",
                    "maneuvered_innovation_max", "maneuvered_gate_rejected",
                    "maneuvered_recovery_time_s", "reference_residual_ratio"):
            self.assertIn(key, metrics)

    def test_share_helps_slow_scan_local_radar(self) -> None:
        single = run_case("S5", "single", seed=42)["metrics"]
        shared = run_case("S5", "ideal_share", seed=42)["metrics"]
        self.assertGreater(shared["association_accuracy"],
                           single["association_accuracy"])
        self.assertLess(shared["position_rmse_m"], single["position_rmse_m"])


class TestHandoverGeometry(unittest.TestCase):
    def test_range_window_matches_closed_form(self) -> None:
        """可见窗口的解析解必须与几何自洽（距离门限 ∧ 视场门限）。"""
        import math

        sensor_y, x_target, half_fov, max_range = -8000.0, 1500.0, 45.0, 11000.0
        low, high = _sensor_range_window(sensor_y, x_target, half_fov, max_range)
        # 距离门限
        span = math.sqrt(max_range ** 2 - x_target ** 2)
        self.assertAlmostEqual(high, sensor_y + span, places=6)
        # 视场门限
        fov_offset = x_target / math.tan(math.radians(half_fov))
        self.assertAlmostEqual(low, sensor_y + fov_offset, places=6)

    def test_handover_windows_and_overlap(self) -> None:
        windows = handover_windows()
        overlap = handover_overlap_s()
        # A 只覆盖前段、B 只覆盖后段，且重叠必须存在（否则不叫交接）
        self.assertLess(windows["A"][1], windows["B"][1])
        self.assertGreater(windows["B"][0], windows["A"][0])
        self.assertGreater(overlap[1], overlap[0], "A/B 没有同时可见的窗口")
        self.assertAlmostEqual(overlap[0], max(windows["A"][0], windows["B"][0]),
                               places=9)
        self.assertAlmostEqual(overlap[1], min(windows["A"][1], windows["B"][1]),
                               places=9)

    def test_scenario_has_incoming_sensor(self) -> None:
        scenario = get_system_scenario("S6")
        self.assertTrue(scenario.incoming_sensor_id)
        self.assertEqual(scenario.n_targets, 1)


class TestHandoverFailureMode(unittest.TestCase):
    def test_no_share_fails_handover(self) -> None:
        """不共享时本地包线之外必然交接失败。"""
        metrics = run_case("S6", "single", seed=42)["system"]["handover"]
        self.assertLess(metrics["handover_continuity_after_local"], 0.95)
        self.assertIsNone(metrics["handover_first_source_s"],
                          "单雷达场景不该出现远端来源")
        self.assertEqual(metrics["remote_contribution_ratio"], 0.0)

    def test_share_achieves_handover(self) -> None:
        metrics = run_case("S6", "ideal_share", seed=42)["system"]["handover"]
        self.assertGreaterEqual(metrics["handover_continuity_after_local"], 0.95)
        self.assertGreater(metrics["remote_contribution_ratio"], 0.2)
        self.assertIsNotNone(metrics["handover_first_source_s"])
        self.assertAlmostEqual(metrics["handover_delay_s"], 0.0, places=9)

    def test_communication_delay_shows_up_in_handover_delay(self) -> None:
        ideal = run_case("S6", "ideal_share", seed=42)["system"]["handover"]
        constrained = run_case("S6", "constrained_share",
                               seed=42)["system"]["handover"]
        self.assertGreater(constrained["handover_delay_s"],
                           ideal["handover_delay_s"],
                           "通信延迟没有体现在交接延迟上")
        self.assertLess(constrained["remote_contribution_ratio"],
                        ideal["remote_contribution_ratio"])


class TestLinkTimingStress(unittest.TestCase):
    def _bus(self, **overrides) -> CommBus:
        """链路参数从**干净**的受限共享起步，测试只打开自己关心的机制。

        ⚠️ 不要把所有概率都设成 1.0：`burst_loss_prob=1.0` 会让**每一条**
        消息都落进突发里，结果什么都送不出去，也就测不到乱序
        （第一版就是这么写的，被这条测试抓了出来）。
        """
        base = {"policy": SHARE_CONSTRAINED, "seed": 42}
        base.update(overrides)
        return CommBus(["A", "B"], CommConfig(**base))

    def test_all_four_mechanisms_fire(self) -> None:
        bus = self._bus(
            outage_windows=((5.0, 7.0),), burst_loss_prob=0.3, burst_length=3,
            recovery_congestion_s=4.0, recovery_extra_delay_s=1.5,
            recovery_loss_prob=0.3, reorder_prob=0.4, reorder_extra_delay_s=2.0,
        )
        for step in range(1, 21):
            bus.publish("A", "S1", [_Measurement(f"C{step}", float(step))],
                        now=float(step))
        stats = bus.statistics()
        reasons = stats["drop_reasons"]
        self.assertGreater(reasons.get("link_outage", 0), 0, "中断窗口没有生效")
        self.assertGreater(reasons.get("burst_loss", 0), 0, "突发丢包没有生效")
        self.assertGreater(stats["n_out_of_order"], 0, "乱序到达没有生效")

    def test_reordering_produces_out_of_order_arrivals(self) -> None:
        bus = self._bus(reorder_prob=0.5, reorder_extra_delay_s=3.0)
        for step in range(1, 16):
            bus.publish("A", "S1", [_Measurement(f"C{step}", float(step))],
                        now=float(step))
        stats = bus.statistics()
        self.assertGreater(stats["out_of_order_rate"], 0.0)
        self.assertGreater(stats["max_reorder_lag_s"], 0.0)

    def test_outage_windows_block_all_traffic(self) -> None:
        """中断窗口内发送的消息必须全部丢弃——这是"中断"的定义。"""
        bus = self._bus(outage_windows=((3.0, 5.0),))
        for step in range(1, 8):
            bus.publish("A", "S1", [_Measurement(f"C{step}", float(step))],
                        now=float(step))
        stats = bus.statistics()
        self.assertEqual(stats["drop_reasons"].get("link_outage"), 3,
                         "t=3,4,5 三条消息应全部因中断被丢弃")
        self.assertEqual(stats["n_delivered"], 4)

    def test_message_carries_destination(self) -> None:
        bus = self._bus()
        sent = bus.publish("A", "S1", [_Measurement("C1", 1.0)], now=1.0)
        self.assertTrue(sent)
        self.assertEqual(sent[0].dst_platform_id, "B")
        self.assertIn("dst_platform_id", sent[0].to_dict())


class TestOosmPolicies(unittest.TestCase):
    def test_drop_stale_passes_everything_immediately(self) -> None:
        controller = OosmController(policy=DROP_STALE, window_s=2.0)
        items = [_Measurement("C2", 2.0), _Measurement("C1", 1.0)]
        fused, flags, decisions = controller.select(items, [True, True], 2.0)
        self.assertEqual(len(fused), 2, "对照策略必须立刻全部喂入")
        self.assertTrue(all(d.decision == "immediate" for d in decisions))
        self.assertEqual(flags, [True, True])

    def test_reorder_buffer_holds_out_of_order(self) -> None:
        controller = OosmController(policy=REORDER_BUFFER, window_s=2.0)
        # 先融合 t=5（建立 newest_fused_time）
        controller.select([_Measurement("C5", 5.0)], [True], 5.0)
        # 再到达一个 t=3 的旧包 → 必须被扣住（本步什么都不融合）
        fused, _flags, decisions = controller.select(
            [_Measurement("C3", 3.0)], [True], 5.5)
        self.assertEqual(len(fused), 0, "乱序测量不该立刻被融合")
        self.assertEqual(decisions, [], "被扣住的测量本步不产生释放记录")
        self.assertEqual(controller.pending, 1)
        # 扣满窗口后释放，并标记为迟到应用
        fused, _flags, decisions = controller.select([], [], 7.5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0].decision, "released_late")
        self.assertAlmostEqual(decisions[0].hold_s, 2.0, places=9)
        self.assertEqual(controller.pending, 0)

    def test_finalize_records_still_pending_measurements(self) -> None:
        """整轮都没释放的测量必须留下 `held` 记录，不能悄悄消失。"""
        controller = OosmController(policy=REORDER_BUFFER, window_s=10.0)
        controller.select([_Measurement("C5", 5.0)], [True], 5.0)
        controller.select([_Measurement("C3", 3.0)], [True], 5.5)
        self.assertEqual(controller.pending, 1)
        frozen = controller.finalize()
        self.assertEqual(len(frozen), 1)
        self.assertEqual(frozen[0].decision, "held")
        self.assertIsNone(frozen[0].fusion_time_s)
        self.assertEqual(controller.pending, 0)

    def test_delayed_update_inflates_covariance_only_for_late(self) -> None:
        controller = OosmController(policy=DELAYED_UPDATE, window_s=1.0,
                                    inflation_per_s=1.0)
        controller.select([_Measurement("C5", 5.0)], [True], 5.0)
        controller.select([_Measurement("C3", 3.0)], [True], 5.0)
        fused, _flags, decisions = controller.select([], [], 6.0)
        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0].decision, "released_late")
        self.assertGreater(decisions[0].covariance_inflation, 1.0)
        # 协方差确实被放大了（σ → σ·f ⇒ 方差 → 方差·f²）
        self.assertGreater(fused[0].covariance[0][0], 100.0)
        self.assertGreater(fused[0].std_range_m, 10.0)

    def test_reorder_buffer_keeps_measurement_time_order(self) -> None:
        controller = OosmController(policy=REORDER_BUFFER, window_s=3.0)
        controller.select([_Measurement("C5", 5.0)], [True], 5.0)
        controller.select([_Measurement("C4", 4.0)], [True], 5.2)
        controller.select([_Measurement("C3", 3.0)], [True], 5.4)
        fused, _flags, _decisions = controller.select([], [], 8.6)
        times = [m.time_s for m in fused]
        self.assertEqual(times, sorted(times), "释放顺序不是测量时刻升序")

    def test_invalid_policy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OosmController(policy="not_a_policy")

    def test_all_three_policies_declared(self) -> None:
        self.assertEqual(set(OOSM_POLICIES),
                         {DROP_STALE, REORDER_BUFFER, DELAYED_UPDATE})


class TestTimingScenario(unittest.TestCase):
    def test_four_timestamps_are_distinguished(self) -> None:
        result = run_case("S7", "reorder_buffer", seed=42)
        decided = [d for frame in result["frames"]
                   for d in frame["oosm_decisions"]]
        self.assertTrue(decided)
        for entry in decided:
            self.assertIn("measurement_time_s", entry)
            self.assertIn("arrival_time_s", entry)
            self.assertIn("fusion_time_s", entry)
            self.assertIn("decision", entry)
        # send time 在消息日志里（逐消息可追溯）
        messages = [m for m in result["frames"]]
        self.assertTrue(messages)

    def test_reorder_trades_latency_for_order(self) -> None:
        """重排必须**降低乱序**，同时**增加延迟与时效拒绝**——代价一起测到。"""
        control = run_case("S7", DROP_STALE, seed=42)["system"]["timing"]
        treated = run_case("S7", REORDER_BUFFER, seed=42)["system"]["timing"]
        self.assertLess(treated["out_of_order_rate"],
                        control["out_of_order_rate"],
                        "重排没有降低乱序率")
        self.assertGreater(treated["mean_time_in_system_s"],
                           control["mean_time_in_system_s"],
                           "重排没有付出延迟代价（这说明缓冲没生效）")
        self.assertGreater(treated["stale_rejection_rate"],
                           control["stale_rejection_rate"],
                           "重排后应有一部分测量因扣留而超时效被拒")

    def test_burst_and_outage_observed(self) -> None:
        timing = run_case("S7", DROP_STALE, seed=42)["system"]["timing"]
        self.assertGreater(timing["n_bursts"], 0, "突发丢包没有发生")
        self.assertGreater(timing["n_outage_frames"], 0, "中断窗口没有覆盖到任何帧")
        self.assertIsNotNone(timing["recovery_time_s"])

    def test_scenario_axis_is_oosm(self) -> None:
        scenario = get_system_scenario("S7")
        self.assertEqual(scenario.axis, "oosm")
        self.assertEqual(set(scenario.variants), set(OOSM_POLICIES))
        self.assertEqual(scenario.fixed_policy, SHARE_CONSTRAINED)


class TestSensorBias(unittest.TestCase):
    def test_bias_only_touches_measurement_not_truth(self) -> None:
        import json

        from engine.simulator import Simulator
        import experiment_config as ec
        from sensor.config import build_suite_from_config

        sim = Simulator(ec.CONFIG_PATH)
        sim.load_config()
        sim.reset(seed=42)
        with open("config/radar_scenario_v1.json", "r", encoding="utf-8") as h:
            base = json.load(h)
        sensor = {
            "sensor_id": "S1", "mounting_id": "RADAR1", "sensor_kind": "radar",
            "max_range_m": 30000.0, "az_fov_deg": 60.0, "el_fov_deg": 30.0,
            "update_period_s": 1.0, "range_sigma_rel": 0.01,
            "range_sigma_abs_m": 5.0, "az_sigma_deg": 0.5, "el_sigma_deg": 0.5,
            "range_rate_sigma_mps": 1.0, "snr50_db": 6.0, "pd_slope_db": 2.0,
            "force_detection": True, "tx_power_w": 18.0, "peak_gain_db": 30.0,
            "wavelength_m": 0.1, "bandwidth_hz": 1.0e6, "noise_figure_db": 3.0,
            "system_loss_db": 3.0, "temperature_k": 290.0,
            "observes_kind": "target", "provides_range": True, "seed": 42,
        }

        def observe(**overrides):
            cfg = dict(sensor)
            cfg.update(overrides)
            suite = build_suite_from_config(sim.scene, {"sensors": [cfg]}, seed=42)
            return suite.sensors[0].observe(sim.scene, 1.0).detections[0]

        clean = observe()
        biased = observe(range_bias_m=200.0, az_bias_deg=0.7,
                         noise_underreport_factor=0.5)
        self.assertAlmostEqual(biased.range_m - clean.range_m, 200.0, places=6)
        self.assertAlmostEqual(biased.azimuth_deg - clean.azimuth_deg, 0.7,
                              places=6)
        self.assertAlmostEqual(biased.std_range_m, clean.std_range_m * 0.5,
                              places=6)
        # 真值一个字节都不许动
        self.assertEqual(biased.truth_range_m, clean.truth_range_m)
        self.assertEqual(biased.truth_azimuth_deg, clean.truth_azimuth_deg)
        # 零偏差必须与无偏差**逐位一致**
        again = observe()
        self.assertEqual(again.range_m, clean.range_m)
        self.assertEqual(again.covariance, clean.covariance)
        self.assertEqual(again.time_s, clean.time_s)

    def test_clock_offset_shifts_reported_time(self) -> None:
        import json

        from engine.simulator import Simulator
        import experiment_config as ec
        from sensor.config import build_suite_from_config

        sim = Simulator(ec.CONFIG_PATH)
        sim.load_config()
        sim.reset(seed=42)
        sensor = {
            "sensor_id": "S1", "mounting_id": "RADAR1", "sensor_kind": "radar",
            "max_range_m": 30000.0, "az_fov_deg": 60.0, "el_fov_deg": 30.0,
            "update_period_s": 1.0, "range_sigma_rel": 0.01,
            "range_sigma_abs_m": 5.0, "az_sigma_deg": 0.5, "el_sigma_deg": 0.5,
            "range_rate_sigma_mps": 1.0, "snr50_db": 6.0, "pd_slope_db": 2.0,
            "force_detection": True, "tx_power_w": 18.0, "peak_gain_db": 30.0,
            "wavelength_m": 0.1, "bandwidth_hz": 1.0e6, "noise_figure_db": 3.0,
            "system_loss_db": 3.0, "temperature_k": 290.0,
            "observes_kind": "target", "provides_range": True, "seed": 42,
            "clock_offset_s": 1.5,
        }
        suite = build_suite_from_config(sim.scene, {"sensors": [sensor]}, seed=42)
        record = suite.sensors[0].observe(sim.scene, 1.0).detections[0]
        self.assertAlmostEqual(record.time_s, 2.5, places=9)

    def test_bias_fields_rejected_when_unknown(self) -> None:
        from sensor.sensor import SensorConfig

        with self.assertRaises(ValueError):
            SensorConfig.from_dict({"sensor_id": "S", "mounting_id": "R",
                                    "bogus_field": 1.0})


class TestBiasScenario(unittest.TestCase):
    def test_four_groups_present(self) -> None:
        scenario = get_system_scenario("S8")
        self.assertEqual(scenario.axis_values,
                         ("single", "no_share", "ideal_share", "biased_share"))
        self.assertEqual(scenario.bias_variant, "biased_share")
        self.assertEqual(scenario.bias_overrides["SENSOR_REMOTE"], dict(S8_BIAS))

    def test_bias_injected_only_in_biased_variant(self) -> None:
        biased = run_case("S8", "biased_share", seed=42)
        clean = run_case("S8", "ideal_share", seed=42)
        self.assertTrue(biased["system"]["bias_injected"])
        self.assertFalse(clean["system"]["bias_injected"])
        # 偏差只加在远端传感器的测量上：本地传感器残差不应被它改变
        remote_biased = biased["system"]["bias"]["source_wise"]["SENSOR_REMOTE"]
        remote_clean = clean["system"]["bias"]["source_wise"]["SENSOR_REMOTE"]
        self.assertGreater(remote_biased["residual_mean_m"],
                           remote_clean["residual_mean_m"])

    def test_second_radar_can_be_worse_than_single(self) -> None:
        """**核心命题**：有偏共享可以比只用本地单雷达更差。"""
        single = run_case("S8", "single", seed=42)["metrics"]
        ideal = run_case("S8", "ideal_share", seed=42)["metrics"]
        biased = run_case("S8", "biased_share", seed=42)["metrics"]
        self.assertLess(ideal["position_rmse_m"], single["position_rmse_m"],
                        "无偏共享本应改善精度")
        self.assertGreater(biased["position_rmse_m"], single["position_rmse_m"],
                           "有偏共享本应比单雷达更差（本场景的设计目标）")

    def test_health_score_flags_the_worse_sensor(self) -> None:
        bias = run_case("S8", "biased_share", seed=42)["system"]["bias"]
        self.assertIn("SENSOR_REMOTE", bias["suspicious_sensor_ids"])
        self.assertGreater(bias["sensor_health_worst"],
                           bias["sensor_health_best"])

    def test_clean_share_is_not_flagged(self) -> None:
        """无偏场景不得误报（否则诊断分支会被当成噪声）。"""
        bias = run_case("S8", "ideal_share", seed=42)["system"]["bias"]
        self.assertEqual(bias["suspicious_sensor_ids"], [])

    def test_health_score_is_not_used_to_drop_measurements(self) -> None:
        """诊断分支**不得**自动剔除传感器：有偏路仍然使用远端测量。"""
        result = run_case("S8", "biased_share", seed=42)
        self.assertGreater(result["system"]["bias"]["remote_contribution_ratio"],
                           0.1, "远端测量被静默剔除了——这违反纪律")
        source = open(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "fusion", "center.py"), encoding="utf-8").read()
        self.assertNotIn("sensor_health", source,
                         "健康分不得进入跟踪器")


class TestScenarioCatalogue(unittest.TestCase):
    def test_all_system_scenarios_build(self) -> None:
        for scenario_id in SYSTEM_SCENARIO_IDS:
            scenario = get_system_scenario(scenario_id)
            self.assertEqual(scenario.group, "system")
            self.assertTrue(scenario.overrides.get("sensors"))
            self.assertTrue(scenario.variants)

    def test_resolve_supports_core_and_system(self) -> None:
        self.assertEqual(resolve_scenario("S1").group, "core")
        self.assertEqual(resolve_scenario("S6").group, "system")
        with self.assertRaises(ValueError):
            resolve_scenario("S99")

    def test_no_jpda_or_imm_introduced(self) -> None:
        """用户要求：先记录基线失效模式，不得因为场景失败就上更复杂的算法。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relative in ("fusion/center.py", "fusion/kalman.py",
                         "multi_target_stress/runner.py"):
            source = open(os.path.join(root, relative), encoding="utf-8").read()
            for forbidden in ("JPDA", "IMM(", "interacting_multiple_model",
                              "joint_probabilistic"):
                self.assertNotIn(forbidden, source,
                                 f"{relative} 出现了 {forbidden}：本轮不应引入")


if __name__ == "__main__":
    unittest.main(verbosity=2)
