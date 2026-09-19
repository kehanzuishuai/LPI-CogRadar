"""多目标压力测试与关联层审计测试（v4.5 P2）。

钉住四件事：
1. **关联层可审计**：每条测量的全部候选航迹、门限距离、拒绝原因、
   最终选择都留痕；歧义（≥2 条过门限）能被识别；
2. **跟踪器的 miss/coasting/删除真的生效**：这是 v4.5 修掉的一个严重 bug
   （`predict_to` 覆盖 `last_update_time` 导致 miss 分支永远走不到，
   `misses` 恒为 0、航迹永不删除）；
3. **多目标指标口径正确**：一对一分配、重复 vs 假航迹的区分、
   交叉时不得把"另一个目标的航迹"误判成重复航迹；
4. **跟踪器与评测的真值隔离**：`fusion/` 不读 `truth_id`，
   共享测量对象的 `truth_id` 恒为 None。

不需要 torch。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.geometry import Vec3  # noqa: E402
from fusion.lifecycle import (  # noqa: E402
    REJECT_AZIMUTH,
    REJECT_MAHALANOBIS,
    AssociationCandidate,
    LifecycleLog,
)
from multi_target_stress.metrics import (  # noqa: E402
    ASSOC_GATE_M,
    MultiTargetMetrics,
    assign_tracks_to_truth,
)
from multi_target_stress.runner import run_case  # noqa: E402
from multi_target_stress.scenarios import (  # noqa: E402
    SCENARIO_IDS,
    _radius_for_occlusion_seconds,
    _shadow_half_width_m,
    get_scenario,
)


class TestScenarioGeometry(unittest.TestCase):
    def test_all_scenarios_build(self) -> None:
        for scenario_id in SCENARIO_IDS:
            scenario = get_scenario(scenario_id)
            self.assertTrue(scenario.overrides.get("targets"))
            self.assertTrue(scenario.overrides.get("sensors"))
            self.assertGreaterEqual(len(scenario.overrides["radars"]), 2,
                                    "四路对照需要远端雷达")

    def test_physics_copied_from_base_config_not_invented(self) -> None:
        """场景只改几何：雷达/目标的物理参数必须与基础配置完全一致。"""
        import json

        with open("config/radar_scenario_v1.json", "r", encoding="utf-8") as handle:
            base = json.load(handle)
        scenario = get_scenario("S1")
        radar = scenario.overrides["radars"][0]
        for key in ("tx_power_w", "peak_gain_db", "sidelobe_gain_db", "wavelength_m",
                    "bandwidth_hz", "noise_figure_db", "system_loss_db",
                    "temperature_k", "snr50_db", "pd_slope_db", "required_pd",
                    "energy_budget_j"):
            self.assertEqual(radar[key], base["radar"][key],
                             f"雷达物理参数 {key} 被改动")
        target = scenario.overrides["targets"][0]
        self.assertEqual(target["rcs_m2"], base["targets"][0]["rcs_m2"])

    def test_shadow_half_width_is_exact_not_far_field(self) -> None:
        """遮挡阴影半宽必须用精确解，远场锥近似在近距球上会差一倍以上。"""
        radius = _radius_for_occlusion_seconds(4.0)
        half_width = _shadow_half_width_m(radius)
        # 反解出来的半径与半宽必须自洽（±1e-6）
        self.assertAlmostEqual(half_width, 4.0 * 150.0 / 2.0, places=6)
        # 远场近似给的是 301 m 量级，精确解是 300 m 附近（此处一致），
        # 但半径 400 时两者分别是 710 m 与 454 m —— 断言这个差异存在
        far_field = (7000.0 - 4000.0) * (400.0 / (4000.0 ** 2 - 400.0 ** 2) ** 0.5)
        self.assertLess(far_field, _shadow_half_width_m(400.0) * 0.7)

    def test_occlusion_window_matches_radius(self) -> None:
        short = get_scenario("S3").occlusion_window_s
        long_ = get_scenario("S3L").occlusion_window_s
        # 配置里的半径保留 3 位小数，因此窗口会有 ~1e-5 级的偏差
        self.assertAlmostEqual(short[1] - short[0], 4.0, places=4)
        self.assertAlmostEqual(long_[1] - long_[0], 9.4, places=4)
        # 短遮挡必须短于 drop_after_misses=5，长遮挡必须长于它
        self.assertLess(short[1] - short[0], 5)
        self.assertGreater(long_[1] - long_[0], 5)


class TestOneToOneAssignment(unittest.TestCase):
    def test_greedy_is_one_to_one(self) -> None:
        tracks = [("T1", Vec3(0.0, 0.0, 0.0)), ("T2", Vec3(100.0, 0.0, 0.0))]
        truth = [("A", Vec3(10.0, 0.0, 0.0))]
        t2tr, tr2t = assign_tracks_to_truth(tracks, truth, gate_m=1000.0)
        self.assertEqual(len(t2tr), 1, "一个目标只能分配给一条航迹")
        self.assertEqual(len(tr2t), 1)
        self.assertEqual(t2tr["T1"], "A")

    def test_gate_excludes_far_pairs(self) -> None:
        tracks = [("T1", Vec3(0.0, 0.0, 0.0))]
        truth = [("A", Vec3(5000.0, 0.0, 0.0))]
        t2tr, _ = assign_tracks_to_truth(tracks, truth, gate_m=1000.0)
        self.assertEqual(t2tr, {})

    def test_assignment_prefers_cheaper_pair(self) -> None:
        tracks = [("T1", Vec3(0.0, 0.0, 0.0)), ("T2", Vec3(0.0, 200.0, 0.0))]
        truth = [("A", Vec3(0.0, 150.0, 0.0)), ("B", Vec3(0.0, 0.0, 0.0))]
        t2tr, _ = assign_tracks_to_truth(tracks, truth, gate_m=1000.0)
        self.assertEqual(t2tr["T1"], "B")
        self.assertEqual(t2tr["T2"], "A")


def _frame(time_s, tracks, truth, associations=(), measurement_truth=None):
    return {
        "time_s": time_s,
        "tracks": tracks,
        "truth": truth,
        "associations": list(associations),
        "measurement_truth": dict(measurement_truth or {}),
    }


class TestMultiTargetMetrics(unittest.TestCase):
    def test_crossing_does_not_count_other_targets_track_as_duplicate(self) -> None:
        """**回归**：交叉时两目标互相靠近，不得把对方的航迹判成重复航迹。

        第一版按"目标附近有几条航迹"数重复，S1 报出 34 次重复，
        全是假阳性；正确口径必须用一对一分配的结果。
        """
        metrics = MultiTargetMetrics()
        for step in range(5):
            metrics.add_frame(_frame(
                step,
                tracks=[("T1", Vec3(7000.0, -10.0, 0.0), Vec3(), "confirmed"),
                        ("T2", Vec3(7000.0, 10.0, 0.0), Vec3(), "confirmed")],
                truth=[("A", Vec3(7000.0, -15.0, 0.0), Vec3(), True),
                       ("B", Vec3(7000.0, 15.0, 0.0), Vec3(), True)],
            ))
        result = metrics.result()
        self.assertEqual(result["duplicate_track_count"], 0)
        self.assertEqual(result["false_track_rate"], 0.0)
        self.assertEqual(result["track_purity"], 1.0)

    def test_real_duplicate_track_is_counted(self) -> None:
        """真正的第三条第航迹追同一目标时必须记成重复航迹。"""
        metrics = MultiTargetMetrics()
        metrics.add_frame(_frame(
            0,
            tracks=[("T1", Vec3(7000.0, 0.0, 0.0), Vec3(), "confirmed"),
                    ("T2", Vec3(7000.0, 60.0, 0.0), Vec3(), "confirmed")],
            truth=[("A", Vec3(7000.0, 0.0, 0.0), Vec3(), True)],
        ))
        result = metrics.result()
        self.assertEqual(result["duplicate_track_count"], 1)
        self.assertEqual(result["n_false_track_frames"], 0)

    def test_ghost_track_is_counted_as_false(self) -> None:
        metrics = MultiTargetMetrics()
        metrics.add_frame(_frame(
            0,
            tracks=[("T1", Vec3(7000.0, 0.0, 0.0), Vec3(), "confirmed"),
                    ("T9", Vec3(40000.0, 0.0, 0.0), Vec3(), "confirmed")],
            truth=[("A", Vec3(7000.0, 0.0, 0.0), Vec3(), True)],
        ))
        result = metrics.result()
        self.assertEqual(result["duplicate_track_count"], 0)
        self.assertEqual(result["n_false_track_frames"], 1)
        self.assertAlmostEqual(result["false_track_rate"], 0.5)

    def test_id_switch_counted_without_gap(self) -> None:
        metrics = MultiTargetMetrics()
        for step, track_id in enumerate(("T1", "T1", "T2")):
            metrics.add_frame(_frame(
                step,
                tracks=[(track_id, Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed")],
                truth=[("A", Vec3(0.0, 0.0, 0.0), Vec3(), True)],
            ))
        result = metrics.result()
        self.assertEqual(result["id_switch_count"], 1)
        self.assertEqual(result["track_fragmentation_count"], 0)
        self.assertAlmostEqual(result["continuity_rate"], 0.5)

    def test_fragmentation_counted_across_gap(self) -> None:
        metrics = MultiTargetMetrics()
        sequence = ["T1", None, None, "T2"]
        for step, track_id in enumerate(sequence):
            tracks = ([] if track_id is None else
                      [(track_id, Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed")])
            metrics.add_frame(_frame(
                step, tracks=tracks,
                truth=[("A", Vec3(0.0, 0.0, 0.0), Vec3(), True)],
            ))
        result = metrics.result()
        self.assertEqual(result["track_fragmentation_count"], 1)
        self.assertEqual(result["id_switch_count"], 0)

    def test_same_track_after_gap_is_not_fragmentation(self) -> None:
        """遮挡后靠外推保住同一 Track ID —— 这正是 S3 想验证的行为。"""
        metrics = MultiTargetMetrics()
        for step, track_id in enumerate(["T1", None, None, "T1"]):
            tracks = ([] if track_id is None else
                      [(track_id, Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed")])
            metrics.add_frame(_frame(
                step, tracks=tracks,
                truth=[("A", Vec3(0.0, 0.0, 0.0), Vec3(), True)],
            ))
        result = metrics.result()
        self.assertEqual(result["track_fragmentation_count"], 0)
        self.assertEqual(result["id_switch_count"], 0)

    def test_missed_and_completeness_relationship(self) -> None:
        """两个口径在覆盖不均时必须分开（macro vs 合并口径）。"""
        metrics = MultiTargetMetrics()
        # 目标 A 只活跃 1 帧且被跟上；目标 B 活跃 3 帧只跟上 1 帧。
        # B 必须放在关联门限之外，否则 A 的航迹会被分配给它（口径就测不出来了）。
        far = Vec3(0.0, 5000.0, 0.0)
        metrics.add_frame(_frame(
            0,
            tracks=[("T1", Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed"),
                    ("T2", far, Vec3(), "confirmed")],
            truth=[("A", Vec3(0.0, 0.0, 0.0), Vec3(), True),
                   ("B", far, Vec3(), True)],
        ))
        for step, track_id in ((1, "T1"), (2, "T1")):
            metrics.add_frame(_frame(
                step,
                tracks=[(track_id, Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed")],
                truth=[("B", far, Vec3(), True)],
            ))
        result = metrics.result()
        # 合并口径：(0 + 2) / (1 + 3) = 0.5
        self.assertAlmostEqual(result["missed_track_rate"], 0.5)
        # 逐目标平均：(1/1 + 1/3) / 2 = 0.6667
        self.assertAlmostEqual(result["track_completeness"], 2.0 / 3.0)
        self.assertNotAlmostEqual(result["missed_track_rate"],
                                  1.0 - result["track_completeness"])

    def test_association_accuracy_uses_measurement_truth(self) -> None:
        metrics = MultiTargetMetrics()
        metrics.add_frame(_frame(
            0,
            tracks=[("T1", Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed")],
            truth=[("A", Vec3(0.0, 0.0, 0.0), Vec3(), True)],
            associations=[("S1", "C1", "T1", False),   # 正确
                          ("S1", "C2", "T1", False),   # C2 属于别的目标 → 错
                          ("S1", "FA1", "T1", False)],  # 虚警进了真实航迹 → 错
            measurement_truth={("S1", "C1"): "A",
                               ("S1", "C2"): "B",
                               ("S1", "FA1"): None},
        ))
        result = metrics.result()
        self.assertEqual(result["n_associations"], 3)
        self.assertEqual(result["n_associations_correct"], 1)
        self.assertAlmostEqual(result["association_accuracy"], 1 / 3)
        self.assertEqual(result["n_false_alarms_into_real_track"], 1)

    def test_metrics_do_not_require_truth_keys(self) -> None:
        """指标里绝不能出现真值字段以外的东西被当真值用。"""
        metrics = MultiTargetMetrics()
        metrics.add_frame(_frame(
            0, tracks=[], truth=[],
        ))
        result = metrics.result()
        self.assertEqual(result["n_track_frames"], 0)
        self.assertEqual(result["position_rmse_m"], 0.0)


class TestAssociationAudit(unittest.TestCase):
    def test_candidates_recorded_with_reject_reasons(self) -> None:
        """关联审计必须留下"比较过谁、距离多少、为什么没选"。"""
        result = run_case("S1", "single", seed=42)
        lifecycle = result["lifecycle"]
        audited = [t for t in lifecycle.traces if t.association_candidate_tracks]
        self.assertTrue(audited, "没有任何测量留下关联审计")
        chosen = [t for t in audited if t.chosen_track_id]
        self.assertTrue(chosen, "没有任何测量记录最终选择")
        for trace in chosen:
            flags = [c for c in trace.association_candidate_tracks if c.chosen]
            self.assertLessEqual(len(flags), 1, "最终选择只能有一个")
            # 若最终选中的航迹当时已存在，它必须在候选列表里并且被标记；
            # 若选中的是**本帧新建**的航迹，它当然不在候选列表里（当时还不存在）。
            existing = [c for c in trace.association_candidate_tracks
                        if c.track_id == trace.chosen_track_id]
            if existing:
                self.assertTrue(existing[0].chosen,
                                "选中了已存在的航迹却没标记 chosen")

    def test_reject_reason_values_are_from_fixed_set(self) -> None:
        result = run_case("S1", "single", seed=42)
        allowed = {"", REJECT_MAHALANOBIS, REJECT_AZIMUTH, "elevation_gate",
                   "no_track", "track_dropped"}
        for trace in result["lifecycle"].traces:
            for candidate in trace.association_candidate_tracks:
                self.assertIn(candidate.reject_reason, allowed)

    def test_ambiguity_detected_at_crossing(self) -> None:
        """交叉场景必须真的产生关联歧义，否则这个"压力"场景没有检验力。"""
        result = run_case("S1", "single", seed=42)
        self.assertGreater(result["metrics"]["ambiguous_rate"], 0.05)
        self.assertGreater(result["center"].stats["ambiguous"], 0)

    def test_association_csv_has_one_row_per_pair(self) -> None:
        """关联审计 CSV：一行 = 一个 (测量, 候选航迹) 对。"""
        lifecycle = LifecycleLog(True)
        trace = lifecycle.open_trace(candidate_id="C1", sensor_id="S1")
        lifecycle.record_association(trace, [
            AssociationCandidate("T1", 1.5, 9.0, 0.4, 0.1, passed=True),
            AssociationCandidate("T2", 88.0, 9.0, 12.0, 3.0, passed=False,
                                 reject_reason=REJECT_MAHALANOBIS),
        ], chosen_track_id="T1")
        rows = trace.association_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["chosen_track_id"], "T1")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[1]["gate_reject_reason"], REJECT_MAHALANOBIS)
        self.assertFalse(rows[1]["passed"])
        self.assertEqual(trace.n_tracks_in_gate, 1)
        self.assertFalse(trace.ambiguous)

    def test_ambiguous_flag_set_when_two_pass(self) -> None:
        lifecycle = LifecycleLog(True)
        trace = lifecycle.open_trace(candidate_id="C1", sensor_id="S1")
        lifecycle.record_association(trace, [
            AssociationCandidate("T1", 1.5, 9.0, passed=True),
            AssociationCandidate("T2", 3.0, 9.0, passed=True),
        ], chosen_track_id="T1")
        self.assertTrue(trace.ambiguous)
        self.assertEqual(trace.n_tracks_in_gate, 2)


class _Measurement:
    """最小测量对象：字段与 `MeasurementRecord` 对齐（只放融合中心用到的）。"""

    def __init__(self, sensor_id: str, candidate_id: str, time_s: float,
                 range_m: float, sensor_kind: str = "radar") -> None:
        self.sensor_id = sensor_id
        self.candidate_id = candidate_id
        self.sensor_kind = sensor_kind
        self.time_s = time_s
        self.range_m = range_m
        # 方位 0° = +y，配合原点处的传感器 → 目标在 (0, range, 0)
        self.azimuth_deg = 0.0
        self.elevation_deg = 0.0
        self.std_range_m = 10.0
        self.std_az_deg = 0.5
        self.std_el_deg = 0.5
        self.range_rate_mps = 0.0
        self.platform_id = ""
        self.msg_id = ""


class TestTrackerMissCoastDropUnit(unittest.TestCase):
    """对 `FusionCenter` 的直接单测：miss → coasting → 删除必须真的发生。

    ⚠️ 这是 v4.5 修的严重 bug 的**最强回归**：
    `predict_to(now)` 把 `last_update_time` 全部写成 `now`，
    而旧代码正是用 `abs(last_update_time - now) < 1e-12` 判"本帧有没有测量"，
    于是 miss 分支永远走不到——`misses` 恒为 0、航迹永不 coasting、永不删除。
    从场景指标反推看不出来（幽灵航迹只会让 RMSE 变大），必须直接测。
    """

    def _center(self):
        from fusion import FusionCenter, FusionConfig

        center = FusionCenter("P", FusionConfig(), own_sensor_ids={"S1"})
        sensor_positions = {"S1": Vec3(0.0, 0.0, 0.0)}
        return center, sensor_positions

    def test_miss_increments_then_coasts_then_drops(self) -> None:
        center, positions = self._center()
        # 两帧有测量 → 航迹确认（confirm_hits=2）
        center.update([_Measurement("S1", "C1", 0.0, 1000.0)], 0.0, positions)
        center.update([_Measurement("S1", "C2", 1.0, 1100.0)], 1.0, positions)
        self.assertEqual(len(center.tracks), 1)
        track = center.tracks[0]
        self.assertEqual(track.misses, 0)

        # 此后连续无测量：misses 必须逐帧增长
        for step, expected in enumerate((1, 2, 3, 4), start=2):
            center.update([], float(step), positions)
            self.assertEqual(track.misses, expected,
                             f"第 {step} 帧 misses 应为 {expected}")
            self.assertIn(track.status, ("coasting", "confirmed"))
            if expected >= 1:
                self.assertEqual(track.status, "coasting",
                                 "coast_after_misses=1，misses≥1 就该进入 coasting")
            self.assertEqual(len(center.tracks), 1, "misses<5 时不应删除")

        # 第 5 次 miss 达到 drop_after_misses → 删除
        center.update([], 6.0, positions)
        self.assertEqual(len(center.tracks), 0, "misses=5 必须删除航迹")
        self.assertEqual(center.stats["dropped"], 1)
        self.assertEqual(center.retired_track_ids, ["P-T1"])

    def test_ghost_track_stops_appearing_after_drop(self) -> None:
        """删除的那一刻起，航迹就不得再出现在快照里。

        coast_after_misses=1、drop_after_misses=5：
        miss 1~4 帧仍在（coasting），miss 5 帧被删除。
        """
        center, positions = self._center()
        center.update([_Measurement("S1", "C1", 0.0, 1000.0)], 0.0, positions)
        center.update([_Measurement("S1", "C2", 1.0, 1100.0)], 1.0, positions)
        for step in range(2, 6):                      # misses 1..4
            snapshot = center.update([], float(step), positions)
            self.assertEqual([t.track_id for t in snapshot.tracks], ["P-T1"],
                             f"第 {step} 帧 misses={step - 1}，应仍在外推")
        for step in range(6, 9):                      # misses 5 起删除
            snapshot = center.update([], float(step), positions)
            self.assertEqual([t.track_id for t in snapshot.tracks], [],
                             f"第 {step} 帧仍输出已删除的幽灵航迹")

    def test_updated_track_is_not_counted_as_miss(self) -> None:
        """本帧拿到测量的航迹不能被记成 miss（修 bug 不能修反）。"""
        center, positions = self._center()
        center.update([_Measurement("S1", "C1", 0.0, 1000.0)], 0.0, positions)
        center.update([_Measurement("S1", "C2", 1.0, 1100.0)], 1.0, positions)
        track = center.tracks[0]
        # 匀速 100 m/s，让卡尔曼的预测与量测一致
        for step in range(2, 6):
            center.update([_Measurement("S1", f"C{step}", float(step),
                                        1000.0 + 100.0 * step)], float(step),
                          positions)
            self.assertEqual(track.misses, 0, "有测量却记了 miss")
            self.assertEqual(track.status, "confirmed")
        self.assertEqual(len(center.tracks), 1)

    def test_long_occlusion_drops_the_track(self) -> None:
        """遮挡 9.4 s（> drop_after_misses=5）必须**删除**航迹。"""
        result = run_case("S3L", "single", seed=42)
        center = result["center"]
        self.assertGreaterEqual(center.stats["dropped"], 1,
                                "长遮挡后航迹从未被删除 —— miss 分支又失效了")
        self.assertTrue(center.retired_track_ids)

    def test_short_occlusion_keeps_track_identity(self) -> None:
        """遮挡 4 s（< drop_after_misses=5）必须**保住**同一 Track ID。"""
        result = run_case("S3", "single", seed=42)
        metrics = result["metrics"]
        self.assertEqual(metrics["track_fragmentation_count"], 0)
        self.assertEqual(result["center"].stats["dropped"], 0)

    def test_coasting_status_is_observable_in_frames(self) -> None:
        """被挡住的航迹必须以 coasting 状态出现（这是"外推维持"的可观测证据）。"""
        result = run_case("S3L", "single", seed=42)
        statuses = {t[3] for frame in result["frames"] for t in frame["tracks"]}
        self.assertIn("coasting", statuses,
                      "遮挡期间没有任何航迹进入 coasting —— "
                      "miss 分支可能又失效了")

    def test_no_track_survives_whole_run_without_measurements(self) -> None:
        """任何一条航迹都不该在整轮几乎没有观测的情况下一直存活。"""
        result = run_case("S3L", "single", seed=42)
        for track in result["center"].tracks:
            self.assertLess(track.misses, 5,
                            f"{track.track_id} 的 misses={track.misses}，"
                            "本应在达到阈值时被删除")


class TestTruthIsolation(unittest.TestCase):
    def test_shared_measurement_never_carries_truth(self) -> None:
        from multi_target_stress.runner import SharedMeasurement

        measurement = SharedMeasurement({"sensor_id": "S", "candidate_id": "C",
                                        "truth_id": "TGT_A", "range_m": 100.0})
        self.assertIsNone(measurement.truth_id,
                          "共享测量对象不得携带真值 ID")

    def test_fusion_module_does_not_read_truth_fields(self) -> None:
        """跟踪器不得**真的读到**真值字段（AST 级，不看文档字符串）。"""
        from validation.checks import check_truth_isolation

        results = check_truth_isolation()
        for result in results:
            self.assertTrue(result.passed,
                            f"{result.check_id} 失败：{result.detail}")


class TestFourScenarioOutcomesAreReproducible(unittest.TestCase):
    """四类场景的关键结论必须可复现（防将来改动把结论悄悄改掉）。"""

    def test_s1_crossing_causes_id_switch(self) -> None:
        metrics = run_case("S1", "single", seed=42)["metrics"]
        self.assertGreaterEqual(metrics["id_switch_count"], 1,
                                "交叉场景应当产生 ID 换号；若变成 0，"
                                "说明场景失去了检验力或指标退化了")

    def test_s1_share_reduces_id_switch(self) -> None:
        single = run_case("S1", "single", seed=42)["metrics"]
        shared = run_case("S1", "ideal_share", seed=42)["metrics"]
        self.assertLess(shared["id_switch_count"], single["id_switch_count"],
                        "S1 上理想共享应当减少 ID 换号（远端消歧）")

    def test_s3_long_occlusion_share_prevents_fragmentation(self) -> None:
        single = run_case("S3L", "single", seed=42)["metrics"]
        shared = run_case("S3L", "ideal_share", seed=42)["metrics"]
        self.assertGreater(single["track_fragmentation_count"], 0)
        self.assertEqual(shared["track_fragmentation_count"], 0,
                         "S3L 上远端补盲应当避免航迹碎裂")

    def test_s4_false_alarms_create_false_tracks(self) -> None:
        metrics = run_case("S4", "single", seed=42)["metrics"]
        self.assertGreater(metrics["false_track_rate"], 0.1,
                           "目标附近虚警应当造出假航迹")
        self.assertGreater(metrics["max_tracks_in_a_frame"], 2,
                           "虚警应当使单帧航迹数超过真实目标数")

    def test_s2_dense_formation_is_ambiguous(self) -> None:
        metrics = run_case("S2", "single", seed=42)["metrics"]
        self.assertGreater(metrics["ambiguous_rate"], 0.5,
                           "编队间距约 2.1σ，关联歧义率应当很高")

    def test_no_share_differs_from_share_only_when_a_remote_exists(self) -> None:
        """`no_share` 与 `single` 在指标上可以相同（远端不共享≈不存在），
        但共享路必须真的建立了远端传感器。"""
        single = run_case("S2", "single", seed=42)["metrics"]
        no_share = run_case("S2", "no_share", seed=42)["metrics"]
        shared = run_case("S2", "ideal_share", seed=42)["metrics"]
        self.assertEqual(single["n_remote_sensors"], 0)
        self.assertEqual(no_share["n_remote_sensors"], 1)
        self.assertEqual(shared["n_remote_sensors"], 1)
        self.assertNotEqual(shared["n_associations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
