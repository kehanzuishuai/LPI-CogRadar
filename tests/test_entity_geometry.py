"""v4.1 多平台几何升级的单元测试。

覆盖用户明确点名的五类一致性：
1. **坐标变换**——ENU ↔ 球坐标 ↔ 机体坐标的往返与正交性；
2. **实体间距离对称性**——逐位（不是近似）对称；
3. **运动更新**——与闭式解逐位一致，且不污染其它维度；
4. **时间同步**——不同时刻的位姿之间不允许求几何量；
5. **多实体索引**——跨类型唯一、按 ID 可取、重复必须报错。

另有一组**向后兼容**断言：新的几何路径必须与升级前的距离实现逐位一致。
这不是"差不多就行"——最后一位变了，Pd/能耗/奖励都会跟着漂，
旧实验就不再可复现。

本文件属于仿真层测试，**不需要 torch**，base 环境即可运行。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.geometry import (  # noqa: E402
    Attitude,
    GeometryError,
    Pose,
    Spherical,
    TimeSyncError,
    Vec3,
    angular_separation_deg,
    azimuth_elevation,
    distance_matrix,
    enu_range,
    enu_range_2d,
    enu_to_spherical,
    los_unit,
    normalize_angle_deg,
    pairwise_relations,
    radial_velocity_mps,
    relation,
    relative_bearing_deg,
    spherical_to_enu,
)
from engine.scene import Scene, SceneError  # noqa: E402
from engine.simulator import Simulator  # noqa: E402
from models import EnemyInterceptor, Jammer, Radar, Target  # noqa: E402
from models.entity import KIND_RADAR, KIND_TARGET  # noqa: E402

LEGACY_CONFIG = "config/radar_scenario_v1.json"
MULTI_CONFIG = "config/multi_platform_scenario.json"


# ----------------------------------------------------------------------
# 1. 坐标变换
# ----------------------------------------------------------------------

class TestCoordinateTransforms(unittest.TestCase):
    def test_spherical_roundtrip(self) -> None:
        """ENU → 球坐标 → ENU 必须回到原点（浮点意义下）。"""
        samples = [
            Vec3(1000.0, 0.0, 0.0),
            Vec3(0.0, 1000.0, 0.0),
            Vec3(0.0, 0.0, 1000.0),
            Vec3(-3000.0, 4000.0, 0.0),
            Vec3(6000.0, 0.0, 3000.0),
            Vec3(-8000.0, -6000.0, 2600.0),
            Vec3(12345.6, -7890.1, 234.5),
        ]
        for vector in samples:
            with self.subTest(vector=vector):
                back = spherical_to_enu(enu_to_spherical(vector))
                self.assertTrue(vector.is_close(back, tol=1e-6), f"{vector} -> {back}")

    def test_azimuth_reference_directions(self) -> None:
        """方位角以 +y（正北）为 0°，顺时针为正：东 = +90°，南 = ±180°。"""
        self.assertAlmostEqual(enu_to_spherical(Vec3(0.0, 100.0, 0.0)).azimuth_deg, 0.0)
        self.assertAlmostEqual(enu_to_spherical(Vec3(100.0, 0.0, 0.0)).azimuth_deg, 90.0)
        self.assertAlmostEqual(abs(enu_to_spherical(Vec3(0.0, -100.0, 0.0)).azimuth_deg), 180.0)
        self.assertAlmostEqual(enu_to_spherical(Vec3(-100.0, 0.0, 0.0)).azimuth_deg, -90.0)

    def test_elevation_sign(self) -> None:
        self.assertAlmostEqual(enu_to_spherical(Vec3(0.0, 100.0, 100.0)).elevation_deg, 45.0)
        self.assertAlmostEqual(enu_to_spherical(Vec3(0.0, 100.0, -100.0)).elevation_deg, -45.0)

    def test_spherical_range_matches_distance(self) -> None:
        vector = Vec3(3000.0, 4000.0, 1200.0)
        self.assertAlmostEqual(enu_to_spherical(vector).range_m, vector.norm(), places=12)

    def test_body_basis_is_orthonormal(self) -> None:
        for heading in (0.0, 37.5, 90.0, 180.0, -120.0, 359.0):
            for pitch in (-30.0, 0.0, 15.0):
                for roll in (-20.0, 0.0, 25.0):
                    with self.subTest(h=heading, p=pitch, r=roll):
                        attitude = Attitude(heading, pitch, roll)
                        forward, right, up = attitude.body_basis()
                        for vec in (forward, right, up):
                            self.assertAlmostEqual(vec.norm(), 1.0, places=12)
                        self.assertAlmostEqual(forward.dot(right), 0.0, places=12)
                        self.assertAlmostEqual(forward.dot(up), 0.0, places=12)
                        self.assertAlmostEqual(right.dot(up), 0.0, places=12)

    def test_heading_definition(self) -> None:
        """heading=0 时机头指北；heading=90 时指东。"""
        north = Attitude(0.0, 0.0, 0.0).body_basis()[0]
        east = Attitude(90.0, 0.0, 0.0).body_basis()[0]
        self.assertTrue(north.is_close(Vec3(0.0, 1.0, 0.0), tol=1e-12))
        self.assertTrue(east.is_close(Vec3(1.0, 0.0, 0.0), tol=1e-12))

    def test_body_enu_roundtrip(self) -> None:
        attitude = Attitude(35.0, -12.0, 8.0)
        samples = [
            Vec3(100.0, 0.0, 0.0),
            Vec3(0.0, -50.0, 0.0),
            Vec3(0.0, 0.0, 25.0),
            Vec3(1234.5, -678.9, 12.3),
        ]
        for vector in samples:
            with self.subTest(vector=vector):
                back = attitude.body_to_enu(attitude.enu_to_body(vector))
                self.assertTrue(vector.is_close(back, tol=1e-9), f"{vector} -> {back}")

    def test_bearing_zero_when_target_ahead(self) -> None:
        """目标正好在机头方向时，机体方位角为 0。"""
        attitude = Attitude(50.0, 0.0, 0.0)
        forward = attitude.body_basis()[0]
        pose = Pose(position=Vec3(0.0, 0.0, 0.0), attitude=attitude)
        target = Pose(position=forward * 1000.0, attitude=attitude)
        rel = relation("A", pose, "B", target)
        self.assertAlmostEqual(rel.bearing_deg, 0.0, places=9)

    def test_normalize_angle(self) -> None:
        self.assertAlmostEqual(normalize_angle_deg(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle_deg(190.0), -170.0)
        self.assertAlmostEqual(normalize_angle_deg(-190.0), 170.0)
        self.assertAlmostEqual(normalize_angle_deg(360.0), 0.0)
        self.assertAlmostEqual(relative_bearing_deg(90.0, 45.0), 45.0)
        self.assertAlmostEqual(relative_bearing_deg(0.0, 90.0), -90.0)


# ----------------------------------------------------------------------
# 2. 距离对称性
# ----------------------------------------------------------------------

class TestDistanceSymmetry(unittest.TestCase):
    def test_enu_range_is_bitwise_symmetric(self) -> None:
        """距离对称必须是**逐位相等**，不是"约等于"。"""
        rng_pairs = [
            (Vec3(0.0, 0.0, 0.0), Vec3(6000.0, 0.0, 3000.0)),
            (Vec3(-1234.5, 6789.0, 12.0), Vec3(9876.5, -4321.0, -7.0)),
            (Vec3(1e-9, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)),
            (Vec3(1e5, 1e5, 1e5), Vec3(-1e5, -1e5, -1e5)),
        ]
        for a, b in rng_pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(enu_range(a, b), enu_range(b, a))
                self.assertEqual(a.distance_to(b), b.distance_to(a))

    def test_two_dimensional_range_is_bitwise_symmetric(self) -> None:
        self.assertEqual(enu_range_2d(0.0, 0.0, 6000.0, 0.0),
                         enu_range_2d(6000.0, 0.0, 0.0, 0.0))
        self.assertEqual(enu_range_2d(-5000.0, 1234.5, 8000.0, -6000.0),
                         enu_range_2d(8000.0, -6000.0, -5000.0, 1234.5))

    def test_zero_distance(self) -> None:
        self.assertEqual(enu_range(Vec3(1.0, 2.0, 3.0), Vec3(1.0, 2.0, 3.0)), 0.0)

    def test_flat_scene_2d_equals_3d(self) -> None:
        """z=0 的旧场景里，三维距离与二维距离必须逐位一致。"""
        for ax, ay, bx, by in [
            (0.0, 0.0, 6000.0, 0.0), (0.0, 0.0, 0.0, 4000.0),
            (8000.0, 6000.0, 0.0, 0.0), (80000.0, -60000.0, 0.0, 0.0),
            (-1234.5, 6789.0, 9876.5, -4321.0),
        ]:
            with self.subTest(a=(ax, ay), b=(bx, by)):
                self.assertEqual(
                    enu_range_2d(ax, ay, bx, by),
                    enu_range(Vec3(ax, ay, 0.0), Vec3(bx, by, 0.0)),
                )

    def test_distance_matrix_symmetry(self) -> None:
        entries = [
            ("A", Pose(position=Vec3(0.0, 0.0, 0.0))),
            ("B", Pose(position=Vec3(6000.0, 0.0, 3000.0))),
            ("C", Pose(position=Vec3(-5000.0, 15000.0, 50.0))),
        ]
        matrix = distance_matrix(entries)
        for (ida, idb), value in matrix.items():
            self.assertEqual(value, matrix[(idb, ida)])
        for ida, _ in entries:
            self.assertEqual(matrix[(ida, ida)], 0.0)

    def test_entity_range_to_symmetric(self) -> None:
        a = Target("T1", 0.0, 0.0)
        b = Target("T2", 6000.0, 4000.0)
        self.assertEqual(a.range_to(b.x, b.y), b.range_to(a.x, a.y))
        self.assertEqual(a.range_to_entity(b), b.range_to_entity(a))


# ----------------------------------------------------------------------
# 3. 运动更新
# ----------------------------------------------------------------------

class TestMotionUpdate(unittest.TestCase):
    def test_closed_form_linear_motion(self) -> None:
        target = Target("T1", 1000.0, 2000.0, rcs_m2=1.0, vx=10.0, vy=-5.0, vz=1.0)
        for step in range(1, 11):
            target.advance(1.0)
            self.assertAlmostEqual(target.x, 1000.0 + 10.0 * step, places=9)
            self.assertAlmostEqual(target.y, 2000.0 - 5.0 * step, places=9)
            self.assertAlmostEqual(target.z, 0.0 + 1.0 * step, places=9)

    def test_timestamp_advances_with_motion(self) -> None:
        target = Target("T1", 0.0, 0.0, vx=1.0)
        self.assertEqual(target.timestamp_s, 0.0)
        target.advance(2.5)
        self.assertAlmostEqual(target.timestamp_s, 2.5)
        target.update_position(1.5)  # 旧接口名也必须推进时间
        self.assertAlmostEqual(target.timestamp_s, 4.0)

    def test_velocity_unchanged_by_motion(self) -> None:
        target = Target("T1", 0.0, 0.0, vx=7.0, vy=-3.0, vz=2.0)
        before = target.velocity
        target.advance(3.0)
        self.assertEqual(target.velocity, before)

    def test_zero_velocity_keeps_position(self) -> None:
        """静止平台推进后位置必须**逐位不变**（旧场景的雷达/侦察机都是静止的）。"""
        radar = Radar("R1", 1234.5, -678.9)
        before = (radar.x, radar.y, radar.z)
        for _ in range(10):
            radar.advance(1.0)
        self.assertEqual((radar.x, radar.y, radar.z), before)
        self.assertAlmostEqual(radar.timestamp_s, 10.0)

    def test_radar_uses_legacy_velocity_field_names(self) -> None:
        radar = Radar("R1", 0.0, 0.0, velocity_x=5.0, velocity_y=-2.0, velocity_z=1.0)
        self.assertEqual(radar.velocity, Vec3(5.0, -2.0, 1.0))
        radar.advance(2.0)
        self.assertAlmostEqual(radar.x, 10.0)
        self.assertAlmostEqual(radar.y, -4.0)
        self.assertAlmostEqual(radar.z, 2.0)

    def test_update_position_delegates_to_advance(self) -> None:
        for cls, kwargs in (
            (Target, {"target_id": "T"}),
            (EnemyInterceptor, {"interceptor_id": "E"}),
            (Jammer, {"jammer_id": "J"}),
            (Radar, {"radar_id": "R"}),
        ):
            with self.subTest(cls=cls.__name__):
                entity = cls(x=0.0, y=0.0, **kwargs)
                entity.set_pose(velocity=Vec3(3.0, 4.0, 0.0))
                entity.update_position(2.0)
                self.assertAlmostEqual(entity.x, 6.0)
                self.assertAlmostEqual(entity.y, 8.0)
                self.assertAlmostEqual(entity.timestamp_s, 2.0)

    def test_set_pose_writes_all_components(self) -> None:
        jammer = Jammer("J1", 0.0, 0.0)
        jammer.set_pose(
            position=Vec3(1.0, 2.0, 3.0),
            velocity=Vec3(4.0, 5.0, 6.0),
            attitude=Attitude(10.0, 20.0, 30.0),
            time_s=7.5,
        )
        self.assertEqual(jammer.position, Vec3(1.0, 2.0, 3.0))
        self.assertEqual(jammer.velocity, Vec3(4.0, 5.0, 6.0))
        self.assertEqual(jammer.attitude.heading_deg, 10.0)
        self.assertEqual(jammer.pitch_deg, 20.0)
        self.assertEqual(jammer.roll_deg, 30.0)
        self.assertEqual(jammer.timestamp_s, 7.5)


# ----------------------------------------------------------------------
# 4. 时间同步
# ----------------------------------------------------------------------

class TestTimeSynchronization(unittest.TestCase):
    def test_mismatched_pose_times_raise(self) -> None:
        a = Pose(position=Vec3(0.0, 0.0, 0.0), time_s=0.0)
        b = Pose(position=Vec3(1000.0, 0.0, 0.0), time_s=5.0)
        with self.assertRaises(TimeSyncError):
            relation("A", a, "B", b, time_s=0.0)

    def test_query_time_must_match_poses(self) -> None:
        a = Pose(position=Vec3(0.0, 0.0, 0.0), time_s=0.0)
        b = Pose(position=Vec3(1000.0, 0.0, 0.0), time_s=0.0)
        with self.assertRaises(TimeSyncError):
            relation("A", a, "B", b, time_s=3.0)

    def test_default_time_taken_from_observer(self) -> None:
        a = Pose(position=Vec3(0.0, 0.0, 0.0), time_s=4.0)
        b = Pose(position=Vec3(1000.0, 0.0, 0.0), time_s=4.0)
        rel = relation("A", a, "B", b)
        self.assertEqual(rel.time_s, 4.0)

    def test_propagate_keeps_time_consistent(self) -> None:
        a = Pose(position=Vec3(0.0, 0.0, 0.0), velocity=Vec3(0.0, 0.0, 0.0), time_s=0.0)
        b = Pose(position=Vec3(1000.0, 0.0, 0.0), velocity=Vec3(-10.0, 0.0, 0.0), time_s=0.0)
        later = (a.propagate(5.0), b.propagate(5.0))
        rel = relation("A", later[0], "B", later[1])
        self.assertEqual(rel.time_s, 5.0)
        self.assertAlmostEqual(rel.range_m, 950.0, places=9)
        self.assertTrue(rel.closing)

    def test_scene_synchronized_after_advance(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        sim.assert_scene_time_synchronized()
        for _ in range(7):
            sim.step(6)
        sim.assert_scene_time_synchronized()
        self.assertAlmostEqual(sim.scene.reference_time, sim.current_time, places=9)

    def test_scene_detects_time_drift(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        sim.scene.by_id("ESM_AIR").timestamp_s = 99.0  # 人为制造漂移
        with self.assertRaises(SceneError):
            sim.scene.assert_time_synchronized()

    def test_set_reference_time_aligns_without_moving(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        positions = {e.entity_id: e.position for e in sim.scene.entities}
        sim.scene.set_reference_time(12.0)
        sim.assert_scene_time_synchronized()
        for entity in sim.scene.entities:
            self.assertEqual(entity.position, positions[entity.entity_id])


# ----------------------------------------------------------------------
# 5. 多实体索引
# ----------------------------------------------------------------------

class TestEntityIndex(unittest.TestCase):
    def test_multi_scene_counts(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        counts = sim.scene.count_by_kind()
        self.assertEqual(counts[KIND_RADAR], 2)
        self.assertEqual(counts[KIND_TARGET], 3)
        self.assertEqual(counts["interceptor"], 2)
        self.assertEqual(counts["jammer"], 2)
        self.assertEqual(len(sim.scene.entities), 9)

    def test_by_id_returns_same_object_as_legacy_lists(self) -> None:
        """索引与旧列表必须指向**同一个对象**，否则状态会分叉。"""
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        self.assertIs(sim.scene.by_id("RADAR_A"), sim.radar)
        self.assertIs(sim.scene.by_id("RADAR_B"), sim.radars[1])
        self.assertIs(sim.scene.by_id("TGT_ESCORT"), sim.targets[2])

    def test_index_reflects_live_mutation(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        sim.scene.by_id("TGT_LOW_SLOW").x = 12345.0
        self.assertEqual(sim.targets[1].x, 12345.0)

    def test_unknown_id_raises(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        with self.assertRaises(SceneError):
            sim.scene.by_id("NO_SUCH_ENTITY")
        self.assertIsNone(sim.scene.maybe_by_id("NO_SUCH_ENTITY"))

    def test_duplicate_id_across_kinds_raises(self) -> None:
        radar = Radar("DUP", 0.0, 0.0)
        target = Target("DUP", 1000.0, 0.0)
        with self.assertRaises(SceneError):
            Scene([radar], [target], [EnemyInterceptor("E1", 0.0, 0.0)], [], name="dup")

    def test_duplicate_id_within_kind_raises(self) -> None:
        with self.assertRaises(SceneError):
            Scene(
                [Radar("R", 0.0, 0.0)],
                [Target("T", 0.0, 0.0), Target("T", 1.0, 1.0)],
                [EnemyInterceptor("E", 0.0, 0.0)],
                [],
            )

    def test_empty_id_raises(self) -> None:
        with self.assertRaises(SceneError):
            Scene([Radar("R", 0.0, 0.0)], [Target("", 0.0, 0.0)],
                  [EnemyInterceptor("E", 0.0, 0.0)], [])

    def test_entity_ids_unique_and_orderable(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        ids = sim.scene.entity_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_of_kind_and_active_filter(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        self.assertEqual(len(sim.scene.of_kind(KIND_TARGET)), 3)
        sim.targets[0].is_active = False
        self.assertEqual(len(sim.scene.of_kind(KIND_TARGET, active_only=True)), 2)

    def test_relation_count_is_ordered_product(self) -> None:
        """雷达→目标的关系数必须是 N_radar × N_target（有向、去掉自反）。"""
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        relations = sim.radar_target_relations()
        self.assertEqual(len(relations), 2 * 3)
        observers = {r.observer_id for r in relations}
        self.assertEqual(observers, {"RADAR_A", "RADAR_B"})

    def test_all_relations_count(self) -> None:
        sim = Simulator(MULTI_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        n = len(sim.scene.entities)
        self.assertEqual(len(sim.scene.all_relations()), n * (n - 1))


# ----------------------------------------------------------------------
# 6. 关系语义（谁相对谁）
# ----------------------------------------------------------------------

class TestRelationSemantics(unittest.TestCase):
    def _poses(self):
        observer = Pose(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            attitude=Attitude(0.0, 0.0, 0.0),
            time_s=0.0,
        )
        target = Pose(
            position=Vec3(3000.0, 4000.0, 0.0),
            velocity=Vec3(10.0, 0.0, 0.0),
            time_s=0.0,
        )
        return observer, target

    def test_reverse_relation_flips_angles_but_not_range_rate(self) -> None:
        """交换双方：角度量翻转，而距离与**距离变化率**不变。

        这里刻意分成两组断言，因为很容易搞混：
        * `range_m` / `range_rate_mps` / `closing` 是**这一对实体**的属性，对称；
        * `los_enu` / `azimuth_deg` / `elevation_deg` / `elevation_body_deg`
          是**有向**的，交换后取反或差 180°。
        早期版本的文档曾把径向速度写成"符号相反"，那是错的——
        有向的是相对速度向量，而径向速度是它在连线上的投影。
        """
        observer, target = self._poses()
        forward = relation("A", observer, "B", target)
        backward = relation("B", target, "A", observer)

        # --- 对称量 ---
        self.assertEqual(forward.range_m, backward.range_m)
        self.assertEqual(forward.range_rate_mps, backward.range_rate_mps)
        self.assertEqual(forward.closing, backward.closing)
        self.assertGreater(forward.range_rate_mps, 0.0)  # 该构型下两者在远离

        # --- 有向量 ---
        self.assertTrue(
            forward.los_enu.is_close(-backward.los_enu, tol=1e-12),
            "视线向量应当互为反向",
        )
        self.assertAlmostEqual(
            abs(normalize_angle_deg(forward.azimuth_deg - backward.azimuth_deg)),
            180.0,
            places=9,
        )
        self.assertAlmostEqual(forward.elevation_deg, -backward.elevation_deg, places=9)
        self.assertAlmostEqual(
            forward.elevation_body_deg, -backward.elevation_body_deg, places=9
        )

    def test_bearing_depends_on_each_observers_own_heading(self) -> None:
        """机体方位各自依赖自己的航向，两者之间没有固定关系。

        方位角自**正北（+y）**起顺时针为正，因此目标在 (3000, 4000) 时
        `azimuth = atan2(3000, 4000) = 36.87°`（先东后北的顺序容易记反，
        写成 atan2(4000, 3000) 会得到 53.13°，那是以正东为基准的角度）。
        """
        position = Vec3(0.0, 0.0, 0.0)
        target_position = Vec3(3000.0, 4000.0, 0.0)
        a = Pose(position=position, attitude=Attitude(0.0), time_s=0.0)
        b = Pose(position=target_position, attitude=Attitude(0.0), time_s=0.0)

        forward = relation("A", a, "B", b)
        self.assertAlmostEqual(forward.azimuth_deg, 36.8698976458, places=6)
        # A 机头朝北（heading=0）时，机体方位等于绝对方位
        self.assertAlmostEqual(forward.bearing_deg, forward.azimuth_deg, places=9)

        # 若 A 把机头转到东（heading=90°），绝对方位不变，机体方位减少 90°
        a_turned = Pose(position=position, attitude=Attitude(90.0), time_s=0.0)
        turned = relation("A", a_turned, "B", b)
        self.assertAlmostEqual(turned.azimuth_deg, forward.azimuth_deg, places=9)
        self.assertAlmostEqual(turned.bearing_deg, forward.bearing_deg - 90.0, places=6)

    def test_los_points_observer_to_target(self) -> None:
        observer, target = self._poses()
        rel = relation("A", observer, "B", target)
        expected = los_unit(observer.position, target.position)
        self.assertTrue(rel.los_enu.is_close(expected, tol=1e-12))
        self.assertAlmostEqual(rel.los_enu.norm(), 1.0, places=12)

    def test_radial_velocity_sign_convention(self) -> None:
        """正值 = 远离，负值 = 接近（写进 docstring 的约定，用测试钉住）。"""
        los = Vec3(1.0, 0.0, 0.0)
        zero = Vec3(0.0, 0.0, 0.0)
        self.assertGreater(radial_velocity_mps(zero, Vec3(5.0, 0.0, 0.0), los), 0.0)
        self.assertLess(radial_velocity_mps(zero, Vec3(-5.0, 0.0, 0.0), los), 0.0)
        # 切向运动没有径向分量
        self.assertAlmostEqual(
            radial_velocity_mps(zero, Vec3(0.0, 5.0, 0.0), los), 0.0
        )

    def test_colocated_entities_have_no_direction(self) -> None:
        same = Vec3(1.0, 2.0, 3.0)
        rel = relation(
            "A", Pose(position=same, time_s=0.0), "B", Pose(position=same, time_s=0.0)
        )
        self.assertEqual(rel.range_m, 0.0)
        self.assertEqual(rel.los_enu, Vec3())
        self.assertTrue(rel.is_colocated)

    def test_los_unit_of_zero_vector_raises(self) -> None:
        with self.assertRaises(GeometryError):
            los_unit(Vec3(1.0, 1.0, 1.0), Vec3(1.0, 1.0, 1.0))

    def test_angular_separation(self) -> None:
        self.assertAlmostEqual(
            angular_separation_deg(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)), 0.0
        )
        self.assertAlmostEqual(
            angular_separation_deg(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)), 90.0
        )
        self.assertAlmostEqual(
            angular_separation_deg(Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0)), 180.0
        )

    def test_pairwise_relations_skips_self(self) -> None:
        observers = [("A", Pose(position=Vec3(0.0, 0.0, 0.0)))]
        targets = [
            ("A", Pose(position=Vec3(0.0, 0.0, 0.0))),
            ("B", Pose(position=Vec3(100.0, 0.0, 0.0))),
        ]
        self.assertEqual(len(pairwise_relations(observers, targets)), 1)


# ----------------------------------------------------------------------
# 7. 向后兼容（逐位）
# ----------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    def test_range_to_matches_legacy_hypot_bitwise(self) -> None:
        """新的 range_to 必须与升级前 `math.hypot(self.x-x, self.y-y)` 逐位一致。"""
        import random

        rng = random.Random(20260918)
        for _ in range(20000):
            x, y = rng.uniform(-2e5, 2e5), rng.uniform(-2e5, 2e5)
            tx, ty = rng.uniform(-2e5, 2e5), rng.uniform(-2e5, 2e5)
            target = Target("T", tx, ty)
            self.assertEqual(target.range_to(x, y), math.hypot(tx - x, ty - y))

    def test_legacy_config_builds_single_radar(self) -> None:
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        self.assertEqual(len(sim.radars), 1)
        self.assertIs(sim.radar, sim.radars[0])
        self.assertEqual(sim.radar.entity_id, "RADAR1")
        self.assertEqual(sim.scene.primary_radar.radar_id, "RADAR1")

    def test_legacy_scene_geometry_unchanged(self) -> None:
        """旧场景在 z=0 下的三维关系距离必须等于旧的二维距离。"""
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        for _ in range(5):
            sim.step(6)
        for target in sim.targets:
            legacy = target.range_to(sim.radar.x, sim.radar.y)
            geometric = sim.relation(sim.radar.entity_id, target.target_id).range_m
            self.assertEqual(legacy, geometric)

    def test_legacy_episode_metrics_unchanged(self) -> None:
        """固定 80 W 跑完一段，核心指标必须与升级前完全一致。

        注意：旧配置 `terminate_on_energy_exhausted = true`，
        80 W × 1 s = 80 J/步，1400 J 预算只够 17.5 步，
        因此 episode 会**提前终止**（约 18~20 步），而不是跑满 61 步。
        未执行的步按口径计入"未满足"，这正是满足率只有 0.2951 的原因
        （18/61 ≈ 0.295）。升级不得改变这两个数。
        """
        from metrics import summarize_run

        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        results = sim.run_fixed_power()

        # 能量耗尽导致提前终止
        self.assertLess(len(results), 61)
        self.assertEqual(results[-1].cumulative_energy_j, 1400.0)
        self.assertTrue(sim.terminated_by_energy)

        summary = summarize_run(
            results,
            label="固定功率",
            policy="固定功率",
            energy_budget_j=sim.radar.energy_budget_j,
            lpi_pint_threshold=sim.scenario.lpi_pint_threshold,
            horizon_steps=sim.scenario.num_steps,
        )
        # 与 main.py 输出的旧值逐位对齐
        self.assertAlmostEqual(summary["horizon_satisfaction_rate"], 0.2951, places=4)
        self.assertAlmostEqual(summary["composite_reward"], -3.1912, places=4)
        self.assertAlmostEqual(summary["avg_tx_power_w"], 70.0, places=6)

    def test_entity_state_dict_has_required_fields(self) -> None:
        target = Target("T1", 1000.0, 2000.0, rcs_m2=2.0, vx=1.0, vz=2.0,
                        heading_deg=30.0, platform_id="P1")
        record = target.to_state_dict()
        for key in ("entity_id", "entity_kind", "platform_id", "time_s",
                    "x", "y", "z", "vx", "vy", "vz", "heading_deg",
                    "pitch_deg", "roll_deg", "is_active"):
            self.assertIn(key, record)
        self.assertEqual(record["entity_id"], "T1")
        self.assertEqual(record["entity_kind"], KIND_TARGET)
        self.assertEqual(record["platform_id"], "P1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
