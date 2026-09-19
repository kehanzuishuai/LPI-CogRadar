"""多平台场景与快照导出的单元测试（v4.1）。

覆盖：
* 多平台配置能正确构造成 2 雷达 + 3 目标 + 2 侦察机 + 2 干扰源；
* 实体注册表索引、按类型查询、平台归属分组正确；
* 有向几何关系数量 = N_观察者 × N_目标；
* 聚合值（最近/最远/最小 RCS）确实**等于**由显式关系算出来的结果
  —— 这钉住"聚合值降级为派生视图"这件事没有走样；
* CSV/JSON 快照导出的行数、列名与可回读性。

不需要 torch，base 环境即可运行。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.scene import (  # noqa: E402
    ENTITY_SNAPSHOT_FIELDS,
    RELATION_SNAPSHOT_FIELDS,
    SceneError,
)
from engine.simulator import Simulator  # noqa: E402
from models.entity import (  # noqa: E402
    KIND_INTERCEPTOR,
    KIND_JAMMER,
    KIND_RADAR,
    KIND_TARGET,
)

MULTI_CONFIG = "config/multi_platform_scenario.json"
LEGACY_CONFIG = "config/radar_scenario_v1.json"


def _multi_sim() -> Simulator:
    sim = Simulator(MULTI_CONFIG)
    sim.load_config()
    sim.reset(seed=42)
    return sim


class TestMultiPlatformScene(unittest.TestCase):
    def test_composition(self) -> None:
        sim = _multi_sim()
        counts = sim.scene.count_by_kind()
        self.assertEqual(counts[KIND_RADAR], 2)
        self.assertEqual(counts[KIND_TARGET], 3)
        self.assertEqual(counts[KIND_INTERCEPTOR], 2)
        self.assertEqual(counts[KIND_JAMMER], 2)

    def test_primary_radar_is_first(self) -> None:
        sim = _multi_sim()
        self.assertEqual(sim.radar.radar_id, "RADAR_A")
        self.assertEqual(sim.scene.primary_radar.radar_id, "RADAR_A")
        self.assertEqual([r.radar_id for r in sim.radars], ["RADAR_A", "RADAR_B"])

    def test_all_entities_have_three_dimensional_pose(self) -> None:
        sim = _multi_sim()
        for entity in sim.scene.entities:
            pose = entity.pose
            self.assertEqual(len(pose.position.to_dict()), 3)
            self.assertEqual(len(pose.velocity.to_dict()), 3)
            self.assertIsInstance(pose.time_s, float)
            self.assertTrue(entity.entity_id)

    def test_non_primary_radar_also_moves(self) -> None:
        """非主雷达也必须运动，否则多平台几何是假的。"""
        sim = _multi_sim()
        before = sim.radars[1].position
        for _ in range(5):
            sim.step(6)
        after = sim.radars[1].position
        self.assertNotEqual(before, after)
        self.assertAlmostEqual(after.x, before.x - 5.0 * 5, places=9)

    def test_platform_grouping(self) -> None:
        sim = _multi_sim()
        groups: dict = {}
        for entity in sim.scene.entities:
            groups.setdefault(entity.platform_id, []).append(entity.entity_id)
        self.assertEqual(
            sorted(groups["AIRCRAFT_1"]), ["JAM_ESCORT", "TGT_ESCORT", "TGT_HIGH_FAST"]
        )
        self.assertEqual(groups["AIRCRAFT_2"], ["TGT_LOW_SLOW"])

    def test_relation_counts(self) -> None:
        sim = _multi_sim()
        self.assertEqual(len(sim.scene.relations_between(KIND_RADAR, KIND_TARGET)), 6)
        self.assertEqual(len(sim.scene.relations_between(KIND_RADAR, KIND_JAMMER)), 4)
        self.assertEqual(
            len(sim.scene.relations_between(KIND_INTERCEPTOR, KIND_RADAR)), 4
        )

    def test_derived_aggregates_match_explicit_relations(self) -> None:
        """聚合值必须与显式关系算出来的一致（聚合只是派生视图）。"""
        sim = _multi_sim()
        observer = sim.radar.radar_id
        relations = sim.scene.relations_from(observer, target_kinds=(KIND_TARGET,))

        nearest = sim.scene.nearest(observer, KIND_TARGET)
        farthest = sim.scene.farthest(observer, KIND_TARGET)
        self.assertAlmostEqual(nearest.range_m, min(r.range_m for r in relations))
        self.assertAlmostEqual(farthest.range_m, max(r.range_m for r in relations))

        # "最小 RCS 目标"是派生量，但仍必须能被显式关系里的目标 ID 解释
        min_rcs_target = sim.scene.min_rcs_target()
        self.assertIn(min_rcs_target.target_id, {r.target_id for r in relations})

    def test_nearest_is_direction_aware(self) -> None:
        """"最近目标"依赖于**哪部雷达在看**，这正是升级前的聚合值做不到的。"""
        sim = _multi_sim()
        nearest_a = sim.scene.nearest("RADAR_A", KIND_TARGET)
        nearest_b = sim.scene.nearest("RADAR_B", KIND_TARGET)
        self.assertNotEqual(nearest_a.observer_id, nearest_b.observer_id)
        # 至少有一部雷达的最近目标与另一部不同，否则这个测试没有意义
        relations_a = sim.scene.relations_from("RADAR_A", target_kinds=(KIND_TARGET,))
        relations_b = sim.scene.relations_from("RADAR_B", target_kinds=(KIND_TARGET,))
        order_a = sorted(relations_a, key=lambda r: r.range_m)
        order_b = sorted(relations_b, key=lambda r: r.range_m)
        self.assertNotEqual(
            [r.target_id for r in order_a], [r.target_id for r in order_b]
        )

    def test_multi_scene_physics_still_runs(self) -> None:
        """多平台场景也能跑物理评估（只针对主雷达），且能量约束仍然成立。"""
        sim = _multi_sim()
        for _ in range(20):
            sim.step(6)
        self.assertLessEqual(
            sim.cumulative_energy_j, sim.radar.energy_budget_j + 1e-9
        )
        self.assertTrue(sim.results)


class TestSceneSnapshotExport(unittest.TestCase):
    """导出测试。

    刻意**不用 `tempfile.TemporaryDirectory()`**：本工程的执行环境把
    临时目录映射到工作区内的一个临时路径，而该路径对新建子目录的写入会被
    沙箱拒绝（实测 `PermissionError: [WinError 5]`）。
    因此统一写到项目内已知可写的 `output/_test_scene`，并在 tearDown 里清理。
    """

    SCRATCH = os.path.join("output", "_test_scene")

    def setUp(self) -> None:
        import shutil

        shutil.rmtree(self.SCRATCH, ignore_errors=True)
        os.makedirs(self.SCRATCH, exist_ok=True)
        self.out = self.SCRATCH

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.SCRATCH, ignore_errors=True)

    def test_entity_csv_columns_and_rows(self) -> None:
        sim = _multi_sim()
        path = os.path.join(self.out, "entities.csv")
        sim.scene.write_entities_csv(path)
        with open(path, "r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), ENTITY_SNAPSHOT_FIELDS)
        self.assertEqual(len(rows), len(sim.scene.entities))

    def test_relation_csv_columns_and_rows(self) -> None:
        sim = _multi_sim()
        path = os.path.join(self.out, "relations.csv")
        sim.scene.write_relations_csv(path)
        with open(path, "r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), RELATION_SNAPSHOT_FIELDS)
        n = len(sim.scene.entities)
        self.assertEqual(len(rows), n * (n - 1))

    def test_relation_csv_has_explicit_semantics(self) -> None:
        """每一行都必须能回答"谁相对谁、哪一刻"。"""
        sim = _multi_sim()
        path = os.path.join(self.out, "relations.csv")
        sim.scene.write_relations_csv(path)
        with open(path, "r", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        self.assertTrue(row["observer_id"])
        self.assertTrue(row["target_id"])
        self.assertEqual(row["observer_id"] != row["target_id"], True)
        self.assertEqual(float(row["time_s"]), 0.0)
        self.assertGreater(float(row["range_m"]), 0.0)

    def test_pair_subset_export(self) -> None:
        sim = _multi_sim()
        path = os.path.join(self.out, "pairs.csv")
        sim.scene.write_relations_csv(
            path, pairs=[("RADAR_A", "TGT_HIGH_FAST"), ("RADAR_B", "TGT_ESCORT")]
        )
        with open(path, "r", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {(r["observer_id"], r["target_id"]) for r in rows},
            {("RADAR_A", "TGT_HIGH_FAST"), ("RADAR_B", "TGT_ESCORT")},
        )

    def test_appended_multistep_csv_row_count(self) -> None:
        sim = _multi_sim()
        path = os.path.join(self.out, "timeline.csv")
        sim.scene.write_entities_csv(path, append=False)
        for _ in range(4):
            sim.step(6)
            sim.scene.write_entities_csv(path, append=True)
        with open(path, "r", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(sim.scene.entities) * 5)
        times = sorted({float(r["time_s"]) for r in rows})
        self.assertEqual(times, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_json_snapshot_structure(self) -> None:
        sim = _multi_sim()
        path = os.path.join(self.out, "snap.json")
        sim.scene.write_json(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("scene_name", "time_s", "coordinate_system", "counts",
                    "index", "timestamps", "entities", "relations"):
            self.assertIn(key, payload)
        self.assertEqual(payload["coordinate_system"]["frame"], "ENU")
        self.assertEqual(len(payload["entities"]), 9)
        self.assertEqual(len(payload["relations"]), 9 * 8)
        self.assertEqual(
            payload["index"]["entity_ids"],
            [e.entity_id for e in sim.scene.entities],
        )

    def test_json_snapshot_records_relation_direction(self) -> None:
        sim = _multi_sim()
        payload = sim.scene.to_dict()
        forward = next(
            r for r in payload["relations"]
            if r["observer_id"] == "RADAR_A" and r["target_id"] == "TGT_HIGH_FAST"
        )
        backward = next(
            r for r in payload["relations"]
            if r["observer_id"] == "TGT_HIGH_FAST" and r["target_id"] == "RADAR_A"
        )
        self.assertEqual(forward["range_m"], backward["range_m"])
        self.assertAlmostEqual(
            forward["los_enu"]["x"], -backward["los_enu"]["x"], places=12
        )

    def test_legacy_scene_export_works(self) -> None:
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        sim.reset(seed=42)
        path = os.path.join(self.out, "legacy.json")
        sim.scene.write_json(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["counts"][KIND_RADAR], 1)
        self.assertEqual(payload["counts"][KIND_TARGET], 2)

    def test_export_cli_script_exists_and_imports(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("export_scene", "export_scene.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        pairs = module.parse_pairs("A>B, C>D")
        self.assertEqual(pairs, [("A", "B"), ("C", "D")])
        self.assertIsNone(module.parse_pairs("   "))
        with self.assertRaises(ValueError):
            module.parse_pairs("BADFORMAT")


class TestSceneValidation(unittest.TestCase):
    def test_unknown_config_field_is_reported(self) -> None:
        """配置里写错字段名必须报错，并指出是哪一段、支持哪些字段。

        静默忽略未知字段会让"某个平台的姿态没生效"这类问题极难定位。
        """
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        data = json.loads(json.dumps(sim._raw_config))
        data["targets"][0]["vel_z"] = 1.0  # 正确名字是 vz
        with self.assertRaises(ValueError) as ctx:
            sim._build_from_config(data)
        message = str(ctx.exception)
        self.assertIn("targets", message)
        self.assertIn("vel_z", message)

    def test_missing_radar_section_reported(self) -> None:
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        data = json.loads(json.dumps(sim._raw_config))
        data.pop("radar")
        with self.assertRaises(ValueError) as ctx:
            sim._build_from_config(data)
        self.assertIn("radar", str(ctx.exception))

    def test_scene_rejects_duplicate_ids_from_config(self) -> None:
        sim = Simulator(LEGACY_CONFIG)
        sim.load_config()
        data = json.loads(json.dumps(sim._raw_config))
        data["targets"][1]["target_id"] = data["targets"][0]["target_id"]
        with self.assertRaises(SceneError):
            sim._build_from_config(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
