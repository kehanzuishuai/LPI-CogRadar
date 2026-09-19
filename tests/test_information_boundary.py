"""信息边界测试（v4.5 一致性验收）。

要求（用户口径）：**整份快照及其序列化结果**都要过信息边界检查，
不能只查新增字段；真值 ID、真实虚警标签、隐藏平台状态、未来消息、
只有仿真器才知道的缺失原因，只能进离线评测通道。

因此本文件的核心不是"某个字段对不对"，而是：

1. 在线快照**整体**（递归、含序列化结果）不含任何真值通道；
2. 检查器**有牙齿**：把离线快照交给在线检查必须报违规——
   否则"检查通过"可能只是因为检查器什么也不查；
3. 不确定的内容必须**显式标为未知**，不能编造确定值；
4. 缺失原因带**来源标注**（设备已知 / 测量推断 / 仅评测端）。

不需要 torch。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec  # noqa: E402
from ai.context import (  # noqa: E402
    TRUTH_PROVENANCE_FIELDS,
    online_snapshot_from_env,
    snapshot_from_simulator,
)
from ai.snapshot_boundary import (  # noqa: E402
    FORBIDDEN_ONLINE_KEYS,
    ONLINE_UNKNOWN_FIELDS,
    SOURCE_OFFLINE,
    SOURCE_ONLINE,
    snapshot_information_violations,
    strip_eval_only,
)
from sensor.record import REASON_PROVENANCE  # noqa: E402


def _env(mode: str = "realistic", steps: int = 5, seed: int = 42):
    env = ec.make_env(observation_mode=mode)
    env.reset(seed=seed)
    for _ in range(steps):
        env.step(6)
    return env


class TestSnapshotSplit(unittest.TestCase):
    def test_offline_snapshot_declares_itself(self) -> None:
        env = _env()
        snapshot = snapshot_from_simulator(
            env.sim, result=env.sim.results[-1], env=env)
        payload = snapshot.to_dict()
        self.assertEqual(payload["information_boundary"], SOURCE_OFFLINE)
        self.assertTrue(payload["boundary_provenance"])
        # 离线快照**允许**带真值——它是评测通道
        self.assertTrue(payload["targets"])
        self.assertEqual(
            snapshot_information_violations(payload, SOURCE_OFFLINE), [])

    def test_online_snapshot_has_no_truth_roster(self) -> None:
        env = _env()
        snapshot = online_snapshot_from_env(env)
        payload = snapshot.to_dict()
        self.assertEqual(payload["information_boundary"], SOURCE_ONLINE)
        self.assertEqual(payload["targets"], [])
        self.assertEqual(payload["interceptors"], [])
        self.assertEqual(payload["jammers"], [])

    def test_online_snapshot_passes_whole_payload_check(self) -> None:
        """**整份**序列化结果都要过（递归扫描，不只查新增字段）。"""
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertEqual(violations, [], f"在线快照违规：{violations}")


class TestCheckerHasTeeth(unittest.TestCase):
    """检查器必须真的能拦住真值——否则"通过"没有意义。"""

    def test_offline_snapshot_fails_online_check(self) -> None:
        env = _env()
        offline = snapshot_from_simulator(
            env.sim, result=env.sim.results[-1], env=env).to_dict()
        violations = snapshot_information_violations(offline, SOURCE_ONLINE)
        self.assertTrue(violations, "离线快照竟然通过了在线检查")
        joined = " ".join(violations)
        self.assertIn("targets", joined)
        self.assertIn("boundary", joined)

    def test_injected_truth_id_is_caught(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        payload["fusion_state"]["tracks"][0]["truth_id"] = "TGT1" \
            if payload.get("fusion_state", {}).get("tracks") else None
        # 无论有没有航迹，都要能识别出这个真值键
        if not payload.get("fusion_state", {}).get("tracks"):
            payload["extra"] = {"truth_id": "TGT1"}
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertTrue(any("truth_id" in v for v in violations), violations)

    def test_injected_false_alarm_label_is_caught(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        payload["measurement_state"]["n_false_alarms"] = 3
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertTrue(any("n_false_alarms" in v for v in violations),
                        violations)

    def test_reason_counts_are_rejected_online(self) -> None:
        """逐原因计数需要真值实体清单当分母 → 在线不得出现。"""
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        payload["measurement_state"]["reason_counts"] = {"out_of_fov": 2}
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertTrue(any("reason_counts" in v for v in violations))

    def test_undeclared_unknown_fields_are_rejected(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        payload["online_unknown_fields"] = []
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertTrue(any("未知项" in v for v in violations))

    def test_missing_boundary_declaration_is_rejected(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        payload.pop("information_boundary")
        violations = snapshot_information_violations(payload, SOURCE_ONLINE)
        self.assertTrue(any("information_boundary" in v for v in violations))


class TestSerializationRoundTrip(unittest.TestCase):
    """序列化结果也要满足边界——不能只在内存对象上成立。"""

    def test_online_snapshot_json_round_trip_stays_clean(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        restored = json.loads(json.dumps(payload, ensure_ascii=False))
        violations = snapshot_information_violations(restored, SOURCE_ONLINE)
        self.assertEqual(violations, [])

    def test_boundary_provenance_survives_serialization(self) -> None:
        env = _env()
        payload = snapshot_from_simulator(
            env.sim, result=env.sim.results[-1], env=env).to_dict()
        restored = json.loads(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(set(restored["boundary_provenance"]),
                         set(TRUTH_PROVENANCE_FIELDS))

    def test_no_forbidden_key_anywhere_online(self) -> None:
        """递归扫描：在线快照的任何层级都不得出现真值键（非空值）。"""
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        offenders = []

        def walk(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in FORBIDDEN_ONLINE_KEYS and value not in (
                            None, False, 0, [], {}, ""):
                        offenders.append(f"{path}.{key}={value!r}")
                    walk(value, f"{path}.{key}")
            elif isinstance(node, (list, tuple)):
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")

        walk(payload)
        self.assertEqual(offenders, [], f"在线快照出现真值键：{offenders}")


class TestUnknownAndProvenance(unittest.TestCase):
    def test_online_snapshot_declares_what_it_cannot_know(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        for field in ONLINE_UNKNOWN_FIELDS:
            self.assertIn(field, payload["online_unknown_fields"],
                          f"在线快照没有声明未知项 {field}")

    def test_missing_reason_provenance_is_reported(self) -> None:
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        provenance = payload["missing_reason_provenance"]
        self.assertEqual(provenance, REASON_PROVENANCE)
        self.assertEqual(provenance["missed_detection"], "evaluation_only")
        self.assertEqual(provenance["sensor_unavailable"], "device_known")

    def test_eval_only_reason_is_not_reported_as_certain(self) -> None:
        """漏检/虚警这类"只有仿真器知道"的量，在线必须是空的。"""
        env = _env()
        payload = online_snapshot_from_env(env).to_dict()
        measurement = payload["measurement_state"] or {}
        self.assertFalse(measurement.get("reason_counts"))
        self.assertFalse(measurement.get("reason_rates"))
        self.assertEqual(measurement.get("n_false_alarms", 0), 0)

    def test_full_mode_marks_truth_coupled_estimates(self) -> None:
        """full 模式下智能体本来就看得见真值 → 必须如实标注，而不是假装是估计。"""
        env = _env(mode="full")
        payload = online_snapshot_from_env(env).to_dict()
        self.assertIn("pd_min", payload["boundary_provenance"])

    def test_measurement_mode_estimates_are_not_marked_as_truth(self) -> None:
        env = _env(mode="realistic")
        payload = online_snapshot_from_env(env).to_dict()
        self.assertNotIn("pd_min", payload["boundary_provenance"])


class TestStripEvalOnly(unittest.TestCase):
    def test_stripping_makes_offline_payload_usable_online(self) -> None:
        env = _env()
        offline = snapshot_from_simulator(
            env.sim, result=env.sim.results[-1], env=env).to_dict()
        self.assertTrue(snapshot_information_violations(offline, SOURCE_ONLINE))
        cleaned = strip_eval_only(offline)
        self.assertEqual(
            snapshot_information_violations(cleaned, SOURCE_ONLINE), [])

    def test_stripping_removes_truth_rosters(self) -> None:
        env = _env()
        offline = snapshot_from_simulator(
            env.sim, result=env.sim.results[-1], env=env).to_dict()
        cleaned = strip_eval_only(offline)
        self.assertEqual(cleaned["targets"], [])
        self.assertEqual(cleaned["interceptors"], [])
        self.assertNotIn("n_false_alarms", cleaned["measurement_state"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
