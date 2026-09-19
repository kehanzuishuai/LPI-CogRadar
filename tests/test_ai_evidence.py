"""v4.5 AI 证据链测试：只读结构化证据、区分缺失原因、证据校验与回退。

要钉住的三件事：
1. **AI 上下文不含真值**：`measurement_state` / `fusion_state` 等节里
   不得出现 `truth_id`、真实目标位置、在途消息内容；
2. **缺失原因真的被区分**：视场外 / 超距离 / 遮挡 / 未更新 / 漏检 / 通信丢弃
   各自产生独立的发现码，不再笼统写成"观测退化"；
3. **证据校验有效且不误伤**：编造的数值/标识符/原因码被拦下，
   而上下文里真实存在的标识符不会被误判。

不需要 torch，base 环境即可运行。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec  # noqa: E402
from ai.context import snapshot_from_simulator  # noqa: E402
from ai.evidence_check import (  # noqa: E402
    collect_identifiers,
    collect_numbers,
    validate_text_against_context,
)
from ai.rule_provider import RuleProvider  # noqa: E402
from ai.schema import FINDING_CODES  # noqa: E402
from ai.track_explain import explain_cooperation, explain_track  # noqa: E402
from communication import (  # noqa: E402
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    CommBus,
    CommConfig,
)

V45_CODES = (
    "TARGET_OUT_OF_FOV", "TARGET_BEYOND_RANGE", "TARGET_OCCLUDED",
    "SENSOR_NOT_UPDATED", "MISSED_DETECTION", "SENSOR_UNAVAILABLE",
    "COMM_PACKET_LOST", "COMM_MESSAGE_EXPIRED", "COMM_LINK_DOWN",
    "COMM_QUEUE_FULL", "REMOTE_MEASUREMENT_DELAYED", "TRACK_COASTING",
    "TRACK_FRAGMENTED", "TRACK_UNCERTAIN", "ASSOCIATION_AMBIGUOUS",
    "REMOTE_SENSOR_CONTRIBUTION", "COOPERATIVE_TRACK_RECOVERED",
)


def _env_with_bus(policy: str = SHARE_CONSTRAINED, steps: int = 8, **overrides):
    env = ec.make_env(observation_mode="realistic")
    env.reset(seed=42)
    bus = CommBus(["RADAR1", "RADAR2"],
                  CommConfig(policy=policy, seed=42, **overrides))
    env.attach_comm_bus(bus)
    for _ in range(steps):
        _o, _r, terminated, truncated, _i = env.step(6)
        if terminated or truncated:
            break
    return env, bus


class TestFindingCodes(unittest.TestCase):
    def test_all_v45_codes_registered(self) -> None:
        for code in V45_CODES:
            self.assertIn(code, FINDING_CODES, f"发现码 {code} 未登记")

    def test_codes_have_chinese_titles(self) -> None:
        for code in V45_CODES:
            self.assertTrue(FINDING_CODES[code].strip())


class TestNoTruthInAiContext(unittest.TestCase):
    """AI 上下文必须只含算法当前可获得的信息。"""

    def _snapshot_dict(self, policy=SHARE_CONSTRAINED):
        env, _bus = _env_with_bus(policy=policy)
        snap = snapshot_from_simulator(env.sim, result=env.sim.results[-1], env=env)
        return snap.to_dict(), env

    def test_no_truth_keys_anywhere(self) -> None:
        payload, env = self._snapshot_dict()
        forbidden = ("truth_id", "truth_range_m", "truth_azimuth_deg",
                     "truth_elevation_deg", "truth_range_rate_mps")
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertNotIn(key, forbidden, f"AI 上下文出现真值字段 {key}")
                    stack.append(value)
            elif isinstance(node, (list, tuple)):
                stack.extend(node)

    def test_no_real_target_position_in_context(self) -> None:
        """真实目标位置不得出现在 AI 上下文里（航迹估计值不算）。"""
        payload, env = self._snapshot_dict()
        truth_positions = {(round(t.position.x, 3), round(t.position.y, 3))
                           for t in env.sim.targets if t.is_active}
        for track in (payload.get("fusion_state") or {}).get("tracks", []):
            pos = track["position"]
            key = (round(pos["x"], 3), round(pos["y"], 3))
            self.assertNotIn(key, truth_positions,
                             "航迹位置恰好等于真值位置，疑似泄露真值")

    def test_measurement_state_has_no_truth(self) -> None:
        payload, _env = self._snapshot_dict()
        ms = payload["measurement_state"]
        for candidate in ms["candidates"]:
            self.assertNotIn("truth_id", candidate)
            self.assertNotIn("truth_range_m", candidate)

    def test_cooperation_state_has_no_truth(self) -> None:
        payload, _env = self._snapshot_dict()
        co = payload["cooperation_state"]
        self.assertIsNotNone(co)
        self.assertNotIn("position_rmse_m", co, "收益类真值指标不得进入 AI 上下文")

    def test_no_share_context_has_no_links(self) -> None:
        payload, _env = self._snapshot_dict(policy=SHARE_NONE)
        self.assertEqual(payload["communication_state"]["n_links"], 0)


class TestReasonDistinction(unittest.TestCase):
    """缺失原因必须逐项区分，不能笼统写成"观测退化"。"""

    def test_out_of_fov_and_missed_detection_distinguished(self) -> None:
        env, _bus = _env_with_bus(steps=8)
        snap = snapshot_from_simulator(env.sim, result=env.sim.results[-1], env=env)
        codes = {f.code for f in RuleProvider().diagnose(snap).findings}
        # 默认单雷达场景里 TGT1 在视场外、且存在概率漏检
        self.assertIn("TARGET_OUT_OF_FOV", codes)
        self.assertTrue(
            "MISSED_DETECTION" in codes or "SENSOR_NOT_UPDATED" in codes,
            f"应至少出现一种随机缺失原因，实际 {sorted(codes)}",
        )

    def test_comm_drop_reasons_mapped(self) -> None:
        """丢弃原因到发现码的映射必须覆盖四种原因。"""
        from ai.rule_provider import RuleProvider as RP

        self.assertEqual(set(RP._COMM_DROP_CODES),
                         {"lost", "expired", "link_down", "queue_full"})
        self.assertEqual(set(RP._MEASUREMENT_REASON_CODES),
                         {"out_of_fov", "beyond_range", "occluded",
                          "not_updated", "missed_detection",
                          "sensor_unavailable"})

    def test_expired_message_produces_code(self) -> None:
        env, _bus = _env_with_bus(
            policy=SHARE_CONSTRAINED, steps=8,
            base_delay_s=2.0, loss_prob=0.2, expiry_s=0.5,
        )
        snap = snapshot_from_simulator(env.sim, result=env.sim.results[-1], env=env)
        codes = {f.code for f in RuleProvider().diagnose(snap).findings}
        self.assertTrue(
            "COMM_MESSAGE_EXPIRED" in codes or "COMM_PACKET_LOST" in codes,
            f"应出现通信类发现码，实际 {sorted(codes)}",
        )


class TestExplainers(unittest.TestCase):
    def test_explain_track_coasting(self) -> None:
        result = explain_track({"track": {
            "track_id": "LOCAL-T1", "status": "coasting", "hits": 2, "misses": 3,
            "local_updates": 2, "remote_updates": 1, "n_sources": 3,
            "freshness": 0.3, "measurement_age_s": 4.5,
            "sigma_position": {"x": 250.0, "y": 10.0, "z": 5.0},
            "source_sensors": ["SENSOR_A"], "platforms": ["LOCAL", "REMOTE"],
            "has_remote_contribution": True, "recent_sources": [],
        }})
        codes = {f.code for f in result.findings}
        self.assertIn("TRACK_COASTING", codes)
        self.assertIn("REMOTE_SENSOR_CONTRIBUTION", codes)
        self.assertEqual(result.status, "ok")

    def test_explain_track_remote_only_flags_recovery(self) -> None:
        result = explain_track({"track": {
            "track_id": "LOCAL-T2", "status": "confirmed", "hits": 5, "misses": 0,
            "local_updates": 0, "remote_updates": 5, "n_sources": 5,
            "freshness": 0.9, "sigma_position": {"x": 20.0, "y": 20.0, "z": 5.0},
            "source_sensors": ["SENSOR_REMOTE"], "platforms": ["REMOTE"],
            "has_remote_contribution": True, "recent_sources": [],
        }})
        codes = {f.code for f in result.findings}
        self.assertIn("COOPERATIVE_TRACK_RECOVERED", codes)

    def test_explain_track_missing_input_is_error(self) -> None:
        result = explain_track({})
        self.assertEqual(result.status, "error")

    def test_missing_covariance_not_reported_as_certainty(self) -> None:
        """没给协方差时不得输出「不确定度 0.0 m」——那等于宣称完全确定。"""
        result = explain_track({"track": {
            "track_id": "T-NOSIGMA", "status": "confirmed", "hits": 5, "misses": 0,
        }})
        codes = {f.code for f in result.findings}
        self.assertNotIn("TRACK_UNCERTAIN", codes)
        self.assertIn("未提供", result.summary)
        self.assertNotIn("0.0 m", result.summary)
        self.assertIn("TRACK_MAINTAINED", codes)

    def test_missing_freshness_not_reported_as_stale(self) -> None:
        """没给新鲜度时不得判成「数据陈旧」（缺失不等于 0）。"""
        result = explain_track({"track": {
            "track_id": "T-NOFRESH", "status": "confirmed", "hits": 5, "misses": 0,
            "sigma_position": {"x": 10.0, "y": 10.0, "z": 5.0},
        }})
        text = " ".join(f.title + f.message for f in result.findings)
        self.assertNotIn("陈旧", text)

    def test_stale_freshness_still_reported_when_provided(self) -> None:
        """真给了低新鲜度时仍要报出来（修复不能把真信号一起删掉）。"""
        result = explain_track({"track": {
            "track_id": "T-STALE", "status": "confirmed", "hits": 5, "misses": 0,
            "freshness": 0.1,
            "sigma_position": {"x": 10.0, "y": 10.0, "z": 5.0},
        }})
        text = " ".join(f.title + f.message for f in result.findings)
        self.assertIn("陈旧", text)

    def test_explain_cooperation_reports_structural_only(self) -> None:
        result = explain_cooperation({
            "communication": {"policy": "constrained_share", "policy_cn": "受限共享",
                              "n_links": 2, "delivery_rate": 0.83,
                              "latency_mean_s": 1.2, "latency_p95_s": 2.0,
                              "drop_reasons": {"lost": 12, "expired": 3}},
            "fusion": {"n_tracks_with_remote": 2, "n_tracks_local_only": 1,
                       "remote_measurements_arrived": 40,
                       "remote_measurements_used": 31,
                       "remote_measurements_rejected": 9,
                       "tracks_supported_remotely_only": 1},
        })
        codes = {f.code for f in result.findings}
        self.assertIn("COMM_PACKET_LOST", codes)
        self.assertIn("COMM_MESSAGE_EXPIRED", codes)
        self.assertIn("COOPERATIVE_TRACK_RECOVERED", codes)
        # 收益类真值指标不得出现
        text = " ".join(f.message for f in result.findings)
        self.assertNotIn("RMSE", text.replace("精度", ""))

    def test_explain_cooperation_empty_is_error(self) -> None:
        self.assertEqual(explain_cooperation({}).status, "error")


class TestEvidenceCheck(unittest.TestCase):
    def test_invented_code_and_number_flagged(self) -> None:
        context = {"latency_mean_s": 1.2, "sensors": ["SENSOR_A"]}
        check = validate_text_against_context(
            "延迟 1.2 s，SENSOR_A 正常，发现码 FOO_BAR_BAZ，误差 999.0 m", context
        )
        self.assertFalse(check.passed)
        self.assertTrue(any("FOO_BAR_BAZ" in v for v in check.violations))
        self.assertTrue(any("999" in v for v in check.violations))

    def test_real_identifiers_not_false_positive(self) -> None:
        """上下文里真实存在的标识符不得被判为编造（假阳性会导致误回退）。"""
        context = {"sensors": ["SENSOR_A", "SENSOR_B"], "n": 3}
        check = validate_text_against_context("SENSOR_A 与 SENSOR_B 均在线", context)
        self.assertTrue(check.passed, f"误判：{check.violations}")

    def test_grounded_text_passes(self) -> None:
        context = {"delivery_rate": 0.83, "n_links": 2}
        check = validate_text_against_context(
            "送达率 0.83，共 2 条链路", context
        )
        self.assertTrue(check.passed, f"误判：{check.violations}")

    def test_collect_helpers(self) -> None:
        payload = {"a": 1.5, "b": {"c": 2}, "d": [3, "SENSOR_X"]}
        self.assertIn(1.5, collect_numbers(payload))
        self.assertIn(3.0, collect_numbers(payload))
        self.assertIn("SENSOR_X", collect_identifiers(payload))


class TestServiceCompatibility(unittest.TestCase):
    """旧能力必须保持兼容。"""

    def test_routes_include_new_and_old(self) -> None:
        """ROUTES 是「能力名 -> 路径」，新旧能力都要在，且不能写反。"""
        from ai.api import REQUEST_SCHEMA, ROUTES

        expected = {
            "diagnose": "/api/diagnose",
            "explain_decision": "/api/explain_decision",
            "compare_policies": "/api/compare_policies",
            "generate_report": "/api/generate_report",
            "explain_track": "/api/explain_track",
            "explain_cooperation": "/api/explain_cooperation",
        }
        self.assertEqual(ROUTES, expected)
        for capability in expected:
            self.assertIn(capability, REQUEST_SCHEMA, f"{capability} 缺少入参说明")

    def test_dispatch_routes_both_new_capabilities(self) -> None:
        """客户端调用必须真的能落到服务方法上（不是只登记了名字）。"""
        from ai.api import LPI_API

        api = LPI_API()
        reply = api.dispatch("explain_track", {"track": {
            "track_id": "LOCAL-T1", "status": "coasting", "hits": 1, "misses": 2,
        }})
        self.assertNotEqual(reply.get("status"), "error", reply.get("error"))
        reply = api.dispatch("explain_cooperation", {"communication": {
            "policy": "constrained_share", "n_links": 1, "delivery_rate": 0.5,
        }})
        self.assertNotEqual(reply.get("status"), "error", reply.get("error"))

    def test_provider_capabilities_cover_new_methods(self) -> None:
        """每个登记在册的能力都必须真能被 provider 调用到（防"登记未接线"）。"""
        from ai.provider import CAPABILITIES, NullProvider
        from ai.rule_provider import RuleProvider

        for capability in CAPABILITIES:
            self.assertTrue(hasattr(NullProvider(), capability))
            self.assertTrue(callable(getattr(RuleProvider(), capability)))

    def test_service_has_new_methods(self) -> None:
        from ai.service import AIDiagnosisService

        self.assertTrue(hasattr(AIDiagnosisService, "explain_track"))
        self.assertTrue(hasattr(AIDiagnosisService, "explain_cooperation"))
        self.assertTrue(hasattr(AIDiagnosisService, "diagnose"))

    def test_diagnose_still_works(self) -> None:
        env = ec.make_env(observation_mode="full")
        env.reset(seed=42)
        env.step(6)
        snap = snapshot_from_simulator(env.sim, result=env.sim.results[-1], env=env)
        result = RuleProvider().diagnose(snap)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.findings)


class TestAiHasNoControl(unittest.TestCase):
    """AI 只诊断/解释，绝不控制雷达功率或融合器。"""

    def test_no_power_or_fusion_writes(self) -> None:
        import inspect

        import ai.rule_provider as rp
        import ai.track_explain as te

        for module in (rp, te):
            source = inspect.getsource(module)
            for forbidden in ("set_power", "apply_overrides", "tracker.update",
                              "tx_power_w =", "step("):
                self.assertNotIn(forbidden, source,
                                 f"{module.__name__} 出现疑似控制调用：{forbidden}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
