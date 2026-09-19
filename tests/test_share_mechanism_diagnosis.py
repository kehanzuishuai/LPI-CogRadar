"""回归 share 机制诊断本身，不读取 checkpoint 或 test-v3。"""
from __future__ import annotations

import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "share_mechanism_diagnosis",
    os.path.join(ROOT, "tools", "diagnose_share_mechanism.py"),
)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSIS)


class TestShareMechanismDiagnosis(unittest.TestCase):
    def test_forced_legal_share_is_real_comm_and_unblocks_receiver_process(self) -> None:
        idle = DIAGNOSIS._counterfactual(False)
        shared = DIAGNOSIS._counterfactual(True)
        self.assertTrue(idle["before"]["share_legal"])
        self.assertEqual(idle["after_decision"]["bus_messages"], 0)
        self.assertFalse(idle["after_decision"]["node_a_process_legal_next"])
        self.assertEqual(shared["after_decision"]["bus_messages"], 1)
        self.assertEqual(shared["after_decision"]["comm_bytes"], 128.0)
        self.assertTrue(shared["after_decision"]["node_a_process_legal_next"])
        self.assertEqual(shared["next_tick"]["node_a_process_event"]["n_remote_measurements"], 1)

    def test_minimal_scripted_case_has_causal_share_difference(self) -> None:
        no_share = DIAGNOSIS._minimal_scripted_case(False)
        shared = DIAGNOSIS._minimal_scripted_case(True)
        self.assertEqual(no_share["after_share"]["sent_comm_bytes"], 0.0)
        self.assertFalse(no_share["node_a_process_available"])
        self.assertIsNone(no_share["node_a_process_event"])
        self.assertEqual(shared["after_share"]["sent_comm_bytes"], 128.0)
        self.assertEqual(shared["after_share"]["accounted_comm_bytes"], 128.0)
        self.assertTrue(shared["node_a_process_available"])
        self.assertEqual(shared["node_a_process_event"]["n_remote_measurements"], 1)
        self.assertGreater(shared["node_a_after"]["remote_updates"], 0)
        for row in (no_share, shared):
            self.assertTrue(row["runtime_invariants"]["resource_conserved"])
            self.assertEqual(row["runtime_invariants"]["duplicate_runtime_tasks"], 0)
            self.assertEqual(row["runtime_invariants"]["truth_payload_violations"], 0)
