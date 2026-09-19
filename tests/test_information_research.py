"""信息新鲜度/不确定度单一研究分支的接口测试。"""

from __future__ import annotations

import ast
import os
import types
import unittest

from resource_management.information_research import (
    ABLATION_ARMS,
    AblationArm,
    CentralResearchFeatureAdapter,
    rule_config_for_arm,
)
from resource_management.model import NodeState, ResourceBudget
from resource_management.observation import (
    CentralObservation,
    NodeObservation,
    TrackObservation,
    node_observation_from_fusion,
    node_observation_from_payload,
    observation_payload,
    observation_truth_violations,
)
from resource_management.scheduling import RuleScheduler, task_urgency
from resource_management.tasks import QueueTaskKind, QueuedTask
from resource_management.units import BUDGET_UNITS, ResourceUnit
from tools.evaluate_information_research import load_contract


def _track(
    track_id: str,
    age: float,
    sigma: float,
    consistency: float | None,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        position=(1000.0, 100.0, 0.0),
        velocity=(20.0, 0.0, 0.0),
        sigma_position=(sigma, sigma, sigma),
        last_measurement_time_s=10.0 - age,
        last_fusion_time_s=10.0,
        information_age_s=age,
        coasting=False,
        n_sources=3,
        source_sensor_ids=("S",),
        platforms=("P",),
        local_updates=3,
        remote_updates=0,
        source_consistency_indicator=consistency,
        source_evidence_count=2 if consistency is not None else 0,
        normalized_residual_mean=(
            None if consistency is None else (1.0 / consistency) - 1.0
        ),
    )


def _node(*tracks: TrackObservation) -> NodeObservation:
    return NodeObservation(
        node_id="NODE_A",
        observed_at_s=9.0,
        tracks=list(tracks),
        track_valid_mask=[True] * len(tracks),
        capacity={unit.value: 10.0 for unit in BUDGET_UNITS},
        remaining={unit.value: 5.0 for unit in BUDGET_UNITS},
    )


def _central(node: NodeObservation) -> CentralObservation:
    return CentralObservation(
        schema_version="test",
        observed_at_s=10.0,
        nodes=[node],
        node_valid_mask=[True],
        node_information_age_s=[0.5],
        node_content_age_s=[1.0],
    )


class TestObservableSourceConsistency(unittest.TestCase):
    def test_indicator_uses_only_arrived_residual_evidence(self) -> None:
        sources = [
            types.SimpleNamespace(
                sensor_id="S1", residual_m=20.0, reported_sigma_m=10.0
            ),
            types.SimpleNamespace(
                sensor_id="S2", residual_m=10.0, reported_sigma_m=10.0
            ),
            types.SimpleNamespace(
                sensor_id="S3", residual_m=999.0, reported_sigma_m=0.0
            ),
        ]
        track = types.SimpleNamespace(
            track_id="TRK-1",
            position=types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
            velocity=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            sigma_position=types.SimpleNamespace(x=4.0, y=5.0, z=6.0),
            last_measurement_time=9.0,
            last_update_time=10.0,
            status="confirmed",
            sources=sources,
            platforms=["P1"],
            local_updates=2,
            remote_updates=0,
        )
        center = types.SimpleNamespace(tracks=[track])
        budget = ResourceBudget(capacity={unit: 10.0 for unit in BUDGET_UNITS})
        node = NodeState(node_id="NODE_A", budget=budget)
        observation = node_observation_from_fusion(center, node, 10.0)
        value = observation.tracks[0]
        self.assertAlmostEqual(value.normalized_residual_mean, 1.5)
        self.assertAlmostEqual(value.source_consistency_indicator, 0.4)
        self.assertEqual(value.source_evidence_count, 2)
        self.assertEqual(observation_truth_violations(observation.to_dict()), [])

    def test_indicator_survives_arrived_message_round_trip(self) -> None:
        original = _node(_track("T", 2.0, 80.0, 0.25))
        restored = node_observation_from_payload(
            observation_payload(original, include_research_extension=True),
            arrived_at_s=10.0
        )
        self.assertEqual(
            restored.tracks[0].source_consistency_indicator, 0.25
        )
        self.assertEqual(restored.tracks[0].source_evidence_count, 2)
        self.assertAlmostEqual(restored.tracks[0].normalized_residual_mean, 3.0)

    def test_extension_is_explicitly_opt_in(self) -> None:
        original = _node(_track("T", 1.0, 10.0, 0.25))
        base = observation_payload(original)
        extended = observation_payload(
            original, include_research_extension=True
        )
        self.assertNotIn("track_0_source_consistency", base)
        self.assertEqual(extended["track_0_source_consistency"], 0.25)


class TestFourArmLearningAdapter(unittest.TestCase):
    def test_all_arms_have_identical_structure_and_dimension(self) -> None:
        adapter = CentralResearchFeatureAdapter(["NODE_A"], tracks_per_node=2)
        observation = _central(_node(_track("T", 6.0, 300.0, 0.2)))
        names = adapter.feature_names()
        vectors = {arm: adapter.encode(observation, arm) for arm in ABLATION_ARMS}
        self.assertTrue(all(len(vector) == adapter.output_dim
                            for vector in vectors.values()))
        self.assertEqual(len(names), adapter.output_dim)
        self.assertEqual(len({tuple(names) for _arm in ABLATION_ARMS}), 1)

        age_indices = [index for index, name in enumerate(names)
                       if "age_norm" in name]
        sigma_indices = [index for index, name in enumerate(names)
                         if "sigma_position_norm" in name]
        consistency_indices = [index for index, name in enumerate(names)
                               if "source_inconsistency" in name]
        baseline = vectors[AblationArm.MAIN_BASELINE]
        fresh = vectors[AblationArm.FRESHNESS_ONLY]
        uncertain = vectors[AblationArm.UNCERTAINTY_ONLY]
        combined = vectors[AblationArm.FRESHNESS_UNCERTAINTY]
        self.assertTrue(all(baseline[index] == 0.0 for index in
                            age_indices + sigma_indices + consistency_indices))
        self.assertTrue(any(fresh[index] > 0.0 for index in age_indices))
        self.assertTrue(all(fresh[index] == 0.0 for index in
                            sigma_indices + consistency_indices))
        self.assertTrue(all(uncertain[index] == 0.0 for index in age_indices))
        self.assertTrue(any(uncertain[index] > 0.0 for index in
                            sigma_indices + consistency_indices))
        self.assertTrue(any(combined[index] > 0.0 for index in age_indices))
        self.assertTrue(any(combined[index] > 0.0 for index in
                            sigma_indices + consistency_indices))

    def test_rule_baseline_uses_the_same_four_gates(self) -> None:
        expected = {
            AblationArm.MAIN_BASELINE: (False, False, False),
            AblationArm.FRESHNESS_ONLY: (True, False, False),
            AblationArm.UNCERTAINTY_ONLY: (False, True, True),
            AblationArm.FRESHNESS_UNCERTAINTY: (True, True, True),
        }
        for arm, gates in expected.items():
            config = rule_config_for_arm(arm)
            self.assertEqual(
                (config.use_information_age,
                 config.use_estimate_uncertainty,
                 config.use_source_consistency),
                gates,
            )


class TestRuleAblationAttribution(unittest.TestCase):
    def _priority(self, arm: AblationArm, track: TrackObservation) -> float:
        task = QueuedTask(
            task_id="Q", kind=QueueTaskKind.ESTIMATE_UPDATE,
            node_id="NODE_A", release_time_s=10.0, targets=(track.track_id,),
        )
        urgency = task_urgency(task, _node(track))
        score, _reasons, evidence = RuleScheduler(
            rule_config_for_arm(arm)
        )._priority(task, urgency, 10.0)
        self.assertNotIn("probability", str(evidence).lower())
        return score

    def test_baseline_is_invariant_to_added_information(self) -> None:
        low = _track("T", 0.0, 10.0, 1.0)
        high = _track("T", 9.0, 500.0, 0.1)
        self.assertEqual(
            self._priority(AblationArm.MAIN_BASELINE, low),
            self._priority(AblationArm.MAIN_BASELINE, high),
        )

    def test_single_factor_arms_respond_only_to_their_factor(self) -> None:
        fresh_low = _track("T", 0.0, 200.0, 0.2)
        fresh_high = _track("T", 9.0, 200.0, 0.2)
        self.assertGreater(
            self._priority(AblationArm.FRESHNESS_ONLY, fresh_high),
            self._priority(AblationArm.FRESHNESS_ONLY, fresh_low),
        )
        uncertain_low = _track("T", 4.0, 10.0, 1.0)
        uncertain_high = _track("T", 4.0, 500.0, 0.1)
        self.assertGreater(
            self._priority(AblationArm.UNCERTAINTY_ONLY, uncertain_high),
            self._priority(AblationArm.UNCERTAINTY_ONLY, uncertain_low),
        )


class TestResearchContract(unittest.TestCase):
    def test_contract_digest_and_test_seal(self) -> None:
        contract = load_contract()
        self.assertFalse(contract["training_budget"]["test_release"])
        self.assertEqual(len(contract["ablation_arms"]), 4)
        self.assertEqual(contract["development_mechanism_check"]["training_episodes"], 0)

    def test_research_module_has_no_truth_or_perception_import(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "resource_management", "information_research.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), path)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"engine", "sensor", "fusion", "communication"})


if __name__ == "__main__":
    unittest.main()
