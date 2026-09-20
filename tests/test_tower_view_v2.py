"""Tower View v2 只读诊断、对照切换、确定性与真值隔离回归。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.run_global_track_acceptance import run_scenario
from tools.run_tower_view_v2 import (
    _handler_v2,
    generate_replays_v2,
    replay_manifest_v2,
)
from tower_view import (
    TOWER_VIEW_V2_MODES,
    TOWER_VIEW_V2_SCENARIOS,
    TOWER_VIEW_V2_SCHEMA_VERSION,
    build_replay_v2,
)


REQUIRED_SCENARIOS = (
    "dual_radar_single_target",
    "dual_radar_two_targets",
    "handover_disconnect_reconnect",
    "link_outage_loss_recovery",
    "asynchronous_radar_refresh",
    "two_targets_crossing",
)


def _contains_forbidden_truth_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "truth" in str(key).lower()
            or _contains_forbidden_truth_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_truth_key(item) for item in value)
    return False


class TestTowerViewV2Replay(unittest.TestCase):
    def test_required_scenarios_generate_formal_truth_free_replays(self) -> None:
        self.assertTrue(set(REQUIRED_SCENARIOS).issubset(TOWER_VIEW_V2_SCENARIOS))
        for scenario_id in REQUIRED_SCENARIOS:
            replay = build_replay_v2(scenario_id)
            self.assertEqual(replay["schema_version"], TOWER_VIEW_V2_SCHEMA_VERSION)
            self.assertFalse(_contains_forbidden_truth_key(replay))
            self.assertFalse(replay["read_only_contract"]["runtime_mutation"])
            self.assertFalse(
                replay["read_only_contract"]["association_parameters_changed"])
            self.assertTrue(replay["summary"]["resource_conserved"])
            self.assertEqual(len(replay["frames"]), replay["scenario"]["steps"])
            for frame in replay["frames"]:
                self.assertIn("communication", frame)
                self.assertIn("fusion", frame)
                self.assertIn("events", frame)

    def test_four_modes_share_schema_and_preserve_empty_comparators(self) -> None:
        rows = {
            mode: build_replay_v2("dual_radar_single_target", mode=mode)
            for mode in TOWER_VIEW_V2_MODES
        }
        self.assertEqual(set(rows), {
            "no_share", "measurement_share", "track_share",
            "event_triggered_track_share",
        })
        self.assertEqual(rows["no_share"]["comparison_metrics"]["communication_bytes"], 0.0)
        self.assertEqual(rows["no_share"]["summary"]["global_track_ids_observed"], [])
        self.assertGreater(
            rows["track_share"]["comparison_metrics"]["communication_bytes"], 0.0)
        for mode, replay in rows.items():
            self.assertEqual(replay["sharing_mode"], mode)
            self.assertEqual(replay, build_replay_v2(
                "dual_radar_single_target", mode=mode))

    def test_lifecycle_and_evidence_match_frame_audit_counts(self) -> None:
        replay = build_replay_v2("link_outage_loss_recovery")
        ci_events = [row for row in replay["events"]
                     if row["event_type"] == "CI_FUSED"]
        self.assertEqual(
            len(ci_events), replay["frames"][-1]["fusion"]["ci_count"])
        evidence_ids = {row["message_id"] for row in replay["message_evidence"]}
        for event in replay["events"]:
            if event.get("message_id"):
                self.assertIn(event["message_id"], evidence_ids)
        for lifecycle in replay["track_lifecycle"]:
            global_id = lifecycle["global_track_id"]
            observed = {
                track["global_track_id"]
                for frame in replay["frames"] for track in frame["global_tracks"]
            }
            self.assertIn(global_id, observed)
            for state in lifecycle["states"]:
                self.assertIn("time_s", state)

    def test_full_event_vocabulary_is_visible_across_frozen_scenarios(self) -> None:
        scenario_ids = (
            "dual_radar_single_target", "two_targets_crossing",
            "radar_reconnect_new_local_id", "leave_drop_reenter",
            "link_outage_loss_recovery",
        )
        observed = {
            event["event_type"]
            for scenario_id in scenario_ids
            for event in build_replay_v2(scenario_id)["events"]
        }
        required = {
            "GLOBAL_TRACK_CREATED", "ASSOCIATED", "CI_FUSED", "HANDOVER",
            "RECONNECTED", "COASTING", "STALE_REJECTED", "DROPPED",
            "ID_SWITCH", "FRAGMENTATION",
        }
        self.assertTrue(required.issubset(observed), required - observed)

    def test_crossing_negative_result_is_not_hidden(self) -> None:
        replay = build_replay_v2("two_targets_crossing")
        metrics = replay["comparison_metrics"]
        self.assertGreater(metrics["id_switch"], 0)
        self.assertGreater(metrics["fragmentation"], 0)
        event_types = {row["event_type"] for row in replay["events"]}
        self.assertIn("ID_SWITCH", event_types)
        self.assertIn("FRAGMENTATION", event_types)

    def test_debug_overlay_remains_explicit(self) -> None:
        formal = build_replay_v2("dual_radar_single_target")
        debug = build_replay_v2(
            "dual_radar_single_target", debug_truth_overlay=True)
        self.assertNotIn("debug_overlay", formal)
        self.assertTrue(debug["debug_overlay"]["enabled"])
        self.assertTrue(any("debug_truth_tracks" in frame
                            for frame in debug["frames"]))

    def test_subprocess_hash_is_deterministic(self) -> None:
        code = (
            "from tower_view.replay_v2 import build_replay_v2; "
            "print(build_replay_v2('asynchronous_radar_refresh')['replay_sha256'])"
        )
        root = str(Path(__file__).resolve().parents[1])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        first = subprocess.check_output(
            [sys.executable, "-c", code], cwd=root, env=environment, text=True).strip()
        second = subprocess.check_output(
            [sys.executable, "-c", code], cwd=root, env=environment, text=True).strip()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_manifest_mode_switch_and_server_are_read_only(self) -> None:
        before = run_scenario("dual_radar_single_target")
        with tempfile.TemporaryDirectory() as temporary:
            replay_dir = Path(temporary)
            paths = generate_replays_v2(
                replay_dir, ("dual_radar_single_target",),
                TOWER_VIEW_V2_MODES, seed=20260920,
                debug_truth_overlay=False,
            )
            debug_path = generate_replays_v2(
                replay_dir, ("dual_radar_two_targets",), ("track_share",),
                seed=20260920, debug_truth_overlay=True,
            )[0]
            hashes_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
            manifest = replay_manifest_v2(replay_dir)
            allowed = replay_manifest_v2(
                replay_dir, allow_debug_replays=True)
            self.assertEqual(len(manifest), 4)
            self.assertEqual(len(allowed), 5)
            self.assertEqual({row["sharing_mode"] for row in manifest},
                             set(TOWER_VIEW_V2_MODES))

            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), _handler_v2(replay_dir))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for row in manifest:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}{row['url']}",
                        timeout=3.0,
                    ) as response:
                        self.assertEqual(response.status, 200)
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}/replays/{debug_path.name}",
                        timeout=3.0,
                    )
                self.assertEqual(rejected.exception.code, 403)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3.0)
            hashes_after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
            self.assertEqual(hashes_before, hashes_after)
        after = run_scenario("dual_radar_single_target")
        self.assertEqual(before, after)

    def test_static_ui_has_diagnostics_switching_and_exports(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tower_view" / "static_v2"
        html = (root / "index.html").read_text(encoding="utf-8")
        script = (root / "app.js").read_text(encoding="utf-8")
        for control in (
            "scenarioSelect", "modeSelect", "towerCanvas", "trackList",
            "trackDetail", "eventMarkers", "eventStrip", "telemetryGrid",
            "exportPng", "exportJson", "exportHtml", "debugBanner",
        ):
            self.assertIn(control, html)
        for token in (
            "message_evidence", "track_lifecycle", "comparison_metrics",
            "canvas.toBlob", "JSON.stringify(replay", "诊断报告",
            "DEBUG / GROUND TRUTH",
        ):
            self.assertIn(token, script if token != "DEBUG / GROUND TRUTH" else html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
