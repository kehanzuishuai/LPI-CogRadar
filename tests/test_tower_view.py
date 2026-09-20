"""Tower View v1 只读 schema、回放确定性与映射一致性回归。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.run_global_track_acceptance import run_scenario
from tools.run_tower_view import _handler, generate_replays, replay_manifest
from tower_view import (
    CANONICAL_FLOAT_DECIMALS,
    TOWER_VIEW_SCENARIOS,
    TOWER_VIEW_SCHEMA_VERSION,
    build_replay,
    canonicalize,
)


def _contains_forbidden_truth(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "truth" in str(key).lower()
            or _contains_forbidden_truth(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_truth(item) for item in value)
    return False


class TestTowerViewReplay(unittest.TestCase):
    def test_four_formal_replays_are_truth_free_and_schema_complete(self) -> None:
        for scenario_id in TOWER_VIEW_SCENARIOS:
            replay = build_replay(scenario_id)
            self.assertEqual(replay["schema_version"], TOWER_VIEW_SCHEMA_VERSION)
            self.assertFalse(_contains_forbidden_truth(replay))
            self.assertEqual(len(replay["frames"]), replay["scenario"]["steps"])
            self.assertTrue(replay["summary"]["resource_conserved"])
            for frame in replay["frames"]:
                self.assertEqual(
                    set(("time_s", "radar_nodes", "local_tracks", "global_tracks",
                         "mappings", "recent_events")) - set(frame),
                    set(),
                )
                self.assertEqual(len(frame["radar_nodes"]), 2)

    def test_debug_overlay_is_explicit_opt_in(self) -> None:
        formal = build_replay("dual_radar_single_target")
        debug = build_replay(
            "dual_radar_single_target", debug_truth_overlay=True)
        self.assertNotIn("debug_overlay", formal)
        self.assertTrue(debug["debug_overlay"]["enabled"])
        self.assertTrue(any("debug_truth_tracks" in frame
                            for frame in debug["frames"]))

    def test_repeated_generation_is_byte_semantically_deterministic(self) -> None:
        first = build_replay("dual_radar_two_targets")
        second = build_replay("dual_radar_two_targets")
        self.assertEqual(first["frames_sha256"], second["frames_sha256"])
        self.assertEqual(first, second)

    def test_all_nested_floats_are_canonicalized_before_hashing(self) -> None:
        value = canonicalize({
            "distance_m": 1.234567891, "projection": [-0.00000001],
            "weights": {"NODE_B": 0.333333333333, "NODE_A": 0.666666666667},
        })
        self.assertEqual(value["distance_m"], 1.234568)
        self.assertEqual(value["projection"], [0.0])
        self.assertEqual(value["weights"], {"NODE_A": 0.666667, "NODE_B": 0.333333})

        def assert_canonical(nested: object) -> None:
            if isinstance(nested, float):
                self.assertEqual(nested, round(nested, CANONICAL_FLOAT_DECIMALS))
            elif isinstance(nested, dict):
                for item in nested.values():
                    assert_canonical(item)
            elif isinstance(nested, (list, tuple)):
                for item in nested:
                    assert_canonical(item)

        replay = build_replay("handover_disconnect_reconnect")
        assert_canonical(replay["frames"])
        self.assertEqual(
            replay["frames_sha256"],
            __import__("hashlib").sha256(
                json.dumps(replay["frames"], ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
        )

    def test_formal_replay_hash_is_stable_across_independent_subprocesses(self) -> None:
        code = (
            "from tower_view.replay import build_replay; "
            "print(build_replay('dual_radar_two_targets', seed=20260920)['frames_sha256'])"
        )
        root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        first = subprocess.check_output(
            [sys.executable, "-c", code], cwd=root, env=env, text=True,
        ).strip()
        second = subprocess.check_output(
            [sys.executable, "-c", code], cwd=root, env=env, text=True,
        ).strip()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_mapping_and_lifecycle_events_match_each_frame(self) -> None:
        for scenario_id in TOWER_VIEW_SCENARIOS:
            replay = build_replay(scenario_id)
            for frame in replay["frames"]:
                globals_now = {
                    row["global_track_id"] for row in frame["global_tracks"]
                }
                mapping = {
                    (row["source_node_id"], row["local_track_id"]):
                    row["global_track_id"]
                    for row in frame["mappings"]
                }
                self.assertTrue(set(mapping.values()).issubset(globals_now))
                for local in frame["local_tracks"]:
                    self.assertEqual(
                        local["mapped_global_track_id"],
                        mapping.get((local["source_node_id"],
                                     local["local_track_id"])),
                    )
                for event in frame["recent_events"]:
                    if (event.get("event") in ("ci_fused", "association")
                            and event.get("decision") != "rejected"):
                        self.assertIn(event.get("global_track_id"), globals_now)

    def test_crossing_negative_result_is_visible_not_sanitized(self) -> None:
        replay = build_replay("two_targets_crossing")
        note = replay["summary"]["frozen_evaluation_annotation"]
        self.assertEqual(note["global_id_switches"], 4)
        self.assertEqual(note["fragmentation"], 3)
        self.assertGreater(len(replay["summary"]["global_track_ids_observed"]), 2)
        self.assertGreater(replay["summary"]["mapping_reassignment_count"], 0)

    def test_opening_and_closing_view_does_not_change_simulation(self) -> None:
        before = run_scenario("dual_radar_single_target")
        with tempfile.TemporaryDirectory() as temporary:
            replay_dir = Path(temporary)
            generate_replays(
                replay_dir, ("handover_disconnect_reconnect",), seed=20260920,
                debug_truth_overlay=False,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(replay_dir))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/replays",
                    timeout=2.0,
                ) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)
        after = run_scenario("dual_radar_single_target")
        self.assertEqual(before, after)

    def test_manifest_only_lists_tower_schema_and_static_ui_has_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replay_dir = Path(temporary)
            paths = generate_replays(
                replay_dir, ("dual_radar_single_target",), seed=20260920,
                debug_truth_overlay=False,
            )
            generate_replays(
                replay_dir, ("dual_radar_two_targets",), seed=20260920,
                debug_truth_overlay=True,
            )
            (replay_dir / "not-a-replay.json").write_text(
                json.dumps({"schema_version": "other"}), encoding="utf-8")
            manifest = replay_manifest(replay_dir)
            debug_manifest = replay_manifest(replay_dir, allow_debug_replays=True)
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(len(debug_manifest), 2)
            self.assertEqual(manifest[0]["scenario_id"],
                             "dual_radar_single_target")

            default_server = ThreadingHTTPServer(
                ("127.0.0.1", 0), _handler(replay_dir))
            allowed_server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _handler(replay_dir, allow_debug_replays=True),
            )
            default_thread = threading.Thread(
                target=default_server.serve_forever, daemon=True)
            allowed_thread = threading.Thread(
                target=allowed_server.serve_forever, daemon=True)
            default_thread.start()
            allowed_thread.start()
            try:
                debug_url = "/replays/dual_radar_two_targets.debug.json"
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{default_server.server_port}{debug_url}",
                        timeout=2.0,
                    )
                self.assertEqual(rejected.exception.code, 403)
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{allowed_server.server_port}{debug_url}",
                    timeout=2.0,
                ) as response:
                    self.assertEqual(response.status, 200)
            finally:
                default_server.shutdown()
                allowed_server.shutdown()
                default_server.server_close()
                allowed_server.server_close()
                default_thread.join(timeout=2.0)
                allowed_thread.join(timeout=2.0)

        static_root = Path(__file__).resolve().parents[1] / "tower_view" / "static"
        html = (static_root / "index.html").read_text(encoding="utf-8")
        script = (static_root / "app.js").read_text(encoding="utf-8")
        for control in ("playButton", "prevButton", "nextButton", "timeline",
                        "trackList", "trackDetail", "towerCanvas"):
            self.assertIn(control, html)
        self.assertIn("mapped_global_track_id", script)
        self.assertIn("frozen_evaluation_annotation", script)
        self.assertIn("DEBUG / GROUND TRUTH", html)
        self.assertIn("debugBanner", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
