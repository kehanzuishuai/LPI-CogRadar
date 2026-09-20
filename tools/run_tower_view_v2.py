"""生成并启动 Tower View v2 全局航迹只读诊断台（仅 Python 标准库）。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower_view import (
    TOWER_VIEW_V2_MODES,
    TOWER_VIEW_V2_SCENARIOS,
    TOWER_VIEW_V2_SCHEMA_VERSION,
    write_replay_v2,
)


STATIC_DIR = PROJECT_ROOT / "tower_view" / "static_v2"
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "examples" / "tower_view_v2"


def generate_replays_v2(
    output_dir: Path,
    scenario_ids: Sequence[str],
    modes: Sequence[str],
    *,
    seed: int,
    debug_truth_overlay: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for scenario_id in scenario_ids:
        for mode in modes:
            debug_suffix = ".debug" if debug_truth_overlay else ""
            path = output_dir / f"{scenario_id}.{mode}{debug_suffix}.json"
            write_replay_v2(
                path, scenario_id, mode=mode, seed=seed,
                debug_truth_overlay=debug_truth_overlay,
            )
            paths.append(path)
    return paths


def _is_debug_replay(payload: MappingLike) -> bool:
    return bool(payload.get("debug_overlay", {}).get("enabled"))


MappingLike = Dict[str, Any]


def replay_manifest_v2(
    replay_dir: Path, *, allow_debug_replays: bool = False,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(replay_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("schema_version") != TOWER_VIEW_V2_SCHEMA_VERSION:
            continue
        is_debug = _is_debug_replay(payload)
        if is_debug and not allow_debug_replays:
            continue
        scenario = payload.get("scenario", {})
        entries.append({
            "file": path.name,
            "url": f"/replays/{path.name}",
            "scenario_id": scenario.get("scenario_id", path.stem),
            "title": scenario.get("title", path.stem),
            "seed": scenario.get("seed"),
            "sharing_mode": payload.get("sharing_mode"),
            "frame_count": len(payload.get("frames", [])),
            "comparison_metrics": payload.get("comparison_metrics", {}),
            "debug_overlay": is_debug,
        })
    return entries


def _handler_v2(replay_dir: Path, *, allow_debug_replays: bool = False):
    replay_root = replay_dir.resolve()

    class TowerViewV2Handler(BaseHTTPRequestHandler):
        server_version = "TowerViewV2/2.0"

        def log_message(self, format_string: str, *args: Any) -> None:
            sys.stderr.write("[tower-view-v2] " + (format_string % args) + "\n")

        def _send_bytes(self, body: bytes, content_type: str,
                        status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_bytes(b"not found\n", "text/plain; charset=utf-8", 404)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type += "; charset=utf-8"
            self._send_bytes(path.read_bytes(), content_type)

        def do_GET(self) -> None:  # noqa: N802
            route = unquote(urlparse(self.path).path)
            if route in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")
                return
            if route == "/styles.css":
                self._send_file(STATIC_DIR / "styles.css")
                return
            if route == "/app.js":
                self._send_file(STATIC_DIR / "app.js")
                return
            if route == "/api/replays":
                body = json.dumps(
                    replay_manifest_v2(
                        replay_root, allow_debug_replays=allow_debug_replays),
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return
            if route.startswith("/replays/"):
                candidate = (replay_root / route[len("/replays/"):]).resolve()
                try:
                    candidate.relative_to(replay_root)
                except ValueError:
                    self._send_bytes(b"forbidden\n", "text/plain; charset=utf-8", 403)
                    return
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = {}
                if _is_debug_replay(payload) and not allow_debug_replays:
                    self._send_bytes(
                        b"debug replay requires --allow-debug-replays\n",
                        "text/plain; charset=utf-8", 403,
                    )
                    return
                self._send_file(candidate)
                return
            self._send_bytes(b"not found\n", "text/plain; charset=utf-8", 404)

    return TowerViewV2Handler


def serve_v2(replay_dir: Path, host: str, port: int, open_browser: bool,
             *, allow_debug_replays: bool = False) -> None:
    entries = replay_manifest_v2(
        replay_dir, allow_debug_replays=allow_debug_replays)
    if not entries:
        raise RuntimeError(
            f"{replay_dir} 没有 tower-view-v2 回放；先运行 --generate-only")
    server = ThreadingHTTPServer(
        (host, port),
        _handler_v2(replay_dir, allow_debug_replays=allow_debug_replays),
    )
    url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{url_host}:{server.server_port}/"
    print(f"Tower View v2: {url}")
    print(f"Replay directory: {replay_dir.resolve()}")
    print("Ctrl+C 停止；服务和 UI 都不会向仿真对象回写。")
    if allow_debug_replays:
        print("WARNING: DEBUG / GROUND TRUTH replays are enabled for this server.")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTower View v2 stopped.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--scenario", action="append", choices=TOWER_VIEW_V2_SCENARIOS)
    parser.add_argument("--mode", action="append", choices=TOWER_VIEW_V2_MODES)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--debug-truth-overlay", action="store_true")
    parser.add_argument("--allow-debug-replays", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios: Tuple[str, ...] = tuple(args.scenario or TOWER_VIEW_V2_SCENARIOS)
    modes: Tuple[str, ...] = tuple(args.mode or TOWER_VIEW_V2_MODES)
    if args.generate_only or args.regenerate:
        for path in generate_replays_v2(
                args.replay_dir, scenarios, modes, seed=args.seed,
                debug_truth_overlay=args.debug_truth_overlay):
            print(path.resolve())
    if args.generate_only:
        return 0
    serve_v2(
        args.replay_dir, args.host, args.port, not args.no_browser,
        allow_debug_replays=args.allow_debug_replays,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
