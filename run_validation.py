"""环境校核入口：分层检查 + 可信度报告 + 通信三组对照（v4.3）。

    python run_validation.py                          # 四层校核 + 生成可信度报告
    python run_validation.py --seeds 42 7 13          # 多种子
    python run_validation.py --communication          # 额外跑通信三组对照

输出（默认 output/validation/）：
    checks.json / checks.txt        四层校核结果
    trust_report.md / .json         环境可信度报告
    comm_comparison.csv             通信三组对照（--communication）
    message_log.csv                 逐消息通信日志（--communication）
    tracks_<策略>.csv               融合航迹（含溯源）（--communication）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()
import experiment_config as ec  # noqa: E402
from communication import (  # noqa: E402
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    SHARE_POLICY_CN,
    CommBus,
    CommConfig,
)
from fusion import FusionCenter, FusionConfig  # noqa: E402
from validation import (  # noqa: E402
    build_trust_report,
    format_checks,
    run_all_checks,
)

DEFAULT_OUT_DIR = "output/validation"

#: 通信对照用场景：双雷达 + 三目标 + 双侦察机 + 双干扰源
COMM_CONFIG = "config/multi_platform_scenario.json"

#: 通信三组对照的参数
COMM_CASES = (
    (SHARE_NONE, {}),
    (SHARE_IDEAL, {}),
    (SHARE_CONSTRAINED, {"base_delay_s": 1.0, "jitter_s": 0.3,
                         "loss_prob": 0.2, "expiry_s": 3.0}),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="环境校核与可信度报告（v4.3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    parser.add_argument("--measurement-steps", type=int, default=60)
    parser.add_argument("--fusion-steps", type=int, default=40)
    parser.add_argument("--communication", action="store_true",
                        help="额外跑「不共享 / 理想共享 / 受限共享」三组对照")
    parser.add_argument("--comm-steps", type=int, default=61)
    parser.add_argument("--quiet", action="store_true")
    return parser


# ----------------------------------------------------------------------


def run_communication_comparison(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """通信三组对照：不共享 / 理想零延迟共享 / 有延迟丢包共享。

    每个平台维护**自己的**融合中心；它只吃
    ①自己的本地测量、②通信总线送来的**已经到达**的别平台测量。
    因此"共享与否"的差异就体现在融合中心的输入里有没有别人的测量。

    ⚠️ 本阶段**不做多智能体联合控制**：功率控制仍由主雷达单独决策，
    通信与融合的影响体现在**状态估计质量**与由此产生的策略表现上。
    """
    rows: List[Dict[str, Any]] = []
    out_dir = os.path.join(args.out_dir, "communication")
    os.makedirs(out_dir, exist_ok=True)
    log_written = False

    for policy, overrides in COMM_CASES:
        label = SHARE_POLICY_CN.get(policy, policy)
        per_seed: List[Dict[str, Any]] = []

        for seed in args.seeds:
            # 用**双雷达**场景：只有在远端雷达能看到（部分）目标时，
            # 共享才真正带来信息增益。单雷达 + 单 ESM 场景里远端传感器是 ESM
            # （测的是雷达辐射源、不是目标），共享它对目标航迹毫无贡献
            # —— 那样测不出协同收益（实测三组完全相同）。
            env = ec.make_env(COMM_CONFIG, observation_mode="realistic")
            env.reset(seed=seed)
            if env.suite is None:
                continue

            platform_ids = ["RADAR_LOCAL", "RADAR_REMOTE"]
            sensors = list(env.suite.sensors)
            # 把传感器按"平台"划分：第 1 个归本平台，其余算远端（教学用简化划分）
            local_sensors, remote_sensors = sensors[:1], sensors[1:]
            own_ids = [s.sensor_id for s in local_sensors]

            bus = CommBus(platform_ids, CommConfig(
                policy=policy, message_size_bytes=128.0, seed=seed, **overrides
            ))
            center = FusionCenter("RADAR_LOCAL", FusionConfig(),
                                  own_sensor_ids=own_ids)
            sensor_positions = {
                s.sensor_id: env.sim.scene.by_id(s.config.mounting_id).position
                for s in sensors
            }

            track_counts: List[int] = []
            freshness: List[float] = []
            n_local_total = 0
            n_remote_total = 0
            steps_done = 0
            while steps_done < args.comm_steps:
                _o, _r, terminated, truncated, _i = env.step(6)
                now = env.sim.current_time
                report = env.suite_report()
                if report is None:
                    break

                # ① 本地测量（只取本平台传感器的）
                local_meas = [
                    m for sr in report.reports if sr.sensor_id in own_ids
                    for m in sr.detections + sr.held
                ]
                # ② 远端测量经通信发布
                remote_fresh = [
                    m for sr in report.reports if sr.sensor_id not in own_ids
                    for m in sr.detections
                ]
                if remote_fresh and bus.sharing_enabled:
                    bus.publish("RADAR_REMOTE",
                                remote_fresh[0].sensor_id, remote_fresh, now=now)

                # ③ 决策侧**只能**读已到达的消息
                arrived = bus.arrived_from("RADAR_LOCAL", now)
                shared: List[Any] = []
                for message in arrived:
                    payload = dict(message.payload)
                    payload["msg_id"] = message.msg_id
                    payload["platform_id"] = message.src_platform_id
                    shared.append(type("SharedMeasurement", (), payload)())

                snapshot = center.update(
                    local_meas + shared, now, sensor_positions,
                    remote_measurement_flags=[False] * len(local_meas)
                    + [True] * len(shared),
                )
                track_counts.append(snapshot.n_tracks)
                freshness.append(snapshot.freshness_stats()["mean"])
                n_local_total += snapshot.n_local_measurements
                n_remote_total += snapshot.n_remote_measurements
                steps_done += 1
                if terminated or truncated:
                    break

            stats = bus.statistics()
            per_seed.append({
                "steps": steps_done,
                "mean_tracks": (sum(track_counts) / len(track_counts))
                if track_counts else 0.0,
                "max_tracks": max(track_counts) if track_counts else 0,
                "frames_with_tracks": sum(1 for c in track_counts if c > 0),
                "mean_freshness": (sum(freshness) / len(freshness))
                if freshness else 0.0,
                "local_measurements": n_local_total,
                "remote_measurements": n_remote_total,
                "n_messages": stats["n_messages"],
                "n_delivered": stats["n_delivered"],
                "delivery_rate": stats["delivery_rate"],
                "latency_mean_s": stats["latency_mean_s"],
                "latency_p95_s": stats["latency_p95_s"],
                "drop_reasons": stats["drop_reasons"],
            })

            # 逐消息日志与航迹导出（只写第一份，三组共用一个文件会混淆）
            if not log_written:
                _write_csv(os.path.join(out_dir, "message_log.csv"),
                           bus.message_log_rows())
                log_written = True
            _write_csv(
                os.path.join(out_dir, f"tracks_{policy}.csv"),
                [t.to_dict(now=env.sim.current_time) for t in center.tracks],
            )

        row = _aggregate(per_seed, label, policy)
        rows.append(row)
        if not args.quiet:
            print(f"\n  {label}")
            print(f"    平均航迹数 {row['mean_tracks']:.3f}｜"
                  f"有航迹帧占比 {row['track_coverage']:.3f}｜"
                  f"平均新鲜度 {row['mean_freshness']:.4f}")
            print(f"    本地测量 {row['local_measurements']:.1f}｜"
                  f"共享测量 {row['remote_measurements']:.1f}｜"
                  f"送达率 {row['delivery_rate']:.4f}｜"
                  f"平均延迟 {row['latency_mean_s']:.4f}s")
    return rows


def _aggregate(per_seed: Sequence[Dict[str, Any]], label: str,
               policy: str) -> Dict[str, Any]:
    if not per_seed:
        return {"policy": policy, "label": label}

    def mean_of(key: str) -> float:
        values = [float(r[key]) for r in per_seed if key in r]
        return sum(values) / len(values) if values else 0.0

    steps_total = sum(int(r["steps"]) for r in per_seed)
    frames_with = sum(int(r["frames_with_tracks"]) for r in per_seed)
    return {
        "policy": policy,
        "label": label,
        "n_seeds": len(per_seed),
        "mean_tracks": mean_of("mean_tracks"),
        "max_tracks": max(int(r["max_tracks"]) for r in per_seed),
        "track_coverage": (frames_with / steps_total) if steps_total else 0.0,
        "mean_freshness": mean_of("mean_freshness"),
        "local_measurements": mean_of("local_measurements"),
        "remote_measurements": mean_of("remote_measurements"),
        "n_messages": mean_of("n_messages"),
        "delivery_rate": mean_of("delivery_rate"),
        "latency_mean_s": mean_of("latency_mean_s"),
        "latency_p95_s": mean_of("latency_p95_s"),
    }


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# ----------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("======== 环境校核与可信度报告（v4.3）========")
    print(f"配置      : {args.config}")
    print(f"种子      : {list(args.seeds)}")
    print("数据生成链：真实状态 → 可见性 → 测量 → 通信 → 融合 → 决策\n")

    report = run_all_checks(
        config_path=args.config,
        seeds=tuple(args.seeds),
        measurement_steps=args.measurement_steps,
        fusion_steps=args.fusion_steps,
    )

    print("======== 四层校核结果 ========")
    from validation.checks import CheckResult

    results = [CheckResult(**item) for item in report["checks"]]
    print(format_checks(results))

    print("\n======== 汇总 ========")
    for layer, info in report["summary"].items():
        status = "全部通过" if not info["failed"] else f"失败 {info['failed']}"
        print(f"  {layer:<14s} {info['passed']}/{info['total']}  {status}")
    print(f"\n  总体：{'全部通过' if report['all_passed'] else '存在失败项'}")

    # --- 可信度报告 ---
    trust = build_trust_report(report)
    paths = trust.write(args.out_dir)
    print(f"\n→ {paths['markdown']}")
    print(f"→ {paths['json']}")

    import json as _json

    with open(os.path.join(args.out_dir, "checks.json"), "w",
              encoding="utf-8") as handle:
        _json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "checks.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(format_checks(results))
    print(f"→ {os.path.join(args.out_dir, 'checks.json')}")

    print(f"\n已验证层：{trust.verified_modules or '无'}")
    if trust.partially_verified_modules:
        print(f"部分验证层：{trust.partially_verified_modules}"
              f"（失败项 {trust.failed_checks}）")

    # --- 通信三组对照 ---
    if args.communication:
        print("\n======== 通信三组对照（不共享 / 理想共享 / 受限共享）========")
        rows = run_communication_comparison(args)
        csv_path = os.path.join(args.out_dir, "communication", "comm_comparison.csv")
        _write_csv(csv_path, rows)
        print(f"\n{'策略':<28s}{'平均航迹':>10s}{'航迹覆盖':>10s}"
              f"{'新鲜度':>10s}{'共享测量':>10s}{'送达率':>10s}{'平均延迟':>10s}")
        for row in rows:
            print(f"{row['label']:<28s}{row['mean_tracks']:>10.3f}"
                  f"{row['track_coverage']:>10.3f}{row['mean_freshness']:>10.4f}"
                  f"{row['remote_measurements']:>10.1f}"
                  f"{row['delivery_rate']:>10.4f}{row['latency_mean_s']:>10.4f}")
        print(f"\n→ {csv_path}")
        print("→ " + os.path.join(args.out_dir, "communication", "message_log.csv"))
        print("\n读法：**「多平台」不等于完美共享**。")
        print("      理想共享是上界参考；实际收益取决于受限共享的延迟/丢包/过期，")
        print("      因此协同效果必须与通信条件一起解释。")


if __name__ == "__main__":
    main()
