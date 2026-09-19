"""多雷达协同感知验证：三类场景 × 三种共享策略（v4.4）。

三类协同验证场景（都是**程序化构造**，保证传感器作用距离真的覆盖目标）
-----------------------------------------------------------------------
| 场景 | 几何 | 要验证什么 |
| --- | --- | --- |
| **A 共同可见** | 两部雷达都能看到同一目标 | 共享是否**降低估计误差**（多视角融合） |
| **B 遮挡/视场受限** | 本地雷达被遮挡或目标在视场外，远端可见 | 共享是否**保持航迹连续** |
| **C 通信受限** | 同 B，但远端通信有延迟/丢包/过期 | 三种共享策略下**协同收益还剩多少** |

为什么场景必须程序化构造
------------------------
多平台场景当初是**为几何校核**建的，它的传感器作用距离由雷达方程在标称
18 W 下反解得到（约 4~6 km），而目标在 6~26 km —— 实测远端雷达
**30/30 判定全是 beyond_range**，一条测量都产生不了。
于是"共享"根本没有目标信息可传，三组航迹指标自然完全相同。
**那是场景配置问题，不是融合算法问题**，也不是靠改指标能解决的。

因此本脚本显式构造场景：把雷达摆在合适位置、把传感器作用距离设为
覆盖目标所需的值，并记录**为什么这么设**。

指标（真值**只用于离线评测**）
------------------------------
* 位置/速度 RMSE：把每条航迹按最近邻匹配到真值目标后计算（离线）
* 轨迹召回率：被至少一条航迹覆盖的真值目标帧占比
* 误关联率：航迹匹配到的真值目标发生变化的帧占比
* 丢轨率：真值目标仍存活但对应航迹被删除的次数
* 航迹连续性：最长连续覆盖帧 / 总帧数
* 远端测量利用率：到达的远端测量里真正进入航迹的比例（来自生命周期漏斗）

⚠️ 任何数字都**不得**回流入决策算法；本脚本属于离线评测通道。

用法
----
    python evaluate_cooperative_sensing.py
    python evaluate_cooperative_sensing.py --seeds 42 7 13 --out-dir output/cooperative
    python evaluate_cooperative_sensing.py --scenario A --diagnose   # 只看漏斗
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

import experiment_config as ec
from communication import (
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    SHARE_POLICY_CN,
    CommBus,
    CommConfig,
)
from engine.geometry import Vec3
from engine.simulator import Simulator
from fusion import FusionCenter, FusionConfig
from fusion.lifecycle import LifecycleLog
from sensor.config import build_suite_from_config

BASE_CONFIG = ec.CONFIG_PATH
DEFAULT_OUT_DIR = "output/cooperative"

#: 三个场景的共享策略
POLICIES = (SHARE_NONE, SHARE_IDEAL, SHARE_CONSTRAINED)

#: 受限共享的链路参数（场景 C 用）
CONSTRAINED_COMM = {"base_delay_s": 1.2, "jitter_s": 0.4, "loss_prob": 0.25,
                    "expiry_s": 2.5}


# ----------------------------------------------------------------------
# 场景构造
# ----------------------------------------------------------------------


def scenario_overrides(name: str) -> Dict[str, Any]:
    """返回某个场景的配置覆盖（场景构造的**唯一**来源）。

    `build_scenario`（造 Simulator 用）与 `run_case`（造 env 用）都调它，
    避免两处各写一份导致漂移。
    """
    if name == "A":
        return {
            "radars": [
                _radar("RADAR_LOCAL", 0.0, 0.0, heading_deg=90.0),
                _radar("RADAR_REMOTE", 15000.0, 0.0, heading_deg=270.0),
            ],
            "targets": [_target("TGT1", 7500.0, 0.0, rcs_m2=2.0)],
            "sensors": [
                _sensor("SENSOR_LOCAL", "RADAR_LOCAL", max_range_m=30000.0),
                _sensor("SENSOR_REMOTE", "RADAR_REMOTE", max_range_m=30000.0),
            ],
            "occluders": [],
        }
    if name in ("B", "C"):
        return {
            "radars": [
                _radar("RADAR_LOCAL", 0.0, 0.0, heading_deg=90.0),
                _radar("RADAR_REMOTE", 12000.0, 6000.0, heading_deg=200.0),
            ],
            "targets": [_target("TGT1", 7500.0, 0.0, rcs_m2=2.0)],
            "sensors": [
                _sensor("SENSOR_LOCAL", "RADAR_LOCAL", max_range_m=30000.0),
                _sensor("SENSOR_REMOTE", "RADAR_REMOTE", max_range_m=30000.0),
            ],
            "occluders": [
                # 球心在本地雷达到目标的视线中点、半径 1200 m：
                # 本地雷达完全看不到目标，远端雷达不受影响
                {"occluder_id": "BLOCK", "shape": "sphere",
                 "center_x": 3750.0, "center_y": 0.0, "center_z": 0.0,
                 "radius_m": 1200.0},
            ],
        }
    raise ValueError(f"未知场景 {name!r}，只支持 A / B / C")


def build_scenario(name: str, seed: int = 42) -> Tuple[Simulator, Dict[str, Any]]:
    """程序化构造场景，返回 (simulator, 场景说明)。

    所有场景都用同一套雷达/目标物理参数（不动物理），只改
    **几何布置**、**传感器作用距离**与**遮挡体**——这三者都属于"场景配置"。
    """
    sim = Simulator(BASE_CONFIG)
    sim.load_config()

    notes: Dict[str, Any] = {}

    sim.apply_overrides(extra=scenario_overrides(name))

    if name == "A":
        notes = {"idea": "两部雷达同时可见同一目标（分居目标两侧，各 7.5 km）",
                 "purpose": "验证共享是否降低估计误差（多视角融合）"}
    elif name == "B":
        notes = {"idea": "本地雷达到目标的视线被球形遮挡区切断，远端雷达可见",
                 "purpose": "验证共享是否保持航迹连续"}
    else:
        notes = {"idea": "几何同 B，远端通信存在延迟、丢包与过期",
                 "purpose": "比较三种共享策略下协同收益还剩多少",
                 "comm": dict(CONSTRAINED_COMM)}

    sim.reset(seed=seed)
    notes["scenario"] = name
    return sim, notes


def _radar(radar_id: str, x: float, y: float, heading_deg: float = 0.0) -> Dict[str, Any]:
    """构造一部雷达的配置段（物理参数沿用场景默认，只改位置与朝向）。"""
    return {
        "radar_id": radar_id, "x": x, "y": y, "z": 0.0,
        "tx_power_w": 18.0,
        "peak_gain_db": 30.0, "sidelobe_gain_db": 10.0, "main_beam_width_deg": 10.0,
        "wavelength_m": 0.1, "bandwidth_hz": 1.0e6,
        "noise_figure_db": 3.0, "system_loss_db": 3.0, "temperature_k": 290.0,
        "snr50_db": 6.0, "pd_slope_db": 2.0,
        "required_pd": 0.8, "energy_budget_j": 1400.0,
        "terminate_on_energy_exhausted": True,
        "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
        "heading_deg": heading_deg, "pitch_deg": 0.0, "roll_deg": 0.0,
        "timestamp_s": 0.0, "platform_id": radar_id,
        "freq_hz": 3.0e9, "is_active": True,
    }


def _target(target_id: str, x: float, y: float, rcs_m2: float = 2.0) -> Dict[str, Any]:
    return {
        "target_id": target_id, "x": x, "y": y, "z": 0.0, "rcs_m2": rcs_m2,
        "vx": -8.0, "vy": 0.0, "vz": 0.0,
        "heading_deg": 270.0, "pitch_deg": 0.0, "roll_deg": 0.0,
        "timestamp_s": 0.0, "platform_id": target_id, "is_active": True,
    }


def _sensor(sensor_id: str, mounting_id: str, max_range_m: float) -> Dict[str, Any]:
    """构造一个雷达传感器配置段。

    作用距离显式给到 30 km：**这是为了覆盖目标距离而设的**，
    不是为了让结果好看。原默认值（约 4~6 km）来自雷达方程在标称
    18 W 下反解，比目标距离小一个量级，属于明显的配置错误。
    """
    return {
        "sensor_id": sensor_id, "mounting_id": mounting_id, "sensor_kind": "radar",
        "max_range_m": max_range_m, "min_range_m": 0.0,
        "az_fov_deg": 60.0, "el_fov_deg": 30.0, "update_period_s": 1.0,
        "range_sigma_rel": 0.01, "range_sigma_abs_m": 5.0,
        "az_sigma_deg": 0.5, "el_sigma_deg": 0.5, "range_rate_sigma_mps": 1.0,
        "snr50_db": 6.0, "pd_slope_db": 2.0, "false_alarm_rate": 0.0,
        "tx_power_w": 18.0, "peak_gain_db": 30.0, "wavelength_m": 0.1,
        "bandwidth_hz": 1.0e6, "noise_figure_db": 3.0, "system_loss_db": 3.0,
        "temperature_k": 290.0, "observes_kind": "target", "provides_range": True,
        "seed": 42,
    }


# ----------------------------------------------------------------------
# 单次运行
# ----------------------------------------------------------------------


class SharedMeasurement:
    """从通信消息解包出的测量对象（供融合中心消费）。

    刻意用一个轻量容器而不是复用 `MeasurementRecord`：
    接收方拿到的是**消息载荷**，不是原始测量对象，
    这正是"跨平台共享"与"本地自用"的区别所在。
    """

    __slots__ = ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                 "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                 "std_az_deg", "std_el_deg", "confidence", "msg_id", "platform_id",
                 "is_false_alarm", "truth_id", "truth_range_m")

    def __init__(self, payload: Dict[str, Any], msg_id: str = "",
                 platform_id: str = "") -> None:
        for key in ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                    "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                    "std_az_deg", "std_el_deg", "confidence"):
            setattr(self, key, payload.get(key))
        self.msg_id = msg_id
        self.platform_id = platform_id
        # 共享测量不含真值：这两个字段始终为 None（接口一致性保留）
        self.is_false_alarm = False
        self.truth_id = None
        self.truth_range_m = None


def run_case(
    scenario: str,
    policy: str,
    seed: int,
    steps: int = 50,
    track_truth: bool = True,
) -> Dict[str, Any]:
    """跑一个 (场景, 共享策略, 种子) 组合，返回指标与诊断。"""
    _probe, notes = build_scenario(scenario, seed)
    env = ec.make_env_for_seed(
        seed, config_path=BASE_CONFIG, jitter=False,
        observation_mode="realistic", measurement_max_tracks=4,
    )
    env.sim.apply_overrides(extra=scenario_overrides(scenario))
    # ⚠️ 套件是在构造 env 时按**原配置**建的（传感器 mounting_id 指向 RADAR1），
    # 换掉雷达后必须**重建**，否则 mount_pose 找不到平台直接抛 SceneError。
    env.suite = build_suite_from_config(
        env.sim.scene, env.sim._raw_config or {}, seed=seed, noise_scale=1.0
    )
    env.reset(seed=seed)
    sim = env.sim

    if env.suite is None:
        raise RuntimeError("realistic 模式应建立传感器套件")

    # 平台划分：本地 = 第一个传感器所在平台，远端 = 其余雷达传感器
    radar_sensors = [s for s in env.suite.sensors if s.sensor_kind == "radar"]
    if len(radar_sensors) < 2:
        raise RuntimeError(f"场景 {scenario} 需要至少两个雷达传感器")
    local_sensor, remote_sensors = radar_sensors[0], radar_sensors[1:]
    own_ids = {local_sensor.sensor_id}
    remote_ids = {s.sensor_id for s in remote_sensors}

    sensor_positions = {
        s.sensor_id: env.sim.scene.by_id(s.config.mounting_id).position
        for s in env.suite.sensors
    }

    comm_overrides = CONSTRAINED_COMM if policy == SHARE_CONSTRAINED else {}
    bus = CommBus(["LOCAL", "REMOTE"], CommConfig(
        policy=policy, message_size_bytes=128.0, seed=seed, **comm_overrides
    ))
    lifecycle = LifecycleLog(enabled=True)
    center = FusionCenter("LOCAL", FusionConfig(), own_sensor_ids=own_ids,
                          lifecycle=lifecycle)

    frames: List[Dict[str, Any]] = []
    steps_done = 0
    while steps_done < steps:
        _o, _r, terminated, truncated, _i = env.step(6)
        now = env.sim.current_time
        report = env.suite_report()
        if report is None:
            break

        # ① 本地测量
        local_meas = [m for sr in report.reports if sr.sensor_id in own_ids
                      for m in sr.detections + sr.held]
        # ② 远端测量发布到总线
        remote_fresh = [m for sr in report.reports if sr.sensor_id in remote_ids
                        for m in sr.detections]
        if remote_fresh and bus.sharing_enabled:
            bus.publish("REMOTE", remote_fresh[0].sensor_id, remote_fresh, now=now)

        # ③ 决策侧只读**已到达**的消息
        # 用 consume（一次投递）而不是 arrived（纯查询）：
        # 后者会把历史消息重复投递，导致远端测量堆积、大量被判过期。
        shared = [SharedMeasurement(dict(m.payload), msg_id=m.msg_id,
                                    platform_id=m.src_platform_id)
                  for m in bus.consume("LOCAL", now)]

        snapshot = center.update(
            local_meas + shared, now, sensor_positions,
            remote_measurement_flags=[False] * len(local_meas) + [True] * len(shared),
        )

        frame: Dict[str, Any] = {
            "time_s": now,
            "n_local": snapshot.n_local_measurements,
            "n_remote": snapshot.n_remote_measurements,
            "n_kind_rejected": snapshot.n_kind_rejected,
            "n_stale_rejected": snapshot.n_stale_rejected,
            "n_tracks": snapshot.n_tracks,
            "n_confirmed": snapshot.n_confirmed,
            "freshness_mean": snapshot.freshness_stats()["mean"],
            "tracks": [(t.track_id, t.position, t.velocity,
                        t.local_updates, t.remote_updates, t.status)
                       for t in snapshot.tracks],
            "truth": [],
        }
        if track_truth:
            frame["truth"] = [(t.target_id, t.position, t.velocity,
                               t.is_active)
                              for t in sim.targets]
        frames.append(frame)
        steps_done += 1
        if terminated or truncated:
            break

    metrics = _compute_metrics(frames)
    funnel = lifecycle.funnel()
    comm_stats = bus.statistics()
    metrics.update({
        "scenario": scenario,
        "policy": policy,
        "policy_cn": SHARE_POLICY_CN.get(policy, policy),
        "seed": seed,
        "steps": steps_done,
        "remote_measurements_received": sum(f["n_remote"] for f in frames),
        "local_measurements": sum(f["n_local"] for f in frames),
        "kind_rejected": sum(f["n_kind_rejected"] for f in frames),
        "stale_rejected": sum(f["n_stale_rejected"] for f in frames),
        "remote_utilization": funnel["remote"]["utilization"],
        "local_utilization": funnel["local"]["utilization"],
        "gate_rejected": center.stats["gate_rejected"],
        "tracks_initiated": center.stats["initiated"],
        "tracks_dropped": center.stats["dropped"],
        "n_messages": comm_stats["n_messages"],
        "delivery_rate": comm_stats["delivery_rate"],
        "latency_mean_s": comm_stats["latency_mean_s"],
        "latency_p95_s": comm_stats["latency_p95_s"],
    })
    return {"metrics": metrics, "frames": frames, "funnel": funnel,
            "lifecycle": lifecycle, "notes": notes, "center": center}


# ----------------------------------------------------------------------


def _compute_metrics(frames: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """离线评测指标（**唯一**读真值的地方）。"""
    if not frames:
        return {"position_rmse_m": 0.0, "velocity_rmse_mps": 0.0,
                "track_recall": 0.0, "track_coverage": 0.0,
                "continuity": 0.0, "misassociation_rate": 0.0,
                "track_breaks": 0, "longest_run": 0}

    pos_sq: List[float] = []
    vel_sq: List[float] = []
    matched_truth: Dict[str, str] = {}   # track_id -> 真值目标 ID（用于误关联率）
    misassoc_frames = 0
    assoc_frames = 0
    covered_frames = 0
    truth_target_frames = 0
    recall_hits = 0
    run = 0
    longest = 0
    prev_track_ids: set = set()
    breaks = 0

    for frame in frames:
        truth = frame.get("truth", [])
        active_truth = [t for t in truth if t[3]]
        truth_target_frames += len(active_truth)
        tracks = frame["tracks"]
        if tracks:
            covered_frames += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0

        current_ids = {t[0] for t in tracks}
        # 丢轨：上一帧有、这一帧没了，而真值目标仍存活
        if active_truth and prev_track_ids and not prev_track_ids & current_ids:
            breaks += 1
        prev_track_ids = current_ids

        for track_id, position, velocity, _lu, _ru, _status in tracks:
            if not active_truth:
                continue
            # 最近邻匹配到真值目标（离线评测用）
            best = min(active_truth,
                       key=lambda t: (position - t[1]).norm())
            distance = (position - best[1]).norm()
            pos_sq.append(distance ** 2)
            vel_sq.append((velocity - best[2]).norm() ** 2)
            assoc_frames += 1
            previous = matched_truth.get(track_id)
            if previous is not None and previous != best[0]:
                misassoc_frames += 1
            matched_truth[track_id] = best[0]

        # 轨迹召回率：某个真值目标被至少一条航迹匹配到
        matched = {matched_truth.get(t[0]) for t in tracks}
        recall_hits += sum(1 for t in active_truth if t[0] in matched)

    n_frames = len(frames)
    return {
        "position_rmse_m": math.sqrt(sum(pos_sq) / len(pos_sq)) if pos_sq else 0.0,
        "velocity_rmse_mps": math.sqrt(sum(vel_sq) / len(vel_sq)) if vel_sq else 0.0,
        "track_recall": (recall_hits / truth_target_frames) if truth_target_frames else 0.0,
        "track_coverage": covered_frames / n_frames,
        "continuity": longest / n_frames,
        "longest_run": longest,
        "misassociation_rate": (misassoc_frames / assoc_frames) if assoc_frames else 0.0,
        "track_breaks": breaks,
    }


# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多雷达协同感知验证（三场景 × 三共享策略）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--diagnose", action="store_true",
                        help="打印测量生命周期漏斗（回答「远端测量在哪里被丢弃」）")
    parser.add_argument("--quiet", action="store_true")
    return parser


METRIC_COLUMNS = (
    ("position_rmse_m", "位置RMSE(m)", 2),
    ("velocity_rmse_mps", "速度RMSE(m/s)", 3),
    ("track_recall", "轨迹召回", 4),
    ("track_coverage", "航迹覆盖", 4),
    ("continuity", "连续性", 4),
    ("misassociation_rate", "误关联率", 4),
    ("track_breaks", "轨迹断裂", 2),
    ("remote_utilization", "远端利用率", 4),
    ("delivery_rate", "送达率", 4),
    ("latency_mean_s", "平均延迟(s)", 3),
)


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = ["A", "B", "C"] if args.scenario == "all" else [args.scenario]

    print("======== 多雷达协同感知验证（v4.4）========")
    print("场景：A 共同可见｜B 遮挡/视场受限｜C 通信受限")
    print("策略：不共享｜理想零延迟共享｜受限共享")
    print(f"种子：{list(args.seeds)}   步数：{args.steps}\n")

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for scenario in scenarios:
        _sim, notes = build_scenario(scenario, args.seeds[0])
        print(f"---- 场景 {scenario}：{notes['idea']} ----")
        print(f"     目的：{notes['purpose']}")

        for policy in POLICIES:
            per_seed: List[Dict[str, Any]] = []
            last: Optional[Dict[str, Any]] = None
            for seed in args.seeds:
                result = run_case(scenario, policy, seed, steps=args.steps)
                per_seed.append(result["metrics"])
                last = result
            row = _aggregate(per_seed, scenario, policy)
            rows.append(row)
            if last is not None:
                details[f"{scenario}_{policy}"] = {
                    "funnel": last["funnel"],
                    "frames": len(last["frames"]),
                    "tracks_dropped": last["metrics"]["tracks_dropped"],
                    "gate_rejected": last["metrics"]["gate_rejected"],
                }
            if not args.quiet:
                print(f"     {row['policy_cn']:<34s}"
                      f" 位置RMSE={row['position_rmse_m']:>8.2f}m"
                      f" 覆盖={row['track_coverage']:>6.3f}"
                      f" 连续性={row['continuity']:>6.3f}"
                      f" 断裂={row['track_breaks']:>5.1f}"
                      f" 远端利用率={row['remote_utilization']:>6.3f}")
        print()

    # ---------------- 汇总表 ----------------
    print("======== 汇总 ========")
    header = f"{'场景':<5s}{'策略':<34s}" + "".join(
        f"{title:>15s}" for _key, title, _p in METRIC_COLUMNS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['scenario']:<5s}{row['policy_cn']:<34s}"
        for key, _title, places in METRIC_COLUMNS:
            line += f"{row.get(key, 0.0):>15.{places}f}"
        print(line)

    # ---------------- 协同收益结论 ----------------
    print("\n======== 协同收益判定 ========")
    conclusions = _judge(rows)
    for line in conclusions:
        print(line)

    # ---------------- 诊断漏斗 ----------------
    if args.diagnose:
        print("\n======== 测量生命周期漏斗（诊断）========")
        for key, detail in details.items():
            funnel = detail["funnel"]
            print(f"\n[{key}]")
            print(f"  追踪总数 {funnel['total']}｜"
                  f"本地 {funnel['local']['total']}｜远端 {funnel['remote']['total']}")
            print(f"  远端进入航迹 {funnel['remote']['in_track']}"
                  f"（利用率 {funnel['remote']['utilization']:.4f}）")
            if funnel["by_reject_reason"]:
                print("  拒绝原因：")
                for reason, count in sorted(funnel["by_reject_reason"].items(),
                                            key=lambda kv: -kv[1]):
                    print(f"    {reason:<40s} {count}")

    # ---------------- 导出 ----------------
    csv_path = os.path.join(args.out_dir, "cooperative_comparison.csv")
    _write_csv(csv_path, rows)
    print(f"\n→ {csv_path}")

    json_path = os.path.join(args.out_dir, "cooperative_details.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "details": details,
                   "conclusions": conclusions}, handle,
                  ensure_ascii=False, indent=2)
    print(f"→ {json_path}")

    # 生命周期明细（第一个场景的不共享与理想共享各导一份，便于对比）
    for scenario in scenarios[:1]:
        for policy in (SHARE_NONE, SHARE_IDEAL):
            result = run_case(scenario, policy, args.seeds[0], steps=args.steps)
            path = os.path.join(args.out_dir, f"lifecycle_{scenario}_{policy}.csv")
            result["lifecycle"].write_csv(path)
            print(f"→ {path}")

    html_path = _write_html(os.path.join(args.out_dir, "cooperative_report.html"),
                            rows, details, conclusions)
    print(f"→ {html_path}")


def _aggregate(per_seed: Sequence[Dict[str, Any]], scenario: str,
               policy: str) -> Dict[str, Any]:
    def mean_of(key: str) -> float:
        values = [float(r.get(key, 0.0)) for r in per_seed]
        return sum(values) / len(values) if values else 0.0

    row: Dict[str, Any] = {
        "scenario": scenario, "policy": policy,
        "policy_cn": SHARE_POLICY_CN.get(policy, policy),
        "n_seeds": len(per_seed),
    }
    for key, _title, _p in METRIC_COLUMNS:
        row[key] = mean_of(key)
    row["remote_measurements_received"] = mean_of("remote_measurements_received")
    row["local_measurements"] = mean_of("local_measurements")
    row["kind_rejected"] = mean_of("kind_rejected")
    row["gate_rejected"] = mean_of("gate_rejected")
    row["n_messages"] = mean_of("n_messages")
    return row


def _judge(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """按"只有确实观察到改善才能声称收益"的纪律给结论。"""
    by_key = {(r["scenario"], r["policy"]): r for r in rows}
    lines: List[str] = []
    for scenario in ("A", "B", "C"):
        none_row = by_key.get((scenario, SHARE_NONE))
        ideal_row = by_key.get((scenario, SHARE_IDEAL))
        constrained_row = by_key.get((scenario, SHARE_CONSTRAINED))
        if not (none_row and ideal_row):
            continue
        lines.append(f"\n【场景 {scenario}】")
        rmse_gain = none_row["position_rmse_m"] - ideal_row["position_rmse_m"]
        cov_gain = ideal_row["track_coverage"] - none_row["track_coverage"]
        cont_gain = ideal_row["continuity"] - none_row["continuity"]

        # ⚠️ 措辞必须区分"根本没有航迹"与"有航迹但误差大"。
        # 无航迹时 RMSE 记为 0（没有样本），若直接比大小会得出
        # "0 → 84.58 m = 无改善"这种与事实相反的结论。
        # 场景 B/C 的不共享组正是这种情况（本地雷达被遮挡 → 零航迹）。
        if none_row["track_coverage"] <= 0.01 and ideal_row["track_coverage"] > 0.01:
            lines.append(
                f"  ⚠️ 不共享组**完全没有航迹**（覆盖 {none_row['track_coverage']:.3f}，"
                f"位置RMSE 无样本记 0）。此时 RMSE 的数值比较**没有意义**，"
                f"应看航迹覆盖：{none_row['track_coverage']:.3f} → "
                f"{ideal_row['track_coverage']:.3f}（Δ {cov_gain:+.3f}）；"
                f"连续性 Δ {cont_gain:+.3f}"
            )
        else:
            lines.append(
                f"  不共享 → 理想共享：位置RMSE {none_row['position_rmse_m']:.2f} → "
                f"{ideal_row['position_rmse_m']:.2f} m（Δ {rmse_gain:+.2f}，"
                f"{'改善' if rmse_gain > 0 else '无改善'}）；"
                f"航迹覆盖 {none_row['track_coverage']:.3f} → "
                f"{ideal_row['track_coverage']:.3f}（Δ {cov_gain:+.3f}）；"
                f"连续性 Δ {cont_gain:+.3f}"
            )
        if rmse_gain > 1.0 or cov_gain > 0.05 or cont_gain > 0.05:
            lines.append("  ⇒ 观察到**共享带来的改善**（见上）。")
        else:
            lines.append("  ⇒ **未观察到可分辨的改善**。不得声称本场景存在协同收益；"
                         "应结合下面的漏斗诊断说明原因。")
        if constrained_row is not None:
            gap = ((constrained_row["position_rmse_m"] - ideal_row["position_rmse_m"]) /
                   max(ideal_row["position_rmse_m"], 1e-9))
            lines.append(
                f"  理想共享 → 受限共享：位置RMSE {ideal_row['position_rmse_m']:.2f} → "
                f"{constrained_row['position_rmse_m']:.2f} m（+{gap * 100:.1f}%）；"
                f"送达率 {constrained_row['delivery_rate']:.4f}，"
                f"平均延迟 {constrained_row['latency_mean_s']:.3f}s"
            )
    lines.append("\n⚠️ 以上全部指标使用**离线真值**计算，只用于评测，"
                 "不参与任何决策。")
    lines.append("⚠️ 「多平台」不等于完美共享：受限共享的收益必须与"
                 "延迟/丢包/过期条件一起解释。")
    return lines


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


def _write_html(path: str, rows: Sequence[Dict[str, Any]],
                details: Dict[str, Any], conclusions: Sequence[str]) -> str:
    import html as html_mod

    def table(rows_: Sequence[Dict[str, Any]]) -> str:
        head = ("<tr><th>场景</th><th>策略</th>" + "".join(
            f"<th>{html_mod.escape(t)}</th>" for _k, t, _p in METRIC_COLUMNS
        ) + "</tr>")
        body = ""
        for row in rows_:
            cells = "".join(
                f"<td>{row.get(k, 0.0):.{p}f}</td>" for k, _t, p in METRIC_COLUMNS
            )
            body += (f"<tr><td>{html_mod.escape(str(row['scenario']))}</td>"
                     f"<td>{html_mod.escape(str(row['policy_cn']))}</td>{cells}</tr>")
        return f"<table>{head}{body}</table>"

    funnel_blocks = ""
    for key, detail in details.items():
        funnel = detail["funnel"]
        reject = "".join(
            f"<li>{html_mod.escape(r)}: {c}</li>"
            for r, c in sorted(funnel["by_reject_reason"].items(), key=lambda kv: -kv[1])
        )
        funnel_blocks += (
            f"<h3>{html_mod.escape(key)}</h3>"
            f"<p>追踪 {funnel['total']}｜远端 {funnel['remote']['total']}｜"
            f"远端进入航迹 {funnel['remote']['in_track']}"
            f"（利用率 {funnel['remote']['utilization']:.4f}）</p>"
            f"<ul>{reject}</ul>"
        )

    text = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>LPI-CogRadar v4.4 多雷达协同感知验证</title>
<style>
body{{font-family:'Microsoft YaHei',Consolas,monospace;margin:24px;background:#fafafa}}
table{{border-collapse:collapse;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
th{{background:#4472c4;color:#fff}} td:first-child,td:nth-child(2){{text-align:left}}
pre{{background:#fff;border:1px solid #ddd;padding:12px;white-space:pre-wrap}}
h2{{border-left:4px solid #4472c4;padding-left:8px;margin-top:28px}}
</style></head><body>
<h1>LPI-CogRadar v4.4 — 多雷达协同感知验证</h1>
<h2>1. 三类场景</h2>
<p>A 共同可见（验证估计误差是否下降）｜B 遮挡/视场受限（验证航迹连续性）｜
C 通信受限（延迟/丢包/过期）</p>
<h2>2. 指标对照</h2>
{table(rows)}
<h2>3. 协同收益判定</h2>
<pre>{html_mod.escape(chr(10).join(conclusions))}</pre>
<h2>4. 测量生命周期漏斗（远端测量在哪里被丢弃）</h2>
{funnel_blocks}
<h2>5. 纪律声明</h2>
<pre>所有指标使用离线真值计算，只用于评测，不参与决策。
只有在遮挡或通信受限场景下确实观察到航迹连续性或估计精度改善，
才能声称协同感知产生收益；否则如实记录负结果。
「多平台」不等于完美共享。</pre>
</body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


if __name__ == "__main__":
    main()
