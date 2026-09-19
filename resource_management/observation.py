"""融合结果 → 资源调度的**只读适配层**（v4.5）。

要解决的断点（核对记录 E2）
---------------------------
"融合器输出存在" ≠ "调度器已经真正使用了它"。此前：
`FusionCenter` 的航迹只喂 AI 诊断与离线评测；RL 观测走的是
`sensor.fusion.fuse_measurements` 的无状态航迹表。两条链没有打通。

本模块提供**正式适配层**，把 `FusionCenter` 的航迹输出翻译成调度器
可以只读消费的观测。

三条硬约束（由构造方式保证，不靠约定）
--------------------------------------
1. **调度器不持有真值 Simulator**。本模块的函数签名只接受
   `FusionCenter` / `NodeState` / 已到达的消息，**不接受** `Simulator`、`Scene`
   或任何真值对象。模块本身也不 import `engine.simulator` / `engine.scene`
   （由测试钉住）。
2. **中央只能读已到达的数据**。`CentralObservationStore.ingest` 拒绝
   未到达的消息（`arrived_at is None` 或 `arrived_at > now`）；
   没到达就保持缺失——**绝不从全局对象补齐**。
   断开远端消息后，远端节点的观测随之"冻结"、信息年龄增长，
   而不是继续反映远端的新状态。
3. **变长 + 有效掩码**。航迹数与节点数都是变的，观测用列表 + 掩码表达；
   旧 DQN 需要定长输入时走**独立适配器**（`FixedLengthAdapter`），
   它有自己的 `schema_version`，且**拒绝**维度不匹配的 checkpoint——
   不允许"截取前几维"或"硬塞进旧维度"。

字段元数据
----------
每个字段都在 `FIELD_SPECS` 里登记 **单位 / 坐标系 / 可见范围 / 来源**。
没有登记的字段不得出现在观测里（`assert_fields_documented` 会拦）。
"可见范围"三档：

| visibility | 含义 |
| --- | --- |
| `local` | 只有本节点知道（自己的资源余量） |
| `shared` | 必须经通信到达才可见（远端航迹） |
| `derived` | 由前两类推导（信息年龄、外推标志） |

坐标系一律 **ENU**（x=东, y=北, z=天），与 `engine.geometry` 一致；
速度单位为 m/s；时间单位为秒（相对仿真起点）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from communication.message import (
    MESSAGE_KIND_NODE_OBSERVATION,
    MeasurementMessage,
)
from resource_management.units import BUDGET_UNITS, ResourceUnit

#: **本适配层独立 schema 版本**。与旧观测路径（full/pomdp/ideal/realistic）
#: 完全分开：它们的维度与语义是冻结的，本层改版不影响它们。
SCHEMA_VERSION = "rm-obs-1.0"

#: 旧路径标记（保留并可复现，但**不得**与本层混用）
LEGACY_OBSERVATION_MODES: Tuple[str, ...] = ("full", "pomdp", "ideal",
                                             "realistic")

#: 可见范围
VISIBILITY_LOCAL = "local"
VISIBILITY_SHARED = "shared"
VISIBILITY_DERIVED = "derived"

#: 来源
PROVENANCE_LOCAL_FUSION = "local_fusion"
PROVENANCE_LOCAL_RESOURCE = "local_resource"
PROVENANCE_REMOTE_MESSAGE = "remote_message"
PROVENANCE_DERIVED = "derived"

#: 禁止出现在观测里的键前缀（真值通道）——与通信层同一套纪律
FORBIDDEN_OBSERVATION_PREFIXES: Tuple[str, ...] = (
    "truth", "err_", "is_false_alarm", "matched_truth",
)


@dataclass(frozen=True)
class FieldSpec:
    """一个观测字段的元数据。

    | 项 | 说明 |
    | --- | --- |
    | `unit` | 单位（`m` / `m/s` / `s` / `m^2` / `1` / `count`） |
    | `frame` | 坐标系（`ENU` / `none`） |
    | `visibility` | `local` / `shared` / `derived` |
    | `provenance` | 该值的来源 |
    """

    unit: str
    frame: str
    visibility: str
    provenance: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"unit": self.unit, "frame": self.frame,
                "visibility": self.visibility, "provenance": self.provenance,
                "note": self.note}


def _spec(unit: str, frame: str, visibility: str, provenance: str,
          note: str = "") -> FieldSpec:
    return FieldSpec(unit, frame, visibility, provenance, note)


#: **字段元数据登记表**。未登记的字段不得出现在观测里。
FIELD_SPECS: Dict[str, FieldSpec] = {
    # --- 航迹级 ---
    "track_id": _spec("1", "none", VISIBILITY_LOCAL, PROVENANCE_LOCAL_FUSION,
                      "融合中心分配的**稳定**航迹 ID（单调递增、不随排序变化）"),
    "position": _spec("m", "ENU", VISIBILITY_LOCAL,
                      PROVENANCE_LOCAL_FUSION, "估计位置"),
    "velocity": _spec("m/s", "ENU", VISIBILITY_LOCAL,
                      PROVENANCE_LOCAL_FUSION, "估计速度"),
    "sigma_position": _spec("m", "ENU", VISIBILITY_LOCAL,
                            PROVENANCE_LOCAL_FUSION, "位置标准差（对角）"),
    "last_measurement_time_s": _spec(
        "s", "none", VISIBILITY_LOCAL, PROVENANCE_LOCAL_FUSION,
        "最后一次**测量**的时刻（消息延迟大时它显著落后于当前时刻）"),
    "last_fusion_time_s": _spec(
        "s", "none", VISIBILITY_LOCAL, PROVENANCE_LOCAL_FUSION,
        "最后一次**融合更新**的时刻"),
    "information_age_s": _spec(
        "s", "none", VISIBILITY_DERIVED, PROVENANCE_DERIVED,
        "当前时刻 − 最后测量时刻；航迹被外推时它会持续增大"),
    "coasting": _spec("1", "none", VISIBILITY_LOCAL, PROVENANCE_LOCAL_FUSION,
                      "是否处于**预测保持**（外推）状态"),
    "n_sources": _spec("count", "none", VISIBILITY_LOCAL,
                       PROVENANCE_LOCAL_FUSION, "溯源记录条数"),
    "source_sensor_ids": _spec("1", "none", VISIBILITY_LOCAL,
                               PROVENANCE_LOCAL_FUSION, "支持该航迹的传感器集合"),
    "platforms": _spec("1", "none", VISIBILITY_LOCAL,
                       PROVENANCE_LOCAL_FUSION, "支持该航迹的平台集合"),
    "local_updates": _spec("count", "none", VISIBILITY_LOCAL,
                           PROVENANCE_LOCAL_FUSION, "本地测量更新次数"),
    "remote_updates": _spec("count", "none", VISIBILITY_LOCAL,
                            PROVENANCE_LOCAL_FUSION, "远端共享更新次数"),
    # --- 节点级（本地资源）---
    "capacity": _spec("resource", "none", VISIBILITY_LOCAL,
                      PROVENANCE_LOCAL_RESOURCE, "预算容量（单位见 units.py）"),
    "remaining": _spec("resource", "none", VISIBILITY_LOCAL,
                       PROVENANCE_LOCAL_RESOURCE, "预算余量"),
    "available": _spec("1", "none", VISIBILITY_LOCAL,
                       PROVENANCE_LOCAL_RESOURCE, "节点是否可用"),
    "update_period_s": _spec("s", "none", VISIBILITY_LOCAL,
                             PROVENANCE_LOCAL_RESOURCE, "采样更新周期"),
    # --- 通信 ---
    "n_messages_arrived": _spec("count", "none", VISIBILITY_LOCAL,
                                PROVENANCE_LOCAL_RESOURCE,
                                "**已到达**本节点的消息条数"),
    "n_messages_inflight": _spec(
        "count", "none", VISIBILITY_LOCAL, PROVENANCE_LOCAL_RESOURCE,
        "在途消息条数（只报数量，**不看内容**）"),
    # --- 中央视角 ---
    "arrived_at_s": _spec("s", "none", VISIBILITY_SHARED,
                          PROVENANCE_REMOTE_MESSAGE, "该节点摘要的到达时刻"),
    "node_information_age_s": _spec(
        "s", "none", VISIBILITY_DERIVED, PROVENANCE_DERIVED,
        "**到达年龄**：现在 − 摘要到达时刻。刚送达的摘要为 0，"
        "断开消息后它会持续增大"),
    "node_content_age_s": _spec(
        "s", "none", VISIBILITY_DERIVED, PROVENANCE_DERIVED,
        "**内容年龄**：现在 − 摘要**生成**时刻。通信延迟不会让到达年龄变大"
        "（送达瞬间仍为 0），但会让内容年龄正好等于链路延迟；"
        "调度器判「数据还新不新」应当看这一项"),
    "data_available": _spec("1", "none", VISIBILITY_DERIVED,
                            PROVENANCE_DERIVED,
                            "该节点的摘要是否**已经到达**过"),
    "missing": _spec("1", "none", VISIBILITY_DERIVED, PROVENANCE_DERIVED,
                     "缺失原因说明（没到达 / 从未收到 / 航迹为空）"),
    "observed_at_s": _spec("s", "none", VISIBILITY_LOCAL,
                           PROVENANCE_LOCAL_FUSION, "该节点观测自身的生成时刻"),
}


def field_metadata() -> Dict[str, Dict[str, Any]]:
    """字段元数据的可序列化视图（写进观测、供调度器与文档对照）。"""
    return {name: spec.to_dict() for name, spec in FIELD_SPECS.items()}


def assert_fields_documented(names: Iterable[str]) -> None:
    """未登记的字段一律拒绝——防止新增字段时忘记标注单位/坐标系/来源。"""
    undocumented = [name for name in names if name not in FIELD_SPECS]
    if undocumented:
        raise KeyError(
            f"观测字段未登记元数据：{undocumented}。"
            "每个字段都必须标注 单位/坐标系/可见范围/来源。"
        )


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class TrackObservation:
    """一条航迹的**只读**观测（融合中心输出）。"""

    track_id: str
    position: Tuple[float, float, float]      # ENU, m
    velocity: Tuple[float, float, float]      # ENU, m/s
    sigma_position: Tuple[float, float, float]  # m
    last_measurement_time_s: Optional[float]  # s
    last_fusion_time_s: Optional[float]       # s
    information_age_s: float                  # s
    coasting: bool
    n_sources: int
    source_sensor_ids: Tuple[str, ...]
    platforms: Tuple[str, ...]
    local_updates: int
    remote_updates: int

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "track_id": self.track_id,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "sigma_position": list(self.sigma_position),
            "last_measurement_time_s": self.last_measurement_time_s,
            "last_fusion_time_s": self.last_fusion_time_s,
            "information_age_s": round(self.information_age_s, 6),
            "coasting": self.coasting,
            "n_sources": self.n_sources,
            "source_sensor_ids": list(self.source_sensor_ids),
            "platforms": list(self.platforms),
            "local_updates": self.local_updates,
            "remote_updates": self.remote_updates,
        }
        assert_fields_documented(payload)
        return payload


@dataclass
class NodeObservation:
    """一个节点的只读观测：本地航迹 + 本地资源余量 + 已到达的通信状态。"""

    node_id: str
    observed_at_s: float
    tracks: List[TrackObservation] = field(default_factory=list)
    #: 与 `tracks` 等长的有效掩码（航迹列表是**变长**的）
    track_valid_mask: List[bool] = field(default_factory=list)
    available: bool = True
    update_period_s: float = 1.0
    capacity: Dict[str, float] = field(default_factory=dict)
    remaining: Dict[str, float] = field(default_factory=dict)
    n_messages_arrived: int = 0
    n_messages_inflight: int = 0
    missing: List[str] = field(default_factory=list)

    def track_ids(self) -> List[str]:
        """只有**有效**航迹的 ID —— 任务队列只能从这些 ID 建任务。"""
        return [track.track_id for track, valid
                in zip(self.tracks, self.track_valid_mask) if valid]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "observed_at_s": round(self.observed_at_s, 6),
            "tracks": [track.to_dict() for track in self.tracks],
            "track_valid_mask": list(self.track_valid_mask),
            "n_tracks": len(self.tracks),
            "available": self.available,
            "update_period_s": self.update_period_s,
            "capacity": dict(self.capacity),
            "remaining": dict(self.remaining),
            "n_messages_arrived": self.n_messages_arrived,
            "n_messages_inflight": self.n_messages_inflight,
            "missing": list(self.missing),
        }
        return payload


@dataclass
class CentralObservation:
    """中央调度器的只读视图：**只含已经到达的**节点摘要。"""

    schema_version: str
    observed_at_s: float
    nodes: List[NodeObservation] = field(default_factory=list)
    #: 与 `nodes` 等长的有效掩码：False = 该节点**从未**送达过摘要
    node_valid_mask: List[bool] = field(default_factory=list)
    #: 逐节点的摘要年龄（未到达的节点也在这里，值为 None 或很大）
    node_information_age_s: List[Optional[float]] = field(default_factory=list)
    #: 逐节点的**内容年龄** = now − 该摘要的生成时刻（None = 该节点从未送达）
    #:
    #: 与 `node_information_age_s`（到达年龄）的区别很重要：通信延迟 2s 时，
    #: 摘要送达的那一 tick 到达年龄是 0（"刚收到"），但它携带的是 2s 前的
    #: 信息，内容年龄为 2。第一版只记到达年龄，于是"通信延迟"这条机制在
    #: 指标上**完全不可见**（两个基线都记 0.000）。
    node_content_age_s: List[Optional[float]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    n_messages_ingested: int = 0

    def node_ids(self) -> List[str]:
        return [node.node_id for node in self.nodes]

    def node(self, node_id: str) -> Optional[NodeObservation]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def all_track_ids(self) -> List[str]:
        """中央能看到的全部有效航迹 ID（**不含**未到达节点的任何航迹）。"""
        out: List[str] = []
        for node, valid in zip(self.nodes, self.node_valid_mask):
            if not valid:
                continue
            out.extend(node.track_ids())
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observed_at_s": round(self.observed_at_s, 6),
            "nodes": [node.to_dict() for node in self.nodes],
            "node_valid_mask": list(self.node_valid_mask),
            "node_information_age_s": list(self.node_information_age_s),
            "node_content_age_s": list(self.node_content_age_s),
            "n_nodes": len(self.nodes),
            "missing": list(self.missing),
            "n_messages_ingested": self.n_messages_ingested,
            "field_metadata": field_metadata(),
            "legacy_observation_modes": list(LEGACY_OBSERVATION_MODES),
        }


# ----------------------------------------------------------------------
# 从融合中心构造节点观测
# ----------------------------------------------------------------------


def node_observation_from_fusion(
    center: Any,
    node_state: Any,
    now_s: float,
    n_messages_arrived: int = 0,
    n_messages_inflight: int = 0,
) -> NodeObservation:
    """把 `FusionCenter` 的航迹 + `NodeState` 的资源翻成只读观测。

    ⚠️ 只读两样东西：`center.tracks`（融合输出）与 `node_state.budget`
    （本地资源）。**不读**真值场景、不读 `truth_id`、不读未到达的消息。
    """
    tracks: List[TrackObservation] = []
    mask: List[bool] = []
    for track in getattr(center, "tracks", []) or []:
        position = track.position
        velocity = track.velocity
        sigma = track.sigma_position
        age = (max(0.0, now_s - track.last_measurement_time)
               if track.last_measurement_time is not None else float("inf"))
        tracks.append(TrackObservation(
            track_id=str(track.track_id),
            position=(float(position.x), float(position.y), float(position.z)),
            velocity=(float(velocity.x), float(velocity.y), float(velocity.z)),
            sigma_position=(float(sigma.x), float(sigma.y), float(sigma.z)),
            last_measurement_time_s=(None if track.last_measurement_time is None
                                     else float(track.last_measurement_time)),
            last_fusion_time_s=(None if track.last_update_time is None
                                else float(track.last_update_time)),
            information_age_s=age,
            coasting=(str(getattr(track, "status", "")) == "coasting"),
            n_sources=len(getattr(track, "sources", []) or []),
            source_sensor_ids=tuple(sorted({
                str(source.sensor_id) for source in
                (getattr(track, "sources", []) or [])
            })),
            platforms=tuple(sorted(set(getattr(track, "platforms", []) or []))),
            local_updates=int(getattr(track, "local_updates", 0)),
            remote_updates=int(getattr(track, "remote_updates", 0)),
        ))
        # 有效掩码：只有位置/时间戳齐全的航迹才可被调度器使用
        mask.append(
            track.last_measurement_time is not None
            and all(abs(value) != float("inf") for value in
                    (position.x, position.y, position.z))
        )

    budget = getattr(node_state, "budget", None)
    capacity = {unit.value: float(budget.capacity.get(unit, 0.0))
                for unit in BUDGET_UNITS} if budget is not None else {}
    remaining = {unit.value: float(budget.remaining(unit))
                 for unit in BUDGET_UNITS} if budget is not None else {}

    missing: List[str] = []
    if not tracks:
        missing.append("no_track_local")
    if budget is None:
        missing.append("no_resource_budget")

    return NodeObservation(
        node_id=str(node_state.node_id),
        observed_at_s=float(now_s),
        tracks=tracks,
        track_valid_mask=mask,
        available=bool(getattr(node_state, "available", True)),
        update_period_s=float(getattr(node_state, "update_period_s", 1.0)),
        capacity=capacity,
        remaining=remaining,
        n_messages_arrived=int(n_messages_arrived),
        n_messages_inflight=int(n_messages_inflight),
        missing=missing,
    )


# ----------------------------------------------------------------------
# 节点观测 → 通信消息（带类型的白名单载荷）
# ----------------------------------------------------------------------


def observation_payload(observation: NodeObservation) -> Dict[str, Any]:
    """把节点观测摊平成**登记过的**扁平载荷（见通信层的按类型白名单）。"""
    payload: Dict[str, Any] = {
        "node_id": observation.node_id,
        "observed_at_s": round(observation.observed_at_s, 6),
        "n_tracks": len(observation.tracks),
        "n_messages_arrived": observation.n_messages_arrived,
        "n_messages_inflight": observation.n_messages_inflight,
        "missing_note": "|".join(observation.missing),
    }
    for unit in BUDGET_UNITS:
        payload[f"capacity_{unit.value}"] = observation.capacity.get(unit.value, 0.0)
        payload[f"remaining_{unit.value}"] = observation.remaining.get(unit.value, 0.0)
    for index, (track, valid) in enumerate(zip(observation.tracks,
                                               observation.track_valid_mask)):
        if not valid:
            continue
        payload[f"track_{index}_id"] = track.track_id
        payload[f"track_{index}_x"] = track.position[0]
        payload[f"track_{index}_y"] = track.position[1]
        payload[f"track_{index}_z"] = track.position[2]
        payload[f"track_{index}_vx"] = track.velocity[0]
        payload[f"track_{index}_vy"] = track.velocity[1]
        payload[f"track_{index}_vz"] = track.velocity[2]
        payload[f"track_{index}_sigma_x"] = track.sigma_position[0]
        payload[f"track_{index}_sigma_y"] = track.sigma_position[1]
        payload[f"track_{index}_sigma_z"] = track.sigma_position[2]
        payload[f"track_{index}_last_meas_s"] = (
            -1.0 if track.last_measurement_time_s is None
            else track.last_measurement_time_s)
        payload[f"track_{index}_last_fusion_s"] = (
            -1.0 if track.last_fusion_time_s is None
            else track.last_fusion_time_s)
        payload[f"track_{index}_age_s"] = (
            1e9 if track.information_age_s == float("inf")
            else track.information_age_s)
        payload[f"track_{index}_coasting"] = bool(track.coasting)
        payload[f"track_{index}_n_sources"] = track.n_sources
        payload[f"track_{index}_sensors"] = "|".join(track.source_sensor_ids)
        payload[f"track_{index}_platforms"] = "|".join(track.platforms)
    return payload


def node_observation_from_payload(payload: Dict[str, Any],
                                  arrived_at_s: Optional[float]) -> NodeObservation:
    """把到达的载荷还原成节点观测（**只用于已到达的消息**）。"""
    n_tracks = int(payload.get("n_tracks", 0) or 0)
    tracks: List[TrackObservation] = []
    mask: List[bool] = []
    for index in range(n_tracks):
        track_id = payload.get(f"track_{index}_id")
        if not track_id:
            continue
        last_meas = float(payload.get(f"track_{index}_last_meas_s", -1.0))
        last_fusion = float(payload.get(f"track_{index}_last_fusion_s", -1.0))
        age = float(payload.get(f"track_{index}_age_s", 1e9))
        tracks.append(TrackObservation(
            track_id=str(track_id),
            position=(float(payload[f"track_{index}_x"]),
                      float(payload[f"track_{index}_y"]),
                      float(payload[f"track_{index}_z"])),
            velocity=(float(payload[f"track_{index}_vx"]),
                      float(payload[f"track_{index}_vy"]),
                      float(payload[f"track_{index}_vz"])),
            sigma_position=(float(payload[f"track_{index}_sigma_x"]),
                            float(payload[f"track_{index}_sigma_y"]),
                            float(payload[f"track_{index}_sigma_z"])),
            last_measurement_time_s=None if last_meas < 0.0 else last_meas,
            last_fusion_time_s=None if last_fusion < 0.0 else last_fusion,
            information_age_s=(float("inf") if age >= 1e9 else age),
            coasting=bool(payload.get(f"track_{index}_coasting", False)),
            n_sources=int(payload.get(f"track_{index}_n_sources", 0) or 0),
            source_sensor_ids=tuple(filter(None, str(
                payload.get(f"track_{index}_sensors", "")).split("|"))),
            platforms=tuple(filter(None, str(
                payload.get(f"track_{index}_platforms", "")).split("|"))),
            local_updates=0,
            remote_updates=0,
        ))
        mask.append(True)

    observation = NodeObservation(
        node_id=str(payload.get("node_id", "")),
        observed_at_s=float(payload.get("observed_at_s", 0.0) or 0.0),
        tracks=tracks,
        track_valid_mask=mask,
        n_messages_arrived=int(payload.get("n_messages_arrived", 0) or 0),
        n_messages_inflight=int(payload.get("n_messages_inflight", 0) or 0),
        capacity={unit.value: float(payload.get(f"capacity_{unit.value}", 0.0) or 0.0)
                  for unit in BUDGET_UNITS},
        remaining={unit.value: float(payload.get(f"remaining_{unit.value}", 0.0) or 0.0)
                   for unit in BUDGET_UNITS},
        missing=[part for part in str(payload.get("missing_note", "")).split("|")
                 if part],
    )
    return observation


def publish_node_observation(observation: NodeObservation, bus: Any,
                             src_platform_id: str, now_s: float,
                             dst_platform_ids: Optional[Sequence[str]] = None
                             ) -> List[MeasurementMessage]:
    """把节点观测经**通信总线**发出（因此中央只能读到已到达的那些）。"""
    message = MeasurementMessage(
        msg_id=f"OBS-{observation.node_id}-{int(round(now_s * 1000)):08d}",
        src_platform_id=src_platform_id,
        src_sensor_id=f"SENSOR_{observation.node_id}",
        seq=0,
        generated_at=float(now_s),
        sent_at=float(now_s),
        payload=observation_payload(observation),
        size_bytes=float(len(observation_payload(observation))),
        kind=MESSAGE_KIND_NODE_OBSERVATION,
    )
    # 直接进链路：这里复用 `CommBus` 的发送语义，但不经过测量打包路径
    # （载荷类型不同，白名单也不同）
    link = None
    for (src, dst), candidate in getattr(bus, "_links", {}).items():
        if src == src_platform_id and (dst_platform_ids is None
                                       or dst in dst_platform_ids):
            link = candidate
            break
    if link is None:
        return []
    link.transmit(message, now_s)
    bus.log.append(message)
    return [message]


# ----------------------------------------------------------------------
# 中央：只累积**已到达**的摘要
# ----------------------------------------------------------------------


class CentralObservationStore:
    """中央调度器的观测缓存：**只保存已经到达过的节点摘要**。

    没到达就没有——不会从任何全局对象补齐。
    断开远端消息后，该节点的摘要保持最后一次到达时的内容，
    而 `node_information_age_s` 随全局时钟持续增大。
    """

    def __init__(self, expected_node_ids: Sequence[str]) -> None:
        #: 拓扑（调度器先验知道有哪些节点），不是真值
        self.expected_node_ids: List[str] = list(expected_node_ids)
        #: node_id -> (arrived_at_s, NodeObservation)
        self._latest: Dict[str, Tuple[float, NodeObservation]] = {}
        self.n_ingested = 0
        self.rejected_not_arrived = 0

    # ------------------------------------------------------------------

    def ingest(self, message: MeasurementMessage, now_s: float) -> bool:
        """接收一条消息；**未到达的一律拒绝**。

        返回是否被接受。拒绝的原因会计数，便于测试断言
        "调度器没有偷看未到达的信息"。
        """
        if getattr(message, "kind", "") != MESSAGE_KIND_NODE_OBSERVATION:
            self.rejected_not_arrived += 1
            return False
        arrived_at = getattr(message, "arrived_at", None)
        if arrived_at is None or arrived_at > now_s + 1e-12:
            # 未到达（或在途）：绝不能读它的内容
            self.rejected_not_arrived += 1
            return False
        observation = node_observation_from_payload(dict(message.payload),
                                                    arrived_at)
        self._latest[observation.node_id] = (float(arrived_at), observation)
        self.n_ingested += 1
        return True

    def ingest_arrived(self, bus: Any, dst_platform_id: str,
                       now_s: float) -> int:
        """从总线上取**本时刻已到达**的消息并入库（一次投递，不重复）。"""
        accepted = 0
        for message in bus.consume(dst_platform_id, now_s):
            if self.ingest(message, now_s):
                accepted += 1
        return accepted

    # ------------------------------------------------------------------

    def observe(self, now_s: float) -> CentralObservation:
        """构造中央观测：逐节点给出**已到达**的摘要与其年龄。"""
        nodes: List[NodeObservation] = []
        mask: List[bool] = []
        ages: List[Optional[float]] = []
        content_ages: List[Optional[float]] = []
        missing: List[str] = []
        for node_id in self.expected_node_ids:
            entry = self._latest.get(node_id)
            if entry is None:
                # 从未收到 → 该节点**完全缺失**（不编造、不从别处补）
                nodes.append(NodeObservation(
                    node_id=node_id, observed_at_s=0.0,
                    missing=["summary_never_arrived"]))
                mask.append(False)
                ages.append(None)
                content_ages.append(None)
                missing.append(f"{node_id}:summary_never_arrived")
                continue
            arrived_at, observation = entry
            age = max(0.0, now_s - arrived_at)
            # 内容年龄用摘要**自己声明的**生成时刻（它随消息一起过来），
            # 因此它度量的是"中央手上这条信息的实际新旧"，与链路延迟一致。
            content_age = max(0.0, now_s - float(observation.observed_at_s))
            nodes.append(observation)
            mask.append(True)
            ages.append(age)
            content_ages.append(content_age)
            if age > 0.0:
                missing.append(f"{node_id}:summary_age_{age:.3f}s")
            if content_age > 0.0:
                missing.append(f"{node_id}:content_age_{content_age:.3f}s")
        return CentralObservation(
            schema_version=SCHEMA_VERSION,
            observed_at_s=float(now_s),
            nodes=nodes,
            node_valid_mask=mask,
            node_information_age_s=ages,
            node_content_age_s=content_ages,
            missing=missing,
            n_messages_ingested=self.n_ingested,
        )


# ----------------------------------------------------------------------
# 真值隔离自检
# ----------------------------------------------------------------------


def observation_truth_violations(payload: Any) -> List[str]:
    """递归扫描**已序列化**的观测，找真值通道（空列表 = 通过）。"""
    violations: List[str] = []
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                text_key = str(key)
                for prefix in FORBIDDEN_OBSERVATION_PREFIXES:
                    if text_key.startswith(prefix):
                        violations.append(f"观测出现真值键 {text_key}")
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return violations


# ----------------------------------------------------------------------
# 定长适配器（给旧 DQN 用，**独立实现、独立版本**）
# ----------------------------------------------------------------------


class FixedLengthAdapter:
    """把**变长**观测编码成定长向量，供旧 DQN 使用。

    为什么必须是独立适配器，而不是"截取前几维"：

    * 旧路径（full/pomdp/ideal/realistic）的维度与语义是**冻结**的，
      它们由各自的 checkpoint 固定。把变长观测硬塞进那个维度，
      等于改变了输入语义而权重不认识 → 评测数字不可解释。
    * 因此本适配器有自己的 `schema_version` 与 `output_dim`，
      并提供 `check_checkpoint_dim(dim)`：**维度不匹配直接报错**，
      不允许静默加载。

    编码布局（`slot` 个槽位，每槽 `per_slot` 维 + 掩码）：

    ```
    [mask(1), x, y, z, vx, vy, vz, sigma_x, sigma_y, sigma_z,
     age_norm, coasting, n_sources] × slot
    + [available, update_period_norm, remaining_ratio × 3, n_arrived_norm]
    ```

    超过 `slot` 条的航迹**不截断丢弃信息**：掩码之外的信息通过
    `overflow_count` 显式报出（由调用方决定怎么处理），
    而不是假装只有 `slot` 条。
    """

    #: 每槽位维度（与编码布局一一对应）
    PER_SLOT_FIELDS: Tuple[str, ...] = (
        "mask", "x", "y", "z", "vx", "vy", "vz",
        "sigma_x", "sigma_y", "sigma_z", "age_norm", "coasting", "n_sources",
    )
    #: 节点级尾部维度
    TAIL_FIELDS: Tuple[str, ...] = (
        "available", "update_period_norm",
        "remaining_sample_ratio", "remaining_processing_ratio",
        "remaining_comm_ratio", "n_arrived_norm",
    )
    #: 归一化尺度（显式给出，避免"魔法数"）
    SCALES: Dict[str, float] = {
        "range_m": 30000.0, "velocity_mps": 300.0, "sigma_m": 1000.0,
        "age_s": 30.0, "update_period_s": 10.0, "n_sources": 10.0,
        "n_arrived": 20.0,
    }

    def __init__(self, slots: int = 4) -> None:
        if slots <= 0:
            raise ValueError("slots 必须为正")
        self.slots = int(slots)
        self.schema_version = f"{SCHEMA_VERSION}+fixed{self.slots}"
        self.output_dim = (len(self.PER_SLOT_FIELDS) * self.slots
                           + len(self.TAIL_FIELDS))

    # ------------------------------------------------------------------

    def check_checkpoint_dim(self, checkpoint_dim: int) -> None:
        """维度不匹配**必须报错**，不允许静默加载或截取。"""
        if int(checkpoint_dim) != self.output_dim:
            raise ValueError(
                f"checkpoint 维度 {checkpoint_dim} 与本适配器输出维度 "
                f"{self.output_dim} 不匹配（schema={self.schema_version}）。"
                "不允许截取前几维或硬塞进旧维度：旧路径（"
                f"{list(LEGACY_OBSERVATION_MODES)}）的输入语义是冻结的，"
                "改语义会让评测数字不可解释。旧 checkpoint 必须使用"
                "它自己那一套观测。"
            )

    def encode(self, observation: NodeObservation) -> List[float]:
        import math

        scales = self.SCALES
        vector: List[float] = []
        tracks = list(observation.tracks)
        mask = list(observation.track_valid_mask)
        for slot in range(self.slots):
            if slot < len(tracks) and slot < len(mask) and mask[slot]:
                track = tracks[slot]
                vector.extend([
                    1.0,
                    track.position[0] / scales["range_m"],
                    track.position[1] / scales["range_m"],
                    track.position[2] / scales["range_m"],
                    track.velocity[0] / scales["velocity_mps"],
                    track.velocity[1] / scales["velocity_mps"],
                    track.velocity[2] / scales["velocity_mps"],
                    track.sigma_position[0] / scales["sigma_m"],
                    track.sigma_position[1] / scales["sigma_m"],
                    track.sigma_position[2] / scales["sigma_m"],
                    (0.0 if math.isinf(track.information_age_s)
                     else min(1.0, track.information_age_s / scales["age_s"])),
                    1.0 if track.coasting else 0.0,
                    min(1.0, track.n_sources / scales["n_sources"]),
                ])
            else:
                vector.extend([0.0] * len(self.PER_SLOT_FIELDS))
        def ratio(unit: ResourceUnit) -> float:
            capacity = observation.capacity.get(unit.value, 0.0)
            remaining = observation.remaining.get(unit.value, 0.0)
            return (remaining / capacity) if capacity > 0 else 0.0

        vector.extend([
            1.0 if observation.available else 0.0,
            min(1.0, observation.update_period_s / scales["update_period_s"]),
            ratio(ResourceUnit.SAMPLE_SLOT),
            ratio(ResourceUnit.PROCESSING_OP),
            ratio(ResourceUnit.COMM_BYTE),
            min(1.0, observation.n_messages_arrived / scales["n_arrived"]),
        ])
        return vector

    def overflow_count(self, observation: NodeObservation) -> int:
        """超出槽位的**有效**航迹数（显式报出，不静默丢弃）。"""
        valid = sum(1 for flag in observation.track_valid_mask if flag)
        return max(0, valid - self.slots)

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for slot in range(self.slots):
            names.extend(f"slot{slot}_{field}" for field in self.PER_SLOT_FIELDS)
        names.extend(self.TAIL_FIELDS)
        return names
