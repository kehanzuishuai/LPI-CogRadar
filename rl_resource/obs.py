"""定长观测编码器：只编码**调度器本来就可见**的信息。

信息边界（与规则基线完全一致，不额外放宽）
------------------------------------------
输入只有两样东西：

1. `CentralObservation`——中央只读视图（节点可用性、**已到达**摘要的航迹、
   信息年龄、位置协方差、资源容量与剩余量、消息计数）；
2. `TaskQueue`——任务队列（各类型待处理任务数、最紧迫的截止余量）。

**不包含**：真值、未来消息、隐藏对象状态、传感器偏差标签、账本内部字段、
`RuntimeExecutor` 的私有缓冲状态。`tests/test_resource_rl.py` 用 AST 扫描
＋"改真值不改观测"的运行时对照钉住这条边界。

布局（逐字段带单位，顺序固定）
------------------------------
全局 `GLOBAL_FEATURES`（5 项）+ 每节点 `NODE_FEATURES`（11 项）× 节点槽位。

节点槽位数 `max_nodes` 固定（默认 4），实际节点少于它时用 `node_mask` 标记，
多余槽位**全部置 0** 并在 mask 里为 False——这样节点数变化不需要改网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.observation import CentralObservation, NodeObservation
from resource_management.tasks import QueueTaskKind, TaskQueue, TaskStatus
from resource_management.units import BUDGET_UNITS, ResourceUnit

#: 全局特征（名称、单位、含义）
GLOBAL_FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("elapsed_fraction", "1", "本 episode 已用步数占比（有限时域必须显式给）"),
    ("remaining_time_fraction", "1", "剩余步数占比"),
    ("queue_pending_fraction", "1", "待处理任务数 / QUEUE_SCALE（截断到 1）"),
    ("queue_oldest_wait_fraction", "1", "最老待处理任务的已等待时长 / WAIT_SCALE"),
    ("n_active_nodes_fraction", "1", "可用节点数 / max_nodes"),
)

#: 单节点特征
NODE_FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("available", "1", "节点是否可用"),
    ("n_tracks_fraction", "1", "可见航迹数 / TRACK_SCALE"),
    ("mean_track_age_fraction", "1", "可见航迹平均信息年龄 / AGE_SCALE（秒）"),
    ("max_track_age_fraction", "1", "可见航迹最大信息年龄 / AGE_SCALE（秒）"),
    ("mean_sigma_fraction", "1", "可见航迹平均位置 σ / SIGMA_SCALE（米）"),
    ("remaining_sample_fraction", "1", "采样槽剩余 / 容量"),
    ("remaining_process_fraction", "1", "处理操作剩余 / 容量"),
    ("remaining_comm_fraction", "1", "通信字节剩余 / 容量"),
    ("pending_sample_fraction", "1", "待处理采样任务数 / TASK_SCALE"),
    ("pending_process_fraction", "1", "待处理处理任务数 / TASK_SCALE"),
    ("pending_share_fraction", "1", "待处理共享任务数 / TASK_SCALE"),
)

#: 归一化尺度（**显式常数**，不是从数据里估出来的；改它们会改变观测语义）
TRACK_SCALE = 4.0
AGE_SCALE = 10.0
SIGMA_SCALE = 500.0
TASK_SCALE = 6.0
QUEUE_SCALE = 24.0
WAIT_SCALE = 20.0

#: 节点槽位数的默认上限
DEFAULT_MAX_NODES = 4


def observation_dim(max_nodes: int = DEFAULT_MAX_NODES) -> int:
    return len(GLOBAL_FEATURES) + max_nodes * len(NODE_FEATURES)


@dataclass
class EncodedObservation:
    """编码结果：定长向量 + 节点掩码 + 逐字段说明（便于审计）。"""

    values: List[float]
    node_mask: List[bool]
    node_ids: List[str]
    max_nodes: int
    feature_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": len(self.values),
            "node_mask": list(self.node_mask),
            "node_ids": list(self.node_ids),
            "max_nodes": self.max_nodes,
            "feature_names": list(self.feature_names),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _visible(node: NodeObservation) -> List[Any]:
    return [track for track, valid in zip(node.tracks, node.track_valid_mask)
            if valid]


def encode(
    observation: CentralObservation,
    queue: TaskQueue,
    now_s: float,
    steps: int,
    step_index: int,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> EncodedObservation:
    """把（观测, 队列）编码成定长向量。**只读**，不改任何输入。"""
    if steps <= 0:
        raise ValueError("steps 必须为正")
    by_node: Dict[str, NodeObservation] = {
        node.node_id: node for node, valid
        in zip(observation.nodes, observation.node_valid_mask) if valid}

    pending: List[Any] = [task for task in queue.tasks
                          if task.status is TaskStatus.PENDING
                          and task.release_time_s <= now_s + 1e-9]
    oldest_wait = 0.0
    if pending:
        oldest_wait = max(now_s - task.release_time_s for task in pending)

    values: List[float] = [
        _clamp(step_index / steps),
        _clamp((steps - step_index) / steps),
        _clamp(len(pending) / QUEUE_SCALE),
        _clamp(oldest_wait / WAIT_SCALE),
        _clamp(len(by_node) / max(1, max_nodes)),
    ]
    names: List[str] = [f"global_{name}" for name, _unit, _note
                        in GLOBAL_FEATURES]
    node_mask: List[bool] = []
    node_ids: List[str] = []

    node_order = [node.node_id for node in observation.nodes]
    for slot in range(max_nodes):
        if slot >= len(node_order):
            values.extend([0.0] * len(NODE_FEATURES))
            names.extend(f"node{slot}_{name}" for name, _u, _n in NODE_FEATURES)
            node_mask.append(False)
            node_ids.append("")
            continue
        node_id = node_order[slot]
        node = by_node.get(node_id)
        node_ids.append(node_id)
        if node is None:
            values.extend([0.0] * len(NODE_FEATURES))
            names.extend(f"node{slot}_{name}" for name, _u, _n in NODE_FEATURES)
            node_mask.append(False)
            continue
        tracks = _visible(node)
        ages = [float(track.information_age_s) for track in tracks]
        sigmas = [float(max(track.sigma_position)) for track in tracks]

        def remaining_fraction(unit: ResourceUnit) -> float:
            capacity = float(node.capacity.get(unit.value, 0.0))
            remaining = float(node.remaining.get(unit.value, 0.0))
            return _clamp(remaining / capacity) if capacity > 0 else 0.0

        counts = {kind: 0 for kind in QueueTaskKind}
        for task in pending:
            if task.node_id == node_id:
                counts[task.kind] = counts.get(task.kind, 0) + 1

        values.extend([
            1.0 if node.available else 0.0,
            _clamp(len(tracks) / TRACK_SCALE),
            _clamp((sum(ages) / len(ages)) / AGE_SCALE) if ages else 0.0,
            _clamp(max(ages) / AGE_SCALE) if ages else 0.0,
            _clamp((sum(sigmas) / len(sigmas)) / SIGMA_SCALE) if sigmas else 0.0,
            remaining_fraction(ResourceUnit.SAMPLE_SLOT),
            remaining_fraction(ResourceUnit.PROCESSING_OP),
            remaining_fraction(ResourceUnit.COMM_BYTE),
            _clamp(counts.get(QueueTaskKind.PREDEFINED_SAMPLE, 0) / TASK_SCALE),
            _clamp(counts.get(QueueTaskKind.PROCESS, 0) / TASK_SCALE),
            _clamp(counts.get(QueueTaskKind.SHARE, 0) / TASK_SCALE),
        ])
        names.extend(f"node{slot}_{name}" for name, _u, _n in NODE_FEATURES)
        node_mask.append(bool(node.available))

    if len(values) != observation_dim(max_nodes):
        raise AssertionError(
            f"观测维度不一致：{len(values)} != {observation_dim(max_nodes)}")
    return EncodedObservation(values=values, node_mask=node_mask,
                              node_ids=node_ids, max_nodes=max_nodes,
                              feature_names=names)


def feature_metadata(max_nodes: int = DEFAULT_MAX_NODES) -> List[Dict[str, str]]:
    """逐字段元数据（名称/单位/含义），供文档与审计对照。"""
    out: List[Dict[str, str]] = [
        {"name": f"global_{name}", "unit": unit, "note": note}
        for name, unit, note in GLOBAL_FEATURES]
    for slot in range(max_nodes):
        out.extend({"name": f"node{slot}_{name}", "unit": unit, "note": note}
                   for name, unit, note in NODE_FEATURES)
    return out


def scales() -> Dict[str, float]:
    return {
        "TRACK_SCALE": TRACK_SCALE, "AGE_SCALE": AGE_SCALE,
        "SIGMA_SCALE": SIGMA_SCALE, "TASK_SCALE": TASK_SCALE,
        "QUEUE_SCALE": QUEUE_SCALE, "WAIT_SCALE": WAIT_SCALE,
    }


__all__ = [
    "DEFAULT_MAX_NODES", "EncodedObservation", "GLOBAL_FEATURES",
    "NODE_FEATURES", "encode", "feature_metadata", "observation_dim", "scales",
]
