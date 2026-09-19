"""四组**同形**研究观测：只改算法可见输入，不改网络/动作/闭环。

设计约束（用户点名）
--------------------
1. **四组输入维度完全相同**，被消融的特征**置零**而不是删除——
   否则"输入更宽"会被误归因成"信息更有用"；
2. 复现已有的 `resource_management.information_research.CentralResearchFeatureAdapter`
   （它已经把四组的槽位、顺序、维度定义好，并按 arm 置零）；
3. 本模块只**追加**一个与 arm 无关的基础块（时域进度 + 任务队列计数），
   基础块里**不含**任何信息年龄 / 协方差 / 来源一致性内容——
   否则消融就不干净了。

布局
----
```
[ research adapter 块 ]  ← 97/94 维（随节点数与每节点航迹槽位定），按 arm 置零
[ 基础块 ]               ← 4 + 3N 维，与 arm 无关
```

| 组别 | 新鲜度（到达/内容/航迹年龄） | 不确定度（σ + 来源一致性） |
| --- | --- | --- |
| `main_baseline` | 置零 | 置零 |
| `freshness_only` | 开 | 置零 |
| `uncertainty_only` | 置零 | 开 |
| `freshness_uncertainty` | 开 | 开 |

信息纪律
--------
* 新鲜度只用**已到达**摘要的年龄：`node_information_age_s`（到达年龄）、
  `node_content_age_s`（内容年龄）、`track.information_age_s`；
* 不确定度只用已到达航迹的位置协方差与 `source_consistency_indicator`
  （由已到达来源的残差/自报 σ 构造）；
* **禁止**读取 `truth_id`、真实传感器偏差、真实关联标签、未来消息、离线真值；
* `source_inconsistency_indicator` 是**未经概率校准的指示量**，
  **不得**称为"出错概率"或"正确概率"（适配器自己也带
  `source_consistency_calibrated_probability: False` 标注）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.information_research import (
    ABLATION_ARMS, AblationArm, CentralResearchFeatureAdapter, RESEARCH_HYPOTHESIS,
)
from resource_management.observation import CentralObservation
from resource_management.tasks import QueueTaskKind, TaskQueue, TaskStatus
from resource_management.units import BUDGET_UNITS, ResourceUnit

#: 研究观测 schema 版本（改布局/尺度必须升这个号）
RESEARCH_OBS_SCHEMA = "resource-rl-research-1.0"

#: 每个节点的航迹槽位数
TRACKS_PER_NODE = 4

#: 基础块（与 arm 无关；**不含**年龄/协方差/一致性）
GLOBAL_BASE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("elapsed_fraction", "本 episode 已用步数占比"),
    ("remaining_time_fraction", "剩余步数占比"),
    ("queue_pending_fraction", "待处理任务数 / QUEUE_SCALE"),
    ("queue_oldest_wait_fraction", "最老待处理任务已等待时长 / WAIT_SCALE"),
)
NODE_BASE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("pending_sample_fraction", "待处理采样任务数 / TASK_SCALE"),
    ("pending_process_fraction", "待处理处理任务数 / TASK_SCALE"),
    ("pending_share_fraction", "待处理共享任务数 / TASK_SCALE"),
)

QUEUE_SCALE = 24.0
WAIT_SCALE = 20.0
TASK_SCALE = 6.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def arm_from_value(value: Any) -> AblationArm:
    """接受枚举或字符串，返回 `AblationArm`。"""
    if isinstance(value, AblationArm):
        return value
    try:
        return AblationArm(str(value))
    except ValueError as exc:
        raise ValueError(
            f"未知消融组 {value!r}；可选 "
            f"{[arm.value for arm in ABLATION_ARMS]}") from exc


class ResearchObservationEncoder:
    """四组同形编码器：`arm` 只决定哪些槽位被置零。"""

    def __init__(self, node_ids: Sequence[str],
                 tracks_per_node: int = TRACKS_PER_NODE) -> None:
        if not node_ids:
            raise ValueError("node_ids 不能为空")
        self.node_ids: Tuple[str, ...] = tuple(str(item) for item in node_ids)
        self.tracks_per_node = int(tracks_per_node)
        self.adapter = CentralResearchFeatureAdapter(
            list(self.node_ids), tracks_per_node=self.tracks_per_node)
        self.adapter_dim = self.adapter.output_dim
        self.base_dim = len(GLOBAL_BASE_FIELDS) + (
            len(self.node_ids) * len(NODE_BASE_FIELDS))
        self.output_dim = self.adapter_dim + self.base_dim
        self.schema_version = (
            f"{RESEARCH_OBS_SCHEMA}+{self.adapter.schema_version}")

    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        names = list(self.adapter.feature_names())
        names.extend(f"global_{name}" for name, _note in GLOBAL_BASE_FIELDS)
        for node_id in self.node_ids:
            names.extend(f"{node_id}_{name}" for name, _note in NODE_BASE_FIELDS)
        return names

    def metadata(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for name in self.adapter.feature_names():
            out.append({"name": name, "source": "research_adapter",
                        "gated": "true"})
        for name, note in GLOBAL_BASE_FIELDS:
            out.append({"name": f"global_{name}", "source": "base_block",
                        "gated": "false", "note": note})
        for node_id in self.node_ids:
            for name, note in NODE_BASE_FIELDS:
                out.append({"name": f"{node_id}_{name}", "source": "base_block",
                            "gated": "false", "note": note})
        return out

    # ------------------------------------------------------------------

    def encode(self, observation: CentralObservation, queue: TaskQueue,
               now_s: float, steps: int, step_index: int,
               arm: Any) -> List[float]:
        """编码为定长向量。**只读**观测与队列。"""
        if steps <= 0:
            raise ValueError("steps 必须为正")
        resolved = arm_from_value(arm)
        vector = list(self.adapter.encode(observation, resolved))
        if len(vector) != self.adapter_dim:
            raise AssertionError(
                f"适配器维度不一致：{len(vector)} != {self.adapter_dim}")

        pending = [task for task in queue.tasks
                   if task.status is TaskStatus.PENDING
                   and task.release_time_s <= now_s + 1e-9]
        oldest_wait = max((now_s - task.release_time_s for task in pending),
                          default=0.0)
        vector.extend([
            _clamp(step_index / steps),
            _clamp((steps - step_index) / steps),
            _clamp(len(pending) / QUEUE_SCALE),
            _clamp(oldest_wait / WAIT_SCALE),
        ])
        counts: Dict[str, Dict[QueueTaskKind, int]] = {
            node_id: {} for node_id in self.node_ids}
        for task in pending:
            bucket = counts.get(task.node_id)
            if bucket is None:
                continue
            bucket[task.kind] = bucket.get(task.kind, 0) + 1
        for node_id in self.node_ids:
            bucket = counts[node_id]
            vector.extend([
                _clamp(bucket.get(QueueTaskKind.PREDEFINED_SAMPLE, 0)
                       / TASK_SCALE),
                _clamp(bucket.get(QueueTaskKind.PROCESS, 0) / TASK_SCALE),
                _clamp(bucket.get(QueueTaskKind.SHARE, 0) / TASK_SCALE),
            ])
        if len(vector) != self.output_dim:
            raise AssertionError(
                f"研究观测维度不一致：{len(vector)} != {self.output_dim}")
        return vector

    # ------------------------------------------------------------------

    def zeroed_slot_mask(self, arm: Any) -> List[bool]:
        """逐槽位是否被**置零**（用于测试与报告：证明消融真的发生了）。"""
        resolved = arm_from_value(arm)
        flags: List[bool] = []
        freshness_names = ("summary_arrival_age_norm",
                           "summary_content_age_norm", "track_age_norm")
        uncertainty_names = ("sigma_position_norm",
                             "source_inconsistency_indicator")
        for name in self.adapter.feature_names():
            if any(name.endswith(suffix) for suffix in freshness_names):
                flags.append(not resolved.uses_freshness)
            elif any(name.endswith(suffix) for suffix in uncertainty_names):
                flags.append(not resolved.uses_uncertainty)
            else:
                flags.append(False)
        flags.extend([False] * self.base_dim)
        return flags


__all__ = [
    "ABLATION_ARMS", "AblationArm", "GLOBAL_BASE_FIELDS", "NODE_BASE_FIELDS",
    "QUEUE_SCALE", "RESEARCH_HYPOTHESIS", "RESEARCH_OBS_SCHEMA",
    "ResearchObservationEncoder", "TASK_SCALE", "TRACKS_PER_NODE",
    "WAIT_SCALE", "arm_from_value",
]
