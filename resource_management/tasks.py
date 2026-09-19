"""任务队列：从**观测**派生任务，而不是从真值对象。

支持四类任务（用户指定）
------------------------
| 任务类型 | 含义 | 何时产生 |
| --- | --- | --- |
| `predefined_sample` | **预定义采样**：按节点的更新周期对该节点采样 | 由调度器按周期下发，**不需要任何目标知识** |
| `estimate_update` | **已有估计的更新**：对已存在的航迹做一次处理更新 | 只能针对观测里**已存在**的 `track_id` |
| `process` | 处理任务（关联/滤波循环） | 同上，绑定到已有航迹 |
| `share` | 共享任务（把本地航迹摘要发出去） | 与目标无关，只依赖节点自身 |

每个任务都显式带：**释放时间 / 截止时间 / 预估成本 / 所属节点 / 完成状态**，
并且有 `created_from` 溯源字段说明它是从哪次观测/哪条规则来的。

关键纪律：**未知对象不能依据 `truth_id` 提前创建任务**
-----------------------------------------------------
任务的对象键只能是**观测里实际存在的 `track_id`**（并且该航迹的有效掩码为真）。
因此：

* `TaskQueue.create_from_observation()` 只遍历观测的 `track_ids()`，
  调用方**没有**传任意对象键的入口；
* `TaskRequest.targets` 若出现不在观测里的键 → 抛 `UnknownObjectError`；
* 传入形如真值 ID 的键（`TGT1` / `ESM1` 之类）同样会被拒——
  判据不是"它像不像真值 ID"，而是"它有没有出现在观测里"，
  这比模式匹配更可靠。

没有任何"猜测对象"的路径：**看不见的目标不会产生任务**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from resource_management.observation import (
    CentralObservation,
    NodeObservation,
)
from resource_management.units import (
    DEFAULT_DURATION_S,
    TEACHING_COST_MODEL,
    ResourceUnit,
    TaskKind,
    format_cost,
)


class QueueTaskKind(str, Enum):
    """队列任务类型（与执行器的 `TaskKind` 是两套词表，见 `to_task_request`）。"""

    PREDEFINED_SAMPLE = "predefined_sample"
    ESTIMATE_UPDATE = "estimate_update"
    PROCESS = "process"
    SHARE = "share"


QUEUE_TASK_KIND_CN: Dict[QueueTaskKind, str] = {
    QueueTaskKind.PREDEFINED_SAMPLE: "预定义采样",
    QueueTaskKind.ESTIMATE_UPDATE: "已有估计的更新",
    QueueTaskKind.PROCESS: "处理",
    QueueTaskKind.SHARE: "共享",
}

#: 队列任务类型 → 执行器任务类型
_EXECUTOR_KIND: Dict[QueueTaskKind, TaskKind] = {
    QueueTaskKind.PREDEFINED_SAMPLE: TaskKind.SAMPLE,
    QueueTaskKind.ESTIMATE_UPDATE: TaskKind.PROCESS,
    QueueTaskKind.PROCESS: TaskKind.PROCESS,
    QueueTaskKind.SHARE: TaskKind.SHARE,
}


class TaskStatus(str, Enum):
    PENDING = "pending"        # 已入队、释放时间未到
    RELEASED = "released"      # 可以提交给执行器
    SUBMITTED = "submitted"    # 已提交
    COMPLETED = "completed"    # 执行成功
    REJECTED = "rejected"      # 执行器拒绝（原因见 reason）
    EXPIRED = "expired"        # 超过截止时间仍未完成
    CANCELLED = "cancelled"    # 被显式取消


class UnknownObjectError(ValueError):
    """任务引用了**观测里不存在**的对象（可能是真值 ID 或拼写错误）。"""


class DuplicateTaskError(ValueError):
    """同一去重键的任务已存在。"""


@dataclass
class QueuedTask:
    """队列里的一个任务。

    | 字段 | 含义 |
    | --- | --- |
    | `release_time_s` | **释放时间**：早于此时间不可提交 |
    | `deadline_s` | **截止时间**：超过仍未完成即 `expired`（None = 不限） |
    | `estimated_cost` | **预估成本**（逐资源单位） |
    | `node_id` | **所属节点** |
    | `status` | **完成状态** |
    | `targets` | 对象键：只能是观测里存在的 `track_id` |
    | `created_from` | 溯源：由哪次观测/哪条规则创建 |
    """

    task_id: str
    kind: QueueTaskKind
    node_id: str
    release_time_s: float
    deadline_s: Optional[float] = None
    estimated_cost: Dict[ResourceUnit, float] = field(default_factory=dict)
    targets: Tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    created_from: str = ""
    idempotency_key: str = ""
    reason: str = ""
    submitted_plan_id: str = ""
    note: str = ""

    # ------------------------------------------------------------------

    @property
    def dedup_key(self) -> str:
        return self.idempotency_key or f"{self.node_id}:{self.task_id}"

    def is_due(self, now_s: float) -> bool:
        return self.status is TaskStatus.PENDING and now_s >= self.release_time_s

    def is_expired(self, now_s: float) -> bool:
        return (self.deadline_s is not None and now_s > self.deadline_s
                and self.status not in (TaskStatus.COMPLETED,
                                        TaskStatus.EXPIRED,
                                        TaskStatus.CANCELLED))

    def estimated_cost_text(self) -> str:
        return format_cost(self.estimated_cost)

    def to_task_request(self, start_s: float) -> Any:
        """转成执行器的 `TaskRequest`（成本沿用预估值）。"""
        from resource_management.model import TaskRequest

        return TaskRequest(
            task_id=self.task_id,
            node_id=self.node_id,
            kind=_EXECUTOR_KIND[self.kind],
            start_s=float(start_s),
            duration_s=DEFAULT_DURATION_S.get(_EXECUTOR_KIND[self.kind], 1.0),
            cost=dict(self.estimated_cost),
            idempotency_key=self.dedup_key,
            entities=self.targets,
            note=f"[{self.kind.value}] {self.note}".strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "kind_cn": QUEUE_TASK_KIND_CN.get(self.kind, self.kind.value),
            "node_id": self.node_id,
            "release_time_s": round(self.release_time_s, 6),
            "deadline_s": (None if self.deadline_s is None
                           else round(self.deadline_s, 6)),
            "estimated_cost": {unit.value: value
                               for unit, value in self.estimated_cost.items()},
            "estimated_cost_text": self.estimated_cost_text(),
            "targets": list(self.targets),
            "status": self.status.value,
            "created_from": self.created_from,
            "dedup_key": self.dedup_key,
            "reason": self.reason,
            "submitted_plan_id": self.submitted_plan_id,
            "note": self.note,
        }


def default_cost(kind: QueueTaskKind) -> Dict[ResourceUnit, float]:
    """按教学成本模型给出预估成本（可被调用方覆盖）。"""
    return TEACHING_COST_MODEL[_EXECUTOR_KIND[kind]].as_dict()


class TaskQueue:
    """任务队列：入队去重、按释放时间取due、状态流转、超期处理。

    **不接受任意对象键**：所有建任务的入口都会先核对
    "该对象是否出现在给定观测里"。
    """

    def __init__(self) -> None:
        self.tasks: List[QueuedTask] = []
        self._by_key: Dict[str, QueuedTask] = {}
        #: 每次建任务时的观测快照（用于事后追溯"当时看得见什么"）
        self.observation_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def _register(self, task: QueuedTask) -> QueuedTask:
        if task.dedup_key in self._by_key:
            raise DuplicateTaskError(
                f"去重键 {task.dedup_key!r} 已存在（任务 "
                f"{self._by_key[task.dedup_key].task_id}）")
        self._by_key[task.dedup_key] = task
        self.tasks.append(task)
        return task

    def _check_targets(self, targets: Sequence[str],
                       visible: Iterable[str]) -> None:
        """对象必须**出现在观测里**，否则拒绝建任务。

        判据是"可见性"，**不是**模式匹配：即便传入的是一个像真值 ID 的字符串
        （`TGT1`），只要它不在观测的航迹里，就会被拒——
        因为调度器根本没有看到过这个对象。
        """
        visible_set = set(visible)
        unknown = [target for target in targets if target not in visible_set]
        if unknown:
            raise UnknownObjectError(
                f"任务引用了观测中不存在的对象 {unknown}。"
                f"当前可见的航迹 ID 为 {sorted(visible_set)}。"
                "调度器不得依据真值 ID 提前创建任务——看不见的目标不产生任务。"
            )

    # ------------------------------------------------------------------

    def enqueue(self, task: QueuedTask) -> QueuedTask:
        """入队（用于**不依赖目标**的任务，例如预定义采样与共享）。"""
        return self._register(task)

    def enqueue_for_observation(
        self,
        observation: NodeObservation,
        kind: QueueTaskKind,
        task_id: str,
        release_time_s: Optional[float] = None,
        deadline_s: Optional[float] = None,
        estimated_cost: Optional[Dict[ResourceUnit, float]] = None,
        targets: Optional[Sequence[str]] = None,
        note: str = "",
        idempotency_key: str = "",
    ) -> QueuedTask:
        """针对**一个节点观测**建任务；对象键必须是该观测里的有效航迹。"""
        visible = observation.track_ids()
        chosen = tuple(targets) if targets is not None else tuple(visible)
        self._check_targets(chosen, visible)
        if kind in (QueueTaskKind.ESTIMATE_UPDATE,) and not chosen:
            # 没有已有估计 → 不允许凭空建"更新任务"
            raise UnknownObjectError(
                "该观测里没有任何有效航迹，不能创建"
                f"{QUEUE_TASK_KIND_CN[kind]}任务（未知对象不得提前建任务）"
            )
        task = QueuedTask(
            task_id=task_id, kind=kind, node_id=observation.node_id,
            release_time_s=(observation.observed_at_s if release_time_s is None
                            else float(release_time_s)),
            deadline_s=deadline_s,
            estimated_cost=(dict(estimated_cost) if estimated_cost is not None
                            else default_cost(kind)),
            targets=chosen,
            created_from=(f"observation(node={observation.node_id},"
                          f"t={observation.observed_at_s:g},"
                          f"tracks={len(visible)},"
                          f"targets={list(chosen)})"),
            idempotency_key=idempotency_key or f"{kind.value}:{task_id}",
            note=note,
        )
        self.observation_log.append({
            "task_id": task_id,
            "node_id": observation.node_id,
            "observed_at_s": observation.observed_at_s,
            "visible_track_ids": visible,
            "chosen_targets": list(chosen),
        })
        return self._register(task)

    # ------------------------------------------------------------------

    def create_from_observation(
        self,
        observation: NodeObservation,
        now_s: float,
        sample_prefix: str = "sample",
        update_prefix: str = "update",
        share_prefix: str = "share",
        update_deadline_s: Optional[float] = None,
        deadline_offsets: Optional[Dict[QueueTaskKind, float]] = None,
        allow_share: bool = True,
        allow_process: bool = False,
        allow_sample: bool = True,
        allow_estimate_update: bool = True,
    ) -> List[QueuedTask]:
        """按常规规则从一次观测派生任务（**唯一的自动建任务入口**）。

        规则（全部只依赖观测内容，不依赖任何真值）：

        1. **预定义采样**：只要节点可用就下发一次（与有没有目标无关）；
        2. **已有估计的更新**：对观测里**每一条有效航迹**各下一个更新任务，
           同一条航迹在同一观测周期内只下一次（去重键含航迹 ID 与轮次）；
        3. **共享**：节点可用且有航迹时下发一次（把本地摘要发出去）。

        ⚠️ 航迹为空时只下发采样任务：**不会**为看不见的目标造任务。

        `deadline_offsets` 按任务类型给**不同的截止余量**（秒）。
        为什么需要它：如果所有任务都用同一个余量，
        它们的截止时间就完全相同，**EDF 退化成先到先服务**，
        与轮询没有区别——那样"最早截止时间优先"这条基线就没有检验力。
        截止时间在这里表达的是**服务需求**（共享的数据最易过期、
        更新次之、采样最松），不是任何目标价值。
        """
        offsets = dict(deadline_offsets or {})
        if update_deadline_s is not None and QueueTaskKind.ESTIMATE_UPDATE \
                not in offsets:
            offsets[QueueTaskKind.ESTIMATE_UPDATE] = (
                update_deadline_s - now_s)

        def deadline_for(kind: QueueTaskKind) -> float:
            return now_s + float(offsets.get(kind, 5.0))

        created: List[QueuedTask] = []
        if not observation.available:
            return created

        round_key = f"{int(round(now_s * 1000)):08d}"
        if allow_sample:
            created.append(self.enqueue(QueuedTask(
                task_id=f"{sample_prefix}-{observation.node_id}-{round_key}",
                kind=QueueTaskKind.PREDEFINED_SAMPLE,
                node_id=observation.node_id,
                release_time_s=now_s,
                deadline_s=deadline_for(QueueTaskKind.PREDEFINED_SAMPLE),
                estimated_cost=default_cost(QueueTaskKind.PREDEFINED_SAMPLE),
                targets=(),
                created_from=(
                    f"predefined(period={observation.update_period_s:g}s)"
                ),
                idempotency_key=f"{observation.node_id}:sample:{round_key}",
                note="预定义采样：只依赖节点与更新周期，不需要目标知识",
            )))

        if allow_process:
            created.append(self.enqueue(QueuedTask(
                task_id=f"process-{observation.node_id}-{round_key}",
                kind=QueueTaskKind.PROCESS,
                node_id=observation.node_id,
                release_time_s=now_s,
                deadline_s=deadline_for(QueueTaskKind.PROCESS),
                estimated_cost=default_cost(QueueTaskKind.PROCESS),
                targets=(),
                created_from=(
                    f"runtime_buffer(node={observation.node_id},t={now_s:g})"
                ),
                idempotency_key=f"{observation.node_id}:process:{round_key}",
                note="处理已实际采样或已到达的测量；不预先假定目标 ID",
            )))

        visible = observation.track_ids()
        for track_id in visible if allow_estimate_update else ():
            created.append(self.enqueue(QueuedTask(
                task_id=f"{update_prefix}-{track_id}-{round_key}",
                kind=QueueTaskKind.ESTIMATE_UPDATE,
                node_id=observation.node_id,
                release_time_s=now_s,
                deadline_s=deadline_for(QueueTaskKind.ESTIMATE_UPDATE),
                estimated_cost=default_cost(QueueTaskKind.ESTIMATE_UPDATE),
                targets=(track_id,),
                created_from=(f"observation(node={observation.node_id},"
                              f"track={track_id})"),
                idempotency_key=f"{track_id}:update:{round_key}",
                note="已有估计的更新：目标必须是观测里已存在的航迹",
            )))

        if visible and allow_share:
            created.append(self.enqueue(QueuedTask(
                task_id=f"{share_prefix}-{observation.node_id}-{round_key}",
                kind=QueueTaskKind.SHARE,
                node_id=observation.node_id,
                release_time_s=now_s,
                deadline_s=deadline_for(QueueTaskKind.SHARE),
                estimated_cost=default_cost(QueueTaskKind.SHARE),
                targets=(),
                created_from=f"observation(node={observation.node_id})",
                idempotency_key=f"{observation.node_id}:share:{round_key}",
                note="共享本地航迹摘要",
            )))
        return created

    # ------------------------------------------------------------------

    def due(self, now_s: float) -> List[QueuedTask]:
        """释放时间已到、仍待提交的任务（按释放时间与入队序稳定排序）。"""
        ready = [task for task in self.tasks if task.is_due(now_s)]
        return sorted(ready, key=lambda task: (task.release_time_s,
                                               task.task_id))

    def expire_overdue(self, now_s: float) -> List[QueuedTask]:
        """把超期未完成的任务标为 `expired` 并返回它们。"""
        expired: List[QueuedTask] = []
        for task in self.tasks:
            if task.is_expired(now_s):
                task.status = TaskStatus.EXPIRED
                task.reason = (f"超过截止时间 {task.deadline_s:g}s "
                               f"（当前 {now_s:g}s）")
                expired.append(task)
        return expired

    def mark(self, task_id: str, status: TaskStatus, reason: str = "",
             plan_id: str = "") -> QueuedTask:
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                if reason:
                    task.reason = reason
                if plan_id:
                    task.submitted_plan_id = plan_id
                return task
        raise KeyError(f"未知任务 {task_id!r}")

    def by_status(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for task in self.tasks:
            grouped.setdefault(task.status.value, []).append(task.task_id)
        return grouped

    def summary(self) -> Dict[str, Any]:
        return {
            "n_tasks": len(self.tasks),
            "by_status": {key: len(value) for key, value
                          in self.by_status().items()},
            "by_kind": {
                kind.value: sum(1 for task in self.tasks if task.kind is kind)
                for kind in QueueTaskKind
            },
            # 逐任务类型 × 逐状态：只看总完成率看不出"哪一类服务被系统性跳过"，
            # 也看不出"低等级服务到底是被判过期还是被判长期未获服务"。
            "by_kind_status": {
                kind.value: {
                    status.value: sum(
                        1 for task in self.tasks
                        if task.kind is kind and task.status is status)
                    for status in TaskStatus
                }
                for kind in QueueTaskKind
            },
        }
