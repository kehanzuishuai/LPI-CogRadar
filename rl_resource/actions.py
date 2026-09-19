"""结构化动作空间：逐节点分类选择 + 合法动作 mask + 动作 → `ExecutionPlan`。

为什么不用"把所有组合展开成一张大离散动作表"
--------------------------------------------
每个节点每 tick 有 4 种选择（idle / sample / process / share），
N 个节点的联合动作是 4^N。N=2 时是 16，看着不大，但：

* **节点数是场景变量**（本工程 2 个，将来可能更多），动作表规模指数爆炸；
* 动作表里绝大部分条目是**非法或等价**的（例如某节点没有可处理数据时，
  它的 process 选项对该节点恒为空操作）；
* 展平后**丢掉了"节点对称性"**，同一套逻辑要分别学 N 次。

因此采用**因素化（autoregressive / factorized）动作空间**：

    a = (a_0, a_1, …, a_{N-1}),   a_i ∈ {0: idle, 1: sample, 2: process, 3: share}

策略对每个节点输出 4 个 logits（见 `policy.py`），联合 log 概率是各节点之和，
一次前向就得到整个联合动作——**参数与节点数线性相关，而不是指数**。
这仍然是"结构化或分阶段选择"，且天然给出**逐节点的合法动作 mask**。

合法性（mask）的定义
--------------------
mask 决定"哪些动作允许被采样"，判据是**这条任务在运行时真的能做**：

| 动作 | 合法条件 |
| --- | --- |
| `idle` | **恒合法**（不做任何事永远可行） |
| `sample` | 节点可用 **且** 队列里有该节点待处理的采样任务 **且** 预算够 |
| `process` | 节点可用 **且** 队列里有该节点的处理任务 **且** 真的有数据可处理（本地缓冲非空或远端消息已到达）**且** 预算够 |
| `share` | 节点可用 **且** 队列里有该节点的共享任务 **且** outbox 非空 **且** 预算够 |

"预算够"按 `TaskRequest.effective_cost()` 与**观测里报告的** `remaining` 比较——
用的是调度器本来就看得见的量，不额外偷看账本。

⚠️ mask 只排除"必然做不成或必然空转"的动作，**不**替策略做取舍：
当 sample / process / share 同时合法时，选哪个是策略要学的（这正是本层的意义）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.model import ExecutionPlan, NodeState, TaskRequest
from resource_management.observation import CentralObservation, NodeObservation
from resource_management.tasks import QueueTaskKind, QueuedTask, TaskQueue, TaskStatus
from resource_management.units import BUDGET_UNITS, ResourceUnit

#: 逐节点动作编号（**顺序固定**，改动会破坏已保存的 checkpoint 语义）
ACTION_IDLE = 0
ACTION_SAMPLE = 1
ACTION_PROCESS = 2
ACTION_SHARE = 3

ACTION_NAMES: Tuple[str, ...] = ("idle", "sample", "process", "share")
ACTION_KINDS: Dict[int, Optional[QueueTaskKind]] = {
    ACTION_IDLE: None,
    ACTION_SAMPLE: QueueTaskKind.PREDEFINED_SAMPLE,
    ACTION_PROCESS: QueueTaskKind.PROCESS,
    ACTION_SHARE: QueueTaskKind.SHARE,
}
NAME_TO_ACTION: Dict[str, int] = {name: index
                                  for index, name in enumerate(ACTION_NAMES)}
#: 每个节点的动作数（策略输出维度）
N_ACTIONS = len(ACTION_NAMES)

#: 空槽位的掩码行：只允许 idle。用于把节点数补齐到 `max_nodes`
IDLE_ONLY_ROW: Tuple[bool, ...] = (True,) + (False,) * (N_ACTIONS - 1)


def pad_mask(mask: Sequence[Sequence[bool]], max_nodes: int
             ) -> List[List[bool]]:
    """把逐节点掩码补齐到 `max_nodes` 行（多出来的槽位只允许 idle）。

    为什么需要：策略固定输出 `max_nodes` 行（节点数变化不改网络结构），
    而实际存在的节点可能少于它。补齐后"动作张量形状"与"网络输出形状"
    永远一致，训练循环里不需要按 episode 分支。
    """
    rows = [list(bool(value) for value in row) for row in mask]
    if len(rows) > max_nodes:
        raise ValueError(f"节点数 {len(rows)} 超过 max_nodes={max_nodes}")
    while len(rows) < max_nodes:
        rows.append(list(IDLE_ONLY_ROW))
    return rows


@dataclass(frozen=True)
class ActionContext:
    """构造 mask 与计划所需的运行时事实（全部来自**已到达/已执行**的状态）。"""

    node_ids: Tuple[str, ...]
    #: 逐节点的候选任务（kind → 选中的任务），只含 PENDING 任务
    candidates: Dict[str, Dict[int, QueuedTask]]
    #: 节点是否可用（来自观测）
    available: Dict[str, bool]
    #: 逐节点逐单位的**观测剩余量**
    remaining: Dict[str, Dict[ResourceUnit, float]]
    #: 该节点是否真的有数据可处理（本地缓冲或已到达的远端消息）
    processable: Dict[str, bool]
    #: 该节点 outbox 是否有待发送数据
    shareable: Dict[str, bool]

    def mask(self) -> List[List[bool]]:
        """逐节点的合法动作掩码（`[节点][动作]`）。"""
        rows: List[List[bool]] = []
        for node_id in self.node_ids:
            row = [True] + [False] * (N_ACTIONS - 1)
            if self.available.get(node_id, False):
                for action in (ACTION_SAMPLE, ACTION_PROCESS, ACTION_SHARE):
                    task = self.candidates.get(node_id, {}).get(action)
                    if task is None:
                        continue
                    if (action == ACTION_PROCESS
                            and not self.processable.get(node_id, False)):
                        continue
                    if (action == ACTION_SHARE
                            and not self.shareable.get(node_id, False)):
                        continue
                    if self._affordable(node_id, task):
                        row[action] = True
            rows.append(row)
        return rows

    def _affordable(self, node_id: str, task: QueuedTask) -> bool:
        remaining = self.remaining.get(node_id)
        if remaining is None:
            return False
        cost = task.estimated_cost
        for unit in BUDGET_UNITS:
            need = float(cost.get(unit, 0.0) or 0.0)
            if need <= 0.0:
                continue
            if need > float(remaining.get(unit, 0.0)) + 1e-9:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "available": dict(self.available),
            "processable": dict(self.processable),
            "shareable": dict(self.shareable),
            "candidates": {
                node_id: {ACTION_NAMES[action]: task.task_id
                          for action, task in sorted(by_action.items())}
                for node_id, by_action in self.candidates.items()},
            "mask": self.mask(),
        }


# ----------------------------------------------------------------------
# 由观测 + 队列构造动作上下文
# ----------------------------------------------------------------------


def build_action_context(
    observation: CentralObservation,
    queue: TaskQueue,
    now_s: float,
    processable: Dict[str, bool],
    shareable: Dict[str, bool],
) -> ActionContext:
    """从中央观测与任务队列抽取可行动作。

    **只读**：不改队列、不改观测、不碰真值。
    """
    node_ids: List[str] = []
    candidates: Dict[str, Dict[int, QueuedTask]] = {}
    available: Dict[str, bool] = {}
    remaining: Dict[str, Dict[ResourceUnit, float]] = {}

    by_node: Dict[str, NodeObservation] = {
        node.node_id: node for node, valid
        in zip(observation.nodes, observation.node_valid_mask) if valid}

    pending: Dict[str, Dict[int, List[QueuedTask]]] = {}
    for task in queue.tasks:
        if task.status is not TaskStatus.PENDING:
            continue
        if task.release_time_s > now_s + 1e-9:
            continue
        for action, kind in ACTION_KINDS.items():
            if kind is None or task.kind is not kind:
                continue
            pending.setdefault(task.node_id, {}).setdefault(
                action, []).append(task)

    for node in observation.nodes:
        if node.node_id not in by_node:
            continue
        node_id = node.node_id
        node_ids.append(node_id)
        available[node_id] = bool(node.available)
        remaining[node_id] = {
            unit: float(node.remaining.get(unit.value, 0.0))
            for unit in BUDGET_UNITS}
        picked: Dict[int, QueuedTask] = {}
        for action, tasks in pending.get(node_id, {}).items():
            # 确定性挑选：截止最早的优先，其次 task_id 字典序
            best = sorted(tasks, key=lambda item: (
                item.deadline_s if item.deadline_s is not None else float("inf"),
                item.task_id))[0]
            picked[action] = best
        candidates[node_id] = picked

    return ActionContext(
        node_ids=tuple(node_ids),
        candidates=candidates,
        available=available,
        remaining=remaining,
        processable=dict(processable),
        shareable=dict(shareable),
    )


# ----------------------------------------------------------------------
# 动作 → 标准 ExecutionPlan
# ----------------------------------------------------------------------


@dataclass
class PlanBuildResult:
    plan: Optional[ExecutionPlan]
    chosen: Dict[str, Optional[int]]
    notes: List[str]


def build_plan(
    joint_action: Sequence[int],
    context: ActionContext,
    now_s: float,
    plan_id: str,
    mask: Optional[Sequence[Sequence[bool]]] = None,
) -> PlanBuildResult:
    """把逐节点动作翻译成**标准** `ExecutionPlan`（不执行、不记账）。

    `mask` 给出时用于**记录非法选择**：被选中的动作若不在 mask 内，
    仍然照原样翻译（让执行器去拒绝它），并把原因写进 notes——
    这样"非法动作率"才是**测得**的，而不是靠 mask 假装不存在。
    """
    if len(joint_action) != len(context.node_ids):
        raise ValueError(
            f"动作维度 {len(joint_action)} 与节点数 {len(context.node_ids)} 不一致")
    requests: List[TaskRequest] = []
    chosen: Dict[str, Optional[int]] = {}
    notes: List[str] = []
    for index, node_id in enumerate(context.node_ids):
        action = int(joint_action[index])
        if action not in ACTION_KINDS:
            raise ValueError(f"未知动作 {action}（合法范围 0..{N_ACTIONS - 1}）")
        if mask is not None and not mask[index][action]:
            notes.append(f"{node_id}:{ACTION_NAMES[action]} 不在合法动作内")
        chosen[node_id] = action
        if action == ACTION_IDLE:
            continue
        task = context.candidates.get(node_id, {}).get(action)
        if task is None:
            notes.append(f"{node_id}:{ACTION_NAMES[action]} 无对应候选任务")
            continue
        requests.append(task.to_task_request(start_s=now_s))
    if not requests:
        return PlanBuildResult(plan=None, chosen=chosen, notes=notes)
    plan = ExecutionPlan(plan_id=plan_id, submit_time_s=now_s,
                         tasks=requests,
                         note="[learned] centralized RL resource scheduler")
    return PlanBuildResult(plan=plan, chosen=chosen, notes=notes)


__all__ = [
    "ACTION_IDLE", "ACTION_KINDS", "ACTION_NAMES", "ACTION_PROCESS",
    "ACTION_SAMPLE", "ACTION_SHARE", "ActionContext", "IDLE_ONLY_ROW",
    "N_ACTIONS", "NAME_TO_ACTION", "PlanBuildResult", "build_action_context",
    "build_plan", "pad_mask",
]
