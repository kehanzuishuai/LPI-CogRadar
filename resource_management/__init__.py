"""教学仿真资源管理模块（`resource_management`）。

定位
----
**独立的教学沙盒**：不与真实装备控制接口对接。

* **实验入口完全不引用本模块**——`main.py` / `train_dqn.py` / `evaluate_*.py` /
  `sensitivity_energy_budget.py` 等产生实验数字的脚本里**一个字都不出现**，
  因此"新增本模块没有改变历史实验行为"是**可证明**的，而不是靠声明
  （由 `tests/test_resource_management.py` 与 `verify_v4.py` §15 双向钉住）。
* **验收脚本可以只读引用**：`verify_v4.py` §15 做一致性校验，
  但不驱动本模块产生任何实验数据。

本阶段交付**多节点资源模型 + 统一执行器**：

* 每个传感器节点独立维护可用状态、任务占用区间、采样/处理/通信预算与资源账本；
* `NodeState` / `ResourceBudget` / `TaskRequest` / `ExecutionPlan` /
  `ExecutionResult` 五个统一结构，**一份计划可同时描述多个节点**的
  采样 / 处理 / 共享 / 空闲任务；
* **唯一全局时钟**推进世界（禁止每执行一部雷达就把目标、通信队列
  和其他实体再推进一次）；
* 提交前统一校验（非法对象 / 重复任务 / 占用冲突 / 资源不足），
  执行后逐节点记账；计划级失败**零扣费、零状态污染**；
  节点局部资源不足**只影响该节点**，不结束整个网络任务；
* 空闲**不产生采样报告**，历史估计保留但信息年龄增长。

验收标准（用户明确）
--------------------
> 不是性能提升，而是**多节点执行顺序、时间推进和资源守恒完全一致**。
> 两个节点分别在做什么、用了多少资源、为什么某项任务没有执行，
> 都能从账本追溯。

演示::

    python -m resource_management            # 两节点教学场景 + 账本输出

⚠️ 资源成本是**显式、可配置的教学模型**（`units.TEACHING_COST_MODEL`），
与真实装备效能**不对应**。
"""

from resource_management.clock import ClockError, GlobalClock
from resource_management.observation import (
    CentralObservation,
    CentralObservationStore,
    FIELD_SPECS,
    FixedLengthAdapter,
    LEGACY_OBSERVATION_MODES,
    SCHEMA_VERSION,
    FieldSpec,
    NodeObservation,
    TrackObservation,
    field_metadata,
    node_observation_from_fusion,
    observation_payload,
    observation_truth_violations,
    publish_node_observation,
)
from resource_management.tasks import (
    QUEUE_TASK_KIND_CN,
    DuplicateTaskError,
    QueuedTask,
    QueueTaskKind,
    TaskQueue,
    TaskStatus,
    UnknownObjectError,
)
from resource_management.executor import (
    NODE_LOCAL_REASONS,
    PLAN_FATAL_REASONS,
    UnifiedExecutor,
)
from resource_management.ledger import (
    ENTRY_ACTIVATE,
    ENTRY_CONSUME,
    ENTRY_IDLE,
    ENTRY_REJECT,
    ENTRY_RESERVE,
    LedgerEntry,
    ResourceLedger,
)
from resource_management.model import (
    Estimate,
    ExecutionPlan,
    ExecutionResult,
    NodeState,
    Outcome,
    PlanStatus,
    ProblemScope,
    ResourceBudget,
    TaskOutcome,
    TaskRequest,
    ValidationIssue,
)
from resource_management.units import (
    BUDGET_UNITS,
    DEFAULT_DURATION_S,
    TASK_KIND_CN,
    TEACHING_COST_MODEL,
    UNIT_CN,
    UNIT_MEANING,
    UNIT_SYMBOL,
    ResourceUnit,
    TaskKind,
    TeachingCost,
    format_cost,
)

__all__ = [
    "BUDGET_UNITS", "CentralObservation", "CentralObservationStore",
    "DuplicateTaskError", "FIELD_SPECS", "FieldSpec",
    "FixedLengthAdapter", "LEGACY_OBSERVATION_MODES",
    "NodeObservation", "QUEUE_TASK_KIND_CN", "QueuedTask",
    "QueueTaskKind", "SCHEMA_VERSION", "TaskQueue", "TaskStatus",
    "TrackObservation", "UnknownObjectError",
    "field_metadata", "node_observation_from_fusion",
    "observation_payload", "observation_truth_violations",
    "publish_node_observation", "ClockError", "DEFAULT_DURATION_S", "ENTRY_ACTIVATE",
    "ENTRY_CONSUME", "ENTRY_IDLE", "ENTRY_REJECT", "ENTRY_RESERVE",
    "Estimate", "ExecutionPlan", "ExecutionResult", "GlobalClock",
    "LedgerEntry", "NODE_LOCAL_REASONS", "NodeState", "Outcome",
    "PLAN_FATAL_REASONS", "PlanStatus", "ProblemScope", "ResourceBudget",
    "ResourceLedger", "ResourceUnit", "TASK_KIND_CN", "TEACHING_COST_MODEL",
    "TaskKind", "TaskOutcome", "TaskRequest", "TeachingCost", "UNIT_CN",
    "UNIT_MEANING", "UNIT_SYMBOL", "UnifiedExecutor", "ValidationIssue",
    "format_cost",
]
