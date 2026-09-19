"""非学习优化参考（v4.5 大阶段一收尾）。

⚠️ 概念边界（必须先说清楚，用户明确纠正过这一点）
-------------------------------------------------
**「有前瞻的参考方法」不自动等于「理论上界」。**

* 只有在**被完全枚举**的小问题上求出的解，才可以称"该问题的精确最优"；
* 滚动规划、束搜索、任何带启发式的推演，**只能**称"优化参考"，
  **不得**被称为全局最优或理论上界。

本模块把这条规则写成**数据**而不是注释：每个结果都带 `exact` 与
`optimality_claim` 两个字段，且 `exact=True` 只可能由
"组合数已全部枚举完 **且** 计算预算未被触到"这一条路径设置。
`_search()` 里任何一次提前退出都会把 `exact` 置回 False。

优化参考与规则基线的关系
------------------------
**共用**同一套东西（这是可比性的前提）：

| 共用项 | 实现方式 |
| --- | --- |
| 任务定义 | 同一个 `TaskQueue` / `QueuedTask`（含成本、截止、去重键） |
| 资源预算 | 观测里的 `capacity` / `remaining`（**只读**，不碰账本） |
| 信息权限 | 只读 `CentralObservation`；不 import `engine`/`sensor`/`fusion`（AST 钉住） |
| 执行器 | 输出同一个标准 `ExecutionPlan`，由 `UnifiedExecutor` 校验记账 |
| 候选语义 | 继承 `SchedulerBase._classify_due` 与 `_duplicate_rule` |

**不共用**的只有"挑哪一条"：规则基线用打分排序，优化参考用
"显式预测模型 + 声明式目标 + 有预算的搜索"。

不得读取未来的四类信息
----------------------
1. **未来真实测量**：预测只能用**当前可见观测**外推（`PredictionModel`）；
2. **未来故障**：不可用窗口不进入预测（模型里没有这个东西）；
3. **隐藏对象状态**：只能引用 `observation.track_ids()` 里的键；
4. **真值**：模块不 import 真值层，也不接受 `Simulator`。

需要未来时只能走显式预测模型，且**假设必须随结果一起落盘**
（`OptimizationOutcome.assumptions`）。

评价向量（6 项，禁止单一不透明综合分）
--------------------------------------
`EVALUATION_METRICS` 定义 6 个维度（完成度/及时性/估计质量/资源消耗/
通信开销/计算耗时），每项都带单位、方向与**逐字定义**。
`ObjectiveSpec` 是**声明式**标量化（显式权重 + 显式归一化尺度），
它只用来在搜索内部排序；**报告一律给完整向量**，不给综合分。
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.model import ExecutionPlan, TaskRequest
from resource_management.observation import CentralObservation, NodeObservation
from resource_management.scheduling import (
    DECISION_DEFERRED,
    DECISION_PLANNED,
    OPTIMIZATION_POLICIES,
    POLICY_CN,
    SchedulerBase,
    SchedulingConfig,
    SchedulingDecision,
    SelectionResult,
)
from resource_management.tasks import (
    QueueTaskKind,
    QueuedTask,
    TaskStatus,
)
from resource_management.units import BUDGET_UNITS, ResourceUnit


# ----------------------------------------------------------------------
# ① 显式预测模型
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionAssumption:
    """一条预测假设：必须能被逐条指出"这是模型说的，不是观测到的"。"""

    name: str
    statement: str
    formula: str
    validity_limit: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "statement": self.statement,
            "formula": self.formula,
            "validity_limit": self.validity_limit,
        }


#: 预测模型用到的教学常数（**全部显式**，不是拟合出来的）
PREDICTION_TICK_S: float = 1.0
#: 完成一次采样/更新后，位置标准差回到的典型值（米）
SERVICED_SIGMA_M: float = 20.0
#: 未被服务时位置标准差每秒增长量（米/秒）
SIGMA_GROWTH_MPS: float = 25.0

PREDICTION_ASSUMPTIONS: Tuple[PredictionAssumption, ...] = (
    PredictionAssumption(
        name="age_growth",
        statement="未被服务的航迹，其信息年龄按 1 秒/秒线性增长",
        formula="age(t + k) = age(t) + k · Δt，Δt = 1 s",
        validity_limit="航迹仍可见；航迹消失不建模（观测里没有的东西不预测）",
    ),
    PredictionAssumption(
        name="sigma_growth",
        statement="未被服务的航迹，位置标准差按固定速率增长",
        formula=f"σ(t + k) = √(σ(t)² + (k · Δt · {SIGMA_GROWTH_MPS:g})²)",
        validity_limit=f"速率取教学常数 {SIGMA_GROWTH_MPS:g} m/s，非实测标定值",
    ),
    PredictionAssumption(
        name="sigma_after_service",
        statement="一次采样或估计更新后，位置标准差回到典型值",
        formula=f"σ ← {SERVICED_SIGMA_M:g} m",
        validity_limit="忽略单次测量的实际精度差异（预测模型不模拟传感器）",
    ),
    PredictionAssumption(
        name="persistent_tracks",
        statement="当前可见的航迹在预测时域内**始终可见**",
        formula="visible(t + k) = visible(t)",
        validity_limit=(
            "这是**乐观**假设：真实覆盖交接会让航迹离开视野。"
            "因此优化参考的高估风险来自这里，报告里必须一并给出该假设"),
    ),
    PredictionAssumption(
        name="arrival_rate",
        statement="未来每 tick 新到达的任务数量 = 当前 tick 观测到的到达数量",
        formula="arrivals(t + k) = arrivals(t)，按 (节点, 任务类型) 分别统计",
        validity_limit="只能外推已经出现过的任务类型，不预测没见过的对象",
    ),
    PredictionAssumption(
        name="no_future_events",
        statement="**不预测**任何未来故障、未来不可用窗口或未来链路中断",
        formula="unavailable(t + k) = []",
        validity_limit=(
            "用户要求：优化参考不得通过读取未来故障取得优势。"
            "模型里根本没有这条信息——不是「忽略」，是**不可见**"),
    ),
    PredictionAssumption(
        name="deadline_expiry",
        statement="任务在截止时刻之后失去意义（与真实过期语义一致）",
        formula="missed 若 t_served > deadline_s（或时域结束仍未服务）",
        validity_limit="与冻结的任务完成口径一致（见 resource_contract_v1）",
    ),
)


class PredictionModel:
    """**只用当前可见观测**做外推的显式预测模型。

    它不接受 `Simulator`、不接受真值、不接受未来事件——接口上就传不进来。
    """

    def __init__(self, horizon_ticks: int = 1,
                 tick_s: float = PREDICTION_TICK_S) -> None:
        if horizon_ticks < 1:
            raise ValueError("horizon_ticks 至少为 1")
        self.horizon_ticks = int(horizon_ticks)
        self.tick_s = float(tick_s)

    def assumptions(self) -> List[Dict[str, str]]:
        return [item.to_dict() for item in PREDICTION_ASSUMPTIONS]

    def describe(self) -> Dict[str, Any]:
        return {
            "horizon_ticks": self.horizon_ticks,
            "tick_s": self.tick_s,
            "constants": {
                "serviced_sigma_m": SERVICED_SIGMA_M,
                "sigma_growth_mps": SIGMA_GROWTH_MPS,
            },
            "assumptions": self.assumptions(),
            "information_boundary": (
                "预测只用当前可见观测：不读未来真实测量、不读未来故障、"
                "不读隐藏对象状态、不读真值。"),
        }

    # --- 单量外推 ---

    def age_after(self, age_s: float, ticks: int) -> float:
        return max(0.0, float(age_s) + ticks * self.tick_s)

    def sigma_after(self, sigma_m: float, ticks: int) -> float:
        if ticks <= 0:
            return float(sigma_m)
        grown = ticks * self.tick_s * SIGMA_GROWTH_MPS
        return math.sqrt(float(sigma_m) ** 2 + grown ** 2)

    def arrival_rate(self, observation: CentralObservation,
                     queue: Any, now_s: float) -> Dict[Tuple[str, str], int]:
        """当前 tick 观测到的任务到达速率（按 节点 × 任务类型 统计）。

        只数**已经出现过**的任务，且只数 `release_time_s == now` 的那一批——
        这两条限制保证它不引入任何未来信息。
        """
        rates: Dict[Tuple[str, str], int] = {}
        for task in getattr(queue, "tasks", []):
            if abs(float(task.release_time_s) - float(now_s)) > 1e-9:
                continue
            key = (task.node_id, task.kind.value)
            rates[key] = rates.get(key, 0) + 1
        return rates


# ----------------------------------------------------------------------
# ② 计算预算
# ----------------------------------------------------------------------


@dataclass
class ComputeBudget:
    """计算预算：**防止某个方法无限推演**。

    三个维度都要有，缺一个都可能被绕过：

    * `time_limit_s`：墙钟上限（每 tick 一次规划）；
    * `max_expansions`：候选计划评估次数的上限（保证组合爆炸时能停下来）；
    * `max_horizon_ticks`：预测时域上限（防止"往前推 1000 步"式作弊）。
    """

    #: 墙钟上限（每次规划）。**这是安全阀，不是确定性预算。**
    #:
    #: 用户要求"设置计算预算，防止某个方法无限推演"，同时也要求
    #: "规则与优化参考可复现"。这两条会**冲突**：以墙钟为界的方法，
    #: 在机器负载不同时会在不同的展开点停下——实测同一配置连跑四次，
    #: 展开数分别是 1315 / 1270 / 1312 / 1292，计划也因此可能不同。
    #:
    #: 因此本字段的角色是**兜底**：默认取值大到在本沙盒里不会触发
    #: （确定性由 `max_expansions` 与 `max_horizon_ticks` 保证）。
    #: 一旦它真的触发，`deterministic` 会置 False 并如实入账——
    #: 那次结果**不允许**被当作可复现基准。
    #: 取 None 表示完全关闭时间兜底（只靠确定性上限）。
    time_limit_s: Optional[float] = 5.0
    max_expansions: int = 1024
    max_horizon_ticks: int = 3
    #: 束搜索时每节点保留的候选数（仅在无法完全枚举时生效）
    beam_width_per_node: int = 3
    #: 帕累托前沿的**后处理上限**（参与两两比较的向量数）。
    #:
    #: 帕累托是非支配筛选，代价是 O(n²)：实测 713 个向量要 0.022s，
    #: 1140 个就明显拖慢单次规划。预算必须覆盖它，否则
    #: "计算预算"只约束了搜索循环、没约束后处理，就是个假预算。
    pareto_max_points: int = 128

    def __post_init__(self) -> None:
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s 必须为正或 None")
        if self.max_expansions < 1:
            raise ValueError("max_expansions 至少为 1")
        if self.max_horizon_ticks < 1:
            raise ValueError("max_horizon_ticks 至少为 1")
        if self.beam_width_per_node < 1:
            raise ValueError("beam_width_per_node 至少为 1")
        if self.pareto_max_points < 1:
            raise ValueError("pareto_max_points 至少为 1")
        self.reset()

    # --- 运行时状态（不属于"配置"，但连同结果一起落盘）---

    expansions: int = 0
    started_at: float = 0.0
    exhausted: bool = False
    exhausted_reason: str = ""
    #: 墙钟安全阀**是否真的触发过**（触发即意味着该次结果不可复现）
    time_limit_triggered: bool = False

    def reset(self) -> None:
        self.expansions = 0
        self.started_at = time.perf_counter()
        self.exhausted = False
        self.exhausted_reason = ""
        self.time_limit_triggered = False

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    def spend(self, count: int = 1) -> bool:
        """记账一次展开；返回 False 表示**预算已耗尽，必须停**。"""
        self.expansions += count
        if self.expansions > self.max_expansions:
            self.exhausted = True
            self.exhausted_reason = (
                f"候选计划评估次数超过上限 {self.max_expansions}")
            return False
        if self.time_limit_s is not None and self.elapsed_s() > self.time_limit_s:
            self.exhausted = True
            self.time_limit_triggered = True
            self.exhausted_reason = (
                f"规划耗时超过上限 {self.time_limit_s:g}s"
                "（墙钟安全阀触发 → 该次结果**不可复现**）")
            return False
        return True

    @property
    def deterministic(self) -> bool:
        """该次搜索是否不受墙钟影响（只有不受影响才谈可复现）。"""
        return not self.time_limit_triggered

    def clamp_horizon(self, horizon_ticks: int) -> Tuple[int, bool]:
        """把预测时域裁到预算内；返回 (实际时域, 是否被裁剪)。"""
        if horizon_ticks > self.max_horizon_ticks:
            return self.max_horizon_ticks, True
        return int(horizon_ticks), False

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.elapsed_s()
        overrun = (max(0.0, elapsed - self.time_limit_s)
                   if self.time_limit_s is not None else 0.0)
        return {
            "time_limit_s": self.time_limit_s,
            "max_expansions": self.max_expansions,
            "max_horizon_ticks": self.max_horizon_ticks,
            "beam_width_per_node": self.beam_width_per_node,
            "pareto_max_points": self.pareto_max_points,
            "expansions": self.expansions,
            "elapsed_s": round(elapsed, 9),
            "exhausted": self.exhausted,
            "exhausted_reason": self.exhausted_reason,
            "time_limit_triggered": self.time_limit_triggered,
            # 只有墙钟安全阀没触发时，这次搜索才是**可复现**的
            "deterministic": self.deterministic,
            # 超支量：预算检查是在每次展开**之前**做的，
            # 因此最后一次展开与后处理（排序/帕累托）可能把总时长顶过上限。
            # 如实记录，而不是声称"从不超预算"。
            "overrun_s": round(overrun, 9),
        }


# ----------------------------------------------------------------------
# ③ 统一评价向量（6 项）
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """一个评价维度：**逐字定义**，不留解释空间。"""

    key: str
    name_cn: str
    unit: str
    #: "higher"（越大越好）或 "lower"（越小越好）
    direction: str
    definition: str
    #: 标量化时的归一化尺度（与方向配合：score = weight · align(value)/scale）
    scale: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name_cn": self.name_cn, "unit": self.unit,
            "direction": self.direction, "definition": self.definition,
            "scale": self.scale,
        }


EVALUATION_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        key="service_completion", name_cn="服务完成度", unit="比值",
        direction="higher", scale=1.0,
        definition=("已完成任务数 / (已完成 + 过期 + 主动放弃 + 执行器拒绝 + "
                    "长期未获服务)。分母与被冻结的任务完成口径一致，"
                    "**主动放弃不缩小分母**。"),
    ),
    MetricSpec(
        key="task_timeliness", name_cn="任务及时性", unit="比值",
        direction="higher", scale=1.0,
        definition=("截止时间前完成的任务数 / 已完成任务数；没有已完成任务时记 0。"
                    "只统计**真的完成**的任务，因此它回答「做成的那些及时吗」，"
                    "与完成度互补、不可互相替代。"),
    ),
    MetricSpec(
        key="estimate_quality", name_cn="估计质量", unit="1/(1+秒)",
        direction="higher", scale=1.0,
        definition=("1 / (1 + 可见航迹的平均信息年龄)。"
                    "取值落在 (0,1]，年龄 0 时为 1。"
                    "用这个形式是为了避免「年龄无上界」导致量纲失控；"
                    "**这个函数形式是声明的约定，不是拟合出来的**。"),
    ),
    MetricSpec(
        key="resource_consumption", name_cn="资源消耗", unit="容量占比",
        direction="lower", scale=1.0,
        definition=("对三类预算单位各算 consumed/capacity，再取算术平均。"
                    "逐单位的明细必须一并给出（见 per_unit），"
                    "避免「平均」把某一类单位吃穿的事实掩盖掉。"),
    ),
    MetricSpec(
        key="communication_overhead", name_cn="通信开销", unit="字节",
        direction="lower", scale=1024.0,
        definition="通信字节的实际消耗总量（来自账本，不从计划反推）。",
    ),
    MetricSpec(
        key="compute_time", name_cn="计算耗时", unit="秒",
        direction="lower", scale=1.0,
        definition=("本方法在该次运行中花在**规划**上的墙钟时间总和"
                    "（不含执行、不含仿真推进）。"
                    "它是优化参考的**代价项**，必须与收益一起报告。"),
    ),
)

METRIC_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in EVALUATION_METRICS}
METRIC_KEYS: Tuple[str, ...] = tuple(m.key for m in EVALUATION_METRICS)


@dataclass
class EvaluationVector:
    """**多目标评价向量**：6 项一起给，不给综合分。

    综合分（`ObjectiveSpec.score`）只是搜索内部排序用的**声明式偏好**，
    不是评价结果，也不允许被当成"方法好不好"的结论。
    """

    values: Dict[str, float] = field(default_factory=dict)
    #: 逐单位资源明细（`resource_consumption` 的支撑材料）
    per_unit: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: 该向量的来源说明（"实测"还是"预测"）——两者绝不能混为一谈
    provenance: str = "measured"
    #: 支撑材料：例如预测完成度用的是哪个口径的分母
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in METRIC_KEYS:
            self.values.setdefault(key, 0.0)

    def aligned(self, key: str) -> float:
        """按方向对齐到"越大越好"的坐标（仅用于比较/标量化）。"""
        spec = METRIC_BY_KEY[key]
        value = float(self.values.get(key, 0.0))
        return value if spec.direction == "higher" else -value

    def dominates(self, other: "EvaluationVector",
                  tol: float = 1e-12) -> bool:
        """帕累托支配：所有维度不差，且至少一维更好（方向已对齐）。"""
        better = False
        for key in METRIC_KEYS:
            mine, theirs = self.aligned(key), other.aligned(key)
            if mine < theirs - tol:
                return False
            if mine > theirs + tol:
                better = True
        return better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": {k: round(float(v), 9) for k, v in self.values.items()},
            "units": {k: METRIC_BY_KEY[k].unit for k in METRIC_KEYS},
            "directions": {k: METRIC_BY_KEY[k].direction for k in METRIC_KEYS},
            "per_unit": {k: dict(v) for k, v in self.per_unit.items()},
            "provenance": self.provenance,
            "detail": dict(self.detail),
            "note": ("这是**向量**评价，6 项必须一起读；"
                     "报告里不得只给综合分。"),
        }


def _is_canonical_claim(outcome: "OptimizationOutcome") -> bool:
    """声明文本是否为规范文本的**逐字**渲染结果（白名单校验）。

    判定方式是把渲染参数也存进结果里再逐字比对，而不是回猜参数——
    回猜会让审计在"自定义停止原因"这类分支上误报。
    """
    template = CANONICAL_CLAIMS.get(outcome.claim_kind)
    if template is None:
        return False
    try:
        return outcome.optimality_claim == template.format(
            **outcome.claim_args)
    except (KeyError, IndexError):
        return False


def pareto_frontier(vectors: Sequence[EvaluationVector]
                    ) -> List[EvaluationVector]:
    """返回不被任何其他向量支配的向量集合（保持输入顺序）。"""
    frontier: List[EvaluationVector] = []
    for index, candidate in enumerate(vectors):
        if any(other.dominates(candidate) for other in vectors
               if other is not candidate):
            continue
        # 去掉与已有元素完全等价的重复项（逐维相等）
        duplicate = any(
            all(abs(candidate.values[k] - kept.values[k]) <= 1e-12
                for k in METRIC_KEYS)
            for kept in frontier)
        if not duplicate:
            frontier.append(candidate)
        _ = index
    return frontier


# ----------------------------------------------------------------------
# ④ 声明式目标（不是不透明综合分）
# ----------------------------------------------------------------------


@dataclass
class ObjectiveSpec:
    """**声明式**标量化：显式权重 + 各维度显式归一化尺度。

    为什么需要它：搜索必须有一个排序依据。但它**不是**评价：
    它是一份被写下来的偏好声明，改权重就换一个目标，结论随之改变，
    这一点必须在报告里说清楚。

    默认权重把"计算耗时"设为 0，理由要说明白：同一次规划里所有候选计划的
    计算耗时是同一个数（都是这一次调用的耗时），它对**候选之间**的排序没有
    区分能力。它仍然作为独立维度被记录和对照，只是不参与内部排序。
    """

    weights: Dict[str, float] = field(default_factory=lambda: {
        "service_completion": 1.0,
        "task_timeliness": 0.5,
        "estimate_quality": 0.5,
        "resource_consumption": -0.3,
        "communication_overhead": -0.2,
        "compute_time": 0.0,
    })
    normalize: Dict[str, float] = field(default_factory=lambda: {
        "service_completion": 1.0,
        "task_timeliness": 1.0,
        "estimate_quality": 1.0,
        "resource_consumption": 1.0,
        "communication_overhead": 1024.0,
        "compute_time": 1.0,
    })

    def validate(self) -> None:
        missing = [k for k in METRIC_KEYS if k not in self.weights]
        if missing:
            raise ValueError(f"ObjectiveSpec.weights 缺少维度 {missing}")
        missing = [k for k in METRIC_KEYS if k not in self.normalize]
        if missing:
            raise ValueError(f"ObjectiveSpec.normalize 缺少维度 {missing}")

    def score(self, vector: EvaluationVector) -> float:
        self.validate()
        total = 0.0
        for key in METRIC_KEYS:
            spec = METRIC_BY_KEY[key]
            raw = float(vector.values.get(key, 0.0))
            aligned = raw if spec.direction == "higher" else -raw
            scale = float(self.normalize.get(key, 1.0)) or 1.0
            total += float(self.weights[key]) * aligned / scale
        return total

    def describe(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "normalize": dict(self.normalize),
            "note": ("这是**声明的偏好**，不是「唯一正确的目标」；"
                     "换一组权重就会换一个最优解。综合分只用于搜索内部排序，"
                     "报告一律给 6 维向量。"),
            "compute_time_weight_is_zero_reason": (
                "同一次规划内所有候选的计算耗时相同，对候选排序无区分能力；"
                "它仍作为独立维度记录与对照。"),
        }


# ----------------------------------------------------------------------
# ⑤ 显式后果模型（预测向量，不是实测向量）
# ----------------------------------------------------------------------


@dataclass
class ConsequenceModel:
    """候选计划的**后果模型**：显式、可读、只用观测与队列。

    它模拟的是"如果这么排，未来 H 个 tick 会怎样"，用的是
    `PredictionModel` 的假设。产出的是**预测向量**（`provenance="predicted"`），
    与执行后的**实测向量**（`provenance="measured"`）严格分开——
    两者混在一起就会出现"用预测的好处和实测的代价对比"这种错误结论。
    """

    prediction: PredictionModel
    objective: ObjectiveSpec
    horizon_ticks: int = 1
    #: 后续 tick 的 rollout 策略（声明式，不是最优）
    rollout: str = "greedy_single_task_by_objective"

    def evaluate(
        self,
        chosen: Sequence[QueuedTask],
        candidates: Sequence[QueuedTask],
        by_node: Dict[str, NodeObservation],
        now_s: float,
    ) -> EvaluationVector:
        """评估"本 tick 选 chosen"在 H 个 tick 内的后果。"""
        horizon = max(1, self.horizon_ticks)
        tick = self.prediction.tick_s

        # --- 状态：全部从观测初始化（只读）---
        remaining: Dict[str, Dict[ResourceUnit, float]] = {}
        capacity: Dict[str, Dict[ResourceUnit, float]] = {}
        for node_id, node in by_node.items():
            capacity[node_id] = {u: float(node.capacity.get(u.value, 0.0))
                                 for u in BUDGET_UNITS}
            remaining[node_id] = {u: float(node.remaining.get(u.value, 0.0))
                                  for u in BUDGET_UNITS}

        ages: Dict[Tuple[str, str], float] = {}
        sigmas: Dict[Tuple[str, str], float] = {}
        for node_id, node in by_node.items():
            for track, valid in zip(node.tracks, node.track_valid_mask):
                if not valid:
                    continue
                ages[(node_id, track.track_id)] = float(track.information_age_s)
                sigmas[(node_id, track.track_id)] = float(
                    max(track.sigma_position))

        consumed: Dict[str, Dict[ResourceUnit, float]] = {
            node_id: {u: 0.0 for u in BUDGET_UNITS} for node_id in by_node}

        # --- 任务池：当前候选（每 tick 只能做一条/节点）---
        pending: List[QueuedTask] = list(candidates)
        # 每条任务的"已被服务时刻"（用于完成/过期判定）
        served_at: Dict[str, float] = {}
        first_tick_choice = {id(task) for task in chosen}
        #: 逐 tick 的估计质量样本（见下方"期末效应"说明）
        quality_samples: List[float] = []

        for step in range(1, horizon + 1):
            t = now_s + step * tick
            # 本 tick 的处置：首 tick 用给定的选择，之后用声明式 rollout
            if step == 1:
                pick = [task for task in pending
                        if id(task) in first_tick_choice]
            else:
                pick = self._rollout(pending, by_node, ages, sigmas, t, served_at)
            for task in pick:
                budget = remaining.get(task.node_id)
                if budget is None:
                    continue
                cost = task.estimated_cost
                if any(float(cost.get(u, 0.0) or 0.0) > budget[u] + 1e-9
                       for u in BUDGET_UNITS):
                    continue                    # 资源不足 → 预测里也不执行
                for u in BUDGET_UNITS:
                    spend = float(cost.get(u, 0.0) or 0.0)
                    budget[u] -= spend
                    consumed[task.node_id][u] += spend
                served_at[task.task_id] = t
                self._apply_effect(task, ages, sigmas, t)

            # 时间推进：未服务的航迹变旧
            for node_id in by_node:
                for key in list(ages):
                    if key[0] != node_id:
                        continue
                    if key in {(task.node_id, target)
                               for task in pick for target in task.targets}:
                        continue
                    ages[key] = self.prediction.age_after(ages[key], 1)
                    sigmas[key] = self.prediction.sigma_after(sigmas[key], 1)

            # 逐 tick 采样估计质量
            step_mean_age = (sum(ages.values()) / len(ages)) if ages else 0.0
            quality_samples.append(1.0 / (1.0 + step_mean_age))

        # --- 汇总预测向量 ---
        #
        # ⚠️ 完成度只统计**时域内可判定去留**的任务，这是一个必须写明的口径：
        # 每个 tick 只能服务"节点数 × 1"条，而候选池有 20+ 条，
        # 若分母取整个候选池，则无论首 tick 选谁，比值都≈2/25，
        # **对选择完全不敏感**——实测后果是资源罚项一家独大，
        # 优化参考学会"什么都不做"（24 个 tick 有 13 次 plan=None）。
        # 因此分母只算**截止时间落在预测时域内**的任务：这些任务在时域内
        # 要么被救下、要么作废，首 tick 的选择对它们的去留是决定性的。
        horizon_end = now_s + horizon * tick
        resolvable = [
            task for task in candidates
            if task.deadline_s is not None
            and task.deadline_s <= horizon_end + 1e-9]
        served_ids = set(served_at)
        served_resolvable = [task for task in resolvable
                             if task.task_id in served_ids]
        served = [task for task in candidates if task.task_id in served_ids]
        missed = [task for task in candidates if task.task_id not in served_ids]
        if resolvable:
            n_served, denominator = len(served_resolvable), len(resolvable)
            completion_scope = "deadline_within_horizon"
        else:
            n_served, denominator = len(served), len(served) + len(missed)
            completion_scope = "full_candidate_pool"
        n_missed = denominator - n_served
        on_time = sum(
            1 for task in served_resolvable
            if task.deadline_s is None
            or served_at[task.task_id] <= task.deadline_s + 1e-9)
        if not resolvable:
            on_time = sum(
                1 for task in served
                if task.deadline_s is None
                or served_at[task.task_id] <= task.deadline_s + 1e-9)
        mean_age = (sum(ages.values()) / len(ages)) if ages else 0.0
        # **估计质量取时域内的平均，不取期末值。**
        #
        # 第一版只取时域末尾的年龄，结果出现典型的**期末效应**：
        # "现在刷新"要让年龄再涨满整个时域才被测量，而"晚一个 tick 再刷新"
        # 反而在末尾更新鲜——模型于是在**奖励拖延**。
        # 实测后果：束搜索里"什么都不做"压过了合法的更新任务。
        # 取时域平均后，早刷新严格不差于晚刷新。
        horizon_quality = ((sum(quality_samples) / len(quality_samples))
                           if quality_samples else 1.0 / (1.0 + mean_age))
        per_unit: Dict[str, Dict[str, float]] = {}
        ratios: List[float] = []
        for node_id in consumed:
            per_unit[node_id] = {}
            for u in BUDGET_UNITS:
                cap = capacity[node_id].get(u, 0.0)
                used = consumed[node_id].get(u, 0.0)
                per_unit[node_id][u.value] = round(used, 9)
                per_unit[node_id][f"{u.value}_capacity"] = round(cap, 9)
                if cap > 0:
                    ratios.append(used / cap)
        comm = sum(consumed[node_id].get(ResourceUnit.COMM_BYTE, 0.0)
                   for node_id in consumed)
        return EvaluationVector(
            values={
                "service_completion": (n_served / denominator)
                if denominator else 0.0,
                "task_timeliness": (on_time / n_served) if n_served else 0.0,
                "estimate_quality": horizon_quality,
                "resource_consumption": (sum(ratios) / len(ratios))
                if ratios else 0.0,
                "communication_overhead": comm,
                "compute_time": 0.0,
            },
            per_unit=per_unit,
            provenance="predicted",
            detail={
                "completion_scope": completion_scope,
                "n_resolvable": len(resolvable),
                "horizon_end_s": round(horizon_end, 6),
                "n_served": n_served,
                "n_missed": n_missed,
                "estimate_quality_scope": "horizon_mean",
                "terminal_mean_age_s": round(mean_age, 6),
            },
        )

    # ------------------------------------------------------------------

    def _apply_effect(self, task: QueuedTask,
                      ages: Dict[Tuple[str, str], float],
                      sigmas: Dict[Tuple[str, str], float],
                      t: float) -> None:
        """任务执行后的状态效果（显式、逐类型）。"""
        if task.kind in (QueueTaskKind.PREDEFINED_SAMPLE,
                         QueueTaskKind.ESTIMATE_UPDATE):
            for target in task.targets:
                key = (task.node_id, target)
                if key in ages:
                    ages[key] = 0.0
                    sigmas[key] = SERVICED_SIGMA_M
        # PROCESS / SHARE 只消耗资源，不刷新估计（与教学成本模型一致）

    def _rollout(self, pending: Sequence[QueuedTask],
                 by_node: Dict[str, NodeObservation],
                 ages: Dict[Tuple[str, str], float],
                 sigmas: Dict[Tuple[str, str], float],
                 t: float,
                 served_at: Dict[str, float]) -> List[QueuedTask]:
        """声明式 rollout：每个节点挑"单条边际目标值最高"的任务。

        **这不是最优**——它只是一条写下来的、可复现的推演规则。
        因此带 lookahead 的结果一律 `exact=False`。
        """
        pick: List[QueuedTask] = []
        for node_id in by_node:
            best: Optional[Tuple[float, str, QueuedTask]] = None
            for task in pending:
                if task.node_id != node_id or task.task_id in served_at:
                    continue
                if task.deadline_s is not None and t > task.deadline_s + 1e-9:
                    continue
                value = self._single_task_value(task, ages, sigmas)
                key = (value, task.task_id)
                if best is None or key > (best[0], best[1]):
                    best = (value, task.task_id, task)
            if best is not None:
                pick.append(best[2])
        return pick

    def _single_task_value(self, task: QueuedTask,
                           ages: Dict[Tuple[str, str], float],
                           sigmas: Dict[Tuple[str, str], float]) -> float:
        """单条任务的边际价值（显式）：刷新估计 + 赶在截止前 + 少花资源。"""
        value = 0.0
        for target in task.targets:
            key = (task.node_id, target)
            if key not in ages:
                continue
            age = ages[key]
            sigma = sigmas.get(key, 0.0)
            value += (min(1.0, age / 3.0) + min(1.0, sigma / 150.0))
        cost = task.estimated_cost
        value -= 1e-4 * float(cost.get(ResourceUnit.COMM_BYTE, 0.0) or 0.0)
        value -= 1e-3 * float(cost.get(ResourceUnit.SAMPLE_SLOT, 0.0) or 0.0)
        return value


# ----------------------------------------------------------------------
# ⑥ 搜索结果容器
# ----------------------------------------------------------------------


class OptimizerKind(str, Enum):
    ENUMERATION = "enumeration"
    ROLLING_HORIZON = "rolling_horizon"


#: 最优性声明的**规范文本**：`optimality_claim` 只允许是这四个之一。
#:
#: 为什么要做「白名单 + 结构化类型」而不是关键词扫描：第一版的自审用
#: 「声明里是否出现『理论上界』」来判断，结果 `CANONICAL_CLAIMS` 里那句
#: **否定**表述（"不是整个调度问题的理论上界"）被误判成"自称理论上界"。
#: 关键词扫描既会误报、也能被措辞绕过，所以改成：
#: ① 文本必须**逐字**等于规范文本的渲染结果；② 另有一个结构化 `claim_kind`
#: 必须与 `exact` 一致。
CLAIM_KIND_EXACT = "exact_small_problem"
CLAIM_KIND_LOOKAHEAD = "lookahead_reference"
CLAIM_KIND_BEAM = "beam_reference"
CLAIM_KIND_BUDGET = "budget_limited"

CANONICAL_CLAIMS: Dict[str, str] = {
    CLAIM_KIND_EXACT: (
        "该单 tick 决策问题已**完全枚举**（{n} 个候选计划全部评估），"
        "因此这个解是**该问题在该声明式目标下的精确最优**。"
        "只对这个被枚举的小问题成立；它**不是**整个调度问题的理论上界。"),
    CLAIM_KIND_LOOKAHEAD: (
        "首 tick 的选择已完全枚举，后续 tick 用**声明式 rollout** 推演，"
        "因此这是**优化参考**：比规则基线多用了前瞻，但**没有**全局最优性证明。"),
    CLAIM_KIND_BEAM: (
        "组合数超出计算预算，已改用**束搜索**，因此这是**优化参考**，"
        "既不是全局最优也不构成理论上界。"),
    CLAIM_KIND_BUDGET: (
        "计算预算在枚举完成前耗尽（{reason}），返回的是"
        "**已评估部分中的最好解**，不构成任何最优性声明。"),
}


@dataclass
class OptimizationOutcome:
    """一次规划的完整交代：计划 + 目标值 + 向量 + 最优性声明 + 预算 + 假设。"""

    kind: str
    #: 是否**完全枚举**了该问题（只有它为真才谈得上"精确最优"）
    exact: bool
    #: 结构化声明类型（必须是 `CANONICAL_CLAIMS` 的键之一），
    #: 与 `exact` 必须一致：`exact=True` ⟺ `claim_kind == CLAIM_KIND_EXACT`
    claim_kind: str = CLAIM_KIND_BUDGET
    #: 最优性声明原文——必须**逐字**等于 `CANONICAL_CLAIMS[claim_kind]` 渲染结果。
    #: 用白名单而非关键词扫描：关键词既会误报（否定句）也能被措辞绕过。
    optimality_claim: str = ""
    #: 渲染声明文本用到的**参数**（审计要逐字比对，不能靠猜参数）
    claim_args: Dict[str, Any] = field(default_factory=dict)
    objective: float = 0.0
    vector: Optional[EvaluationVector] = None
    n_candidates: int = 0
    n_combinations_total: int = 0
    n_combinations_evaluated: int = 0
    n_combinations_infeasible: int = 0
    best_plan: List[str] = field(default_factory=list)
    runner_up: Optional[Dict[str, Any]] = None
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    #: 帕累托前沿是否因预算被截断（截断了就只是"近似前沿"）
    pareto_truncated: bool = False
    budget: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    objective_spec: Dict[str, Any] = field(default_factory=dict)
    explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "exact": self.exact,
            "claim_kind": self.claim_kind,
            "optimality_claim": self.optimality_claim,
            "claim_args": dict(self.claim_args),
            "objective": round(float(self.objective), 9),
            "vector": self.vector.to_dict() if self.vector else None,
            "n_candidates": self.n_candidates,
            "n_combinations_total": self.n_combinations_total,
            "n_combinations_evaluated": self.n_combinations_evaluated,
            "n_combinations_infeasible": self.n_combinations_infeasible,
            "best_plan": list(self.best_plan),
            "runner_up": self.runner_up,
            "pareto_frontier": list(self.pareto_frontier),
            "pareto_truncated": self.pareto_truncated,
            "n_pareto_points": len(self.pareto_frontier),
            "budget": dict(self.budget),
            "prediction": dict(self.prediction),
            "objective_spec": dict(self.objective_spec),
            "explanations": dict(self.explanations),
        }


# ----------------------------------------------------------------------
# ⑦ 优化参考实现
# ----------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """优化参考的配置（与规则基线的 `SchedulingConfig` 分开，职责不混）。"""

    kind: OptimizerKind = OptimizerKind.ENUMERATION
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    #: 预测时域（1 = 只规划当前 tick）
    horizon_ticks: int = 1
    #: 是否计算帕累托前沿（小规模枚举时便宜；束搜索时跳过）
    compute_pareto: bool = True

    def __post_init__(self) -> None:
        self.objective.validate()


class OptimizationScheduler(SchedulerBase):
    """优化参考的调度器：**同一入口、同一计划格式、同一执行器**。"""

    def __init__(self, config: Optional[SchedulingConfig] = None,
                 optimizer_config: Optional[OptimizerConfig] = None) -> None:
        super().__init__(config)
        self.optimizer_config = optimizer_config or OptimizerConfig()
        clamped, clipped = self.optimizer_config.budget.clamp_horizon(
            self.optimizer_config.horizon_ticks)
        self._horizon_clipped = clipped
        self._horizon = clamped
        self.prediction = PredictionModel(horizon_ticks=clamped)
        self.consequence = ConsequenceModel(
            prediction=self.prediction,
            objective=self.optimizer_config.objective,
            horizon_ticks=clamped)
        #: 最近一次规划的完整交代（供报告与测试读取）
        self.last_outcome: Optional[OptimizationOutcome] = None
        #: 逐 tick 的规划交代：报告需要回答"24 次规划里几次完全枚举、
        #: 几次触发计算预算、总展开多少次"。只留最近一次是不够的。
        self.outcomes: List[OptimizationOutcome] = []

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return POLICY_CN[self.policy]

    def _priority(self, task: QueuedTask, urgency: Dict[str, Any],
                  now_s: float) -> Tuple[float, List[str], Dict[str, Any]]:
        """优化参考的「优先级」= 单条边际价值。

        它只用于两件事：同一任务多节点时的取舍、以及 `task.reason` 的可读说明。
        计划级排序走的是**目标值**（`ObjectiveSpec`），不是这个数——
        两者的区别在决策理由里写清楚了。
        """
        targets = tuple(task.targets)
        ages = [urgency.get("information_age_s")] if targets else []
        sigmas = [urgency.get("sigma_position_m")] if targets else []
        age = ages[0] if ages and ages[0] is not None else None
        sigma = sigmas[0] if sigmas and sigmas[0] is not None else None
        value = 0.0
        if age is not None:
            value += min(1.0, float(age) / 3.0)
        if sigma is not None:
            value += min(1.0, float(sigma) / 150.0)
        cost = task.estimated_cost
        value -= 1e-4 * float(cost.get(ResourceUnit.COMM_BYTE, 0.0) or 0.0)
        value -= 1e-3 * float(cost.get(ResourceUnit.SAMPLE_SLOT, 0.0) or 0.0)
        evidence = {
            "marginal_value": round(value, 9),
            "information_age_s": age,
            "sigma_position_m": sigma,
            "note": ("单条边际价值只用于同类比较；计划级取舍看 6 维目标值，"
                     "不是这个数"),
        }
        reasons = [
            f"单条边际价值 {value:.6f}（刷新估计 {min(1.0, (age or 0.0) / 3.0):.3f}"
            f" + 改善协方差 {min(1.0, (sigma or 0.0) / 150.0):.3f}"
            " − 资源代价）",
            "该值**不参与**计划级排序；计划级取舍由 6 维目标值决定",
        ]
        return value, reasons, evidence

    # ------------------------------------------------------------------

    def _select(self, candidates: List[QueuedTask],
                observation: CentralObservation,
                by_node: Dict[str, NodeObservation],
                now_s: float,
                node_order: List[str]) -> SelectionResult:
        budget = self.optimizer_config.budget
        budget.reset()
        decisions: List[SchedulingDecision] = []

        # ③ 重复判定：与规则基线共用同一实现（保证"什么算重复"一致）
        singles = [(self._single_value(task, by_node, now_s), task)
                   for task in candidates]
        singles = self._duplicate_rule(singles, decisions)
        active = [task for _value, task in singles]

        # 搜索
        outcome = self._search(active, candidate_pool=candidates,
                               by_node=by_node, now_s=now_s,
                               node_order=node_order)
        self.last_outcome = outcome
        self.outcomes.append(outcome)
        best_ids = set(outcome.best_plan)

        # 逐条决策：选中的给"为什么是最优/参考解"，未选中的给差额
        chosen = [task for task in active if task.task_id in best_ids]
        for task in active:
            if task.task_id in best_ids:
                continue
            delta = self._single_value(task, by_node, now_s) \
                - (outcome.runner_up or {}).get("objective", outcome.objective)
            decisions.append(SchedulingDecision(
                task_id=task.task_id, node_id=task.node_id,
                kind=task.kind.value, decision=DECISION_DEFERRED,
                priority=self._single_value(task, by_node, now_s),
                policy=self.policy.value,
                reasons=[
                    (f"未被选入最优解：{outcome.kind} 在 "
                     f"{outcome.n_combinations_evaluated} 个候选计划里选出目标值 "
                     f"{outcome.objective:.6f} 的方案"),
                    f"单条边际价值 {self._single_value(task, by_node, now_s):.6f}"
                    f"（边际价值只用于同类比较，不等于计划级目标值）",
                    outcome.optimality_claim,
                ],
                evidence={"objective": round(outcome.objective, 9),
                          "best_plan": list(outcome.best_plan),
                          "exact": outcome.exact,
                          "note": "该任务在本 tick 未被执行，仍在队列中等待"},
                deferred_reason="not_selected_by_optimizer"))
            _ = delta
        for task in chosen:
            decisions.append(SchedulingDecision(
                task_id=task.task_id, node_id=task.node_id,
                kind=task.kind.value, decision=DECISION_PLANNED,
                priority=self._single_value(task, by_node, now_s),
                policy=self.policy.value,
                reasons=[
                    (f"{outcome.kind}：在 {outcome.n_combinations_evaluated} "
                     f"个可行候选计划中，含本任务的方案目标值 "
                     f"{outcome.objective:.6f} 最高"),
                    f"预测向量：{self._vector_text(outcome.vector)}",
                    outcome.optimality_claim,
                ],
                evidence={
                    "objective": round(outcome.objective, 9),
                    "vector": (outcome.vector.to_dict() if outcome.vector
                               else None),
                    "exact": outcome.exact,
                    "n_combinations_evaluated":
                        outcome.n_combinations_evaluated,
                    "budget_exhausted": bool(outcome.budget.get("exhausted")),
                    "estimated_cost": task.estimated_cost_text(),
                    "deadline_s": task.deadline_s,
                }))
        return SelectionResult(chosen=chosen, decisions=decisions)

    # ------------------------------------------------------------------

    def outcome_summary(self) -> Dict[str, Any]:
        """逐 tick 规划交代的汇总（报告用它回答预算与最优性声明的真实性）。"""
        if not self.outcomes:
            return {"n_plans": 0}
        exact = [o for o in self.outcomes if o.exact]
        budget_hit = [o for o in self.outcomes
                      if o.budget.get("exhausted")]
        claims = sorted({o.optimality_claim.split("。")[0]
                         for o in self.outcomes})
        return {
            "n_plans": len(self.outcomes),
            "n_exact": len(exact),
            "n_budget_exhausted": len(budget_hit),
            "total_expansions": sum(o.budget.get("expansions", 0)
                                    for o in self.outcomes),
            "max_expansions": max(o.budget.get("expansions", 0)
                                  for o in self.outcomes),
            "max_overrun_s": max((o.budget.get("overrun_s", 0.0)
                                  for o in self.outcomes), default=0.0),
            "n_time_limit_triggered": sum(
                1 for o in self.outcomes
                if o.budget.get("time_limit_triggered")),
            "deterministic": all(o.budget.get("deterministic", True)
                                 for o in self.outcomes),
            "determinism_note": (
                "`deterministic=False` 表示墙钟安全阀至少触发过一次——"
                "那次搜索的停止点与机器负载有关，**该次结果不可作为可复现基准**。"
                "要可复现请把 `time_limit_s` 调大或设为 None，"
                "只依赖 `max_expansions`（确定性上限）。"),
            "n_pareto_truncated": sum(1 for o in self.outcomes
                                      if o.pareto_truncated),
            "planning_time_s": round(self.planning_time_s, 9),
            "planning_calls": self.planning_calls,
            "mean_planning_ms": round(
                1000.0 * self.planning_time_s / max(1, self.planning_calls), 6),
            "distinct_claims": claims,
            "claim_kinds": sorted({o.claim_kind for o in self.outcomes}),
            "claim_audit": {
                # ① 文本必须是规范文本的逐字渲染（防止注入"全局最优"之类措辞）
                "all_claims_canonical": all(
                    _is_canonical_claim(o) for o in self.outcomes),
                # ② 结构化类型必须与 exact 一致
                "exact_iff_exact_claim": all(
                    o.exact == (o.claim_kind == CLAIM_KIND_EXACT)
                    for o in self.outcomes),
                # ③ exact 声明必须真的把组合枚举完了（不谎报总数）
                "all_exact_claims_backed": all(
                    (not o.exact)
                    or (o.n_combinations_total > 0
                        and o.budget.get("expansions", 0)
                        >= o.n_combinations_total)
                    for o in exact),
                "note": ("三项都必须为 True。`exact` 表示完全枚举；"
                         "本模块的任何方法都不得自称全局最优或理论上界。"),
            },
        }

    # ------------------------------------------------------------------

    def _search(self, active: List[QueuedTask],
                candidate_pool: List[QueuedTask],
                by_node: Dict[str, NodeObservation],
                now_s: float,
                node_order: List[str]) -> OptimizationOutcome:
        """枚举 / 束搜索。**`exact=True` 只有一条产生路径**（全部枚举且未触预算）。"""
        cfg = self.optimizer_config
        budget = cfg.budget
        objective = cfg.objective
        kind = cfg.kind.value

        by_node_tasks: Dict[str, List[QueuedTask]] = {}
        for task in active:
            by_node_tasks.setdefault(task.node_id, []).append(task)
        for node_id in by_node_tasks:
            by_node_tasks[node_id].sort(key=lambda t: t.task_id)
        node_ids = sorted(by_node_tasks)

        # 每节点至多 1 条（执行器硬约束，见 SchedulingConfig 的说明）
        cap = self.config.max_tasks_per_node_per_tick
        if cap != 1:
            return self._budget_outcome(
                kind, candidate_pool, "每节点每 tick 限额不是 1，本版本不支持枚举")

        # 组合数与"是否可能完全枚举"（先算，再决定走哪条路）
        per_node_options = [[None] + by_node_tasks[node_id]
                            for node_id in node_ids]
        total = 1
        for options in per_node_options:
            total *= len(options)

        if total > budget.max_expansions:
            if cfg.kind is OptimizerKind.ENUMERATION:
                # 枚举模式要求完全枚举；组合数超预算就不允许声明精确最优，
                # 降级为束搜索并如实标注。
                return self._beam_search(
                    by_node_tasks, candidate_pool, by_node, now_s, node_order,
                    downgraded_from_exact=True)
            return self._beam_search(
                by_node_tasks, candidate_pool, by_node, now_s, node_order)

        # --- 完全枚举 ---
        evaluated: List[Tuple[float, Tuple[str, ...],
                             EvaluationVector]] = []
        infeasible = 0
        frontier: List[EvaluationVector] = []
        for combination in itertools.product(*per_node_options):
            if not budget.spend(1):
                return self._budget_outcome(
                    kind, candidate_pool, budget.exhausted_reason,
                    evaluated=evaluated, total=total)
            pick = [task for task in combination if task is not None]
            ok, _why = self._affordable(pick, by_node)
            if not ok:
                infeasible += 1
                continue
            vector = self.consequence.evaluate(
                pick, candidate_pool, by_node, now_s)
            score = objective.score(vector)
            evaluated.append((score, tuple(sorted(t.task_id for t in pick)),
                              vector))
            if cfg.compute_pareto:
                frontier.append(vector)
        if not evaluated:
            return self._budget_outcome(
                kind, candidate_pool, "没有任何资源可行的候选计划")

        evaluated.sort(key=lambda row: (-row[0], row[1]))
        best_score, best_ids, best_vector = evaluated[0]
        runner = None
        if len(evaluated) > 1:
            runner = {"plan": list(evaluated[1][1]),
                      "objective": evaluated[1][0],
                      "gap": best_score - evaluated[1][0]}
        pareto_truncated = False
        frontier: List[EvaluationVector] = []
        if cfg.compute_pareto:
            pool = [row[2] for row in evaluated[:budget.pareto_max_points]]
            pareto_truncated = len(evaluated) > len(pool)
            frontier = pareto_frontier(pool)
        exact = bool(not budget.exhausted
                     and budget.expansions >= total
                     and self._horizon == 1)
        if self._horizon > 1:
            claim_kind = CLAIM_KIND_LOOKAHEAD
            claim_args: Dict[str, Any] = {"reason": ""}
        elif budget.exhausted:
            claim_kind = CLAIM_KIND_BUDGET
            claim_args = {"reason": budget.exhausted_reason}
        else:
            claim_kind = CLAIM_KIND_EXACT
            claim_args = {"n": budget.expansions, "reason": ""}
        claim = CANONICAL_CLAIMS[claim_kind].format(**claim_args)
        return OptimizationOutcome(
            kind=kind, exact=exact, claim_kind=claim_kind,
            optimality_claim=claim, claim_args=claim_args,
            objective=best_score, vector=best_vector,
            n_candidates=len(active), n_combinations_total=total,
            n_combinations_evaluated=len(evaluated),
            n_combinations_infeasible=infeasible,
            best_plan=list(best_ids), runner_up=runner,
            pareto_frontier=[v.to_dict() for v in frontier],
            pareto_truncated=pareto_truncated,
            budget=budget.to_dict(),
            prediction=self.prediction.describe(),
            objective_spec=objective.describe(),
            explanations={
                "search": ("完全枚举：每个节点在 [不选, 各候选任务] 上做笛卡尔积，"
                           "逐个评估预测向量"),
                "pareto_note": (
                    f"帕累托前沿在目标值最高的 "
                    f"{min(len(evaluated), budget.pareto_max_points)} 个向量上计算"
                    + ("（**已截断**：候选更多，这是预算内的近似前沿）"
                       if pareto_truncated else "（未截断）")),
                "why_not_theoretical_bound": (
                    "预测向量来自**显式预测模型**（含「航迹始终可见」这类乐观假设），"
                    "不是真值，因此即使枚举完全，也只是"
                    "**该模型下的精确最优**，不是现实问题的理论上界。"),
            })

    # ------------------------------------------------------------------

    def _beam_search(self, by_node_tasks: Dict[str, List[QueuedTask]],
                     candidate_pool: List[QueuedTask],
                     by_node: Dict[str, NodeObservation],
                     now_s: float,
                     node_order: List[str],
                     downgraded_from_exact: bool = False
                     ) -> OptimizationOutcome:
        cfg = self.optimizer_config
        budget = cfg.budget
        objective = cfg.objective
        width = budget.beam_width_per_node
        # 每节点只保留"单条边际价值"最高的 width 条，再枚举
        trimmed: List[List[Optional[QueuedTask]]] = []
        for node_id in sorted(by_node_tasks):
            ranked = sorted(
                by_node_tasks[node_id],
                key=lambda t: (-self._single_value(t, by_node, now_s),
                               t.task_id))
            trimmed.append([None] + ranked[:width])
        evaluated: List[Tuple[float, Tuple[str, ...],
                             EvaluationVector]] = []
        infeasible = 0
        frontier: List[EvaluationVector] = []
        for combination in itertools.product(*trimmed):
            if not budget.spend(1):
                break
            pick = [task for task in combination if task is not None]
            ok, _why = self._affordable(pick, by_node)
            if not ok:
                infeasible += 1
                continue
            vector = self.consequence.evaluate(
                pick, candidate_pool, by_node, now_s)
            evaluated.append((objective.score(vector),
                              tuple(sorted(t.task_id for t in pick)), vector))
            if cfg.compute_pareto:
                frontier.append(vector)
        if not evaluated:
            return self._budget_outcome(
                cfg.kind.value, candidate_pool,
                budget.exhausted_reason or "没有任何资源可行的候选计划")
        evaluated.sort(key=lambda row: (-row[0], row[1]))
        best_score, best_ids, best_vector = evaluated[0]
        runner = ({"plan": list(evaluated[1][1]), "objective": evaluated[1][0],
                   "gap": best_score - evaluated[1][0]}
                  if len(evaluated) > 1 else None)
        claim_kind = (CLAIM_KIND_BUDGET if downgraded_from_exact
                      else CLAIM_KIND_BEAM)
        claim_args = {"reason": "组合数超过计算预算，已降级为束搜索"}
        claim = CANONICAL_CLAIMS[claim_kind].format(**claim_args)
        return OptimizationOutcome(
            kind=cfg.kind.value, exact=False, claim_kind=claim_kind,
            optimality_claim=claim, claim_args=claim_args,
            objective=best_score, vector=best_vector,
            n_candidates=len([t for tasks in by_node_tasks.values()
                              for t in tasks]),
            n_combinations_total=-1,          # 未枚举完，不谎报总数
            n_combinations_evaluated=len(evaluated),
            n_combinations_infeasible=infeasible,
            best_plan=list(best_ids), runner_up=runner,
            pareto_frontier=[v.to_dict() for v in pareto_frontier(frontier)]
            if cfg.compute_pareto else [],
            budget=budget.to_dict(),
            prediction=self.prediction.describe(),
            objective_spec=objective.describe(),
            explanations={
                "search": (f"束搜索：每节点只保留单条边际价值最高的 {width} 条，"
                           "再枚举组合"),
                "downgraded_from_exact": (
                    "本来要求完全枚举，但组合数超过 max_expansions，"
                    "因此**不声明**精确最优" if downgraded_from_exact else ""),
            })

    # ------------------------------------------------------------------

    def _budget_outcome(self, kind: str, candidate_pool: List[QueuedTask],
                        reason: str,
                        evaluated: Optional[List[Any]] = None,
                        total: int = -1) -> OptimizationOutcome:
        """预算耗尽/无可行解时的交代。

        `total` 在**枚举路径**里是已知的真实组合数（先算出来的），
        必须原样报出——记成"未知"会让报告丢掉"评估了 2 / 共 1140"
        这个关键事实。只有束搜索路径才用 -1（那里报总数就是虚报）。
        """
        evaluated = evaluated or []
        best: Optional[Tuple[float, Tuple[str, ...],
                             EvaluationVector]] = None
        for row in evaluated:
            if best is None or (-row[0], row[1]) < (-best[0], best[1]):
                best = row
        claim_args = {"reason": reason}
        claim = CANONICAL_CLAIMS[CLAIM_KIND_BUDGET].format(**claim_args)
        return OptimizationOutcome(
            kind=kind, exact=False, claim_kind=CLAIM_KIND_BUDGET,
            optimality_claim=claim, claim_args=claim_args,
            objective=(best[0] if best else 0.0),
            vector=(best[2] if best else None),
            n_candidates=len(candidate_pool),
            n_combinations_total=total,
            n_combinations_evaluated=len(evaluated),
            best_plan=list(best[1]) if best else [],
            budget=self.optimizer_config.budget.to_dict(),
            prediction=self.prediction.describe(),
            objective_spec=self.optimizer_config.objective.describe(),
            explanations={"stop_reason": reason})

    # ------------------------------------------------------------------

    def _single_value(self, task: QueuedTask,
                      by_node: Dict[str, NodeObservation],
                      now_s: float) -> float:
        node = by_node.get(task.node_id)
        ages: Dict[Tuple[str, str], float] = {}
        sigmas: Dict[Tuple[str, str], float] = {}
        if node is not None:
            for track, valid in zip(node.tracks, node.track_valid_mask):
                if valid:
                    ages[(node.node_id, track.track_id)] = \
                        float(track.information_age_s)
                    sigmas[(node.node_id, track.track_id)] = \
                        float(max(track.sigma_position))
        return self.consequence._single_task_value(task, ages, sigmas)

    def _affordable(self, pick: Sequence[QueuedTask],
                    by_node: Dict[str, NodeObservation]
                    ) -> Tuple[bool, List[str]]:
        """资源是否够（**只读观测里的 remaining**，不碰账本）。"""
        need: Dict[str, Dict[ResourceUnit, float]] = {}
        for task in pick:
            bucket = need.setdefault(task.node_id,
                                     {u: 0.0 for u in BUDGET_UNITS})
            for u in BUDGET_UNITS:
                bucket[u] += float(task.estimated_cost.get(u, 0.0) or 0.0)
        shortfalls: List[str] = []
        for node_id, bucket in need.items():
            node = by_node.get(node_id)
            if node is None:
                shortfalls.append(f"{node_id}: 本 tick 没有到达的观测摘要")
                continue
            for u in BUDGET_UNITS:
                available = float(node.remaining.get(u.value, 0.0))
                if bucket[u] > available + 1e-9:
                    shortfalls.append(
                        f"{node_id}.{u.value}: 需要 {bucket[u]:g}，可用 "
                        f"{available:g}")
        return (not shortfalls), shortfalls

    @staticmethod
    def _vector_text(vector: Optional[EvaluationVector]) -> str:
        if vector is None:
            return "（无预测向量）"
        parts = [f"{METRIC_BY_KEY[k].name_cn}={vector.values.get(k, 0.0):.4f}"
                 for k in METRIC_KEYS]
        return "，".join(parts)


class EnumerationOptimizer(OptimizationScheduler):
    """小规模**完全枚举**优化参考。"""

    policy = OPTIMIZATION_POLICIES[0]


class RollingHorizonOptimizer(OptimizationScheduler):
    """带计算预算的**滚动规划**优化参考（不声明最优性）。"""

    policy = OPTIMIZATION_POLICIES[1]


def default_optimizer_config(kind: OptimizerKind) -> OptimizerConfig:
    """按优化参考类型给出**语义上有区别**的默认配置。

    为什么要分开给：两种方法的差别必须真的存在。第一版两者都用
    `horizon_ticks=1`，实测输出**逐位相同**——"滚动规划"名不副实，
    等于同一个方法起了两个名字。

    * `ENUMERATION`：时域 1（只规划当前 tick），要求完全枚举；
      只有它能给出"该单 tick 问题的精确最优"。
    * `ROLLING_HORIZON`：时域取预算允许的最大值（默认 3），
      首 tick 枚举 + 后续声明式 rollout；因为含前瞻推演，
      **永远**不声明精确最优。
    """
    budget = ComputeBudget()
    if kind is OptimizerKind.ROLLING_HORIZON:
        return OptimizerConfig(kind=kind, budget=budget,
                               horizon_ticks=budget.max_horizon_ticks)
    return OptimizerConfig(kind=kind, budget=budget, horizon_ticks=1)


def build_optimizer(policy: Any,
                    config: Optional[SchedulingConfig] = None,
                    optimizer_config: Optional[OptimizerConfig] = None
                    ) -> OptimizationScheduler:
    """按策略名构造优化参考（`build_scheduler` 的唯一入口）。"""
    table = {
        OPTIMIZATION_POLICIES[0]: (EnumerationOptimizer, OptimizerKind.ENUMERATION),
        OPTIMIZATION_POLICIES[1]: (RollingHorizonOptimizer,
                                   OptimizerKind.ROLLING_HORIZON),
    }
    if policy not in table:
        raise KeyError(f"未知优化参考策略 {policy!r}")
    cls, kind = table[policy]
    scheduler = cls(config, optimizer_config or default_optimizer_config(kind))
    # 保证策略标识与实现一致（防止 subclass 与枚举错配）
    if scheduler.policy is not policy:
        raise RuntimeError(
            f"策略标识错配：要求 {policy!r}，构造出 {scheduler.policy!r}")
    return scheduler


__all__ = [
    "CLAIM_BEAM", "CLAIM_BUDGET", "CLAIM_EXACT", "CLAIM_LOOKAHEAD",
    "ComputeBudget", "ConsequenceModel", "EnumerationOptimizer",
    "EVALUATION_METRICS", "EvaluationVector", "METRIC_BY_KEY", "METRIC_KEYS",
    "MetricSpec", "ObjectiveSpec", "OptimizationOutcome", "OptimizationScheduler",
    "OptimizerConfig", "OptimizerKind", "PREDICTION_ASSUMPTIONS",
    "PredictionAssumption", "PredictionModel", "RollingHorizonOptimizer",
    "SERVICED_SIGMA_M", "SIGMA_GROWTH_MPS", "build_optimizer",
    "default_optimizer_config", "pareto_frontier",
]
