"""集中式资源调度学习环境：**直接跑在真闭环上**。

它凭什么可信
------------
* 世界、传感器、通信、融合、执行器**全部复用** `resource_management.closed_loop`
  与 `RuntimeExecutor`——不是另写一套简化模型，因此学到的策略能直接对回
  规则基线（同一观测、同一队列、同一执行器、同一真闭环）；
* 训练固定 `runtime_mode="plan_controlled_feedback"`：只有被分配 `sample`
  的节点才会触发传感器，未调度节点只做航迹预测，只有执行 `share` 才真实发送。
  **在旧路径上训练是没有意义的**——那里调度影响不了感知链（README §11K）；
* 智能体只能通过**标准 `ExecutionPlan`** 影响世界；它拿不到 `Simulator`、
  拿不到 `Sensor`/`FusionCenter`/`CommBus` 的引用。

奖励与代价（与 `docs/learning_protocol.md` 的语义一致）
-------------------------------------------------------
单步奖励分解（原样写进 `info["reward_components"]`）：

    completion  = +1.0 × 本 tick 被真正执行（APPLIED）的任务数
    expiry      = −1.0 × 本 tick 过期任务数
    invalid     = −0.5 × 本 tick 被执行器拒绝的任务数
    waiting     = −0.25 × 本 tick 超阈值仍未被服务的任务数 / 节点数
    terminal    = −1.5 × 自然终止时仍未解决的任务数（**外部截断不补计**）

单步约束代价（与协议 §2.3 同定义）：

    c_t = mean_{节点,单位}( Δconsumed[u] / capacity[u] )

因此 `Σ_t c_t` **必须**等于 episode 末由账本重算的资源消耗
（`evaluation_vector.resource_consumption`）——这条恒等式是"实现正确"
与"算法不行"的分界线，测试逐 episode 校验。

终止 / 截断（与协议 §2.2 同语义）
---------------------------------
| 事件 | 标志 | 终端补计 | bootstrap |
| --- | --- | --- | --- |
| 全部任务已完成或过期 | `terminated` | 是 | 否 |
| 任务时域（`steps`）到达 | `terminated` | 是 | 否 |
| 调用方施加的更短步数上限 | `truncated` | 否 | **是** |

`info["bootstrap_allowed"]` 恒等于 `not terminated`，由测试钉住。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.closed_loop import (
    DEFAULT_MECHANISMS,
    FeedbackLoopDriver,
    RUNTIME_MODE_FEEDBACK,
    TASK_GATING_EXPOSE_ALL,
    _build_feedback_world,
)
from resource_management.clock import GlobalClock
from resource_management.executor import UnifiedExecutor
from resource_management.ledger import ResourceLedger
from resource_management.model import Outcome
from resource_management.scheduling import SchedulingConfig
from resource_management.tasks import QueueTaskKind, TaskStatus
from resource_management.units import BUDGET_UNITS, ResourceUnit

from rl_resource.actions import (
    ACTION_NAMES, ActionContext, N_ACTIONS, build_action_context, build_plan,
    pad_mask,
)
from rl_resource.obs import (
    DEFAULT_MAX_NODES, EncodedObservation, encode, observation_dim,
)


#: 奖励权重（显式常数；改动即改变学习问题的定义，需升版本）
REWARD_COMPLETION = 1.0
REWARD_EXPIRY = -1.0
REWARD_INVALID = -0.5
REWARD_WAITING = -0.25
REWARD_TERMINAL_UNRESOLVED = -1.5

#: 判定"长期未获服务"的阈值（秒）——与闭环的饥饿阈值保持一致
WAITING_THRESHOLD_S = 8.0

REWARD_VERSION = "resource-rl-reward-v1"


@dataclass
class EnvConfig:
    """环境配置（一次 episode 的全部外部参数）。"""

    scenario: str = "rm_train_base"
    seed: int = 101
    steps: int = 24
    external_step_limit: Optional[int] = None
    max_nodes: int = DEFAULT_MAX_NODES
    gamma: float = 0.99
    #: 消融组（`information_research.AblationArm` 的取值）。
    #: **空字符串 = 使用 §11L 的 49 维基线观测**（保持该基线逐位可复现）；
    #: 取四组之一时改用 `research_obs.ResearchObservationEncoder`，
    #: 四组维度完全相同、被消融的特征置零。
    arm: str = ""
    #: 是否记录逐 tick 明细（训练时关掉可省内存）
    keep_trace: bool = True

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps 必须为正")
        if self.external_step_limit is not None:
            if self.external_step_limit <= 0:
                raise ValueError("external_step_limit 必须为正")
            if self.external_step_limit >= self.steps:
                raise ValueError(
                    "外部截断必须严格早于自然时域；否则应使用自然 terminated")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes 必须为正")


class CentralizedResourceSchedulingEnv:
    """中央智能体视角的环境：`reset()` → 观测；`step(动作)` → 下一观测。

    动作是**逐节点**的动作序列（长度 = 节点数），见 `rl_resource.actions`。
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self.config.validate()
        self.action_dim_per_node = N_ACTIONS
        self._driver: Optional[FeedbackLoopDriver] = None
        self._world: Optional[Dict[str, Any]] = None
        self._queue: Any = None
        self._runtime: Any = None
        self._executor: Any = None
        self._ledger: Any = None
        self._clock: Any = None
        self._scheduler_config = SchedulingConfig()
        self._prev_consumed: Dict[str, Dict[ResourceUnit, float]] = {}
        self._prev_counts: Dict[str, int] = {}
        self._cumulative_cost = 0.0
        self._metrics_total: Dict[str, float] = {}
        self._n_rejected_total = 0
        self._n_planned_total = 0
        self._n_unmasked_illegal = 0
        self._n_node_steps = 0
        self._trace: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {}
        self._context: Optional[ActionContext] = None
        self._mask: List[List[bool]] = []
        self._observation: List[float] = []
        self._research: Any = None
        self.node_ids: Tuple[str, ...] = ()
        self.observation_dim = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> Tuple[List[float], Dict[str, Any]]:
        if seed is not None:
            self.config.seed = int(seed)
        from rl_resource.scenarios import closed_loop_kwargs, load_scenarios

        spec = load_scenarios()[self.config.scenario]
        kwargs = closed_loop_kwargs(spec)
        world = _build_feedback_world(
            seed=self.config.seed,
            steps=self.config.steps,
            mechanisms=kwargs["mechanisms"],
            node_budgets=kwargs["node_budgets"],
            deadline_offsets=kwargs["deadline_offsets"],
            scheduler_config=self._scheduler_config,
        )
        self._world = world
        self._queue = world["queue"]
        self._runtime = world["runtime"]
        self._executor = world["executor"]
        self._ledger = world["ledger"]
        self._clock = world["clock"]
        self._driver = world["driver"]
        self.node_ids = tuple(world["node_ids"])
        self._research = None
        if self.config.arm:
            from rl_resource.research_obs import ResearchObservationEncoder
            self._research = ResearchObservationEncoder(self.node_ids)
            self.observation_dim = self._research.output_dim
        else:
            self.observation_dim = observation_dim(self.config.max_nodes)
        self._prev_consumed = self._consumed_snapshot()
        self._prev_counts = self._status_counts()
        self._cumulative_cost = 0.0
        self._metrics_total = {
            "completion": 0.0, "expiry": 0.0, "invalid": 0.0,
            "waiting": 0.0, "terminal_supplement": 0.0}
        self._n_rejected_total = 0
        self._n_planned_total = 0
        self._n_unmasked_illegal = 0
        self._n_node_steps = 0
        self._trace = []
        self._last_info = {"bootstrap_allowed": True, "reward_components": {},
                           "termination_reason": "running", "step": 0}
        # 先推进到第 1 个 tick 并准备好**决策上下文**（观测 + 合法动作 mask）。
        # 为什么 reset 就要 begin：策略必须在**决策之前**看到 mask。
        # 第一版把 begin() 放在 step() 里，结果 mask 与决策同时产生——
        # 策略等于闭着眼睛选，mask 只能事后用来统计非法率，起不到约束作用。
        self._begin_tick()
        return list(self._observation), self._info()

    # ------------------------------------------------------------------

    def _begin_tick(self) -> None:
        """推进一个 tick 并缓存本 tick 的决策上下文。"""
        assert self._driver is not None
        observation = self._driver.begin()
        context = self._action_context(observation)
        self._context = context
        # 掩码补齐到 max_nodes：策略固定输出 max_nodes 行，
        # 空槽位只允许 idle（见 actions.pad_mask 的说明）。
        self._mask = pad_mask(context.mask(), self.config.max_nodes)
        self._observation = self._observe()

    def _info(self, **extra: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self._driver.step_index if self._driver else 0,
            "time_s": self._driver.current_time_s if self._driver else 0.0,
            "mask": [list(row) for row in (self._mask or [])],
            "node_ids": list(self.node_ids),
            "bootstrap_allowed": True,
            "action_names": list(ACTION_NAMES),
            "cumulative_resource_cost": self._cumulative_cost,
        }
        info.update(extra)
        return info

    def step(self, joint_action: Sequence[int]
             ) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        if self._driver is None or self._context is None:
            raise RuntimeError("必须先 reset()")
        context = self._context
        mask = self._mask
        n_real = len(self.node_ids)

        # 记录**未加 mask 时的非法选择**（"非法动作率"必须是测出来的）。
        # 只统计**真实节点**：补齐槽位不算数，否则分母被虚高。
        for index in range(n_real):
            self._n_node_steps += 1
            if not mask[index][int(joint_action[index])]:
                self._n_unmasked_illegal += 1

        built = build_plan(
            list(joint_action)[:n_real], context, self._driver.current_time_s,
            plan_id=f"learned-{self._driver.step_index:04d}",
            mask=mask[:n_real])
        stats = self._driver.commit(built.plan)
        self._n_planned_total += int(stats["n_planned"])
        self._n_rejected_total += int(stats["n_rejected"])

        reward, components = self._reward(stats)
        self._cumulative_cost += self._cost_delta()

        # --- 终止判定 ---
        #
        # 顺序很关键：`resource_exhausted` 必须在**下一个 tick 的任务已派生之后**
        # 判定，否则看到的是"本 tick 刚执行完的空队列"——那会被误判成资源耗尽
        # （第一版就是这样：每个 episode 都在第 1 个 tick 结束）。
        terminated = False
        truncated = False
        reason = "running"
        step_index = self._driver.step_index
        if (self.config.external_step_limit is not None
                and step_index >= self.config.external_step_limit
                and step_index < self.config.steps):
            truncated, reason = True, "external_step_limit"
        elif step_index >= self.config.steps:
            terminated, reason = True, "task_horizon"
        else:
            self._begin_tick()
            if self._resources_exhausted():
                terminated, reason = True, "resource_exhausted"

        if terminated:
            unresolved = self._n_open_tasks()
            supplement = REWARD_TERMINAL_UNRESOLVED * unresolved
            components["terminal_supplement"] = supplement
            reward += supplement
        for key, value in components.items():
            self._metrics_total[key] = self._metrics_total.get(key, 0.0) + value

        info = self._info(
            reward_components=dict(components),
            bootstrap_allowed=not terminated,
            termination_reason=reason,
            plan_status=stats["plan_status"],
            n_planned=int(stats["n_planned"]),
            n_applied=int(stats["n_applied"]),
            n_rejected=int(stats["n_rejected"]),
            note=built.notes,
            chosen={node: ACTION_NAMES[action]
                    for node, action in built.chosen.items()
                    if action is not None},
        )
        self._last_info = info
        if self.config.keep_trace:
            self._trace.append(info)
        if not (terminated or truncated):
            info["next_mask"] = [list(row) for row in self._mask]
            return list(self._observation), float(reward), False, False, info
        return list(self._observation), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------

    def _observe(self) -> List[float]:
        assert self._driver is not None
        observation = self._driver.runtime.central_observation(
            self._driver.current_time_s)
        if self._research is not None:
            return self._research.encode(
                observation, self._queue, self._driver.current_time_s,
                self.config.steps, self._driver.step_index, self.config.arm)
        encoded = encode(observation, self._queue, self._driver.current_time_s,
                         self.config.steps, self._driver.step_index,
                         self.config.max_nodes)
        return list(encoded.values)

    def observe_with_mask(self) -> Tuple[List[float], List[List[bool]]]:
        """返回当前决策上下文（观测 + 补齐到 `max_nodes` 的 mask）。

        学习侧评测用：**必须**在决策前拿到 mask（见 `_begin_tick` 的说明）。
        """
        return list(self._observation), [list(row) for row in self._mask]

    def encoded(self) -> EncodedObservation:
        assert self._driver is not None
        observation = self._driver.runtime.central_observation(
            self._driver.current_time_s)
        return encode(observation, self._queue, self._driver.current_time_s,
                      self.config.steps, self._driver.step_index,
                      self.config.max_nodes)

    def _action_context(self, observation: Any) -> ActionContext:
        assert self._driver is not None
        now = self._driver.current_time_s
        processable = {node_id: bool(
            self._driver.runtime.has_processable(node_id, now))
            for node_id in self._driver.runtime.centers}
        shareable = {node_id: bool(self._driver.runtime.has_shareable(node_id))
                     for node_id in self._driver.runtime.centers}
        return build_action_context(observation, self._queue, now,
                                    processable, shareable)

    def action_context(self) -> ActionContext:
        assert self._driver is not None
        return self._action_context(self._driver.runtime.central_observation(
            self._driver.current_time_s))

    # ------------------------------------------------------------------
    # 奖励 / 代价 / 终止
    # ------------------------------------------------------------------

    def _reward(self, stats: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        counts = self._status_counts()
        completed = counts.get("completed", 0) - self._prev_counts.get(
            "completed", 0)
        expired = counts.get("expired", 0) - self._prev_counts.get("expired", 0)
        self._prev_counts = counts
        waiting = self._n_waiting_over_threshold()
        components = {
            "completion": REWARD_COMPLETION * max(0, completed),
            "expiry": REWARD_EXPIRY * max(0, expired),
            "invalid": REWARD_INVALID * int(stats["n_rejected"]),
            "waiting": (REWARD_WAITING * waiting
                        / max(1, len(self.node_ids))),
        }
        return sum(components.values()), components

    def _consumed_snapshot(self) -> Dict[str, Dict[ResourceUnit, float]]:
        return {
            node_id: {unit: float(node.budget.consumed.get(unit, 0.0))
                      for unit in BUDGET_UNITS}
            for node_id, node in self._executor.nodes.items()}

    def _cost_delta(self) -> float:
        """本 tick 的约束代价 = 逐(节点,单位) 消耗增量占容量比例的算术平均。

        分母用**容量**（与 `evaluation_vector.resource_consumption` 同一口径），
        因此 `Σ_t c_t` 与 episode 末的实测资源消耗**逐位相等**（测试校验）。
        """
        current = self._consumed_snapshot()
        nodes = sorted(self._executor.nodes)
        ratios: List[float] = []
        for node_id in nodes:
            node = self._executor.node(node_id)
            for unit in BUDGET_UNITS:
                capacity = float(node.budget.capacity.get(unit, 0.0))
                if capacity <= 0.0:
                    continue
                delta = (current[node_id][unit]
                         - self._prev_consumed.get(node_id, {}).get(unit, 0.0))
                ratios.append(max(0.0, delta) / capacity)
        self._prev_consumed = current
        return (sum(ratios) / len(ratios)) if ratios else 0.0

    def _status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self._queue.tasks:
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return counts

    def _n_open_tasks(self) -> int:
        return sum(1 for task in self._queue.tasks
                   if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED))

    def _n_waiting_over_threshold(self) -> int:
        now = self._driver.current_time_s
        return sum(1 for task in self._queue.tasks
                   if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED)
                   and now - task.release_time_s >= WAITING_THRESHOLD_S)

    def _termination(self) -> Tuple[bool, bool, str]:
        """闭环环境的终止/截断语义。

        ⚠️ 与 `docs/learning_protocol.md` §2.2 那张表有**一处刻意的差别**，
        必须写明理由：

        | 事件 | 本环境 | 为什么 |
        | --- | --- | --- |
        | 任务时域（`steps`）到达 | `terminated` | 有限时域问题的自然终点 |
        | 剩余任务均不可由剩余资源执行 | `terminated` | 协议原义：`resource_exhausted` |
        | 外部更短步数上限 | `truncated` | 外部截断，保留 bootstrap |
        | **全部任务已完成** | **不作为终止条件** | 见下 |

        `all_tasks_resolved` 在**独立单节点环境**里成立，因为那里的任务集在
        episode 开始时就是固定的。但**闭环环境的任务是由实时观测逐 tick 派生**的：
        第 1 个 tick 只会派生"采样"任务（还没有数据可处理、还没有航迹可共享），
        执行完队列就空了——若沿用 `all_tasks_resolved`，**每个 episode 都会在
        第 1 个 tick 结束**（实测如此），学不到任何东西。
        空队列在这里是**瞬态**，不是"问题已解决"。
        """
        assert self._driver is not None
        step_index = self._driver.step_index
        if (self.config.external_step_limit is not None
                and step_index >= self.config.external_step_limit
                and step_index < self.config.steps):
            return False, True, "external_step_limit"
        if step_index >= self.config.steps:
            return True, False, "task_horizon"
        if self._resources_exhausted():
            return True, False, "resource_exhausted"
        return False, False, "running"

    def _resources_exhausted(self) -> bool:
        """所有**可用**节点都无法承担任何一类候选任务（协议原义）。

        判据直接看**当前 tick 已算好的 mask**（`_begin_tick()` 之后调用），
        不额外构造上下文、也不改动任何状态。
        """
        if not self._mask:
            return False
        for row in self._mask:
            if any(row[1:]):
                return False
        return True

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """跑完（或提前结束）后取回闭环结果与对账信息。"""
        if self._driver is None:
            raise RuntimeError("必须先 reset()")
        result = self._driver.finalize()
        measured = float(result.metrics["evaluation_vector"]["values"]
                         ["resource_consumption"])
        return {
            "result": result,
            "cumulative_resource_cost": self._cumulative_cost,
            "measured_resource_consumption": measured,
            "cost_reconciliation_error": abs(self._cumulative_cost - measured),
            "reward_totals": dict(self._metrics_total),
            "n_planned": self._n_planned_total,
            "n_rejected_by_executor": self._n_rejected_total,
            "n_node_steps": self._n_node_steps,
            "n_unmasked_illegal_actions": self._n_unmasked_illegal,
            "unmasked_illegal_action_rate": (
                self._n_unmasked_illegal / self._n_node_steps
                if self._n_node_steps else 0.0),
            "executor_rejection_rate": (
                self._n_rejected_total / self._n_planned_total
                if self._n_planned_total else 0.0),
            "plan_fatal_ticks": sum(
                1 for row in result.plan_log if row["status"] == "rejected"),
            "conservation_ok": bool(result.metrics["conservation_all"]),
            "trace": list(self._trace),
        }


__all__ = [
    "CentralizedResourceSchedulingEnv", "EnvConfig", "REWARD_COMPLETION",
    "REWARD_EXPIRY", "REWARD_INVALID", "REWARD_TERMINAL_UNRESOLVED",
    "REWARD_VERSION", "REWARD_WAITING", "WAITING_THRESHOLD_S",
]
