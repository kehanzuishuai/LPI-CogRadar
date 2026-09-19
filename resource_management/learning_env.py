"""教学资源调度的独立学习环境。

本环境只用于在冻结的 ``resource-contract-v1`` 之上校核学习问题，
不复用旧雷达功率环境的终止语义，也不接触仿真真值、传感器或跟踪器。

时间语义
--------
一步是固定 ``tick_seconds`` 的调度区间；每个 tick 最多完成一个原子任务。
有限任务时域是 MDP 定义的一部分，因此时域结束是 ``terminated``；只有调用方
施加的更短步数限制才是 ``truncated``。剩余时间显式进入观测，保证有限时域
问题保持 Markov 性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resource_management.units import BUDGET_UNITS, ResourceUnit
from resource_management.spaces import Box, Discrete


class LearningTaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class LearningTaskSpec:
    """一个可手算的原子调度任务。时间字段均为 tick 索引。"""

    task_id: str
    release_step: int
    deadline_step: int
    completion_reward: float
    cost: Mapping[ResourceUnit, float] = field(default_factory=dict)

    def validate(self, horizon_steps: int) -> None:
        if not self.task_id:
            raise ValueError("task_id 不能为空")
        if self.release_step < 0:
            raise ValueError(f"{self.task_id}: release_step 不能为负")
        if self.deadline_step < self.release_step:
            raise ValueError(f"{self.task_id}: deadline 早于 release")
        if self.release_step >= horizon_steps:
            raise ValueError(f"{self.task_id}: release_step 必须落在任务时域内")
        for unit in BUDGET_UNITS:
            if float(self.cost.get(unit, 0.0)) < 0.0:
                raise ValueError(f"{self.task_id}: {unit.value} 成本不能为负")


@dataclass(frozen=True)
class LearningCase:
    """资源调度学习问题的完整、确定性定义。"""

    name: str
    horizon_steps: int
    tick_seconds: float
    budgets: Mapping[ResourceUnit, float]
    tasks: Tuple[LearningTaskSpec, ...]
    gamma: float = 0.9
    waiting_penalty: float = 0.25
    expiry_penalty: float = 1.0
    terminal_unresolved_penalty: float = 1.5
    invalid_action_penalty: float = 0.5

    def validate(self) -> None:
        if not self.name:
            raise ValueError("case name 不能为空")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps 必须为正")
        if self.tick_seconds <= 0.0:
            raise ValueError("tick_seconds 必须为正")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if not self.tasks:
            raise ValueError("至少需要一个任务")
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id 必须唯一")
        for unit in BUDGET_UNITS:
            if float(self.budgets.get(unit, 0.0)) <= 0.0:
                raise ValueError(f"预算 {unit.value} 必须为正")
        for task in self.tasks:
            task.validate(self.horizon_steps)


@dataclass
class _TaskState:
    spec: LearningTaskSpec
    status: LearningTaskStatus = LearningTaskStatus.PENDING
    completed_step: Optional[int] = None


class ResourceSchedulingLearningEnv:
    """Gymnasium 风格的确定性资源调度校核环境。

    动作 0 表示空闲；动作 ``i + 1`` 表示选择 ``case.tasks[i]``。
    任务槽位在整个 episode 内保持稳定，适合离散动作与动作掩码。
    """

    GLOBAL_FEATURES: Tuple[str, ...] = (
        "elapsed_fraction",
        "remaining_time_fraction",
        "sample_remaining_fraction",
        "processing_remaining_fraction",
        "comm_remaining_fraction",
    )
    TASK_FEATURES: Tuple[str, ...] = (
        "valid",
        "released",
        "completed",
        "expired",
        "deadline_remaining_fraction",
        "completion_reward_normalized",
        "affordable",
        "sample_cost_fraction",
        "processing_cost_fraction",
        "comm_cost_fraction",
    )

    def __init__(
        self,
        case: LearningCase,
        external_step_limit_steps: Optional[int] = None,
    ) -> None:
        case.validate()
        if external_step_limit_steps is not None:
            if external_step_limit_steps <= 0:
                raise ValueError("external_step_limit_steps 必须为正或 None")
            if external_step_limit_steps >= case.horizon_steps:
                raise ValueError(
                    "外部截断必须严格早于自然时域；否则应使用自然 terminated"
                )
        self.case = case
        self.external_step_limit_steps = external_step_limit_steps
        self.action_space = Discrete(len(case.tasks) + 1)
        self.observation_features = list(self.GLOBAL_FEATURES)
        for index in range(len(case.tasks)):
            self.observation_features.extend(
                f"task_{index}_{name}" for name in self.TASK_FEATURES
            )
        low = [0.0] * len(self.observation_features)
        high = [1.0] * len(self.observation_features)
        for index, name in enumerate(self.observation_features):
            if name.endswith("deadline_remaining_fraction"):
                low[index] = -1.0
        self.observation_space = Box(low, high)
        self._episode_done = False
        self._tasks: List[_TaskState] = []
        self.current_step = 0
        self.elapsed_steps = 0
        self.remaining: Dict[ResourceUnit, float] = {}
        self.cumulative_resource_cost = 0.0
        self.undiscounted_return = 0.0
        self.discounted_return = 0.0
        self.reward_trace: List[Dict[str, float]] = []
        self.cost_trace: List[float] = []
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        del options
        if seed is not None:
            self.action_space.seed(seed)
        self._tasks = [_TaskState(task) for task in self.case.tasks]
        self.current_step = 0
        self.elapsed_steps = 0
        self.remaining = {
            unit: float(self.case.budgets[unit]) for unit in BUDGET_UNITS
        }
        self.cumulative_resource_cost = 0.0
        self.undiscounted_return = 0.0
        self.discounted_return = 0.0
        self.reward_trace = []
        self.cost_trace = []
        self._episode_done = False
        return self._observation(), self._info("", False, False, 0.0, {})

    def action_masks(self) -> List[bool]:
        return self._action_masks(ignore_episode_done=False)

    def _action_masks(self, ignore_episode_done: bool) -> List[bool]:
        if self._episode_done and not ignore_episode_done:
            return [False] * self.action_space.n
        masks = [True]
        for state in self._tasks:
            masks.append(
                state.status is LearningTaskStatus.PENDING
                and state.spec.release_step <= self.current_step
                and state.spec.deadline_step >= self.current_step
                and self._affordable(state.spec)
            )
        return masks

    def step(
        self, action: int
    ) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode 已结束，请先 reset")
        if not self.action_space.contains(action):
            raise ValueError(f"action={action} 不在动作空间内")

        decision_step = self.current_step
        reward_parts = {
            "completion": 0.0,
            "invalid_action": 0.0,
            "expiry": 0.0,
            "waiting": 0.0,
            "terminal_supplement": 0.0,
        }
        step_cost = 0.0
        masks = self._action_masks(ignore_episode_done=True)
        if action > 0:
            if not masks[action]:
                reward_parts["invalid_action"] = -self.case.invalid_action_penalty
            else:
                state = self._tasks[action - 1]
                self._consume(state.spec)
                state.status = LearningTaskStatus.COMPLETED
                state.completed_step = decision_step
                reward_parts["completion"] = state.spec.completion_reward
                step_cost = self._normalized_incremental_cost(state.spec)

        self.current_step += 1
        self.elapsed_steps += 1

        expired_now = 0
        for state in self._tasks:
            if (
                state.status is LearningTaskStatus.PENDING
                and self.current_step > state.spec.deadline_step
            ):
                state.status = LearningTaskStatus.EXPIRED
                expired_now += 1
        reward_parts["expiry"] = -self.case.expiry_penalty * expired_now

        waiting = sum(
            1
            for state in self._tasks
            if state.status is LearningTaskStatus.PENDING
            and state.spec.release_step < self.current_step
        )
        reward_parts["waiting"] = -self.case.waiting_penalty * waiting

        all_resolved = all(
            state.status is not LearningTaskStatus.PENDING for state in self._tasks
        )
        resource_exhausted = (
            not all_resolved
            and not any(
                self._affordable(state.spec)
                for state in self._tasks
                if state.status is LearningTaskStatus.PENDING
            )
        )
        horizon_reached = self.current_step >= self.case.horizon_steps

        terminated = False
        truncated = False
        reason = ""
        if all_resolved:
            terminated, reason = True, "all_tasks_resolved"
        elif resource_exhausted:
            terminated, reason = True, "resource_exhausted"
        elif horizon_reached:
            terminated, reason = True, "task_horizon"
        elif (
            self.external_step_limit_steps is not None
            and self.elapsed_steps >= self.external_step_limit_steps
        ):
            truncated, reason = True, "external_step_limit"

        if terminated:
            unresolved = self.unresolved_count
            reward_parts["terminal_supplement"] = (
                -self.case.terminal_unresolved_penalty * unresolved
            )

        reward = sum(reward_parts.values())
        self.cumulative_resource_cost += step_cost
        self.undiscounted_return += reward
        self.discounted_return += (self.case.gamma ** decision_step) * reward
        self.reward_trace.append(dict(reward_parts))
        self.cost_trace.append(step_cost)
        self._episode_done = terminated or truncated

        if terminated:
            next_mask = [False] * self.action_space.n
        else:
            next_mask = self._action_masks(ignore_episode_done=True)
        info = self._info(reason, terminated, truncated, step_cost, reward_parts)
        info["next_action_mask"] = next_mask
        return self._observation(), reward, terminated, truncated, info

    @property
    def unresolved_count(self) -> int:
        return sum(
            state.status is LearningTaskStatus.PENDING for state in self._tasks
        )

    def _affordable(self, task: LearningTaskSpec) -> bool:
        return all(
            float(task.cost.get(unit, 0.0)) <= self.remaining[unit] + 1e-12
            for unit in BUDGET_UNITS
        )

    def _consume(self, task: LearningTaskSpec) -> None:
        for unit in BUDGET_UNITS:
            amount = float(task.cost.get(unit, 0.0))
            self.remaining[unit] -= amount
            if self.remaining[unit] < -1e-9:
                raise RuntimeError(f"资源守恒被破坏：{unit.value} < 0")

    def _normalized_incremental_cost(self, task: LearningTaskSpec) -> float:
        return sum(
            float(task.cost.get(unit, 0.0)) / float(self.case.budgets[unit])
            for unit in BUDGET_UNITS
        ) / len(BUDGET_UNITS)

    def _observation(self) -> List[float]:
        horizon = float(self.case.horizon_steps)
        values = [
            min(1.0, self.current_step / horizon),
            max(0.0, (self.case.horizon_steps - self.current_step) / horizon),
        ]
        values.extend(
            self.remaining[unit] / float(self.case.budgets[unit])
            for unit in BUDGET_UNITS
        )
        reward_scale = max(
            1.0, max(abs(task.completion_reward) for task in self.case.tasks)
        )
        for state in self._tasks:
            spec = state.spec
            deadline_fraction = (
                spec.deadline_step - self.current_step + 1
            ) / horizon
            values.extend(
                [
                    1.0,
                    1.0 if spec.release_step <= self.current_step else 0.0,
                    1.0 if state.status is LearningTaskStatus.COMPLETED else 0.0,
                    1.0 if state.status is LearningTaskStatus.EXPIRED else 0.0,
                    max(-1.0, min(1.0, deadline_fraction)),
                    max(0.0, min(1.0, spec.completion_reward / reward_scale)),
                    1.0 if (
                        state.status is LearningTaskStatus.PENDING
                        and self._affordable(spec)
                    ) else 0.0,
                ]
            )
            values.extend(
                min(
                    1.0,
                    float(spec.cost.get(unit, 0.0))
                    / float(self.case.budgets[unit]),
                )
                for unit in BUDGET_UNITS
            )
        return self.observation_space.clip(values)

    def evaluation_metrics(self) -> Dict[str, Any]:
        completed = [
            state for state in self._tasks
            if state.status is LearningTaskStatus.COMPLETED
        ]
        expired = sum(
            state.status is LearningTaskStatus.EXPIRED for state in self._tasks
        )
        timely = sum(
            state.completed_step is not None
            and state.completed_step <= state.spec.deadline_step
            for state in completed
        )
        consumed = {
            unit.value: float(self.case.budgets[unit]) - self.remaining[unit]
            for unit in BUDGET_UNITS
        }
        recomputed_cost = sum(
            consumed[unit.value] / float(self.case.budgets[unit])
            for unit in BUDGET_UNITS
        ) / len(BUDGET_UNITS)
        return {
            "case": self.case.name,
            "steps": self.elapsed_steps,
            "completed": len(completed),
            "expired": expired,
            "unresolved": self.unresolved_count,
            "service_completion_ratio": len(completed) / len(self._tasks),
            "timely_completion_ratio": timely / len(completed) if completed else 0.0,
            "resource_consumed": consumed,
            "resource_consumption": recomputed_cost,
            "cumulative_cost": self.cumulative_resource_cost,
            "undiscounted_return": self.undiscounted_return,
            "discounted_return": self.discounted_return,
            "complete_episode": self._episode_done and self.unresolved_count == 0,
        }

    def _info(
        self,
        reason: str,
        terminated: bool,
        truncated: bool,
        step_cost: float,
        reward_parts: Mapping[str, float],
    ) -> Dict[str, Any]:
        return {
            "case": self.case.name,
            "time_s": self.current_step * self.case.tick_seconds,
            "step_index": self.current_step,
            "remaining_steps": max(0, self.case.horizon_steps - self.current_step),
            "remaining_time_fraction": max(
                0.0,
                (self.case.horizon_steps - self.current_step)
                / self.case.horizon_steps,
            ),
            "terminated_reason": reason if terminated else "",
            "truncated_reason": reason if truncated else "",
            "bootstrap_allowed": not terminated,
            "step_cost": step_cost,
            "cumulative_cost": self.cumulative_resource_cost,
            "reward_components": dict(reward_parts),
            "task_status": {
                state.spec.task_id: state.status.value for state in self._tasks
            },
            "metrics": self.evaluation_metrics(),
        }


def bootstrap_multiplier(terminated: bool, truncated: bool) -> float:
    """Bellman target 的 bootstrap 系数。

    ``truncated`` 只表示外部中止，不能关闭 bootstrap；参数保留在签名中是为了
    让调用处显式区分两个标志，而不是重新合并成 ``done``。
    """

    del truncated
    return 0.0 if terminated else 1.0
