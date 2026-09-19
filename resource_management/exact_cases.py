"""学习协议的可手算排队案例与解析期望值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from resource_management.learning_env import LearningCase, LearningTaskSpec
from resource_management.units import ResourceUnit


def _budgets(sample: float = 2.0) -> Dict[ResourceUnit, float]:
    return {
        ResourceUnit.SAMPLE_SLOT: sample,
        ResourceUnit.PROCESSING_OP: 2.0,
        ResourceUnit.COMM_BYTE: 100.0,
    }


def _task(task_id: str = "T1", reward: float = 2.0) -> LearningTaskSpec:
    return LearningTaskSpec(
        task_id=task_id,
        release_step=0,
        deadline_step=5,
        completion_reward=reward,
        cost={ResourceUnit.SAMPLE_SLOT: 1.0},
    )


@dataclass(frozen=True)
class ExactExpectation:
    actions: Tuple[int, ...]
    rewards: Tuple[float, ...]
    # 顺序：completion, invalid_action, expiry, waiting, terminal_supplement
    reward_components: Tuple[Tuple[float, float, float, float, float], ...]
    terminated_reason: str
    undiscounted_return: float
    discounted_return: float
    cumulative_cost: float


def exact_cases() -> Dict[str, Tuple[LearningCase, ExactExpectation]]:
    """返回四个无需仿真统计、可逐项人工复核的案例。"""

    single = LearningCase(
        name="single_on_time",
        horizon_steps=3,
        tick_seconds=1.0,
        budgets=_budgets(),
        tasks=(_task(),),
    )
    wait = LearningCase(
        name="wait_then_complete",
        horizon_steps=3,
        tick_seconds=1.0,
        budgets=_budgets(),
        tasks=(_task(),),
    )
    horizon = LearningCase(
        name="horizon_unresolved",
        horizon_steps=2,
        tick_seconds=1.0,
        budgets=_budgets(),
        tasks=(_task(),),
    )
    exhausted = LearningCase(
        name="resource_exhaustion",
        horizon_steps=4,
        tick_seconds=1.0,
        budgets=_budgets(sample=1.0),
        tasks=(_task("T1", 2.0), _task("T2", 3.0)),
    )
    one_sixth = 1.0 / 6.0
    one_third = 1.0 / 3.0
    return {
        single.name: (
            single,
            ExactExpectation(
                (1,), (2.0,), ((2.0, 0.0, 0.0, 0.0, 0.0),),
                "all_tasks_resolved", 2.0, 2.0, one_sixth,
            ),
        ),
        wait.name: (
            wait,
            ExactExpectation(
                (0, 1), (-0.25, 2.0),
                (
                    (0.0, 0.0, 0.0, -0.25, 0.0),
                    (2.0, 0.0, 0.0, 0.0, 0.0),
                ),
                "all_tasks_resolved", 1.75, 1.55, one_sixth,
            ),
        ),
        horizon.name: (
            horizon,
            ExactExpectation(
                (0, 0), (-0.25, -1.75),
                (
                    (0.0, 0.0, 0.0, -0.25, 0.0),
                    (0.0, 0.0, 0.0, -0.25, -1.5),
                ),
                "task_horizon", -2.0, -1.825, 0.0,
            ),
        ),
        exhausted.name: (
            exhausted,
            ExactExpectation(
                (1,), (0.25,), ((2.0, 0.0, 0.0, -0.25, -1.5),),
                "resource_exhausted", 0.25, 0.25, one_third,
            ),
        ),
    }
