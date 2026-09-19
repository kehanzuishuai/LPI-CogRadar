"""任务收益模型（综合收益）。

放在 models 而不是 engine，是为了保持清晰的依赖方向：
    models  <- engine <- strategy / metrics
models 只依赖标准库，因此 engine.equations 可以安心被 models 引用，
而 models 不反向依赖 engine，杜绝循环 import。

第二版（时序决策版）的奖励
--------------------------
    r = w_det · min(Pd_min / required_pd, 1)         探测任务完成度（达标即封顶）
      − w_int · Pint_eff                             有效截获概率（含累计暴露）
      − w_energy · (Pt / Pt_max)                      归一化发射功率
      − w_violation · 1[Pd_min < required_pd]         探测任务未达标的**固定惩罚**

与第一版相比新增最后一项：`violation` 权重（默认 1.5）大于探测项的上限 1.0，
因此「为了省电而放弃探测」在单步上永远不划算——这是把探测任务满足率
拉到 95% 以上的主要手段。

另外，能量耗尽导致 episode 提前结束时，未执行的任务步按**全部失败**计入一次性终端惩罚：

    penalty_terminal = −w_violation · 剩余任务步数

这条与评测指标里的 `horizon_satisfaction_rate`（满足步数 / 完整任务步数）口径一致，
防止智能体通过「早早把能量烧光、提前结束」来规避未达标惩罚。

`engine.env`（训练）与 `metrics.collector`（报告）共用本函数，
保证 RL 训练目标与实验报告里的「综合收益」是同一个定义。
"""

from __future__ import annotations

from typing import Dict

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "detection": 1.0,
    "intercept": 0.6,
    "energy": 0.3,
    "violation": 1.5,  # 第二版新增：Pd < required_pd 的固定惩罚
    "terminal": 1.5,  # 能量耗尽终端惩罚的权重（**独立于 violation**）
}


def composite_reward(
    detection_term: float,
    intercept_prob: float,
    power_fraction: float,
    weights: Dict[str, float] | None = None,
    violation: float = 0.0,
) -> float:
    """单步综合收益。

    参数
    ----
    detection_term : 探测任务完成度，建议传 min(Pd_min / required_pd, 1.0)
    intercept_prob : 有效截获概率 Pint_eff（含累计暴露）
    power_fraction : Pt / Pt_max
    weights        : 权重字典；缺项按默认值补齐
    violation      : 1.0 表示本步 Pd < required_pd，0.0 表示达标
    """
    w = weights or DEFAULT_REWARD_WEIGHTS
    return (
        w.get("detection", 1.0) * detection_term
        - w.get("intercept", 0.6) * intercept_prob
        - w.get("energy", 0.3) * power_fraction
        - w.get("violation", 1.5) * violation
    )


def terminal_energy_penalty(
    remaining_steps: int,
    weights: Dict[str, float] | None = None,
) -> float:
    """能量耗尽导致提前结束时的一次性终端惩罚。

    把「没跑完的任务步」按未达标处理，与 horizon_satisfaction_rate 口径一致，
    避免智能体把「提前结束」当作规避惩罚的手段。

    ⚠️ 权重键是 **`terminal`**（默认回退到 `violation`，再回退到 1.5），
    **必须与 `violation` 分开**：安全 RL 分支会把逐步的违反惩罚置零
    （改由代价 critic + λ 显式约束），但**能量耗尽的终端惩罚必须保留**——
    否则智能体会发现「烧光能量提前结束」不再有任何代价，
    于是疯狂提功率避免当前违反、把能量提前烧光，反而让满足率崩掉。
    """
    w = weights or DEFAULT_REWARD_WEIGHTS
    weight = w.get("terminal", w.get("violation", 1.5))
    return -weight * max(0, int(remaining_steps))
