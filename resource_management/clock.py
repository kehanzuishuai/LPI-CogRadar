"""唯一全局时钟。

纪律（用户明确要求）
--------------------
> 由唯一的全局时钟推进世界，**禁止每执行一部雷达就把目标、通信队列和其他
> 实体再推进一次。**

因此本模块只提供**一个**时钟对象，并把它设计成"唯一的 `now` 来源"：

* 执行器只**读取** `clock.now_s`，从不自己算时间；
* 推进世界只能通过 `clock.advance_to(t)` / `clock.advance(dt)`，
  且**每个 tick 只调用一次**（`step_count` 记录了调用次数，
  测试用它证明"N 个节点、M 个任务只推进一次"）；
* 时钟不持有任何实体的引用，因此**不可能**顺手把目标/通信队列再推一次。

`advance_to` 只允许**向前**：时间倒流会让"信息年龄"与占用区间失去意义，
所以直接报错而不是静默接受。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ClockError(RuntimeError):
    """时间用法违规（倒流、负数步长、未对齐）。"""


@dataclass
class GlobalClock:
    """世界唯一时间源。

    | 字段 | 含义 |
    | --- | --- |
    | `now_s` | 当前世界时刻（秒），所有模块唯一的时间依据 |
    | `step_count` | **推进次数**——用来证明"一次 tick 只推进一次" |
    | `history` | 推进轨迹 `(from, to, reason)`，便于追溯 |
    """

    now_s: float = 0.0
    step_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: 每次推进后要调用的钩子（执行器挂在这里做预留激活）
    _on_advance: List[Callable[[float, float], None]] = field(default_factory=list)

    # ------------------------------------------------------------------

    def subscribe(self, hook: Callable[[float, float], None]) -> None:
        """注册"时钟推进时"的回调，签名为 `(previous_s, now_s)`。

        这是**唯一**允许在推进时做副作用的通道：执行器用它把到期预留
        转成消耗。回调里拿不到任何世界实体，因此不会出现
        "顺手再推一次目标"的写法。
        """
        self._on_advance.append(hook)

    def advance_to(self, target_s: float, reason: str = "") -> float:
        """把世界推进到 `target_s`（只允许向前）。返回实际推进量。"""
        target = float(target_s)
        if target < self.now_s - 1e-12:
            raise ClockError(
                f"时间不能倒流：now={self.now_s:g}s，请求 {target:g}s。"
                "信息年龄与占用区间都依赖单调时间。"
            )
        previous = self.now_s
        self.now_s = target
        self.step_count += 1
        self.history.append({"from_s": round(previous, 6),
                             "to_s": round(target, 6),
                             "reason": reason})
        for hook in list(self._on_advance):
            hook(previous, self.now_s)
        return self.now_s - previous

    def advance(self, dt_s: float, reason: str = "") -> float:
        """按步长推进；负数步长直接报错。"""
        delta = float(dt_s)
        if delta < 0.0:
            raise ClockError(f"步长不能为负：{delta:g}s")
        return self.advance_to(self.now_s + delta, reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "now_s": round(self.now_s, 6),
            "step_count": self.step_count,
            "history": list(self.history),
        }
