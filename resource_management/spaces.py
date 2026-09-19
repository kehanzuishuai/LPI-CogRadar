"""资源调度环境使用的最小 Gymnasium 风格空间。

该实现刻意留在 ``resource_management`` 包内，避免教学调度
模块因复用便利类而反向依赖历史仿真 ``engine``。
"""

from __future__ import annotations

import random
from typing import List, Sequence


class Discrete:
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("Discrete 的 n 必须为正整数")
        self.n = int(n)
        self._rng = random.Random(0)

    def seed(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample(self) -> int:
        return self._rng.randrange(self.n)

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and 0 <= value < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box:
    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        if len(low) != len(high):
            raise ValueError("Box 的 low 与 high 长度必须一致")
        self.low: List[float] = [float(value) for value in low]
        self.high: List[float] = [float(value) for value in high]
        self.shape = (len(self.low),)

    def contains(self, value: object) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != len(self.low):
            return False
        return all(
            low - 1e-9 <= float(item) <= high + 1e-9
            for item, low, high in zip(value, self.low, self.high)
        )

    def clip(self, value: Sequence[float]) -> List[float]:
        return [
            max(low, min(high, float(item)))
            for item, low, high in zip(value, self.low, self.high)
        ]

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, low={self.low}, high={self.high})"
