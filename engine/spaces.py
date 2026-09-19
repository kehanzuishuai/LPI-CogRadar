"""极简动作/观测空间定义（Gymnasium 风格，零第三方依赖）。

工程定位是「Gymnasium 风格」而不是「强制依赖 Gymnasium」：
本仿真只需要离散动作与定长浮点向量的语义，装一个 numpy/gymnasium
只为拿两个空间类并不划算。这里提供与 Gymnasium 同名同形的最小实现：

    Discrete(n)                     -> .n / .sample() / .contains(x)
    Box(low, high, shape, dtype)    -> .low / .high / .shape / .sample() / .contains(x)

因此 LpiPowerEnv 的 reset()/step() 契约与 Gymnasium 完全一致，
后续接 DQN 时若已安装 gymnasium，可写两行适配把它包装成 gym.Env：

    class GymWrapper(gym.Env):
        def __init__(self, **kw): self._env = LpiPowerEnv(**kw)
        def reset(self, *, seed=None, options=None): return self._env.reset(seed=seed, options=options)
        def step(self, action): return self._env.step(action)
"""

from __future__ import annotations

import random
from typing import List, Sequence


class Discrete:
    """离散动作空间 {0, 1, ..., n-1}，对应 11 档发射功率。"""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("Discrete 的 n 必须为正整数")
        self.n = int(n)
        self._rng = random.Random(0)

    def seed(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample(self) -> int:
        return self._rng.randrange(self.n)

    def contains(self, value) -> bool:
        return isinstance(value, int) and 0 <= value < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Discrete) and other.n == self.n


class Box:
    """定长浮点向量空间（观测）。"""

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        shape: Sequence[int] | None = None,
        dtype=float,
    ) -> None:
        if len(low) != len(high):
            raise ValueError("Box 的 low 与 high 长度必须一致")
        self.low: List[float] = [float(v) for v in low]
        self.high: List[float] = [float(v) for v in high]
        self.shape = tuple(shape) if shape is not None else (len(self.low),)
        self.dtype = dtype
        self._rng = random.Random(0)

    def seed(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample(self) -> List[float]:
        return [
            self._rng.uniform(lo, hi) if hi > lo else lo
            for lo, hi in zip(self.low, self.high)
        ]

    def contains(self, value) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != len(self.low):
            return False
        return all(
            lo - 1e-9 <= float(v) <= hi + 1e-9
            for v, lo, hi in zip(value, self.low, self.high)
        )

    def clip(self, value: Sequence[float]) -> List[float]:
        return [
            max(lo, min(hi, float(v))) for v, lo, hi in zip(value, self.low, self.high)
        ]

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, low={self.low}, high={self.high})"
