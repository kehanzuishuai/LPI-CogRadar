r"""经验回放池（DQN 的 off-policy 数据来源）。

为什么用 torch.Tensor 而不是 numpy
----------------------------------
**本模块刻意不依赖 numpy。** 原因有两个：

1. 工程里用户的 PyTorch 环境（`D:\anaconda\envs\pytorch_env`）中
   `torch 2.3.1` 是按 NumPy 1.x 编译的，而该环境装的是 `numpy 2.2.6`，
   torch 的 numpy 桥接在这种组合下会报
   `Failed to initialize NumPy: _ARRAY_API not found` —— 任何
   `torch.from_numpy` / `tensor.numpy()` 都会失败。
2. 观测本身就是 `List[float]`，直接 `torch.tensor()` 转换比绕道 numpy 更直接。

因此全工程 RL 路径只使用 torch，`D:\anaconda\envs\pytorch_env\python.exe`
即可直接运行，无需改动任何现有环境。

设计要点
--------
* 预分配环形缓冲区，超出容量后覆盖最旧数据。
* `done` 存的是 **terminated** 而不是 truncated：时间截断（到达场景时长上限）
  不是 MDP 的终止态，仍然需要 bootstrap，否则会把「没时间了」误当成
  「任务失败」，系统性低估末段动作的价值。
* 随机采样使用独立的 `torch.Generator`，种子由 DQNAgent 统一下发，
  与全局随机流解耦，保证同种子下采样序列可复现。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


class ReplayBuffer:
    """定容经验回放池（torch 张量实现）。"""

    def __init__(
        self, capacity: int, obs_dim: int, n_actions: int = 11, seed: int | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正")
        if obs_dim <= 0:
            raise ValueError("obs_dim 必须为正")
        if n_actions <= 0:
            raise ValueError("n_actions 必须为正")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        self._states = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32)
        self._next_states = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32)
        self._actions = torch.zeros(self.capacity, dtype=torch.int64)
        self._rewards = torch.zeros(self.capacity, dtype=torch.float32)
        self._dones = torch.zeros(self.capacity, dtype=torch.float32)
        # 约束代价（安全 RL 用）：c_t = 1[Pd < required_pd]。默认 0，不影响普通 DQN。
        self._costs = torch.zeros(self.capacity, dtype=torch.float32)
        # 下一状态的动作可行性掩码（能量硬约束）：用于对目标 Q 值的 max 做 mask，
        # 否则「当前买不起的档位」会因为从未被更新而保留随机初始值、污染 bootstrap 目标
        self._next_masks = torch.ones((self.capacity, n_actions), dtype=torch.float32)

        self._position = 0
        self._size = 0

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(0 if seed is None else int(seed))

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= max(batch_size, 1)

    # ------------------------------------------------------------------
    # 写入 / 采样
    # ------------------------------------------------------------------

    def push(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool | float,
        next_mask: Sequence[bool] | None = None,
        cost: float = 0.0,
    ) -> None:
        """写入一条转移。缓冲区满后按环形覆盖最旧数据。

        next_mask：**下一状态**的动作可行性掩码（True = 能量买得起）。
        不传则视为全可行（兼容未启用能量硬约束的场景）。
        cost     ：约束代价（安全 RL 用），普通 DQN 恒为 0。
        """
        index = self._position
        # torch.tensor 直接吃 Python 序列，不经过 numpy
        self._states[index] = torch.tensor(state, dtype=torch.float32)
        self._next_states[index] = torch.tensor(next_state, dtype=torch.float32)
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._dones[index] = float(done)
        self._costs[index] = float(cost)
        if next_mask is not None:
            self._next_masks[index] = torch.tensor(next_mask, dtype=torch.float32)
        else:
            self._next_masks[index] = 1.0

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """均匀随机采样一个批次。

        返回 (states, actions, rewards, next_states, dones, next_masks, costs)。
        """
        if not self.is_ready(batch_size):
            raise ValueError(
                f"缓冲区内样本不足：{self._size} < {batch_size}，请先检查 learning_starts"
            )

        indices = torch.randint(
            0, self._size, (int(batch_size),), generator=self._generator
        )
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._next_masks[indices],
            self._costs[indices],
        )

    def clear(self) -> None:
        self._position = 0
        self._size = 0

    # ------------------------------------------------------------------
    # 存取（供 checkpoint 使用）
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "position": self._position,
            "size": self._size,
            "states": self._states[: self._size].clone(),
            "next_states": self._next_states[: self._size].clone(),
            "actions": self._actions[: self._size].clone(),
            "rewards": self._rewards[: self._size].clone(),
            "dones": self._dones[: self._size].clone(),
            "next_masks": self._next_masks[: self._size].clone(),
            "costs": self._costs[: self._size].clone(),
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        size = int(payload["size"])  # type: ignore[arg-type]
        if size > self.capacity:
            raise ValueError(
                f"checkpoint 内样本数 {size} 超过当前缓冲区容量 {self.capacity}"
            )
        self._size = size
        self._position = int(payload["position"])  # type: ignore[arg-type]
        self._states[:size] = payload["states"]  # type: ignore[index]
        self._next_states[:size] = payload["next_states"]  # type: ignore[index]
        self._actions[:size] = payload["actions"]  # type: ignore[index]
        self._rewards[:size] = payload["rewards"]  # type: ignore[index]
        self._dones[:size] = payload["dones"]  # type: ignore[index]
        if "next_masks" in payload:
            self._next_masks[:size] = payload["next_masks"]  # type: ignore[index]
        else:
            self._next_masks[:size] = 1.0
        if "costs" in payload:
            self._costs[:size] = payload["costs"]  # type: ignore[index]
        else:
            self._costs[:size] = 0.0

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}/{self.capacity}, obs_dim={self.obs_dim}, "
            f"n_actions={self.n_actions})"
        )
