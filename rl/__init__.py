"""强化学习包（DQN）。

    replay_buffer.py  经验回放池
    dqn_agent.py      Q 网络 + 目标网络 + ε-greedy + 批量训练 + 目标网络周期同步 + 保存/加载

训练入口在工程根目录的 train_dqn.py，评测入口在 evaluate_dqn.py。
环境直接复用 engine/env.py::LpiPowerEnv（11 维观测 / 11 档离散功率 / 综合收益奖励）。
"""

from .dqn_agent import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    silence_numpy_bridge_warning,
)
from .lagrangian_agent import LagrangianConfig, LagrangianDQNAgent
from .replay_buffer import ReplayBuffer

__all__ = [
    "DQNAgent",
    "DQNConfig",
    "QNetwork",
    "ReplayBuffer",
    "silence_numpy_bridge_warning",
    "LagrangianDQNAgent",
    "LagrangianConfig",
]
