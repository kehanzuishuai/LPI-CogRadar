r"""DQN 智能体：Q 网络 + 目标网络 + ε-greedy + 批量训练 + 目标网络周期同步 + 保存/加载。

与现有工程的接法
----------------
* 状态：直接使用 `engine/env.py::LpiPowerEnv` 的 11 维观测（`observation_space: Box(11)`），
  不额外手工构造特征。
* 动作：11 档离散发射功率（`action_space: Discrete(11)`）。
* 奖励：环境返回的 reward 就是 `models/reward.py::composite_reward` 的综合收益，
  因此**训练目标与 main.py / 指标报告里的「综合收益」是同一个定义**，
  不存在训练奖励与评价指标口径不一致的问题。

算法
----
标准 DQN（Mnih et al. 2015）：
    目标 y = r + γ · (1 - done) · max_a' Q_target(s', a')
    损失   = Huber( Q_online(s, a), y )
    done  只取 terminated（时间截断仍需 bootstrap）

可选的 Double DQN（`double_dqn=True`）：用在线网络选动作、目标网络估值，
缓解 max 算子带来的过估计，默认关闭以保持「标准 DQN」。

能量硬约束下的动作掩码
----------------------
第三版把能量变成**执行前检查**的硬约束：某档功率若 Pt·Δt 超过剩余能量就不可选。
因此本实现支持 **masked DQN**：

* `select_action(obs, greedy, action_mask)` —— 探索与利用都只在可行档位内进行；
* 回放池额外存**下一状态**的可行性掩码，`train_step` 对目标 Q 的 max 做 mask。
  这一步是必要的：若只掩码动作选择而不掩码目标，那些从未被执行过的档位
  会一直保留随机初始化的 Q 值，一旦它偏大就会污染 bootstrap 目标。
* 掩码全为 False 的样本只可能出现在 `terminated=True` 的那一步（此时 bootstrap
  被 (1−done) 归零），代码里对这种行做数值保护，避免出现 −inf/NaN。

关于依赖：本模块**不使用 numpy**
--------------------------------
观测是 `List[float]`，直接 `torch.tensor()` 转换即可，无需绕道 numpy。
这一点是刻意的：用户的 PyTorch 环境（`D:\anaconda\envs\pytorch_env`）里
`torch 2.3.1` 按 NumPy 1.x 编译、而 `numpy 2.2.6` 已安装，torch 的 numpy
桥接在该组合下不可用（`Failed to initialize NumPy: _ARRAY_API not found`）。
纯 torch 路径可以直接在该环境运行，不必改动用户的任何环境。

关于折扣因子 γ
--------------
本场景经实验验证为 **contextual bandit**：动作不改变后续状态
（最低档、最高档、随机档三条轨迹的状态序列完全一致，见 README）。
此时 γ 不影响最优策略，只影响 Q 值的尺度与数值条件。
默认 γ = 0.99 保持标准 DQN 形态；若训练出现数值困难，可用 `--gamma 0`
把 Q 值尺度拉回单步收益量级（对 bandit 而言等价且条件更好）。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer

CHECKPOINT_FORMAT_VERSION = 1


# ----------------------------------------------------------------------
# 超参数
# ----------------------------------------------------------------------

@dataclass
class DQNConfig:
    """DQN 全部超参数。会被完整写进 checkpoint 与侧车 JSON，保证可复现。"""

    obs_dim: int = 11
    n_actions: int = 11
    hidden_sizes: Tuple[int, ...] = (64, 64)

    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 50000

    target_sync_steps: int = 200
    learning_starts: int = 500
    train_frequency: int = 1
    grad_clip_norm: float = 10.0
    huber_beta: float = 1.0
    double_dqn: bool = False

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000

    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DQNConfig":
        raw = json.loads(text)
        if "hidden_sizes" in raw:
            raw["hidden_sizes"] = tuple(int(v) for v in raw["hidden_sizes"])
        return cls(**raw)


# ----------------------------------------------------------------------
# Q 网络
# ----------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP Q 网络：obs_dim -> hidden... -> n_actions。

    输出是每个离散动作的 Q 值，不做 softmax（DQN 输出的是价值而非概率）。
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.ReLU())
            prev = int(hidden)
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ----------------------------------------------------------------------
# 智能体
# ----------------------------------------------------------------------

class DQNAgent:
    """DQN 智能体。"""

    def __init__(self, config: Optional[DQNConfig] = None, device: str = "cpu") -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device)

        self.set_seed(self.config.seed)

        self.online_network = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_sizes
        ).to(self.device)
        self.target_network = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_sizes
        ).to(self.device)
        self.sync_target()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss(beta=self.config.huber_beta)

        self.buffer = ReplayBuffer(
            self.config.buffer_capacity,
            self.config.obs_dim,
            n_actions=self.config.n_actions,
            seed=self.config.seed,
        )

        self.epsilon = float(self.config.epsilon_start)
        self.env_steps = 0
        self.train_steps = 0
        self.last_loss: Optional[float] = None
        self._rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # 随机种子
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """固定全部随机源，保证同种子下训练可复现。"""
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def q_values(self, obs: Sequence[float]) -> torch.Tensor:
        """返回该观测下 n_actions 个动作的 Q 值（CPU 上的 1 维张量）。"""
        with torch.no_grad():
            tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online_network(tensor).squeeze(0).cpu()

    def select_action(
        self,
        obs: Sequence[float],
        greedy: bool = False,
        action_mask: Sequence[bool] | None = None,
    ) -> int:
        """ε-greedy 选动作；greedy=True 时不探索（用于评测）。

        action_mask：能量硬约束下的动作可行性掩码。传入后，**探索与利用都只在
        可行档位内进行**——这是硬约束系统里避免「请求买不起的功率」的标准做法。
        不传则视为全部动作可行（兼容未启用能量约束的场景）。
        """
        if action_mask is None:
            feasible = list(range(self.config.n_actions))
        else:
            feasible = [i for i, ok in enumerate(action_mask) if ok]
            if not feasible:
                raise ValueError(
                    "action_mask 全为 False（本步没有任何可行档位），"
                    "episode 应当已结束，请先 reset"
                )

        if not greedy and self._rng.random() < self.epsilon:
            return feasible[self._rng.randrange(len(feasible))]

        values = self.q_values(obs)
        # 在可行档位内取 Q 最大者；平局时取索引最小者，保证确定性
        best = feasible[0]
        best_value = float(values[best])
        for level in feasible[1:]:
            value = float(values[level])
            if value > best_value:
                best_value = value
                best = level
        return best

    # ------------------------------------------------------------------
    # 交互与学习
    # ------------------------------------------------------------------

    def store(
        self,
        obs: Sequence[float],
        action: int,
        reward: float,
        next_obs: Sequence[float],
        terminated: bool,
        next_action_mask: Sequence[bool] | None = None,
    ) -> None:
        """写入一条转移。

        注意传的是 **terminated**（真实终止）而非 truncated（时间截断）；
        next_action_mask 是下一状态的动作可行性掩码，用于对 bootstrap 目标做 mask。
        """
        self.buffer.push(
            obs, action, reward, next_obs, float(terminated), next_action_mask
        )

    def update_epsilon(self) -> None:
        """ε 线性衰减：epsilon_start -> epsilon_end，在 epsilon_decay_steps 步内完成。"""
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            self.epsilon = float(cfg.epsilon_end)
            return
        progress = min(1.0, self.env_steps / float(cfg.epsilon_decay_steps))
        self.epsilon = cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    def begin_step(self) -> None:
        """在每个环境步开始时调用，推进步计数与 ε。"""
        self.env_steps += 1
        self.update_epsilon()

    def maybe_train(self) -> Optional[float]:
        """到达训练条件就做一次梯度更新，返回 loss（未训练则返回 None）。"""
        cfg = self.config
        if self.env_steps < cfg.learning_starts:
            return None
        if not self.buffer.is_ready(cfg.batch_size):
            return None
        if cfg.train_frequency > 1 and self.env_steps % cfg.train_frequency != 0:
            return None
        return self.train_step()

    def train_step(self) -> float:
        """一次批量梯度下降（含动作可行性掩码）。"""
        cfg = self.config
        # 缓冲区统一返回 7 项（最后一项 costs 供拉格朗日分支使用）。
        # 普通 DQN 不用代价信号，但必须解包全部 7 项——
        # 否则一旦缓冲区扩展到 7 项，这里会直接 ValueError。
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
            _costs,
        ) = self.buffer.sample(cfg.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        next_masks = next_masks.to(self.device)

        # --- 目标 Q 值（只在下一状态**可行**的动作上取 max）---
        with torch.no_grad():
            next_mask_bool = next_masks > 0.5
            # 全是 False 的行只可能出现在 terminated=True 的那一步（此时 bootstrap
            # 会被 (1-done) 归零）；为数值安全把这种行整体放开，避免 max 出 -inf/NaN
            has_feasible = next_mask_bool.any(dim=1, keepdim=True)
            safe_mask = torch.where(
                has_feasible, next_mask_bool, torch.ones_like(next_mask_bool)
            )

            q_next = self.target_network(next_states)
            if cfg.double_dqn:
                # 在线网络选动作（同样只在可行集合内），目标网络估值
                q_online_next = self.online_network(next_states).masked_fill(
                    ~safe_mask, float("-inf")
                )
                next_actions = q_online_next.argmax(dim=1, keepdim=True)
                next_q = q_next.gather(1, next_actions)
            else:
                next_q = q_next.masked_fill(~safe_mask, float("-inf")).max(
                    dim=1, keepdim=True
                )[0]

            targets = rewards + cfg.gamma * (1.0 - dones) * next_q

        # --- 当前 Q 值（只取实际执行的动作）---
        q_values = self.online_network(states).gather(1, actions)
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online_network.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        self.last_loss = float(loss.item())

        # --- 目标网络周期同步 ---
        if cfg.target_sync_steps > 0 and self.train_steps % cfg.target_sync_steps == 0:
            self.sync_target()

        return self.last_loss

    def sync_target(self) -> None:
        """硬同步：把在线网络的参数整体复制到目标网络。"""
        self.target_network.load_state_dict(self.online_network.state_dict())

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        include_buffer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存 checkpoint。配置以 JSON 字符串存放，便于 weights_only 安全加载。"""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        payload: Dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config_json": self.config.to_json(),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            "online_state_dict": self.online_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "env_steps": int(self.env_steps),
            "train_steps": int(self.train_steps),
        }
        if include_buffer:
            payload["buffer_state_dict"] = self.buffer.state_dict()

        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        load_buffer: bool = False,
    ) -> "DQNAgent":
        """加载 checkpoint，返回可直接继续训练或评测的智能体。"""
        payload = _torch_load(path, device=device)

        config = DQNConfig.from_json(payload["config_json"])
        agent = cls(config, device=device)
        agent.online_network.load_state_dict(payload["online_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.epsilon = float(payload.get("epsilon", config.epsilon_end))
        agent.env_steps = int(payload.get("env_steps", 0))
        agent.train_steps = int(payload.get("train_steps", 0))

        if load_buffer and "buffer_state_dict" in payload:
            agent.buffer.load_state_dict(payload["buffer_state_dict"])

        agent.online_network.eval()
        agent.target_network.eval()
        return agent

    @staticmethod
    def read_metadata(path: str) -> Dict[str, Any]:
        """读取 checkpoint 里的元数据（不构造完整智能体）。"""
        payload = _torch_load(path)
        return json.loads(payload.get("metadata_json", "{}"))

    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.online_network.parameters())

    def describe(self) -> str:
        cfg = self.config
        return (
            f"DQN obs={cfg.obs_dim} actions={cfg.n_actions} "
            f"hidden={tuple(cfg.hidden_sizes)} params={self.parameter_count()} "
            f"lr={cfg.learning_rate} gamma={cfg.gamma} batch={cfg.batch_size} "
            f"buffer={cfg.buffer_capacity} target_sync={cfg.target_sync_steps} "
            f"eps={cfg.epsilon_start}->{cfg.epsilon_end}/{cfg.epsilon_decay_steps}步 "
            f"double_dqn={cfg.double_dqn}"
        )


def _torch_load(path: str, device: str = "cpu") -> Dict[str, Any]:
    """加载 checkpoint。

    这里显式使用 ``weights_only=False``：checkpoint 内除网络权重外还包含
    Adam 优化器状态（含参数组等非张量结构），torch 的 weights_only unpickler
    在部分版本上无法重建它。

    **安全提示**：``torch.load`` 会反序列化 pickle，可能执行任意代码。
    因此只加载本工程自己训练产出的 ``.pt`` 文件，不要加载来源不明的 checkpoint。

    另：torch 2.3.x 在加载本工程的 checkpoint 时会向 stderr 打印一段
    `_rebuild_tensor` 的 traceback（内部 legacy-storage 回退路径），
    **加载本身是成功的**。这里把 stderr 临时接住，避免无害噪声误导使用者；
    若加载真的抛错，则原样重放捕获内容后再抛出，不吞掉任何真实诊断信息。
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        captured = buffer.getvalue()
        if captured:
            print(captured, file=sys.stderr)
        raise


def silence_numpy_bridge_warning() -> None:
    """屏蔽 torch 初始化 numpy 桥接失败的无害告警。

    本工程 RL 路径完全不使用 numpy；但当 torch 版本按 NumPy 1.x 编译、
    环境里却是 NumPy 2.x 时（例如 `D:\\anaconda\\envs\\pytorch_env`），
    torch 首次做张量初始化会打印一次
    `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`。
    该告警不影响任何计算，这里在入口处显式抑制，避免刷屏误导使用者。
    """
    import warnings

    warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")
