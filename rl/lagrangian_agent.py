"""安全强化学习分支：拉格朗日约束 DQN（**不替换**主 DQN）。

问题设定
--------
主 DQN 把「探测未达标」写成一个**固定惩罚**塞进奖励里：

    r = … − w_violation · 1[Pd < required_pd]

这有两个弱点：
1. 惩罚权重得手调，且与低截获/能耗目标在同一量纲上直接相加，**无法指定目标违反率**；
2. 训练过程中无法知道「现在约束满足得怎么样」，只能看奖励总量。

约束版换成标准做法：把违反率当成**显式约束**

    maximize  E[Σ r_t]        （r 里**去掉**违反惩罚项）
    s.t.      E[平均代价] ≤ d   （代价 c_t = 1[Pd_t < required_pd]，d 为目标违反率）

用拉格朗日松弛把约束搬到目标里：

    L(θ, λ) = E[Σ (r_t − λ · c_t)] − λ · d · T

对偶上升更新乘子（每个 episode 一次）：

    λ ← clip( λ + η_λ · ( ĉ_episode − d ), 0, λ_max )

实现要点
--------
* **两个 critic**：奖励 critic `Q_r` 与代价 critic `Q_c`，各自带目标网络、
  同一套动作可行性掩码与 Huber 损失；
* 动作选择用 `Q_r − λ·Q_c` 在**可行档位**内取最大——
  这正是「在满足约束的前提下最大化收益」的贪心近似；
* `λ = 0` 时退化为普通 DQN，因此本实现与主 DQN 完全兼容、可平滑对比；
* **不修改主 DQN**：本类继承 `DQNAgent` 只覆写训练与动作选择，
  主实验路径完全不变（旧结果可复现）。

诚实性说明
----------
本实现是「拉格朗日松弛 + 对偶上升」的标准工程近似：
λ 用**本 episode 的实际违反率**做随机逼近更新（而不是对 Q_c 求期望），
这样更稳、更容易复现；代价 critic 仍被正常训练并用于动作选择。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .dqn_agent import DQNAgent, DQNConfig, QNetwork

logger = logging.getLogger("lpi.rl.lagrangian")


@dataclass
class LagrangianConfig:
    """安全 RL 的约束参数。

    关于 λ 的量级：本任务**满足一步探测的奖励约 +1.0，未达标一次代价 1.0**，
    因此有意义的 λ 区间大致是 **[0.5, 1.5]**——
    λ 太大（如 5、20）会让代价项彻底压倒奖励项，策略退化成
    「不惜一切代价不违反当前这一步」，于是疯狂提功率、把能量提前烧光，
    结果**违反率反而上升**（跑不完的步同样算违反）。
    这是把朴素的拉格朗日对偶上升直接套到"带硬资源约束"环境上的典型失效模式，
    `lambda_max` 默认 1.5 就是为了把 λ 限制在有效区间内。
    """

    cost_limit: float = 0.08  # 目标违反率上限 d（对标规则策略约 0.082）
    lambda_lr: float = 0.01  # 对偶上升步长 η_λ
    lambda_max: float = 1.5  # 乘子上界（见上：与"探测项 +1.0"的量级对齐）
    lambda_init: float = 0.0  # 初始乘子


class LagrangianDQNAgent(DQNAgent):
    """拉格朗日约束 DQN：双 critic + 动态 λ。"""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        constraint: Optional[LagrangianConfig] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, device=device)
        self.constraint = constraint or LagrangianConfig()

        # --- 代价 critic（与奖励 critic 同结构）---
        self.cost_network = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_sizes
        ).to(self.device)
        self.cost_target_network = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_sizes
        ).to(self.device)
        self.cost_target_network.load_state_dict(self.cost_network.state_dict())

        self.cost_optimizer = torch.optim.Adam(
            self.cost_network.parameters(), lr=self.config.learning_rate
        )
        self.cost_loss_fn = nn.SmoothL1Loss(beta=self.config.huber_beta)

        self.lambda_cost = float(self.constraint.lambda_init)
        self.last_cost_loss: Optional[float] = None
        self.lambda_history: List[Dict[str, Any]] = []
        self._episode_costs: List[float] = []
        self._episode_horizon: int = 0

    def set_episode_horizon(self, horizon: int) -> None:
        """设置本 episode 的**完整任务步数**。

        代价率必须以完整任务步数为分母：能量耗尽导致没执行到的步同样算"没完成探测任务"，
        否则会出现「早早烧光能量 -> 违反率为 0」的漏洞（与评测口径保持一致）。
        """
        self._episode_horizon = int(horizon)

    # ------------------------------------------------------------------
    # 动作选择：max (Q_r − λ·Q_c)
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Sequence[float],
        greedy: bool = False,
        action_mask: Sequence[bool] | None = None,
    ) -> int:
        if action_mask is None:
            feasible = list(range(self.config.n_actions))
        else:
            feasible = [i for i, ok in enumerate(action_mask) if ok]
            if not feasible:
                raise ValueError("action_mask 全为 False，episode 应当已结束")

        if not greedy and self._rng.random() < self.epsilon:
            return feasible[self._rng.randrange(len(feasible))]

        q_reward = self.q_values(obs)
        with torch.no_grad():
            tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_cost = self.cost_network(tensor).squeeze(0).cpu()

        score = q_reward - self.lambda_cost * q_cost
        best = feasible[0]
        best_value = float(score[best])
        for level in feasible[1:]:
            value = float(score[level])
            if value > best_value:
                best_value = value
                best = level
        return best

    def q_cost_values(self, obs: Sequence[float]) -> torch.Tensor:
        with torch.no_grad():
            tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.cost_network(tensor).squeeze(0).cpu()

    # ------------------------------------------------------------------
    # 交互：额外记录代价
    # ------------------------------------------------------------------

    def store(  # type: ignore[override]
        self,
        obs: Sequence[float],
        action: int,
        reward: float,
        next_obs: Sequence[float],
        terminated: bool,
        next_action_mask: Sequence[bool] | None = None,
        cost: float = 0.0,
    ) -> None:
        self.buffer.push(
            obs, action, reward, next_obs, float(terminated), next_action_mask, cost
        )
        self._episode_costs.append(float(cost))

    def reset_episode_cost(self) -> None:
        self._episode_costs = []

    @property
    def episode_cost_rate(self) -> float:
        """本 episode 的约束违反率 = 未满足任务步 / **完整任务步数**。"""
        if not self._episode_costs:
            return 0.0
        denominator = self._episode_horizon or len(self._episode_costs)
        return sum(self._episode_costs) / denominator

    # ------------------------------------------------------------------
    # 训练：两个 critic
    # ------------------------------------------------------------------

    def train_step(self) -> float:  # type: ignore[override]
        cfg = self.config
        (
            states, actions, rewards, next_states, dones, next_masks, costs,
        ) = self.buffer.sample(cfg.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        next_masks = next_masks.to(self.device)
        costs = costs.to(self.device).unsqueeze(1)

        next_mask_bool = next_masks > 0.5
        has_feasible = next_mask_bool.any(dim=1, keepdim=True)
        safe_mask = torch.where(
            has_feasible, next_mask_bool, torch.ones_like(next_mask_bool)
        )

        # --- 奖励 critic ---
        with torch.no_grad():
            q_next = self.target_network(next_states)
            if cfg.double_dqn:
                q_online_next = self.online_network(next_states).masked_fill(
                    ~safe_mask, float("-inf")
                )
                next_actions = q_online_next.argmax(dim=1, keepdim=True)
                next_q = q_next.gather(1, next_actions)
            else:
                next_q = q_next.masked_fill(~safe_mask, float("-inf")).max(
                    dim=1, keepdim=True
                )[0]
            reward_targets = rewards + cfg.gamma * (1.0 - dones) * next_q

        q_values = self.online_network(states).gather(1, actions)
        reward_loss = self.loss_fn(q_values, reward_targets)

        # --- 代价 critic ---
        with torch.no_grad():
            qc_next = self.cost_target_network(next_states).masked_fill(
                ~safe_mask, float("-inf")
            ).max(dim=1, keepdim=True)[0]
            cost_targets = costs + cfg.gamma * (1.0 - dones) * qc_next

        qc_values = self.cost_network(states).gather(1, actions)
        cost_loss = self.cost_loss_fn(qc_values, cost_targets)

        # --- 依次更新（两个优化器互不干扰）---
        self.optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        if cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online_network.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()

        self.cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        if cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.cost_network.parameters(), cfg.grad_clip_norm)
        self.cost_optimizer.step()

        self.train_steps += 1
        self.last_loss = float(reward_loss.item())
        self.last_cost_loss = float(cost_loss.item())

        if cfg.target_sync_steps > 0 and self.train_steps % cfg.target_sync_steps == 0:
            self.sync_target()

        return self.last_loss

    def sync_target(self) -> None:  # type: ignore[override]
        super().sync_target()
        if hasattr(self, "cost_network"):
            self.cost_target_network.load_state_dict(self.cost_network.state_dict())

    # ------------------------------------------------------------------
    # 对偶上升
    # ------------------------------------------------------------------

    def update_lambda(
        self, episode: int = 0, cost_rate_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """按违反率做一次对偶上升。

        ⚠️ **必须用「贪心策略」的违反率，而不是 ε-greedy 训练 rollout 的违反率。**

        训练 rollout 带 ε 探索噪声（早期 ε=1），其违反率远高于部署时真正执行的策略。
        若直接用它做对偶上升，λ 会被探索噪声一路顶到上界，
        结果部署策略被过大的 λ 压得"不惜一切代价不违反当前这一步"——
        疯狂提功率、把能量提前烧光，而**跑不完的步同样算违反**，
        违反率反而从 0.082 升到 0.197。这是本项目实际踩过的坑。

        因此调用方应在**周期性贪心评测**后用 `cost_rate_override` 传入真实违反率。
        """
        cost_rate = (
            self.episode_cost_rate if cost_rate_override is None else float(cost_rate_override)
        )
        limit = self.constraint.cost_limit
        old = self.lambda_cost
        self.lambda_cost = min(
            self.constraint.lambda_max,
            max(0.0, self.lambda_cost + self.constraint.lambda_lr * (cost_rate - limit)),
        )
        record = {
            "episode": episode,
            "cost_rate": round(cost_rate, 6),
            "cost_limit": limit,
            "lambda_before": round(old, 6),
            "lambda_after": round(self.lambda_cost, 6),
            "violation_excess": round(cost_rate - limit, 6),
            "source": "greedy_eval" if cost_rate_override is not None else "train_rollout",
        }
        self.lambda_history.append(record)
        return record

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(  # type: ignore[override]
        self,
        path: str,
        include_buffer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(metadata or {})
        meta.update(
            {
                "agent_kind": "lagrangian_dqn",
                "cost_limit": self.constraint.cost_limit,
                "lambda_lr": self.constraint.lambda_lr,
                "lambda_max": self.constraint.lambda_max,
                "lambda_cost": round(self.lambda_cost, 6),
            }
        )
        super().save(path, include_buffer=include_buffer, metadata=meta)
        # 额外把代价 critic 与 λ 追加写进同一个 checkpoint
        import os

        payload = _torch_load_local(path, self.device)
        payload["cost_network_state_dict"] = self.cost_network.state_dict()
        payload["cost_target_state_dict"] = self.cost_target_network.state_dict()
        payload["cost_optimizer_state_dict"] = self.cost_optimizer.state_dict()
        payload["lambda_cost"] = float(self.lambda_cost)
        torch.save(payload, path)

    @classmethod
    def load(  # type: ignore[override]
        cls, path: str, device: str = "cpu", load_buffer: bool = False
    ) -> "LagrangianDQNAgent":
        payload = _torch_load_local(path, device)
        config = DQNConfig.from_json(payload["config_json"])
        meta = __import__("json").loads(payload.get("metadata_json", "{}"))
        constraint = LagrangianConfig(
            cost_limit=float(meta.get("cost_limit", 0.05)),
            lambda_lr=float(meta.get("lambda_lr", 0.05)),
            lambda_max=float(meta.get("lambda_max", 20.0)),
            lambda_init=float(payload.get("lambda_cost", 0.0)),
        )
        agent = cls(config, constraint=constraint, device=device)

        agent.online_network.load_state_dict(payload["online_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "cost_network_state_dict" in payload:
            agent.cost_network.load_state_dict(payload["cost_network_state_dict"])
            agent.cost_target_network.load_state_dict(payload["cost_target_state_dict"])
            agent.cost_optimizer.load_state_dict(payload["cost_optimizer_state_dict"])
        agent.lambda_cost = float(payload.get("lambda_cost", constraint.lambda_init))
        agent.epsilon = float(payload.get("epsilon", config.epsilon_end))
        agent.env_steps = int(payload.get("env_steps", 0))
        agent.train_steps = int(payload.get("train_steps", 0))

        if load_buffer and "buffer_state_dict" in payload:
            agent.buffer.load_state_dict(payload["buffer_state_dict"])

        agent.online_network.eval()
        agent.target_network.eval()
        agent.cost_network.eval()
        agent.cost_target_network.eval()
        return agent

    # ------------------------------------------------------------------

    def describe(self) -> str:
        base = super().describe()
        return (
            f"{base} | 【约束版】cost_limit={self.constraint.cost_limit} "
            f"λ={self.lambda_cost:.4f} η_λ={self.constraint.lambda_lr} "
            f"λ_max={self.constraint.lambda_max} 双 critic"
        )


def _torch_load_local(path: str, device: Any) -> Dict[str, Any]:
    import torch as _torch

    return _torch.load(path, map_location=device, weights_only=False)
