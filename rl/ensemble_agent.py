"""集成 DQN（Bootstrapped Ensemble）：给「智能体知不知道自己在犯错」提供量化依据。

为什么需要它
------------
普通 DQN 只输出一组 Q 值，`argmax` 之后没有任何「这个决策有多可靠」的信息。
部分可观测条件下这一点很危险：观测丢测、延迟、噪声大时，网络仍会给出一个
看似确定的 argmax，而这个 argmax 可能只是因为输入落在训练分布之外。

本模块用 **N 个独立初始化的 Q 网络 + 自助采样（bootstrap）掩码** 组成集成：

    Q_mean(a) = (1/N) Σ_i Q_i(a)
    Q_std(a)  = sqrt( (1/N) Σ_i (Q_i(a) - Q_mean(a))^2 )

Q_std 就是**认知不确定度**（epistemic uncertainty）：网络之间越不一致，
说明这个状态下的估值越不可靠。这是深度集成在 RL 里的标准用法，
不需要额外标签，也不需要改动环境。

两个不确定度信号（互相独立，缺一不可）
--------------------------------------
1. **集成分歧** `q_std_max` / `q_std_at_best`：认知不确定度。
2. **分布外评分** `ood_score`：观测向量相对训练期观测统计量的标准化偏离
   （对每个维度算 |z| 再取均值）。集成分歧在「N 个网络犯同一个错」时会失效，
   而 OOD 评分能在输入本身就很陌生时报警。两者都高才最危险。

诚实说明
--------
* 这不是「学习到的置信度」，而是**基于集成分歧的启发式不确定度**。
  它是有理论依据的（Lakshminarayanan 等 2017 的深度集成；Osband 等 2016
  的 Bootstrapped DQN），但**不是校准过的概率**，不能解释成「犯错的概率」。
* 集成分歧小 **不保证** 决策正确——N 个网络可能在同一个状态上一致地错。
  因此 P2 的回退规则同时看集成分歧、OOD 评分和观测质量，而不是只看一个。
* 自助掩码让各成员看到不同的数据子集，但它们共享同一个回放缓冲区
  （不是各训各的缓冲区），这是为控制训练成本做的取舍，会**降低**成员多样性，
  进而让集成分歧偏小。这一点在报告里必须写清楚，不能说成「完全独立的 N 个模型」。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rl.dqn_agent import (
    CHECKPOINT_FORMAT_VERSION,
    DQNConfig,
    DQNAgent,
    QNetwork,
    _torch_load,
    silence_numpy_bridge_warning,
)

silence_numpy_bridge_warning()

ENSEMBLE_FORMAT_VERSION = 1


@dataclass
class EnsembleConfig:
    """集成超参数。"""

    ensemble_size: int = 5  # 成员数（v4.0 要求 3~5）
    bootstrap_prob: float = 0.8  # 每个成员看到每个样本的概率（自助掩码）
    member_seed_stride: int = 1000  # 成员 i 的种子 = base_seed + i*stride，保证初始化不同
    ood_ema_decay: float = 0.01  # 观测统计量的在线更新率
    ood_warmup_steps: int = 200  # 前多少步不报 OOD（统计量还没稳）

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnsembleConfig":
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> None:
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size 至少为 2（否则无法计算分歧）")
        if not 0.0 < self.bootstrap_prob <= 1.0:
            raise ValueError("bootstrap_prob 必须落在 (0, 1] 内")


@dataclass
class UncertaintyInfo:
    """一次决策的不确定度画像。"""

    q_mean_best: float = 0.0
    q_std_at_best: float = 0.0
    q_std_max: float = 0.0
    q_margin: float = 0.0  # 最优与次优的 Q 均值之差
    disagreement: float = 0.0  # 成员 argmax 与集成 argmax 不一致的比例
    ood_score: float = 0.0
    n_feasible: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "q_mean_best": round(self.q_mean_best, 6),
            "q_std_at_best": round(self.q_std_at_best, 6),
            "q_std_max": round(self.q_std_max, 6),
            "q_margin": round(self.q_margin, 6),
            "disagreement": round(self.disagreement, 6),
            "ood_score": round(self.ood_score, 6),
            "n_feasible": self.n_feasible,
        }


class EnsembleDQNAgent(DQNAgent):
    """N 个 Q 网络的集成，共用回放缓冲区，按自助掩码分别更新。"""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        ensemble_config: Optional[EnsembleConfig] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, device=device)
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.ensemble_config.validate()

        cfg = self.config
        # 用父类建好的成员 0 作为第一个成员，其余成员重新初始化
        self.online_networks: List[QNetwork] = [self.online_network]
        self.target_networks: List[QNetwork] = [self.target_network]
        self.optimizers: List[torch.optim.Optimizer] = [self.optimizer]

        for index in range(1, self.ensemble_config.ensemble_size):
            member_seed = int(cfg.seed) + index * int(self.ensemble_config.member_seed_stride)
            torch.manual_seed(member_seed)
            online = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
            target = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
            target.load_state_dict(online.state_dict())
            self.online_networks.append(online)
            self.target_networks.append(target)
            self.optimizers.append(
                torch.optim.Adam(online.parameters(), lr=cfg.learning_rate)
            )

        # 观测统计量（Welford 在线均值/方差）——用于分布外评分
        self._obs_mean = torch.zeros(cfg.obs_dim, dtype=torch.float32)
        self._obs_m2 = torch.zeros(cfg.obs_dim, dtype=torch.float32)
        self._obs_count = 0
        self._ood_warned = False

    # ------------------------------------------------------------------
    # 观测统计量 / OOD
    # ------------------------------------------------------------------

    def _update_obs_stats(self, obs: Sequence[float]) -> None:
        """在线更新观测均值与方差（Welford 算法，无需保存全部历史）。"""
        vector = torch.tensor(list(obs), dtype=torch.float32)
        self._obs_count += 1
        delta = vector - self._obs_mean
        self._obs_mean = self._obs_mean + delta / self._obs_count
        self._obs_m2 = self._obs_m2 + delta * (vector - self._obs_mean)

    def _ood_score(self, obs: Sequence[float]) -> float:
        """观测相对训练分布的标准化偏离（各维 |z| 的均值）。"""
        if self._obs_count < self.ensemble_config.ood_warmup_steps:
            return 0.0
        variance = self._obs_m2 / max(self._obs_count - 1, 1)
        std = torch.sqrt(torch.clamp(variance, min=1e-12))
        vector = torch.tensor(list(obs), dtype=torch.float32)
        z = torch.abs(vector - self._obs_mean) / std
        return float(z.mean().item())

    @property
    def obs_stats(self) -> Dict[str, Any]:
        variance = self._obs_m2 / max(self._obs_count - 1, 1)
        return {
            "count": int(self._obs_count),
            "mean": [round(float(v), 8) for v in self._obs_mean],
            "std": [round(float(math.sqrt(max(float(v), 0.0))), 8) for v in variance],
        }

    # ------------------------------------------------------------------
    # 存储（顺带更新统计量）
    # ------------------------------------------------------------------

    def store(self, *args: Any, **kwargs: Any) -> None:
        # 父类签名是 store(obs, action, reward, next_obs, done, next_mask, cost)
        obs = args[0] if args else kwargs.get("obs")
        if obs is not None:
            self._update_obs_stats(obs)
        super().store(*args, **kwargs)

    # ------------------------------------------------------------------
    # 推理：Q 均值 / 分歧
    # ------------------------------------------------------------------

    def q_ensemble(self, obs: Sequence[float]) -> torch.Tensor:
        """返回 (N, n_actions) 的 Q 矩阵。"""
        tensor = torch.tensor(list(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            rows = [net(tensor).squeeze(0).cpu() for net in self.online_networks]
        return torch.stack(rows, dim=0)

    def q_values(self, obs: Sequence[float]) -> torch.Tensor:
        """集成平均 Q 值（覆盖父类，使父类 select_action 也能直接用）。"""
        return self.q_ensemble(obs).mean(dim=0)

    def uncertainty(
        self, obs: Sequence[float], action_mask: Optional[Sequence[bool]] = None
    ) -> Tuple[UncertaintyInfo, torch.Tensor]:
        """计算不确定度画像，并返回 (info, Q 均值向量)。"""
        q_all = self.q_ensemble(obs)  # (N, A)
        q_mean = q_all.mean(dim=0)
        q_std = q_all.std(dim=0, unbiased=False)

        n_actions = q_mean.shape[0]
        if action_mask is None:
            feasible = list(range(n_actions))
        else:
            feasible = [i for i, ok in enumerate(action_mask) if ok]
            if not feasible:
                feasible = list(range(n_actions))

        # 集成 argmax 与各成员 argmax 的一致性
        best = feasible[0]
        for level in feasible[1:]:
            if float(q_mean[level]) > float(q_mean[best]):
                best = level
        member_winners = [
            max(feasible, key=lambda a: float(q_all[m][a]))
            for m in range(q_all.shape[0])
        ]
        disagreement = sum(1 for w in member_winners if w != best) / len(member_winners)

        # 次优动作的 Q 均值
        others = [a for a in feasible if a != best]
        second = max((float(q_mean[a]) for a in others), default=float("-inf"))
        margin = float(q_mean[best]) - second if others else float("inf")

        info = UncertaintyInfo(
            q_mean_best=float(q_mean[best]),
            q_std_at_best=float(q_std[best]),
            q_std_max=max(float(q_std[a]) for a in feasible),
            q_margin=margin if math.isfinite(margin) else 0.0,
            disagreement=disagreement,
            ood_score=self._ood_score(obs),
            n_feasible=len(feasible),
        )
        return info, q_mean

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train_step(self) -> float:
        """每个成员在**自助采样的子批次**上各做一次更新。"""
        cfg = self.config
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

        next_mask_bool = next_masks > 0.5
        has_feasible = next_mask_bool.any(dim=1, keepdim=True)
        safe_mask = torch.where(
            has_feasible, next_mask_bool, torch.ones_like(next_mask_bool)
        )

        total_loss = 0.0
        probability = float(self.ensemble_config.bootstrap_prob)
        batch_size = int(cfg.batch_size)

        for member in range(self.ensemble_config.ensemble_size):
            # 自助掩码：每个成员看到 batch 的一个随机子集
            if probability >= 1.0:
                mask = torch.ones(batch_size, 1, device=self.device)
            else:
                bernoulli = torch.rand(batch_size, 1, device=self.device)
                mask = (bernoulli < probability).float()
                if float(mask.sum()) < 1.0:
                    mask = torch.ones(batch_size, 1, device=self.device)

            target_net = self.target_networks[member]
            online_net = self.online_networks[member]
            optimizer = self.optimizers[member]

            with torch.no_grad():
                q_next = target_net(next_states)
                if cfg.double_dqn:
                    q_online_next = online_net(next_states).masked_fill(
                        ~safe_mask, float("-inf")
                    )
                    next_actions = q_online_next.argmax(dim=1, keepdim=True)
                    next_q = q_next.gather(1, next_actions)
                else:
                    next_q = q_next.masked_fill(~safe_mask, float("-inf")).max(
                        dim=1, keepdim=True
                    )[0]
                targets = rewards + cfg.gamma * (1.0 - dones) * next_q

            q_values = online_net(states).gather(1, actions)
            per_sample = self.loss_fn(q_values, targets)
            # 按掩码加权，并除以掩码均值，使损失量级与全体本训练一致
            loss = (per_sample * mask).sum() / mask.sum().clamp(min=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.item())

        self.train_steps += 1
        self.last_loss = total_loss / self.ensemble_config.ensemble_size

        if cfg.target_sync_steps > 0 and self.train_steps % cfg.target_sync_steps == 0:
            self.sync_target()
        return self.last_loss

    def sync_target(self) -> None:
        """同步全部成员的目标网络。"""
        networks = getattr(self, "online_networks", None)
        if not networks:
            # __init__ 期间父类会先调用一次，此时成员列表还没建好
            if hasattr(self, "online_network") and hasattr(self, "target_network"):
                self.target_network.load_state_dict(self.online_network.state_dict())
            return
        for online, target in zip(networks, self.target_networks):
            target.load_state_dict(online.state_dict())

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        include_buffer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload: Dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "ensemble_format_version": ENSEMBLE_FORMAT_VERSION,
            "is_ensemble": True,
            "config_json": self.config.to_json(),
            "ensemble_config_json": json.dumps(
                {
                    k: getattr(self.ensemble_config, k)
                    for k in self.ensemble_config.__dataclass_fields__
                },
                sort_keys=True,
            ),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            "online_state_dicts": [net.state_dict() for net in self.online_networks],
            "target_state_dicts": [net.state_dict() for net in self.target_networks],
            "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
            "epsilon": float(self.epsilon),
            "env_steps": int(self.env_steps),
            "train_steps": int(self.train_steps),
            "obs_mean": self._obs_mean,
            "obs_m2": self._obs_m2,
            "obs_count": int(self._obs_count),
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
    ) -> "EnsembleDQNAgent":
        payload = _torch_load(path, device=device)
        if not payload.get("is_ensemble", False):
            raise ValueError(
                f"{path} 不是集成 checkpoint（单模型请用 DQNAgent.load）"
            )
        config = DQNConfig.from_json(payload["config_json"])
        ensemble_config = EnsembleConfig.from_dict(
            json.loads(payload.get("ensemble_config_json", "{}"))
        )
        agent = cls(config, ensemble_config=ensemble_config, device=device)

        online_states = payload["online_state_dicts"]
        target_states = payload["target_state_dicts"]
        if len(online_states) != agent.ensemble_config.ensemble_size:
            raise ValueError(
                f"checkpoint 成员数 {len(online_states)} 与配置 "
                f"{agent.ensemble_config.ensemble_size} 不一致"
            )
        for net, state in zip(agent.online_networks, online_states):
            net.load_state_dict(state)
            net.eval()
        for net, state in zip(agent.target_networks, target_states):
            net.load_state_dict(state)
            net.eval()
        for optimizer, state in zip(agent.optimizers, payload["optimizer_state_dicts"]):
            optimizer.load_state_dict(state)

        agent.epsilon = float(payload.get("epsilon", config.epsilon_end))
        agent.env_steps = int(payload.get("env_steps", 0))
        agent.train_steps = int(payload.get("train_steps", 0))
        agent._obs_mean = payload["obs_mean"].to("cpu")
        agent._obs_m2 = payload["obs_m2"].to("cpu")
        agent._obs_count = int(payload.get("obs_count", 0))

        if load_buffer and "buffer_state_dict" in payload:
            agent.buffer.load_state_dict(payload["buffer_state_dict"])
        return agent

    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return sum(
            sum(p.numel() for p in net.parameters()) for net in self.online_networks
        )

    def describe(self) -> str:
        cfg = self.config
        return (
            f"集成DQN N={self.ensemble_config.ensemble_size} "
            f"(bootstrap p={self.ensemble_config.bootstrap_prob}) "
            f"obs={cfg.obs_dim} actions={cfg.n_actions} hidden={tuple(cfg.hidden_sizes)} "
            f"总参数={self.parameter_count()} lr={cfg.learning_rate} gamma={cfg.gamma} "
            f"batch={cfg.batch_size} target_sync={cfg.target_sync_steps} "
            f"eps={cfg.epsilon_start}->{cfg.epsilon_end}/{cfg.epsilon_decay_steps}步"
        )
