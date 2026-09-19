"""Actor-Critic 策略：逐节点因素化分类头 + 合法动作 mask（torch）。

结构
----
```
观测(定长) ──► 共享 MLP [128, 128] ──┬──► actor 头 ──► (max_nodes, 4) logits
                                     └──► critic 头 ─► V(s)
```

* **actor 是因素化的**：对每个节点槽位输出 4 个 logits，联合动作的
  log 概率 = 各节点 log 概率之和。参数量与节点数**线性**相关，
  不是把 4^N 个联合动作展平（见 `actions.py` 的说明）。
* **mask 在 logits 上做**：非法动作的 logit 被置为极小值，
  因此采样**不可能**产生非法动作，`log_prob` 也只覆盖合法集合。
* **观测与动作张量形状在 episode 间恒定**（`max_nodes` 固定），
  训练循环里不需要按场景分支。

这个工程刻意**不用 numpy**：`torch 2.3.1` 按 NumPy 1.x 编译而环境里是
NumPy 2.x，桥接不可用（见 `rl/` 的同一条说明），因此全程用 list/torch。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rl_resource.actions import IDLE_ONLY_ROW, N_ACTIONS
from rl_resource.obs import DEFAULT_MAX_NODES, observation_dim

#: mask 里非法动作的 logit 取值（用大负数而非 -inf：避免与 0 相乘出 nan）
MASKED_LOGIT = -1e8


@dataclass
class PolicyConfig:
    obs_dim: int = 0
    max_nodes: int = DEFAULT_MAX_NODES
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    init_log_std: float = 0.0
    seed: int = 0
    #: ``concat`` keeps the historical state+preference MLP. ``film`` keeps
    #: state and preference inputs separate and modulates each hidden layer.
    conditioning: str = "concat"
    state_obs_dim: int = 0
    preference_dim: int = 0
    preference_hidden_size: int = 8

    def resolved(self) -> "PolicyConfig":
        if self.obs_dim <= 0:
            return PolicyConfig(
                obs_dim=observation_dim(self.max_nodes),
                max_nodes=self.max_nodes, hidden_sizes=self.hidden_sizes,
                activation=self.activation, init_log_std=self.init_log_std,
                seed=self.seed, conditioning=self.conditioning,
                state_obs_dim=self.state_obs_dim,
                preference_dim=self.preference_dim,
                preference_hidden_size=self.preference_hidden_size)
        if self.conditioning not in {"concat", "film"}:
            raise ValueError("conditioning 只能是 'concat' 或 'film'")
        if self.conditioning == "film":
            if self.state_obs_dim <= 0 or self.preference_dim <= 0:
                raise ValueError("FiLM policy 必须声明 state_obs_dim 和 preference_dim")
            if self.state_obs_dim + self.preference_dim != self.obs_dim:
                raise ValueError("FiLM state/preference 维度必须恰好组成 obs_dim")
            if self.preference_hidden_size <= 0:
                raise ValueError("FiLM preference_hidden_size 必须为正")
        return self


def _activation(name: str) -> nn.Module:
    table = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
    if name not in table:
        raise ValueError(f"未知激活函数 {name!r}；可选 {sorted(table)}")
    return table[name]()


class ActorCritic(nn.Module):
    """共享主干 + 逐节点 actor 头 + 标量 critic 头。"""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__()
        self.config = (config or PolicyConfig()).resolved()
        last = self.config.obs_dim
        if self.config.conditioning == "concat":
            layers: List[nn.Module] = []
            for size in self.config.hidden_sizes:
                layers.append(nn.Linear(last, int(size)))
                layers.append(_activation(self.config.activation))
                last = int(size)
            self.trunk: Optional[nn.Sequential] = nn.Sequential(*layers)
            self.state_layers: Optional[nn.ModuleList] = None
            self.film_layers: Optional[nn.ModuleList] = None
            self.film_activations: Optional[nn.ModuleList] = None
            self.preference_encoder: Optional[nn.Sequential] = None
        else:
            # v3 representation-only change: the state trunk retains [128,128]
            # capacity, while a small preference encoder produces a scale/shift
            # pair for every hidden layer.  The 0.1 bounded modulation keeps the
            # initial policy numerically close to the established MLP scale.
            self.trunk = None
            self.state_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            self.film_activations = nn.ModuleList()
            self.preference_encoder = nn.Sequential(
                nn.Linear(self.config.preference_dim,
                          self.config.preference_hidden_size),
                _activation(self.config.activation))
            last = self.config.state_obs_dim
            for size in self.config.hidden_sizes:
                width = int(size)
                self.state_layers.append(nn.Linear(last, width))
                self.film_layers.append(nn.Linear(
                    self.config.preference_hidden_size, 2 * width))
                self.film_activations.append(_activation(self.config.activation))
                last = width
        self.actor = nn.Linear(last, self.config.max_nodes * N_ACTIONS)
        self.critic = nn.Linear(last, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.conditioning == "concat":
            assert self.trunk is not None
            features = self.trunk(obs)
        else:
            assert (self.state_layers is not None and self.film_layers is not None
                    and self.film_activations is not None
                    and self.preference_encoder is not None)
            state = obs[..., :self.config.state_obs_dim]
            preference = obs[..., self.config.state_obs_dim:]
            pref_features = self.preference_encoder(preference)
            features = state
            for layer, film, activation in zip(
                    self.state_layers, self.film_layers, self.film_activations):
                raw = layer(features)
                gamma, beta = film(pref_features).chunk(2, dim=-1)
                features = activation(raw * (1.0 + 0.1 * torch.tanh(gamma))
                                      + 0.1 * torch.tanh(beta))
        logits = self.actor(features).view(
            -1, self.config.max_nodes, N_ACTIONS)
        value = self.critic(features).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------

    @staticmethod
    def masked_logits(logits: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """把非法动作的 logit 压到 `MASKED_LOGIT`。"""
        return logits.masked_fill(~mask.bool(), MASKED_LOGIT)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor
                     ) -> Tuple[torch.distributions.Categorical,
                                torch.Tensor]:
        logits, value = self.forward(obs)
        masked = self.masked_logits(logits, mask)
        return torch.distributions.Categorical(logits=masked), value

    def act(self, obs: torch.Tensor, mask: torch.Tensor,
            deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """采样（或确定性地选）一个联合动作，返回动作/对数概率/熵/价值。

        `deterministic=True` 时取 masked logits 的 argmax——**评测与
        checkpoint 选择必须用它**。用随机采样做评测会把"选哪个 checkpoint"
        变成抽奖：实测同一策略两次采样的验证回报能差 ±7，
        而学习信号本身也在同一量级，选出来的是噪声不是进步。
        """
        dist, value = self.distribution(obs, mask)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return {
            "action": action,
            "log_prob": dist.log_prob(action).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "value": value,
        }

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor,
                 action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """给定动作重算 log 概率 / 熵 / 价值（PPO 更新用）。"""
        dist, value = self.distribution(obs, mask)
        return {
            "log_prob": dist.log_prob(action).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "value": value,
        }

    # ------------------------------------------------------------------

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "obs_dim": self.config.obs_dim,
                "max_nodes": self.config.max_nodes,
                "hidden_sizes": list(self.config.hidden_sizes),
                "activation": self.config.activation,
                "conditioning": self.config.conditioning,
                "state_obs_dim": self.config.state_obs_dim,
                "preference_dim": self.config.preference_dim,
                "preference_hidden_size": self.config.preference_hidden_size,
            },
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        return path

    @staticmethod
    def load(path: str, map_location: str = "cpu"
             ) -> Tuple["ActorCritic", Dict[str, Any]]:
        payload = torch.load(path, map_location=map_location)
        raw = payload.get("config") or {}
        config = PolicyConfig(
            obs_dim=int(raw.get("obs_dim", 0)),
            max_nodes=int(raw.get("max_nodes", DEFAULT_MAX_NODES)),
            hidden_sizes=tuple(raw.get("hidden_sizes") or (128, 128)),
            activation=str(raw.get("activation", "tanh")),
            conditioning=str(raw.get("conditioning", "concat")),
            state_obs_dim=int(raw.get("state_obs_dim", 0)),
            preference_dim=int(raw.get("preference_dim", 0)),
            preference_hidden_size=int(raw.get("preference_hidden_size", 8)),
        ).resolved()
        model = ActorCritic(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload.get("extra") or {})


def mask_to_tensor(mask: Sequence[Sequence[bool]],
                   device: torch.device) -> torch.Tensor:
    return torch.tensor([[bool(value) for value in row] for row in mask],
                        dtype=torch.bool, device=device)


def legal_action_rates(mask: Sequence[Sequence[bool]]) -> Dict[str, float]:
    """掩码覆盖率统计：每个动作在多少比例的(节点,tick)上合法。

    它是"mask 到底约束了多少"的直接证据；同时能看出某类动作是否
    **从未**进入候选（那是环境问题，不是策略问题）。
    """
    counts = {index: 0 for index in range(N_ACTIONS)}
    total = 0
    for row in mask:
        total += 1
        for index, allowed in enumerate(row):
            if allowed:
                counts[index] += 1
    names = ("idle", "sample", "process", "share")
    return {names[index]: (counts[index] / total if total else 0.0)
            for index in range(N_ACTIONS)}


__all__ = [
    "MASKED_LOGIT", "ActorCritic", "PolicyConfig", "legal_action_rates",
    "mask_to_tensor",
]
