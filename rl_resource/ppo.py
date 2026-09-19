"""PPO + GAE（单文件实现，`rl_resource` 专用）。

为什么用 PPO 而不是 DQN
-----------------------
动作空间是**逐节点因素化**的（`max_nodes × 4`），而且**合法性随 tick 变化**
（能不能 sample / process / share 取决于运行时缓冲与预算）。DQN 需要
"对每个离散动作求 max"，在因素化 + 变长合法集上要么退化成动作表展开、
要么需要为每个节点单独学一个 Q——都比直接学一个带 mask 的分类策略别扭。
PPO 只需要 log 概率与 mask，天然适配。

实现要点（都是为了"结果可信"）
------------------------------
* **GAE(λ)** 用 `(1 − terminated)` 切断 bootstrap，**不用**
  `done = terminated or truncated`——外部截断必须保留 bootstrap
  （`docs/learning_protocol.md` §2.2）。`info["bootstrap_allowed"]` 是接口审计字段。
* **优势归一化**、**策略裁剪**、**价值裁剪**、**熵奖励**、**梯度裁剪**全部显式列出。
* 记录 `approx_kl` / `clip_fraction` / `entropy` / `value_loss`，
  用来判断"到底有没有在学"，而不是只看回报曲线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rl_resource.policy import ActorCritic, mask_to_tensor


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    normalize_advantage: bool = True
    target_kl: Optional[float] = 0.03
    seed: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda 必须在 [0, 1]")
        if self.n_epochs < 1 or self.batch_size < 1:
            raise ValueError("n_epochs 与 batch_size 必须为正")


# ----------------------------------------------------------------------
# 经验缓冲（纯 python 列表，不用 numpy）
# ----------------------------------------------------------------------


class RolloutBuffer:
    """一次 rollout 的转移。

    ⚠️ 这里显式区分两个概念，**不能合并成一个 `done`**：

    * `terminated`：episode **自然**结束（任务时域到达 / 资源耗尽）
      → bootstrap 必须切断，`next_value` 记为 0；
    * `episode_end`：本转移是该 episode 的**最后一条**（terminated 或 truncated）
      → 只用来**切断 GAE 的时间反向传播**，不让上一条 episode 的优势
      串到下一条里。

    外部截断（`truncated`）时 `episode_end=True` 但 `terminated=False`，
    因此 `next_value` 由调用方按 `V(s_{t+1})` 给出、**保留 bootstrap**——
    这正是 `docs/learning_protocol.md` §2.2 要求的语义。
    """

    def __init__(self) -> None:
        self.obs: List[List[float]] = []
        self.mask: List[List[List[bool]]] = []
        self.action: List[List[int]] = []
        self.log_prob: List[float] = []
        self.value: List[float] = []
        self.reward: List[float] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
        self.episode_end: List[bool] = []
        self.next_value: List[float] = []

    def add(self, obs: Sequence[float], mask: Sequence[Sequence[bool]],
            action: Sequence[int], log_prob: float, value: float,
            reward: float, terminated: bool, truncated: bool,
            next_value: float) -> None:
        self.obs.append(list(obs))
        self.mask.append([[bool(v) for v in row] for row in mask])
        self.action.append([int(v) for v in action])
        self.log_prob.append(float(log_prob))
        self.value.append(float(value))
        self.reward.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.episode_end.append(bool(terminated or truncated))
        self.next_value.append(float(next_value))

    def __len__(self) -> int:
        return len(self.reward)

    def clear(self) -> None:
        self.__init__()

    # ------------------------------------------------------------------

    def compute_gae(self, gamma: float, gae_lambda: float
                    ) -> Tuple[List[float], List[float]]:
        """广义优势估计（逐转移 next_value；episode 边界切断反向传播）。"""
        n = len(self.reward)
        advantages = [0.0] * n
        returns = [0.0] * n
        running = 0.0
        for index in range(n - 1, -1, -1):
            non_terminal = 0.0 if self.terminated[index] else 1.0
            delta = (self.reward[index]
                     + gamma * non_terminal * self.next_value[index]
                     - self.value[index])
            # 边界处不再向后串联（`episode_end`），但 terminated 的
            # bootstrap 切断由上面的 `non_terminal` 单独负责。
            carry = 0.0 if self.episode_end[index] else running
            running = delta + gamma * gae_lambda * non_terminal * carry
            advantages[index] = running
            returns[index] = running + self.value[index]
        return advantages, returns


# ----------------------------------------------------------------------
# 更新
# ----------------------------------------------------------------------


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    n_updates: int = 0
    early_stopped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "entropy": round(self.entropy, 6),
            "approx_kl": round(self.approx_kl, 6),
            "clip_fraction": round(self.clip_fraction, 6),
            "explained_variance": round(self.explained_variance, 6),
            "grad_norm": round(self.grad_norm, 6),
            "n_updates": self.n_updates,
            "early_stopped": self.early_stopped,
        }


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer, config: PPOConfig,
               device: torch.device) -> UpdateStats:
    """一次 PPO 更新（多 epoch、小批量、按 target_kl 提前停）。"""
    advantages, returns = buffer.compute_gae(config.gamma, config.gae_lambda)
    obs = torch.tensor(buffer.obs, dtype=torch.float32, device=device)
    mask = torch.tensor(buffer.mask, dtype=torch.bool, device=device)
    action = torch.tensor(buffer.action, dtype=torch.long, device=device)
    old_log_prob = torch.tensor(buffer.log_prob, dtype=torch.float32,
                                device=device)
    advantage_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    return_t = torch.tensor(returns, dtype=torch.float32, device=device)
    old_value = torch.tensor(buffer.value, dtype=torch.float32, device=device)

    if config.normalize_advantage and len(advantage_t) > 1:
        advantage_t = ((advantage_t - advantage_t.mean())
                       / (advantage_t.std(unbiased=False) + 1e-8))

    stats = UpdateStats()
    n = len(buffer)
    indices = torch.arange(n, device=device)
    for _epoch in range(config.n_epochs):
        permutation = indices[torch.randperm(n, device=device)]
        for start in range(0, n, config.batch_size):
            batch = permutation[start:start + config.batch_size]
            evaluated = model.evaluate(obs[batch], mask[batch], action[batch])
            ratio = torch.exp(evaluated["log_prob"] - old_log_prob[batch])
            unclipped = ratio * advantage_t[batch]
            clipped = torch.clamp(ratio, 1.0 - config.clip_ratio,
                                  1.0 + config.clip_ratio) * advantage_t[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_pred = evaluated["value"]
            if config.value_clip > 0:
                value_clipped = old_value[batch] + torch.clamp(
                    value_pred - old_value[batch],
                    -config.value_clip, config.value_clip)
                value_loss = torch.max(
                    (value_pred - return_t[batch]) ** 2,
                    (value_clipped - return_t[batch]) ** 2).mean()
            else:
                value_loss = ((value_pred - return_t[batch]) ** 2).mean()

            entropy = evaluated["entropy"].mean()
            loss = (policy_loss + config.value_coef * value_loss
                    - config.entropy_coef * entropy)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((old_log_prob[batch] - evaluated["log_prob"])
                             .mean().item())
                clip_fraction = ((ratio - 1.0).abs()
                                 > config.clip_ratio).float().mean().item()
            stats.policy_loss += float(policy_loss.item())
            stats.value_loss += float(value_loss.item())
            stats.entropy += float(entropy.item())
            stats.approx_kl += float(approx_kl)
            stats.clip_fraction += float(clip_fraction)
            stats.grad_norm += float(grad_norm)
            stats.n_updates += 1
            if (config.target_kl is not None
                    and approx_kl > config.target_kl):
                stats.early_stopped = True
                break
        if stats.early_stopped:
            break

    if stats.n_updates:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "clip_fraction", "grad_norm"):
            setattr(stats, key, getattr(stats, key) / stats.n_updates)
    with torch.no_grad():
        if len(return_t) > 1:
            variance = return_t.var(unbiased=False)
            stats.explained_variance = float(
                1.0 - (return_t - old_value).var(unbiased=False)
                / (variance + 1e-8)) if float(variance) > 1e-12 else 0.0
        else:
            stats.explained_variance = 0.0
    return stats


__all__ = ["PPOConfig", "RolloutBuffer", "UpdateStats", "ppo_update"]
