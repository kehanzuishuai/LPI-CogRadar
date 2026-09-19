"""拉格朗日双 critic 是否与部署策略估计同一对象。"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
    from rl.dqn_agent import DQNConfig
    from rl.lagrangian_agent import LagrangianConfig, LagrangianDQNAgent
except ModuleNotFoundError:  # base 环境无 torch 时只跳过本模块，不伪造通过
    torch = None
    nn = None


if torch is not None:
    class _FixedNetwork(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, states):
            return self.values.unsqueeze(0).repeat(states.shape[0], 1)
else:
    class _FixedNetwork:  # pragma: no cover - 仅供无 torch 时完成模块导入
        pass


@unittest.skipIf(torch is None, "需要 pytorch_env")
class TestLagrangianCriticSemantics(unittest.TestCase):
    def test_target_and_execution_share_lagrangian_action(self) -> None:
        cfg = DQNConfig(
            obs_dim=1,
            n_actions=3,
            hidden_sizes=(4,),
            batch_size=1,
            buffer_capacity=4,
            learning_starts=1,
        )
        agent = LagrangianDQNAgent(
            cfg, LagrangianConfig(lambda_init=1.0), device="cpu"
        )
        # 奖励单独最大会选 0；Q_r-λQ_c 会选 1。
        agent.online_network = _FixedNetwork([10.0, 9.0, 0.0])
        agent.cost_network = _FixedNetwork([100.0, 0.0, 0.0])
        states = torch.tensor([[0.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True, True]])
        target_action = int(agent._lagrangian_greedy_actions(states, mask).item())
        executed_action = agent.select_action([0.0], greedy=True, action_mask=[True] * 3)
        self.assertEqual(target_action, 1)
        self.assertEqual(executed_action, target_action)

    def test_feasibility_mask_applies_to_shared_action(self) -> None:
        cfg = DQNConfig(obs_dim=1, n_actions=3, hidden_sizes=(4,))
        agent = LagrangianDQNAgent(
            cfg, LagrangianConfig(lambda_init=1.0), device="cpu"
        )
        agent.online_network = _FixedNetwork([0.0, 9.0, 8.0])
        agent.cost_network = _FixedNetwork([0.0, 0.0, 0.0])
        states = torch.tensor([[0.0]], dtype=torch.float32)
        mask = torch.tensor([[True, False, True]])
        self.assertEqual(
            int(agent._lagrangian_greedy_actions(states, mask).item()), 2
        )


if __name__ == "__main__":
    unittest.main()
