# 大阶段二第三步结论冻结：七方法统一公平评测

## 冻结与范围

评测输入已由
[`freeze_manifest.json`](../output/unified_resource_evaluation/freeze_manifest.json)
冻结：`resource-contract-v1 = 380dad61d2efb15aaf6dadd532b831d1`、
`plan_controlled_feedback`、`expose_all` 任务候选集、场景映射、数据划分、
两个 PPO checkpoint 及相关源码均记录 SHA-256。validation 先完成 63 格一致性
检查，随后一次性显式解封 test，完成 7 方法 × 3 测试场景 × 5 环境种子 = 105 格。

所有格子的资源守恒、运行时去重和真值隔离均通过；资源违反率和执行器拒绝率
均为 0。完整机器可读报告见
[`test_report.json`](../output/unified_resource_evaluation/test_report.json)、
[`test_per_episode.csv`](../output/unified_resource_evaluation/test_per_episode.csv)
和 [`test_report.html`](../output/unified_resource_evaluation/test_report.html)。

## test 观察（跨场景均值仅作描述，不作排名）

| 方法 | 完成度 | 及时性 | 估计质量 | 资源消耗 | 通信 B | 计算 s | 平均等待 s | 过期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 0.4392 | 0.8621 | 0.1540 | 0.5288 | 256 | 0.0037 | 3.314 | 58.6 |
| EDF | 0.5323 | 0.3314 | 0.1523 | 0.2756 | 2346.7 | 0.0035 | 2.490 | 40.1 |
| rule | 0.4980 | 0.6943 | 0.1769 | 0.4558 | 42.7 | 0.0054 | 2.626 | 46.3 |
| enumeration | 0.4980 | 0.8547 | 0.1116 | 0.2743 | 3541.3 | 0.2555 | 1.848 | 46.7 |
| rolling_horizon | 0.5351 | 0.9919 | 0.1634 | 0.3747 | 1792 | 0.2385 | 1.572 | 36.3 |
| PPO-baseline | 0.5885 | 0.8237 | 0.1787 | 0.4195 | 1732.3 | 0.0106 | 1.070 | 32.1 |
| PPO-freshness_uncertainty | 0.5179 | 0.8609 | 0.2106 | 0.4698 | 554.7 | 0.0112 | 1.028 | 42.7 |

这些均值混合了预算、时限和通信条件不同的场景，**不能**当作总冠军排序。
正式比较应读取 JSON 中“同场景、同环境种子、相对 rule 的配对差异”及其
95% CI。

## 可以说与不能说

- PPO-baseline 在三个 test 场景均比 rule 有更高完成度、更少过期、更短等待，
  但通信开销显著更高；它不是对所有六维都更好。
- PPO-freshness_uncertainty 在三个 test 场景均比 rule 有更高估计质量与更短等待，
  但完成度提升较小，且资源消耗在其中两个场景更高；不能说该分支全面优于基线。
- rolling_horizon 给出最强的及时性和较低过期，但付出约两个数量级更高的规划耗时
  与较大通信开销；仍只是优化参考，不能称为理论上界。
- PPO 的带 mask 非法率为 0 是构造保证。去 mask 后，PPO-baseline 的 argmax
  非法率为 0.5069、非法概率质量为 0.4192；组合 PPO 分别为 0.4306、0.3699。
  因此两者都**没有**学会独立满足合法性，部署必须保留 mask。

## 诚实性边界

1. 每种 PPO 当前只有一个冻结 checkpoint。训练随机性 `n=1`，不可估计；报告的
   标准差和置信区间只反映环境种子变化，不能外推为训练稳定性。
2. test 已一次性释放。之后不得用 test 结果更换模型、奖励、超参、种子、mask 或
   场景；任何新研究必须另起版本和新的封存划分。
3. 位置误差与估计质量中的真值对齐只在离线评测层使用，不进入七种方法的决策输入。
4. 本阶段没有单一综合分；结论是多维取舍，不是算法总排名。
