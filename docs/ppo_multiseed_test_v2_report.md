# PPO 多训练 seed 复现与 test-v2 冻结报告

## 冻结与隔离

旧 `test` 仅保留为单 checkpoint 历史记录，未用于本轮的模型选择。新的
`config/learning_splits_v2.json` 在训练前冻结（SHA-256 见
`config/learning_splits_v2.sha256`）：训练仍为既有 4 个场景 × `{101,103}`，
validation 仍为既有 3 个场景 × `{211,223,227}`；test-v2 为三个新的组合压力
场景（高丢包交接、预算饥饿、延迟恢复）× `{503,509,521,523,541}`，场景名和
环境 seed 都不与 train/validation 重叠。

两个 arm 分别以训练 seed `701,703,709,719,727` 训练五次。奖励、PPO 参数、
`[128,128]` 网络、24 tick、12 updates × 8 episodes、动作空间、`expose_all`、
action mask、确定性 validation 及 checkpoint 选择规则完全不变。每个 seed 的
checkpoint 都只按 validation 的“守恒闸门 → 最大回报 → 更低代价 → 更早 update”
规则选择；十个 checkpoint 的 SHA-256 与选择 update 写在
`output/rl_resource/multiseed_v2/test_v2_freeze_manifest.json`。

在所有 checkpoint 哈希冻结后，test-v2 一次性解封。四方法为 `rule`、
`rolling_horizon`、PPO baseline、PPO freshness+uncertainty；总计 180 格。所有格
通过资源守恒、零资源违反、无重复运行时执行及无真值载荷检查。

审计说明：首次正式运行在 180 格运行完成后，报告聚合错误地尝试将规则方法
不适用的“去 mask”指标做数值配对，因而**没有写出报告或释放记录**。修复为仅对
双方均定义的指标配对后，使用未改变的冻结 checkpoint 与同一清单重新执行并写出
本报告；这是报告程序修复，不是依据 test-v2 指标调整任何模型或选择。

## 训练 seed 与环境 seed 分离统计

下表的均值/标准差/95% CI 是“每个训练 seed 先跨 15 个 test-v2 环境格求均值，
再在 5 个训练 seed 间统计”；JSON 同时保存每个训练 seed 内的 15 格环境波动。

| arm | 指标 | 训练 seed 均值 | seed 间标准差 | 95% CI |
| --- | --- | ---: | ---: | --- |
| PPO baseline | 完成率 | 0.560 | 0.030 | [0.530, 0.590] |
| PPO baseline | 估计质量 | 0.160 | 0.020 | [0.130, 0.190] |
| PPO baseline | 通信(B) | 1414.83 | 468.61 | [833.06, 1996.59] |
| PPO baseline | 去 mask 非法 argmax | 0.650 | 0.120 | [0.500, 0.800] |
| PPO freshness+uncertainty | 完成率 | 0.570 | 0.030 | [0.530, 0.600] |
| PPO freshness+uncertainty | 估计质量 | 0.180 | 0.010 | [0.170, 0.190] |
| PPO freshness+uncertainty | 通信(B) | 1522.35 | 467.72 | [941.68, 2103.01] |
| PPO freshness+uncertainty | 去 mask 非法 argmax | 0.600 | 0.070 | [0.510, 0.680] |

同场景、同环境 seed 的配对差值（PPO − 参考）再按训练 seed 聚合：

| PPO arm | 参考 | 完成率 Δ 95% CI | 估计质量 Δ 95% CI | 通信 Δ(B) 95% CI |
| --- | --- | --- | --- | --- |
| baseline | rule | +0.050 [ +0.020, +0.080 ] | −0.020 [ −0.050, +0.000 ] | +1329.49 [ +747.73, +1911.26 ] |
| baseline | rolling_horizon | +0.000 [ −0.030, +0.040 ] | −0.070 [ −0.100, −0.040 ] | −121.17 [ −702.94, +460.59 ] |
| freshness+uncertainty | rule | +0.060 [ +0.030, +0.090 ] | −0.000 [ −0.020, +0.010 ] | +1437.01 [ +856.35, +2017.68 ] |
| freshness+uncertainty | rolling_horizon | +0.010 [ −0.020, +0.050 ] | −0.050 [ −0.070, −0.040 ] | −13.65 [ −594.32, +567.01 ] |

## 三项预先声明结论

1. **PPO baseline 的完成率优势相对 rule 在这五个训练 seed 下保留**：配对完成率
   CI 为正；但相对 rolling horizon 不稳定，不能称为普遍优势。
2. **原单 checkpoint 的 freshness+uncertainty“估计质量/通信优势且完成率损失”
   不可复现**：相对 rule 的完成率反而为正，通信显著更高；相对 rolling 的估计质量
   更低。因此该旧结论降级为单 checkpoint、旧 test 场景下的历史观察。
3. **去掉 action mask 的非法 argmax 仍处于同一高量级**（0.60–0.65 的训练 seed
   均值，且 CI 很宽）。它不是策略学会约束的证据；部署继续依赖 mask。

## 边界

这不是 Pareto、MARL、JPDA 或网络结构研究；没有重挑 seed、改奖励、改 PPO 参数、
改网络规模、改动作空间、改 mask、改训练预算或在 test-v2 后作任何选择。五个训练
seed 能估计该固定预算下的初始化随机性，但不能外推到其它训练预算、环境模型或
未登记的场景族。

复现顺序：`prepare` → 十次 `train` → `freeze` → 一次 `test`，命令在
`tools/run_multiseed_v2.py` 中受阶段闸门约束。原始逐格数据为
`output/rl_resource/multiseed_v2/test_v2_raw.csv`，完整分层统计为
`output/rl_resource/multiseed_v2/test_v2_report.json`。
