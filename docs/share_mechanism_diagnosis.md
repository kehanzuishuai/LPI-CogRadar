# Share 机制诊断（v1；只读 train/validation）

## 边界与可复现性

本阶段只读取 `preference_ppo_v1` 的三个既有 checkpoint（seed 907/911/919）和
train/validation 场景。没有训练、没有修改 v1 的奖励/偏好/网络/mask，也没有读取、
解封或运行 `test-v3`。输入配置和 checkpoint 的 SHA-256 写在
`output/share_mechanism_diagnosis/share_diagnosis.json` 中。

运行：

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools/diagnose_share_mechanism.py
```

## 可用性漏斗

统计范围是三个冻结模型 × 十个冻结偏好；每一步只统计真实的两个节点。
“候选”表示队列已有 share 任务；“合法”还要求 outbox 非空、节点可用且预算足够；
“策略概率”是未 mask 与施加 mask 后的 softmax 概率，argmax 则始终在合法 mask 内取。

| 分区 | 节点决策步 | share 候选 | 候选率 | 合法 / 候选 | 被 mask / 候选 | share 条件 raw 概率 | share 条件 mask 概率 | masked argmax share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 11,520 | 3,612 | 31.35% | 100.00% | 0.00% | 12.80% | 20.39% | **0 / 3,612** |
| validation | 12,960 | 3,966 | 30.60% | 97.96% | 2.04% | 12.89% | 20.46% | **0 / 3,885** |

因此结论不是“share 根本没有可选”或“mask 把它系统屏蔽”。它经常出现、几乎总是
合法，且策略给了非零概率；但在所有 7,497 个合法 share 节点状态中，三个模型、十个
偏好都没有把它排为合法动作 argmax。完整逐 checkpoint × 偏好 × 分区漏斗和动作计数
保留在原始 JSON。

## 同状态运行时反事实

反事实使用 `rm_validation_handover`、seed 211，并先执行真实的
`NODE_B sample → NODE_B process`，使 outbox 非空且 share 已经是合法候选。两分支
在这个相同状态分别选择 idle 或强制**合法** share；随后两分支按自己的合法 mask 前进。

| 项 | idle | 强制合法 share |
| --- | ---: | ---: |
| 当步真实消息 / 通信字节 | 0 / 0 B | 1 / 128 B |
| 当步五项效用加权和 | 0.466667 | 0.464583 |
| 当步通信节约贡献 | 0.200000 | 0.000000 |
| 下一 tick NODE_A process 是否合法 | 否 | 是 |
| 下一 tick NODE_A 远端测量数 | 0 | 1 |
| 下一 tick NODE_A remote updates | 0 | 1 |

这证明 share 的执行链是有效的：它真实占用 128 B、向 CommBus 发一条无真值载荷的
消息，并在到达后让接收节点获得 process 的可行性和融合更新。它不是只改“任务统计”。
当步分数略低 0.002083，来自通信节约从 0.2 降到 0；完成/及时性各增加 0.1，但资源
节约也轻微下降。下一 tick 的 0.646667 对 0.450000 是被 share 解锁的 process 路径的
回报，不能误写成 share 单独即时带来的估计收益。

## 最小受控因果案例

脚本只用真实 `RuntimeExecutor` 和标准 `ExecutionPlan`：A 在 t=1 sample、t=2
process，之后仅预测；B 在 t=3 sample，持有更新的真实测量；t=4 分别执行 share 或
idle；t=5 A 仅在消息实际到达时 process。

| t=5 的 A 状态 | 不 share | 执行 share |
| --- | ---: | ---: |
| 真实发送 / 账本通信 | 0 / 0 B | 128 / 128 B |
| A process 可用 | 否 | 是 |
| A 消费远端测量 | 0 | 1 |
| A remote updates | 0 | 1 |
| A 平均信息年龄 | 4.0 s | 3.0 s |
| A 年龄质量 `1/(1+age)` | 0.2000 | 0.2667 |
| A 平均最大位置 σ | 158.08 m | 146.66 m |

两个分支均验证资源守恒、运行时无重复任务、通信载荷无真值字段。这是一个机制证据，
不是 test 结果，也不用于重选 checkpoint。

## 冻结奖励的尺度检查与根因

`balanced_protocol_v1` 的通信节约是 `1 - min(1, Δcomm_bytes / 128)`。一次真实
share 恰好为 128 B，所以在通信节约极端偏好下，share 的**即时**通信效用必从 1 降到
0；它只有经“消息到达 → 接收方 process → 航迹改善”的延迟路径才可能获得补偿。等权
下反事实也显示通信惩罚几乎抵消 share 完成/及时性的即时奖励。冻结的估计质量又只用
已到达航迹的年龄 `1/(1+age)`，不直接奖励协方差下降；协方差改善只能间接通过更鲜的
航迹体现。

另有一个执行语义摩擦：`RuntimeExecutor` 的 share 实际发送 raw outbox 测量，但自动
建 share 候选仍要求本地 `visible` 航迹。因此一次采样后的“outbox 有数据但还没有本地
航迹”状态不能立刻 share；典型路径至少是 `sample → process → share`。它没有使 share
不可能（漏斗已否定这一点），但增加了延迟和信用分配长度。

**根因结论：** v1 的“通信偏好机制失败”不是通信链路断裂，也不是 mask 使 share 不可用；
是三个冻结 PPO 在频繁合法的 share 状态始终把其排在其他合法动作之后，同时面对通信
成本的即时负向信号、延迟的融合收益和 sample→process→share 的门控链。该结论仅定位
机制，不把它夸大为某种算法必然失效。

所以本阶段结束后仍不得开始正式 `preference_ppo_v2`，除非先提出新的、版本化的
协议（包括场景/奖励修改、重新冻结 train/validation/test-v4），并保留 v1 的失败结果。
