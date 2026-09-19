# Preference-Conditioned PPO v2（机制验证失败；test-v4 继续封存）

## 协议与边界

v2 是独立协议，**不覆盖、不回写或重新解释** `preference_ppo_v1`。v1 配置 SHA-256
仍匹配其冻结清单；v1 的“通信偏好机制失败”保留为历史负结果。

v2 的唯一依据是 v1 诊断：share 在 7,497 个合法状态中从未是 masked argmax，但强制
合法 share 会真实发送 128 B、解锁远端 process，并改善可观测信息年龄与协方差。
因此修订只限于信用分配和门控语义，而不是更换网络、动作、mask、物理模型或使用真值。

冻结输入：

- [协议](../config/preference_ppo_v2.json) 与 SHA：`581ea684e9c2607acc9f7917bab0165efd481d269eb4283c6729bb359ffe40cf`
- [train/validation/test-v4 划分](../config/preference_ppo_v2_splits.json) 与 SHA：`6b3a337c9a7dcd584fc0e23a3f7946dceac0c62ba032a88722de06b144073e46`
- 三训练 seed：1009、1013、1019；同 v1 的 `[128,128]` Actor-Critic、PPO、预算、
  动作空间、action mask 和 validation-only checkpoint 选择。
- test-v4 场景族和环境 seed 与 train/validation 区分，且始终封存；旧 test/test-v2/
  test-v3 仅为历史记录。

## 唯二的 v2 改动

1. 当 share 的消息实际到达并被远端 `process` 消费时，估计质量维度加入仅由已到达融合
   航迹计算的延迟增益：年龄质量与协方差质量各占一半，`σ` 代理尺度固定为 200 m，增益
   仅取正值并截断至 1。没有 `truth_id`、真实误差、未来真值、离线标签或未到达消息。
   该回报出现在远端 process 时，现有 GAE 将其回传给早先的 share 决策。
2. 通信节约从 v1 的硬截断改为预声明的平滑函数
   `1 / (1 + Δcomm_bytes / 128)`：0 B 为 1、128 B 为 0.5，通信越多严格越差。
   同时 v2 允许真实 raw-measurement outbox 直接派生 share 候选，不再要求先产生本地可见
   航迹；这只缩短 `sample → share` 的无谓等待，所有副作仍只经 `RuntimeExecutor` 执行。

资源守恒、去重、真值隔离与唯一 `ExecutionPlan → RuntimeExecutor` 入口不变。

## 三 seed validation 机制验证

完成 3 seed × 10 偏好 × 3 validation 场景 × 3 环境 seed = **270** 格。三个预注册闸门
全部失败，因此本阶段按协议停止，不运行 test-v4，也不调整权重、奖励、seed、场景或超参。

| 闸门 | 结果 | 证据 |
| --- | --- | --- |
| 每个训练 seed 至少一次合法 share argmax | 失败 | seed 1009 全部偏好 share=0 |
| 通信节约偏好 share/通信不高于估计质量偏好 | 失败 | 通信节约 share=90、426.67 B；估计质量 share=0、0 B |
| handover 中估计质量偏好 share 多于通信节约 | 失败 | 0 对 36，方向相反 |

| 偏好 | share 动作 | 通信(B) | 信息年龄 / σ 语义 |
| --- | ---: | ---: | --- |
| communication_saving | 90 | 426.67 | 9.68 s / 357.22 m（observed） |
| estimate_quality | 0 | 0.00 | N/A（无可见航迹，非“0 最好”） |
| completion | 24 | 113.78 | 10.98 s / 405.02 m（observed） |
| service_quality | 6 | 28.44 | 9.50 s / 333.62 m（observed） |
| 其余六个偏好 | 0 | 0.00 | N/A（无可见航迹） |

这不是 v2 成功：虽然两个 seed 在部分偏好下开始选择 share，但跨 seed 不稳定，且通信节约
的方向反了。结果只说明原始“不选 share”的症状可被改变，**并不**说明偏好控制或多目标
调度机制已学成。

checkpoint SHA-256：1009=`793767c7224a66e717cea8ecc73a326f12296f8102425324acbd950431aac35d`；
1013=`338457a338586ce653d1819b584f34bc017716a9486712c278c6793746031def`；
1019=`778be4f7519acf1b747f9df5a17b79b2b35d8f72ca8ab8454229b3e157219315`。
原始 270 行 CSV、JSON 及冻结清单位于 `output/rl_resource/preference_ppo_v2/`。
空样本语义及 reward/策略可控性区分见 `docs/preference_controllability_audit.md`。

## 诚实性边界

- test-v4 未读取或解封；没有 test 结论、Pareto 结论或正式方法比较。
- 机制失败后不重挑 seed 或场景，不重训，也不通过再改奖励“救回”结果。
- 若未来另行探索，必须再立新协议与全新封存分区；当前 v2 不是该探索的授权。
