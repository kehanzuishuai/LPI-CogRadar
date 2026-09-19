# Preference-Conditioned PPO v3（FiLM 表示机制验证失败；test-v5 继续封存）

v3 是对 v2 的独立、**仅表示方式**实验，不是对 v2 负结果的覆盖或“救回”。v1、v2 的
失败结论保持原样。固定状态审计已经确认 v2 的五项目标 reward 对偏好方向正确；本阶段仅
检验更显式的条件化表示能否让策略学习该映射。

## 冻结边界

- [v3 协议](../config/preference_ppo_v3.json) SHA-256：
  `ce67ba95c9c31252b3a179fe69e6090a69086aa558211b0cb2413b3cda4812f9`
- [v3 train/validation/test-v5 划分](../config/preference_ppo_v3_splits.json) SHA-256：
  `f4c5c858e375ba7be173ad4b13e52fd49b2d898671743855c8bef9d4110d9fab`
- 固定 audit 配置 SHA-256：
  `c3e56b1dda7882e0aa42fc58da18f27b9b7a19af03dc62d62d61e63787dd90c6`
- `resource-contract-v1`：`380dad61d2efb15aaf6dadd532b831d1`。
- `test-v5` 新建即封存；本工具只有 `prepare`、`train --seed`、`report` 三个命令，
  没有测试入口。旧 test/test-v2/test-v3/test-v4 仅是历史记录。

不改变 v2 已校核的五维效用、平滑通信代价、到达后远端 process 的可观测延迟质量代理、
`RuntimeExecutor`、`plan_controlled_feedback`、物理模型、四种节点动作、action mask、
PPO 参数、12 update × 每 update 8 episode 的预算，或 deterministic validation
checkpoint 选择规则。训练偏好集合和每 episode 等概率伪随机抽样也与 v2 冻结定义一致。

## 唯一变量：条件化表示

严格基线是三个**原封不动**的 concat-v2 checkpoint（seed 1009/1013/1019，均 32,785
参数）：将 104 维状态和 5 维偏好直接连接后输入 `[128,128]` MLP。

v3 仍保留 104 维状态的 `[128,128]` 主干和相同 actor/critic heads，但偏好先走 `5→8`
encoder，再为每层生成 bounded FiLM scale/shift：

```
h = activation(raw_state_hidden * (1 + 0.1 tanh(scale(pref)))
               + 0.1 tanh(shift(pref)))
```

每个 FiLM 模型为 36,801 参数（比 concat 多 4,016，约 12.3%）；这不是扩大主干或
增加 PPO 容量，而是小型、显式的条件接口。训练 seed 为预先声明的 1103、1109、1117，
每个 checkpoint 仅由 validation 选择。其 SHA-256 依次为：

| seed | checkpoint SHA-256 |
| --- | --- |
| 1103 | `cc6f655577c385bd9849e175ddfc231cecb4ba6001f056ca71ef1b8287feca9f` |
| 1109 | `a34afbf8b00e036d29246ef17c63b8d216b4668831d0749ca41943c5fe919e81` |
| 1117 | `5032b68f10ed0199930c867f97fada3ce1b3fc5003e4f4b26b21dc8bc4cd7147` |

## 机制闸门结果（validation；不运行 test）

复用 audit 的固定 `stale_local_fresh_remote`、`resource_pressure`、`sample_legal`
状态。下表是质量偏好 vs 通信节约偏好在远端信息有价值状态的协同行为概率
`P(process)+P(share)`、share 概率及 raw-logit L1 有限差分。

| seed | 质量协同 P | 通信协同 P | 质量 share P | 通信 share P | logit L1 | 通过全部方向闸门 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1103 | 0.60007 | 0.60025 | 0.16835 | 0.17004 | 0.01514 | 否 |
| 1109 | 0.43558 | 0.43610 | 0.22349 | 0.22383 | 0.01961 | 否 |
| 1117 | 0.62297 | 0.62108 | 0.47710 | 0.47423 | 0.03366 | 否 |

三个 seed 的 raw-logit 有限差分均非零，说明 FiLM 输入没有被完全忽略；但这不等于
学成控制：1103/1109 的质量偏好协同和 share 概率反而略低于通信节约偏好；1117 虽在该
方向成立，却在资源节约降低 sample、完成偏好提高服务动作两项均失败。因此没有一个
跨 seed 的稳定因果响应，预注册机制闸门为 **失败**。

validation 六维汇总也只作描述性记录。例如质量偏好合并三 seed 有 54 次 share、256 B
通信，通信节约偏好有 33 次 share、156.44 B；但资源节约偏好反而有 63 次 share、298.67 B，
不能把这些局部差异解释为可控 Pareto 行为。

## 结论与边界

v3 没有证明“偏好提高 → 策略按对应方向改变动作”。可见的 logit 响应与稳定的、方向正确
的动作控制是两件不同的事；本次后者未通过。按协议：

- `test-v5` 不读取、不解封；不报告正式 test 或经验 Pareto 前沿。
- 不改 reward、通信尺度、动作、mask、PPO 参数、网络宽度，不挑 seed，也不扩大网络。
- Preference-Conditioned PPO 主线在 v3 处停止，保留 v1/v2/v3 负结果。

可复查原始文件：`output/rl_resource/preference_ppo_v3/training_freeze.json`、
`validation_raw.csv`、`policy_response_matrix.json`、`sensitivity.json` 和
`validation_report.json`。运行入口为 `tools/run_preference_ppo_v3.py`。
