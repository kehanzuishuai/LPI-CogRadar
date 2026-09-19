# Preference Controllability Audit（v2 只读审计）

本审计暂停 v2 后续训练，保持其负结果与 test-v4 封存。它不训练模型、不修改 v2 reward、
不读取 test-v4；只读取冻结 v2 checkpoint、train/validation 场景和预先固定的状态清单。

## 空样本语义

此前 validation 报告把“没有观测到任何航迹/协方差”的均值写为 `0.0`，这会把
**没有信息**误读成“年龄为零、协方差为零”的最好状态。现已改为：

| 情形 | `mean_information_age_s` / `mean_sigma_m` | 状态 |
| --- | --- | --- |
| 至少一个可见航迹 | 实测均值 | `observed` |
| 无可见航迹或协方差 | `null` | `not_applicable` |

这只修正报告语义，不改变 v2 reward。reward 中无可见航迹的年龄质量本来就是 `0`，
而不是 `1`，因此不存在“空集得到最佳估计质量回报”的漏洞。

## 固定状态与反事实

状态来自 train/validation，场景、seed、前置动作、目标节点、有限时域（3 tick）和
后续脚本策略 `process > share > sample > idle` 均冻结在
`config/preference_controllability_audit_v1.json`。后续脚本不读取偏好或 PPO；因此每一格
只改变偏好，或只改变首动作。

| 状态 | 覆盖 | 关键事实 |
| --- | --- | --- |
| `sample_legal` | sample 合法 | 初始采样决策 |
| `share_process_legal` | sample/process/share 合法 | outbox 2 条真实测量 |
| `stale_local_fresh_remote` | share/process 合法、远端信息价值 | A 航迹年龄 4.0 s；B outbox 2 条新测量 |
| `resource_pressure` | sample/process/share 合法、资源紧张 | B 最小剩余资源比例 0.44 |

全部原始“偏好 → 首动作 → 五项原始效用 → 折扣累计回报”矩阵见
`output/preference_controllability_audit/sensitivity_matrix.{json,csv}`。

## 奖励敏感度结果

在 `stale_local_fresh_remote` 中，相对 idle 的 3-tick 折扣回报为：

| 比较 | share / process 相对回报 |
| --- | ---: |
| 通信节约偏好下 share − idle | −0.6584 |
| 估计质量偏好下 share − idle | +0.5228 |
| 估计质量偏好下 process − idle | +0.0667 |
| 资源节约偏好下（资源紧张状态）sample − idle | −0.0627 |
| 完成度偏好下（同状态）sample − idle | +0.5000 |

这三项方向均符合协议：通信权重抑制 share；远端信息存在价值时，质量权重提高 share/
process 的相对回报；资源权重降低高成本 sample 的相对回报。因此 **v2 reward 的固定状态
因果方向正确**。

## 冻结策略响应与根因

对三个冻结 PPO、相同 `stale_local_fresh_remote` 状态、仅替换输入偏好后：

| 输入偏好 | masked share 概率均值 | share argmax 次数 |
| --- | ---: | ---: |
| estimate_quality | 0.1750 | 0 |
| communication_saving | 0.1777 | 0 |

策略不仅没有把质量偏好转成更多 share，概率方向还略微相反。因此根因判定是：

> **奖励方向已被固定状态反事实支持，但冻结 PPO 没有学会该偏好—动作映射。**

这将“奖励写反”与“优化/表示/训练未学成”明确分开。审计本身不授权创建 v3；若后续要做
v3，必须另立协议并重新冻结 train/validation/test 分区。完整 logits、masked 概率和
argmax 矩阵在 `output/preference_controllability_audit/policy_response_matrix.json`。
