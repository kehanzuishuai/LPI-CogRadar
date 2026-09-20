# Preference-Conditioned PPO 探索分支归档（已暂停）

这是 v1→诊断→v2→可控性审计→v3 的**只读归档索引**。它保留失败证据，而不把
局部动作变化包装为偏好可控或 Pareto 成功。`test-v3`、`test-v4`、`test-v5` 都不曾在
各自未通过的机制阶段解封；尤其 `test-v5` 必须继续封存。

| 阶段 | 协议 / SHA | 训练 checkpoint | validation 证据 | 结论 |
| --- | --- | --- | --- | --- |
| v1 concat | `preference_ppo_v1.json` / `b049a997…` | 907, 911, 919 | 合法 share 候选存在，但 7,497 个合法状态的 masked argmax 为 0 | 偏好未能控制 share；test-v3 封存 |
| 机制诊断 | [share diagnosis](share_mechanism_diagnosis.md) | 读取 v1 冻结 checkpoint | 强制 share 真实发送 128 B、解锁远端 process 并改善信息年龄/协方差 | 链路与 mask 不是根因 |
| v2 concat | `preference_ppo_v2.json` / `581ea684…` | 1009, 1013, 1019 | 修正延迟归因与通信尺度后 share 开始出现，但通信/质量偏好方向错误 | 不稳定控制；test-v4 封存 |
| 可控性审计 | [audit](preference_controllability_audit.md) | 读取 v2 冻结 checkpoint | 固定状态反事实确认 reward 因果方向正确；策略 logits/argmax 不同向 | reward 正确，策略未学成映射 |
| v3 FiLM | `preference_ppo_v3.json` / `ce67ba95…` | 1103, 1109, 1117 | logits 对偏好有非零响应，但动作概率方向不能跨 seed 复现 | **策略对偏好敏感，但当前证据不支持策略可被偏好稳定控制；test-v5 封存** |

v1/v2/v3 的配置、SHA sidecar、冻结清单、checkpoint metadata、原始 validation CSV/JSON
和报告分别位于：

- `config/preference_ppo_v1.{json,sha256}` 与 `output/rl_resource/preference_ppo/`
- `config/preference_ppo_v2.{json,sha256}`、`config/preference_ppo_v2_splits.{json,sha256}` 与 `output/rl_resource/preference_ppo_v2/`
- `config/preference_ppo_v3.{json,sha256}`、`config/preference_ppo_v3_splits.{json,sha256}` 与 `output/rl_resource/preference_ppo_v3/`

`tools/verify_preference_ppo_archive.py` 和
`tests/test_preference_ppo_archive.py` 强制检查这些路径、协议 SHA、每个 checkpoint
SHA、三次负结果、`test-v5` 封存，以及基础 PPO 多训练 seed 的独立正结论没有被改写。
为支持归档迁移，若 metadata 内的历史 Windows 绝对 checkpoint 路径已不存在，校验器只会回退到
该 metadata 所在 seed 目录的 `policy.pt`，并继续以 metadata 已记录的 SHA-256 验证；它不会重写
metadata、替换 checkpoint 或改变任何负结果。

本分支的暂停**不否定**基础 PPO 资源调度结论：
`output/rl_resource/multiseed_v2/final_release.json` 仍记录在固定预算下，基础 PPO
相对 rule 的完成率优势在五训练 seed 配对 95% CI 中复现。它回答的是单目标/固定目标
资源调度的可复现性，不回答偏好条件化多目标控制是否成立。

若未来重新研究多目标/Pareto RL、层次化策略、偏好课程学习、更长训练预算或更多训练
seed，必须新建协议、全新 train/validation/test 分区与封存测试集；不得改写、覆盖或
重新解释这里的 v1–v3 负结果。
