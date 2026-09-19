# Preference-Conditioned PPO：三 seed 机制验证

一个 Actor-Critic 接收 104 维既有可见观测和 5 维非负 simplex 偏好；主干仍为
`[128,128]`，动作、mask、PPO 参数、训练预算与 validation checkpoint 规则均未改变。
奖励严格复用 `balanced_protocol_v1` 的五项效用/固定归一化，只由偏好权重线性组合。
偏好集合、逐 episode 伪随机采样规则和三训练 seed 在
`config/preference_ppo_v1.json` 训练前冻结。

validation 上 10 个偏好 × 3 个 seed × 3 场景 × 3 环境 seed = 270 格。完成偏好
完成率为 0.35，资源节约偏好为 0.34 且资源消耗从 0.28 降到 0.24；说明至少这两条
偏好发生了行为/结果响应。但所有偏好 share 均为 0、通信开销均为 0，因此通信节约
偏好没有控制通信行为；严格机制闸门**失败**。

故 test-v3 继续封存，不冻结为正式 test 结论，也不扩大实验或挑选偏好。validation
经验 Pareto 非支配集合和被支配偏好位于
`output/rl_resource/preference_ppo/validation_report.json`；原始数据在
`validation_raw.csv`。Balanced 等权点仅是一个偏好点，既有 Balanced 失败结果不变。

后续只读机制诊断（未修改本 v1、未读取 test-v3）已证明 share 的真实通信—到达—融合
链路有效：问题不是 share 不存在或被 mask 恒屏蔽，而是它在频繁合法的状态中从未成为
masked argmax。完整可用性漏斗、同状态反事实、冻结奖励尺度与最小因果案例见
`docs/share_mechanism_diagnosis.md`。这仍不是启动 v2 的授权；任何 v2 都必须另行版本化
协议并重新封存 train/validation/test-v4。
