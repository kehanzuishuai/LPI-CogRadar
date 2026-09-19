# Balanced Resource Scheduling Mode（固定均衡锚点）

Balanced 是后续偏好条件化/Pareto 工作的固定参考点，不是“总分最优”算法。
它固定优化五个任务级效用：完成度、及时性、估计质量、资源节约、通信节约；
计算耗时只评测、不进入 PPO reward。

冻结配置见 `config/balanced_protocol_v1.json`（等权 `[0.2,0.2,0.2,0.2,0.2]`）。
资源节约使用 `1-min(1, cost_delta/0.25)`，通信节约使用
`1-min(1, comm_delta_bytes/128)`；完成和按时完成各按每 tick 最多 2 个任务截断，
估计质量为已到达航迹信息年龄的 `1/(1+age)`。上述都是预声明的固定范围，
不从 test/test-v2/test-v3 做 min-max。

三 seed（811/821/823）机制验证只在 train/validation 完成；test-v3 已预先封存，
未读取或解封。网络 `[128,128]`、PPO 参数、12 updates × 8 episodes、动作空间、
`expose_all`、mask 及 validation checkpoint 规则均沿用既有 PPO。

| validation 方法 | 完成率 | 及时性 | 估计质量 | 资源消耗 | 通信(B) | 等待(s) | 过期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rule | 0.51 | 0.78 | 0.19 | 0.43 | 0.0 | 3.03 | 44.7 |
| rolling_horizon | 0.47 | 1.00 | 0.13 | 0.24 | 3413.3 | 1.34 | 48.3 |
| Balanced PPO（3 seed 合并） | 0.27 | 0.96 | 0.17 | 0.20 | 298.7 | 0.34 | 47.7 |

Balanced 的动作组成是 idle 59%、sample 31%、process 5%、share 5%。它以较少资源、
较短等待换来明显更低完成率，属于**极端节约/及时性行为**，未形成所期望的均衡锚点。
该失败原样保留：不得更换权重、奖励边界、seed、checkpoint 或超参补救。
去 mask 非法 argmax 率为 0.49，仍依赖 action mask。

原始验证数据：`output/rl_resource/balanced/balanced_validation_raw.csv`；完整 JSON、
训练冻结清单和三个 checkpoint 哈希分别在同目录的
`balanced_validation_report.json`、`balanced_training_freeze.json` 和 `seed_*/balanced_metadata.json`。
