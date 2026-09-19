# 大阶段二第三步：统一公平评测与结论冻结

本协议冻结七种方法的同口径比较：`round_robin`、`edf`、`rule`、
`enumeration`、`rolling_horizon`、`ppo_baseline`、
`ppo_freshness_uncertainty`。它们逐 episode 共用同一真闭环世界、
`expose_all` 候选任务集、资源预算、场景映射、通信条件、中央可见观测与
`UnifiedExecutor → RuntimeExecutor` 执行链。

## 冻结输入

- 反馈模式固定为 `plan_controlled_feedback`；
- 任务门控固定为 `expose_all`，避免手工门控替任一方法提前决定动作；
- `resource-contract-v1`、学习数据划分和场景映射均写入 SHA-256 清单；
- PPO 使用已存在的研究消融 checkpoint：`main_baseline/policy.pt` 与
  `freshness_uncertainty/policy.pt`；不训练、不改奖励、不换 checkpoint；
- test 默认拒绝访问。必须先完成 validation 报告并保存 `freeze_manifest.json`，
  才允许一次性带 `--release-test` 执行。

## 统计和解释纪律

报告六维实测向量：服务完成度、及时性、估计质量、资源消耗、通信开销、
计算耗时；并列出平均/最差完成率、等待、过期、执行器拒绝率、资源违反率及
PPO 的三项 mask 诊断。每个场景内按相同环境种子与 `rule` 做配对差异，
汇总给出均值、样本标准差与 Student-t 95% CI。

当前每个 PPO 方法只有一个冻结 checkpoint。因此训练随机性为 `n=1`，
不可估计；不得把环境种子变异写成训练随机性，也不得虚构多训练种子 CI。
这是一项结论边界，而不是允许重训或另挑种子的理由。

## 命令

```powershell
# 先验证口径；不会读取 test
D:\anaconda\envs\pytorch_env\python.exe tools/evaluate_unified_resource_methods.py --split validation

# validation 无错误、冻结清单已落盘后：唯一允许的 test 解封命令
D:\anaconda\envs\pytorch_env\python.exe tools/evaluate_unified_resource_methods.py --split test --release-test
```

每次运行输出逐 episode CSV、完整 JSON 和 HTML。没有综合总分，也不根据
test 结果回调模型选择；任何退化、未超过规则或优化参考的情况都原样保留。
