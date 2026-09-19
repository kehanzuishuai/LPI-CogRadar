# 学习问题与评测协议 v1

> 状态：协议与接口校核阶段。`resource-contract-v1` 已冻结；本阶段不进行大规模训练，不打开封存测试集。

## 1. 目标与边界

本协议用于把“实现错误”与“算法能力不足”分开。只有本文的接口、精确案例、数据划分和评测闸门通过后，才能开始学习调度器的小规模训练。大规模多种子统计仍放在最后。

独立环境为 `resource_management.learning_env.ResourceSchedulingLearningEnv`。它只依赖冻结资源单位与包内的最小空间类，不导入 `engine`、`sensor`、`fusion` 或 `communication`，不把旧雷达功率环境的终止处理照搬过来。

## 2. 学习问题定义

### 2.1 时间与动作

- 一步表示固定长度 `tick_seconds` 的调度区间，不是“一次不定时长的事件”。
- 每个 tick 最多完成一个原子任务。动作 `0` 为空闲，动作 `i+1` 对应稳定的任务槽位 `i`。
- 发布时间、截止时间、自然时域都用 tick 索引表示。
- 观测显式包含 `elapsed_fraction` 和 `remaining_time_fraction`，以及三类剩余资源和任务状态，使有限时域问题保持 Markov 性。

### 2.2 终止、截断与 bootstrap

| 事件 | Gymnasium 标志 | 是否终端补计 | Bellman bootstrap | 原因字段 |
| --- | --- | --- | --- | --- |
| 全部任务已完成或过期 | `terminated=True` | 是 | 否 | `all_tasks_resolved` |
| 剩余任务均不可由剩余资源执行 | `terminated=True` | 是 | 否 | `resource_exhausted` |
| 问题定义内的有限任务时域到达 | `terminated=True` | 是 | 否 | `task_horizon` |
| 调用方施加的更短运行步数上限 | `truncated=True` | 否 | 是 | `external_step_limit` |

Bellman 目标固定为

`y_t = r_t + gamma * (1 - terminated_t) * Q_target(s_{t+1}, a*)`。

不得用 `done = terminated or truncated` 切断 bootstrap，也不得把任务内自然时域结束伪装成外部截断。`info.bootstrap_allowed` 是接口审计字段，真值必须等于 `not terminated`。

旧 `LpiPowerEnv` 现在默认 `horizon_semantics="finite_task"`。只有复现旧 checkpoint 时才可显式选择 `legacy_truncation`；该选项必须记入实验配置，不得用于新结论。

### 2.3 奖励、代价和指标

单步奖励分解为 `completion + invalid_action + expiry + waiting + terminal_supplement`，并原样写入 `info.reward_components`。只有自然终止会对尚未解决的任务加终端补计；外部截断不补计。

单步约束代价是三类资源增量消耗比例的算术平均：

`c_t = mean_u(delta_resource[t,u] / budget[u])`。

因此 `sum_t c_t` 必须与 episode 末由资源账本重算的 `resource_consumption` 一致。评测同时报告不折扣回报、折扣回报、完成率、及时完成率、三类物理资源消耗和累计代价。

## 3. 可手算精确案例

案例定义在 `resource_management/exact_cases.py`，测试不依赖随机统计。默认 `gamma=0.9`。

| 案例 | 动作 | 逐步奖励 | 不折扣 / 折扣回报 | 累计代价 | 结束 |
| --- | --- | --- | --- | --- | --- |
| `single_on_time` | `[1]` | `[2.0]` | `2.0 / 2.0` | `1/6` | 全部解决 |
| `wait_then_complete` | `[0,1]` | `[-0.25,2.0]` | `1.75 / 1.55` | `1/6` | 全部解决 |
| `horizon_unresolved` | `[0,0]` | `[-0.25,-1.75]` | `-2.0 / -1.825` | `0` | 自然时域 |
| `resource_exhaustion` | `[1]` | `[0.25]` | `0.25 / 0.25` | `1/3` | 资源耗尽 |

外部截断另有定向用例：第一步空闲后奖励只是 `-0.25`，终端补计为零，下一状态动作掩码仍可用，bootstrap 系数为 1。

## 4. 拉格朗日分支审计

历史负结果和数值记录继续保留，但其归因降级为“校核前历史结果”。审计发现，旧训练目标中奖励 critic 与代价 critic 在下一状态各自选动作，而执行策略选择 `argmax(Q_r - lambda*Q_c)`；三者不是同一个估计对象。

未来训练已改为：在同一个可行动作掩码下，先用在线双 critic 选出唯一的拉格朗日贪心动作，再让两个目标网络都评估该动作。`tests/test_lagrangian_semantics.py` 用固定网络值检查动作与掩码。本修复尚未重训，因此不宣称已改善结果，也不宣称某类方法必然失效。

## 5. 训练、验证和测试封存

唯一划分登记表是 `config/learning_splits_v1.json`，SHA-256 在 `config/learning_splits_v1.sha256`。场景名和种子在三个集合间必须两两不相交。

| 分区 | 用途 | 场景数 | 种子 |
| --- | --- | ---: | --- |
| train | 参数更新 | 4 | 101, 103, 107, 109, 113, 127, 131, 137 |
| validation | checkpoint 选择、阈值与超参选择 | 3 | 211, 223, 227 |
| test | 所有选择冻结后的一次最终报告 | 3 | 401, 409, 419, 421, 431 |

`get_split("test")` 默认抛出 `SealedTestSplitError`。只有进入最终阶段的专用评测入口才能显式传入 `release_test=True`。打开前须冻结代码提交、环境配置、算法、超参、checkpoint 选择规则和指标。

checkpoint 只看验证集：先剔除资源守恒或接口检查失败的候选，再在预先声明的代价限制内按平均折扣回报最大选择；并列时依次选平均累计代价更低、checkpoint 更早者。代价限制必须在训练 manifest 中预先写明，不得根据测试结果回调。

## 6. 环境限制与固定感知链

已知限制单独列示，不归因给调度算法：

- 融合航迹尚未完整进入原有 RL 观测链；
- 无状态测量汇聚与 `FusionCenter` 航迹仍是两条路径；
- 已知存在乱序/延迟消息、交接失败、远程传感器偏差和机动模型失配等通信与跟踪局限；
- 在学习调度器对比期间，传感器、测量模型、通信、关联、跟踪、融合和信息边界配置必须冻结为同一份摘要。

禁止在同一组实验中同时更换跟踪器/感知链和调度算法，却将收益只归结为调度算法。若必须升级感知链，应建立独立因子实验，并重做基线。

## 7. 开始训练前的命令

```powershell
python -m unittest tests.test_learning_protocol -v
conda run -n pytorch_env python -m unittest tests.test_lagrangian_semantics -v
```

两组测试与 `docs/learning_evaluation_checklist.md` 全部通过前，不得启动大规模训练。
