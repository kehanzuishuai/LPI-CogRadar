# 集中式学习资源调度基线（大阶段二 · 第一步）

> 状态：**冒烟 + 小规模验证完成**。目标是证明"能稳定学习且资源守恒"，
> **不要求**此时超过规则或优化参考。test 分区**未解封**。

## 1. 这一层要解决什么

大阶段一交付了资源管理的**问题定义**（任务、预算、执行器、冻结契约），
并补完了真闭环（`plan_controlled_feedback`）。但"提交什么任务"此前一直由
**手工门控**决定：每 tick 只派生一种可执行的任务类型。

本层把那件事交给**一个中央智能体**：

```
CentralObservation + 任务队列（算法可见信息）
        ↓  观测编码（定长）
    Actor-Critic（逐节点因素化分类头 + 合法动作 mask）
        ↓  逐节点动作 (idle/sample/process/share)^N
    标准 ExecutionPlan
        ↓
UnifiedExecutor（唯一记账点） → RuntimeExecutor（唯一副作点）
        ↓
Sensor / Fusion / CommBus（**智能体不得直接触碰**）
```

## 2. 六条硬约束（都能被机器检查）

| 约束 | 实现 | 检查 |
| --- | --- | --- |
| 不改冻结契约 | 完全不碰 `resource_contract_v1` | `verify_frozen()` 仍通过（测试） |
| 不改传感器/通信/融合/完成口径 | 世界构造复用 `closed_loop._build_world` | 旧路径逐位复现（见 §7） |
| 训练跑**真闭环** | 固定 `runtime_mode="plan_controlled_feedback"` | 测试断言 `provenance.runtime_mode` |
| 智能体不越权 | `rl_resource/{obs,actions,env}.py` 不 import `engine`/`sensor`/`fusion`/`communication` | AST 扫描 + **全 idle 动作下测量/融合/消息恒为 0**（运行时） |
| 结构化动作，不做动作表展开 | 逐节点因素化分类（4 类 × N 节点），不是 4^N 展平 | 参数量与节点数线性；测试断言形状 |
| 只用现有任务类型 | sample / process / share / idle | 测试断言计划内 `kind ∈ {SAMPLE, PROCESS, SHARE}` |

**信息边界**与规则基线**完全一致**：观测只来自 `CentralObservation` 与任务队列，
不读真值、不读未来消息、不读传感器偏差标签、不读账本内部字段。

## 3. 结构化动作空间与合法 mask

**为什么不做动作表**：N 个节点的联合动作是 4^N。N=2 时是 16，看着不大，但
节点数是场景变量、表里绝大多数条目非法或等价、展平后丢掉节点对称性。
因此采用**因素化（autoregressive）**表示：

```
a = (a_0, …, a_{N-1}),  a_i ∈ {0:idle, 1:sample, 2:process, 3:share}
log π(a|s) = Σ_i log π_i(a_i|s)          ← 一次前向，参数与 N 线性
```

| 动作 | 合法条件 |
| --- | --- |
| `idle` | **恒合法** |
| `sample` | 节点可用 + 队列里有该节点待处理采样任务 + 预算够 |
| `process` | 上述 + **真的有数据可处理**（本地缓冲非空或远端消息已到达） |
| `share` | 上述 + **outbox 非空** |

后两条刻意**不**用"队列里有任务就放行"这种偷懒判据——那会让策略选到必然
空转的动作、白烧资源（测试 `test_mask_requires_feasibility_not_just_a_queued_task` 钉住）。

⚠️ **任务派生门控必须换成 `expose_all`**。默认的 `loop_gate` 已经替策略
决定了该提交哪类任务，动作空间里就没东西可学了。`expose_all` 把三种候选
**同时**暴露给策略（仍沿用真实可行性），它改变的是**候选构成**、
不改传感器/通信/融合语义，因此两种模式的完成率**不可直接比较**。

## 4. 奖励、代价与终止（与 `docs/learning_protocol.md` 同语义）

单步奖励（原样写进 `info["reward_components"]`）：

```
completion = +1.0 × 本 tick 真正执行(APPLIED)的任务数
expiry     = −1.0 × 本 tick 过期任务数
invalid    = −0.5 × 本 tick 被执行器拒绝的任务数
waiting    = −0.25 × 超阈值仍未被服务的任务数 / 节点数
terminal   = −1.5 × 自然终止时仍未解决的任务数（**外部截断不补计**）
```

单步约束代价 `c_t = mean_{节点,单位}(Δconsumed[u] / capacity[u])`，
因此 **`Σ_t c_t` 必须等于 episode 末由账本重算的资源消耗**——这条恒等式
是"实现正确"与"算法不行"的分界线，**每个 episode 都校验**。

**终止/截断**：

| 事件 | 标志 | 终端补计 | bootstrap |
| --- | --- | --- | --- |
| 任务时域（`steps`）到达 | `terminated` | 是 | 否 |
| 剩余任务均不可由剩余资源执行 | `terminated` | 是 | 否 |
| 外部更短步数上限 | `truncated` | **否** | **是** |

`info["bootstrap_allowed"]` 恒等于 `not terminated`（测试逐 tick 校验）。
GAE 里 `terminated` 负责切断 bootstrap、`episode_end` 负责切断优势的
时间反向传播——**两者不能合并成一个 `done`**，否则截断会把上一条
episode 的优势串下来。

⚠️ **与协议表的一处刻意差别**：`all_tasks_resolved` **不作为**本环境的终止
条件。闭环任务由实时观测**逐 tick 派生**，第 1 个 tick 只会派生采样任务，
执行完队列就空了——沿用该条会让**每个 episode 都在第 1 个 tick 结束**
（实测如此）。空队列在闭环里是**瞬态**，不是"问题已解决"。

## 5. 场景目录 → 闭环机制的映射（**本轮首次实现**）

`config/learning_splits_v1.json` 的 `scenario_catalog` 在此之前**只被校验、
从未被任何环境消费**，因此"`load_multiplier` 是什么意思"此前没有定义。
`rl_resource/scenarios.py` 第一次给出可执行语义，并用
`SCENARIO_MAPPING_VERSION` + `mapping_digest()` 单独标识（**不修改**冻结登记表）。

| 目录字段 | 映射 | 说明 |
| --- | --- | --- |
| `budget_multiplier` | 逐节点容量 × 系数 | 直接 |
| `node_outage_windows` | `unavailable_windows`，**作用于远端节点 NODE_B** | 目录只有一对 `[start,end]`、未写作用于哪个节点；作用在远端才制造覆盖/交接压力 |
| `comm_delay_ticks` / `comm_drop_probability` | `SHARE_CONSTRAINED` + `base_delay_s` / `loss_prob` | 1 tick = 1 s |
| `sensor_bias_sigma_multiplier` | 远端节点偏置：`range_bias_m=150m`、`az_bias_deg=0.8m`、`noise_underreport_factor=1/(1+m)` | 实现为**来源不一致**（自报 σ 低估），不是 σ 整体缩放；只污染测量、不动真值 |
| `load_multiplier` | **截止余量 ÷ 系数** | ⚠️ 见下 |

⚠️ **`load_multiplier` 不是"任务数量倍数"**。真闭环里每节点每 tick 的候选
任务类型上限就是 3 种，**结构上做不到**按倍数增加任务（实测目标数 1→3 只让
任务总数从 54 变到 56）。本实现把它定义为**服务压力**：截止余量 ÷ m
（m>1 更难、m<1 更容易）。**不得**读成"任务多了 40%"。

## 6. 训练、checkpoint 选择与数据纪律

* **train 分区**训练（4 场景 × 8 种子）；**validation 分区**选 checkpoint
  （3 场景 × 3 种子）；**test 分区封存**——`get_split("test")` 默认抛
  `SealedTestSplitError`，训练脚本里**没有任何路径**能打开它。
* episode 来源是**可无限延伸的纯函数**（场景轮转、种子循环）。
  ⚠️ 第一版把它写成固定长度列表，跑完就没了 → 第 6 次更新起 `trans 0`、
  后面 15 次"训练"全是空转（entropy/kl 恒为 0），报出来的"改善"只是验证噪声。
* **评测用确定性动作（argmax）**。用随机采样评测会把"选哪个 checkpoint"
  变成抽奖：实测同一策略两次采样的验证回报能差 ±7，与学习信号同量级。
* **checkpoint 选择规则（先声明、后执行）**：
  ① 淘汰守恒或代价对账失败的候选（硬闸门）；② 通过者里取 validation
  平均折扣回报最大；③ 并列时比平均累计代价更小 → 更早的 update。

## 7. 实测结果（冒烟 + 小规模）

### 7.1 训练曲线（24 次更新，每次 8 episode × 24 tick = 192 转移）

| update | 训练回报 | validation 回报 | validation 资源消耗 | 熵 | KL | 守恒 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | −66.53 | −64.33 | 0.2417 | 2.363 | 0.0043 | ✅ |
| 8 | −57.02 | −4.00 | 0.3871 | 2.227 | 0.0127 | ✅ |
| 16 | −44.83 | **−2.67** | 0.5305 | 1.929 | 0.0138 | ✅ |
| 21 | **−28.77** | −14.50 | 0.4814 | 1.558 | 0.0051 | ✅ |
| 24 | −33.67 | −4.00 | 0.3871 | 1.560 | 0.0051 | ✅ |

* **训练回报 −66.5 → −33.7**、**validation 回报 −64.3 → −2.7**、
  熵 2.36 → 1.55 → **确实在学**，且每次更新资源守恒成立；
* 学习轨迹可解释：前 5 次更新先学会"少做事"（validation 资源消耗一度降到
  0.0000），随后发现**花资源把任务做完更划算**（消耗升到 0.53、回报升到 −2.7）；
* 代价对账最大误差 **2.68e-10**（必须 ~0）；
* 执行器拒绝率 **0.0000**、PLAN_FATAL **0 次**。

**本结果不构成"优于规则基线"的结论**：回报量纲与规则基线不同
（规则基线不优化这套奖励），跨方法比较需要先用同一评价向量重跑，未做。

### 7.2 动作合法性

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 带 mask 的非法动作率 | **0.0000** | 构造保证（采样只在合法集内） |
| 执行器拒绝率 | **0.0000** | 计划合法性的操作定义 |
| mask 覆盖率：idle / sample / process / share | 1.000 / 0.965 / 0.597 / 0.326 | mask 对 process/share 的约束最大 |
| **不带 mask** 的 argmax 非法率 | 1.0000 | 见下 |
| **不带 mask** 的非法动作概率质量 | 0.70 | 更公允的度量 |

⚠️ **必须说明的事实**：**mask 是承重的**。只靠 mask 训练的 PPO **没有**
学会规避非法动作——被掩掉的 logit 拿不到梯度，argmax 比的其实是未训练的
初值，所以"不带 mask 时 100% 非法"是**必然结果**，不是"策略学坏了"。
因此：**部署必须带 mask**；"非法动作率"要分成"带 mask 的 0"、"
执行器拒绝率"与"不带 mask 的反事实"三个数一起报，不能只报第一个。

## 8. 复现

```powershell
# 冒烟
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train --smoke
# 小规模（约 3–5 分钟）
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train `
    --rollout-episodes 8 --updates 24 --steps 24 --tag baseline
# 评测（默认 validation；test 需显式 --release-test）
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.evaluate `
    --policy output/rl_resource/baseline/policy.pt
# 定向测试
D:\anaconda\envs\pytorch_env\python.exe -m unittest tests.test_resource_rl -v
```

产物（`output/rl_resource/<tag>/`，已被 `.gitignore` 忽略）：
`summary.json`、`training_curve.csv`、`selection.json`、`policy.pt`
（checkpoint 内嵌场景映射摘要、奖励版本、数据划分摘要、训练/评测分区）。

## 9. 本层**没有**做到的事

1. **没有与规则/优化参考做同口径对比**：本层的回报是自定的学习信号，
   规则基线不优化它。要比较必须先在同一评价向量上重跑两边（未做）；
2. **没有跑多种子统计**：训练用 2 个种子（`101/103`）起步，
   validation 用 3 个种子；**不宣称任何统计显著性**；
3. **没有超参搜索**：PPO 超参是文献常用值，未调；
4. **策略只在 mask 下有效**（见 §7.2），没有把合法性内化进策略；
5. **没有多智能体**：只有一个中央智能体，节点之间没有独立策略；
6. **没有接 AI 解释层**：学习决策尚未接入 `ai/` 的诊断与解释；
7. **test 分区仍未解封**：本轮**没有**读取测试集，也**没有**用任何测试
   结果回调选择；
8. **没有改任何物理模型、历史实验与冻结契约**。
