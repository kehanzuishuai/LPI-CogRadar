# Global Track / Track-to-Track Fusion v1

## 状态与边界

v1 在现有多雷达资源—感知真闭环之上新增一个**可选**中央全局航迹层；默认
`global_track_mode="off"`，旧 `measurement_share`、local `FusionCenter`、
`resource-contract-v1`、`rm-obs-1.0` 以及 PPO / Rule / Rolling 的历史语义均不变。

`track_fusion` 不是 JPDA、MHT 或新的学习调度器。它只做解释性距离门控、稳定映射、
**Covariance Intersection (CI) 保守融合**与 coast；没有真值、离线关联标签或未来消息的
读取路径。

## 旧链路审计（冻结）

反馈模式 `plan_controlled_feedback` 的旧数据流保持如下：

```text
ExecutionPlan
  └─ RuntimeExecutor（唯一副作用入口；先经 UnifiedExecutor 校验/账本）
       ├─ sample → Sensor.observe → local_pending + measurement share_outbox
       ├─ process → local_pending + 已到达 MeasurementMessage
       │             → node-local FusionCenter.update → local tracks
       ├─ share → CommBus.publish(MeasurementMessage) → CommLink
       │             → 对端 process 才消费已到达测量
       └─ begin_tick → local FusionCenter.predict_to（未调度时只预测）

local FusionCenter → node_observation_from_fusion → CentralObservation
                 → scheduler → 下一份 ExecutionPlan
```

`CentralObservation` 仍是 `rm-obs-1.0` 的调度控制面，**没有**塞入 global track。
其作用域和历史 PPO/规则/滚动实验均被冻结。新增功能关闭时，默认运行与显式
`global_track_mode="off"` 的导出在去除墙钟计算耗时后逐项相同；计算耗时是唯一
非确定性测量，不是路径语义。

## `global-track-v1` 通信协议

`communication.TrackMessage` 是独立于 `MeasurementMessage` 的版本化消息：

| 字段 | 语义 |
| --- | --- |
| `source_node_id`, `local_track_id` | local track 的来源与稳定本地标识 |
| `message_id`, `sequence_no` | 通信去重/时序审计标识 |
| `state_timestamp_s`, `send_time_s` | 状态时刻与发送时刻，明确消息新旧 |
| `position_m`, `velocity_mps`, `covariance_position_m2` | 本地已估计状态及对角协方差 |
| `track_status`, `information_age_s` | 生命周期和信息新鲜度 |
| `source_provenance` | 可见的来源平台/传感器与更新计数 |

构造器递归拒绝 `truth*`、`true_*`、`ground_truth`、`offline_association`、关联/误差
标签等字段；无 `truth_id`、真实位置、未来状态或离线标签。

TrackMessage 与 MeasurementMessage 走同一个 `CommBus → CommLink`：同样受延迟、丢包、
过期、乱序、带宽和队列约束；全局端点只调用 `consume_kind(..., "track")`，因此只会
读取已经到达且路由给它的消息。

## 通信基线与真实记账

所有通信模式都由 `RuntimeExecutor` 在 `UnifiedExecutor` 已接受的 `share` 任务上执行，
因此没有“只统计不占资源”的旁路：

| 模式 | 实际消息 | 一次 share 的 `COMM_BYTE` | 定位 |
| --- | --- | --- | --- |
| `no_share` | 无 | 0 B | 无通信下界 |
| `measurement_share` | 1 条 128B MeasurementMessage | 128 B | 保留的原始测量共享 |
| `track_share` | 每条 pending active local track 各 1 条 128B TrackMessage | `128 × N_pending` B | v1.1 一次任务批量覆盖全部待上报航迹 |
| `event_triggered_track_share` | 每条满足冻结事件条件的 local track 各 1 条 128B TrackMessage | `128 × N_triggered` B | 可解释规则基线，非学习策略 |
| `measurement_and_track` | 至多 1 条 MeasurementMessage + 全部 pending TrackMessage | `128 × (N_pending + I_measurement)` B | 兼容模式；按实际载荷预记账 |

事件触发阈值在 `EventTriggeredTrackShareConfig` 中预声明：新航迹、位置变化 ≥250m、
协方差 trace ≥100,000m²、信息年龄 ≥3s、handover 或显式远端刷新请求。协方差/年龄
仅在本地有新测量后允许再次触发，避免“持续超阈值 = 每 tick 都上报”的伪事件。

原 `measurement_share` 没有被废弃；它是底层通信方式之一。未执行 `share` 时不会发送
TrackMessage，且每个实际发送事件必须满足
`accounted_comm_bytes == sent_comm_bytes`。

## 全局管理器

`global_fusion.GlobalTrackManager` 独立于 `fusion.FusionCenter`：

1. 对 global track 进行常速度 coast，信息年龄/协方差在失联期间增长；
2. 先使用 `(source_node_id, local_track_id)` 的既有映射，并在较宽的重接入门内恢复原 ID；
3. 否则使用距离门控关联最近 global track；无法关联才新建 `GLOBAL_TRACK_x`；
4. 保存 `local → global` 映射、来源记录、每次关联的距离、选择或拒绝原因；
5. 对不同节点来源以 CI 融合位置和位置协方差；不假设 local track 独立，不引入 JPDA/MHT。

### CI 的保守性与数值边界

对于两个位置估计，v1 使用固定 21 点权重网格，选择使位置协方差 trace 最小的
`ω ∈ [0,1]`：

```text
P_CI^-1 = ω P_A^-1 + (1-ω) P_B^-1
x_CI = P_CI [ω P_A^-1 x_A + (1-ω) P_B^-1 x_B]
```

它不是独立高斯融合，**不声称统计最优**。单节点时 global state 退化为该节点估计；
同协方差的两个来源不会被错误缩为一半方差；互补各向异性协方差可以降低整体 trace，
但每个轴都不会优于对应节点的最小方差。位置协方差对角项先检查非负，再以 `1e-6m²`
数值下界稳定求逆，报告会带 `fusion_method`、各来源权重和数值守卫。local track 目前
未提供速度协方差，因此速度用 CI 位置权重作保守加权，**不报告虚构的速度协方差收缩**。
若 TrackMessage 抵达时已晚于其 `state_timestamp_s`，中央只用报文携带的速度和既有
`process_noise_m2_per_s` 推演到当前融合时刻；不会把旧位置伪装为刚测得的位置，也不会读
未来量测。

## 塔台只读视图：`rm-obs-2.0`

`GlobalObservation` 是独立的只读快照，提供 global ID、位置/速度/位置协方差、更新时间、
信息年龄、参与节点、覆盖状态（single/overlap/handover/coasting）以及 CI 元数据：

```text
local track → TrackMessage → CommBus/CommLink → GlobalTrackManager
                                              → GlobalObservation (rm-obs-2.0)
                                              → Rule/诊断/离线评测（当前只读）
```

`CentralObservation` 仍是 `rm-obs-1.0`，PPO、奖励、动作 mask 和
`resource-contract-v1` 均没有读取或改变该塔台视图。本阶段尚未建立资源契约 v2，
因为没有把 `rm-obs-2.0` 接入任一资源调度器；未来若接入，必须另立 v2 契约、开关和
封存评测，不能覆盖 v1 实验。

开启方式（仅用于新的全局航迹实验，不可与历史结果混比）：

```python
from global_fusion import GLOBAL_TRACK_MODE_TRACK_FUSION
from resource_management.closed_loop import RUNTIME_MODE_FEEDBACK, run_closed_loop
from resource_management.scheduling import SchedulerPolicy

result = run_closed_loop(
    SchedulerPolicy.RULE, runtime_mode=RUNTIME_MODE_FEEDBACK,
    global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
)
report = result.global_track_report
tower = result.global_observation_report  # 独立 rm-obs-2.0，只读
```

## 固定 development 对照（非正式统计结论）

```powershell
python evaluate_global_tracking.py --out-dir output/global_tracking_development
```

该命令固定 Rule、seeds `41/73/109`、18 ticks，输出 CSV/JSON/HTML，绝不训练 RL。
它报告 coverage、ID switch、fragmentation、duplicate global tracks、离线 RMSE、协方差/
信息年龄、通信字节、规划/墙钟耗时和消息利用率；没有单一综合分。

本轮开发结果必须如实保留：`event_triggered_track_share` 为 3840B，低于周期
`track_share` 的 4096B（少 256B），但二者 global coverage 都只有 `0.111`、RMSE
约 `1094.54m`。这只能说明事件规则在该小场景减少了少量上报，**不能**说明其保持了
主要融合收益或改善了连续性。所有 12 个开发格资源守恒、真值载荷违规为零。

## 全链路漏斗诊断（只读）

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools/diagnose_global_track_pipeline.py `
  --out-dir output/global_track_pipeline_diagnosis
```

工具输出 `events.csv`、`summary.csv`、`lifecycle.csv`、JSON 和 Markdown，逐条记录
`local_track_created → generated/sent/arrived → global_gate → associated → ci_fused →
maintained/dropped`。运行时记录只来自 local FusionCenter 前后快照、RuntimeExecutor、
CommBus 和 GlobalTrackManager；真值只在最外层聚合 coverage/RMSE，绝不写入 TrackMessage、
manager、GlobalObservation 或调度器。

在同一冻结的三 development seeds（Rule、18 ticks）中，两个实际 TrackMessage 模式均显示：

| 模式 | local 创建 → 唯一上报 | 发送/到达/关联/CI | 最终活跃 CI 来源 | coverage | 解释 |
| --- | --- | --- | --- | --- | --- |
| `track_share` | 3 → 2 | 32 / 32 / 30 / 30 | 2 条均单源 | 0.111 | 首个损失是已创建 local track 未获航迹上报；无传输/关联拒绝。 |
| `event_triggered_track_share` | 5 → 4 | 30 / 30 / 28 / 28 | 2 条均单源 | 0.111 | 同样首先卡在 local→TrackMessage；最终虽保留过多节点来源，旧来源已超 age 不参与最终 CI。 |

每个模式还有 2 条消息在最后一个 tick 已到达、但没有下一次 `begin_tick` 可进入 global gate；
它们被单列为 `arrived_unconsumed_at_horizon`，不是丢包或关联拒绝。消息利用率分别为
0.9375 与 0.9333。因而当前低 coverage 的**首个可观测缺口**是 local track 覆盖到航迹上报，
而不是 CommBus 丢包、global gate 拒绝或 CI 未执行；但这只是固定小场景的诊断，不足以单独
证明最终估计误差的唯一因果来源。该段记录的是 v1 诊断时的历史行为：当时没有真正删除
global track；v1.2 已在独立稳健性阶段补齐超龄 drop 和墓碑审计，不能回写旧诊断结果。

## A–K 端到端与最终稳健性验收

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools/run_global_track_acceptance.py `
  --out-dir output/global_track_acceptance

# 单独复现交接/失联/重接入场景
D:\anaconda\envs\pytorch_env\python.exe tools/run_global_track_acceptance.py `
  --scenario handover_disconnect_reconnect
```

`global-track-acceptance-v1.3` 保留 v1.2 的 A–H，再新增 I 链路中断/丢包恢复、J 三个分离
目标并发和 K 异步多雷达刷新。它们走真实
Sensor→local FusionCenter→RuntimeExecutor→CommBus→
GlobalTrackManager，不使用手工 TrackMessage；仅外层离线计算 coverage、ID switch、
fragmentation、duplicate 与 RMSE。输出统一
`output/global_track_acceptance/global_track_acceptance_report.md/json`，其中包括 local track、
TrackMessage 覆盖、association、global ID/source history、CI、生命周期和通信字节。

| 场景 | 基础验收 | 关键证据与结论 |
| --- | --- | --- |
| 双雷达单目标 | 通过 | 2→2 local 上报，一条 stable global ID；最终保持 NODE_A/NODE_B 双 active source，coverage=0.833。 |
| 双雷达双目标 | 通过 | 4→4 local 上报，一次 share 可批量发送两条 pending 航迹；最终形成两条 global track，且各自保持双 active source，coverage=0.833。 |
| 两目标交叉 | 仅关联压力诊断，保留负结果 | 8→8 上报后出现 4 次 ID switch、fragmentation=3、duplicate=0.292；这是简单门控的能力边界。 |
| handover/失联/重接入 | 通过 | 10 条重建 local track 均重接同一 global ID；coverage=0.615、RMSE=128.48m，最终双 active source。 |
| E 延迟 + 乱序 | 通过 | 真实 CommBus 产生 5 次乱序到达，3 条晚到旧 sequence 被拒；global ID 稳定，coverage=0.900。 |
| F 掉线重启 + 新 local ID | 通过 | NODE_A tracker 重启后由 T2 重接原 GLOBAL_TRACK_1；审计含 `source_reconnect_associated`，coverage=0.912。 |
| G 离场/drop/重入 | 通过 | GLOBAL_TRACK_1 在 t=38 超龄删除并留下墓碑；重入的 A/B-T2 建立 GLOBAL_TRACK_2，不复活旧 ID。离线 ID switch/fragmentation 各 1 是合理删除后重建。 |
| H 单节点虚假 local track | 通过 | 同源同状态时刻竞争留下拒绝候选；假航迹独立为从未 confirmed 的 GLOBAL_TRACK_2，真实 GLOBAL_TRACK_1 仍为 A/B 双源。duplicate 上升是隔离证据，不伪装成零。 |
| I 链路中断/丢包恢复 | 通过 | 本地跟踪在固定中断窗继续；中央 global track coast，active source 数由 2 降至 0；恢复后回到双源且 `GLOBAL_TRACK_1` 不变，CommBus 明确记录 `link_outage`。 |
| J 3 个分离目标并发 | 通过 | 6 条 local track 全部独立上报，形成 3 条 global track；每条恰含 A/B 两源，逐 local stream sequence 严格递增，无串线、重复 ID、ID switch 或 fragmentation。 |
| K 异步多雷达刷新 | 通过 | CI 审计逐条保存 message/state/arrival/fusion 时间、投影时长和投影前后状态；存在异步 CI 与正投影，投影公式误差为 0，旧状态未直接当作当前状态融合，global ID 稳定。 |

v1.1 修复了验收暴露的两类基础问题：每条 active local track 现在有独立
outbox/sequence/revision，一次 share 的资源成本按实际多消息字节预记账；retained source 与
active source 分开报告，持续收到新状态的两源保持 active，超龄来源才退出。handover 的旧
`coverage=0/RMSE=None` 被定位为高速目标下处理/发送滞后叠加“重建 local track 速度为 0”；
v1.1 只用同源前后已到达状态差恢复重接速度，不读 truth/未来信息、不改 CI 数学或 gate 数值。
原 v1 报告原样保存在 `global_track_acceptance_report_v1_baseline.md/json`。交叉场景现在明确
暴露复杂关联负结果，仍不引入或宣称 JPDA。

v1.2 只修复验收明确暴露的基础语义：CommBus 报文按真实 `arrived_at` 消费；同一 local stream
同时守卫 sequence 和 state timestamp 单调性；达到既有 `max_coast_s` 后 global track 真正从
active 容器移除、删除映射并留下墓碑；同一来源/同一状态时刻的并发 local ID 不能覆盖成熟来源。
零测量扫描也会经既有 sample→process 链推进 local miss/coast/drop。CI 公式、CI 网格、空间门限、
物理模型、动作空间、资源契约和 PPO 均未修改。G 因持续 62 tick，按最大服务上限预声明充足
教学预算，避免把资源耗尽误判为生命周期失败；成本单位和守恒规则不变。

A/B/D/E/F/G/H/I/J/K 基础工程闸门全部通过后，报告写入
`foundation_freeze.frozen=true`、`v1x_final_freeze=true` 和
`no_more_foundation_scenarios=true`。Global Track v1.x 基础机制最终冻结，下一主线正式进入
Tower View。C 交叉场景继续作为简单最近门控的负结果；近距离编队、系统偏差和复杂关联也留给
后续增强分支，冻结不等于 JPDA/MHT 能力。v1.1/v1.2 报告分别另存 baseline，不被 v1.3 回写。

K 的场景配置为 NODE_A 1.0s、NODE_B 2.5s 传感器刷新；现有
sample→process→share 门控使观测到的 local state cadence 均约 3s、相位错开 1s。因此本轮证据
严格支持“异步 `state_timestamp_s` 会按融合时刻外推”，不声称实际端到端报告吞吐达到配置周期。

## 回归检查

```powershell
python -m pytest tests/test_evaluate_global_tracking.py tests/test_global_track_fusion.py tests/test_runtime_feedback.py -q
python -m pytest tests/test_diagnose_global_track_pipeline.py -q
python -m pytest tests/test_global_track_acceptance.py -q
python -m pytest -q
```

专项测试覆盖：默认关闭回归、TrackMessage 真值隔离和延迟边界、跨节点门控关联、短时
失联 coast/重连 ID 恢复、CI 保守性/数值稳定、`rm-obs-2.0` 隔离、四个通信基线、
同目标/handover/过期乱序/近距双目标的确定性验收、measurement sharing 保留、
TrackMessage 账本字节守恒、无重复副作用和全局层不读取在途消息；新诊断测试还覆盖
漏斗/生命周期 CSV、CI 审计、真实乱序旧消息拒绝、global drop/reentry、同源虚假 local track
隔离和运行时真值隔离。
