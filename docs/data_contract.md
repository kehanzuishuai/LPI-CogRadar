# LPI-CogRadar 数据契约与接口契约（v4.5 一致性验收）

本文档是**资源管理前置一致性验收**的交付物之一。它只描述**代码里实际存在**
的行为，不重复 README 的完成度叙述。凡"代码里没有接通"的地方，本文一律
标为 **未接通**，不写成能力。

验收日期：2026-09-19（承接外部复核 `LPI_CogRadar_review_20260919.md`）。

---

## 1. 能力清单（以代码为准）

图例：✅ 已接通并可用｜⚠️ 部分接通（下面注明边界）｜❌ 未接通

### 1.1 仿真与物理层

| 能力 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 雷达方程 / ROC / 干扰 / 暴露模型 | ✅ | `engine/equations.py` | 纯标准库，无第三方依赖 |
| 单雷达功率控制主循环 | ✅ | `engine/simulator.py` | `step(power_level: int)` 接收**一个标量档位** |
| 多雷达**几何** | ✅ | `engine/scene.py`、`engine/geometry.py` | 任意多雷达的实体、坐标、有向关系查询 |
| 多雷达**联合资源管理** | ❌ | — | 见 §4.1：能量/动作/回报仍是**仿真器级标量**，只有 `radars[0]` 受控 |
| 统一实体模型 | ✅ | `models/entity.py` | 唯一 ID / 三维位置速度 / 姿态 / 时间戳 / 平台归属 |

### 1.2 测量与通信层

| 能力 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 传感器可见性（距离/视场/遮挡/周期/可用性） | ✅ | `sensor/sensor.py` | 六类"没有数据"原因按固定优先级判定 |
| **动作 → 主动传感器测量** | ✅（本轮修复） | `engine/env.py::_sensor_context` + `sensor/sensor.py::effective_tx_power_w` | 见 §3.1 |
| 被动 ESM 观测 | ✅ | `sensor/sensor.py::EsmSensor` | 无距离量测，不参与定位 |
| 传感器**系统偏差**注入 | ✅ | `sensor/sensor.py` | 距离/方位/俯仰偏差、时钟偏移、噪声低估；默认全零 |
| 通信：延迟/抖动/丢包/带宽/队列/过期 | ✅ | `communication/message.py` | |
| 通信：突发丢包/中断窗口/恢复拥塞/乱序 | ✅ | `communication/message.py` | 默认全零，旧行为位级不变 |
| 航迹通信：`global-track-v1` TrackMessage | ✅（默认关闭） | `communication/message.py`、`global_fusion/` | 只含 local track 估计/协方差/来源；拒绝真值、离线关联与未来信息 |
| 乱序测量（OOSM）处理策略 | ✅ | `multi_target_stress/timing.py` | `drop_stale` / `reorder_buffer` / `delayed_update` |

### 1.3 融合与跟踪层

| 能力 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 测量生命周期审计 | ✅ | `fusion/lifecycle.py` | 逐测量全链路漏斗 + 关联层候选审计 |
| 关联 + 常速度卡尔曼跟踪 | ✅ | `fusion/center.py`、`fusion/kalman.py` | 马氏门限 + 最近邻贪心；**基线未改** |
| 航迹外推 / 删除 | ✅（v4.5 修复） | `fusion/center.py` | 曾因 `predict_to` 覆盖 `last_update_time` 而是死代码 |
| 逐来源残差 / 新息落盘 | ✅ | `fusion/track.py::TrackSource` | 支撑机动失配与传感器健康诊断 |
| Global Track / CI 融合 | ✅（默认关闭） | `global_fusion/manager.py` | 距离门控 + stable ID + coast + 保守 CI；非 JPDA/MHT，非统计最优声明 |
| 塔台观测 `rm-obs-2.0` | ✅（只读） | `global_fusion/observation.py` | 不替换 `CentralObservation`/`rm-obs-1.0`，尚未接入 PPO/资源调度 |
| **融合航迹 → RL 观测** | ❌ | — | 见 §4.2：53 维观测走 `fuse_measurements`，**不是** `FusionCenter` |

### 1.4 认知与学习层

| 能力 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 结构化证据诊断（功率/能量/暴露） | ✅ | `ai/context.py`、`ai/rule_provider.py` | |
| 测量/通信/融合/协同四节证据 | ✅ | `ai/context.py` | 51 个发现码 |
| 系统级压力证据（机动/交接/时序/健康） | ✅ | `ai/system_stress.py` | |
| **在线快照与离线快照分离** | ✅（本轮新增） | `ai/snapshot_boundary.py` | 见 §3.2 |
| 远程 LLM 证据校验与回退 | ✅ | `ai/evidence_check.py` | 只拦"编造发现码/不可溯源数字"，**非语义级** |
| DQN / 历史窗口 / 集成 / 拉格朗日 | ✅ | `rl/` | 语义审计见 §4.4 |
| Preference-Conditioned PPO 探索分支 | 🟡 已完成探索，机制验证未通过，当前暂停 | `rl_resource/`、`config/preference_ppo_v*.json` | v1→诊断→v2→可控性审计→v3 已归档；不否定多训练 seed 基础 PPO 资源调度基线；`test-v5` 继续封存，见 `docs/preference_ppo_archive.md` |
| 基础 PPO 资源调度基线（多训练 seed） | ✅ | `output/rl_resource/multiseed_v2/` | 固定预算下相对 rule 的完成率优势在五训练 seed 配对 CI 中复现；不应与已暂停的偏好条件化探索混写 |

### 1.5 评测与压力测试

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 8 类压力场景（4 核心 + 4 系统级） | ✅ | `multi_target_stress/` |
| 观测模式三组对照 | ✅ | `evaluate_observation_modes.py` |
| 协同感知三场景对照 | ✅ | `evaluate_cooperative_sensing.py` |
| 运行隔离与产物清单 | ✅（本轮新增） | `run_manifest.py`，见 §3.3 |

---

## 2. 数据流图（代码级）

```
                       ┌──────────────────────────────────────────┐
   动作 a_t            │  LpiPowerEnv.step(action)                │
  ────────────────────▶│   1. executed = clip(action)             │
                       │   2. sim.step(executed)                  │
                       │        └─ 主仿真按 executed 记账：        │
                       │           能量/暴露/Pd/Pint（radars[0]）  │
                       │   3. _build_base_observation()           │
                       │        └─ 此时 previous_power_level      │
                       │           == executed（同一时刻、同一动作）│
                       └───────────────┬──────────────────────────┘
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
  ┌────────────────────────┐                    ┌────────────────────────────┐
  │ 测量层 _build_measurement_observation        │ 自状态（非测量，精确已知） │
  │  _sensor_context():    │                    │  own_previous_power_norm   │
  │   受控雷达 → tx_power_w=executed  ← §3.1     │  own_remaining_energy_norm │
  │   所有雷达 → interference_w                  │  own_time_norm             │
  │                        │                    │  own_required_pd           │
  │  SensorSuite.observe() │                    └────────────┬───────────────┘
  │   ├ 可见性 → 六类缺失原因│                                 │
  │   ├ 检测 Pd(执行功率)   │                                 │
  │   ├ 量测 + 协方差       │                                 │
  │   └ 系统偏差注入        │                                 │
  └───────┬────────────────┘                                 │
          │ MeasurementRecord（含 truth_* 仅评测通道）          │
          ├──────────────────────────────┐                   │
          ▼                              ▼                   ▼
  ┌──────────────────┐        ┌──────────────────┐  ┌────────────────────┐
  │ 通信总线 CommBus │        │ 融合中心         │  │ 观测向量（53 维）  │
  │  publish/consume │        │ FusionCenter     │  │ 逐字段上下界 ← §3.4│
  │  只投递已到达     │        │  关联+卡尔曼      │  └────────────────────┘
  └────────┬─────────┘        │  生命周期审计     │            │
           │ 已到达的共享测量   │  逐来源残差       │            ▼
           └──────────────────▶└────────┬─────────┘   ┌────────────────────┐
                                       │              │ 决策算法（RL）      │
                                       ▼              └────────────────────┘
                        ┌──────────────────────────────┐
                        │ AI 证据链                     │
                        │  online_snapshot_from_env()  │ ← 只含已知 + 已收到
                        │  snapshot_from_simulator()   │ ← 离线评测，含真值
                        │  snapshot_information_violations() 校验 ← §3.2
                        └──────────────────────────────┘
```

### 2.1 三层数据字典（谁能读什么）

| 层 | 内容 | 在线算法 | 在线 AI | 离线评测 |
| --- | --- | --- | --- | --- |
| 仿真真值 | `Scene` 实体真实位置/速度、`StepResult` 的真实 Pd/暴露/能耗 | ❌ | ❌ | ✅ |
| 传感器测量 | `MeasurementRecord` 的**测量字段** | ✅（经观测打包） | ✅ | ✅ |
| 传感器真值通道 | `MeasurementRecord.truth_*`、`Sensor.truth_of_candidate()` | ❌ | ❌ | ✅ |
| 算法可见输入 | `FusedObservation.vector`、航迹、已到达消息 | ✅ | ✅ | ✅ |

---

## 3. 接口契约

### 3.1 动作—测量一致性契约（本轮修复）

**契约**：同一时刻、同一节点、同一次动作下，主仿真的执行状态与传感器
测量报告必须来自**同一个发射功率**。

**违反情形（修复前，实测）**：同一默认配置、`realistic` 模式、`seed=42`，
分别执行档位 0 与档位 10——

| 量 | 档位 0 | 档位 10 | 结论 |
| --- | --- | --- | --- |
| 主仿真记录功率 | 0.5 W | 80 W | 主路径正确 |
| 主仿真 Pd_min | 0.002321 | 0.993015 | 主路径正确 |
| 主动传感器 `cfg.tx_power_w` | 18.0 W | 18.0 W | **未随动作变化** |
| 主动传感器测量记录 | `range=3983.6915, az=0.0847, snr=15.3011` | **完全相同** | **双账本** |

**根因**：`sensor/config.py` 构造时把 `radar.tx_power_w` 写死进
`SensorConfig`，而 `RadarSensor._detection_probability` / `_measure`
只读这个静态值；`engine/env.py::_sensor_context` 只给雷达传感器传了
`interference_w`（干扰），**没有传本步执行功率**。

**修复**：
1. `_sensor_context()` 对**受控雷达**（`mounting_id == sim.radar.radar_id`）
   的传感器注入 `tx_power_w = 本步执行功率`；非受控平台不注入。
2. `Sensor.effective_tx_power_w()` 统一取值：有注入用注入值，否则回退
   `cfg.tx_power_w`（因此不经过 env 的旧路径与旧测试**逐位不变**）。
3. `RadarSensor._measure` 的正演回波功率与**反演 RCS** 必须用同一个 Pt：
   否则 σ̂ 被 `Pt_cfg / Pt_actual` 系统性缩放（档位 0 配 18 W 配置时偏小 36 倍）。

**修复后验证**：档位 0 → 无检测（Pd≈0.002）；档位 10 → `snr=21.33 dB`、
`rcs_est=1.5 m²`。动作**确实改变**了传感器测量质量。

**时序依据**：`LpiPowerEnv.step()` 先 `sim.step(executed)`，之后才
`_build_base_observation()` 调测量层，因此此刻 `sim.previous_power_level`
就是本步执行档位——不是上一步，也不是初始化值。

### 3.2 信息边界契约（本轮新增）

**两条路径，必须显式声明来源**：

| 来源常量 | 允许内容 | 只读接口 |
| --- | --- | --- |
| `SOURCE_ONLINE` (`"online"`) | 本平台已知自状态、已到达的测量与估计、航迹、协同结构 | `ai.context.online_snapshot_from_env(env)` |
| `SOURCE_OFFLINE` (`"offline_evaluation"`) | 上述全部 + 真值目标/侦察机/干扰机清单、真实 Pd/Pint/暴露、真实虚警标签、逐原因真值计数 | `ai.context.snapshot_from_simulator(sim, ...)` |

**禁止进入在线路径的内容**（`FORBIDDEN_ONLINE_KEYS`）：
`truth_id`、`truth_*`、`is_false_alarm`、`n_false_alarms`、`in_fov`、
`is_occluded`、`matched_truth`，以及顶层真值清单
`targets` / `interceptors` / `jammers`。

**缺失原因的信息来源（`sensor.record.REASON_PROVENANCE`）**：

| 原因 | 来源 | 在线可否报确定值 |
| --- | --- | --- |
| `sensor_unavailable` | `device_known` | ✅ |
| `not_updated` | `device_known` | ✅ |
| `beyond_range` | `measurement_inferred` | ✅（标注为推断） |
| `out_of_fov` | `measurement_inferred` | ✅（标注为推断） |
| `occluded` | `measurement_inferred` | ✅（标注为推断） |
| `missed_detection` | `evaluation_only` | ❌ 没有真值就分不清"漏检"与"这里什么都没有" |

**校验接口**：`ai.snapshot_boundary.snapshot_information_violations(payload, source)`
——**递归扫描已序列化的整份快照**，不只查新增字段；
`strip_eval_only(payload)` 提供"离线快照降级为在线视图"的单向通道。

**必须显式声明未知**（`ONLINE_UNKNOWN_FIELDS`）：`target_count`、
`target_truth_ids`、`true_missing_reasons`、`true_false_alarm_labels`、
`hidden_platform_state`、`future_messages`。不确定就写未知，不编确定值。

**已知残留问题**：`full` 模式下智能体本来就"看得见真值"，此时
在线快照把 `pd_min`/`pint_eff`/`pint_inst` **如实记入 `boundary_provenance`**，
而不是假装它们是在线估计。这是标注而非修复；`full` 模式本身不是
可上线配置。

### 3.3 运行与产物契约（本轮新增）

- **run_id**：`<UTC时间戳>_<配置摘要前8位>_s<首个种子>`，例
  `20260919T030417_d68f13fa_s42`。
- **目录**：每次运行独占 `output/runs/<run_id>/`；`manifest.json` 记录
  `run_id` / `command` / `config` / `config_digest` / `source_digest` /
  `seeds` / `artifacts`（每个产物的路径、字节数、sha256）。
- **索引**：`output/runs/index.json` 追加式记录每次运行。
- **禁止覆盖**：`RunManifest.claim(path)` 在目标文件属于**另一次运行**时
  直接抛错。已实测拦截成功。
- **纪律**：结论只允许引用 manifest 登记过的产物；清单外的数字不可追溯。

### 3.4 观测空间契约（本轮修复）

- **逐字段上下界**：`engine.env.observation_bounds(features)`。
  只有显式登记在 `SIGNED_OBSERVATION_FIELDS` 里的维取 `[-1, 1]`。
- **有符号维**：`bearing_norm`、`elevation_norm`、`range_rate_norm`
  （`ideal`/`realistic` 模式下 4 槽 × 3 = **12 维**）。
- **禁止统一 `[0, 1]`**：统一裁剪会**静默抹掉**方位/俯仰/径向速度的符号，
  属信息破坏而非归一化。`full` 模式 12 维全为非负量，旧契约不变。
- **往返要求**：编码 → `clip` → 取值，符号必须存活；越界值仍要被夹到界内。
  由 `tests/test_observation_contract.py` 覆盖。

### 3.5 缺失率分母契约（本轮修复）

`meas_rate_*` 九维的**分母**不得使用真值实体数。

**修复前**：`total_outcomes = Σ len(report.outcomes)`——每个 outcome 对应
一条**真值**实体。若 3 个目标全部视场外，三个率都是 1.0，等于直接告诉
算法"有 3 个目标"，即使一个都没探到。

**修复后**：分母改为**本平台自己的感知容量**
（`本帧实际扫描的传感器数 × 跟踪槽位数`），完全由平台配置决定。

**残留问题（已标记，未修）**：九维的**分子**仍是仿真器侧的逐原因计数，
因此仍隐含"本帧有多少实体被考虑过"。彻底修法是把缺失统计移出在线观测向量、
只留在离线评测通道；这需要同时改动观测维度与已训练 checkpoint，
**留待下一阶段**（见 §5）。在那之前，不得声称观测向量与真值完全隔离。

---

## 4. 未接通路径（明确标记）

以下路径**在代码里没有接通**，不得被叙述为已完成能力。

### 4.1 多雷达联合资源管理 ❌

- `engine/simulator.py` 读入 `radars` 后取 `radars[0]` 存为 `self.radar`；
- `cumulative_energy_j` / `previous_power_level` / episode 回报都是
  **仿真器级标量**，没有逐雷达账本；
- `step(power_level: int)` 是**单标量**动作，不是逐雷达联合动作；
- `preview(pt_w)` 只作用于 `self.radar`。

**结论**：场景里有多个雷达 ≠ 多雷达资源管理已接通。

### 4.2 融合航迹未进入 RL 观测 ❌

- 观测向量来自 `sensor.fusion.fuse_measurements`（**无状态航迹表**）；
- `FusionCenter` 的航迹只喂 AI 诊断、协同评测与压力测试；
- 两条路径**没有打通**（这也是 v4.5 修跟踪器 bug 时观测模式天梯
  **逐位未变**的原因）。

### 4.3 在线 AI 与决策算法信息权限仍不等价 ⚠️

- 在线快照已去掉真值清单与评测专用统计（§3.2）；
- 但决策算法的观测向量仍含 §3.5 的残留项，且 `full` 模式下二者
  都不是"只看得见估计"。

### 4.4 学习层语义审计（原始问题记录）

- `engine/env.py`：只有**能量耗尽**才 `terminated`，任务时长到达记 `truncated`；
- `train_dqn.py`：bootstrap 只看 `terminated` → 任务终点与外部截断需要重新定义；
- `rl/lagrangian_agent.py`：奖励 critic 与代价 critic **各自独立**取下一状态最大值，
  该实现**不足以**支撑"机制正确、失败仅来自资源耦合"的归因；
- 训练种子逐 episode 递增、默认验证种子 42 → **训练/验证/测试种子划分未定义**。

以上是校核前的**原始审计需求**，本文不宣称已证明某个现象的唯一原因。现已在
`docs/learning_protocol.md` 中冻结新语义，并由
`tests/test_learning_protocol.py` 与 `tests/test_lagrangian_semantics.py` 校验。

### 4.5 多雷达资源—感知闭环（原缺口，**现已接通**）✅

**缺口原文（审计时的实际代码行为）**：

- `run_closed_loop` 每个 tick 先**无条件**调用 `suite.observe`，
  再**无条件**更新各节点 `FusionCenter`，之后才创建任务与提交计划
  → 调度**不可能**影响当 tick 的传感器或融合行为；
- `UnifiedExecutor._apply_task` 只改教学用 `NodeState.estimates` 与资源账本，
  `SAMPLE` 的产出数是按任务实体数构造的**记账数**，从未调用真实 `SensorSuite`；
- 节点航迹摘要每 tick **无条件** `publish_node_observation`；
  `SHARE` 任务只扣账，与 `CommBus` 发送**无关**。

后果很具体：**不同调度策略只改变任务与资源统计，不改变航迹质量**。
实测旧路径下轮询 / EDF / 规则三个策略的 `estimate_quality` **完全相同**。

**现已被 `plan_controlled_feedback` 模式接通**：

| 环节 | 旧路径 | 新路径（`RuntimeExecutor`） |
| --- | --- | --- |
| 传感器 | 每 tick 无条件全传感器扫描 | **只有拿到 `sample` 任务的节点**才 `Sensor.observe` 一次 |
| 融合 | 每 tick 无条件 `update` | **只有 `process` 任务**才 `FusionCenter.update`；未调度节点只 `predict_to` |
| 通信 | 每 tick 无条件发布摘要 | **只有 `share` 任务**才 `CommBus.publish` 一条真实测量（128 B），且与 `comm_byte` 账本逐字节一致 |
| 远端数据 | —— | 只从 `bus.consume(dst, now)` 取**实际到达**的测量 |
| 计费 | —— | `UnifiedExecutor` 仍是唯一记账点；`RuntimeExecutor` 只对 `APPLIED` 结果做副作，**不二次扣费** |

**默认仍是旧路径**（`runtime_mode="legacy_observation_first"`），
历史实验逐位可复现；新路径是显式 opt-in。
回归与端到端证据：`tests/test_runtime_feedback.py`、
`resource_management.acceptance.check_plan_controlled_feedback`（阶段验收第 6 项）、
`verify_v4.py` §19。

**新路径带来的新限制（必须一并说明）**：

1. **通信预算耗尽会停住节点**：`share` 无法执行时 outbox 永远非空，
   门控 `allow_sample=(not processable and not shareable)` 于是不再派采样任务
   → 节点进入纯预测。这是**资源耗尽**语义的直接后果，不是 bug，
   但它是硬门控而非软降级（真要保留本地采样需改门控设计）；
2. **估计误差整体变大**：旧路径等于"每个节点每 tick 免费获得一次测量"，
   新路径按实际任务量给测量，因此同一场景下航迹更新更少、误差更大
   （本轮 24 tick 实测：离线最近航迹误差均值由 155.68 m 变为 1000 m 量级）。
   这**不是回归**，而是把此前被隐式补上的测量量显式化了；
3. **多雷达功率控制仍未接通**（§4.1 不变）：本次接通的是**资源管理的多节点
   感知/通信/融合闭环**，`radars[0]` 的联合功率动作仍未实现。

---

## 5. 下一阶段闸门

本阶段（一致性验收）通过后方可进入多节点资源管理。进入条件：

1. 观测空间逐字段上下界与往返测试通过（✅ 本轮完成）；
2. 动作 → 主仿真 → 传感器测量三者同源（✅ 本轮完成，含实测对照）；
3. 在线/离线快照分离 + 整份快照信息边界测试（✅ 本轮完成）；
4. run_id / 配置摘要 / 源码摘要 / 产物清单 + 禁止覆盖（✅ 本轮完成）；
5. `docs/` 中记录修复前后对照与结果变化原因（✅ 见 `docs/change_report.md`）。

**遗留到下一阶段**（进入前至少要在设计上解决）：

- 缺失统计（九维分子）移出在线观测向量，或改用测量侧可算的替代量；
- `terminated` / `truncated` 与 bootstrap 的有限任务终点定义；
- 训练 / 验证 / 测试种子划分；
- 拉格朗日双 critic 的实现审计。

> 后续进展：上述四项已在 `docs/learning_protocol.md`（学习协议 v1）中冻结并校验；
> §4.5 的多雷达资源—感知闭环也已接通（阶段验收第 6 项）。
> 当前闸门是**集中式学习基线**：见 `docs/learning_evaluation_checklist.md`。

---

## 6. Global Track / 塔台式全局航迹管理（v1）

```text
local FusionCenter（节点 local track）
  → TrackMessage (global-track-v1, 无真值)
  → CommBus / CommLink（实际延迟、丢包、乱序、过期、带宽）
  → GlobalTrackManager（门控、stable GLOBAL_TRACK_x、CI、coast、审计）
  → GlobalObservation (rm-obs-2.0, 只读)
  → Rule/诊断/离线评测；当前不进入 PPO 或 CentralObservation
```

`measurement_share` 仍存在且不变；Global Track 新增 `no_share`、
`measurement_share`、`track_share`、`event_triggered_track_share` 四种可对照通信基线。
事件触发是固定阈值规则，**不是学习策略**。任何实际 TrackMessage 都经
`RuntimeExecutor` 触发并由 `UnifiedExecutor` 计入 `COMM_BYTE`，不得旁路记账。

CI 仅融合位置及位置协方差（local track 尚无速度协方差），以未知相关性下的保守
信息凸组合取代独立逆方差融合。迟到状态只凭报文速度和既有过程噪声推演至当前融合时刻；
它不是 JPDA/MHT，不宣称统计最优。`rm-obs-2.0`
目前仅只读，因此 `resource-contract-v1`、`rm-obs-1.0`、PPO、奖励和动作空间仍完整
复现。未来若要用 global track 调度，必须另立 resource-contract-v2、显式开关、训练协议
和新的封存评测集。

开发评测命令：`python evaluate_global_tracking.py --out-dir output/global_tracking_development`。
它固定 seeds 41/73/109，不做 RL 训练或显著性结论；本轮结果显示事件模式少发 256B，
但覆盖与连续性没有改善，负结果保留在输出报告与 `docs/global_track_fusion_v1.md`。

链路诊断命令：`python tools/diagnose_global_track_pipeline.py --out-dir output/global_track_pipeline_diagnosis`。
`global-track-pipeline-diagnosis-v1` 将 local track 创建、TrackMessage 生成/发送/抵达、
global gate、关联、CI 和最终 maintained/dropped 分开记录；运行时事件无真值，coverage/RMSE
只在外层离线汇总。当前固定 development 诊断显示首个缺口是 local track 未覆盖到航迹上报，
不是 CommBus 丢失、gate 拒绝或 CI 未执行；这不是对最终估计误差的充分因果归因。

稳健性验收命令：`python tools/run_global_track_acceptance.py --out-dir output/global_track_acceptance`。
`global-track-acceptance-v1.3` 使用显式、默认不启用的 closed-loop `target_specs` 和预声明事件
仅构造 A–K 验收几何；
默认 `TARGET_LIBRARY` 及历史实验不变。运行时 trace 只含 local track、TrackMessage、CI、
global ID/source history 与资源账本；coverage/ID switch/fragmentation/duplicate/RMSE 均是闭环
外层的 truth-only 离线评测。v1.1 只把发送状态细化为逐 local-track outbox/sequence/revision、
按实际多消息字节预记账，并分离 retained/active source；同源 local ID 重建使用已有 reconnect
gate 与前后已到达状态差恢复连续性。v1.2 再冻结：真实 arrival 顺序、sequence/state timestamp
双重单调守卫、超龄 global track 删除墓碑、删除映射后重入新建，以及同源同状态并发 local ID
隔离。v1.3 再增加 I 链路中断恢复、J 分离多目标并发、K 异步时间戳外推的只读证据字段；不改变
TrackMessage schema、CI 公式或门限。A/B/D/E/F/G/H/I/J/K 基础闸门已通过并最终冻结；交叉
出现 ID switch/fragmentation/duplicate，近距离编队与系统偏差也仍作为复杂关联增强问题保留。
下一主线正式进入 Tower View；这不是 JPDA/MHT 或近距/交叉关联能力声明。

### 4.8 Tower View v1 只读回放契约

`tower-view-v1` 独立于 `rm-obs-1.0`、`rm-obs-2.0` 和 `resource-contract-v1`。顶层包含
`scenario / coordinate_frame / frames / summary / frames_sha256`；每帧包含
`time_s / radar_nodes / local_tracks / global_tracks / mappings / recent_events`。所有运行时字段
只读采自 local FusionCenter 和 GlobalTrackManager 已有状态/审计，不形成动作或写回入口。

正式回放不含目标真值 ID、真实位置或离线关联标签。只有显式
`debug_truth_overlay=True` 才增加 `debug_truth_tracks`，并必须带开发用途警告。交叉场景可携带冻结
外层评测的 ID switch/fragmentation 聚合注释，用于暴露既有负结果；该注释不是运行时输入。

v1.0.1 对所有输出浮点字段递归 canonicalize 到 6 位小数（`-0.0` 归为 `0.0`），再计算
`frames_sha256`；因此 association/lifecycle 的距离、投影和 CI 权重不会因进程/平台最低位差异
改变 formal replay 哈希。debug replay 默认不进入 manifest 且 HTTP 访问被拒，只有服务显式传入
`--allow-debug-replays` 才可加载，UI 必须显示 `DEBUG / GROUND TRUTH` 警示。

### 4.9 Tower View v2 只读诊断契约

`tower-view-v2` 不替换 v1，也不属于调度观测。它增加 `sharing_mode / events /
message_evidence / track_lifecycle / comparison_metrics / communication / fusion`，来源仍只限已有
local/global 快照、RuntimeExecutor 结果、CommBus log 和 GlobalTrackManager audit。顶层
`read_only_contract.runtime_mutation=false`，服务仅提供静态 GET。

正式 v2 回放不保存目标身份、真实位置或真实轨迹。coverage/RMSE/ID switch/fragmentation/duplicate
在最外层计算为聚合显示值，计算后立即丢弃目标身份与位置，不得进入 message evidence、关联或 CI。
debug truth overlay 沿用 v1.0.1 的双重显式 opt-in 和醒目警告。v2 使用相同 6 位递归
canonicalization，并同时冻结 `frames_sha256` 与完整 `replay_sha256`。
