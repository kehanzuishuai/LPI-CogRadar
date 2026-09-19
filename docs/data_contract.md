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
| 乱序测量（OOSM）处理策略 | ✅ | `multi_target_stress/timing.py` | `drop_stale` / `reorder_buffer` / `delayed_update` |

### 1.3 融合与跟踪层

| 能力 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 测量生命周期审计 | ✅ | `fusion/lifecycle.py` | 逐测量全链路漏斗 + 关联层候选审计 |
| 关联 + 常速度卡尔曼跟踪 | ✅ | `fusion/center.py`、`fusion/kalman.py` | 马氏门限 + 最近邻贪心；**基线未改** |
| 航迹外推 / 删除 | ✅（v4.5 修复） | `fusion/center.py` | 曾因 `predict_to` 覆盖 `last_update_time` 而是死代码 |
| 逐来源残差 / 新息落盘 | ✅ | `fusion/track.py::TrackSource` | 支撑机动失配与传感器健康诊断 |
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

### 4.4 学习层语义待审计（记录，未结论） ⚠️

- `engine/env.py`：只有**能量耗尽**才 `terminated`，任务时长到达记 `truncated`；
- `train_dqn.py`：bootstrap 只看 `terminated` → 任务终点与外部截断需要重新定义；
- `rl/lagrangian_agent.py`：奖励 critic 与代价 critic **各自独立**取下一状态最大值，
  该实现**不足以**支撑"机制正确、失败仅来自资源耦合"的归因；
- 训练种子逐 episode 递增、默认验证种子 42 → **训练/验证/测试种子划分未定义**。

以上为**审计需求**，本文不宣称已证明某个现象的唯一原因。

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
