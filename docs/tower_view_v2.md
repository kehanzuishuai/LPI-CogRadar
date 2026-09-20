# Tower View v2：Global Track 实验诊断台

Tower View v2 在冻结的 Global Track v1.x 和 Tower View v1 上新增只读诊断与共享模式对照。
它不会修改 Sensor、local `FusionCenter`、TrackMessage、CommBus、`GlobalTrackManager`、CI、
资源调度、PPO、关联参数或历史结果。v1 的 schema、入口和四个示例仍可独立复现。

## 启动与生成

独立入口默认使用端口 `8766`：

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view_v2.py

# 默认生成全部注册场景 × 四种共享模式
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view_v2.py --generate-only

# 只生成通信中断场景的两种航迹共享模式
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view_v2.py --generate-only `
  --scenario link_outage_loss_recovery `
  --mode track_share --mode event_triggered_track_share
```

正式示例覆盖双雷达单目标、双目标、handover/失联/重接入、新 local ID 重接、离场 drop/重入、
通信中断恢复、异步刷新和交叉目标。至少六类主验收场景均接入；额外 F/G 场景用于真实显示
`RECONNECTED` 和 `DROPPED` 生命周期。

## 页面能力

- 中央二维 ENU 态势图保留雷达、local/global track、global trail 和 local→global 映射；
- global track 使用 `tentative / confirmed / coasting / stale_coasting / handover / reconnect /
  dropped` 状态徽标、coast 轮廓和生命周期历史；
- 点击航迹可查看 active/retained sources、local track、最近更新时间、信息年龄、位置/速度、
  CI 来源/权重、fusion method 和 lifecycle；
- 事件时间轴标记并可跳转 `GLOBAL_TRACK_CREATED / ASSOCIATED / CI_FUSED / HANDOVER /
  RECONNECTED / COASTING / STALE_REJECTED / DROPPED / ID_SWITCH / FRAGMENTATION`；
- 点击事件显示同一 message 的 `local track → generated → sent → arrived → gate → associated → CI`
  证据链和 reject reason；
- 通信/融合面板逐帧显示 sent/arrived/used/rejected、累计字节、message utilization、local/global
  数、单源/多源数、CI 次数、active radar/source；
- 顶部以同场景、同 seed 切换 `no_share / measurement_share / track_share /
  event_triggered_track_share`，展示 coverage、RMSE、ID switch、fragmentation、duplicate、信息年龄、
  通信字节和利用率。它只是切换冻结 JSON，不会重新运行或控制仿真；
- 可导出当前 Canvas 帧 PNG、完整 replay JSON 和当前 replay 的单文件 HTML 诊断报告。

交叉目标的 ID switch 和 fragmentation 由外层离线评测派生为无目标身份事件并原样显示；UI 不会
隐藏或平滑这些负结果。聚合质量指标只用于显示，不参与运行时关联或融合。

## `tower-view-v2` schema

顶层主要字段：

```text
schema_version / scenario / sharing_mode / coordinate_frame
read_only_contract / frames / events / message_evidence
track_lifecycle / comparison_metrics / summary
frames_sha256 / replay_sha256
```

每帧包含：

```text
time_s / radar_nodes / local_tracks / global_tracks / mappings
communication / fusion / events
```

`events` 与 `message_evidence` 由现有 GlobalTrackManager audit、CommBus log、RuntimeExecutor 结果和
local FusionCenter 快照只读派生。回放只持有 canonical JSON，不持有任何运行时对象引用。

`comparison_metrics` 中 coverage/RMSE/ID switch/fragmentation/duplicate 是最外层最近 global track
聚合评测；导出前已丢弃目标身份和真实位置。正式 replay 不包含真值字段、目标 ID、真实轨迹或离线
关联标签。

## Debug truth 隔离

只有生成时显式使用 `--debug-truth-overlay` 才会创建 `.debug.json`。即使 debug 文件位于回放目录，
默认 manifest 和 HTTP 也不会列出或加载；服务还必须显式加入：

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view_v2.py `
  --replay-dir output\tower_view_v2 --allow-debug-replays
```

允许后页面顶部持续显示 `DEBUG / GROUND TRUTH OVERLAY`。debug replay 不能作为正式报告输入。

## 确定性和只读验收

- 全部浮点复用 v1.0.1 的递归 6 位 canonicalization；相同 scenario/mode/seed 的
  `frames_sha256` 和 `replay_sha256` 跨独立 subprocess 稳定；
- formal replay 递归禁止 truth key；debug 必须双重显式 opt-in；
- CI 事件数量、mapping、lifecycle、message evidence 与原审计日志一致；
- 模式切换仅 GET 不写文件，服务前后底层验收结果一致；
- 交叉负结果、链路拒绝、coast/reconnect/drop 均来自实际链路证据；
- `resource-contract-v1`、`rm-obs-1.0`、Global Track v1.x 参数和历史实验不变。

## 文件

- `tower_view/replay_v2.py`：v2 schema、诊断事件、对照指标与 canonical replay；
- `tower_view/static_v2/`：单页诊断台；
- `tools/run_tower_view_v2.py`：生成与标准库只读 HTTP 服务；
- `examples/tower_view_v2/`：正式示例；
- `tests/test_tower_view_v2.py`：场景、事件、模式、truth 隔离、只读与确定性回归。

## 边界

v2 不是新的 association/fusion 算法，不接 PPO 控制，不引入 JPDA/MHT，不做 3D、地图瓦片或
四屏联动。模式差异只表示既有规则通信路径的代价与结果；近距离编队、系统偏差和复杂关联仍保留为
后续算法增强问题。
