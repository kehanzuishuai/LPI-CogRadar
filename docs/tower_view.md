# Tower View v1：只读全局航迹回放

Tower View v1 是冻结 Global Track v1.x 之上的独立展示层，目标只有“能用、能看、能回放”。
它不提供调度或仿真控制，也不修改 Sensor、local `FusionCenter`、CommBus、
`GlobalTrackManager`、CI、资源账本、PPO、`resource-contract-v1` 或 `rm-obs-1.0`。

## 启动

仓库已带四个正式无真值示例：双雷达单目标、双雷达双目标、handover/失联/重接入、两目标交叉。

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view.py
```

默认打开 `http://127.0.0.1:8765/`。也可以只重新生成回放而不启动页面：

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view.py --generate-only

# 只生成一个场景到指定目录
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view.py `
  --generate-only --scenario two_targets_crossing `
  --replay-dir output\tower_view
```

`--debug-truth-overlay` 是开发专用显式开关。它会生成 `.debug.json` 并在地图上以红色叉号叠加
真实轨迹；正式示例和默认命令都不开启该选项。

即使 debug JSON 已放进回放目录，默认 manifest 和 HTTP 服务也不会列出或加载它。开发者必须再
显式加入 `--allow-debug-replays`：

```powershell
D:\anaconda\envs\pytorch_env\python.exe tools\run_tower_view.py `
  --replay-dir output\tower_view --allow-debug-replays
```

允许后，页面顶部会显示红色 `DEBUG / GROUND TRUTH OVERLAY` 警示；这不是正式演示模式。

## 页面能力

- 二维 ENU 平面显示 NODE_A/NODE_B、local track、global track 和 global 轨迹线；
- local track 采用较淡节点色，虚线表示当前 local→global 归属；
- 右侧列表显示 global ID、状态、来源、active source、位置、速度、信息年龄和融合方法；
- 点击 global track 后高亮其轨迹与对应 local tracks，并显示最近 CI 权重、更新/coast、handover
  与本帧 lifecycle/association 事件；
- 播放、暂停、上一帧、下一帧、当前仿真时间和时间轴拖动；
- 交叉场景顶部明确显示冻结验收负结果：ID switch=4、fragmentation=3、
  duplicate=0.292，并回放真实出现的 4 个 global ID，不在 UI 层清洗结果。

第一版使用原生 Canvas、CSS、JavaScript 和 Python 标准库 HTTP server，不安装任何第三方前端
依赖。它不是复杂空管大屏，也不包含地图瓦片、3D、统计大屏、多页面、PPO 控制或在线交互控制。

## `tower-view-v1` schema

顶层字段为 `schema_version / scenario / coordinate_frame / frames / summary / frames_sha256`。
每一帧至少包含：

```text
time_s
radar_nodes
local_tracks
global_tracks
mappings
recent_events
```

`global_tracks` 来自同一时刻 `GlobalTrackManager` 的只读状态；`mappings` 来自
`local_to_global`；`recent_events` 只选取 manager 已产生的发送、关联、CI、handover/drop 等
审计字段。回放层不持有 manager 引用，JSON 修改也不能反向影响仿真。

正式 schema 不包含真值字段、目标真值 ID 或真实位置。交叉场景的 ID switch/fragmentation 是
已冻结外层离线评测的聚合注释，只用于诚实标注已知负结果，不会进入运行时或关联器。仅当显式
使用 `--debug-truth-overlay` 时，每帧才增加 `debug_truth_tracks`，顶层同时写出开发用途警告。

所有输出浮点数（包括 lifecycle/association 的 `distance_m`、时间外推字段、CI 权重和嵌套
handover 数据）在序列化及 `frames_sha256` 前统一 canonicalize 到 6 位小数，并将 `-0.0` 归一为
`0.0`。这样相同场景/seed 即使由不同进程运行，formal replay 的哈希也应稳定。

## 确定性与只读边界

- 相同场景/seed 重复导出得到相同 JSON 语义和 `frames_sha256`；
- 独立 subprocess 重复生成同一 formal replay 的 SHA 有回归保护；
- exporter 前后重跑同一验收场景，运行结果逐项一致；
- 页面服务只实现 GET，前端没有写入 API；关闭页面或停止服务器不影响仿真；默认拒绝 debug replay；
- 每帧 local 的 `mapped_global_track_id`、`mappings` 和 CI/association lifecycle 审计有回归一致性检查；
- 示例由冻结的 Rule、真实 RuntimeExecutor、CommBus 和 GlobalTrackManager 链路生成，不是手写动画。

## 文件

- `tower_view/replay.py`：只读回放导出与 schema；
- `tower_view/static/`：单页 UI；
- `tools/run_tower_view.py`：生成和标准库 HTTP 启动入口；
- `examples/tower_view/`：四个正式示例；
- `tests/test_tower_view.py`：只读、无真值、确定性与映射回归。

## 已知限制

交叉目标会出现标签重叠、ID switch 和 fragmentation；v1 原样展示这些问题。轨迹标签避让、复杂
关联解释、地图/3D、长回放分块和视觉优化留到 v2。近距离编队、系统偏差以及 JPDA/MHT 也不属于
本展示阶段。
