# 文档索引

## 资源调度与学习

- [学习环境与评测协议](learning_protocol.md)
- [统一公平评测结论](unified_evaluation_conclusions.md)
- [PPO 多训练 seed test-v2 报告](ppo_multiseed_test_v2_report.md)
- [Preference-Conditioned PPO 探索分支归档（已暂停）](preference_ppo_archive.md)
- [v1 偏好条件化报告](preference_conditioned_ppo.md)
- [v2 信用分配机制验证](preference_ppo_v2.md)
- [偏好可控性审计](preference_controllability_audit.md)
- [v3 FiLM 表示机制验证](preference_ppo_v3.md)

## 多雷达全局航迹（当前主线）

- [Global Track / Track-to-Track Fusion v1](global_track_fusion_v1.md)
- [Tower View v1：只读全局航迹回放](tower_view.md)
- [Tower View v2：Global Track 实验诊断台](tower_view_v2.md)

`global-track-v1` 是独立、默认关闭的 global track 层；它不修改本索引中的
Preference-Conditioned PPO 归档或旧 measurement-sharing 实验。该页同时定义保守 CI、
含 v1.1 多航迹上报/source 生命周期修复、v1.2 A–H 稳健性验收、v1.3 I/J/K 最终补测与
A/B/D/E/F/G/H/I/J/K 基础机制冻结、只读 `rm-obs-2.0`、四种真实通信模式、确定性验收和
固定 development 对照；v1.x 不再新增基础测试，下一主线正式进入 Tower View。当前仍没有
PPO 重训、JPDA/MHT、MARL 或联合功率控制结论。

Tower View v1 在冻结 Global Track v1.x 上提供独立 `tower-view-v1` JSON、二维 ENU 单页界面和
时间轴回放；默认正式数据无真值，只有显式 debug overlay 才允许显示真实轨迹。它是只读展示层，
不向仿真、调度或融合对象写入。

Tower View v2 保留 v1 播放能力，并新增 lifecycle、事件跳转、消息证据链、通信/融合状态、四种共享
模式切换及 PNG/JSON/HTML 导出。v2 仍是只读层；交叉场景的 ID switch/fragmentation 原样展示，
不构成 JPDA/MHT、PPO 控制或新关联结论。
