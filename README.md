# LPI-CogRadar

**LPI-CogRadar: An AI-Driven Cognitive Radar Power Control and Electromagnetic Adversarial Simulation Platform**

> 版本 **4.5.0** ｜ 在 v4.4（多雷达协同感知跑通）基础上新增：
> **① AI 对测量—通信—融合—航迹证据链的认知诊断**——`ai/` 层的结构化证据从
> 「功率/能量/暴露」扩展到 `measurement_state` / `communication_state` /
> `fusion_state` / `cooperation_state`，**17 个新的发现码**把「没有数据」的
> 原因（视场外/超距离/遮挡/未更新/漏检/传感器不可用）与「信息被通信吃掉」的
> 原因（丢包/过期/链路断/队列满/远端迟到）**逐项区分开**，不再笼统写成
> 「观测退化」；新增只读接口 `/api/explain_track`、`/api/explain_cooperation`；
> **远程 LLM 输出必须通过证据字段 + 发现码白名单校验**，不通过就标记
> `evidence_check_failed` 并回退本地规则 provider。
> **② 多目标压力测试：4 核心关联场景 + 4 系统级场景**——核心场景 S1–S4
> （两目标交叉 / 密集编队 / 短时遮挡 / 长时遮挡 / 目标附近虚警+异步远端+通信延迟）
> 配**关联层审计**（`association_candidate_tracks` + 门限距离 + 最终选择 + 拒绝原因）
> 与十项多目标指标；系统级场景 S5–S8 补上**机动目标模型失配**、
> **多雷达覆盖交接（handover）**、**通信时序压力**（突发丢包 / 链路中断 /
> 恢复拥塞 / 乱序到达 + 三种 OOSM 处理策略）、**多传感器系统偏差**
> （距离 / 方位 / 时钟偏移 / 噪声低估，**只污染测量、不动真值**）。
> 全部按**单雷达 / 不共享 / 理想共享 / 受限共享**（S8 为有偏共享、
> S7 为三种时序策略）对照，见 §11G。
>
> ⚠️ **本轮没有引入 JPDA / IMM / 复杂运动模型**：用户要求先把
> NN + Kalman 基线的失效模式**完整记录**下来，再决定是否升级。
> 系统级场景确实暴露了新失效（S6 交接期 ID 换号 22 次、重复航迹 44 条；
> S8 有偏共享比单雷达**更差 28.2%**），这些都已如实写入 §11G.9，
> **但"该不该升级"留给下一阶段判断**。
>
> **③ 教学仿真资源管理（大阶段一，已冻结）**——把「哪台设备、什么时刻、做什么」
> 从注释里的约定变成**可校验的实现**：统一资源单位（采样槽 / 处理操作 / 通信字节，
> 另有只作时间占用量的 `occupancy_second`）、唯一全局时钟、两阶段校验
> （`PLAN_FATAL` 整份拒绝零扣费 / `NODE_LOCAL` 只拒该节点）、逐节点账本；
> **融合结果 → 资源调度适配层**（独立 `schema_version = rm-obs-1.0`、
> 25 个字段全部标注单位/坐标系/可见范围/来源）；**非学习规则基线**
> （轮询 / EDF / 规则三策略共用同一观测·队列·执行器）；**非学习优化参考**
> （小规模完全枚举 + 带计算预算的滚动规划，只用显式预测模型、不读未来）；
> **六维评价向量**（完成度 / 及时性 / 估计质量 / 资源消耗 / 通信开销 / 计算耗时，
> **不给单一综合分**）；**冻结契约 `resource-contract-v1`**
> （摘要 `380dad61…`，改一处即校验失败）；**阶段验收 6 项全部通过**。
> 见 §11H。
>
> ⚠️ **「有前瞻的参考方法」不等于「理论上界」**：只有在一个**被完全枚举**的
> 小问题上求出的解，才可以称"该问题的精确最优"。滚动规划、束搜索一律只称
> **优化参考**，且这一条由**代码结构**保证（`exact=True` 只有一条产生路径；
> 声明文本必须逐字等于规范白名单；结构化 `claim_kind` 必须与 `exact` 一致）。
> 不得把优化参考写成"最优""上界"。
>
> ⚠️ 资源管理阶段**没有**触碰任何雷达/侦察/干扰物理公式，也**没有**改动
> `full` / `pomdp` / `ideal` / `realistic` 四条旧观测路径；旧实验逐位可复现
> （§13.1 的兼容性检查仍通过）。**没有**跨 tick 资源预留（执行器仍只支持
> 立即执行，因此每节点每 tick 最多落 1 条任务）；**没有**把共享/刷新的收益
> 接进资源约束；**没有**做统计意义上的方法比较；**没有**接学习算法
> （闸门见 §11H.8）。
>
> **④ 闭环补完 + 学习入口闸门**——
> **§11K 真闭环**：此前"多雷达资源管理"的调度结果**并未反向控制感知链**
> （未分配采样任务的雷达照常测量、节点摘要逐 tick 无条件发布、`share` 只扣账不发送），
> 因此不同策略只改变任务/资源统计、**不改变航迹质量**（实测三策略估计质量
> 完全相同：0.498522947 ×3）。现已建立唯一 `ExecutionPlan → RuntimeExecutor` 入口：
> 只有拿到 `sample` 的节点才触发传感器、未调度节点只做航迹预测（年龄与 σ 自然增长）、
> 只有执行 `share` 才真实发送并占用通信资源、融合只消费**实际到达**的测量。
> 实测新路径下 EDF 与轮询/规则在测量数、融合更新、消息数、估计质量与误差上
> **全部不同**。旧路径保留为默认值、逐位可复现。
> **§11I 学习协议 v1**：先修掉三处会让负结果无法归因的语义缺陷
> （`truncated` 伪装任务终点、拉格朗日双 critic **不是同一个估计对象**、
> 训练/验证/测试划分未定义），再冻结终止/bootstrap 语义、四个可手算精确案例、
> 独立学习环境与**封存**的测试分区。**尚未训练、未解封测试集。**
> **§11J 研究分支**：唯一可检验假设 + 四组同结构消融；小开发种子结果显示
> 组合特征把平均等待降了 0.227 s 但**最差完成率退化**，
> 因此 **当前证据不支持完整假设**（负结果如实保留）。
>
> ⚠️ **阶段验收已扩到 6 项**（§11H.8）：第 6 项"调度反向控制感知链"是**后补的**——
> 前 5 项通过时闭环其实还没成立，因此当时的"通过"并**不**意味着闭环成立。
>
> ⚠️ **跨模式数字不可直接比较**：新闭环下任务由实际运行状态派生，
> 测量量按调度给（旧路径等于每节点每 tick 免费获得一次测量），
> 因此完成率、估计质量、离线误差在两种模式之间**没有可比性**。
>
> **⑤ 大阶段二第一步：集中式学习资源调度基线（§11L）**——
> `rl_resource/` 用一个中央 Actor-Critic（PPO，逐节点因素化分类头 + 合法动作
> mask）读取算法可见信息、输出**标准 `ExecutionPlan`**，仍由
> `UnifiedExecutor` + `RuntimeExecutor` 校验执行，**智能体不得直接改 Sensor/
> Fusion/CommBus**。训练固定跑**真闭环** `plan_controlled_feedback`，
> 只做 sample/process/share/idle 四类现有任务，**不新增物理动作**；
> 严格 train 训练、validation 选 checkpoint、**test 保持封存**。
> 24 次更新实测：训练回报 −66.5→−33.7、validation 回报 −64.3→−2.7、
> 熵 2.36→1.55、**每次更新资源守恒**、代价对账误差 2.7e-10、
> 执行器拒绝率 0.0000。**没有**修改冻结契约、传感器、通信、融合与完成口径。
>
> ⚠️ **mask 是承重的**：只靠 mask 训练的 PPO 没有学会规避非法动作
> （不带 mask 的 argmax 非法率 100%——被掩掉的 logit 拿不到梯度）。
> **部署必须带 mask**，且"非法动作率"要分三个数一起报（§11L.3）。
>
> ⚠️ **本层结果不构成"优于规则基线"的结论**：回报是自定的学习信号，
> 规则基线不优化它；同口径比较**未做**。
>
> 命令见第 2 章，架构见第 3 章与 §3A，AI API 数据格式见第 9 章，自适应干扰闭环见第 8 章，
> 安全 RL 原理见第 11 章，部分可观测见第 11A 章，实体模型与坐标系见第 11B 章，
> **分层测量仿真与三层数据字典见第 11C 章**，**协同感知见第 11E 章**，
> **证据链认知诊断见第 11F 章**，**多目标压力测试见第 11G 章**，
> **教学资源管理与阶段冻结见第 11H 章**，**学习协议见第 11I 章**，
> **研究分支见第 11J 章**，**真闭环见第 11K 章**，
> **集中式学习调度基线见第 11L 章**，
> v3.1→v4.0 结论差异见第 13A 章。
>
> 三层依赖：**仿真层零依赖**（纯标准库）｜学习层需 torch ｜**AI 层也零依赖**（默认本地 provider）。
> 资源管理层同样**零依赖**（纯标准库，`resource_management/` 不 import torch）；
> **`rl_resource/` 需要 torch**（与 `rl/` 同一环境 `pytorch_env`）。
>
> ⚠️ **v4.5 的多目标压力测试顺手抓出并修掉了跟踪器的一个严重 bug**：
> `FusionCenter.predict_to()` 会把 `last_update_time` 写成 `now`，
> 而旧代码正是用它判断「本帧有没有拿到测量」，于是 `misses` 恒为 0、
> **coasting 与航迹删除全是死代码**（一条航迹连续 27 帧无观测仍是 `confirmed`）。
> 修复改变了 v4.4 的协同数字：场景 A 的相对改善从 −57.6% 变为 **−29.1%**，
> 详见 §11G.5 与 §11E.6。**引用旧数字的地方必须更新。**
>
> ⚠️ 原计划五项优先级中**只完成了 P1、P2**（部分可观测、可信认知决策）。
> P3 学习型干扰机、P4 轨迹级反事实、P5 自动压力测试与课程学习**尚未实现**，
> 详见 §13A.1 的诚实性清单。不要把它们写进论文。
>
> ⚠️ v4.1 的**多雷达只是几何层面的**：`Scene` 支持任意多雷达的几何查询，
> 但物理评估与功率控制**仍然只针对主雷达**（`radars[0]`）。
> **没有**多雷达协同决策、没有多雷达数据融合、没有雷达间干扰协调。
>
> ⚠️ v4.5 的**多目标压力测试已跑通，但 JPDA-lite 关联分支尚未实现**：
> 压力测试显示 S1/S2/S4 都**触发了预置的误关联告警阈值**
> （S2 关联准确率仅 0.733、关联歧义率 0.809；S4 假航迹率 0.474），
> 因此引入 JPDA-lite 的依据是充分的，但它**还没有做**（见 §11G.6）。
> 不得把「接口存在」或「阈值已触发」读成「能力已经验证」。
>
> ⚠️ 三层数据（真值 / 测量 / 算法可见输入）的边界见 §11C.1 —— **这是最该先看的一张表**。
> AI 上下文**只允许包含算法此刻真正拿得到的信息**，不含 `truth_id`、
> 不含真实目标位置、不含尚未到达的消息（由 `tests/test_ai_evidence.py` 钉住）。
> 资源管理层的对应边界见 §11H.3：**调度器只读 `CentralObservation`**，
> 不持有 `Simulator`，也不 import `engine` / `sensor` / `fusion`（AST 扫描钉住），
> 并有**运行时对照**证明它没有偷看未来（§11H.8 验收第 3 项）。

---

## 1. 项目定位

面向复杂电磁对抗环境的**低截获（LPI）雷达发射功率智能调控**仿真平台。
平台构建「雷达 — 目标 — 敌方侦察接收机 — 干扰源」四方对抗场景，
用简化雷达方程算探测性能、用简化侦察链路模型算被截获风险，
把发射功率 `Pt` 作为唯一动作变量，在**保证探测可靠性的同时压低被截获概率、累计暴露与能耗**。

在第三版（时序决策 + 能量硬约束 + Masked DQN）的基础上，本版把平台从
「跑实验的仿真器」向「**认知雷达研究平台**」推进：能自我诊断、能对抗自适应对手、
能解释自己的决策、能显式满足可靠性约束。

| 层 | 内容 | 依赖 |
| --- | --- | --- |
| 仿真层 | 场景建模、雷达/侦察方程、暴露模型、能量硬约束、规则基线、指标与报告 | **仅标准库** |
| 学习层 | Gymnasium 风格环境 + Masked DQN + **拉格朗日约束 DQN** | torch（`rl/` 不用 numpy） |
| 认知层 | **AI 认知诊断、可解释决策、provider 抽象、HTTP API** | **仅标准库** |

---

## 2. 快速开始

```bash
# ---------- 仿真与基线（Anaconda base 即可，零第三方依赖）----------
python main.py                            # 链路预算 → 三组规则基线 → CSV/HTML → env 演示
python diagnose_temporal_coupling.py      # 时序耦合 / 能量硬约束 / 实验一致性（6 项检验）

# ---------- 学习层（用你自己的 pytorch 环境）----------
D:\anaconda\envs\pytorch_env\python.exe train_dqn.py          # 训练 Masked DQN
D:\anaconda\envs\pytorch_env\python.exe evaluate_dqn.py       # 统一评测（6 策略对照）

# 约束 DQN 分支（安全 RL，不替换主 DQN）
D:\anaconda\envs\pytorch_env\python.exe train_dqn.py --safe-rl --cost-limit 0.07 \
    --lambda-lr 0.05 --lambda-max 1.5 \
    --episodes 4500 --gamma 0.95 --jitter --eps-decay-steps 30000 --eps-end 0.02 \
    --lr-decay-every 700 --out-dir output/rl_safe
D:\anaconda\envs\pytorch_env\python.exe evaluate_dqn.py --safe-model output/rl_safe/dqn_agent.pt

# ---------- 对手与泛化实验 ----------
D:\anaconda\envs\pytorch_env\python.exe evaluate_jammer_modes.py       # 固定干扰 vs 自适应智能干扰机
D:\anaconda\envs\pytorch_env\python.exe evaluate_multiseed.py          # 10 种子泛化评测
D:\anaconda\envs\pytorch_env\python.exe sensitivity_energy_budget.py   # 能量预算敏感性

# ---------- AI 认知诊断层 ----------
python diagnose_ai.py                     # 认知诊断 + 决策解释 + 对比 + 总结（无需 API Key）
python diagnose_ai.py --policy dqn --model output/rl/dqn_agent.pt
python diagnose_ai.py --adaptive-jammer   # 自适应干扰机场景下诊断
python ai_server.py                       # 启动 HTTP 服务（默认 127.0.0.1:8765）
python ai_server.py --provider deepseek   # 换成远程大模型（需要 DEEPSEEK_API_KEY）

# ---------- v4.0 P1：部分可观测 ----------
# 1) 训练部分可观测条件下的 DQN（默认 full，加 --observation-mode pomdp 切换）
D:\anaconda\envs\pytorch_env\python.exe train_dqn.py --observation-mode pomdp \
    --observation-preset moderate --episodes 1200 --jitter --gamma 0.95 \
    --eps-decay-steps 30000 --eps-end 0.02 --lr-decay-every 700 \
    --out-dir output/rl_pomdp_1200
# 2) 历史窗口分支（把最近 K 帧观测拼成长向量，K=4）
D:\anaconda\envs\pytorch_env\python.exe train_dqn.py --observation-mode pomdp \
    --history-len 4 --episodes 1200 --jitter --gamma 0.95 \
    --eps-decay-steps 30000 --eps-end 0.02 --lr-decay-every 700 \
    --out-dir output/rl_pomdp_hist_1200
# 3) 三组对照评测（全状态参考 / 仅观测信念 / 学习），含逐项噪声消融
D:\anaconda\envs\pytorch_env\python.exe evaluate_pomdp.py --observation-mode pomdp \
    --observation-preset moderate --no-lookahead --ablation \
    --pomdp-model output/rl_pomdp_1200/dqn_agent_best.pt \
    --pomdp-hist-model output/rl_pomdp_hist_1200/dqn_agent_best.pt \
    --out-dir output/pomdp

# ---------- v4.0 P2：不确定度感知的可信决策 ----------
# 1) 训练集成 DQN（5 成员，共享缓冲区 + 自助掩码）
D:\anaconda\envs\pytorch_env\python.exe train_ensemble.py --episodes 1200 \
    --observation-mode pomdp --observation-preset moderate --jitter --gamma 0.95 \
    --ensemble-size 5 --bootstrap-prob 0.8 --ood-warmup-steps 2000 \
    --eval-every 200 --eps-decay-steps 30000 --eps-end 0.02 --lr-decay-every 700 \
    --out-dir output/rl_ensemble
# 2) 评测自主率 / 回退率 / 错误决策率 / 高风险满足率，含阈值灵敏度与信号消融
D:\anaconda\envs\pytorch_env\python.exe evaluate_uncertainty.py \
    --ensemble-model output/rl_ensemble/ensemble_best.pt \
    --pomdp-model output/rl_pomdp_1200/dqn_agent_best.pt \
    --observation-mode pomdp --observation-preset moderate \
    --threshold-sweep --signal-ablation --out-dir output/uncertainty

# ---------- v4.1 多平台几何与场景导出 ----------
python export_scene.py --config config/multi_platform_scenario.json \
    --policy fixed --max-steps 10 --out-dir output/scene_multi
python export_scene.py \
    --pairs "RADAR_A>TGT_HIGH_FAST,RADAR_B>TGT_ESCORT"   # 只导出关心的有向对

# ---------- v4.2 分层测量仿真 ----------
# 测量级统计验证 + 记录导出（含距离剖面与更新周期扫描）
python analyze_measurements.py --mode realistic --range-profile --period-sweep
python analyze_measurements.py --mode ideal            # 理想测量对照（无噪声/无漏检）
python analyze_measurements.py --no-truth-columns      # 导出不含真值的版本

# 三组观测对照：full-truth / ideal-measurement / realistic-measurement
D:\anaconda\envs\pytorch_env\python.exe evaluate_observation_modes.py \
    --seeds 42 7 13 21 33 --measurement-stats

# 训练测量模式下的 DQN（观测 53 维，与 full 的 12 维不同，必须各自训练）
D:\anaconda\envs\pytorch_env\python.exe train_dqn.py --observation-mode realistic \
    --episodes 1200 --jitter --gamma 0.95 --eps-decay-steps 30000 --eps-end 0.02 \
    --lr-decay-every 700 --out-dir output/rl_realistic

# ---------- 单元测试与端到端验收（仿真层与 AI 层零依赖，base 环境即可跑）----------
python verify_v4.py                       # 端到端验收：19 章（模块/不变式/旧实验逐位复现/AI发现码/证据链/压力测试/资源管理/优化参考/契约冻结/真闭环/版本号/产物）
python -m unittest discover -s tests -v   # 最新回归：570 项通过、1 项条件跳过
python tests/test_ai_evidence.py -v       # v4.5：AI 证据链（无真值泄漏/原因区分/证据校验/路由接线）
python tests/test_multi_target_stress.py -v  # v4.5：多目标压力测试（关联审计/指标口径/跟踪器生命周期）
python tests/test_system_stress.py -v     # v4.5：系统级压力场景 S5–S8
python tests/test_resource_management.py -v      # 教学资源管理（两阶段校验/守恒/时钟/账本）
python tests/test_fusion_scheduling_adapter.py -v # 融合→调度适配层（断开远端后中央不得继续知道新信息）
python tests/test_resource_scheduling.py -v      # 规则基线闭环（分工差异/失败语义/四机制/只读 AI 桥）
python tests/test_resource_optimization.py -v    # 优化参考与阶段冻结（精确性靠独立暴力枚举验证）
python tests/test_runtime_feedback.py -v         # ★ 计划控制感知链：真闭环端到端（停采样→σ 上升→恢复）
python tests/test_learning_protocol.py -v        # ★ 学习协议（终止/截断/bootstrap/精确案例/数据划分封存）
python tests/test_lagrangian_semantics.py -v     # ★ 拉格朗日双 critic 是否评估同一个部署动作
python tests/test_information_research.py -v     # ★ 研究分支（特征门控/真值隔离/消息往返/规则归因）
python tests/test_pomdp_env.py            # 只跑 POMDP 环境
python tests/test_belief_policy.py        # 只跑信念桥接

# ---------- v4.5 P2：多目标压力测试（零依赖）----------
python -m multi_target_stress                      # 五场景 × 四路 × seed=42
python -m multi_target_stress --scenario S1 S2      # 只跑指定场景
python -m multi_target_stress --seeds 42 7 13      # 多种子（接口就绪；本轮只跑单种子）
# 集成/回退相关测试需要 torch
D:\anaconda\envs\pytorch_env\python.exe tests\test_uncertainty_fallback.py

# ---------- 大阶段一：教学资源管理（零依赖）----------
python -m resource_management                    # 教学演示：7 步（双节点/不同任务/空闲/重复/节点不可用/未知节点/超额请求）
python tools/compare_schedulers.py               # 三个规则基线的分工对比（4 场景 × 3 策略 × 3 种子）
python tools/compare_schedulers.py --seeds 42 --scenes base --no-write   # 快速自检
python evaluate_resource_management.py            # 规则基线 vs 优化参考：逐任务计划/拒绝原因/约束违反/耗时/六维对照/验收清单
python evaluate_resource_management.py --quick    # 少种子少场景（自检用）
python evaluate_resource_management.py --acceptance   # 只跑阶段验收 6 项
python evaluate_resource_management.py --freeze   # 重算并写出冻结契约快照（口径改动后必须走这一步）
python -m unittest tests.test_resource_scheduling tests.test_resource_optimization -q

# ---------- 真闭环与学习入口闸门（零依赖）----------
python tools/evaluate_information_research.py     # 研究分支：四组消融的机制检查（只写 output/development/，被 Git 忽略）
# 真闭环开关（默认仍是旧路径，历史实验逐位可复现）：
python -c "from resource_management.closed_loop import run_closed_loop; \
print(run_closed_loop(steps=24, runtime_mode='plan_controlled_feedback').metrics['runtime_feedback'])"

# ---------- 大阶段二第一步：集中式学习调度基线（需要 torch）----------
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train --smoke            # 冒烟（约 10 秒）
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train `
    --rollout-episodes 8 --updates 24 --steps 24 --tag baseline                 # 小规模训练（约 3–5 分钟）
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.evaluate `
    --policy output/rl_resource/baseline/policy.pt                              # 评测（默认 validation）
# 最终评测需显式解封测试分区（仅在冻结所有选择后）：
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.evaluate `
    --policy output/rl_resource/baseline/policy.pt --split test --release-test
D:\anaconda\envs\pytorch_env\python.exe -m unittest tests.test_resource_rl -v
```

> **本工程不会安装任何第三方包。** 仿真层与 AI 层零依赖；
> 学习层只用 `pytorch_env` 里已有的 torch 与 matplotlib。
> 除 `tests/test_uncertainty_fallback.py` 外的测试都不需要 torch。
> `verify_v4.py` 会打印 `↔`/`→` 等符号，脚本已在启动时把 stdout 强制为
> UTF-8（Windows 控制台默认 GBK 会在中途抛 `UnicodeEncodeError`）。

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          认知层  ai/  explain/  （零依赖）                    │
│                                                                              │
│   diagnose_ai.py ──┐                          ┌── ai_server.py               │
│                    ├─► AIDiagnosisService ────┤   POST /api/diagnose         │
│   ai/LPI_API ──────┘   异常隔离 + 自动降级     │   POST /api/explain_decision │
│                              │                │   POST /api/compare_policies │
│                    ┌─────────┴─────────┐      │   POST /api/generate_report  │
│                    ▼                   ▼      └──────────────────────────────┘
│            AIProvider 接口        结构化 JSON 协议（ai/schema.py）
│            ├── RuleProvider  ← 默认，无 Key，离线确定
│            ├── MockProvider  ← 测试 + 降级路径验证
│            └── RemoteHTTPProvider ← OpenAI/DeepSeek/通义（OpenAI 兼容）
│                                                                              │
│   explain/counterfactual.py：对候选功率做**反事实试算** → 结构化证据           │
│   ⚠️ AI 层**没有控制接口**：只读状态、只给诊断与建议；失败绝不影响主仿真        │
└───────────────────────────────▲──────────────────────────────────────────────┘
                                │ 只读 StateSnapshot（时间/距离/J-N/功率/Pd/Pint/暴露/能量/Q值
                                │   + v4.0 新增：observability 观测链路 / trust 不确定度与回退）
┌───────────────────────────────┴──────────────────────────────────────────────┐
│                        学习层  rl/  （需要 torch）                            │
│   DQNAgent（Masked DQN，主分支）        LagrangianDQNAgent（约束分支）          │
│   · Q 网络 D→64→64→11                 · 双 critic：奖励 Q_r + 代价 Q_c         │
│   · 目标网络周期同步                   · 动作 = argmax (Q_r − λ·Q_c)           │
│   · 动作掩码（选择 + bootstrap 目标）   · λ 对偶上升：λ ← λ + η(ĉ − d)          │
│                                        EnsembleDQNAgent（v4.0 集成分支）       │
│                                        · N=5 成员，共享缓冲区 + 自助掩码        │
│                                        · Q 均值/方差 → 认知不确定度            │
│                                        · 观测均值/方差（Welford）→ OOD 评分     │
│   ReplayBuffer：状态/动作/奖励/掩码/代价                                       │
│   ⚠️ D 随观测模式变化：full=12，pomdp=16，pomdp+历史窗口=16×K                  │
└───────────────────────────────▲──────────────────────────────────────────────┘
                                │ (obs, reward, terminated, truncated, info) + action_masks()
┌───────────────────────────────┴──────────────────────────────────────────────┐
│                     仿真层  engine/  models/  strategy/  metrics/（零依赖）    │
│                                                                              │
│   LpiPowerEnv（Gymnasium 风格，11 档离散功率 / 硬能量约束）                     │
│        │                                                                     │
│        │  observation_mode="full"  → 真值 12 维观测（= v3.1，逐位一致）        │
│        │  observation_mode="pomdp" → 16 维：12 维**估计值** + 4 维不确定度     │
│        │        └─ engine/observation_model.py：测量噪声/延迟/丢测/隐藏真值    │
│        ▼                                                                     │
│   Simulator ── 雷达方程 ── 侦察方程 ── 累计暴露递推 ── 能量硬约束              │
│        │            │            │            │            │                 │
│        │       models/radar  models/interceptor  models/exposure  InfeasibleActionError
│        ▼                                                                     │
│   对抗环境：Target × N ｜ EnemyInterceptor(ESM) ｜ Jammer                      │
│                                            ├── mode="fixed"    固定时间窗（默认）│
│                                            └── mode="adaptive" 规则自适应 ★闭环 │
│                                                                              │
│   闭环：雷达辐射 → ESM 累计暴露↑ → 干扰机升级压制 → Pd↓ → 雷达重新调功率        │
│                                                                              │
│   策略基线：固定 / 规则 / 随机 / 逐档贪心(短视) / 前瞻规划(非短视)              │
│   v4.0 新增：strategy/belief_policy.py   —— 让上述脚本策略只用带噪观测决策      │
│              strategy/uncertainty_policy.py —— 不确定度监测 + 安全护盾/回退    │
│   指标与报告：metrics/collector.py（CSV） + metrics/report.py（HTML/曲线）      │
│   统一日志：logging_utils.py → output/logs/<run>.log                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

**信息水平必须分组汇报**（v4.0 最重要的实验纪律）：

```
  A 组 全状态参考    固定80W / 规则 / 随机 / 短视 / 前瞻   ← 读真值，信息量最高，只作上界
  B 组 仅观测        规则(仅观测) / 短视(仅观测) / 前瞻(仅观测) ← 读带噪信念，与 DQN 同信息量
  C 组 学习          DQN(全可观) / DQN(部分可观测) / DQN(POMDP+历史窗口)
```

把 A 组和 C 组混在一张表里排名是**错的**：A 组多出「侦察机真实位置、真实 Pint、
真实累计暴露」这些 DQN 根本看不到的量。`evaluate_pomdp.py` 因此按组分别排版，
并在结论里显式写出「这是同一信息水平下的对比 / 这是上界参考」。

**五类模块都可用配置/开关独立启停**，旧实验保持逐位可复现：

| 模块 | 开关 | 默认 |
| --- | --- | --- |
| 自适应智能干扰机 | `--adaptive-jammer`（或 `Simulator.apply_overrides(adaptive_jammer=True)`） | **关**（固定时间窗） |
| AI 认知诊断层 | `--provider rule/mock/...`、`--no-ai` | **开**（本地 rule，零依赖） |
| 可解释决策 | `diagnose_ai.py`（含 `--max-moments`、`--focus-powers`） | 仅在诊断脚本中启用 |
| 安全强化学习 | `train_dqn.py --safe-rl`、`evaluate_dqn.py --safe-model` | **关**（主 DQN 不变） |
| 域随机化（泛化） | `train_dqn.py --jitter`、`evaluate_multiseed.py --no-jitter` | 训练开、评测开 |
| **部分可观测（POMDP）** | `--observation-mode full\|pomdp` + `--observation-preset mild\|moderate\|severe` | **`full`**（= v3.1 行为） |
| **历史窗口编码** | `--history-len K`（K>1 时观测维度 ×K） | **1**（不堆叠） |
| **观测模型逐项噪声** | `make_pomdp_env(preset=..., range_sigma_m=..., dropout_prob=..., delay_steps=..., hide_*_truth=...)` | moderate 预设 |
| **不确定度感知回退** | `FallbackConfig(enabled=..., fallback_mode="shield"\|"fallback_rule", use_*_signal=...)` | 仅 `evaluate_uncertainty.py` 启用 |
| **教学资源管理（大阶段一）** | 独立入口 `python -m resource_management` / `evaluate_resource_management.py`；`SchedulingConfig` 与 `OptimizerConfig` 显式配置 | **不接入**任何旧实验路径（旧数字逐位不变） |

---

## 3A. 资源管理层在架构中的位置（大阶段一）

资源管理是**与功率控制并列的另一条线**，不改变上面任何一层的行为：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  编排层  resource_management/closed_loop.py                                  │
│  （**唯一**接触真值的资源模块；真值只在这里，且只喂给各节点自己的传感器）      │
│                                                                              │
│   ① 世界：Simulator + 场景逐 tick 推进（目标真的在动）                        │
│   ② 各节点：自己的 FusionCenter ← 只吃本节点传感器的测量                      │
│   ③ 通信：CommBus ← 只传**节点观测摘要**，中央只能读已到达的                   │
│   ④ 调度：scheduler.plan() → 标准 ExecutionPlan                              │
│   ⑤ 执行：UnifiedExecutor.submit() → 两阶段校验 + 逐节点记账                  │
│                                                                              │
│  ⚠️ scheduling.py / observation.py / tasks.py / optimization.py               │
│     **不 import engine / sensor / fusion**（AST 扫描钉住），                  │
│     只读 CentralObservation；改掉"未来"不会改变"当前"的规划（运行时对照证明）   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ 只读：节点列表 + 掩码 + 逐节点摘要年龄 + 可见航迹
┌───────────────────────────────┴──────────────────────────────────────────────┐
│  方法层（**同一份观测 / 同一个队列 / 同一个执行器**）                          │
│                                                                              │
│  规则基线（scheduling.py）        优化参考（optimization.py）                  │
│   · round_robin 节点轮流+先到先服务  · enumeration  单 tick **完全枚举**       │
│   · edf         只认截止时间        · rolling_horizon 首 tick 枚举 + rollout  │
│   · rule        等级+等待+新鲜度+质量 · 显式预测模型 + 声明式目标 + 计算预算    │
│                                                                              │
│  共用：_classify_due（谁能被服务/什么算放弃）、_duplicate_rule（什么算重复）、   │
│        SchedulerBase.plan 里的**对称计时**（计算耗时维度对两条路径公平）        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ 标准 ExecutionPlan（唯一输出形式）
┌───────────────────────────────┴──────────────────────────────────────────────┐
│  契约层  resource_management/contract_v1.py（冻结，摘要可校验）               │
│   观测 schema rm-obs-1.0 ｜ 任务完成口径 ｜ 资源模型 ｜ 基准配置 ｜ 六维评价    │
│   ⚠️ 改一处即 verify_frozen() 失败 → 必须升版本号 + 同步文档 + 重跑验收        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 目录结构

```
LPI-CogRadar/
├── main.py                          链路预算 → 规则基线 → CSV/HTML → env 演示
├── experiment_config.py             统一实验配置（项目名/种子/视野/策略工厂/运行器/聚合）
├── logging_utils.py                 统一日志（控制台 + output/logs/<run>.log）
│
├── train_dqn.py                     DQN 训练（--safe-rl / --jitter / --adaptive-jammer / --lr-decay）
├── evaluate_dqn.py                  单种子统一测试（--safe-model / --adaptive-jammer / --energy-budget）
├── evaluate_multiseed.py            多种子泛化评测（均值±标准差 + 配对 t 检验）
├── evaluate_jammer_modes.py ★       固定干扰 vs 自适应智能干扰机 分别评测
├── sensitivity_energy_budget.py     能量预算敏感性（CSV + HTML + PNG）
├── diagnose_temporal_coupling.py    时序耦合 / 能量约束 / 实验一致性（6 项检验）
├── diagnose_ai.py ★                 AI 认知诊断 + 可解释决策 + 对比 + 总结
├── ai_server.py ★                   AI 诊断 HTTP 服务（零依赖）
│
├── ai/ ★                            AI 认知诊断层
│   ├── schema.py                    结构化 JSON 协议（只读快照 + 四类结果）
│   ├── provider.py                  AIProvider 接口 + 工厂（与厂商解耦）
│   ├── rule_provider.py             默认 provider：规则+模板，无 Key，离线确定
│   ├── mock_provider.py             确定性桩（可注入故障，验证降级路径）
│   ├── http_provider.py             远程 provider（OpenAI/DeepSeek/通义，兼容协议）
│   ├── context.py                   仿真侧 → 只读快照（边界层）
│   ├── service.py                   四能力门面 + 异常隔离 + 自动降级
│   └── api.py                       /api/diagnose 等四能力 + 零依赖 HTTP 服务
│
├── explain/ ★                       可解释决策
│   └── counterfactual.py            反事实试算 → 结构化证据（不含自然语言）
│
├── sensor/ ★                      分层测量层（零第三方依赖）
│   ├── record.py                  测量记录 + 六种「没有数据」原因的分类
│   ├── occlusion.py               简化遮挡模型（AABB / 球，线段相交）
│   ├── sensor.py                  传感器本体（主动雷达 / 被动 ESM）与套件
│   ├── fusion.py                  测量汇聚为定长观测（**剥掉真值**）
│   ├── config.py                  从场景配置构建套件 / 噪声缩放（ideal 对照）
│   └── reporting.py               测量级统计（误差-距离/逐原因缺失/虚警/周期）
│
├── engine/
│   ├── geometry.py ★               统一坐标与几何（叶子模块，任意实体对的
│   │                                 距离/方位/俯仰/径向速度/视线，含时刻语义）
│   ├── scene.py ★                  场景实体注册表（跨类型唯一索引 + 有向关系查询
│   │                                 + CSV/JSON 快照导出）
│   ├── equations.py                 简化雷达方程 / 侦察方程 / ROC logistic（叶子模块）
│   ├── simulator.py                 Simulator + InfeasibleActionError + 动作可行性 + 自适应干扰机调度
│   ├── env.py                       LpiPowerEnv（Gymnasium 风格，action_masks/裁剪）
│   └── spaces.py                    Discrete / Box 极简实现（零依赖）
│
├── models/
│   ├── entity.py ★                 SceneEntity 实体基类（唯一 ID/三维位置/速度/
│   │                                 航向姿态/时间戳/平台归属 + 唯一的距离实现）
│   ├── radar.py / target.py / interceptor.py / jammer.py
│   ├── adaptive_jammer.py ★         规则自适应智能干扰机（四动作 + 三信号 + 决策轨迹）
│   ├── exposure.py                  累计暴露 / 侦察证据递推
│   ├── reward.py                    综合收益 + 未达标惩罚 + 终端惩罚
│   ├── step_result.py               单步结果（含暴露/剩余能量/未达标/干扰机模式）
│   ├── scenario.py / receive_record.py / behavior.py
│   └── node.py / flow.py / link_state.py / disturbance.py    遗留（通信抗干扰阶段）
│
├── strategy/
│   ├── power_policy.py              固定/规则/随机/逐档贪心(短视)/前瞻规划（均遵守可行性）
│   └── anti_jam.py                  遗留：跳频策略（已弱化）
│
├── rl/
│   ├── dqn_agent.py                 Masked DQN（主分支）
│   ├── lagrangian_agent.py ★        拉格朗日约束 DQN（双 critic + 动态 λ）
│   └── replay_buffer.py             经验回放（状态/动作/奖励/掩码/代价）
│
├── metrics/
│   ├── collector.py                 逐步 + 汇总指标（含 violation_rate / 暴露 / 干扰机动作）
│   └── report.py                    HTML 报告 + 通用曲线页
│
├── resource_management/ ★         教学仿真资源管理（零依赖；**唯一接触真值的是 closed_loop.py**）
│   ├── units.py                    资源单位（sample_slot/processing_op/comm_byte + occupancy_second）
│   │                                 与**教学**成本模型（显式声明：与真实装备参数无关）
│   ├── model.py                    五结构：ResourceBudget / NodeState / TaskRequest /
│   │                                 ExecutionPlan / TaskOutcome·ExecutionResult
│   ├── clock.py                    唯一全局时钟（一次 tick 只推进一次；拒绝时间倒流）
│   ├── ledger.py                   逐条账本（消耗/预留/激活/拒绝/空闲）+ 逐节点汇总
│   ├── executor.py                 UnifiedExecutor：两阶段校验 + **唯一扣费点**
│   ├── observation.py              融合结果 → 调度器只读观测（独立 schema rm-obs-1.0，
│   │                                 25 字段标注单位/坐标系/可见范围/来源；定长适配器）
│   ├── tasks.py                    任务队列（未知对象不得提前建任务；状态机；允许门控参数）
│   ├── scheduling.py ★             三个规则基线 + 模板方法（_classify_due/_duplicate_rule 共用）
│   ├── optimization.py ★           非学习优化参考（显式预测模型 / 声明式目标 / 计算预算 /
│   │                                 规范最优性声明白名单 / 帕累托前沿）
│   ├── closed_loop.py ★            闭环编排 + **RuntimeExecutor**（真闭环唯一副作入口）
│   │                                 两种模式：legacy_observation_first（默认，逐位可复现）
│   │                                 / plan_controlled_feedback（计划控制感知链）
│   ├── acceptance.py ★             阶段验收 6 项（含"改未来不改当前规划"的运行时对照
│   │                                 与"调度是否真能改变航迹质量"的闭环判据）
│   ├── contract_v1.py ★            冻结契约 resource-contract-v1（摘要可校验）
│   ├── learning_env.py ★           独立资源调度学习环境（不 import engine/sensor/fusion/communication）
│   ├── spaces.py ★                 包内最小空间类（零依赖，不引入 gymnasium）
│   ├── exact_cases.py ★            四个可手算的精确案例（解析期望值，测试不依赖随机统计）
│   ├── learning_protocol.py ★      终止/截断/bootstrap 语义 + 数据划分封存（SealedTestSplitError）
│   ├── information_research.py ★   研究分支特征（年龄/协方差/来源一致性指示量）+ 四组门控
│   ├── ai_bridge.py ★              只读 AI 诊断桥（远端超时不阻塞仿真时钟）
│   ├── reporting.py                账本/执行日志/nodes.json/manifest 落盘
│   └── __main__.py                 教学演示（7 步：双节点/不同任务/空闲/重复/不可用/未知/超额）
│
├── config/ ★                       摘要受保护的实验契约（改一处即校验失败）
│   ├── learning_splits_v1.json     训练/验证/测试场景与种子划分（+ .sha256）
│   └── information_research_v1.json 研究分支契约：唯一假设 + 四组消融（+ .sha256）
│
├── tools/ ★                        实验工具
│   ├── capture_baseline.py         基线快照（--out / --skip-ladder）
│   ├── compare_baselines.py        修复前后基线对比（按节解析，避免把汇总行算进上一节）
│   ├── compare_schedulers.py ★     规则基线分工对比（4 场景 × 3 策略 × 3 种子 + 产物）
│   └── evaluate_information_research.py ★ 研究分支四组消融的机制检查
│
├── rl_resource/ ★                 集中式学习资源调度基线（**需要 torch**）
│   ├── scenarios.py                场景目录 → 真闭环机制的映射（首次实现，独立摘要标识）
│   ├── obs.py                      定长观测编码（49 维；只吃调度器本就可见的字段）
│   ├── actions.py                  结构化动作空间（逐节点分类）+ 合法 mask + 动作→ExecutionPlan
│   ├── env.py                      CentralizedResourceSchedulingEnv（真闭环，Gymnasium 风格）
│   ├── policy.py                   Actor-Critic（因素化分类头 + mask，保存/恢复）
│   ├── ppo.py                      PPO + GAE（截断保留 bootstrap、终止切断）
│   ├── train.py                    训练 CLI（train 训练 / validation 选 checkpoint / test 封存）
│   └── evaluate.py                 评测 CLI（默认 validation；`--release-test` 显式解封）
│
├── evaluate_resource_management.py ★  规则基线 vs 优化参考：逐任务计划/拒绝原因/
│                                      约束违反/运行时间/六维向量/验收清单/契约快照
│
├── docs/ ★
│   ├── data_contract.md            能力清单 / 数据流图 / 接口契约 / **未接通路径**（含闭环前后对照）
│   ├── change_report.md            一致性验收阶段的修改记录（E3/E4 等）
│   ├── resource_management.md      资源管理设计说明（§1–§12）+ **§13 真闭环**
│   ├── resource_contract_v1.md ★   大阶段一冻结契约（口径 + 纪律 + 局限）
│   ├── resource_contract_v1.json ★ 冻结快照 + 摘要（人工核对与跨版本对比）
│   ├── learning_protocol.md ★      学习问题与评测协议 v1（终止/bootstrap/精确案例/划分封存）
│   ├── learning_evaluation_checklist.md ★ 开始训练前的评测闸门清单
│   ├── information_research_protocol.md ★ 研究分支协议（唯一假设 + 四组消融 + 信息纪律）
│   └── information_research_dev_report.md ★ 研究分支开发机制报告（含**负结果**）
│
├── task_plan.md / findings.md / progress.md ★ 第三方阅读与开发的过程记录（阶段/决策/错误/发现）
│
├── output/
│   ├── step_*.csv / summary.csv / lpi_power_report.html    规则基线
│   ├── rl/                          主模型（Masked DQN）训练与评测
│   ├── rl_safe/ ★                   约束版模型（拉格朗日 DQN）训练产物
│   ├── jammer_modes/ ★              固定 vs 自适应干扰机对照（含干扰机动作轨迹）
│   ├── ai_diagnosis/ ★              AI 诊断报告（ai_report.json + ai_report.html）
│   ├── runs/<run_id>/ ★             运行隔离目录（每次运行独占；含 manifest.json）
│   │   ├── resource_management/     教学演示的账本/执行日志/节点状态
│   │   ├── scheduler_baselines/     规则基线对比（逐条决策/时间线/观测年龄）
│   │   └── resource_eval/ ★         规则 vs 优化参考（逐任务计划/拒绝原因/约束违反/
│   │                                  运行时/六维对照/acceptance/contract_snapshot）
│   ├── rl_multiseed/ rl_sensitivity/ rl_v1/ legacy_comm_sim/
│   └── logs/ ★                      统一日志
└── backup_original|iter2|iter3/     历史代码快照
```

★ = 本版新增。

---

## 5. 物理与任务模型（自第三版起未改动）

### 5.1 探测链路

```
S = Pt · G² · λ² · σ / ( (4π)³ · R⁴ · L )        N = k·T·B·F
SINR = S / (N + J_eff)
Pd = 1 / (1 + exp( -(SINR_dB − snr50_db) / pd_slope_db ))
```

### 5.2 侦察（截获）链路 —— 低截获的关键在旁瓣

```
Pr = Pt · G_radar→ESM · G_ESM · λ² / ( (4π)² · R_ESM² · L_ESM )
Pint_inst = 1 / (1 + exp( -(Pr/N_ESM|dB − snr50_db_ESM) / pint_slope_db_ESM ))
```

### 5.3 累计暴露 / 侦察证据

```
e(0) = 0
Pint_eff(t) = 1 − (1 − Pint_inst(t)) · (1 − e(t))        ← 用 step 之前的暴露量
e(t+1) = min(1, decay · e(t) + gain · Pint_inst(t))       ← step 之后更新
```

因果顺序：本步动作只影响**未来**的 `e`，形成
`提高功率 → 本步 Pint_inst↑ → 未来 e↑ → 未来 Pint_eff↑ → 未来奖励↓` 的传导链。
参数 `decay=0.85, gain=0.15` 使稳态暴露量 ≈ 瞬时截获概率。

### 5.4 能量硬约束（执行前拦截）

```
E(t+1) = E(t) − Pt · Δt         E 不足则动作被否决，累计能耗永不越预算
```

| 规则 | 位置 | 行为 |
| --- | --- | --- |
| 可行性 = `Pt·Δt ≤ E_remaining` | `Simulator.is_level_feasible()` | 判定 |
| 可行集合/掩码/裁剪 | `feasible_levels()` / `action_mask()` / `max_feasible_level()` / `clip_to_feasible()` | 供策略与智能体使用 |
| 请求不可行动作 | `Simulator.step()` | **抛 `InfeasibleActionError`**，无副作用 |
| 环境层容错 | `LpiPowerEnv.step()` | 裁剪到可行最高档 + `info["action_clipped"]` |
| 策略层 | `strategy/power_policy.py` | 只在 `feasible_levels()` 内选动作 |
| 终止 | `Simulator.is_done` | 时间到 **或** 连最低档都买不起 |

预算 **1400 J**（满足全部 61 步需 1450 J，故约束真正 binding；背包最优 59/61 = 96.7%）。

### 5.5 奖励

```
r = w_det · min(Pd_min/required_pd, 1) − w_int · Pint_eff
  − w_energy · (Pt/Pt_max) − w_violation · 1[Pd_min < required_pd]
能量耗尽终端惩罚：− w_violation × 剩余任务步数
```

---

## 6. 环境接口

```python
from engine import LpiPowerEnv

env = LpiPowerEnv("config/radar_scenario_v1.json")     # 可选 energy_budget_j / reward_weights
obs, info = env.reset(seed=42)

done = False
while not done:
    action = agent.select_action(obs, action_mask=env.action_masks())   # 掩码选动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

* `action_space`：`Discrete(11)`；`observation_space`：`Box(12)`

  | # | 特征 | # | 特征 |
  | --- | --- | --- | --- |
  | 0 | `primary_target_range_norm` | 6 | `remaining_energy_norm` |
  | 1 | `min_target_rcs_norm` | 7 | `time_norm` |
  | 2 | `max_target_range_norm` | 8 | `previous_pd` |
  | 3 | `interceptor_range_norm` | 9 | `previous_pint` |
  | 4 | `jam_noise_ratio_norm` | 10 | `required_pd` |
  | 5 | `previous_power_norm` | 11 | `exposure_norm`（累计暴露） |

* 终止：`terminated=True` 能量耗尽；`truncated=True` 到时长上限。
* `env.action_masks()`：当前可行档位掩码；越界动作被裁剪并记录。

---

## 7. 指标

| 指标 | 含义 |
| --- | --- |
| `horizon_satisfaction_rate` | **主指标**：满足步数 / **完整任务步数** |
| **`violation_rate`** | **约束违反率 = 1 − 主指标**（安全 RL 的 cost 口径） |
| `explicit_violation_steps` | 显式 Pd 未达标步数（与上者口径区分） |
| `avg_tx_power_w` / `cumulative_energy_j` / `remaining_energy_j` | 功率与能量 |
| `avg_intercept_prob` / `avg_instant_intercept_prob` | Pint_eff / Pint_inst |
| `avg_exposure` / `final_exposure` / `cumulative_exposure` | 平均 / 结束 / 累计暴露 |
| `composite_reward` | 综合收益（与 RL 奖励同源） |
| `jammer_modes` | 自适应干扰机出现过的动作集合 |

> ⚠️ **`violation_rate` 的口径很关键**：分母必须是**完整任务步数**。
> 若只用「显式未达标步数 / 完整步数」，会出现「早早烧光能量 → 一步都没执行 → 违反率为 0」
> 的指标漏洞（固定 80 W 基线就会靠这个拿到"最低违反率"的假象）。
> 本工程统一为 `1 − horizon_satisfaction_rate`，并在安全 RL 的 cost 里同步采用。

---

## 8. 自适应智能干扰机（规则型对手）

### 8.1 定位与诚实性声明

**它是「规则型自适应」对手，不是学习型对手。** 它按预先写死的态势判据在四种动作间切换，
没有在训练中优化过自己的策略。工程、报告、答辩里都**不得**称其为「学习型对手」或「RL 对手」。
真正的学习型对手需要独立训练一个干扰方智能体——那属于后续工作。

### 8.2 四种干扰动作

| 动作 | 功率倍数 | 用途 |
| --- | --- | --- |
| `no_jam` 不干扰 | 0.0 | 暴露低、威胁不足时省电静默 |
| `low_power` 低功率压制 | 0.6× | 轻度压制，观察雷达反应 |
| `high_power` 高功率压制 | 1.5× | 雷达持续达标（干扰无效）时全力压制 |
| `intermittent` 间歇干扰 | 1.5× 按周期开/关 | 已压制住雷达或连续高功率过久时省电、规避反辐射 |

### 8.3 三类驱动信号（全部来自敌方视角可观测的量）

```
threat = w_exp · exposure                        ① ESM 累计暴露
       + w_rad · (最近窗口雷达平均功率 / 最大档)   ② 雷达辐射强度
       + w_ineff · (1 − 最近窗口雷达达标率)        ③ 历史探测行为
```

按 `threat` 与 `engage_threshold=0.35` / `escalate_threshold=0.65` 映射到四种动作，
并带模式滞回（`mode_hold_steps`）与连续高功率占空比管理（`high_power_streak_limit`）。

**因果顺序**：本步干扰功率由上一步末决定的模式算出；
本步结束后才观测新证据并决定下一步——绝不回头改变本步。

### 8.4 闭环实录（规则功率控制，seed=42，自适应模式）

```
  t     下一步动作      threat   暴露    雷达功率  是否达标
  0.0     不干扰        0.232   0.052    18.0 W   True
  4.0   低功率压制       0.373   0.193    18.0 W   True   ← 暴露上升，开始压制
  8.0   高功率压制       0.703   0.387    35.0 W   True   ← 雷达被迫提功率
 12.0   高功率压制       1.017   0.547    50.0 W   True   ← 暴露进一步上升
 17.0   间歇干扰         1.036   0.600    50.0 W   True   ← 连续高功率过久，转间歇省电
 24.0   间歇干扰         0.935   0.563    35.0 W   True   ← 压制减弱，雷达降功率，暴露回落
```

形成完整的双向闭环：**雷达辐射 → ESM 积累暴露 → 干扰升级 → Pd 下降 →
雷达重新调功率（往往提功率）→ 暴露进一步上升 → 干扰维持高功率**；
当干扰机转间歇后雷达又能恢复，暴露回落，威胁度随之下降。

### 8.5 固定 vs 自适应：分别评测结果（seed=42）

| 策略 | 固定干扰 满足率 | 自适应干扰 满足率 | 固定 收益 | 自适应 收益 | 自适应 平均功率 | 自适应 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| 固定功率基线(80 W) | 0.2951 | 0.2787 | −3.1912 | −3.2698 | 70.00 W | 0.5637 |
| **规则功率控制** | **0.9180** | 0.6557 | **+0.3510** | −0.3977 | 33.33 W | 0.4657 |
| 随机策略 | 0.4426 | 0.3934 | −0.6289 | −0.7270 | 22.93 W | 0.3160 |
| **DQN(贪心)** | **0.9180** | 0.6393 | **+0.3766** | **−0.3106** | 24.14 W | **0.3714** |
| 逐档贪心(短视) | 0.9180 | 0.6557 | +0.3510 | −0.3929 | 33.33 W | 0.4649 |
| **前瞻规划(非短视)** | **0.9344** | **0.8033** | **+0.4052** | **+0.0831** | 22.91 W | 0.3454 |

**结论**：
1. 自适应对手显著更难——规则策略满足率 **0.9180 → 0.6557**，收益从 +0.3510 跌到 −0.3977；
2. **对手变自适应后，DQN 相对规则策略的优势反而变大**：收益 −0.3106 vs −0.3977，
   且平均功率低 27%（24.14 vs 33.33 W）、Pint 低（0.5975 vs 0.7386）、暴露低（0.3714 vs 0.4657）。
   规则策略被对手"牵着"被迫加大功率，而 DQN 学会了不去过度刺激对手
   （它触发的干扰动作分布与规则不同）。**这是"对手越强、学习型策略越有价值"的直接证据。**
3. 前瞻规划仍最强（0.8033 / +0.0831），它拥有完整模型知识，属于不可部署的参考上界。

干扰机动作轨迹写入 `output/jammer_modes/jammer_trace_JAM1.csv`（敌我双方逐步动作记录）。

---

## 9. AI 认知诊断层

### 9.1 架构约束（不可违反）

1. **AI 层没有控制接口**：`StateSnapshot` 里没有任何动作指令字段，
   provider 接口里也没有 `set_power` 之类方法。AI 只描述、解释、建议。
2. **AI 失败绝不影响主仿真**：provider 的任何异常都在 `AIDiagnosisService` 内被捕获，
   按 `fallback` 链降级（默认回退本地 `rule`）；HTTP 层始终返回 HTTP 200 + `status:"error"`。
3. **无 Key 即可运行**：默认 provider 是本地 `rule`（纯规则+模板，离线、确定、毫秒级）。
4. **解释必须可追溯**：自然语言只允许引用结构化证据里的**代码与数值**（见第 10 章）。

### 9.2 Provider 抽象（与厂商解耦）

```python
class AIProvider(ABC):
    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult
    def explain_decision(self, payload: dict) -> ExplainResult
    def compare_policies(self, payload: dict) -> CompareResult
    def generate_report(self, payload: dict) -> ReportResult
    def info(self) -> ProviderInfo
    def health(self) -> dict
```

| provider | 需要 Key | 说明 |
| --- | --- | --- |
| `rule`（默认） | ❌ | 规则+模板；离线确定；远程 provider 的兜底 |
| `mock` | ❌ | 确定性桩；`fail_rate=1.0` 可注入故障，用于验证降级路径 |
| `openai` / `deepseek` / `qwen` / `http` | ✅ | OpenAI 兼容 `/chat/completions`；只发送结构化状态；Key 只从参数或环境变量读取，不落盘不进日志 |

### 9.3 四个能力 / 路由

| 能力 | 路由 | 输入 | 输出 |
| --- | --- | --- | --- |
| 实时态势诊断 | `POST /api/diagnose` | `StateSnapshot` | `DiagnosisResult` |
| 策略解释 | `POST /api/explain_decision` | `{"counterfactual": {...}}` | `ExplainResult` |
| 多策略对比 | `POST /api/compare_policies` | `{"summaries":[...], "metric_order":[...], "context":{...}}` | `CompareResult` |
| 实验总结 | `POST /api/generate_report` | `{"title":..., "summaries":[...], "context":{...}}` | `ReportResult` |

`GET /api` 返回自我描述（路由、入参 schema、provider 信息）。
能力既可通过 `ai/LPI_API` 直接调用（不起服务），也可通过 `ai_server.py` 走 HTTP。

### 9.4 输入数据格式（StateSnapshot，节选）

```json
{
  "project": "LPI-CogRadar", "schema_version": "1.0",
  "scenario": "minimal_comm_jam_runable", "step_index": 30, "time": 30.0, "horizon_steps": 61,
  "targets": [{"target_id": "TGT1", "range_m": 5700.0, "rcs_m2": 2.0,
               "snr_db": 9.42, "pd": 0.832}],
  "interceptors": [{"interceptor_id": "ESM1", "range_m": 100000.0, "beam": "sidelobe",
                    "snr_db": 28.43, "pint_inst": 0.599}],
  "jammers": [{"jammer_id": "JAM1", "active": true, "mode": "adaptive",
               "action": "high_power", "action_cn": "高功率压制",
               "jam_noise_ratio": 1.52, "threat": 0.703}],
  "detection":   {"pd_min": 0.832, "required_pd": 0.8,
                  "task_satisfied": true, "task_violated": false},
  "interception":{"pint_eff": 0.818, "pint_inst": 0.599, "exposure": 0.232},
  "power":  {"level": 8, "tx_power_w": 35.0,
             "power_levels_w": [0.5,1,2,4,8,12,18,25,35,50,80],
             "feasible_levels": [0,1,2,3,4,5,6,7,8],
             "action_mask": [true, "...", false], "previous_level": 8, "switched": false},
  "energy": {"budget_j": 1400.0, "remaining_j": 655.0, "cumulative_j": 745.0,
             "fraction_used": 0.532, "min_step_energy_j": 0.5},
  "agent":  {"kind": "dqn", "action": 8, "tx_power_w": 35.0, "q_values": ["..."],
             "q_margin": 0.42, "epsilon": 0.02, "lambda_cost": null, "cost_rate": null},
  "reward": 0.378,
  "recent": {"violation_rate": 0.033, "avg_power_w": 24.1}
}
```

### 9.5 输出数据格式（DiagnosisResult，节选）

```json
{
  "provider": "rule", "model": "rule-engine-v1", "status": "ok",
  "severity": "warning",
  "summary": "t=30.0s（第 30/61 步）需要关注。Pd=0.832（要求 0.800），Pint_eff=0.818，暴露=0.232，当前功率 35.0 W（档位 8），剩余能量 655.0/1400 J；主要问题：探测概率接近门限、按当前功率无法支撑到任务结束、有效截获概率偏高。",
  "findings": [
    {"code": "DETECTION_AT_RISK", "severity": "warning", "title": "探测概率接近门限",
     "message": "Pd=0.832 仅高于 required_pd=0.800 +0.032，几何或干扰稍有变化就会跌破。",
     "evidence": {"pd_min": 0.832, "required_pd": 0.8, "margin": 0.032}},
    {"code": "ENERGY_OVERRUN_RISK", "severity": "warning",
     "title": "按当前功率无法支撑到任务结束",
     "message": "以当前 35.0 W 再发 31 步需要 1085.0 J，超过剩余 655.0 J。",
     "evidence": {"needed_j": 1085.0, "remaining_j": 655.0}}
  ],
  "recommendations": ["降低平均功率或主动放弃部分最贵的任务步，把能量留给后段。"],
  "confidence": 0.85, "latency_ms": 0.03, "generated_at": "2026-09-18 16:17:44",
  "error": "", "schema_version": "1.0"
}
```

诊断结论代码（`FINDING_CODES`）覆盖：探测风险/未达标、能量低/耗尽/超支风险、
暴露偏高、截获风险高、干扰升级、功率到顶、低截获状态等。

### 9.6 失败隔离与降级（实测）

| 场景 | 行为 |
| --- | --- |
| provider 构造失败（如远程缺 Key） | 记录 warning，自动降级到 `rule`，仿真照常 |
| provider 调用抛异常 | 捕获 + `logger.warning`，尝试 fallback 链 |
| 全部 provider 失败 | 返回 `status="error"` 的结构化结果，**不抛异常** |
| `--no-ai` | 使用 `NullProvider`/`disabled()`，全链路仍可完整运行 |
| HTTP 层 | 一律 HTTP 200 + `status` 字段，不返回 5xx |

用 `MockProvider(fail_rate=1.0)` 可以把上述路径全部走一遍。

---

## 10. 可解释决策（反事实证据）

### 10.1 做法

对**同一时刻**的若干候选功率档位逐一试算（`Simulator.preview()` 是纯函数，不改变状态），
算出各自的 Pd、Pint_inst、Pint_eff、暴露（下一步）、单步能耗、剩余能量、
可支撑步数与**单步收益**，再归纳成结构化归因。

默认试算 **18 / 25 / 35 W**（用户可读性优先）以及执行档位的相邻档位，共 6 个候选。

### 10.2 证据代码（`explain/counterfactual.py::EVIDENCE_CODES`）

| 代码 | 含义 |
| --- | --- |
| `MINIMAL_SATISFYING` | 当前档位已是满足探测要求的最低可行档 |
| `AVOID_VIOLATION` | 更低档位会跌破 required_pd，触发未达标固定惩罚 |
| `VIOLATION_ACCEPTED` | 当前未达标，是为省电/降暴露而主动放弃该步 |
| `ENERGY_LIMITED` / `ENERGY_MARGIN` | 受剩余能量限制 / 能量余量不足 |
| `HIGHER_POWER_WORSE` | 更高档位单步收益更低（暴露与能耗代价超过探测收益） |
| `LOWER_POWER_BETTER` / `REDUCE_EXPOSURE` | 更低档位更优 / 降功率可显著降低暴露 |
| `DETECTION_DOMINANT` | 探测收益占主导，宜优先保 Pd |
| `GREEDY_OPTIMAL` | 与逐档贪心的单步最优一致 |

### 10.3 实测输出（规则策略 t=30s）

```
本步维持功率：执行档位 8（35.0 W），判定代码 MINIMAL_SATISFYING。
依据：当前 35 W 已是满足探测要求的最低可行档（Pd=0.832 ≥ required_pd=0.8）；
      升到 50 W 只能把 Pd 提到 0.915（探测项已封顶），却让 Pint_eff 从 0.818 升到 0.874、多耗 15.0 J；
      降到 18 W 会让 Pd 掉到 0.539（< 0.8），触发未达标惩罚。
反事实试算：18W → Pd 0.539、Pint_eff 0.704、暴露(下一步) 0.5167、单步收益 −1.3159；
            25W → Pd 0.705、Pint_eff 0.760、暴露(下一步) 0.5351、单步收益 −1.1684；
            35W → …… （✔ 实际执行）
证据代码：MINIMAL_SATISFYING、HIGHER_POWER_WORSE、AVOID_VIOLATION
```

**每一句都能追溯到仿真里的具体数字**。自然语言由 AI 层生成时只允许引用这些
证据代码与数值——大模型没有编造理由的空间。

---

## 11. 安全强化学习（拉格朗日约束 DQN）

### 11.1 为什么需要它

主 DQN 把「探测未达标」写成一个**固定惩罚**塞进奖励里：

```
r = … − w_violation · 1[Pd < required_pd]      （w_violation = 1.5）
```

两个弱点：① 惩罚权重得手调，且与低截获/能耗目标同量纲相加，**无法指定目标违反率**；
② 训练中无法直接知道「约束满足得怎么样」。

### 11.2 约束形式与拉格朗日松弛

```
maximize  E[Σ r_t]                       （r 里**去掉**违反惩罚项）
s.t.      E[平均代价] ≤ d                 c_t = 1[Pd_t < required_pd]，d = 0.08

L(θ, λ) = E[Σ (r_t − λ·c_t)] − λ·d·T
对偶上升：λ ← clip( λ + η_λ · (ĉ_episode − d), 0, λ_max )
```

**λ 的量级很重要**：本任务单步奖励量级约 **0.3**，单次违反的代价是 **1.0**，
因此有意义的 λ 区间大致是 **[0.5, 3]**。λ 取到 20 会让代价项彻底压倒奖励项，
策略退化成「不惜一切代价不违反」，等于把软约束变成硬约束并放弃低截获与能耗优化。
默认 `lambda_lr=0.01`、`lambda_max=5.0` 就是按这个量级选的。

`d` 取 **0.08** 而不是 0.05：能量预算下**最优策略的违反率下限约 0.066**
（前瞻规划实测），0.05 是不可行的，λ 会一路顶到上界、实验失去区分度；
取 0.08 让约束恰好落在可行区间内、且对规则策略（实测 0.082）是激活的。

### 11.3 实现要点

* **双 critic**：奖励 critic `Q_r` 与代价 critic `Q_c`，各自带目标网络、
  共用同一套动作可行性掩码与 Huber 损失；
* **动作选择**：在可行档位内取 `argmax (Q_r − λ·Q_c)`——「满足约束前提下最大化收益」的贪心近似；
* **target 动作同源**：奖励 critic 和代价 critic 都评估同一个
  `argmax (Q_r − λ·Q_c)` 下一动作，不再各自选一个与执行策略不一致的动作；
* **λ = 0 时退化为普通 DQN**，因此两者完全兼容、可平滑对比；
* **不修改主 DQN**：`LagrangianDQNAgent` 继承 `DQNAgent` 只覆写训练与选择，主实验路径不变；
* **代价口径与评测一致**：`ĉ` 以**完整任务步数**为分母，能量耗尽导致没执行到的步同样计入违反。

### 11.3.1 两个踩过的坑（重要，写下来避免重犯）

把朴素的拉格朗日对偶上升直接套到「带硬资源约束」的环境上会失效，本工程实际踩了两个坑：

**坑一：终端惩罚被连带清零。**
安全 RL 需要把逐步的违反惩罚从奖励里去掉（改由代价 critic 表达），
但如果 `violation` 权重同时被终端惩罚复用，`violation=0` 会让
「能量耗尽终端惩罚」也变成 0——智能体立刻发现「烧光能量提前结束」不再有代价，
于是疯狂提功率规避当前违反、把能量提前烧光，
结果**违反率从 0.082 升到 0.197**（跑不完的步同样算违反）。
修法：给终端惩罚独立权重键 **`terminal`**，与 `violation` 解耦。

**坑二：用错违反率做对偶上升。**
对偶上升必须用**部署策略（贪心）**的违反率，而不是 ε-greedy 训练 rollout 的违反率。
训练早期 ε=1，探索噪声让 rollout 违反率高达 0.15~0.46，
λ 会被这个噪声一路顶到上界；而部署时策略是贪心的，根本不需要那么大的 λ，
结果被过大的 λ 压成「不惜一切代价不违反当前这一步」——同样把能量烧光。
修法：λ 只在**周期性贪心评测**之后更新（`update_lambda(cost_rate_override=...)`）。

**坑三：λ 的量级必须与奖励量级匹配。**
本任务满足一步探测的奖励约 **+1.0**，违反一次代价 **1.0**，
而 `Q_c` 估计的是**折扣累计**未来代价（γ=0.95 时有效视野约 20 步），
所以 `λ·Q_c` 天然比「单步惩罚」大一个数量级。
λ 取 5 或 20 会让代价项彻底压倒奖励项；有效区间大致是 **[0.5, 1.5]**，
默认 `lambda_max=1.5`。

> 这三条合起来说明一件事：**拉格朗日方法在「动作会消耗不可再生资源」的环境里
> 不能照搬**——约束的"代价"和资源的"消耗"通过未来状态耦合在一起，
> 过强的乘子会让智能体做出比无约束时更差的长期决策。

### 11.3.2 约束目标 `d` 的取法

`d` 必须落在「可行」与「激活」之间：

| 策略 | 实测违反率 |
| --- | --- |
| 前瞻规划（最优参考） | 0.0656 |
| 普通 DQN | 0.0820 |
| 规则功率控制 | 0.0820 |

因此默认 **`d = 0.07`**：低于普通 DQN 的 0.082（约束是**激活**的，必须牺牲一点收益才能压下来），
又高于前瞻的 0.0656（约束是**可行**的，不会让 λ 一路顶到上界失去区分度）。

### 11.4 对比结果（含**校核前历史负结果**，如实保留）

统一评测（seed=42，固定干扰，**全部策略都用标准奖励权重评测**，口径一致）：

| 策略 | 满足率* | **违反率** | 执行步数 | 平均功率 | 累计能耗 | 平均 Pint | 平均暴露 | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 固定功率基线(80 W) | 0.2951 | 0.7049 | 20 | 70.00 W | 1400.0 J | 0.9051 | 0.5637 | −3.1912 |
| 规则功率控制 | 0.9180 | 0.0820 | 58 | 24.14 W | 1400.0 J | 0.6432 | 0.4022 | +0.3510 |
| 随机策略 | 0.4426 | 0.5574 | 61 | 22.93 W | 1399.0 J | 0.5502 | 0.3160 | −0.6289 |
| **普通 DQN（固定惩罚）** | **0.9180** | **0.0820** | 61 | 22.95 W | 1400.0 J | 0.6246 | 0.3878 | **+0.3766** |
| 逐档贪心(短视) | 0.9180 | 0.0820 | 58 | 24.14 W | 1400.0 J | 0.6432 | 0.4022 | +0.3510 |
| **前瞻规划(非短视)** | **0.9344** | **0.0656** | 61 | 22.89 W | 1396.5 J | 0.6068 | 0.3777 | **+0.4052** |
| **DQN(约束版)** | 0.8689 | **0.1311** | 61 | 22.74 W | 1387.0 J | 0.6155 | 0.3909 | **+0.2230** |

**约束版没能达成约束目标（d=0.07），综合收益也低于普通 DQN。这是一个必须保留的历史负结果。**

> 2026-09 学习协议审计发现：该次训练的两个 critic 曾在下一状态各自选动作，
> 与部署时的拉格朗日执行策略不一致。因此下表能证明“当时实现跑出了负结果”，
> 但不能证明“某类方法必然失效”。修复尚未重训，新协议见 `docs/learning_protocol.md`。

为此做了三次配置实验（同一架构、同一场景）：

| 运行 | λ 初值 | η_λ | λ 上界 | episode | 最终 λ | 贪心违反率（目标 0.07） | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 冷启动 | 0.0 | 0.05 | 1.5 | 4500 | 0.2015 | 0.0984 ✗ | +0.3852 |
| B 热启动 | 0.6 | 0.50 | 1.5 | 3000 | 1.5（饱和） | **0.5246 ✗（发散）** | −1.2272 |
| C 封顶 | 0.9 | 0.05 | 1.0 | 3000 | 1.0（饱和） | 0.1311 ✗ | +0.2230 |

### 11.5 历史观察：违反率对 λ **非单调**（待修复后复核）

三次实验共同指向同一个机理：

```
  λ ≈ 0.92 ~ 0.96  ->  违反率最低（实测 0.0656 ~ 0.066）  ← 约束在这一窗口内是能满足的
  λ ≥ 1.0          ->  违反率反而上升：0.13 -> 0.31 -> 0.52
                       同时平均功率飙到 45 W、满足率掉到 0.48、综合收益变负
```

**物理原因**：本环境里「避免违反」的唯一手段是**提高功率**，
而功率消耗的是**不可再生的能量预算**。λ 越大，智能体越倾向于
「牺牲整个未来去保住当前这一步」——提功率 → 能量提前耗尽 →
后面几十步全都无法执行 → 而**未执行的步同样算违反**。
于是违反率随 λ **上升**，对偶上升拿到这个反馈后继续加大 λ，形成**正反馈发散**。

这是把朴素拉格朗日对偶上升直接套到「带硬资源约束」环境上的典型失效模式：
**约束的代价与资源的消耗通过未来状态耦合在一起**，把 λ 调大反而让长期决策更差。

**为什么三次都没达标**：
* A：λ 爬得太慢（η_λ 小 + 每 100 episode 才更新一次），整个训练预算内只爬到 0.20，约束形同虚设；
* B / C：λ 一旦越过 ~1.0 就进入发散区，`lambda_max` 只是把发散**截断**在某个较差的点上，
  并不能把它拉回有效窗口。

**现在能确认的范围**：
1. 旧实现下确实观测到上表数值，原始产物不删除；
2. 已修正双 critic target 与执行策略的动作不一致，并加入固定网络值接口测试；
3. `λ = 0` 退化、有效窗口和失效机理都需在新终止语义、固定评测划分下重新实验；
   在此之前只能视为候选解释，不是已排除实现错误后的算法结论。

**改进方向**（已列入第 16 章后续工作）：把**能量**也纳入约束（而不是只当作状态）、
改用信任域类方法（CPO）、用 `Q_c` 的期望而非实测违反率做对偶上升、
对 λ 加阻尼/动量、或在 λ 上做一维线搜索而不是梯度上升。

> **当前可对外表述**：「校核前的拉格朗日 DQN 运行未达到目标违反率，该负结果已保留；
> 后续审计发现 critic target 与执行策略不一致，因此暂不将该结果归因为方法必然失效。」
> 同样**不得**宣称约束版优于普通 DQN。

训练过程中 λ 与**贪心违反率**的逐次演化记录在 `output/rl_safe/training_log.csv`
（`lambda_cost`）与 `eval_log.csv`（`violation_rate`）里，可直接画出对偶上升曲线。

---

## 11A. 部分可观测与可信认知决策（v4.0 核心）

### 11A.1 为什么必须补上「部分可观测」

v3.1 及其之前的所有版本，环境给智能体的都是**真值**：目标距离、RCS、干扰强度、
剩余能量、侦察机方位、真实 Pint、真实累计暴露，全部精确已知。
这等于假设雷达拥有一套完美的传感器与完美的对手状态情报。

现实里这些量都要靠测量与估计：距离有测距误差、J/N 有估计误差、
侦察机在被动工作模式下**无法被精确定位**、累计暴露是内部递推量且其模型本身有误差。
把这个假设去掉，会直接改变结论——规则策略在 moderate 噪声下综合收益
从 `+0.4636` 掉到 `+0.2362`，满足率掉 13.1 个百分点。

**实现方式**：新增独立模块 `engine/observation_model.py`，把「真值 → 可观测量」
这一步从仿真内核里分离出来。仿真照旧演进真值，只有观测被污染。

```
真值 state ──► ObservationModel ──► 估计值 + σ + 新鲜度 ──► 观测向量 ──► 策略/DQN
                    ▲
                    └─ 测量噪声 / 延迟缓冲 / 随机丢测 / hide_* 隐藏开关
```

### 11A.2 观测模型

| 机制 | 配置项 | 说明 |
| --- | --- | --- |
| 逐项测量噪声 | `range_sigma_m`、`rcs_sigma_m2`、`jam_ratio_sigma`、`energy_sigma_j`、`interceptor_range_sigma_m`、`pint_sigma`、`exposure_sigma`、`pd_sigma` | 零均值高斯，量纲各自独立 |
| 观测延迟 | `delay_steps` | 延迟缓冲，取 `delay_steps` 步之前的干净测量 |
| 随机丢测 | `dropout_prob`、`hold_last_on_dropout` | 逐量独立伯努利；丢测时沿用上次测量并放大 σ |
| 隐藏真值 | `hide_interceptor_truth`、`hide_pint_truth`、`hide_exposure_truth` | 关闭即等于给真值（消融对照） |
| 置信度上报策略 | `report_honest_sigma`、`sigma_underreport_factor` | 设为 False 可模拟**过度自信**的智能体 |

三档预设（`experiment_config.OBSERVATION_PRESETS`，只调测量噪声，不动物理参数）：

| 预设 | 距离 σ | 干扰 σ | 侦察机 σ | 延迟 | 丢测率 | 实测观测质量 |
| --- | --- | --- | --- | --- | --- | --- |
| `mild` | 60 m | 0.05 | 2000 m | 1 步 | 3% | 0.656 ~ 0.691 |
| `moderate`（默认） | 120 m | 0.10 | 4000 m | 1 步 | 10% | 0.580 ~ 0.685 |
| `severe` | 250 m | 0.20 | 8000 m | 2 步 | 25% | 0.497 ~ 0.671 |

观测维度：`full` = 12（与 v3.1 完全一致）；`pomdp` = 12 + 4 维不确定度通道
（观测质量、侦察机距离 σ、暴露 σ、Pint σ）。`history_len=K` 时维度 ×K。

> **踩坑记录（观测质量口径）**：初版用 `1/(1+σ/0.1)` 计算观测质量，
> 而侦察机距离的 σ 是 4000 m，代入后该项得分 `2.5e-5`，
> 把整体质量压到 0.3 左右，于是「观测严重退化」在任何一步都成立——
> 实测「高风险步占比」恒为 `1.000`，回退机制形同虚设。
> 按各量程分别归一化（`ObservationModel.SIGMA_REFERENCE`）后质量回到 0.50~0.69，
> 触发条件才重新具备区分度。**归一化尺度写错会让一整类实验静默失效**，
> 因此它现在是显式常量表 + 有测试覆盖。

### 11A.3 关键不变式：噪声只影响「看到什么」，不影响「发生什么」

这是整个 P1 实验成立的前提，有专门的单元测试钉住
（`tests/test_pomdp_env.py::TestPomdpInvariant`）：

> 同一串动作在 `full` 与 `pomdp` 两个模式下，
> 真实轨迹（Pt / Pd / Pint / J-N / 暴露 / 能耗）与**逐步奖励必须完全相等**。

如果这条不变式被破坏，说明观测模型越界改动了物理层，
那么所有「部分可观测 vs 全可观」的对比都失去意义。

### 11A.4 信念状态桥接：让脚本基线公平参赛

工程里所有脚本基线（规则 / 短视 / 前瞻）签名都是 `select_level(sim)`，
直接读真值并调用 `sim.preview()` 试算。一旦 DQN 只能看带噪观测，
拿它们直接对比就是**信息不对称**对比。

`strategy/belief_policy.py` 解决这个问题：从真值仿真器深拷贝一份**信念仿真器**，
把「智能体测到的估计值」写进去，使副本在 `preview()` 下表现得像那些估计值是真值。

| 量 | 写入方式 | 精度 |
| --- | --- | --- |
| 剩余能量 | 按估计值反推累计能耗 | **精确** |
| 主目标距离 | 统一径向缩放 | 精确（其余目标被同一比例缩放） |
| 最小 RCS | 统一缩放 | 精确（保留目标间相对关系） |
| 干扰 J/N | 探针法标定「每瓦峰值功率产生的干扰功率」后反解 | 当前步精确；未来起伏沿用真值序列 |
| 累计暴露 | 直接写入 | 精确 |

`tests/test_belief_policy.py` 逐项验证信念复现了估计值，**并验证信念在噪声开启时
显著偏离真值**（否则桥接形同虚设），以及构造信念不会改动真值仿真器。

### 11A.5 不确定度从哪来：集成 DQN 的两路独立信号

`rl/ensemble_agent.py` 用 N=5 个独立初始化的 Q 网络组成集成，
每个成员在**自助采样的子批次**上更新（Bootstrapped DQN 的做法）：

```
Q_mean(a) = (1/N) Σ_i Q_i(a)
Q_std(a)  = sqrt( (1/N) Σ_i (Q_i(a) - Q_mean(a))² )     ← 认知不确定度
```

再加一路**与网络无关**的信号：

```
ood_score = mean_d | (obs_d - μ_d) / σ_d |                ← 观测相对训练分布的偏离
             μ, σ 用 Welford 算法在线维护
```

两路互不依赖：集成分歧在「N 个网络一起错」时会失效，OOD 评分则在输入本身陌生时报警。
第三路信号是环境给的**观测质量**（丢测/延迟/精度）。三路独立，只用其中一个都有盲区——
`evaluate_uncertainty.py --signal-ablation` 就是为验证这一点而写的。

> **诚实性说明**：这不是「学习到的置信度」，而是基于集成分歧的启发式不确定度。
> 有理论依据（Lakshminarayanan 2017 深度集成；Osband 2016 Bootstrapped DQN），
> 但**不是校准过的概率**，不能解释成「犯错的概率」。
> 另外，各成员**共享同一个回放缓冲区**（只靠初始化与自助掩码区分），
> 多样性低于标准 Bootstrapped DQN，分歧信号偏保守。不得声称「N 个完全独立的模型」。

### 11A.6 三种决策模式与触发条件

| 模式 | 行为 |
| --- | --- |
| `ai` | 执行集成 DQN 的掩码内 argmax |
| `shield`（安全护盾） | 若 AI 动作**低于**「按估计状态刚好够用」的最低档，抬升到那一档；**只抬不降** |
| `fallback_rule` | 整体交给可解释的规则策略（在信念状态上决策） |

触发条件（任一命中即进入高风险状态，按优先级取 `reason_code`）：

| 原因码 | 条件 | 防的是什么 |
| --- | --- | --- |
| `observation_severely_degraded` | 观测质量 < `obs_quality_threshold`（默认 0.60） | 丢测/延迟太严重 |
| `out_of_distribution` | `ood_score` > 3.0 | 输入陌生 |
| `high_ensemble_disagreement` | `q_std_max` > 0.35 或 `disagreement` > 0.60 | 网络之间互相矛盾 |
| `small_q_margin` | `q_margin` < 0.05 | 动作排序接近随机 |

**模式与风险状态是正交的**：`triggered` 非空表示「处于高风险状态」，
`mode` 表示「最终动作是谁定的」。因此「高风险但护盾判断无需干预」会被如实记成
`mode=ai + triggered 非空`，不会被伪装成一次回退。这个区分直接决定了指标能不能信。

### 11A.7 四个必须汇报的指标与判定纪律

| 指标 | 定义 |
| --- | --- |
| AI 自主率 | 最终执行 AI argmax 的步数 / 总步数 |
| 回退率 / 护盾率 | 整体回退的步数 / 被护盾抬升的步数 占比 |
| 错误决策率 | 该步探测未达标（**真值**口径）的比例，**分别**统计 AI 自主步与被干预步 |
| 高风险状态任务满足率 | `triggered` 非空那些步的满足率 |

**判定纪律（很重要，写下来避免自欺）**：

1. 被干预步的错误率通常**高于** AI 自主步——这不是回退失败，
   而是触发条件本就倾向于在困难步上命中（**选择性偏差**）。
   直接比这两个错误率会得出完全错误的结论。
2. 判断「回退有没有用」必须看**单步反事实**：
   在决策时刻的真实状态上，分别试算「AI 原本要选的功率」与「实际执行的功率」，
   看两者的 Pd 是否达标。据此定义
   `shield_rescue_rate`（AI 本会失败、干预后达标）与 `shield_harm_rate`（反而变差）。
3. 阈值是**人工设定**的，所以回退率主要反映阈值选择而非智能体的内省能力。
   必须同时给出 `--threshold-sweep` 的松/中/紧三档结果。
4. 当高风险步占比 ≥0.90 或 ≤0.10 时，「高风险 vs 低风险满足率」**不可解读**
   （其中一组样本太少）。脚本会显式打印该警告，而不是照样输出一个看起来正常的数。

### 11A.8 AI 诊断层如何解释「我不确定」与「为什么回退」

`StateSnapshot` 新增两节（`ai/schema.py`）：

* `observability`（`ObservabilityState`）：观测模式、观测质量、本步丢测/延迟的量、
  各量上报 σ、历史窗口长度。**只含估计值与上报 σ，不含真值**——
  AI 要解释的是「智能体以为自己看到什么」，混入真值会让诊断失真。
* `trust`（`TrustState`）：集成分歧、成员投票分歧、OOD 评分、Q 优势、
  决策模式、触发原因码，以及自主率/回退率/护盾率。

对应的诊断发现码：

| 代码 | 含义 |
| --- | --- |
| `OBSERVATION_DEGRADED` / `OBSERVATION_MISSING` | 观测链路退化 / 本步丢测 |
| `ESM_POSITION_UNKNOWN` | 侦察机真实位置不可直接观测（附 σ） |
| `EXPOSURE_ESTIMATE_UNCERTAIN` | 累计暴露只能估计 |
| `AI_HIGH_UNCERTAINTY` | 集成各成员判断分歧 |
| `AI_OOD_INPUT` | 当前观测偏离训练分布 |
| `AI_SMALL_Q_MARGIN` | 最优与次优 Q 接近，决策缺乏区分度 |
| `AI_FALLBACK_TRIGGERED` | 已回退到规则策略（并**明确指出回退不保证更好**） |
| `AI_SHIELD_APPLIED` | 安全护盾生效（只抬升功率） |

### 11A.9 一条真实踩坑：`strategy/` 不得依赖 `rl/`（否则 torch 会渗进零依赖层）

本工程的分层约定是硬约束：

| 层 | 目录 | 允许的依赖 |
| --- | --- | --- |
| 仿真层 | `engine/` `models/` `strategy/` `metrics/` | **仅标准库** |
| 学习层 | `rl/` | torch（且不用 numpy） |
| 认知层 | `ai/` `explain/` | **仅标准库** |

v4.0 新增的 `strategy/uncertainty_policy.py` 一开始写了
`from rl.ensemble_agent import UncertaintyInfo` —— 只是为了一个**类型标注**。
后果是 `import strategy.uncertainty_policy` 会把 torch 一起拖进来，
于是本该零依赖的仿真层在 base 环境（无 torch）下直接 `ImportError`，
而这一点在 pytorch 环境里跑测试**完全看不出来**。

修法：改成 `if TYPE_CHECKING:` 下的导入。运行时不需要该类型，
因为回退策略对智能体只用**鸭子类型**（调用 `agent.uncertainty(...)` 并读属性）。

现在有自动化检查守住这条线：
`verify_v4.py` 用 base python 跑一遍全部导入，任何越界依赖都会立刻失败。

> 教训：**「某个环境里能跑」不等于「依赖关系正确」**。
> 跨层类型标注必须走 `TYPE_CHECKING`，
> 否则一个纯注释性的 import 就能悄悄破坏整个分层设计。

---

### 11A.10 真值泄漏审计（v4.3）：`belief_policy` 曾经偷看未来与敌方位置

**这是本项目发现并修复的一个真实缺陷，不是理论担忧。**

问题
----
`strategy/belief_policy.py::build_belief_simulator` 用 `copy.deepcopy(env.sim)`
构造信念仿真器。深拷贝会把**整个**真值仿真器复制过来，其中包含两项
本不该被信念知道的信息：

1. **未来的干扰起伏**。干扰强度起伏是**开局一次性预生成整段**的
   （`jammer._fluctuation_series`，长度 = 总步数 + 1），
   因此副本里带着未来每一步的真实干扰强度；
2. **观测不到的平台真值位置**。雷达没有对敌方侦察机的定位手段，
   但深拷贝把 ESM 的**真实坐标**原样带进了信念。

审计证据（可证伪的行为级判据）
------------------------------
判据不是"字段是否相等"（太脆弱），而是
**篡改真值后信念行为是否跟着变**：

| 检验 | 结果 |
| --- | --- |
| 信念是否携带真值的未来起伏序列 | **是**（尾部逐位相同） |
| 篡改真值**未来**起伏后，信念推演 J/N | **1.519854 → 6.908427**（变了 ⇒ 泄漏） |
| 把真值 ESM 挪走 100 km 后，信念算出的 Pint | **0.530 → 0.329**（变了 ⇒ 泄漏） |

> 第一次做这个检验时选在 t=10 s，J/N 恒为 0（干扰机时间窗是 20~45 s 还没开机），
> 得出"没有泄漏"的**假阴性**。换到 t=30 s 才测出来。
> **测试必须选在信号真正非零的时刻**，否则断言毫无意义。

修复
----
`build_belief_simulator` 现在先执行 `_sanitize_belief()`：

| 项 | 处理 | 可选值 |
| --- | --- | --- |
| 未来干扰起伏 | 截断到当前步，未来用**持续性模型**（沿用最后一个已知值） | `persistence`（默认）/ `resample`（用信念自己的 RNG 重采样）/ `keep_truth`（**显式泄漏**，仅供"先知"上界对照） |
| 观测不到的侦察机 | 在信念里置为 `is_active=False`，`preview()` 不再产生截获记录 | `deactivate`（默认）/ `keep_truth`（显式泄漏） |
| 随机流 | 给信念自己的 `random.Random(seed)`，不与真值共享 | `seed` |

修复后同一组检验全部翻转为"不变"：信念推演 J/N 对真值未来**不再敏感**
（恒为 1.519854），信念 Pint 对真值 ESM 位置**不再敏感**（恒为 0.380554）。
回归测试见 `tests/test_belief_policy.py::TestNoFutureLeak`（9 项）。

对既有结论的影响（逐一量化，A/B 同策略同种子）
----------------------------------------------
用 `BeliefOptions(future_jammer_model="keep_truth", unobservable_interceptor="keep_truth")`
可以复现修复前的行为，从而直接量化"泄漏带来了多少偏差"。5 个种子：

| 模式 | 策略 | 泄漏版（修复前） | 修复版 | Δ |
| --- | --- | --- | --- | --- |
| pomdp | 规则功率控制 | +0.1485 | +0.1485 | **+0.0000** |
| pomdp | 逐档贪心(短视) | +0.1740 | +0.1740 | **+0.0000** |
| realistic | 规则功率控制 | +0.4313 | +0.4313 | **+0.0000** |
| realistic | 逐档贪心(短视) | +0.4227 | +0.4227 | **+0.0000** |
| pomdp | **前瞻规划(H=6)** | +0.1598 | **+0.1717** | **+0.0119** |
| realistic | **前瞻规划(H=6)** | +0.4241 | **+0.4271** | **+0.0030** |

**怎么读**

1. **规则与短视的 Δ 精确为 0**：它们只用**单步** `preview()` 做决策，
   而单步评估不涉及未来起伏；侦察机是否 active 也不改变它们的动作
   （它们的判据是"能不能满足探测要求"，与截获无关）。
   ⇒ **§12.5、§12.6、§11C.5 里已汇报的这些数字无需修正**。
2. **只有会向前推演的策略受影响**（前瞻规划）。方向上，修复后反而**略好**
   （+0.0119 / +0.0030）——即"偷看未来"在本场景下并没有带来好处，
   反而因为相信了未来的强干扰而做出更保守的前期决策。
   量级很小（相对种子标准差），**不应**解读为"泄漏有害"。
3. **这不改变一个基本事实**：泄漏是真实存在的，任何**新**接入的策略
   （尤其是会 rollout、或把截获/暴露纳入目标的）都会受影响。
   因此修复是必需的，而不是"反正结果没变所以可以留着"。
4. ⚠️ `keep_truth` 只用于**构造"先知"上界基线**与量化历史偏差。
   任何正式实验都不得使用它——`BeliefOptions.validate()` 不会阻止它，
   因为它是审计工具，需要人工纪律。

---

## 11B. 实体模型、坐标系与多平台场景（v4.1）

### 11B.1 升级前的场景结构有什么问题

v4.0 及之前，场景真值是 `Simulator` 上几个平行列表加一堆**聚合值**：

```
sim.radar        # 单个雷达
sim.targets      # 目标列表
sim.interceptors # 侦察机列表
sim.jammers      # 干扰机列表
```

观测里用的是"最近目标距离""最远目标距离""最小 RCS""最近侦察机距离"。
这套表示法在单雷达/双目标下够用，但撑不起多平台，有两个硬伤：

1. **没有身份**。聚合值回答不了"**是哪个**目标最近""哪个干扰源在压制**哪部**雷达"。
   多雷达场景下连"这个距离属于哪一对实体"都无法表达。
2. **没有方向与时刻**。距离对称，但方位、俯仰、视线方向都是有向的；
   而"此刻"与"上一步"的几何量也没有任何显式区分。
3. **距离计算被抄了 4 遍**：`models/target.py`、`models/interceptor.py`、
   `models/jammer.py` 各写一份 `math.hypot(self.x - x, self.y - y)`，
   `strategy/anti_jam.py` 又直接调了一次 `math.hypot`。
   同一个量在多处实现，是"某处改了别处没改"的经典温床。

### 11B.2 统一实体模型（`models/entity.py`）

四类实体（Radar / Target / EnemyInterceptor / Jammer）现在都继承 `SceneEntity`，
统一具备：

| 能力 | 说明 |
| --- | --- |
| 唯一 ID | `entity_id`（雷达 `radar_id`、目标 `target_id`、侦察机 `interceptor_id`、干扰源 `jammer_id` 的统一别名）；**跨类型唯一**，重复直接报错 |
| 三维位置 | ENU 下的 `x, y, z`（米）。`z` 为新增，默认 0 |
| 三维速度 | `velocity` 属性返回 `Vec3`。字段名保持历史命名（目标/干扰/侦察机用 `vx,vy,vz`，雷达用 `velocity_x,velocity_y,velocity_z`），由 `VELOCITY_FIELDS` 适配 |
| 航向/姿态 | `heading_deg`（自正北顺时针）、`pitch_deg`（抬头为正）、`roll_deg`（右滚为正），`attitude` 属性返回 `Attitude` |
| 时间戳 | `timestamp_s`：**该实体状态所对应的时刻**，随 `advance(dt)` 前进 |
| 平台归属 | `platform_id`（可选）：把多个实体绑定到同一载具，例如"某架飞机 + 它带的干扰吊舱" |
| 位姿对象 | `pose` 属性返回 `Pose(position, velocity, attitude, time_s, entity_id)` |
| 唯一距离实现 | `range_to(x, y)`（二维，兼容旧接口）与 `range_to_entity(other)`（三维） |
| 有向关系 | `relation_to(other)` → `GeometricRelation` |

**为什么不重命名字段**：`Target(target_id=..., x=..., y=...)` 这种写法散布在配置、
脚本、测试和外部工具里。把 `target_id` 改名成 `entity_id` 会让它们**全部失效**，
而收益只是一个更好看的名字。因此基类提供**只读别名**，字段名原样保留。

**为什么用"类属性默认值"而不是 `@property`**：四个子类都是 `@dataclass`，
会在类体里把这些名字声明成数据字段，从而遮蔽基类属性。用 `@property` 时
"读到的是字段还是描述符"取决于子类有没有声明，非常容易出微妙 bug；
普通类属性没有这个歧义，而且允许**分步迁移**（先挂基类，再逐个补字段）。

### 11B.3 统一坐标与几何（`engine/geometry.py`）

这是一个**叶子模块**（只用标准库，不 import 工程内任何模块，
与 `engine/equations.py` 同一约定），因此任何层都可以引用而不产生环。

#### 坐标系约定（全工程统一，禁止各模块自定）

| 项 | 定义 |
| --- | --- |
| 直角坐标 | **ENU**：`x` 东、`y` 北、`z` 天；位置 m，速度 m/s |
| 球坐标 | `range`(m)、`azimuth`(度，自 **+y 正北** 起**顺时针**为正，0°=北、90°=东)、`elevation`(度，水平面以上为正) |
| 姿态 | `heading` 自正北顺时针、`pitch` 抬头为正、`roll` 右滚为正 |
| 机体坐标 | `forward`(机头) / `right`(右) / `up`(上)，由姿态唯一确定 |
| 机体方位 | `bearing`：相对**本机机头**，0°=正前、+90°=右 |
| 机体系俯仰 | `elevation_body`：相对本机机体水平面 |

上一版配置里的 `x`/`y` 语义**完全不变**，新增的 `z` 默认 0。

#### 每条几何关系都带语义（`GeometricRelation`）

`relation(observer_pose, target_pose, time_s)` 返回一个**有向**关系对象，
字段自带"谁相对谁、哪一刻"：

```
observer_id, target_id, time_s          ← 语义本身
range_m, range_rate_mps, closing        ← 对称量（这一对实体的属性）
los_enu                                  ← observer → target 的单位向量
azimuth_deg, elevation_deg               ← 绝对方位/俯仰
bearing_deg, elevation_body_deg          ← 相对 observer 机体的方位/俯仰
```

⚠️ **哪几个量在交换双方时会变，必须记准**（这里踩过一次坑）：

| 量 | 交换 `A→B` 与 `B→A` |
| --- | --- |
| `range_m` | **不变** |
| `range_rate_mps` | **不变**（它是**距离变化率** d\|R\|/dt，属于这一对实体） |
| `closing` | **不变** |
| `los_enu` | 取反 |
| `azimuth_deg` | 相差 180° |
| `elevation_deg` / `elevation_body_deg` | 变号 |
| `bearing_deg` | 各自依赖自己的航向，无固定关系 |

早期版本在文档里把径向速度写成"符号相反"，那是**错的**：
按定义 `v_r = (v_target − v_observer)·u(observer→target)`，
反向时 `(v_observer − v_target)·(−u)` 与原式恒等。
有向的是**相对速度向量**，而径向速度是它在连线上的投影，连线翻转时投影值不变。
现在这条约定由 `tests/test_entity_geometry.py` 直接钉住。

#### 距离计算为什么必须继续用 `math.hypot`

`math.hypot` 用的是防溢出缩放算法，与 `math.sqrt(dx*dx+dy*dy)` **不是一回事**。
实测在 5 万个随机构型里有 **8280 个（16.6%）** 最低位不同：

```
hypot 与 sqrt(dx²+dy²) 不等：8280 / 50000
```

最低位一变，Pd / 能耗 / 奖励会跟着漂，旧实验就不再逐位可复现。
因此几何模块**一律继续用 `math.hypot`**，并且二维路径直接调两参版本、
刻意不绕道"补一个 z=0 再走三维"（虽然实测两者在 20 万样本上逐位一致，
但不把正确性建立在浮点实现细节上）。

### 11B.4 场景注册表（`engine/scene.py`）

`Simulator` 现在额外持有 `sim.scene`，它是**同一批实体对象**的统一视图
（不是副本——`sim.scene.by_id("TGT1").x` 改了会同步反映到 `sim.targets`）。

```
sim.scene.entities                      # 全部实体（雷达→目标→侦察机→干扰源）
sim.scene.by_id("RADAR_A")              # 按 ID 取；不存在直接抛错
sim.scene.count_by_kind()               # {'radar':2,'target':3,'interceptor':2,'jammer':2}
sim.scene.of_kind(KIND_TARGET)          # 按类型取，可只取活跃的

sim.relation("RADAR_A", "TGT_HIGH_FAST")             # 单条有向关系（推荐入口）
sim.radar_target_relations()                          # 所有雷达→所有目标
sim.scene.relations_between(KIND_RADAR, KIND_JAMMER)  # 任意两类之间
sim.scene.all_relations()                             # 全部有向关系

sim.scene.nearest("RADAR_B", KIND_TARGET)             # 派生量（不是场景真值）
sim.scene.farthest("RADAR_A", KIND_TARGET)
sim.scene.distance_matrix()                           # 对称距离矩阵
```

**聚合值降级为派生视图**：`nearest` / `farthest` / `min_rcs_target` 仍然存在
（环境的 12 维观测在用），但它们现在是**由显式关系算出来的**，
而不是场景真值本身。多平台场景下"最近目标"必须说明是**哪部雷达的**
最近目标——`sim.scene.nearest(observer_id, kind)` 因此要求显式给出观察者。

**时间同步是硬约束**：每个实体带 `timestamp_s`，`relation()` 会校验双方时间戳
与查询时刻一致，不一致抛 `TimeSyncError`，并附上两个具体时刻。
"两个不同时刻的平台之间没有良定义的几何关系"——这是错误而不是近似。
`Scene.assert_time_synchronized()` 可整体校验，`export_scene.py` 每步都调它。

### 11B.5 多平台场景配置格式

**单雷达（旧格式，继续支持）**：

```json
{
  "scenario_name": "lpi_radar_power_v1",
  "radar": { "radar_id": "RADAR1", "x": 0.0, "y": 0.0, ... },
  "targets": [ ... ], "interceptors": [ ... ], "jammers": [ ... ]
}
```

**多平台（新格式）**：把 `radar` 换成 `radars` 列表，**其余段落名字不变**。
列表**首部即主雷达**（物理评估与功率控制只针对它）。

```json
{
  "scenario_name": "multi_platform_v1",
  "radars": [
    { "radar_id": "RADAR_A", "x": 0.0,  "y": 0.0,   "z": 0.0,
      "heading_deg": 45.0, "platform_id": "SITE_ALPHA", ... },
    { "radar_id": "RADAR_B", "x": 25000.0, "y": 12000.0, "z": 300.0,
      "velocity_x": -5.0, "velocity_y": 3.0,
      "heading_deg": 120.0, "pitch_deg": 1.5, "platform_id": "SITE_BRAVO", ... }
  ],
  "targets": [
    { "target_id": "TGT_HIGH_FAST", "x": 6000.0, "y": 0.0, "z": 3000.0,
      "rcs_m2": 2.0, "vx": -10.0, "vz": -2.0,
      "heading_deg": 270.0, "pitch_deg": -5.0, "platform_id": "AIRCRAFT_1" },
    ...
  ],
  "interceptors": [ { "interceptor_id": "ESM_AIR", "z": 8000.0, ... } ],
  "jammers":      [ { "jammer_id": "JAM_ESCORT", "z": 2600.0,
                      "platform_id": "AIRCRAFT_1", ... } ]
}
```

所有实体通用新增字段（全部有默认值，**旧配置无需改动**）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `z` | 0.0 | 高度（ENU 天向），米 |
| `vz`（雷达为 `velocity_z`） | 0.0 | 垂直速度，m/s |
| `heading_deg` | 0.0 | 航向，自正北顺时针 |
| `pitch_deg` | 0.0 | 俯仰，抬头为正 |
| `roll_deg` | 0.0 | 横滚，右滚为正 |
| `timestamp_s` | 0.0 | 状态对应时刻 |
| `platform_id` | `""` | 平台归属 |

**未知字段会明确报错**（而不是静默忽略）：

```
配置段 'targets' 出现未知字段 ['vel_z']；Target 支持的字段为 ['is_active', ...]
```

因为静默忽略会让"某个平台的姿态没生效"这种现象极难定位。

### 11B.6 内置多平台回归场景

`config/multi_platform_scenario.json` = **2 雷达 + 3 目标 + 2 侦察机 + 2 干扰源**，
9 个实体，72 条有向关系。它刻意包含几种几何上"不好看"的构型：

* 高空高速目标（z=3000 m，俯冲）与低空慢速目标（z=120 m）并存；
* 两架飞机（`AIRCRAFT_1`）各自带着目标与干扰吊舱，用 `platform_id` 绑定；
* 空中侦察机（`ESM_AIR`，z=8000 m，在运动）与地面侦察站并存；
* 第二部雷达 `RADAR_B` 自己在运动且带俯仰角，使"最近目标"在两部雷达看来**不同**。

实测输出（节选）：

```
场景 multi_platform_v1 @t=0s：2 雷达 / 3 目标 / 2 侦察机 / 2 干扰源；
实体总数 9，有向关系 72 条

[RADAR_A → TGT_HIGH_FAST] @t=0s  距离 6708.2 m  方位 +90.00°(绝对) / +45.00°(机体)
                                 俯仰 +26.57°  径向速度 -9.84 m/s（接近）
[RADAR_A → TGT_LOW_SLOW]  @t=0s  距离 4001.8 m  方位  +0.00°(绝对) / -45.00°(机体)
                                 俯仰  +1.72°  径向速度 +5.00 m/s（远离）
[RADAR_B → TGT_HIGH_FAST] @t=0s  距离 22633.8 m 方位 -122.28°(绝对) / +117.56°(机体)
                                 俯仰  +6.85°  径向速度 +5.55 m/s（远离）
```

手算核对：`hypot(6000, 0, 3000) = 6708.2` ✓；
`azimuth = atan2(6000, 0) = 90°`（正东）✓；
`elevation = asin(3000/6708.2) = 26.57°` ✓；
`bearing = 90° − 45°(航向) = 45°` ✓；
`v_r = (−10,0,−2)·(0.894,0,0.447) = −9.84`（负=接近）✓。

⚠️ **本阶段只验证几何与实体关系**：多雷达**协同决策**未实现，
物理评估（探测/截获/干扰）与功率控制仍然只针对主雷达 `RADAR_A`。

### 11B.7 场景快照导出（`export_scene.py`）

```bash
python export_scene.py --config config/multi_platform_scenario.json \
    --policy fixed --max-steps 10 --out-dir output/scene_multi
python export_scene.py --pairs "RADAR_A>TGT_HIGH_FAST,RADAR_B>TGT_ESCORT"
```

输出四个文件：

| 文件 | 内容 |
| --- | --- |
| `scene_entities.csv` | 每行 = 某时刻的某实体：id / 类型 / 平台 / t / x,y,z / vx,vy,vz / speed / heading,pitch,roll |
| `scene_relations.csv` | 每行 = 某时刻的某一对**有向**关系：observer / target / t / 距离 / 方位 / 俯仰 / 机体方位 / 径向速度 / 是否接近 / 视线向量 |
| `scene_initial.json` | t=0 完整快照（实体 + 关系 + 坐标体系声明） |
| `scene_final.json` | 末步完整快照 |

脚本末尾会**回读校验** CSV 行数是否等于 `实体数 × (步数+1)`，
不一致直接以非零码退出——避免"导出成功但少写了一段"这种静默错误。

### 11B.8 测试与验收

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_entity_geometry.py`（53 项） | 坐标变换往返/正交性、距离**逐位**对称、运动更新闭式解、时间同步、多实体索引、关系语义、向后兼容 |
| `tests/test_multiplatform_scene.py`（21 项） | 多平台构型、实体索引与旧列表共享对象、关系计数、派生聚合与显式关系一致、CSV/JSON 导出与回读、配置校验 |
| `verify_v4.py` §10 | 端到端几何验收（距离对称、坐标往返、时间同步报错、多平台构成、索引一致） |
| `tests/test_sensor_layer.py`（59 项） | 六种缺失原因可区分 + 判定顺序、真值不泄漏、噪声不改真值（逐位）、测量字段完整性、虚警无真值、被动传感器无距离、遮挡几何、更新周期语义 |

---

## 11C. 分层测量仿真：真值世界 → 传感器 → 测量 → 观测（v4.2）

### 11C.1 三层数据字典（**先看这张表，其余都可以慢慢读**）

这是本工程里最容易搞混的地方，因此放在最前面。**三层数据严格分离**，
混用会导致"以为在评测传感器、其实在评测真值"这类无法察觉的错误。

| 层 | 类型 / 位置 | 内容 | 谁能读 |
| --- | --- | --- | --- |
| **① 仿真真值** | `Scene` 实体、`StepResult` | 实体真实位置/速度/姿态；真实 Pd、Pint_eff、暴露、能耗 | 仿真器、指标、**评测脚本** |
| **② 传感器测量** | `MeasurementRecord` | 带噪声/协方差/时间戳的**估计值**，以及"这一帧没有数据"的**原因** | 传感器、融合层、**评测脚本**（算误差） |
| **③ 算法可见输入** | `FusedObservation.vector` | 定长向量：候选 ID 槽位 + 自状态 + 缺失统计 | **决策算法**（规则 / DQN / 前瞻） |

关键纪律（都有测试钉住）：

* `MeasurementRecord.truth_id` / `truth_*` 与 `Sensor.truth_of_candidate()`
  属于**评测专用通道**；
* `sensor/fusion.py` 打包观测时**不读**上述任何字段，
  并且每次打包都会跑 `_assert_slot_clean()`；`assert_no_truth_leak()`
  可对任意结构递归扫描；
* `is_false_alarm` 同样**不进**观测向量——算法若能看见这个标记等于开了上帝视角；
* `tests/test_sensor_layer.py::TestNoTruthLeak` 把这件事变成可执行断言，
  其中一个测试会把 `truth_of_candidate` 换成"一调用就抛错"的桩，
  确保打包路径根本不碰它。

### 11C.2 为什么必须把「没有数据」拆成多种原因

v4.0 的 POMDP 层用一个 `dropout_prob` 代表全部观测缺失。这在工程上是**错的**：
不同原因对应完全不同的物理含义、不同的统计归属、不同的对策。

| 原因码 | 含义 | 维度 | 工程对策 |
| --- | --- | --- | --- |
| `out_of_fov` | 目标不在视场（方位/俯仰超限） | 空间-指向 | 转雷达 / 调整扫描 |
| `beyond_range` | 超出作用距离或落在近界盲区 | 空间-能量 | 提功率 / 降门限 |
| `occluded` | 视线被障碍物切断 | 空间-环境 | 换位置（**功率无用**） |
| `not_updated` | 尚未到该传感器更新时刻 | 时间 | 等下一帧（旧值仍有效） |
| `missed_detection` | 通过全部检查但本帧概率未命中 | 概率 | 下一帧可能就有 |
| `sensor_unavailable` | 传感器关机 / 故障 | 可用性 | 换传感器 |

**判定顺序是语义的一部分**，不可随意调换：

```
1 SENSOR_UNAVAILABLE → 2 NOT_UPDATED → 3 BEYOND_RANGE
→ 4 OUT_OF_FOV → 5 OCCLUDED → 6 MISSED_DETECTION → 7 DETECTED
```

* 把 `MISSED_DETECTION` 放前面，会把"目标在视场外"错记成"概率丢帧"，
  消融实验直接失效（关掉 dropout 却没有任何变化）；
* `NOT_UPDATED` 必须排在几何判定**之前**：否则会出现
  "没到更新时刻却报出精确距离"的矛盾——那是拿当前真值去算几何了；
* `tests/test_sensor_layer.py::TestNoDataReasons` 对顺序有专门断言。

### 11C.3 每个传感器自己的四类属性

| 属性 | 配置项 | 说明 |
| --- | --- | --- |
| 作用范围 | `min_range_m` / `max_range_m` | 近界盲区 + 远界 |
| 视场 | `az_fov_deg` / `el_fov_deg` | 相对自身机头的半角；随姿态一起转 |
| 扫描/更新周期 | `update_period_s` | 用 `floor(t/period)` 判据，**与步长解耦** |
| 误差模型 | `range_sigma_rel`/`range_sigma_abs_m`/`az_sigma_deg`/`el_sigma_deg`/`range_rate_sigma_mps` | `σ_R = R·rel + abs`；角度误差与距离无关 |
| 检测模型 | `snr50_db` / `pd_slope_db` | ROC logistic，复用雷达方程 |
| 虚警 | `false_alarm_rate` | 每帧虚警概率 |
| 可用状态 | `available` | 关机/故障 |

两种传感器**物理上本来就不同**，不能统一：

* `RadarSensor`（主动）：测**距离、方位、俯仰、径向速度**，由雷达方程算 SNR → Pd；
* `EsmSensor`（被动）：**只有方位/俯仰，没有距离量测**。单站被动测距物理上做不到
  （需要多站时差/相位差或平台机动），因此它的距离维标准差是 `inf`，
  融合层必须能正确处理"有些测量没有距离"。观测槽位里有 `has_range` 标志区分。

### 11C.4 三组对照：一条干净的信息阶梯

| 组 | `observation_mode` | 观测来源 | 可见性约束 | 噪声/漏检/虚警 | 维度 |
| --- | --- | --- | --- | --- | --- |
| ① 全真值 | `full` | 仿真真值 | 无 | 无 | 12 |
| ② 理想测量 | `ideal` | 传感器测量 | **有** | **无** | 53 |
| ③ 真实测量 | `realistic` | 传感器测量 | 有（同②） | **有** | 53 |

阶梯的意义：**①→② 的落差是"信息可得性"的代价，②→③ 才是"测量不完美"的代价。**
把两者混在一起报，就说不清性能下降该怪谁。

实现上 `ideal` 与 `realistic` **共享完全相同的可见性判定代码路径**
（同一个 `Sensor.observe`），差异只由 `sensor/config.py::apply_noise_scale` 控制：
`scale=0` 时把全部 σ 与虚警率置零，并打开 `force_detection`
（理想传感器只要几何可见就一定检测到）。
如果改成"另写一个无噪声的传感器类"，两条路径迟早漂移，对照就失去意义。

### 11C.5 实测结果：信息阶梯落差分解

5 个种子（42/7/13/21/33），初始条件域随机化开启，`evaluate_observation_modes.py`：

| 组 | 规则功率控制 | 逐档贪心(短视) | DQN | 固定功率基线 | 随机策略 |
| --- | --- | --- | --- | --- | --- |
| ① 全真值 | **+0.4636**±0.1244 | **+0.4949**±0.0832 | +0.4739±0.0394 | −3.1954 | −0.5866 |
| ② 理想测量 | **+0.4636**±0.1244 | **+0.4949**±0.0832 | +0.3901±0.0477 | −3.1954 | −0.5866 |
| ③ 真实测量 | +0.4313±0.0674 | +0.4227±0.0747 | **+0.4385**±0.0932 | −3.1954 | −0.5866 |

（DQN 满足率：① 0.9574、② 0.9082、③ **0.9541**；③ 组里 DQN 的满足率 0.9541 **高于**规则 0.9475 与短视 0.9377。）

**落差分解：**

```
规则功率控制：  ①+0.4636 →②+0.4636（Δ +0.0000）→③+0.4313（Δ −0.0322）
逐档贪心(短视)：①+0.4949 →②+0.4949（Δ +0.0000）→③+0.4227（Δ −0.0722）
DQN：          ①+0.4739 →②+0.3901（Δ −0.0838）→③+0.4385（Δ +0.0484）
固定功率基线：  ①−3.1954 →②−3.1954（Δ +0.0000）→③−3.1954（Δ +0.0000）
随机策略：      ①−0.5866 →②−0.5866（Δ +0.0000）→③−0.5866（Δ +0.0000）
```

**怎么读这个结果（重要，别读错）**

1. **①→② 落差为 0，不是 bug，而是一个真实结论**：
   这两个策略**只用**了"目标距离 + 目标 RCS + 自身剩余能量"——
   而这些恰好都是**真实雷达测得到**的量。所以即使剥夺了真值，
   它们需要的信息一个都没少。换句话说：
   **v3.1/v4.0 里"给智能体真值"这个不现实假设，对这两个策略其实没有帮助**，
   因为它们从没用过那些"不该知道"的量。
2. **②→③ 的落差（−0.032 / −0.072）才是测量不完美的代价**：量测噪声、
   概率漏检、虚警、更新周期一起作用的结果。
3. **随机策略三组完全相同**——它根本不读观测，这是一个有用的对照：
   它证明三组之间的差异确实来自观测通道，而不是别处的偶然差异。
4. ⚠️ **不能把①→②=0 推广成「真值假设无所谓」**。会用到那些量的策略
   （例如读取完整仿真器做推演的前瞻规划，或未来接入 ESM 方位的协同策略）
   在①→②之间会出现明显落差。本表只说明：
   *当前这两个单目标、无记忆控制器*对不可观测信息没有依赖。
5. ⚠️ **DQN 那一行不能按同样的方式读**。三组 DQN 是**三个分别训练的模型**
   （观测维度 12 vs 53，必须各自训练），所以它的跨组落差里混着
   「观测变化」与「两次独立训练」两个因素。实测：
   * ①→② 落差 −0.0838：既有信息变少（侦察机位置、真实暴露、真实 Pint 都不再可见），
     也有「53 维输入本身更难学」的成分，两者没有分离；
   * ②→③ 差值 **+0.0484**，但差值标准误 ≈0.0468，**t≈1.03，不显著**。
     因此**不能**说「噪声训练更好」，也不能说「理想测量更好」——
     两组在 5 个种子上不可分辨。能说的只是：
     **在理想（无噪声）测量下训练并不比在真实测量下训练更好，也没有更差**，
     训练期噪声没有带来可分辨的退化。
   要严格归因 DQN 的跨组差异，需要固定随机种子做多次独立训练并给出训练方差，
   本版**没有**做这件事。

### 11C.6 测量级统计验证（`analyze_measurements.py`）

```
python analyze_measurements.py --mode realistic --range-profile --period-sweep
```

**(a) 误差随距离的变化**——两遍标定，避免一个统计陷阱：

* **A 遍（误差标定，`force_detection=True`）**：只要几何可见就产出测量。
  实测：σ_R 从近距箱 42.7 m 增长到远距箱 193.4 m；
  **σ_方位 全箱基本恒定（0.40°~0.55°）**，与"角度误差来自波束、不随距离变化"的模型预期吻合。
  ⚠️ σ_R 的增长倍数（4.5×）**小于**名义距离比（9×），
  原因是每个分箱内部本身跨越 6 km、样本量只有十几个（σ 估计有 ~20% 波动），
  因此**这张表只能支持"σ_R 随距离增长、σ_方位 不随距离变化"两条定性结论**，
  不能用来精确标定 rel/abs 系数。
* **B 遍（检测标定，正常检测器）**：检测率随距离 1.0000 → 0.3333 → 0.0000，
  缺失原因全部是 `missed_detection`——这正是低截获雷达"越远越难发现"的物理表现。

> **踩坑记录**：第一次跑距离剖面时 5 个箱里只有 1 个有样本。原因有两层：
> ① 只用一遍（正常检测器）时距离一远 Pd→0，**根本没有测量**可以拿来算误差；
> ② 目标沿 +x（正东）摆放，而雷达机头朝正北、视场只有 ±60°，
> 于是全程 `out_of_fov`，统计到的其实是另一个目标。
> 修法是"两遍标定 + 沿机头方向扫"。这两个错误都不会报错，只会静默给出错误结论。

**(b) 逐原因缺失率**（一次 5 种子 realistic 运行实测）：

```
总判定 168 次，检测成功 67 次，整体检测率 0.3988
  超出传感器作用距离      beyond_range        46 次  0.2738
  目标不在传感器视场内     out_of_fov          10 次  0.0595
  本帧检测遗漏（概率性丢测） missed_detection    45 次  0.2679
  按维度：概率=45，空间-指向=10，空间-能量=46
```

三种原因在**同一次运行**里同时出现且被分开统计——这正是"不能用一个
dropout 概率代表全部情况"的具体体现。

**(c) 更新周期**——**扫描节奏与输出节奏必须分开看**：

```
SENSOR_RADAR1  扫描 58 次，扫描间隔 1.0000s ± 0.000000s   ← 周期是否正确的判据
               出数 55 次，出数间隔 1.0556s ± 0.231212s   ← 含漏检跳帧，属正常
SENSOR_ESM1    扫描 58 次，扫描间隔 1.0000s ± 0.000000s
               出数 18 次，出数间隔 2.5882s ± 1.872793s
```

> **踩坑记录**：早期版本只统计"测量时刻"，把一次漏检误报成
> "周期 1.21±0.43 s"，看起来像扫描节奏乱了——实际是**统计口径错了**。
> 现在用 `Sensor.scan_times`（与是否检测到目标无关）判周期，
> 实测标准差精确为 0，才真正验证了 `update_period_s` 生效。

**(d) 离散时间与连续周期的关系**：仿真只在整数秒求值，
因此 `update_period_s=2.5` 实际表现为 **0, 3, 5, 8**（而非 0, 2.5, 5.0, 7.5）。
这是必然结果而非 bug；换成 dt=0.5 s 的网格就会落在 0, 2.5, 5.0, 7.5。
判据用 `floor(t/period)` 是否增加，因此**与步长解耦**，
不会像"按步数取模"那样把节奏写死。

### 11C.7 测量记录导出

`analyze_measurements.py` 输出（默认**含**真值与误差列，属评测通道）：

| 文件 | 内容 |
| --- | --- |
| `measurements.csv` | 测量记录：`time_s` / `sensor_id` / `candidate_id` / 距离·方位·俯仰·径向速度 / σ / 协方差对角 / `confidence` / `snr_db` / `rcs_est_m2` / `jam_ratio_est` / 虚警与新鲜度标记（+ 真值与误差列） |
| `outcomes.csv` | 逐 (传感器, 目标) 判定：`status` / `reason` / `reason_cn` / `reason_dimension` / `detected`（+ 真值列） |
| `measurements.json` | 上述内容 + 完整统计（误差分箱、逐原因缺失、虚警、更新序列） |
| `range_profile/measurements_forced.csv` | A 遍（误差标定）原始记录 |
| `range_profile/measurements_normal.csv` | B 遍（检测标定）原始记录 |

加 `--no-truth-columns` 导出**不含真值**的版本，可直接给算法/外部工具消费。
`MeasurementRecord.to_dict()` 默认同样是 `include_truth=False`。

### 11C.8 v4.0 POMDP 路径的处置（诚实说明）

v4.0 的 `observation_mode="pomdp"`（`engine/observation_model.py` 的全局噪声/延迟/丢测模型）
**保留且未改动**，目的是让 v4.0 的实验结果仍可复现。

但要说清楚：**它的机制已经被"更正确地"实现在测量层了**。两者的区别是：

| | `pomdp`（v4.0） | `ideal` / `realistic`（v4.2） |
| --- | --- | --- |
| 噪声作用对象 | 全局状态量（距离/干扰/能量/暴露…） | **每个传感器各自**的原始量测 |
| 缺失原因 | 单一 `dropout_prob` | **六种**可区分原因 |
| 可见性 | 无（全向、无限远） | 作用距离 / 视场 / 遮挡 / 周期 |
| 参数量纲 | 各量独立 σ | `σ_R = R·rel + abs`（随距离增长） |
| 多传感器 | 不支持 | 支持，含主/被动差异 |
| 真值隔离 | `info` 中可选暴露 | 三层结构 + 递归泄漏断言 |

**新工作请用 `realistic`**；`pomdp` 仅用于复现 v4.0 的既有数字。

### 11C.9 本阶段**没有**做到的（写论文时必须写明）

* **没有跟踪/数据关联**：融合层只是"按置信度排序取前 K 条"，**不是**最近邻/JPDA/MHT。
  同一目标在不同步可能落在不同槽位，航迹编号不跨时间稳定。不得称其为"跟踪器"；
* **没有航迹外推**：扫描帧会作废旧记忆，目标被漏检一次即彻底丢失，
  不会用速度推算位置（避免幻影航迹的保守取舍）；
* **遮挡是简化几何**：只用 AABB 与球，**没有**地形高程、地球曲率、大气折射、
  多径与绕射；`is_los` 之外没有更细的通视建模；
* **天线方向图未接入姿态**：`heading/pitch/roll` 目前只影响**视场**与机体方位，
  不影响发射/接收增益（增益仍是主瓣/旁瓣二值判决）；
* **协方差是对角阵**：方位-俯仰的交叉耦合未建模；
* **虚警是泊松式独立事件**：没有做杂波图/CFAR 门限随环境自适应；
* **多传感器没有做时空配准与融合估计**：当前只是把各传感器的测量并排放进槽位，
  **没有**卡尔曼滤波/协方差加权融合；因此"多传感器"目前只增加信息条数，
  不产生最优融合估计；
* **ESM 仍假定能观测到雷达**：真实的被动侦收还有截获概率、天线扫描相遇概率、
  信号分选与识别等环节，本工程只建到"单程链路 + ROC"这一层。

---

## 11D. 环境校核与智能算法重新验证（v4.3）

### 11D.1 数据生成链：任何结论都要能追溯到明确环节

```
真实状态（Scene/Simulator，唯一真值）
   ↓ 只读
可见性（sensor：作用距离 / 视场 / 遮挡 / 更新周期 / 可用状态）
   ↓ 产生
测量（MeasurementRecord：带噪声、协方差、时间戳、候选 ID）
   ↓ 传输
通信（communication：生成/发送/到达时刻、延迟、丢包、带宽、过期）
   ↓ 消费
融合（fusion：关联 + 加权估计 + 溯源）
   ↓ 输入
决策（规则 / DQN / 集成 / AI 诊断，只读已到达的融合结果）
```

`validation/` 模块对**前五环**分别做检查，因此任何实验结果都可以回答
"它是在这条链的哪一环上产生的、那一环是否合格"。

### 11D.2 分层校核结果（`run_validation.py`）

实测（2 个种子，测量层 60 步、融合层 40 步）：**四层 29 项检查全部通过**。

| 层 | 通过/总数 | 抽查的实测值 |
| --- | --- | --- |
| 几何层 | 8/8 | 距离逐位对称；方位 0°=北 / 90°=东；俯仰 ±45°；机体正交误差 1.1e-16 |
| 测量层 | 5/5 | 距离误差均值 0.52 m（n=100+）；方位误差均值 0.19°；缺失原因可区分 2 种 |
| 通信层 | 11/11 | 延迟 3 s 的消息在 t=2 不可见、t=4 可见；投递率 0.685 vs 期望 0.7；平均延迟 0.5 s = 配置值；过期按时丢弃 |
| 融合层 | 5/5 | 融合位置误差 30.44 m；溯源字段完整；航迹连续性 1.0000；峰值航迹 1 条（目标 2 个） |

**「已验证」不是手写的**：`validation/report.py::build_trust_report` 从
`check_report` 推导——某层全部检查通过才进 `verified_modules`，
有失败项则降级为"部分验证"并列出失败 ID。手写清单迟早会和测试脱节。

### 11D.3 环境可信度报告

`run_validation.py` 生成 `output/validation/trust_report.md` / `.json`，四节结构：

1. **已验证模块**（由上表推导）
2. **尚未验证的假设**（9 条，如"误差独立零均值高斯"、"匀速直线无机动模型"、
   "通信无突发错误与重传"、"单站被动不参与位置更新"、"最近邻关联在密集目标下会串"）
3. **可支持的结论**（只在对应层已验证时列出）
4. **不可支持的结论**（7 条明令禁止的表述）

**外部校核的边界**（同时写进报告与本节）：允许比较通用雷达量测误差随距离的关系、
异步多传感器跟踪的融合误差量级、多雷达观测几何的结构性结论；
**禁止**据此声称验证了真实装备、真实战场性能或电子战效能。
本仿真的参数是**教学与研究用等效参数**，不与任何具体型号对应。

### 11D.4 通信三组对照：**协同收益尚未被证明**

`run_validation.py --communication` 跑「不共享 / 理想零延迟共享 / 受限共享」。
在双雷达场景下（2 雷达 + 3 目标 + 2 侦察机 + 2 干扰源）：

| 策略 | 平均航迹 | 航迹覆盖 | 新鲜度 | 送达共享测量 | 送达率 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| 不共享 | 1.000 | 1.000 | 0.9648 | 0 | — | — |
| 理想零延迟共享 | 1.000 | 1.000 | 0.9648 | 5270 | 1.0000 | 0.0000 s |
| 受限共享 | 1.000 | 1.000 | 0.9648 | 4197 | 0.8294 | 1.0016 s |

**必须如实说明**：共享测量确实到达并进入融合中心的输入
（理想 5270 条、受限 4197 条，受限组送达率 0.8294、平均延迟 1.0016 s
与配置一致），但**三组的航迹指标完全相同**。

这意味着**协同收益目前没有被证明**，也**不能**据此说"共享无用"。可能的解释有三条，
本阶段**尚未诊断**：

1. 本地雷达已经覆盖了远端雷达能看到的目标，共享没有新增信息；
2. 远端测量在融合中心被上游环节拦掉了（关联门限 / 时效检查 / 观测对象过滤），
   即"送达了但没被用上"；
3. 融合中心的航迹起始条件与场景目标数不匹配（当前峰值航迹 1 条，而场景有 3 个目标）。

**在诊断清楚之前，任何协同收益的结论都不成立。**
诊断入口已就绪：`TracksSnapshot` 里的 `n_kind_rejected` / `n_stale_rejected` /
`n_local_measurements` / `n_remote_measurements` 会告诉你测量走到哪一步被拦下。

### 11D.5 AI 诊断的证据边界（表述已收紧）

原表述"**AI 解释不会编造**"是**无法证明的强主张**，现收紧为：

> **AI 解释被限制在结构化证据范围内，并通过规则检查。**

含义上的差别很重要：不是声称"它不会错"，而是
①它**只能引用**结构化证据（测量 / 融合 / 决策 / 校核结果），
②有**规则检查**（发现码枚举、证据字段数值化、provider 降级路径）。
"未检测"的**原因区分**（遮挡 / 视场外 / 传感器未更新 / 漏检 / 通信未到达）
在本阶段的数据结构里**已经具备**（`NoDataReason` 六种原因 +
通信层的 `drop_reason`：`lost` / `expired` / `queue_full` / `link_down`），
但**AI 诊断层的取用与措辞尚未改造完成**——这是本版明确未完成的一项。

### 11D.6 本阶段**没有**做到的事

* **AI 诊断层尚未接入新的原因分类**：数据结构已备好，但
  `ai/rule_provider.py` 还没有把"通信未到达"等新原因变成独立的发现码；
* **协同收益未证明**：见 §11D.4，三组航迹指标相同且原因未诊断；
* **DQN / 集成 / 历史窗口未在融合输出上重新评测**：
  它们目前仍接测量层（§11C），**没有**接入 `fusion/` 的航迹输出，
  因此"旧简化环境 vs 新测量级环境"的对照只覆盖到测量层，未覆盖融合与通信；
* **`validation/` 的检查项有限**：几何/测量/通信/融合各 5~11 项，
  没有覆盖机动目标、密集目标、突发丢包等压力场景；
* **未使用任何公开数据做外部校核**：只写了允许与禁止的边界，
  没有实际执行外部对比。

---

## 11E. 多雷达协同感知跑通（v4.4）

### 11E.1 先修根因：为什么"数千条共享测量、航迹指标却不变"

按纪律，先诊断再改。诊断脚本输出的原始数据是决定性的：

```
SENSOR_RADAR_A  max_range =  5801 m   → 目标在 6708 / 4018 / 13032 m
SENSOR_RADAR_B  max_range =  3836 m   → 目标在 22634 / 26249 / 16470 m
SENSOR_RADAR_B  10 步 30 次判定，**全部 beyond_range**
```

**远端雷达一条测量都产生不了。** 多平台场景当初只为几何校核而建，
它的传感器作用距离由雷达方程在标称 18 W 下反解得到（约 4~6 km），
比目标距离小一个量级，从未做过可探测性校核。
所以"共享不改变航迹"与融合算法无关 —— **是场景配置错误**。

> 教训：`_radar_default_max_range()` 用**标称功率**反解作用距离，
> 而动作空间最高到 80 W。用它当"覆盖范围"会系统性低估。
> 凡是要参与性能实验的场景，都必须显式校核传感器包络。

### 11E.2 测量生命周期追踪（`fusion/lifecycle.py`）

每条测量有贯穿全链路的身份：

```
generated → sent → arrived → stale_rejected / kind_rejected / gate_rejected
          → associated → track_created / track_updated
```

逐步记录 `sensor_id`、`source_platform`、`candidate_id`、各阶段时刻、
`reject_reason`、`track_id`。`LifecycleLog.funnel()` 输出漏斗与
**远端测量利用率**，直接回答"某条远端测量在哪里被丢弃、为什么没进航迹"。

**这个工具立刻抓出了第二个真 bug**：漏斗里出现

```
age=4.000s > 3s   10 条
age=5.000s > 3s   10 条
...  一直到  age=28.000s > 3s
```

而链路配置的延迟只有 **1.2 s** —— 年龄 28 s 在物理上不可能。
根因：`CommBus.arrived()` 只按 `arrival <= now` 过滤、**从不消费**，
于是每一步都把自 t=0 以来的**全部历史消息重新投递一遍**。
远端测量因此堆积（179 条 vs 本地 9 条），远端利用率 0.134 是假象。

**修法**：新增 `CommBus.consume(dst, now)`（一次投递、记账不重复），
`arrived()` 保留为纯查询。修后远端利用率 **0.134 → 1.0000**。

### 11E.3 融合层升级为基础多目标跟踪器（`fusion/kalman.py` + `center.py`）

| 能力 | v4.3 旧 | v4.4 新 |
| --- | --- | --- |
| 状态 | 仅位置 | 位置 + 速度（6 维） |
| 预测 | 无 | 常速度 Kalman（连续白噪声加速度模型） |
| 关联 | 固定米/度门限 | **马氏距离门限**（预测协方差归一化）+ 角度兜底 |
| 速度 | 相邻位置差分 | Kalman 估计 |
| 漏检 | 直接记 miss | **coasting 短时保持**，靠预测外推 |
| track_id | 会重排 | **稳定**（单调递增） |

**第三个真 bug（也是"共享没收益"的直接原因）**：`update()` 原本对所有候选做
**一次性批量关联**。同一时刻两部雷达测到同一目标时，两条测量竞争同一条航迹，
一条胜出、另一条关联不上 → **起始重复航迹**；两条信息从未先后作用在
同一条航迹上，多传感器融合降低协方差的机制根本没发生。
表现为"共享后 RMSE 反而更差、受限共享比理想共享还好"这种自相矛盾的结果。

**修法**：改为**逐条顺序关联**——每处理一条就重新关联，
第二条测量会关联到刚被第一条更新过的航迹，两次 Kalman 更新真正合成。

### 11E.4 三类协同验证场景（`evaluate_cooperative_sensing.py`）

场景**程序化构造**（显式给定雷达位置、传感器作用距离、遮挡体），
避免再次出现 11E.1 那种配置错误。2 个种子、30 步。

> ⚠️ **下面这一组数字已被 v4.5 的跟踪器 bug 修复更新过**，
> 旧值（修复前）保留在 §11E.6 供对照。**请以本节为准。**

| 场景 | 指标 | 不共享 | 理想零延迟共享 | 受限共享 |
| --- | --- | --- | --- | --- |
| **A 共同可见** | 位置 RMSE | 129.51 m | **91.83 m（−29.1%）** | 107.27 m |
| | 航迹覆盖 | 0.840 | 0.960 | 0.940 |
| **B 遮挡** | 位置 RMSE | 无航迹 | 84.32 m | 106.57 m |
| | **航迹覆盖** | **0.000** | **0.760（+0.760）** | 0.540 |
| **C 通信受限** | 航迹覆盖 | 0.000 | 0.760 | 0.540 |
| | 送达率 / 平均延迟 | — | 1.0000 / 0 s | **0.6471 / 1.149 s** |

**结论（只在确实观察到改善时才声称收益）**：

1. **场景 A：共享确实降低估计误差** —— RMSE 129.51 → 91.83 m（**−29.1%**）。
   机制是多视角融合：两部雷达从不同方位测同一目标，
   顺序 Kalman 更新把两个独立信息合成，协方差显著收缩。
2. **场景 B：共享确实保持航迹连续** —— 不共享时本地雷达被遮挡，
   **完全没有航迹**（覆盖 0.000）；共享后覆盖升到 **0.760**。
   这是"共享把看不见的目标补上"的直接证据。
   注意此时 RMSE 的数值比较**没有意义**（不共享组无样本记 0），
   判据看覆盖与连续性 —— 脚本里有专门的分支处理这一点，
   避免输出"0 → 84.32 m = 无改善"这种与事实相反的结论。
3. **场景 C：通信条件确实吃掉一部分协同收益** ——
   覆盖 0.760 → 0.540（−22 个百分点），RMSE 84.32 → 106.57 m（+26.4%），
   送达率 0.6471、平均延迟 1.149 s。
   **协同效果必须与通信条件一起解释**，不能只报理想共享的数字。

⚠️ 所有指标用**离线真值**计算，只用于评测，不参与任何决策。
⚠️ 场景 C 的几何与 B 相同，因此 C 的"不共享/理想共享"两行与 B 相同，
这是设计使然；C 的增量信息在"受限共享"那一行。

### 11E.6 v4.5 修 bug 后协同数字的变化（**必须与 §11E.4 一起读**）

§11G.5 记录的跟踪器 bug（`misses` 恒为 0、航迹永不删除）会让**幽灵航迹永久累积**。
在单雷达场景里，幽灵航迹把"不共享"基线的误差抬高了，于是**共享的相对收益被高估**。
修复前后的对照：

| 场景 | 指标 | 修复前（v4.4 报告值） | 修复后（当前值） |
| --- | --- | --- | --- |
| A | 不共享 位置 RMSE | 206.00 m | **129.51 m** |
| A | 理想共享 位置 RMSE | 87.29 m | **91.83 m** |
| A | 相对改善 | −57.6% | **−29.1%** |
| B | 理想共享 航迹覆盖 | 0.967 | **0.760** |
| B | 受限共享 航迹覆盖 | 0.900 | **0.540** |
| C | 受限共享 送达率 / 延迟 | 0.6538 / 1.198 s | 0.6471 / 1.149 s |

**怎么读这张表**：

* **共享仍然有效、方向不变**（A 降误差、B 补盲、C 通信吃掉一部分收益），
  但**幅度明显缩小**：A 的相对改善从 −57.6% 降到 −29.1%；
* 覆盖类指标**下降**（0.967 → 0.760）不是"共享变差了"，而是
  **现在真的会删除航迹**：没有观测支撑的航迹按时删除，不再永久挂在航迹表里；
* **v4.4 已发布文档里的 −57.6% 不再成立**，凡引用该数字处（含本 README 旧稿、
  论文草稿）都应改为 −29.1%，或明确标注为"修复前的值"；
* 这一条也说明**审计类改动必须重跑受影响的全部实验**：
  修复本身只在 `fusion/`，却改变了协同评测的所有数字。


### 11E.5 本阶段**没有**做到的事

* **基于融合航迹的新观测模式尚未接入**：`fusion/` 的航迹输出还没有变成
  环境的一种 `observation_mode`，因此**规则策略与 DQN 仍未在"融合航迹输入"
  下重新评测**。用户要求的"单传感器测量输入 vs 融合航迹输入"对照**未完成**；
* **`diagnose_fusion.py` 尚未单独成脚本**：诊断能力已具备
  （`--diagnose` 打印漏斗 + `lifecycle_*.csv` 逐测量导出），
  但没有按用户要求拆成独立诊断脚本；
* **协同收益只在 2 个种子上验证**：样本量偏小，
  且**没有做统计显著性检验**，因此上面的"改善"是描述性的，
  不能声称统计显著；
* ~~**场景只覆盖单目标**~~：v4.5 P2 已补齐多目标压力测试（见 §11G），
  并且**恰恰是多目标压力测试暴露了跟踪器的一个严重 bug**（§11G.5）。
  ⚠️ 但 §11G 的结论**不能反向保证** §11E 的单目标结论更强——
  S2 就出现了"共享反而更差"的反例；
* **未做机动目标**：跟踪器是常速度模型，无机动检测与自适应 Q；
* **未使用任何公开数据做外部校核**（与 §11D 同一约束）。

---

## 11F. AI 对测量—通信—融合—航迹证据链的认知诊断（v4.5）

### 11F.1 为什么需要这一层

v4.2–v4.4 把「真值 → 可见性 → 测量 → 通信 → 融合 → 航迹」这条链修好了，
但 **AI 认知层还停在 v4.0 的信息水平**：它只看得到 `pd_min`、`snr_db`、
累计暴露这些**聚合标量**。于是它只能说出「观测退化」——

而 `观测退化` 在 v4.2 之后已经是**一个被拆开的词**：目标可能在视场外、
可能超距离、可能被遮挡、可能只是这一帧没扫描到、可能是传感器下线、
也可能是雷达探到了但没过检测门限。**这六种原因的处置完全不同**
（转雷达指向 / 换传感器 / 等下一帧 / 修链路），用同一个词概括等于没有诊断能力。

所以 v4.5 P1 的任务不是"加个新模型"，而是**把已经存在于仿真里的证据
喂给 AI，并且只喂它此刻真正拿得到的那部分**。

### 11F.2 结构化证据的四个新节

| 节 | 内容 | 关键约束 |
| --- | --- | --- |
| `measurement_state` | 每个传感器的可见性结论、六类"没有数据"原因计数与占比、候选/虚警计数、新鲜度、观测质量 | 只含**测量报告**，不含 `truth_id` 与真值距离/角度 |
| `communication_state` | 每条链路的送达率、平均/p95 时延、丢弃原因分布（丢包/过期/链路断/队列满） | 只统计**已到达**的消息；在途消息内容不可见 |
| `fusion_state` | 每条航迹的 `track_id`、状态（tentative/confirmed/coasting）、命中/漏检、本地与远端更新次数、协方差、新鲜度、溯源最近源 | 提供的是**估计值与协方差**，不是真值 |
| `cooperation_state` | 含远端贡献的航迹数、仅本地支撑的航迹数、远端测量到达/采用/被拒计数、远端利用率 | **不含任何真值收益指标**（RMSE 之类只存在于离线评测） |

**「不含未来」不只是"不写 key"**：`CommBus.arrived()` 本身是纯查询，
所以上下文构造走的是 `consume()` 的单次投递账本与 `measurement_time_s`
（测量时刻），而不是"按到达时刻看起来像新的"——延迟大的共享测量即使刚到，
新鲜度也仍然是低的。这条纪律和 §11A.10 的真值泄漏审计是同一套标准。

### 11F.3 发现码：把"没有数据"和"信息被吃掉"逐项拆开

v4.5 新增 17 个发现码（`FINDING_CODES` 共 40 个）：

| 类别 | 发现码 |
| --- | --- |
| 传感器看不到 | `TARGET_OUT_OF_FOV`、`TARGET_BEYOND_RANGE`、`TARGET_OCCLUDED`、`SENSOR_NOT_UPDATED`、`MISSED_DETECTION`、`SENSOR_UNAVAILABLE` |
| 信息被通信吃掉 | `COMM_PACKET_LOST`、`COMM_MESSAGE_EXPIRED`、`COMM_LINK_DOWN`、`COMM_QUEUE_FULL`、`REMOTE_MEASUREMENT_DELAYED` |
| 航迹层 | `TRACK_COASTING`、`TRACK_MAINTAINED`、`TRACK_FRAGMENTED`、`TRACK_UNCERTAIN`、`ASSOCIATION_AMBIGUOUS` |
| 协同贡献 | `REMOTE_SENSOR_CONTRIBUTION`、`COOPERATIVE_TRACK_RECOVERED` |

映射关系是显式的（`_MEASUREMENT_REASON_CODES` / `_COMM_DROP_CODES`），
六种 `NoDataReason` 与四种通信丢弃原因各自对应一个码——**不是**由一个
"观测退化"统一兜底。

**实测样例**（`realistic` 模式 + 受限共享，step 8）：

```
TARGET_OUT_OF_FOV          ← TGT1 不在 RADAR1 视场内
MISSED_DETECTION           ← 在视场内但这一帧没过门限
REMOTE_MEASUREMENT_DELAYED ← 远端测量到了但已过时效
REMOTE_SENSOR_CONTRIBUTION ← 本地航迹确实拿到了远端贡献
```

### 11F.4 两个新的只读接口

| 接口 | 回答的问题 |
| --- | --- |
| `/api/explain_track` | 这条航迹是怎么形成的？为什么在 coasting？不确定度从哪来？远端贡献占比多少？ |
| `/api/explain_cooperation` | 远端信息在哪些时刻真的进了航迹、多少被延迟/丢包吃掉？ |

两者都**只讲结构性事实**。像「共享把 RMSE 降低了 57.6%」这种**收益大小**
必须由离线评测用真值算，接口的输入里根本没有这类数字——**编不出来**，
这也正好让下面的证据校验有东西可查。

### 11F.5 证据校验与回退（针对远程 LLM）

远程 provider 的自由文本必须通过两道校验才被采信：

1. **发现码白名单**：文本里出现的发现码必须真的在 `FINDING_CODES` 里；
2. **数值/标识符可溯源**：文本里的数字与 `SENSOR_A` 这类标识符必须能在
   上下文里找到（`ai/evidence_check.py`）。

不通过则标记 `evidence_check_failed` + 记录 `evidence_violations`，
并**回退本地规则 provider**（`fell_back_to_rule=True`），
同时在结论里附加说明。实测：

```
输入： "延迟 1.2 s，发现码 FOO_BAR_BAZ，误差 999.0 m"
上下文：{"latency_mean_s": 1.2}
结果： passed=False
  - 发现码不在白名单：FOO_BAR_BAZ（可能是编造的代码）
  - 数值无法在上下文中溯源：999.0
```

**诚实措辞**：这一层保证的是「AI 解释被限制在结构化证据范围内，
并通过规则检查」。它**不是**"AI 解释不会编造"的证明——
校验只覆盖可列举的发现码与可解析的数值，语义层面的错误陈述仍可能漏过。
第一版校验还曾把上下文中真实存在的 `SENSOR_A` 误判为编造标识符，
现已由 `known_ids` 豁免修正（该回归由测试钉住）。

### 11F.6 AI 不控制任何东西

`ai/` 层**没有** `set_power`、`apply_overrides` 之类的写路径，
不修改雷达功率，也不修改融合器输出。`tests/test_ai_evidence.py`
用源码扫描（`inspect.getsource`）把这条纪律钉住——
诊断、解释、出报告可以，控制不行。

复现命令：

```bash
python -m unittest tests.test_ai_evidence -v      # 28 项：无真值泄漏 / 原因区分 / 证据校验 / 路由接线
python diagnose_ai.py --observation-mode realistic --share constrained
```

---

## 11G. 多目标压力测试（v4.5 P2）

v4.4 的协同结论是在**单目标**下得到的。单目标时最近邻关联几乎不会出错，
所以"协同有效"这个结论**从未经过关联层的压力考验**。本阶段补上这一课。

### 11G.1 四类压力场景

`multi_target_stress/scenarios.py`，每个场景都先声明"要暴露什么失效模式"，
跑之前就能说清"如果关联没问题应该看到什么"：

| 场景 | 几何 | 问题 |
| --- | --- | --- |
| **S1 两目标交叉** | 两目标相向 120 m/s，在本地雷达正前方 7 km 处交叉通过 | 交叉点附近会不会**换号**（ID switch）？ |
| **S2 密集编队（3 目标）** | 横向间距 200 m，而 7 km 处量测标准差约 93 m（约 2.1σ） | 航迹会不会**合并 / 互换 / 重复**？ |
| **S3 短时遮挡（4.0 s）** | 球形遮挡体切断本地视线，目标 150 m/s 穿过 | Track ID 是否保持（coasting）？ |
| **S3L 长时遮挡（9.4 s）** | 遮挡 9.4 s > `drop_after_misses=5` | 重先后是同一 ID 还是新航迹？ |
| **S4 虚警 + 异步远端 + 延迟** | 目标附近 σ=300 m 可控虚警（本地 0.6/帧、远端 0.3/帧）+ 远端 2.5 s 异步周期 + 1.2 s 基延迟 / 25% 丢包 | 虚警会不会**夺走真实航迹**？远端能否消歧？ |

三条实验纪律：

1. **物理参数一律从 `config/radar_scenario_v1.json` 复制**，只覆盖位置、朝向、
   速度、传感器作用距离/视场/更新周期、虚警率与遮挡体。
   `tests/test_multi_target_stress.py` 逐字段断言雷达的
   `tx_power_w / peak_gain_db / noise_figure_db / snr_db / energy_budget_j …`
   与基础配置完全一致 —— **"压力"来自几何与测量条件，不是来自改物理**。
2. **遮挡半径由目标遮挡时长反解**，不用远场近似猜：
   球心到视线距离 `d(y) = x_o·|y| / sqrt(x_t² + y²)`，令 `d(y) = r` 得
   `|y| = r·x_t / sqrt(x_o² − r²)`，于是 `r = x_o·|y_half| / sqrt(x_t² + y_half²)`。
   （第一版按 `asin(r/x_o)` 的远场锥给半径，把 9.4 s 的遮挡写成了 4.0 s——
   球离雷达只有 4 km 而目标在 7 km，近似在这里根本不成立。）
3. **四个场景统一把检测设为确定性**（几何可见即检测到）。理由：本模块要归因的是
   关联/遮挡/虚警/通信这四类失效，而 7 km、RCS 2 m² 下基线检测概率约 0.5，
   随机的漏检会把"被遮挡"和"这一帧没探到"混在一起。概率漏检已在
   v4.2 测量层单独量化，不在这里重复。

### 11G.2 关联层审计：回答"这一帧为什么关联错了"

`fusion/lifecycle.py` 新增 `AssociationCandidate`，把**每一次候选评估**留痕：

| 字段 | 含义 |
| --- | --- |
| `association_candidate_tracks` | 本帧评估过的**全部**候选航迹（含被门限拒绝的） |
| `mahalanobis_sq` / `gate_mahalanobis_sq` | 门限距离与当时生效的门限 |
| `az_diff_deg` / `el_diff_deg` | 角度兜底门限的残差 |
| `reject_reason` | `mahalanobis_gate` / `azimuth_gate` / `elevation_gate` / `track_dropped` |
| `chosen_track_id` | 最终选择（起始新航迹时是新 Track ID） |
| `n_tracks_in_gate` / `ambiguous` | 通过门限的候选数；**≥2 即关联存在歧义** |

导出为 `output/stress/association_<场景>_<路>.csv`，**一行 = 一个 (测量, 候选航迹) 对**。
审计记录里**只有航迹 ID 与几何量，没有真值 ID**——审计本身也不携带真值。

由此得到本阶段最关键的一个"检验力"证据：

```
S1 单雷达：关联歧义率 0.2200（22% 的测量面对 ≥2 条候选航迹都过门限）
S2 单雷达：关联歧义率 0.8090
```

也就是说交叉与编队场景**确实制造了真实的关联歧义**，不是"看起来难其实没歧义"。

### 11G.3 十项多目标指标（`multi_target_stress/metrics.py`）

单目标的指标（§11E）在多目标下会算错：两条航迹可以同时"匹配"到同一个真值目标，
丢掉目标再以新 ID 重新捕获（碎裂）根本不会被记到。本模块改为**每帧做一次一对一
离线分配**（贪心、按距离升序），再在分配结果上算指标。

| 指标 | 定义要点 |
| --- | --- |
| `id_switch_count` | 真值目标的分配航迹发生了变化，且**中间没有丢失** |
| `track_fragmentation_count` | 丢失 ≥1 帧后重新分配到**不同**的 Track ID |
| `duplicate_track_count` | 未被分配的**多余**航迹，且身边有真值目标（该目标已被别的航迹占用） |
| `false_track_rate` | 未被分配、**且身边没有任何真值目标**的航迹帧占比 |
| `missed_track_rate` | 合并口径：所有帧所有目标中"没有航迹"的比例 |
| `track_completeness` | 逐目标平均覆盖率（macro）；与上一条在覆盖不均时不相等，是有意分开的 |
| `association_accuracy` | 关联决策里"测量进了它自己真值目标的那条航迹"的比例（虚警进真实航迹即算错） |
| `track_purity` | 每条航迹历史上主导真值目标的占比，取平均 |
| `continuity_rate` | 相邻两帧都被覆盖时，分配航迹保持不变的比例 |
| `position_rmse_m` / `velocity_rmse_mps` | 只在被分配的 (航迹, 目标) 对上计算 |

**"重复 vs 假航迹"的口径是踩坑改对的**：第一版按"目标附近有几条航迹"数重复，
S1 交叉时报出 **34 次重复，全是假阳性**——因为两个目标本身就互相靠近，
对方的航迹被当成了重复航迹。正确口径必须建立在**一对一分配的结果**上
（`tests/test_multi_target_stress.py` 用交叉几何把这个回归钉住了）。

⚠️ 评测门限（关联 1000 m / 重复 1000 m）是**离线参数，不回流入任何算法**。
真实目标误差超过 1 km 的航迹会被计成假航迹，所以报告同时给出
`position_error_p95_m` 便于区分"真有假航迹"与"误差过大被误判"。

### 11G.4 结果（seed=42，30 步，描述性统计）

| 场景 | 共享路 | ID换号 | 碎裂 | 重复 | 假航迹率 | 漏跟率 | 关联准确率 | 纯度 | 连续性 | 位置RMSE(m) | 歧义率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | single | **6** | 0 | 0 | 0.000 | 0.000 | 0.967 | 0.500 | 0.897 | 81.9 | 0.220 |
| S1 | ideal_share | **2** | 0 | 0 | 0.000 | 0.000 | 0.942 | 0.500 | 0.966 | 75.0 | 0.244 |
| S1 | constrained_share | **2** | 0 | 0 | 0.000 | 0.000 | 0.952 | 0.500 | 0.966 | 81.8 | 0.233 |
| S2 | single | **4** | 1 | 0 | 0.000 | 0.156 | 0.733 | 0.879 | 0.943 | 118.4 | 0.809 |
| S2 | ideal_share | 1 | **2** | 0 | 0.000 | **0.333** | **0.622** | 0.800 | 0.980 | 89.3 | 0.821 |
| S2 | constrained_share | 0 | **2** | 0 | 0.000 | **0.333** | **0.608** | 0.800 | 1.000 | 84.5 | 0.796 |
| S3 | single | 0 | 0 | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 101.6 | 0.000 |
| S3 | ideal_share | 0 | 0 | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 74.1 | 0.000 |
| S3 | constrained_share | **1** | 0 | **18** | 0.000 | 0.000 | 0.765 | 1.000 | 0.966 | 137.3 | 0.540 |
| S3L | single | 0 | **1** | 0 | 0.000 | 0.167 | 1.000 | 1.000 | 1.000 | 166.1 | 0.000 |
| S3L | ideal_share | 0 | **0** | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 84.4 | 0.000 |
| S4 | single | 2 | 0 | 0 | **0.474** | 0.000 | 0.753 | 0.967 | 0.966 | 76.1 | 0.342 |
| S4 | ideal_share | 2 | 0 | 0 | **0.474** | 0.000 | 0.792 | 0.967 | 0.966 | 74.1 | 0.380 |
| S4 | constrained_share | 2 | 0 | 0 | **0.474** | 0.000 | **0.656** | 0.967 | 0.966 | 73.8 | 0.368 |

逐场景结论：

* **S1 交叉：换号确实发生，且共享能显著减少它。**
  单雷达 6 次换号、纯度 0.500（逐帧核对过：换号就发生在 t=15 那一帧，
  即两目标真实位置重合的那一帧，此后身份永久互换）。
  理想共享把换号降到 **2 次**、连续性 0.897 → 0.966 —— 远端视角提供了消歧信息。
  这是**关联层**（而不只是精度层）的协同收益。
* **S2 编队：基线误关联最严重，而共享在这里"帮了倒忙"。**
  单雷达关联歧义率 **0.809**、关联准确率 **0.733**、4 次换号、漏跟 15.6%。
  加上理想共享后换号降到 1 次、精度改善（118.4 → 89.3 m），
  但**碎裂 1 → 2、漏跟 15.6% → 33.3%、关联准确率 0.733 → 0.622 都变差了**，
  单帧航迹数从 3 降到 2 —— 3 条航迹里有 1 条从头就没建立起来（合并）。
  诚实结论：**密集编队下"更多测量"不等于"更好关联"**，
  这属于**负结果**，不能只报精度改善那一半。
* **S3 短遮挡（4 s）：基线外推扛住了。**
  4 帧 < `drop_after_misses=5`，航迹 coasting 后重现时重新关联上，
  **0 碎裂、0 重复**，同一 Track ID 保持。远端共享只是把 RMSE 从 101.6 降到 74.1 m。
  但**受限共享反而制造了 18 次重复航迹和 1 次换号**：延迟到达的远端测量
  带着"旧位置"参与关联，在旋转/平移的目标附近造出了落后的第二条航迹。
* **S3L 长遮挡（9.4 s）：删除 + 重新起始，共享能补盲。**
  单雷达 **1 次碎裂**（旧航迹按配置被删除、重现时新建 ID）、漏跟 16.7%；
  理想共享 → **0 碎裂、0 漏跟**（远端雷达视线不经遮挡区，补上了盲区）。
  这是干净的正结果。
* **S4 虚警：虚警确实造出假航迹，且延迟共享损害关联准确率。**
  假航迹率 **0.474**、单帧航迹数峰值 **5**（真实目标只有 2 个）、
  关联准确率 0.753（单雷达）→ **0.792**（理想共享，远端帮助消歧）
  → **0.656**（受限共享，过时的远端测量把关联带偏）。
  虚警**没有**把真实目标"夺走"（漏跟率 0.000），但确实占据了航迹槽位。

### 11G.5 顺带修掉的一个严重 bug：coasting 与删除曾是**死代码**

压力测试第一次跑 S3 就暴露了它：一条航迹**连续 27 帧没有任何观测**，
却仍然是 `confirmed`、`misses=0`、`dropped=0`。

根因在 `FusionCenter.update()`：

```python
self.predict_to(now)      # ← 这里把每条航迹的 last_update_time 都写成了 now
...
for track in list(self.tracks):
    if abs(track.last_update_time - now) < 1e-12:
        continue          # ← 于是永远成立，miss 分支永远走不到
    track.misses += 1
```

`predict_to()` 的用途是"把状态推进到当前时刻"，它更新 `last_update_time` 是合理的；
**错在把同一个字段当成了"本帧有没有拿到测量"的判据**。后果：

* `misses` 恒为 0 → `coast_after_misses` / `drop_after_misses` **全是死代码**；
* `TRACK_COASTING` / `TRACK_DROPPED` 从不出现，`retired_track_ids` 恒为空；
* 幽灵航迹永久累积，只会在指标里表现为"RMSE 莫名其妙偏大"。

修法：`update()` 内改用**本步被更新过的航迹身份集合**（`id(track)`，因为 `Track`
是 `eq=True` 的 dataclass、不可哈希），不再比较时间戳。
`tests/test_multi_target_stress.py` 的 `TestTrackerMissCoastDropUnit`
直接对 `FusionCenter` 单测 `miss → coasting → 删除` 的完整过程——
从场景指标反推是看不出来的，必须直接测。

**这个修复改变了 v4.4 的协同数字**（幽灵航迹消失后，单雷达基线本身变好了），
更新后的值见 §11E.6；它也把 `center.stats["gate_rejected"]` 从
"真实值的 2 倍"改回正确值（`update()` 与 `_associate` 曾各记一次同一条件）。

### 11G.6 JPDA-lite：阈值已触发，但**尚未实现**

按用户要求，"只有在压力测试显示出明显误关联时才考虑加 JPDA-lite"。
预置阈值（先写死再看数据，见 `multi_target_stress/report.py`）：

| 触发条件 | S1 | S2 | S3 | S3L | S4 |
| --- | --- | --- | --- | --- | --- |
| ID换号 ≥ 1 | **6** ✅ | **4** ✅ | 0 | 0 | **2** ✅ |
| 关联歧义率 ≥ 0.30 | 0.220 | **0.809** ✅ | 0.000 | 0.000 | **0.342** ✅ |
| 关联准确率 < 0.85 | 0.967 | **0.733** ✅ | 1.000 | 1.000 | **0.753** ✅ |
| 假航迹率 ≥ 0.20 | 0.000 | 0.000 | 0.000 | 0.000 | **0.474** ✅ |
| 漏跟率 ≥ 0.20 | 0.000 | 0.156 | 0.000 | 0.167 | 0.000 |

**S1 / S2 / S4 都触发了告警阈值**，因此引入 JPDA-lite 的**依据是充分的**。
但本阶段**没有实现它** —— 用户给定的顺序是"先补齐这四类压力测试"，
JPDA-lite 是这一段的**后续**。在实现之前，本工程**不得**声称：

* "关联已经足够可靠"（S2 的关联准确率只有 0.733）；
* "协同总能改善感知"（S2 就是反例）；
* "已经评估过 JPDA 的收益"（没有，只有一个"该做"的判断）。

下一阶段若引入 JPDA-lite，必须同时报告**改善了哪些失效模式**
（预期是 S2 的合并与 S4 的虚警夺轨）与**算力代价**（每帧关联计算量、
与 NN 的耗时比），并且**保持 NN + 卡尔曼为可复现的基线**。

### 11G.7 复现

```bash
python -m multi_target_stress                       # 五场景 × 四路 × seed=42
python -m multi_target_stress --scenario S2 --seeds 42 7 13
python -m unittest tests.test_multi_target_stress -v   # 36 项
```

产物：`output/stress/stress_metrics.csv`、`stress_metrics.json`、
`stress_report.md`，以及逐测量的 `association_*.csv` 与 `lifecycle_*.csv`。

### 11G.8 本阶段**没有**做到的事

* **单种子**（默认 42）。多种子接口已就绪（`--seeds`）并支持均值±标准差聚合，
  但**没有**跑 20–30 种子，**没有**做显著性检验——所有差值都是描述性的；
* **JPDA-lite 未实现**（见 §11G.6）；
* **离线一对一分配用贪心**（按距离升序），不是匈牙利最优解。
  目标数 ≤3 时与最优解一致，但这**未被证明**；
* **目标只做匀速直线运动**：机动目标（转弯、加速）会让常速度模型的
  预测与关联同时退化，本阶段没有覆盖；
* **没有做"融合航迹观测模式"下的 RL 重评**（§11E.5 的遗留项仍未完成）：
  53 维观测走的是 `fuse_measurements()` 的无状态航迹表，
  **不是**本阶段的 `FusionCenter`（这也是观测模式天梯在本轮修复后
  **逐位未变**的原因）；
* **没有外部数据校核**（与 §11D 同一约束）。

---

### 11G.9 四类**系统级**压力场景（S5–S8）

前四类问的是"关联本身会不会错"；这四类问的是**整条链条**在更真实的不确定
条件下还稳不稳。压力测试由此整理为 **4 个核心关联场景 + 4 个系统级场景**。

| 场景 | 打击的目标 | 关键结果（seed=42，描述性） |
| --- | --- | --- |
| **S5 机动目标模型失配** | 常速度卡尔曼的**运动模型** | 目标 t=12 s 转弯 110°、t=22 s 加速 +100 m/s；机动目标残差放大 **1.85 倍** vs 同场景匀速对照 **1.24 倍**，创新峰值 **11.40 > χ²₉₅(3)=7.815** → 1 次门限拒绝，**恢复时间 2.5 s**。基线**没有丢轨**——它吸收了失配 |
| **S6 多雷达覆盖交接** | **跨传感器航迹接力** | 不共享：本地包线外连续性 **0.500**（交接必然失败）。理想共享：连续性 1.000、远端贡献 0.483、交接延迟 **0.00 s**；受限共享交接延迟 **2.00 s**、远端贡献降到 0.363 |
| **S7 通信时序压力** | **时间语义** | 重排缓冲把乱序率 **0.1892 → 0.1351（−28.6%）**，代价是在途时间 **0.459 → 0.730 s**、时效拒绝率 **0.0541 → 0.1351**、5 条测量因扣留而超时效 |
| **S8 多传感器系统偏差** | **融合的加权** | 单雷达 84.76 m｜理想共享 **71.61 m**（−15.5%）｜**有偏共享 108.70 m** → **比单雷达还差 28.2%**；ID换号 2、重复航迹 39、航迹 σ 从 114.4 涨到 138.3 m |

四条**必须一起读**的诚实限定：

1. **S5 是"失配被观测到"，不是"失配导致丢轨"。** 在本场景强度下，常速度滤波器
   用 1 次门限拒绝与 2.5 s 恢复换来了航迹不中断。**不得**据此声称
   "机动必然导致丢轨"——反过来也**不得**声称"基线能处理任何机动"，
   因为只测了一个机动强度。
2. **S6 的"连续性 1.000"只等于"目标覆盖没断"，不等于"航迹交接成功"。**
   新增指标 `within_track_handover_achieved` 专门区分这两件事：
   它要求**同一条 track_id** 先后拿到两个传感器的来源。
   本场景下该条件**成立**（例如 `LOCAL-T7/T8/T9`），但同一批实验里
   ID 换号 22 次、重复航迹 44 条——**接力发生了，身份却没保住**。
   所以正确说法是「共享让目标持续有航迹，但基线在这一过程中大量换号与重复建轨」。
3. **S7 的重排是"有代价的改善"**：乱序降了 28.6%，未达到**先设定**的 30% 门槛
   （阈值写在 `report.py`，没有看到数据后再改），所以在报告里**不成立**
   "重排值得做"这个结论；能确认的只有方向与代价。
   `delayed_update` 是**简化的回溯处理**（按扣留时长放大协方差），
   **不等价于**任何严格的 OOSM 滤波器。
4. **S8 的 health score 是相对判据**：只有两部传感器时，它只能指出
   "更差的那部"，**不能**断定偏差就在它身上；而且它只作**诊断分支**，
   **默认不自动剔除**任何传感器——自动剔除会掩盖基线本身的问题。

### 11G.10 通信时序压力的四个时间戳与三种处理策略

`measurement / send / arrival / fusion` 四个时刻被**显式区分**并可逐条导出
（`oosm_decisions` + `lifecycle_*.csv`）。`multi_target_stress/timing.py`
提供三种策略：

| 策略 | 语义 | 代价 |
| --- | --- | --- |
| `drop_stale`（**对照**，v4.3–v4.5 既有行为） | 来什么立刻喂什么，过时交给跟踪器时效门限 | 乱序时旧包照常参与更新 |
| `reorder_buffer`（新增） | 把"比已融合最新测量还旧"的到达先扣住，按测量时刻升序释放 | 被扣的测量延迟 `window_s`，部分会超时效 |
| `delayed_update`（新增，简化） | 同上，但释放时按扣留时长**放大该测量的协方差** | 同上，另加"旧信息权重更低"的人为设计 |

链路侧新增**突发丢包**（连续丢若干条，而不是逐条独立丢）、**中断窗口**
（窗口内发送的消息全部丢弃）、**恢复后拥塞**（额外丢包 + 额外延迟，线性衰减）
与**乱序到达**（命中消息额外延迟 1.5~4.5 s）。这些默认全零，
因此旧实验的通信行为**逐位不变**。

### 11G.11 传感器系统偏差（S8）的注入方式

偏差加在**传感器测量层**，通过 `SensorConfig` 新增字段：
`range_bias_m`、`az_bias_deg`、`el_bias_deg`、`range_bias_drift_mps`、
`az_bias_drift_degps`、`clock_offset_s`、`noise_underreport_factor`、`bias_start_s`。

三条纪律（由测试钉住）：

* **只污染测量，不动真值**：`record.truth_*` 一个字节都不许改；
* **算法不可见**：偏差不写任何标记字段，消息载荷白名单里也没有；
* **默认全零 / 1 时逐位一致**：`has_bias()` 为假就直接返回。

### 11G.12 AI 对系统级压力的诊断（接 §11F 的证据链）

`ai/system_stress.py` 新增上下文节 `system_stress_state`
（`maneuver` / `handover` / `timing` / `sensor_health`），
并新增 11 个发现码（`FINDING_CODES` 共 **51** 个）：

| 事实 | 发现码 | 是否已在场景中触发 |
| --- | --- | --- |
| 目标机动导致预测模型失配 | `MANEUVER_MODEL_MISMATCH`、`INNOVATION_INCONSISTENT` | ✅ S5 |
| 航迹跨传感器交接 | `TRACK_HANDOVER_COMPLETED` | ✅ S8；S6 **不触发**（身份未保住，见下） |
| 交接失败（有链路却只有单一来源） | `TRACK_HANDOVER_FAILED` | ✅ 受限共享环境 |
| 远端信息因突发中断未被采用 | `COMM_LINK_OUTAGE`、`COMM_BURST_LOSS`、`COMM_RECOVERY_CONGESTION` | ✅ 中断 / 突发 |
| 乱序到达 | `MEASUREMENT_OUT_OF_ORDER` | 代码已通；基线几何下乱序为 0，默认场景未触发 |
| 某传感器长期残差异常 | `SENSOR_BIAS_SUSPECTED`、`SENSOR_NOISE_UNDERREPORTED` | ⚠️ **未触发**，见下 |

⚠️ **两条尚未达成的事实，必须如实记录**：

1. **`SENSOR_BIAS_SUSPECTED` 在 S8 有偏路没有触发**。原因是口径差异：
   离线评测用**全部**来源累计健康分（远端 2.911 vs 本地 1.661，能标出远端），
   而 AI 只看到**当前存活航迹保留的来源窗口**（每条航迹最多 16 条），
   分数被稀释到阈值以下。这**不属于"能力已验证"**——
   该发现码目前只证明了"接线正确"，没有证明"能检出本场景的偏差"。
2. **AI 只能看见当前存活的航迹**：S6 里短暂同时持有两个传感器来源的
   航迹（`LOCAL-T7/T8/T9`）在运行结束时已被删除，因此
   `TRACK_HANDOVER_COMPLETED` 在 S6 不触发。在线诊断看不到"曾经发生过"的证据，
   这是**能力边界**，不是 bug。

---

### 11G.13 一致性验收阶段：外部复核的逐条核对与修复（跨全链路）

> 这一阶段**不新增能力**，只做一件事：把外部复核意见**逐条核实**，
> 是缺陷就修、并在修完后如实说明数字为什么变。
> 原则（用户给定）：**保留旧版本供复现，但修复错误后允许结果变化，
> 必须说明变化原因，不得为保持旧数字而保留缺陷。**

三处修复，影响面逐条对照（完整表格见 `docs/change_report.md`）：

| 编号 | 修复 | 为什么是缺陷 | 影响面 |
| --- | --- | --- | --- |
| F1 | **执行功率进入主动传感器** | 修复前把功率档从 0 调到 10，传感器看到的**检测记录逐位相同**——"功率控制"这条链路在测量层是断的。改为 `_sensor_context` 注入本步执行功率 + `effective_tx_power_w()` 统一正演与反演（RCS 反演必须用同一个 `Pt`，否则 σ̂ 会被 `Pt_cfg/Pt_actual` 整体缩放，实测在 0 档差 36 倍） | `ideal`/`realistic` 的检测结果与整条测量序列；`full` 不经测量层，**不受影响** |
| F2 | **观测空间逐字段上下界** | 有符号维（`bearing_norm`/`elevation_norm`/`range_rate_norm`）被裁剪到 0，负值信息全部丢失。修复前默认场景 5 种子×12 步**从未**产生负值，因此它是**潜在的契约违规**而不是已观测到的损失——这一点如实标注为"契约违规"而非"观测到损失" | `ideal`/`realistic` 的 12 个有符号维（53 维中）；`full` 无有符号维 |
| F3 | **缺失率分母改为平台容量** | 原分母是**真值实体数**：3 个目标全在视场外时，九维缺失率仍全是 1.0（"什么都没缺"）——真值从分母漏进了观测 | `ideal`/`realistic` 的九维缺失率；`full` 不含该组特征 |

**对照锚点**：`full` 模式的 5 个策略（固定 80 W `-3.1954`、规则 `+0.4636`、
随机 `-0.5866`、短视 `+0.4949`、DQN `+0.4739`）**全部逐位未变**——
这是本轮最重要的对照，证明三处修复都没有越界影响 legacy 路径。

**同时落地的基础设施**（都是为了"结论可追溯"）：

* **区分在线/离线快照**：`ai/snapshot_boundary.py` 显式标注 `online` / `offline`
  来源，并禁止真值顶层字段进入在线快照（`boundary_provenance` 字段名曾被
  `truth` 前缀的泄漏扫描器误判，选择**改名**而不是加白名单，避免留下泄漏旁路）；
* **运行隔离与清单**：`run_manifest.py` 给每次运行独占 `output/runs/<run_id>/`
  目录 + `manifest.json`（配置摘要/源码摘要/产物 sha256），
  并**拒绝**把不同运行的产物写进同一目录；契约见 `docs/data_contract.md`；
* **修复前后基线对照**：`docs/baseline_pre_fix.json` / `baseline_post_fix.json`
  + `tools/compare_baselines.py`（只解析逐节的天梯块，避免把汇总行算进上一节——
  第一版解析器就因此伪造出一条"realistic/rule 与 myopic 一起塌陷到 −0.9837"
  之外的错误结论）。

⚠️ **一个未隔离的信号，写论文时不得使用**：修复后 `realistic` 模式下
`rule` 与 `myopic` **落到完全相同的 −0.9837**。两个不同策略得到同一个数值
是**轨迹退化**的特征。已确定这两行是观测耦合的（脚本策略必须走信念桥接）、
已排除 F3（`belief_policy.py` 不含 `meas_rate` 任何引用），
但 **F1 与 F2 各自的贡献尚未分开测**（隔离方法已写明：临时只回退 F1/只回退 F2，
比较动作序列；本轮**没有做**）。

因此：
* **旧 checkpoint 不能用于修复后的策略对比**（训练/测试输入语义不一致），
  可归因的策略对比必须**重新训练**；
* **不得引用修复后的 `realistic` 行做任何能力判断**（`docs/change_report.md` §4.3）。

---

## 11H. 教学仿真资源管理（大阶段一，已冻结）

### 11H.1 这一阶段的交付是什么

> **资源管理问题已经定义清楚、并且能运行。**

不是"多了一个调度器文件"。要能说这句话，必须能回答"口径是什么、
它有没有被动过"——因此本阶段的核心产物之一是**可校验的冻结契约**。

四件事构成一个闭环：

| 层次 | 交付 | 关键约束 |
| --- | --- | --- |
| 资源管理层 | 单位 / 时钟 / 两阶段校验 / 逐节点账本 | 守恒式恒成立、失败不吞资源 |
| 适配层 | 融合结果 → 调度器只读观测 | 独立 schema、字段全标注、只读已到达 |
| 方法层 | 3 个规则基线 + 2 个非学习优化参考 | 共用观测/队列/执行器，**不预设谁更好** |
| 契约层 | `resource-contract-v1` | 改一处即校验失败 |

⚠️ 本阶段**没有**触碰任何雷达/侦察/干扰公式，**没有**改动
`full`/`pomdp`/`ideal`/`realistic` 四条旧观测路径。旧实验逐位可复现。

### 11H.2 资源单位与两阶段校验

| 单位 | 含义 | 是否预算单位 |
| --- | --- | --- |
| `sample_slot` | 一次采样占用的采样槽 | ✅ |
| `processing_op` | 一次处理占用的处理操作 | ✅ |
| `comm_byte` | 一次共享占用的通信字节 | ✅ |
| `occupancy_second` | 时间占用量 | ❌ **只作时间占用，不参与守恒式** |

守恒不变量：对每个预算单位都有
`consumed + reserved + remaining == capacity`，且三者均非负。

执行器只在**一处**扣费，校验分两类：

| 类别 | 触发条件 | 后果 |
| --- | --- | --- |
| `PLAN_FATAL` | 未知节点/类型、负成本、时长非法、`start_s != now`、计划内重复、去重键重复、占用冲突 | **整份计划被拒**，一个单位都不扣，状态快照逐位不变 |
| `NODE_LOCAL` | 节点不可用、资源不足、未到更新周期 | **只拒该节点的任务**，其余节点照常执行，且不得因此结束整个网络任务 |

⚠️ 每个节点每 tick 最多提交 **1** 条任务，这不是保守取值而是执行器的硬约束：
执行器只支持立即执行（计划内 `start_s` 必须等于当前时钟），同一节点上两段
从当前时刻开始的占用**必然重叠** → `PLAN_FATAL` → 整份计划被拒。
这个坑很隐蔽：把上限设成 2 时结果是"计划数翻倍、完成数归零"，
而错误只出现在执行器拒绝记录里，指标表上看起来像"策略变差了"。
现在 `SchedulingConfig.validate()` **直接拒绝**非法取值。

### 11H.3 融合结果 → 资源调度适配层（信息边界）

"融合器输出存在" ≠ "调度器已经真正使用了它"。本层把融合输出翻译成调度器
可只读消费的观测，三条约束由**构造方式**保证，不靠约定：

| 约束 | 实现方式 | 钉住它的测试 |
| --- | --- | --- |
| 调度器不持有真值 `Simulator` | import 里没有 `engine.*` / `sensor` / `torch` | AST 扫 import（子串匹配会把文档字符串里的声明误判成违规） |
| 中央只读**已到达**的数据 | `ingest()` 拒绝未到达；`ingest_arrived()` 走 `CommBus.consume` | 延迟链路下逐条验证 + 绕过总线的二层防线 |
| 变长 + 有效掩码 | 航迹与节点都是列表 + 掩码；定长需求走**独立适配器** | 掩码/溢出/维度检查 |

**验收核心**：

> **断开远端消息后，调度器不能继续"知道"远端的新信息。**

实测（`verify_v4.py` §16）：本地视图冻结在 `3239.20 m`，而底层真值已到
`3480.0 m` —— 中央**没有**跟上。

观测 schema 使用**独立**的 `SCHEMA_VERSION = "rm-obs-1.0"`，
25 个字段全部标注单位/坐标系/可见范围/来源（`local` / `shared` / `derived`）。
旧路径的 4 个模式在 `LEGACY_OBSERVATION_MODES` 里显式标记并保留，
定长适配器会**拒绝**旧维度（实测拦下 12 / 16 / 53）。

**任务队列**也守住同一条边界：**未知对象不得依据 `truth_id` 提前创建任务**，
判据是**可见性**而不是模式匹配——即便传入一个像真值 ID 的字符串
（`TGT1`），只要调度器没看见过它就会被拒。

**为通信延迟新增的一项指标**：原来只记"到达年龄（now − 到达时刻）"，
而延迟 2 s 的摘要在**送达那一 tick 的到达年龄仍是 0**（"刚收到"），
于是"通信延迟"这条机制在指标上完全不可见（快/慢两个配置都记 0.000）——
这不是机制没生效，是**指标口径选错了**。现在同时记
`node_information_age_s`（到达年龄）与 `node_content_age_s`（= now − 摘要生成时刻）。

### 11H.4 非学习规则基线：三个策略共用一切

差别**只在排序与选择**：

| 策略 | 依据 | 特点 |
| --- | --- | --- |
| `round_robin` | 节点轮流 + 节点内**先到先服务** | 最弱基线：不看紧迫性、不看新鲜度 |
| `edf` | 只看 `deadline_s` | 没有截止时间的排最后 |
| `rule` | 教学任务等级 + 等待时间 + 数据新鲜度 + 估计质量 | 四项**全部可观测**，逐项可算出来 |

`rule` 的打分（权重全部显式在 `SchedulingConfig` 里）：

```
priority = 0.3 × 任务等级
         + 0.5 × min(1, 等待 / starvation_threshold)
         + 0.6 × min(1, 信息年龄 / max_information_age)
         + 0.4 × min(1, 位置σ / max_sigma_position)
```

**没有**军事目标价值、威胁度、对抗效能之类的项；证据里也禁止出现这些字段。

**验收**（用户原话）：**系统确实会产生不同分工，而且每次分工都有原因和实际
执行记录。** 实测四个场景 `work_division_differs` 全为真：轮询/EDF 两类服务
都做，规则只做等级最高的 `estimate_update`。

### 11H.5 语义定义（完成 / 过期 / 放弃 / 重复 / 长期未获服务）

| 语义 | 定义 | 关键性质 |
| --- | --- | --- |
| **完成** | 执行器 `applied` 且记账 | 计入分子 |
| **过期** | 截止时间到、任务失去意义 | **不删除**，仍计入分母 |
| **主动放弃** | 等待超 `abandon_after_s` 且仍无法服务 | 状态 `cancelled`，**仍留在队列**，分母不变 |
| **重复请求** | 同 `task_id` / 同去重键 / 计划内重复 | 拦下且**不重复扣费** |
| **长期未获服务** | 等待超阈值且**不是**因截止时间到期而离开队列 | 与"过期"**互斥** |

完成率分母 = 完成 + 过期 + 主动放弃 + 执行器拒绝 + 长期未获服务。
**主动放弃不会让分母变小，因此"删掉难任务"只会让完成率下降。**

"互斥"这一条是修出来的：第一版只在运行结束时统计"仍在排队且等待超阈值"的
任务，而默认场景里所有任务都带 2~6 s 截止余量、饿死阈值是 8 s，
于是任务**总是先过期**，`n_starved` 恒为 0——这条语义形同不存在。
反过来若两个计数都记同一条任务，完成率分母会被重复计入（偏低也是错）。

### 11H.6 非学习优化参考：**"有前瞻"≠"理论上界"**

| 情形 | 允许的表述 | 禁止的表述 |
| --- | --- | --- |
| 单 tick 决策问题**完全枚举**完毕 | "该问题在该声明式目标下的**精确最优**" | "理论上界" |
| 首 tick 枚举 + 后续声明式 rollout | "**优化参考**" | 任何最优性主张 |
| 组合数超预算 → 束搜索 | "**优化参考**（束搜索）" | 任何最优性主张 |
| 预算在枚举完成前耗尽 | "已评估部分中的最好解" | 任何最优性主张 |

这条纪律由**代码结构**保证，不靠自觉：

* `exact=True` **只有一条**产生路径——组合全部枚举完 **且** 未触预算
  **且** 预测时域为 1；
* 声明文本必须**逐字**等于 `CANONICAL_CLAIMS` 的渲染结果（白名单），
  渲染参数一并存进结果供逐字比对；
* 结构化 `claim_kind` 必须与 `exact` 一致；`exact` 还必须"有据"
  （报出的组合总数 > 0 且展开次数 ≥ 该总数）。

> 为什么不用关键词扫描：第一版自审用"声明里是否出现『理论上界』"判断，
> 结果**否定句**（"不是整个调度问题的理论上界"）被误判成"自称理论上界"。
> 关键词既会误报、也能被措辞绕过。

**不得读取未来的四类信息**：

| 禁止 | 保证方式 |
| --- | --- |
| 未来真实测量 | 预测只用当前可见观测外推（`PredictionModel`） |
| 未来故障 | 模型里**没有**这条信息：`unavailable(t+k) = []` 是一条显式假设 |
| 隐藏对象状态 | 只能引用 `observation.track_ids()` 里的键 |
| 真值 | 优化层不 import `engine`/`sensor`/`fusion`（AST 扫描钉住） |

7 条预测假设随结果一起落盘，其中两条是**已知的乐观假设**，报告里必须一并给出：
`persistent_tracks`（假设航迹在时域内始终可见）与
`arrival_rate`（未来到达数 = 当前观测到的到达数）。

### 11H.7 六维评价向量（**不给单一综合分**）

| 维度 | 单位 | 方向 | 定义 |
| --- | --- | --- | --- |
| 服务完成度 | 比值 | 越大越好 | 已完成 / (完成+过期+放弃+拒绝+饿死) |
| 任务及时性 | 比值 | 越大越好 | 截止前完成数 / 已完成数 |
| 估计质量 | 1/(1+秒) | 越大越好 | 1/(1+平均信息年龄)；预测取**时域内平均** |
| 资源消耗 | 容量占比 | 越小越好 | 三类单位 `consumed/capacity` 的算术平均（逐单位明细另附） |
| 通信开销 | 字节 | 越小越好 | 账本里的通信字节总量 |
| 计算耗时 | 秒 | 越小越好 | **规划**的墙钟时间总和（不含执行与仿真推进） |

**预测向量与实测向量严格分开**：优化参考内部排序用 `provenance="predicted"`，
报告对照用 `provenance="measured"`。混在一起就会得出
"用预测的好处去比实测的代价"这种错误结论。

**计算预算是硬约束**，四个维度：`max_expansions`（候选计划评估次数）、
`max_horizon_ticks`（预测时域）、`pareto_max_points`（**后处理**上限——
帕累托是非支配筛选、代价 O(n²)，实测 713 个向量要 0.022 s；
第一版预算只管搜索循环不管后处理，那是假预算）、`time_limit_s`（墙钟）。

⚠️ **墙钟预算与"可复现"直接冲突**，这一点是被阶段验收逼出来的：

> 同一配置连跑四次，展开数分别是 **1315 / 1270 / 1312 / 1292**——
> 机器负载不同 → 停止点不同 → 计划也因此可能不同。
> `verify_v4.py` 第 18 节曾因此**偶发失败**（4/5），不是随机噪声，
> 而是一个真实的性质缺陷。

处置方式是让方法变确定，而不是把验收条件放松：`time_limit_s` 降级为
**安全阀**（默认 5 s，在本沙盒里不会触发），一旦触发即
`deterministic=False` 入账，**该次结果不允许作为可复现基准**；
阶段验收第 5 项**显式要求** `non_deterministic_runs` 为空。

### 11H.8 实测结果与阶段验收

`evaluate_resource_management.py`（种子 42/7/13，24 tick；三个种子在本配置下
**逐位相同**，因此下表每个数字都是三次一致的取值）：

| 场景 | 方法 | 完成度 | 及时性 | 估计质量 | 资源消耗 | 通信(B) | 计算耗时(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | rule | 0.1951 | 0.2708 | 0.4985 | 0.2538 | 0 | 0.019 |
| base | enumeration | 0.1920 | **1.0000** | 0.4985 | 0.2289 | 1024 | 0.174 |
| base | rolling_horizon | 0.1920 | **1.0000** | 0.4985 | 0.2251 | 1152 | 0.383 |
| node_outage | rule | 0.1860 | 0.2889 | 0.5386 | 0.2531 | 0 | 0.018 |
| node_outage | enumeration | 0.1829 | **1.0000** | 0.5386 | 0.2382 | 640 | 0.236 |
| node_outage | rolling_horizon | 0.1829 | **1.0000** | 0.5386 | 0.2334 | 896 | 0.642 |

**五条必须一起读的诚实结论**：

1. 优化参考把**及时性从 0.27 提到 1.00**（48 条完成里 13 条按时 → 48 条全按时），
   代价是 **9~35 倍规划耗时**与约 1 KB 通信。这是**真实的取舍**，不是"谁更好"；
2. **没有任何方法支配其他方法**（`base`）：规则基线在通信为 0 与计算耗时上最优，
   优化参考在及时性上最优。这正是必须用向量而不用综合分的原因。
   `node_outage` 场景里枚举参考支配滚动规划参考——但那是**两者之间**的比较；
3. **完成度几乎不变**（0.195 → 0.192）：完成度由**服务上限**封顶
   （需求/能力 ≈ 6.3），不是由调度决定的。**不能**单独用完成度判定方法优劣；
4. **`estimate_quality` 三方法完全相同**：节点摘要每 tick 无条件发布，
   中央看到的新鲜度不由被调度任务决定。这是**建模缺口**（共享/刷新的收益链路
   尚未接进资源约束），不是调度器的成绩或过错；
5. **规则调度把某一类服务永久饿死**：`share` 不绑定航迹 → 拿不到信息年龄/
   协方差项 → 优先级上限 `0.3+0.5=0.8`，低于"零需求"高等级任务的 `0.9`。
   如实记录，未修。

**阶段验收清单（6 项，全部通过）**：

| # | 条件 | 判定方式 |
| --- | --- | --- |
| 1 | 多节点执行真实生效 | ≥2 个节点既有已规划任务、又在账本里有真实消耗；计划数 > 0 |
| 2 | 资源不超支 | 全部策略 × 种子 × 节点 × 单位的守恒残差为 0、`consumed ≤ capacity`（实测 90 个样本 0 违反） |
| 3 | **信息不越权** | **运行时对照**：改掉未来的 NODE_B 不可用窗口、当前观测不变，**故障窗口之前的规划必须逐位相同**（实测前 6 个 tick 逐位相同，窗口内确实不同——否则说明扰动没生效）+ 静态 AST 扫描 |
| 4 | 任务队列可追溯 | 逐状态计数之和 = 任务总数（303 = 303）；每条决策/时间线行都带原因 |
| 5 | 规则与优化参考可复现 | 同配置重跑指纹逐位相同（**排除墙钟耗时**）；且冻结摘要一致、优化参考搜索确定性 |
| 6 | **调度反向控制感知链（真闭环）** | 见 §11K：旧路径下三策略估计质量**完全相同**（闭环未成立的症状），新路径下不同；停采样窗口内扫描 0 次且 σ 单调上升；无重复执行、无真值泄漏、通信字节与账本一致、资源守恒 |

> 未全部通过前**不进入学习算法阶段**；通过后也**不要求**规则方法必须失败、
> 学习方法必须胜出。第 6 项是**后补的**：前 5 项通过时，调度其实还没有
> 反向控制感知链（见 §11K.1），因此当时的"通过"并**不**意味着闭环成立。

### 11H.9 冻结契约 `resource-contract-v1`

| 块 | 内容 |
| --- | --- |
| 观测 schema | 25 字段的单位/坐标系/可见范围/来源 + `SCHEMA_VERSION = rm-obs-1.0` |
| 任务完成口径 | 7 种状态、完成率分母、饿死与过期的互斥定义、**不得删除任务** |
| 资源模型 | 三类预算单位、教学成本模型、时长、守恒不变量 |
| 基准配置 | 节点布局、逐节点预算、场景机制、调度默认值、固定种子与 tick 数 |
| 评价维度 | 六维向量的逐字定义 + 声明式目标的默认权重 |

摘要 `380dad61d2efb15aaf6dadd532b831d1`，三处一致（源码常量 / `docs/resource_contract_v1.json`
/ 实时重算）。冻结**不等于**不能改：改动会让 `verify_frozen()` 失败，
必须走「升版本号 → 重算摘要 → 同步文档 → 重跑验收」。
**禁止的是悄悄改**——一旦口径悄悄漂移，"规则基线 vs 优化参考"的对比
就变成了"两套问题定义的对比"，那不是结论，是误会。

### 11H.10 复现

```bash
python -m resource_management                        # 教学演示（7 步）
python tools/compare_schedulers.py                   # 三规则基线分工对比
python evaluate_resource_management.py               # 规则 vs 优化参考 + 验收清单
python verify_v4.py                                  # 端到端验收（§17 资源调度 / §18 优化参考与冻结）
python -m unittest tests.test_resource_scheduling tests.test_resource_optimization -q
```

### 11H.11 本阶段**没有**做到的事

1. **没有跨 tick 的资源预留**：执行器仍只支持立即执行（`start_s == now`），
   因此每节点每 tick 最多 1 条；`reserved` 机制已就位但未被调度器使用；
2. **没有把共享/刷新的收益接进资源约束**：`share` 任务成本照扣、收益为零，
   因此 §11H.8 第 4 条那个缺口未修；
3. **没有证明滚动规划的任何界**：rollout 是声明式的，既不最优也没有近似比保证；
4. **没有统计意义上的方法比较**：默认场景逐种子同值，需要先打开概率检测/
   虚警或随机化场景，才能谈"重复实验"；
5. **没有跨节点航迹关联**：中央看到的是各节点各自上报的航迹，
   不同节点的同一条真值航迹**不会**被合并成全局航迹 ID；
6. **没有用真实 OOSM 语义**：观测摘要按"最后到达"覆盖，乱序到达的旧摘要
   会被更新的覆盖；
7. **没有接学习算法**：本阶段到此为止（闸门见 §11H.8）。

---

## 11I. 学习问题与评测协议 v1（学习算法阶段的入口闸门）

> 这一阶段**不训练**任何东西。它只做一件事：把"实现错误"与"算法能力不足"
> 分开，否则后面所有训练结果的归因都是不可信的。

### 11I.1 先修语义，再谈能力

审计发现旧链路有三处**语义缺陷**，它们会让负结果无法归因：

| # | 缺陷 | 为什么必须修 |
| --- | --- | --- |
| 1 | `LpiPowerEnv` 把"场景时长到达"统一标为 `truncated=True`，而训练只对 `terminated` 关闭 bootstrap | 任务自然终点被伪装成外部截断 → 终止语义与实际不符 |
| 2 | 拉格朗日分支部署时用 `argmax(Q_r − λ·Q_c)`，但训练时奖励 critic 按奖励 Q 选下一动作、代价 critic 又独立取最大代价动作 | **三个目标不是同一个估计对象** → 负结果不能归因于"方法能力不足" |
| 3 | 训练种子逐 episode 递增、验证种子未定义 | 训练/验证/测试划分未定义 → 无法防止"调参调进测试集" |

修复后的语义（`docs/learning_protocol.md`）：

| 事件 | 标志 | 终端补计 | bootstrap |
| --- | --- | --- | --- |
| 全部任务已完成或过期 | `terminated=True` | 是 | 否 |
| 剩余任务均不可由剩余资源执行 | `terminated=True` | 是 | 否 |
| 问题定义内的有限任务时域到达 | `terminated=True` | 是 | 否 |
| 调用方施加的更短步数上限 | `truncated=True` | 否 | **是** |

Bellman 目标固定为 `y = r + γ·(1−terminated)·Q_target(s')`。
旧行为通过显式 `horizon_semantics="legacy_truncation"` 保留以复现旧 checkpoint。
拉格朗日双 critic 改为**共用部署策略的同一个拉格朗日贪心动作**
（`tests/test_lagrangian_semantics.py` 用固定网络值校验）。
**修复尚未重训，因此历史负结果保留但归因降级为"校核前历史观察"。**

### 11I.2 四个可手算的精确案例

`resource_management/exact_cases.py`（γ=0.9），测试**不依赖随机统计**：

| 案例 | 动作 | 逐步奖励 | 不折扣/折扣回报 | 累计代价 | 结束 |
| --- | --- | --- | --- | --- | --- |
| `single_on_time` | `[1]` | `[2.0]` | `2.0 / 2.0` | `1/6` | 全部解决 |
| `wait_then_complete` | `[0,1]` | `[-0.25, 2.0]` | `1.75 / 1.55` | `1/6` | 全部解决 |
| `horizon_unresolved` | `[0,0]` | `[-0.25,-1.75]` | `-2.0 / -1.825` | `0` | 自然时域 |
| `resource_exhaustion` | `[1]` | `[0.25]` | `0.25 / 0.25` | `1/3` | 资源耗尽 |

单步约束代价定义为 `c_t = mean_u(Δresource[t,u] / budget[u])`，
因此 `Σc_t` **必须**等于 episode 末由账本重算的 `resource_consumption`
（这条一致性是"实现错误"与"算法问题"的分界线）。

### 11I.3 独立环境与数据封存

`resource_management/learning_env.py` 的 `ResourceSchedulingLearningEnv`
**不 import** `engine`/`sensor`/`fusion`/`communication`，
只依赖冻结资源单位与包内最小空间类；观测显式包含
`elapsed_fraction` / `remaining_time_fraction` 以保持有限时域问题的 Markov 性。

`config/learning_splits_v1.json`（+ `.sha256`）是唯一的划分登记表，
场景与种子在两两之间不相交：

| 分区 | 用途 | 种子 |
| --- | --- | --- |
| train | 参数更新 | 101, 103, 107, 109, 113, 127, 131, 137 |
| validation | checkpoint 选择 | 211, 223, 227 |
| test | **封存**，最终一次报告 | 401, 409, 419, 421, 431 |

`get_split("test")` 默认抛 `SealedTestSplitError`，只有最终评测入口才能
显式 `release_test=True`。**本轮未解封，也未训练。**

---

## 11J. 单一研究分支：新鲜度与不确定度感知调度

### 11J.1 唯一可检验假设（预登记）

> 在环境、感知链、资源模型、训练预算与编码结构固定的前提下，
> 当已到达数据存在延迟、缺失或来源不一致时，显式使用**信息年龄**、
> **估计协方差**和**基于可观测残差的来源一致性指示量**，
> 相比不使用它们，能否降低教学服务任务完成率的**最差场景退化**，
> 且**不增加资源超支**。

本轮**不新增** JPDA / IMM / 循环网络 / 注意力 / 多智能体。

### 11J.2 四组消融（同结构、只置零）

唯一机器可读契约是 `config/information_research_v1.json`（SHA-256 同目录）。
四组共用环境、感知链、资源模型、队列、预算、隐藏层 `[128,128]`、输入维度与
特征顺序；**禁用特征是置零占位，不是删维度**。
`CentralResearchFeatureAdapter` 给学习侧提供固定编码，
`rule_config_for_arm` 把**完全相同的特征开关**交给规则基线——
因此额外信息带来的收益**不能被写成学习算法的收益**。

| 组别 | 新鲜度 | 协方差 | 来源一致性 |
| --- | --- | --- | --- |
| `main_baseline` | 关 | 关 | 关 |
| `freshness_only` | 开 | 关 | 关 |
| `uncertainty_only` | 关 | 开 | 开 |
| `freshness_uncertainty` | 开 | 开 | 开 |

来源一致性定义为 `1 / (1 + mean(residual_m / reported_sigma_m))`，
是**未经概率校准的相对指示量**——不得称为"出错概率"或"正确概率"。
调度输入禁止读取真实传感器偏差、真值关联正确性、真值虚警标签与未来消息状态。

### 11J.3 结果：主假设**未获支持**（如实记录负结果）

规则基线机制检查，开发种子 101/103/107，每格 24 tick，复用 S6/S7/S8 机制：

| 消融组 | 完成率均值 | 最差完成率 | 平均过期 | 平均等待(s) | 资源消耗 | RMSE(m) | 计算(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_baseline` | 0.1654 | **0.1103** | 194.00 | 3.720 | 0.2202 | 374.80 | 0.011788 |
| `freshness_only` | 0.1652 | 0.1088 | 194.22 | 3.536 | 0.2202 | 374.80 | 0.012460 |
| `uncertainty_only` | 0.1651 | 0.1088 | 194.33 | 3.325 | 0.2202 | 374.80 | 0.012758 |
| `freshness_uncertainty` | 0.1652 | 0.1088 | 194.22 | **3.493** | 0.2202 | 374.80 | 0.013564 |

组合特征相对主基线：平均等待 **3.720 → 3.493 s（↓0.227 s）**，
但**最差完成率 0.1103 → 0.1088（退化）**，平均过期数 194.00 → 194.22。

> **当前小样本证据不支持"最差完成率改善且不增加资源超支"的完整假设。**
> 失败格子已保留：S7 通信时序场景种子 103 上组合特征多出 2 个过期任务，
> 完成率下降约 0.0015。

⚠️ **四组 RMSE 完全相同（374.80 m）不能用来判定特征无效**：
当时调度执行**尚不会**改变传感器或融合更新，因此误差是对照检查而非可归因结果
（这个限制已在 §11K 被解除——见 §11K.3 的重测提示）。

---

## 11K. 计划控制感知链：真闭环

### 11K.1 修的是什么：闭环其实**没有成立**

大阶段一交付时，多雷达资源管理只有任务分配、预算、规则/优化调度与账本，
**调度结果并没有反向控制 Sensor → Communication → Fusion**：

| 环节 | 修复前的真实行为 |
| --- | --- |
| 传感器 | `run_closed_loop` 每 tick **无条件** `suite.observe` —— 未分配采样任务的雷达**照常**产生测量 |
| 融合 | 每 tick **无条件** `FusionCenter.update` —— 未调度节点也照常刷新航迹 |
| 通信 | 节点摘要每 tick **无条件** `publish_node_observation`；`SHARE` 只扣账，与真实发送**无关** |

后果不是"统计不好看"，而是**闭环根本不成立**——实测旧路径下轮询 / EDF / 规则
三策略的 `estimate_quality` **完全相同**（0.498522947 ×3），
离线误差也完全相同（155.68 m ×3）。**调度只改变任务与资源统计。**

### 11K.2 唯一 `ExecutionPlan → RuntimeExecutor` 入口

分层原则：**记账与副作分离**。

```
scheduler.plan() → ExecutionPlan
                      ↓
        UnifiedExecutor.submit()     ← 仍是**唯一**记账点（校验+去重+扣费）
                      ↓ 只对 Outcome.APPLIED
        RuntimeExecutor.submit()     ← 只做**一次**运行时副作，**不二次扣费**
                      ↓
      sample → Sensor.observe ｜ process → FusionCenter.update ｜ share → CommBus.publish
```

| 任务 | 唯一副作 | 关键性质 |
| --- | --- | --- |
| `sample` | `Sensor.observe` 一次 | 未拿到 sample 的节点**一次都不扫** |
| `process` | `FusionCenter.update(本地缓冲 + 已到达远端)` | 未调度节点只 `predict_to`（**不是** `update([])`——那会偷偷记 miss 并删轨） |
| `share` | `CommBus.publish` 一条真实测量 | 每条 share 最多一条消息，字节与 `comm_byte` 账本一致 |
| 无任务 | `predict_to` | 信息年龄与协方差自然增长 |

**实测（seed=42，24 tick）**：

| 模式 | 策略 | 估计质量 | 传感器测量 | 融合更新 | 消息 | 离线误差(m) |
| --- | --- | --- | --- | --- | --- | --- |
| 旧路径 | 轮询/EDF/规则 | **0.498522947 ×3** | 不产生运行时统计 | — | — | **155.68 ×3** |
| 新路径 | 轮询 | 0.190780461 | 18 | 23 | 15 | 1112.69 |
| 新路径 | EDF | **0.168980994** | **19** | **20** | **18** | **982.22** |
| 新路径 | 规则 | 0.190780461 | 18 | 23 | 15 | 1112.69 |

**停采样 → 不确定度上升**（NODE_B 在 [3,7] 不可用）：
窗口内扫描 **0** 次，位置 σ **167.2 → 201.3 → 240.9 m 单调上升**；
窗口结束后扫描恢复、σ 回落。恢复**不是一步完成的**：门控顺序是
`process → share → sample`，因此窗口不能贴到运行末尾。

**share 门控**：通信预算正常 → 7 次 share / 7 条消息 / 896 B（与账本一致）；
通信预算全置 0 → **0 次 share / 0 条消息 / 0 B**。

**不变量（由实际扫描统计，不是硬编码常数）**：
`n_runtime_tasks == n_unique_task_keys`（24 == 24，每任务只执行一次）；
`observed_payload_violations == observed_context_violations == 0`
（真值隔离，逐 tick 扫实际序列化的中央快照与发送载荷）；资源守恒成立。

### 11K.3 开关、旧路径与已知限制

```python
run_closed_loop(..., runtime_mode="legacy_observation_first")    # 默认：旧路径
run_closed_loop(..., runtime_mode="plan_controlled_feedback")    # 新路径
```

旧路径**保留且逐位未变**（base 场景 `rule` 仍是 48 完成 / 13 按时 / 0 B /
`estimate_quality` 0.4985 / `resource_consumption` 0.2538）。

⚠️ **新路径的限制，必须与收益一起读**：

1. **通信预算耗尽会停住节点**：`share` 无法执行时 outbox 永非空，
   门控于是不再派采样任务 → 节点进入纯预测。这是**资源耗尽**语义的直接后果，
   不是 bug，但它是硬门控而非软降级；
2. **估计误差整体变大**：旧路径等于"每节点每 tick 免费获得一次测量"；
   新路径按实际任务量给测量，误差均值由 155.68 m 变为 1000 m 量级。
   这**不是回归**，而是把此前被隐式补上的测量量显式化了；
3. **完成率不可跨模式比较**：新路径的任务由实际运行状态派生
   （sample → 可 process → 可 share），需求结构与旧路径不同；
4. **多雷达功率控制仍未接通**：本次接通的是**资源调度的多节点感知/通信/融合**
   闭环，`radars[0]` 的联合功率动作仍未实现；
5. **响应带宽受单 tick 单任务限制**：`sample → process → share` 需 3 个 tick
   才能走完一轮；
6. **§11J 的四组消融需要重测**：该分支当时受"调度不影响感知链"的限制
   （四组 RMSE 相同）。该限制现已解除，重测后其结论可能改变——
   在重测完成前，§11J.3 的结论只在"执行—感知回路未接通"的旧前提下成立。

### 11K.4 修复过程中发现并修掉的一个回归

第三方在本阶段改造 `closed_loop.py` 时，把反馈路径的门控代码
（`processable` / `shareable`）**误留在了旧路径里**，而 `runtime` 只在反馈模式
下才定义 → 旧路径每 tick 抛 `UnboundLocalError` → 被 `except Exception` 吞掉
并记成 `task_creation_failed` → **闭环跑完全程但一个任务都不产生**
（实测 `n_tasks=0, n_plans=0`，全量测试 10 项失败）。

两处修复：

1. 删掉旧路径里那两行**未被使用**的门控代码（反馈路径的门控语义不变）；
2. 把包住建任务的 `except Exception` **收窄为领域异常**
   （`DuplicateTaskError` / `UnknownObjectError`）——
   编程错误（`NameError` / `UnboundLocalError` 等）**不允许**被记为
   "任务创建失败"，否则硬 bug 会伪装成软失败。这正是本次事故的成因。

---

## 11L. 集中式学习资源调度基线（大阶段二 · 第一步）

> 状态：PPO 基线训练与消融已完成；跨方法结论以 §11N 的**统一冻结评测**为准。
> 本节保留训练期记录，不单独用于规则/优化/PPO 排名。

### 11L.1 交给智能体的是"提交什么任务"

大阶段一交付了问题定义与真闭环，但"提交什么任务"一直由**手工门控**决定
（每 tick 只派生一种可执行的任务类型）。本层把它交给**一个中央智能体**：

```
CentralObservation + 任务队列（算法可见信息）
   ↓ 定长编码（49 维：5 全局 + 11 × 节点槽位）
Actor-Critic（逐节点因素化分类头 + 合法动作 mask，隐藏层 [128,128]）
   ↓ 逐节点动作 (idle/sample/process/share)^N
标准 ExecutionPlan
   ↓
UnifiedExecutor（唯一记账点）→ RuntimeExecutor（唯一副作点）→ Sensor/Fusion/CommBus
```

⚠️ **任务派生门控必须换成 `expose_all`**：默认的 `loop_gate` 已经替策略决定了
该提交哪类任务，动作空间里就没东西可学了。`expose_all` 把三种候选同时暴露给
策略（仍沿用真实可行性），它改变的是**候选构成**、不改传感器/通信/融合语义，
因此两种模式的完成率**不可直接比较**。

### 11L.2 六条硬约束（都能被机器检查）

| 约束 | 检查方式 |
| --- | --- |
| 不改冻结契约 `resource-contract-v1` | `verify_frozen()` 仍通过 |
| 不改传感器/通信/融合/完成口径 | 世界构造复用 `closed_loop._build_world`，旧路径逐位复现 |
| 训练跑**真闭环** | 固定 `runtime_mode="plan_controlled_feedback"` |
| 智能体不越权 | `rl_resource/{obs,actions,env}.py` 不 import 真值层（AST）+ **全 idle 动作下测量/融合/消息恒为 0**（运行时） |
| 结构化动作，不做 4^N 展平 | 逐节点因素化分类，参数与节点数**线性** |
| 只用现有任务类型 | 计划内 `kind ∈ {SAMPLE, PROCESS, SHARE}` |

信息边界与规则基线**完全一致**：只读 `CentralObservation` 与任务队列，
不读真值、不读未来消息、不读偏差标签、不读账本内部字段。

### 11L.3 合法动作 mask 与"mask 是承重的"

| 动作 | 合法条件 |
| --- | --- |
| `idle` | **恒合法** |
| `sample` | 节点可用 + 队列里有该节点待处理采样任务 + 预算够 |
| `process` | 上述 + **真的有数据可处理**（本地缓冲非空或远端消息已到达） |
| `share` | 上述 + **outbox 非空** |

后两条刻意不用"队列里有任务就放行"这种偷懒判据——那会选到必然空转的动作。

实测（validation 分区）：

| 指标 | 数值 |
| --- | --- |
| 带 mask 的非法动作率 | **0.0000**（构造保证） |
| 执行器拒绝率 | **0.0000** |
| mask 覆盖率 idle/sample/process/share | 1.000 / 0.965 / 0.597 / **0.326** |
| 不带 mask 的 argmax 非法率 | 1.0000 |
| 不带 mask 的非法动作概率质量 | 0.70 |

⚠️ **mask 是承重的**：只靠 mask 训练的 PPO **没有**学会规避非法动作——
被掩掉的 logit 拿不到梯度，argmax 比的其实是未训练初值，所以"不带 mask 时
100% 非法"是**必然结果**，不是"策略学坏了"。**部署必须带 mask**；
三个数（带 mask 的 0 / 执行器拒绝率 / 不带 mask 的反事实）要一起报。

### 11L.4 奖励、代价与终止

```
completion = +1.0 × 本 tick 真正 APPLIED 的任务数
expiry     = −1.0 × 本 tick 过期任务数
invalid    = −0.5 × 本 tick 被执行器拒绝的任务数
waiting    = −0.25 × 超阈值仍未被服务的任务数 / 节点数
terminal   = −1.5 × 自然终止时仍未解决的任务数（**外部截断不补计**）
```

约束代价 `c_t = mean_{节点,单位}(Δconsumed[u]/capacity[u])`，
因此 **`Σ_t c_t` 必须等于账本重算的资源消耗**——每个 episode 都校验
（实测最大误差 **2.68e-10**）。

终止：任务时域到达 → `terminated`；剩余任务均不可执行 → `terminated`；
外部步数上限 → `truncated`（**保留 bootstrap**、**不加终端补计**）。
`info["bootstrap_allowed"]` 恒等于 `not terminated`。

⚠️ **与 `docs/learning_protocol.md` §2.2 的一处刻意差别**：
`all_tasks_resolved` **不作为**本环境的终止条件——闭环任务由实时观测逐 tick
派生，第 1 个 tick 只派生采样任务，沿用该条会让**每个 episode 都在第 1 个 tick
结束**（实测如此）。空队列在闭环里是**瞬态**，不是"问题已解决"。

### 11L.5 场景目录映射（**首次实现**）

`config/learning_splits_v1.json` 的 `scenario_catalog` 此前**只被校验、
从未被任何环境消费**。`rl_resource/scenarios.py` 第一次给出可执行语义
（用独立的映射版本 + 摘要标识，**不修改**冻结登记表）：

| 字段 | 映射 |
| --- | --- |
| `budget_multiplier` | 逐节点容量 × 系数 |
| `node_outage_windows` | `unavailable_windows`，作用于**远端节点**（目录未写节点，作用在远端才制造覆盖/交接压力） |
| `comm_delay_ticks` / `comm_drop_probability` | 受限链路 + 延迟/丢包 |
| `sensor_bias_sigma_multiplier` | 远端节点的测量偏置 + **自报 σ 低估**（来源不一致，只污染测量） |
| `load_multiplier` | **截止余量 ÷ 系数**（服务压力） |

⚠️ **`load_multiplier` 不是"任务数量倍数"**：闭环每节点每 tick 的候选类型
上限就是 3 种，**结构上做不到**按倍数增加任务（实测目标数 1→3 只让任务数
从 54 变到 56）。不得读成"任务多了 40%"。

### 11L.6 实测：确实在学，且资源守恒

24 次更新 × 8 episode × 24 tick：

| update | 训练回报 | validation 回报 | validation 资源消耗 | 熵 | 守恒 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | −66.53 | −64.33 | 0.2417 | 2.363 | ✅ |
| 8 | −57.02 | −4.00 | 0.3871 | 2.227 | ✅ |
| 16 | −44.83 | **−2.67** | 0.5305 | 1.929 | ✅ |
| 21 | **−28.77** | −14.50 | 0.4814 | 1.558 | ✅ |
| 24 | −33.67 | −4.00 | 0.3871 | 1.560 | ✅ |

学习轨迹可解释：前 5 次更新先学会"少做事"（资源消耗一度降到 0.0000），
随后发现**花资源把任务做完更划算**（消耗升到 0.53、回报升到 −2.7）。
执行器拒绝率 0.0000、PLAN_FATAL 0 次。

⚠️ **这不构成"优于规则基线"的结论**：本层回报是自定的学习信号，规则基线
不优化它；同口径比较需先用同一评价向量重跑两边（**未做**）。

### 11L.7 两个被修掉的实现缺陷

| 缺陷 | 症状 | 根因 |
| --- | --- | --- |
| **episode 计划是固定长度列表** | 第 6 次更新起 `trans 0`，后 15 次"训练"全是空转（entropy/kl 恒 0），"改善"只是验证噪声 | 计划跑完就没了；改为可按全局 episode 序号**无限延伸**的纯函数 |
| **评测用随机采样动作** | 同一策略两次采样的验证回报差 ±7，与学习信号同量级 → checkpoint 选择变成抽奖 | 评测与选择改用**确定性 argmax** |

### 11L.8 复现

```powershell
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train --smoke
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.train --rollout-episodes 8 --updates 24 --tag baseline
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.evaluate --policy output/rl_resource/baseline/policy.pt
D:\anaconda\envs\pytorch_env\python.exe -m unittest tests.test_resource_rl -v
```

产物在 `output/rl_resource/<tag>/`（`summary.json` / `training_curve.csv` /
`selection.json` / `policy.pt`，checkpoint 内嵌场景映射摘要、奖励版本、
数据划分摘要、训练与评测分区）。

### 11L.9 本层**没有**做到的事

1. **没有与规则/优化参考做同口径对比**（回报量纲不同，需重跑）；
2. **没有多种子统计**：训练 2 个种子起步，**不宣称统计显著性**；
3. **没有超参搜索**：PPO 超参是文献常用值；
4. **策略只在 mask 下有效**，合法性没有内化；
5. **没有多智能体**（只有一个中央智能体）；
6. **没有接 AI 解释层**；
7. 本节训练阶段的 test 曾封存；正式一次性解封与冻结记录见 §11N，
   **test 结果不得用于回调任何选择**。

---

## 11M. 信息新鲜度与不确定度感知调度：四组同结构消融（大阶段二第二步）

> **结论：§11J 的假设仍未被支持，负结果保留。**
> 独立报告见 `docs/resource_rl_ablation.md`，机器可读结果在
> `output/rl_resource/ablation/ablation_report.{json,md}`。

### 11M.1 与 §11J 的关系：条件变了，必须重验

旧 §11J 的负结果是在**「调度不影响感知」的旧闭环**下得到的（当时四组位置
RMSE 完全相同）。真闭环（§11K）接通后调度会改变测量与融合，因此本轮在
新条件下**重新检验同一假设**，而不是沿用旧结论。

判据与 §11J 原始表述一致（**先声明、后执行**）：
**组合特征组相对主基线，最差完成率不下降 且 资源消耗不增加。**

### 11M.2 冻结了什么（不得按结果调整）

奖励函数 / Actor-Critic 结构 / 隐藏层 `[128,128]` / Adam `lr=3e-4` /
训练预算（12 updates × 8 episodes × 24 tick）/ PPO·GAE 全部参数
（`γ=0.99`、`λ=0.95`、`clip=0.2`、`entropy=0.01`、`epochs=4`、`batch=64`、
`target_kl=0.03`）/ 结构化动作空间 / 合法动作 mask / train-validation 划分 /
`plan_controlled_feedback` 真闭环。

每次运行前 `assert_frozen()` **逐项断言**，改任何一项直接报错。
**唯一变化的是算法可见输入**：四组同形 **104 维**，被消融的特征**置零**
而非删除（复用 `information_research.CentralResearchFeatureAdapter`）。

### 11M.3 结果（validation，确定性 masked argmax）

| 组别 | 回报 | 完成率 | **最差完成率** | 及时性 | 平均等待(s) | 过期数 | 估计质量 | **资源消耗** | 通信(B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_baseline` | −8.667 | 0.6004 | **0.5854** | 0.8330 | 1.126 | 31.0 | 0.1790 | **0.4106** | 1621 |
| `freshness_only` | **−7.500** | 0.5980 | 0.5783 | 0.8469 | 1.112 | 31.3 | 0.1796 | 0.4101 | 1621 |
| `uncertainty_only` | −17.833 | 0.5270 | 0.5161 | **0.8910** | **1.076** | 41.7 | **0.2110** | 0.4564 | **427** |
| `freshness_uncertainty` | −17.833 | 0.5270 | 0.5161 | **0.8910** | **1.076** | 41.7 | **0.2110** | 0.4564 | **427** |

最差完成率 `0.5854 → 0.5161`（**下降**）、资源消耗 `0.4106 → 0.4564`
（**增加**）→ **两个条件都不满足，假设未被支持**。

### 11M.4 三条必须分开读的发现

1. **不确定度信息有真实的、可复现的行为影响**（不是噪声）：与主基线从
   **第 2 个 tick 起动作就不同**，整体是一次**多目标取舍**——及时性
   0.833→0.891、平均等待 1.126→1.076 s、估计质量 0.179→0.211、
   通信 1621→**427 B（↓74%）**，代价是完成率 0.600→0.527、过期 31→42、
   资源 0.411→0.456。方向与判据相反，**不能说它"更好"**。
2. **在不确定度之上再加新鲜度：决策层面完全无差别**。
   `uncertainty_only` 与 `freshness_uncertainty` 八项指标**逐位相同**。
   这不是 arm 没生效——两组观测在第 6 个 tick 有 **7 个槽位不同**；
   决定性证据是两组策略在 3 个验证场景 × **72 个决策步上的动作序列逐位相同**。
   即：不确定度块已覆盖新鲜度在这批场景里能提供的信息。
3. **只开新鲜度：回报略高但主指标未改善**（完成率持平、最差完成率略降）。
   这是 2 训练种子 × 12 update 的运行，validation 只有 9 个格子，
   差值远小于采样波动——**不构成"新鲜度有用"的证据**。

### 11M.5 mask 仍是承重的（三组指标一起报）

| 组别 | masked_invalid_rate | unmasked_argmax_invalid_rate | invalid_probability_mass | 执行器拒绝率 |
| --- | ---: | ---: | ---: | ---: |
| `main_baseline` | **0.0000** | 0.6250 | 0.5572 | 0.0000 |
| `freshness_only` | **0.0000** | 0.6389 | 0.6131 | 0.0000 |
| `uncertainty_only` | **0.0000** | 0.6250 | 0.5540 | 0.0000 |
| `freshness_uncertainty` | **0.0000** | 0.6319 | 0.5455 | 0.0000 |

`masked_invalid_rate = 0` 是构造保证；另两个数说明**四组都没有把合法性内化**。
**所有组正式部署必须带 mask**，三个数一起报，不允许只报第一个。

### 11M.6 「额外信息带来的收益」≠「PPO 本身带来的收益」

* **本报告能回答**：在**同一套 PPO、同一预算、同一 episode 序列**下，
  换掉算法可见输入会不会改变结果；
* **本报告不能回答**：PPO 本身比规则/优化参考好多少——学习回报是自定信号，
  规则基线不优化它；要做那个比较必须把三者放到**同一评价向量**上重跑，
  **未做**（§11L.9 第 1 条不变）。

### 11M.7 复现

```powershell
D:\anaconda\envs\pytorch_env\python.exe -m rl_resource.ablation
D:\anaconda\envs\pytorch_env\python.exe -m unittest tests.test_resource_ablation -v
```

### 11M.8 本阶段**没有**做到的事

1. **没有统计显著性**：训练 2 种子 × 12 update、validation 9 个格子，
   所有差值是**描述性**的；
2. **没有找出新鲜度无效的原因**：只证明了"在这批场景与预算下它不改变决策"，
   没有做特征重要性分析或梯度归因；
3. **没有测其它场景分布**：训练只用 4 个训练场景，泛化结论限于
   交接 / 节点掉线 / 传感器偏差三类验证场景；
4. **没有把合法性内化**：四组仍依赖 mask；
5. 消融训练阶段未解封 test；正式一次性解封与结论冻结见 §11N，
   且没有用 test 结果回调任何选择；
6. **没有修改**冻结契约、奖励、网络、优化器、动作空间与闭环——
   也**没有**通过挑选种子制造优势。

---

## 11N. 统一公平评测与结论冻结（大阶段二 · 第三步）

> **最终口径：不设综合分、不宣布总冠军。** 七种方法在同一
> `plan_controlled_feedback` 真闭环、同一任务队列、资源预算、通信条件、
> 算法可见信息与执行器上重跑；PPO 不读取未来或真值。

### 11N.1 冻结输入与评测矩阵

冻结 `resource-contract-v1`、场景划分、动作 mask、两个 checkpoint
（`main_baseline` 与 `freshness_uncertainty`）、评测源码和
`ExecutionPlan → RuntimeExecutor` 链路。先完成 validation 的 7 方法 ×
3 场景 × 3 环境种子 = **63 格**预检，生成输入 SHA-256 清单，随后才显式
一次性解封 test：7 方法 × 3 场景 × 5 环境种子 = **105 格**。

方法为 `round_robin`、`edf`、`rule`、`enumeration`、`rolling_horizon`、
`PPO-baseline`、`PPO-freshness_uncertainty`。所有格均通过资源守恒、零资源
违反、无重复运行时执行、Fusion 只消费实际到达测量与无真值载荷检查。

### 11N.2 test 描述性结果（跨场景平均）

| 方法 | 完成度 | 及时性 | 估计质量 | 资源消耗 | 通信(B) | 计算(s) | 等待(s) | 过期任务 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 0.4392 | 0.8621 | 0.1540 | 0.5288 | 256.0 | 0.0037 | 3.314 | 58.6 |
| EDF | 0.5323 | 0.3314 | 0.1523 | 0.2756 | 2346.7 | 0.0035 | 2.490 | 40.1 |
| rule | 0.4980 | 0.6943 | 0.1769 | 0.4558 | 42.7 | 0.0054 | 2.626 | 46.3 |
| enumeration | 0.4980 | 0.8547 | 0.1116 | 0.2743 | 3541.3 | 0.2555 | 1.848 | 46.7 |
| rolling_horizon | 0.5351 | 0.9919 | 0.1634 | 0.3747 | 1792.0 | 0.2385 | 1.572 | 36.3 |
| PPO-baseline | **0.5885** | 0.8237 | 0.1787 | 0.4195 | 1732.3 | 0.0106 | 1.070 | **32.1** |
| PPO-freshness_uncertainty | 0.5179 | 0.8609 | **0.2106** | 0.4698 | 554.7 | 0.0112 | **1.028** | 42.7 |

这些跨场景均值仅用于描述；正式比较同时给出同场景、同环境种子的配对差异与
95% 置信区间，见 JSON。PPO-baseline 在每个 test 场景中都比规则参考有更高
完成度、更少过期和更短等待，但通信代价更高；组合特征 PPO 有更高估计质量与更短
等待，却没有稳定的完成度收益。优化参考在及时性/资源/计算开销之间也呈现明显取舍。

### 11N.3 必须保留的边界

* 每个 PPO 候选只有 **1 个冻结训练 checkpoint**：训练随机性及其置信区间
  **不可估计**，环境种子统计不能冒充多训练种子统计；
* PPO 的带 mask 非法率为构造性的 0，不能写成策略自行学会约束。test 中
  PPO-baseline 的去 mask 非法 argmax 率为 0.5069、组合 PPO 为 0.4306；
* test 只在冻结后释放，结果不再用于调奖励、挑 checkpoint、换种子或选模型；
* 无单一综合得分，也不把任何方法写成所有场景/指标下的"总冠军"。

复现和审计产物：`tools/evaluate_unified_resource_methods.py`、
`output/unified_resource_evaluation/{freeze_manifest.json,final_release.json,\
test_report.json,test_per_episode.csv,test_report.html}`，以及
`docs/unified_evaluation_protocol.md`、`docs/unified_evaluation_conclusions.md`。

```powershell
# 仅在已冻结选择后执行；test 必须显式解封
D:\anaconda\envs\pytorch_env\python.exe tools/evaluate_unified_resource_methods.py --split validation
D:\anaconda\envs\pytorch_env\python.exe tools/evaluate_unified_resource_methods.py --split test --release-test
```

---

## 11O. Balanced Resource Scheduling Mode（固定均衡锚点）

Balanced 是为后续 preference-conditioned/Pareto 调度准备的**固定参考点**，不是
总体最优声明。它固定等权 `[0.2,0.2,0.2,0.2,0.2]`，只优化完成度、及时性、
估计质量、资源节约和通信节约；计算耗时只作评测指标。归一化边界在
`config/balanced_protocol_v1.json` 预冻结，禁止按任何 test 动态 min-max。

三训练 seed 的 validation 机制验证显示它退化为偏节约/及时性的策略：完成率
0.27，低于 rule 的 0.51；虽然资源消耗为 0.20、等待 0.34 s，但这不是均衡成功。
idle 占 59%，去 mask 非法 argmax 仍为 0.49，说明 mask 仍是承重约束。
**失败结果保留，未改权重、奖励、PPO 参数、网络、动作、mask、预算或 seed 补救。**
test-v3 已 SHA 封存且未读取/解封。完整口径、checkpoint 哈希与 CSV/JSON 见
`docs/balanced_resource_mode.md` 和 `output/rl_resource/balanced/`。

---

## 11P. Preference-Conditioned PPO（机制未通过，test-v3 保持封存）

单一 PPO 在输入末尾拼接五维偏好，奖励复用 §11O 的固定效用定义。三 seed validation
显示完成/资源偏好存在有限响应，但所有偏好均未执行 share、通信量恒为零，通信偏好
机制未学成。因此**不得**解封 test-v3、不得宣称 Pareto 前沿或偏好控制成功，也不得
通过改权重、奖励、网络、PPO 参数、mask 或挑偏好补救。详见
`docs/preference_conditioned_ppo.md`。

---

## 11Q. Share 机制诊断（v1 失败定位；不运行 test）

独立诊断确认：share 并非一直不存在或被 mask 屏蔽。在三个冻结模型、十个偏好上，
train/validation 分别有 **31.35% / 30.60%** 的节点决策步出现 share 候选，合法率为
100.00% / 97.96%，但 **7,497 个合法状态的 masked argmax 均未选 share**。因此通信偏好
失败不能归因于“没有 share 动作”。

同状态反事实和最小真闭环案例均确认：强制合法 share 会真实发送 128 B、消耗同额通信
账本、使接收节点在消息到达后获得 process 可行性并消费远端测量；资源守恒、无重复执行
和无真值载荷检查都通过。根因是冻结 v1 中的即时通信节约惩罚、延迟的
`share → arrival → process` 收益，以及自动候选要求先 `sample → process → share` 的
门控链共同作用，而不是通信/融合链未接通。

**仍不得**修改 v1、解封 test-v3 或直接启动 `preference_ppo_v2`；任何后继改变都必须
新建版本化协议并重新封存 train/validation/test-v4。完整漏斗、反事实、奖励分解和最小
场景证据见 `docs/share_mechanism_diagnosis.md`，原始 JSON 为
`output/share_mechanism_diagnosis/share_diagnosis.json`。

---

## 11R. Preference-Conditioned PPO v2（机制验证失败；test-v4 封存）

v2 作为**独立协议**保留 v1 的失败结果，只针对已定位的 share 信用分配问题改动：将
128 B 通信代价改为预冻结的平滑单调函数，并在消息真实到达、远端实际 process 后以
仅含航迹年龄/协方差的可观测代理提供延迟估计质量收益；不读取真值、未来或离线标签。
同时只有 v2 允许 raw outbox 直接产生 share 候选，保持 `RuntimeExecutor` 是唯一副作
入口、资源守恒和 action mask 不变。

三预声明训练 seed 的 270 格 validation **未通过**机制闸门：seed 1009 从未选择 share；
通信节约偏好反而有 90 次 share / 426.67 B，而估计质量偏好为 0 / 0，方向与假设相反。
因此停止在 validation，**不解封 test-v4、不调参、不重训、不宣称 Pareto 或偏好控制成功**。
完整协议、checkpoint 哈希、原始 CSV/JSON 和边界见 `docs/preference_ppo_v2.md`。

---

## 11S. Preference Controllability Audit（暂停 v2；不训练、不读 test-v4）

审计先将“无可见航迹/协方差”的信息年龄和 σ 从 `0` 修正为 `null + not_applicable`，
避免把没有信息写成最好信息；v2 reward 的空集估计质量本来为 0，不存在同类奖励漏洞。
随后在四个冻结 train/validation 状态中，仅替换五维偏好、枚举首个
idle/sample/process/share 动作，并用固定 3-tick 后续脚本做反事实。

结果明确区分了两类问题：奖励方向正确——通信偏好令 share 相对 idle 回报为 −0.6584，
质量偏好在远端信息有价值时为 +0.5228，资源偏好降低高成本 sample 回报；但三个冻结 PPO
没有学会相同方向，质量偏好下 share 概率 0.1750 反而略低于通信偏好 0.1777，二者 argmax
都没有 share。因此结论是**奖励因果方向正确、策略偏好—动作映射未学成**，而非通信链、
mask 或 reward 符号错误。v2 继续暂停，test-v4 保持封存；审计不自动授权 v3。
详见 `docs/preference_controllability_audit.md` 和
`output/preference_controllability_audit/`。

---

## 11T. Preference-Conditioned PPO v3（FiLM 表示机制未通过；test-v5 封存）

基于 11S 的结论，v3 **只**替换偏好条件化表示：104 维可见状态继续经 `[128,128]` 主干，
五维 simplex 偏好经独立 `5→8` encoder，以有界 FiLM scale/shift 调制每层隐藏特征；不改
v2 已验证方向正确的五目标 reward、通信代价、RuntimeExecutor、物理模型、动作、mask 或
PPO。三个冻结 concat-v2 checkpoint 继续作为严格基线，未被重训或覆盖。

新建、冻结的 test-v5 未被读取。三个预声明 FiLM seed 在固定远端信息有价值状态上都有
非零的“质量偏好 vs 通信偏好”raw-logit 有限差分（0.01514、0.01961、0.03366），说明网络
数值上看到了偏好；但仅 seed 1117 在质量偏好提升协同/share、通信偏好降低 share 的方向
上成立，1103/1109 反向，且资源/完成服务方向也不能跨 seed 稳定复现。因此预注册机制
闸门失败：这**不是**偏好控制成功的证据，不能解封 test-v5、不能宣称 Pareto 前沿，也不再
通过改 reward、挑 seed 或扩大网络补救。Preference-Conditioned PPO 主线止于 v3，保留
v1/v2/v3 全部负结果。详见 `docs/preference_ppo_v3.md` 和
`output/rl_resource/preference_ppo_v3/`。

复现/审计入口：`config/preference_ppo_v3.json`、
`config/preference_ppo_v3_splits.json`、`tools/run_preference_ppo_v3.py`；原始
validation 行、固定状态 logits/概率矩阵、有限差分敏感度和 checkpoint 哈希均已随报告
保存。该入口**没有** `test` 子命令，防止在机制失败后意外读取 test-v5。

---

## 11U. Preference-Conditioned PPO 探索分支（已暂停）

状态：**🟡 已完成探索，机制验证未通过，当前暂停。** 这不是“未实现”：v1、机制诊断、
v2、固定状态可控性审计和 v3 FiLM 均已按各自冻结协议完成，并保留全部代码、配置、
checkpoint、SHA 与 validation 产物。

1. **v1**：五维偏好 concat 输入未能控制 `share`；诊断发现候选常出现且几乎合法，但
   7,497 个合法状态的 masked argmax 都不选 share。
2. **诊断**：强制合法 share 会真实发送、解锁远端 process，并改善可观测信息年龄/协方差；
   因而不是通信链路或 mask 失效。
3. **v2**：仅修正 share 的延迟信用归因、平滑通信代价和 v2-only 候选门控；share 开始出现，
   但偏好方向错误，机制闸门失败。
4. **可控性审计**：固定状态反事实证明 reward 对偏好的因果方向正确；冻结策略却未学会
   “偏好 → 动作”映射。
5. **v3**：FiLM 后 logits 会响应偏好，但跨三个训练 seed 仍无法稳定形成方向正确的动作控制。

因此唯一允许的结论是：**策略对偏好敏感，但当前证据不支持策略可被偏好稳定控制。**
不得据此生成正式 Pareto/test 结论；`test-v5` 继续封存，不读取、不解封。该失败也**不否定**
基础 PPO 资源调度基线：其多训练 seed 评测已独立复现固定预算下相对 rule 的完成率优势。

完整归档、SHA、路径和回归校验见 `docs/preference_ppo_archive.md`；运行
`python tools/verify_preference_ppo_archive.py` 可验证 v1–v3 未被回写、test-v5 仍封存，
以及基础 PPO 的独立结论仍存在。

---

## 12. 实验结果汇总

### 12.1 固定干扰场景，seed=42（`evaluate_dqn.py`）

| 策略 | 满足率* | 违反率 | 执行步数 | 平均功率 | 累计能耗 | 平均 Pint | 平均暴露 | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 固定功率基线(80 W) | 0.2951 | 0.7049 | 20 | 70.00 W | 1400.0 J | 0.9051 | 0.5637 | −3.1912 |
| **规则功率控制** | **0.9180** | 0.0820 | 58 | 24.14 W | 1400.0 J | 0.6432 | 0.4022 | **+0.3510** |
| 随机策略 | 0.4426 | 0.5574 | 61 | 22.93 W | 1399.0 J | 0.5502 | 0.3160 | −0.6289 |
| **DQN(贪心)** | **0.9180** | 0.0820 | 61 | **22.95 W** | 1400.0 J | **0.6246** | **0.3878** | **+0.3766** |
| 逐档贪心(短视) | 0.9180 | 0.0820 | 58 | 24.14 W | 1400.0 J | 0.6432 | 0.4022 | +0.3510 |
| 前瞻规划(非短视) | **0.9344** | **0.0656** | 61 | 22.89 W | 1396.5 J | 0.6068 | 0.3777 | **+0.4052** |

DQN 相对规则：功率 −4.93%、Pint −2.89%、暴露 −3.58%、综合收益 **+7.31%**，满足率持平。

### 12.2 多种子泛化（10 seed，含初始条件域随机化）

| 策略 | 满足率* | 平均功率 | 累计能耗 | 平均暴露 | 综合收益 |
| --- | --- | --- | --- | --- | --- |
| 规则功率控制 | 0.9459±0.0792 | 23.13±2.87 | 1347.0±78.1 | 0.3818±0.0286 | +0.4219±0.1947 |
| **DQN(贪心)** | 0.9410±0.0652 | **21.88±1.17** | **1334.9±71.6** | **0.3685±0.0158** | **+0.4491±0.1302** |
| 逐档贪心(短视) | 0.9541±0.0716 | 22.72±2.69 | 1335.2±91.5 | 0.3778±0.0312 | +0.4427±0.1811 |
| 前瞻规划(非短视) | 0.9721±0.0346 | 21.86±1.47 | 1333.3±90.0 | 0.3650±0.0195 | +0.4934±0.0928 |

配对差异（DQN − 基线，综合收益）：vs 规则 **+0.0272**（t=+0.944，不显著）、
vs 短视 +0.0064、vs 前瞻 −0.0443（t=−1.892）。
**DQN 的跨场景标准差 0.1302 低于规则 0.1947 与短视 0.1811**——更可预期。

### 12.3 泛化验证：域随机化的必要性

| 模型 | 固定场景收益 | 扰动 10 种子收益 | 相对规则 |
| --- | --- | --- | --- |
| 无扰动训练 | **+0.4329**（最好） | +0.3684±0.1462 | **−0.0535（输给规则）** |
| 域随机化训练（主模型） | +0.3766 | **+0.4491±0.1302** | **+0.0272（优于规则）** |

只在单个场景上报成绩会得出错误结论。**主模型选域随机化版本，论文以多场景均值为准。**

### 12.4 能量预算敏感性（1200/1300/1400/1500/1800 J）

| 预算 J | 规则 满足率 | DQN 满足率 | 前瞻 满足率 | 规则 收益 | DQN 收益 |
| --- | --- | --- | --- | --- | --- |
| 1200 | 0.7213 | **0.7869** | 0.7213 | −0.1178 | **+0.1220** |
| 1300 | 0.7869 | 0.7049 | **0.8361** | +0.0513 | −0.0304 |
| 1400 | 0.9180 | 0.9180 | **0.9344** | +0.3510 | +0.3264 |
| 1500 | **1.0000** | 0.8852 | **1.0000** | **+0.5291** | +0.3204 |
| 1800 | **1.0000** | **1.0000** | **1.0000** | **+0.5291** | **+0.5291** |

1. **预算 ≥1500 J 时约束不再 binding**（满足全部 61 步只需 1450 J），
   规则/短视/前瞻**完全收敛到同一个解**——反证了 1400 J 才是让时序权衡成立的关键；
2. **预算越紧，规划价值越大**：1200 J 时 DQN 收益 +0.1220，而三个基线全是 −0.1178；
3. DQN 曲线不平滑是**每个预算只训练 800 episode 的欠训练**所致（非预算的性质），
   脚本策略曲线才是干净的敏感性信号。

---

### 12.5 v4.0 P1：部分可观测的代价与历史窗口的作用

**设置**：5 个种子（42/7/13/21/33），初始条件域随机化开启，`--observation-preset moderate`。
三个学习臂**训练预算完全相同（1200 episode）**，因此差值可归因于观测模式；
主 DQN（4500 episode）另列，仅作参考。
命令：`evaluate_pomdp.py --observation-mode pomdp --no-lookahead --ablation`

#### A 组：全状态参考（读真值，信息量最高，只作上界）

| 策略 | 满足率 | 平均功率 W | 累计能耗 J | 平均 Pint | 平均暴露 | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- |
| 固定功率基线(80W) | 0.2951±0.0000 | 70.00 | 1400.0 | 0.9051 | 0.5637 | −3.1954 |
| 规则功率控制 | 0.9639±0.0524 | 23.06 | 1372.2 | 0.6252 | 0.3858 | +0.4636±0.1244 |
| 随机策略 | 0.4590±0.0000 | 22.93 | 1399.0 | 0.5502 | 0.3160 | −0.5866±0.0099 |
| 逐档贪心(短视) | 0.9770±0.0320 | 22.71 | 1361.4 | 0.6195 | 0.3806 | +0.4949±0.0832 |

#### B 组：仅观测（读带噪信念状态，与 DQN 同一信息水平）

| 策略 | 满足率 | 平均功率 W | 综合收益 |
| --- | --- | --- | --- |
| 规则(仅观测) | 0.7803±0.0791 | 23.56 | +0.1485±0.1215 |
| 逐档贪心(仅观测) | 0.7902±0.0798 | 23.03 | +0.1740±0.1047 |

#### C 组：学习型策略

| 策略 | 训练预算 | 满足率 | 平均功率 W | 累计能耗 J | 平均 Pint | 平均暴露 | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DQN(全可观) | 4500 ep | 0.9574±0.0220 | 22.30 | 1360.0 | 0.6119 | 0.3748 | +0.4739±0.0394 |
| DQN(全可观,匹配预算) | 1200 ep | 0.9574±0.0220 | 22.30 | 1360.0 | 0.6119 | 0.3748 | +0.4739±0.0394 |
| DQN(部分可观测) | 1200 ep | 0.8295±0.0340 | 25.43 | 1400.0 | 0.6704 | 0.4187 | +0.1238±0.0706 |
| **DQN(部分可观测+历史 K=4)** | 1200 ep | **0.9639±0.0243** | 22.64 | 1381.0 | 0.6163 | 0.3788 | **+0.4613±0.0612** |

#### 结论

1. **部分可观测的代价很大，而且主要落在「距离测量误差」上**：
   * 规则策略：`+0.4636 → +0.1485`（Δ −0.3151），满足率 `0.9639 → 0.7803`（**−18.4 个百分点**）；
   * 短视贪心：`+0.4949 → +0.1740`（Δ −0.3209），满足率 `0.9770 → 0.7902`（−18.7 个百分点）；
   * DQN（同预算）：`+0.4739 → +0.1238`（**Δ −0.3501**）。
2. **无记忆的 DQN 在部分可观测下不再优于规则**：`+0.1238` vs 规则(仅观测) `+0.1485`，
   Δ `−0.0247`。这与 v3.1「DQN 不劣于规则」的结论**方向相反**，
   说明那条结论高度依赖「智能体能看真值」这个隐含假设。
3. **历史窗口把性能基本补回来了**：`+0.1238 → +0.4613`（Δ **+0.3375**），
   接近全可观水平（`+0.4613` vs `+0.4739`，差 0.0126），
   且明显优于仅观测规则（`+0.4613` vs `+0.1485`，Δ +0.3128）。
   **机制解释（不要过度解读）**：历史窗口最直接的作用是**沿时间对独立测量噪声做平均**，
   把有效观测精度抬高；它**不是**学会了 POMDP 信念更新，也不是循环记忆。
   因此「历史窗口 = 学会了处理部分可观测」这种说法不成立；
   它只是用最简单的办法换回了信息。
4. **全可观策略在 1200 episode 就饱和**：4500 ep 与 1200 ep 两个全可观模型的评测结果
   **逐位相同**（满足率 0.9574、综合收益 +0.4739、平均功率 22.2951 W）。
   两个 checkpoint 文件的 SHA-256 不同（确为不同权重），
   说明在该场景下二者的贪心策略在这些种子上完全一致。
   这一条很重要：它保证了上面「同预算对照」不是因为 POMDP 臂训练不足而吃亏。

#### 逐项噪声消融（仅观测规则策略，moderate 为基准）

实现上把每路噪声改成**独立随机流**（`ObservationModel._field_rng`），
因此关掉一路不会扰动其他路的噪声实现——这是**配对单变量**对比，
比共用一条随机流时可信得多。

| 消融 | 满足率 | 综合收益 | 相对基准 |
| --- | --- | --- | --- |
| 全噪声（基准） | 0.7803 | +0.1485 | — |
| **只关距离噪声** | **0.9082** | **+0.3673** | **+0.2188** ← 主因 |
| 只关干扰噪声 | 0.7869 | +0.1871 | +0.0386 |
| 只关能量噪声 | 0.7803 | +0.1494 | +0.0009 |
| 只关暴露/Pint 噪声 | 0.7803 | +0.1485 | 0.0000 |
| 只关延迟 | 0.7738 | +0.1230 | −0.0255 |
| 只关丢测 | 0.7672 | +0.1084 | −0.0401 |
| 真值全开（仅保留延迟/丢测） | 0.7803 | +0.1485 | 0.0000 |

读法（**必须按这个口径解释**）：

* **距离/RCS 噪声是压倒性主因**（+0.2188），远超其他所有项之和；
* 「只关暴露/Pint」与「真值全开」**恰好等于基准**，这不是 bug：
  规则策略只依据信念状态里的**距离、RCS、干扰、剩余能量**做决策，
  它**根本不读**暴露、Pint 与侦察机位置，所以关掉这些量的噪声不可能改变它的动作。
  这反过来是对信念桥接的一次有效校验——干预的量与决策依赖的量一致；
* 关掉延迟/丢测后综合收益反而**略降**（−0.026 / −0.040）：这两个负面效应很小且方向反直觉。
  在 5 个种子上不足以给出可靠解释，**不应**据此声称「延迟和丢测有益」。

#### 探测概率：AI 在部分可观测下仍然有效吗？

**是，但前提是给它时序记忆。** 历史窗口臂在 5 个种子上达到满足率 0.9639、
综合收益 +0.4613，与全可观 DQN 的 0.9574 / +0.4739 基本持平；
而单帧观测的 DQN 只有 0.8295 / +0.1238。结论是：
**POMDP 的困难不在于「算法不够强」，而在于「单帧观测信息不足」**。

---

### 12.6 v4.0 P2：不确定度感知的可信决策

**设置**：同上 5 个种子、moderate 预设；回退默认模式为安全护盾（`shield`）。
命令：`evaluate_uncertainty.py --threshold-sweep --signal-ablation`
集成模型：N=5，`bootstrap_prob=0.8`，共享回放缓冲区，1200 episode。

| 配置 | AI 自主率 | 干预率 | 高风险步占比 | 错误率(AI自主) | 错误率(被干预) | 护盾挽回率 | 干预变差率 | 高风险满足率 | 低风险满足率 | 综合收益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则(仅观测) | 1.000 | 0.000 | — | 0.181 | n/a | n/a | n/a | n/a | n/a | +0.148 |
| 普通DQN(部分可观测) | 1.000 | 0.000 | — | 0.082 | n/a | n/a | n/a | n/a | n/a | +0.124 |
| 集成DQN(纯argmax) | 1.000 | 0.000 | 0.000 | 0.100 | n/a | n/a | n/a | n/a | 0.900 | +0.235 |
| 集成DQN+安全护盾 | 0.934 | 0.066 | 0.520 | 0.055 | 0.233 | **0.080** | **0.000** | 0.969 | 0.895 | +0.164 |
| 集成DQN+完全回退 | 0.384 | 0.616 | 0.616 | 0.023 | 0.204 | 0.000 | 0.155 | 0.796 | 0.977 | **+0.275** |

#### 结论（含负结果）

1. **只有「集成分歧」这一路信号真正在起作用**。
   三路信号消融（安全护盾模式）显示：

   | 使用的信号 | 自主率 | 高风险步占比 | 综合收益 |
   | --- | --- | --- | --- |
   | 只用集成分歧 | 0.934 | 0.491 | +0.164 |
   | 只用 OOD 评分 | 1.000 | **0.000** | +0.235 |
   | 只用观测质量 | 0.986 | 0.072 | +0.220 |
   | 三路全用（默认） | 0.934 | 0.520 | +0.164 |
   | 全部关闭（对照） | 1.000 | 0.000 | +0.235 |

   **负结果：OOD 评分从未触发过。** 「只用 OOD」与「全部关闭」的结果**完全一致**。
   实测平均 OOD 评分只有 0.706~0.721，远低于阈值 3.0——
   原因是训练与评测用了同一套初始条件域随机化，评测分布落在训练分布之内，
   检测器根本没有可检出的偏移。**在当前实验设定下，OOD 这一路是无效的**，
   不应把它算作本方案的贡献。要让它有意义，需要真正的外推场景
   （例如评测分布超出域随机化范围）。
2. **安全护盾：有效，但收益有限，且有代价。**
   单步反事实显示挽回率 8.0%、变差率 **0.0%**——护盾只抬功率不降功率，
   所以从不把一步从「达标」变成「不达标」，这一点与设计一致。
   但**整段 episode 的任务满足率反而从 0.875 降到 0.839**：
   抬功率会多耗能量，使能量预算提前耗尽，未执行的步骤按口径计入未满足。
   这是一个真实的**单步收益与整段代价之间的冲突**，
   只看单步反事实会得出「护盾有益」的片面结论。
3. **完全回退拿到最高的综合收益（+0.275 vs 纯 argmax +0.235），
   但错误率更高（0.135 vs 0.100），且 15.5% 的干预步比 AI 原动作更差。**
   它的收益来自规则策略本身更省功率（对 LPI 目标更友好），
   而不是来自「AI 知道自己不行」。因此**不能**把它宣传成「AI 学会了求助」。
4. **触发条件确实定位到了困难状态**：完全回退模式下高风险步满足率 0.796，
   低风险步 0.977（高风险步占比 0.616，落在可解读区间内）。
   这是「不确定度信号有信息量」的正向证据。
5. **回退率由阈值决定，而非智能体的内省能力**。阈值灵敏度扫描（安全护盾）：

   | 阈值档 | 自主率 | 干预率 | 综合收益 |
   | --- | --- | --- | --- |
   | 松（几乎不回退） | 0.979 | 0.021 | +0.225 |
   | 中（默认） | 0.934 | 0.066 | +0.164 |
   | 紧（频繁回退） | 0.864 | 0.136 | +0.094 |

   回退越多、综合收益越低。任何「回退率 X% 说明 AI 有多谨慎」的说法都必须
   同时给出这张表，否则无法判断 X% 是能力还是设定。
6. **错误决策率不能直接跨组比较**：被干预步的错误率（0.233）高于 AI 自主步（0.055），
   这是触发条件在困难步上命中的**选择性偏差**，不是回退失败。
   脚本在结论里显式写明了这一点。

---

## 13. 版本演进

| 维度 | v1 | v2 | v3 | v3.1 | v4.0 | **v4.1 → v4.5（当前）** |
| --- | --- | --- | --- | --- | --- | --- |
| MDP 结构 | contextual bandit | 时序决策 | 时序决策 | 时序决策 | **POMDP**（部分可观测） |
| 状态可见性 | 真值 | 真值 | 真值 | 真值 | 真值 / **带噪估计 + 置信区间**（可切换） |
| 能量约束 | 软约束 | 硬约束（事后判定） | **执行前拦截** | 执行前拦截 | 执行前拦截（不变） |
| DQN | 标准 | 标准 | **Masked** | Masked + **拉格朗日约束分支** | Masked + **历史窗口分支** + **集成（Bootstrapped）** |
| 对手 | 固定时间窗 | 固定时间窗 | 固定时间窗 | + **规则自适应智能干扰机** | 同 v3.1（学习型对手见 §16 后续工作） |
| 可解释性 | — | — | — | **反事实证据 + AI 自然语言** | + **不确定度与回退原因解释** |
| 认知诊断 | — | — | — | **AI 层（四能力 + HTTP API）** | + **观测链路与可信决策两节** |
| 可信决策 | — | — | — | — | **不确定度监测 + 安全护盾/回退** |
| 实验配置 | 各脚本自定义 | 各脚本自定义 | 收敛到 `experiment_config.py` | 同 v3 | 同 v3.1 |
| 评价方式 | 单种子 | 单种子 | 单种子 + 10 种子 + 敏感性 | 同 v3 + 对手模式对照 | 同 v3.1 + **观测模式/噪声消融/阈值灵敏度** |
| 项目名 | — | — | — | **LPI-CogRadar** | LPI-CogRadar | LPI-CogRadar |
| 场景实体模型 | 各自的 x,y | 各自的 x,y | 各自的 x,y | 各自的 x,y | 各自的 x,y | **统一 `SceneEntity`**（唯一 ID／三维位置与速度／航向姿态／时间戳／平台归属） |
| 几何计算 | 各模型各写一份 hypot | 同左 | 同左 | 同左 | 同左（4 份重复实现） | **收口到 `engine/geometry.py` 唯一实现** |
| 几何关系语义 | — | — | — | — | — | **有向 `GeometricRelation`**（谁相对谁、哪一时刻；距离/方位/俯仰/径向速度/视线） |
| 场景真值 | 聚合值（最近/最远距离等） | 同左 | 同左 | 同左 | 同左 | **显式逐条关系**（聚合值降级为派生视图） |
| 平台数 | 1 雷达 | 1 雷达 | 1 雷达 | 1 雷达 | 1 雷达 | **多雷达几何**（物理仍只作用于主雷达） |
| 场景快照导出 | — | — | — | — | — | **CSV + JSON**（实体 + 有向关系） |
| 测量层（v4.2） | — | — | — | — | 单一 `dropout_prob` 代表全部缺失 | **`真值 → 可见性 → 测量 → 观测` 四层**；六种"没有数据"原因显式区分；逐条测量带时间戳/协方差/置信度 |
| 通信层（v4.3） | — | — | — | — | — | **三种共享策略**；消息生成/发送/到达时刻、固定+随机延迟、丢包、带宽、过期；只读已到达消息 |
| 融合层（v4.4） | — | — | — | — | 按置信度取前 K 条 | **真正的跟踪器**：门限 + 最近邻关联 + 常速度卡尔曼、稳定 `track_id`、外推与删除 |
| 协同验证（v4.4） | — | — | — | — | — | **三类场景**（共同可见 / 遮挡视场 / 通信受限）+ 利用率/覆盖/连续性/精度/延迟指标 |
| AI 证据链（v4.5） | — | — | — | 四能力（功率/能量/暴露） | + 观测链路与可信决策两节 | + **测量/通信/融合/协同四节证据 + 17 个新发现码 + 两个只读接口 + 证据校验与回退** |
| AI 控制权限 | — | — | — | 无 | 无 | **无**（v4.5 仍是只读诊断；源码扫描测试钉住） |
| 多目标压力测试（v4.5 P2） | — | — | — | — | — | **四类场景 + 关联层审计 + 十项指标 + 四路对照**（§11G） |
| 跟踪器生命周期 | — | — | — | — | — | **miss→coasting→删除真的生效**（v4.5 修掉"死代码"bug，见 §11G.5） |
| JPDA-lite 关联 | — | — | — | — | — | **未实现**（压力测试已触发告警阈值，见 §11G.6） |
| 资源管理层（大阶段一） | — | — | — | — | — | **单位/时钟/两阶段校验/逐节点账本**；守恒式恒成立；失败不吞资源（§11H.2） |
| 融合→调度适配（大阶段一） | — | — | — | — | — | **独立 schema `rm-obs-1.0`**（25 字段标注单位/坐标系/可见范围/来源）；断开远端后中央视图**冻结**（§11H.3） |
| 非学习规则基线（大阶段一） | — | — | — | — | — | **轮询 / EDF / 规则三策略共用同一观测·队列·执行器**，产生不同分工且逐条可追溯（§11H.4） |
| 非学习优化参考（大阶段一） | — | — | — | — | — | **完全枚举 + 滚动规划**；显式预测模型 + 声明式目标 + 计算预算；**不读未来**（§11H.6） |
| 评价方式（大阶段一） | 单种子 | 单种子 | 单种子 + 10 种子 + 敏感性 | 同 v3 + 对手模式对照 | 同 v3.1 + 观测模式/噪声消融 | + **六维向量**（完成度/及时性/估计质量/资源消耗/通信开销/计算耗时），**不给综合分**（§11H.7） |
| 口径冻结（大阶段一） | — | — | — | — | — | **`resource-contract-v1`**：改一处即校验失败，须升版本号 + 同步文档 + 重跑验收（§11H.9） |
| 学习算法接入 | — | — | — | — | — | **已接入第一步**（§11L）：中央 Actor-Critic 输出标准 `ExecutionPlan`，真闭环训练，资源守恒 |
| 学习协议（§11I） | — | — | — | — | — | **终止/截断/bootstrap 语义冻结**；任务自然终点 = `terminated`、外部步数上限 = `truncated`；四个可手算精确案例；数据划分**封存**（`SealedTestSplitError`） |
| 拉格朗日语义（§11I.1） | — | — | — | — | 部署用 `argmax(Q_r−λQ_c)`，训练时双 critic **各自取最大** | **双 critic 共用部署策略的同一个拉格朗日贪心动作**（修复未重训，历史负结果保留但归因降级） |
| 研究分支（§11J） | — | — | — | — | — | **唯一可检验假设 + 四组同结构消融**（禁用特征置零而非删维度）；规则基线获得**完全相同**的信息字段 |
| 调度→感知闭环（§11K） | — | — | — | — | 调度只改任务/资源统计，**不改变航迹质量** | **唯一 `ExecutionPlan → RuntimeExecutor`**：只有 sample 任务才触发传感器、未调度节点只预测（年龄/σ 自然增长）、只有 share 才真实发送并占用通信资源、融合只消费已到达测量 |

### 13.1 向后兼容性（已实测）

* **固定干扰场景的全部数值逐位未变**：固定 80 W `0.2951/70.00/1400.0/0.9051/0.5637/−3.1912`、
  规则 `0.9180/24.1379/1400.0/0.6432/0.4022/+0.3510`、
  随机 `0.4426/22.9344/1399.0/0.5502/0.3160/−0.6289`——与 v3.1 完全相同；
* `evaluate_dqn.py` 用 `output/rl/dqn_agent_best.pt` 复现出
  `0.9180/22.9508/1400.0/0.6246/0.3878/+0.3766`，与 v3.1 完全一致；
* 观测模式默认 `full`，**不传参即等价于 v3.1**；旧 checkpoint（观测 12 维）可直接加载评测；
* 自适应干扰机默认**关闭**；安全 RL 默认**关闭**；AI 层失败可降级到无 AI；
* `output/rl_v1/` 保留第一版产物用于跨版本对照。
* **大阶段一（资源管理）不接入任何旧实验路径**：新增模块只在
  `python -m resource_management` / `tools/compare_schedulers.py` /
  `evaluate_resource_management.py` / `tools/evaluate_information_research.py`
  四个入口被调用；历史入口的源码扫描仍为空（`verify_v4.py` §15
  「历史实验入口未引用 resource_management」）。
  完整/部分可观测/理想/真实四条观测路径与旧数字**逐位不变**。
* **真闭环是显式开关、默认关闭**：`runtime_mode` 默认
  `legacy_observation_first`（旧路径，逐位可复现）；
  新路径 `plan_controlled_feedback` 需显式传入。
  实测旧路径下 base 场景 `rule` 仍是 48 完成 / 13 按时 / 24 计划 / 0 B 通信 /
  `estimate_quality` 0.4985 / `resource_consumption` 0.2538。
* **学习协议不改旧环境默认行为之外的东西**：`LpiPowerEnv` 默认改为
  `horizon_semantics="finite_task"`，旧行为通过显式
  `legacy_truncation` 保留以复现旧 checkpoint（该选项**不得**用于新结论）。

### 13.2 本版**没有**改动的物理参数

```
雷达：peak_gain_db, sidelobe_gain_db, main_beam_width_deg, wavelength_m,
      bandwidth_hz, noise_figure_db, system_loss_db, temperature_k,
      snr50_db, pd_slope_db, required_pd
目标：TGT1/TGT2 的 RCS、速度（初始位置仅在多种子鲁棒性评测中做域随机化）
侦察：ESM1 的位置、gain_db, bandwidth_hz, noise_figure_db, system_loss_db,
      snr50_db, pint_slope_db
干扰：JAM1 的 peak_power_w(=200 W 额定), intensity, duty_cycle, fluctuation,
      fluctuation_step；自适应模式下的 0.6×/1.5× 是**动作定义**而非物理量改动
动作：power_levels_w（11 档）、fixed_power_level
```

本版只新增/改动**任务约束、对手行为、状态与奖励设计、算法分支、认知与解释模块**。

**v4.0 新增的 `observation` 配置段同样不触碰物理模型**：它只描述「智能体测到什么」，
真值演进、能量硬约束、奖励函数完全按原样计算。这一点有单元测试钉住
（`tests/test_pomdp_env.py::TestPomdpInvariant`：同一串动作在 full 与 pomdp 两个模式下，
真实轨迹与逐步奖励必须**完全相等**）。

---

## 13A. v3.1 → v4.0 实验结论差异（逐条对照）

这一节是回答"新版本到底多出了什么、哪些结论被推翻"的地方，**不复述 v3.1 已有的结论**。

| 议题 | v3.1 的结论 | v4.0 的新结论 | 依据 |
| --- | --- | --- | --- |
| 状态可见性假设 | 隐含假设智能体知道真值（距离、干扰、能量、侦察机方位、Pint、暴露） | 该假设在 moderate 噪声下**带来可观的高估**：规则策略综合收益 `+0.4636 → +0.1485`，满足率 `0.9639 → 0.7803`（**−18.4 个百分点**）；DQN 同预算 `+0.4739 → +0.1238` | §12.5 |
| DQN 相对规则的结论 | 「不差于规则」（多种子配对 t 检验不显著） | **方向翻转**：无记忆的 DQN 在 POMDP 下 `+0.1238` < 仅观测规则 `+0.1485`；但加上历史窗口后 `+0.4613`，远优于仅观测规则。原结论高度依赖「智能体能看真值」这一隐含假设 | §12.5 |
| 脚本基线是否公平 | 规则/短视/前瞻直接读真值，与 DQN 同信息量（当时两边都是真值） | 引入 POMDP 后**不再公平**。新增 `strategy/belief_policy.py`：让脚本策略只依赖带噪信念状态，与 DQN 站在同一信息水平 | §12.5 |
| 置信度从哪来 | 只有 Q 值 argmax，无任何可靠性信息 | 集成 DQN 给出 **Q 均值/方差**；另加 **OOD 评分**与**观测质量**，共三路独立信号。⚠️ 实测**只有集成分歧真正在起作用**，OOD 评分从未触发（负结果） | §11A / §12.6 |
| AI 能不能说清"我不确定" | 不能 | 能：AI 诊断新增 `AI_HIGH_UNCERTAINTY` / `AI_OOD_INPUT` / `AI_SMALL_Q_MARGIN` / `AI_FALLBACK_TRIGGERED` / `AI_SHIELD_APPLIED` 等发现码，逐条附数值证据 | §9.7 |
| 回退的定位 | — | **明确否定"回退 = 保证更好"**。回退阈值是人工设定的，回退率主要反映阈值选择；有效性必须由**单步反事实**（同样状态下换动作是否达标）而非错误率对比来判定 | §11A.4 |
| 拉格朗日负结果 | 违反率对 λ 非单调，朴素对偶上升发散 | **保持不变、继续保留**，未通过调 λ 掩盖；v4.0 只新增"安全护盾"作为独立的可信决策控制，不替换该负结果 | §11.4 / §11.5 |
| 观测质量的口径 | — | 新增；并记录一次真实踩坑：初版用 `1/(1+σ/0.1)` 计算质量，侦察机距离 σ=4000 m 使质量恒为 0.3 左右，导致回退条件**任何一步都成立**（实测"高风险步占比"恒为 1.000）。按各量程归一化后质量回到 0.50~0.69 | §11A.2 |

### 13A.1 v4.0 明确**还没有**做到的事（避免被误读）

* **P3 学习型干扰机尚未实现**：当前只有**规则型**自适应干扰机（v3.1 已有），
  不能声称存在"学习型对手"的实验；
* **P4 轨迹级反事实尚未实现**：现有反事实仍是**单步**的（同一步换功率试算），
  不能回答"第 12 步若不用 50 W，能否避免第 48 步能量不足"这类跨时问题；
* **P5 自动压力测试与课程学习尚未实现**：尚无 `/api/generate_scenario`、
  `/api/stress_test`，也没有"发现薄弱场景→加入训练→重新评测"的自动闭环；
* **历史窗口分支是 frame-stacking，不是 DRQN**：它把最近 K 帧观测拼成一个长向量，
  **没有**循环网络、没有序列回放。不得在论文里称其为 DRQN/LSTM-DQN；
* **集成成员共享同一回放缓冲区**（只有初始化与自助掩码不同），
  因此成员多样性低于标准 Bootstrapped DQN，集成分歧信号偏保守。
  不得声称"N 个完全独立的模型"。

### 13A.2 v4.1 明确**还没有**做到的事

* **多雷达只是几何层面的**：`Scene` 可以对任意多雷达做几何查询，
  但探测/截获/干扰的物理评估与功率控制**只针对主雷达** `radars[0]`。
  **没有**多雷达协同决策、没有多雷达数据融合、没有雷达间干扰协调，
  也**没有**"多部雷达联合覆盖同一目标"这类能力；
* **多平台场景不参与任何性能实验**：`config/multi_platform_scenario.json`
  只用于验证实体模型与几何一致性，**没有**跑过 DQN 训练或多种子评测，
  不得引用它给出任何策略性能结论；
* **姿态不影响力学模型**：`heading/pitch/roll` 目前只用于几何求值
  （机体方位、机体俯仰），**没有**接入天线方向图、
  没有做波束指向增益的姿态修正、也没有机体遮挡建模。
  因此"雷达朝哪看"只体现在机体方位这个报告量上，不影响发射增益；
* **运动学仍是匀速直线**：没有加速度/转弯率建模，`Pose.propagate()`
  只是用于"在 t+dt 求值"的线性外推，不驱动仿真；
* **没有高程遮挡与地球曲率**：`z` 参与距离计算，但视线不被地形/地球遮挡，
  `is_los`（通视）判断**未实现**。

### 13A.3 v4.2 明确**还没有**做到的事

* **没有跟踪与数据关联**：融合层是「按置信度排序取前 K 条」，
  不是最近邻 / JPDA / MHT；航迹编号跨时间不稳定。**不得称其为跟踪器**；
* **没有航迹外推**：扫描帧作废旧记忆，漏检一次即丢失目标（保守取舍）；
* **多传感器没有融合估计**：只是把各传感器的测量并排放进槽位，
  **没有**卡尔曼滤波或协方差加权——「多传感器」目前只增加信息条数，
  不产生最优融合估计；
* **天线方向图未接入姿态**：姿态只影响视场与机体方位，不影响增益；
* **协方差是对角阵**，方位-俯仰交叉耦合未建模；
* **遮挡只是简化几何**：AABB 与球，无地形高程/地球曲率/大气折射/多径/绕射；
* **虚警是独立伯努利事件**，没有 CFAR 门限随环境自适应；
* **DQN 三组对照已完成，但不构成严格归因**：三组 DQN 是三个分别训练的模型，
  ②→③ 的差值 **t≈1.03 不显著**。可以说「在理想测量下训练不比真实测量下更好、
  也没有更差」，**不可以**说「噪声训练更好」或「理想测量更好」。
  严格归因需要多次独立训练的训练方差，本版**没有**做；
* **DQN 的 ①→② 落差（−0.0838）无法分解**：里面同时含「信息变少」与
  「53 维输入更难学」两个成分，只有 12 维一个模型、53 维两个模型，
  无法把两者分离。不得声称这就是「信息可得性的代价」。

### 13A.4 v4.3 / v4.4 明确**还没有**做到的事

* **协同收益只在 2 个种子上验证**，样本量偏小，**没有统计显著性检验**；
  因此 §11E 里"理想共享把 RMSE 从 206.00 m 降到 87.29 m"这类数字是**描述性**的，
  不能声称统计显著；
* **融合航迹还没有变成环境的一种观测模式**：规则策略与 DQN
  **仍未在"融合航迹输入"下重新评测**，"单传感器测量 vs 融合航迹"对照**未完成**；
* **跟踪器只有常速度模型**：无机动检测、无自适应过程噪声；
* **JPDA/MHT 未引入**，关联就是门限 + 最近邻；
* **协同只发生在几何/测量层面**：没有多雷达联合控制，也没有雷达间干扰协调；
* **没有外部数据校核**。

### 13A.5 v4.5 明确**还没有**做到的事

* ✅ **多目标压力测试（P2）已完成**：四类压力场景（+长遮挡子场景）、
  关联层审计、十项多目标指标、四路共享对照全部落地，详见 §11G。
  ⚠️ 但它同时暴露了跟踪器的一个严重 bug（§11G.5），
  并**改变了 v4.4 的协同数字**（§11E.6）——这两件事都必须一起读；
* ✅ **多目标指标已实现**：`id_switch_count`、`track_fragmentation_count`、
  `false_track_rate`、`missed_track_rate`、`association_accuracy`、
  `track_purity`、`track_completeness`、`continuity_rate`、
  位置/速度 RMSE、`duplicate_track_count` 共十项；
* **JPDA-lite 分支未引入**：按用户要求需要先由压力测试证明存在明显误关联。
  ✅ v4.5 P2 已跑压力测试并**确认 S1/S2/S4 触发告警阈值**（§11G.6），
  因此"是否需要"已有答案（**需要**），但分支**仍未实现**；
* **多目标压力测试仍有两个硬缺口**：
  ① **只跑了单种子**（`--seeds` 接口与均值±标准差聚合已就绪，但没有跑 20–30 种子，
  也没有显著性检验），所有差值都是描述性的；
  ② **目标只做匀速直线运动**，机动目标（转弯/加速）同时打击预测与关联，未覆盖；
  ③ 离线一对一分配用**贪心**而非匈牙利最优解（目标数 ≤3 时一致，但未证明）；
* **P3 多种子统计只留接口**：本版**没有**跑 20–30 种子，
  也没有聚合 CSV 与显著性检验的产物——不能把"接口预留"写成"统计已做"；
* **证据校验不是语义级校验**：它只拦"编造的发现码"和"无法溯源的数字"，
  语义层面的错误陈述**仍可能漏过**。不得把 §11F.5 写成"AI 不会编造"；
* **`/api/explain_track`、`/api/explain_cooperation` 目前只做过进程内调用验证**
  （`verify_v4.py` §11 + `tests/test_ai_evidence.py`），
  **没有**经过真实 HTTP 服务端到端联调；
* **AI 上下文的无真值性质由测试钉住，但不构成形式化证明**：
  测试覆盖的是当前字段与当前构造路径，新增字段时需同步扩展断言。
* **53 维观测仍未接入融合航迹**：`fuse_measurements()` 的无状态航迹表
  与 `FusionCenter` 是两条并行路径，前者喂 RL 观测、后者喂 AI 证据与协同/压力评测，
  两者**没有打通**（这也是本轮跟踪器修复**没有**改变观测模式天梯的原因）。

### 13A.6 大阶段一（资源管理）明确**还没有**做到的事

* ✅ **阶段验收 6 项全部通过**（多节点执行真实生效 / 资源不超支 / 信息不越权 /
  任务队列可追溯 / 规则与优化参考可复现 / 调度反向控制感知链），清单与逐项证据落在
  `output/runs/<run_id>/resource_eval/acceptance/checklist.{json,md}`；
* ✅ **冻结契约 `resource-contract-v1` 已建立**（摘要可校验，改一处即失败）；
* **没有跨 tick 资源预留**：执行器仍只支持立即执行，每节点每 tick 最多 1 条；
  `reserved` 机制已就位但未被调度器使用；
* **没有把"共享/刷新"的收益接进资源约束**：节点摘要每 tick **无条件**发布，
  于是 `share` 任务成本照扣、收益为零，`estimate_quality` 维度对调度
  **完全不敏感**（三个方法同值）。这是**建模缺口**，必须与
  §11H.8 第 4 条一起读，不得把它写成"调度对估计质量没有影响"；
* **规则调度会饿死一类服务**：`share` 不绑定航迹 → 拿不到服务需求项 →
  永远排在高等级任务之后（实测完成 0/42）。**未修**，如实记录；
* **"有前瞻"不等于"理论上界"**：`exact=True` 只对**完全枚举**的单 tick
  小问题成立，且只对该问题成立；滚动规划**没有**任何最优性或近似比保证；
* **优化参考的排名依据是预测向量，报告里的向量是实测向量**：预测模型含
  "航迹始终可见"这类乐观假设，预测目标值高**不代表**实测更好；
* **没有统计意义上的方法比较**：默认场景逐种子同值（检测强制成功、虚警率 0），
  需要先打开概率检测/虚警或随机化场景才能谈"重复实验"；
* **没有跨节点航迹关联**、**没有真实 OOSM 语义**（观测摘要按"最后到达"覆盖）；
* **没有接学习算法**：闸门见 §11H.8——未通过前不进入学习算法阶段，
  通过后也**不要求**规则方法必须失败、学习方法必须胜出。

### 13A.7 真闭环与学习协议明确**还没有**做到的事

* ✅ **多雷达资源—感知闭环已接通**（§11K）：`ExecutionPlan → RuntimeExecutor`
  唯一入口；停采样 → 测量下降、σ 单调上升；恢复 → 重新采样；share 门控生效；
  无重复执行、无真值泄漏、资源守恒。**但**它只是"闭环成立"，
  **不是**性能改进（误差反而变大，见下条）；
* ⚠️ **新闭环下估计误差变大**：旧路径等于每节点每 tick 免费获得一次测量；
  新路径按实际任务给测量。24 tick 实测离线误差均值 155.68 m → 1000 m 量级。
  不得把新路径写成"更准"，也不得用旧路径的数字做新结论；
* ⚠️ **通信预算是硬门控**：预算耗尽后节点 outbox 永非空 → 不再派采样任务 →
  节点停住进入纯预测。这是资源耗尽语义的直接后果，不是软降级；
* **没有多雷达联合功率控制**：本次接通的是资源调度的多节点感知/通信/融合闭环；
  `engine/simulator.py` 的 `radars[0]` 联合功率动作仍未实现；
* **没有跨 tick 资源预留**：执行器仍只支持立即执行，每节点每 tick 最多 1 条，
  因此 `sample → process → share` 需要 3 个 tick 走完一轮，闭环响应带宽受限；
* **学习协议只做语义与接口校核，没有训练**：终止/截断/bootstrap/精确案例/
  数据划分已完成并通过测试，但**没有**跑任何 smoke training，也**没有**解封测试集；
* **拉格朗日语义修复尚未重训**：因此历史负结果保留为"校核前历史观察"，
  **不能**据此说"某类方法必然失效"，也**不能**说"已改善"；
* **研究分支的主假设未获支持**（§11J.3）：组合特征降低平均等待 0.227 s，
  但最差完成率退化 0.0015，且失败格子（S7/种子 103）已保留；
* **研究分支需要重测**：该分支的小样本结论建立在"调度不影响感知链"这一
  当时成立的限制上（四组 RMSE 完全相同）；§11K 已解除该限制，
  重测完成前 §11J.3 的结论只在旧前提下成立；
* **测试分区仍封存**：`get_split("test")` 默认抛异常；本轮**未**解封，
  也**没有**用测试结果回调任何选择。

---


## 14. 环境与依赖

| 用途 | 解释器 | 依赖 |
| --- | --- | --- |
| `main.py` / `diagnose_temporal_coupling.py` / `diagnose_ai.py` / `ai_server.py` / `multi_target_stress` | 任意 Python 3.10+ | **仅标准库** |
| `python -m resource_management` / `tools/compare_schedulers.py` / `evaluate_resource_management.py` / `verify_v4.py` / `run_validation.py` | 任意 Python 3.10+ | **仅标准库**（资源管理层不 import torch） |
| `train_dqn.py` / `evaluate_dqn.py` / `evaluate_multiseed.py` / `sensitivity_energy_budget.py` / `evaluate_jammer_modes.py` | **`D:\anaconda\envs\pytorch_env\python.exe`**（torch 2.3.1+cu118） | torch；`rl/` 不用 numpy；PNG 需 matplotlib |

`rl/` 刻意不用 numpy：该环境的 `torch 2.3.1` 按 NumPy 1.x 编译、而 `numpy 2.2.6`，
torch 的 numpy 桥接不可用；观测是 `List[float]`，直接 `torch.tensor()` 即可绕开。
启动时那行 `A module that was compiled using NumPy 1.x ...` 是 numpy 的 ABI 提示，**无害**。

**统一日志**：所有新脚本经 `logging_utils.setup_logging()` 输出到控制台与
`output/logs/<run>.log`（带时间戳、级别、模块名）。

**API Key 安全**：远程 provider 的 Key 只从参数或环境变量读取，
不写入任何日志、产物或 checkpoint。

---

## 15. 模型局限

1. **Pd 用 logistic 近似 ROC**，未引入虚警概率 `Pf`、未建模 Swerling 起伏与脉冲积累。
2. **`suppression_db` / `snr50_db` 是等效合并参数**，不应当作真实装备指标。
3. **累计暴露是一阶线性递推**；真实 ESM 的证据积累是非线性、多模态的。
4. **自适应干扰机是规则型状态机，不是学习型对手**（见 8.1）——
   它不会针对具体雷达策略优化行为，因此**不能**用它宣称"对抗学习型对手"。
5. **能量模型只算发射能耗**，未含待机/冷却/处理功耗。
6. **单雷达、单波束**，未建模扫描与波位驻留。
7. **AI 层是"认知诊断"，不是"认知控制"**：它不参与决策回路，
   给出的建议也不保证与策略的最优动作一致；这是刻意的安全设计。
8. **AI 解释只覆盖单步反事实**，不提供跨时间的策略级因果解释
   （如"为什么第 30 步的选择导致了第 50 步的能量不足"）——那需要轨迹级反事实。
9. **泛化优势尚未确立**：在 12 维状态、61 步、单一物理模型下，DQN 相对解析规则策略
   只做到"平均不劣于"；**但在自适应对手下其优势明显变大**（见 8.5），这是本版最重要的线索。
10. **拉格朗日约束分支未收敛**（见 11.4 / 11.5）：机制正确但朴素对偶上升在本环境失效，
    约束版违反率 0.1311 > 目标 0.07，综合收益 +0.2230 < 普通 DQN +0.3766。
    **不应当**把它作为"约束版更好"的证据使用。

### 15.1 本版交付的诚实性边界（写论文/答辩时必须遵守）

| 主张 | 可以说吗 | 依据 |
| --- | --- | --- |
| 自适应干扰机形成了闭环对抗 | ✅ 可以 | 8.4 闭环实录 + 四动作全部触发 |
| 自适应干扰机是**规则型**对手 | ✅ 必须同时说明 | 它是状态机，**不是学习型对手** |
| 对手自适应后 DQN 相对规则的优势变大 | ✅ 可以 | 8.5：收益 −0.3106 vs −0.3977 |
| DQN 在固定干扰下显著优于规则策略 | ⚠️ 只能说"不劣于" | 多种子配对 t=+0.944 不显著 |
| AI 层能诊断/解释且不影响主仿真 | ✅ 可以 | 9.6 降级路径 + 路由端到端测试 |
| AI 解释被限制在结构化证据范围内，并通过规则检查 | ✅ 应当这样说 | 10 章 + §11D.5：只引用结构化证据，且有规则检查与降级路径。**不要**说"AI 解释被限制在结构化证据范围内，并通过规则检查"——那是无法证明的强主张 |
| 约束 DQN 优于普通 DQN | ❌ **不可以** | 11.4 负结果 |
| — v4.0 新增 — | | |
| 部分可观测使任务变难、性能下降 | ✅ 可以（附具体数值） | §12.5：同一信息水平下的对照 |
| 观测噪声不影响物理与奖励 | ✅ 可以，且有测试保证 | `tests/test_pomdp_env.py::TestPomdpInvariant` |
| 集成分歧可作为"不确定度"使用 | ⚠️ 必须说明它是**启发式** | §11A.5：不是校准概率 |
| 集成成员是"N 个完全独立的模型" | ❌ **不可以** | 共享同一回放缓冲区 |
| 历史窗口分支是 DRQN / LSTM-DQN | ❌ **不可以** | 它只是 frame-stacking，无循环网络 |
| 回退机制让 AI"更安全/更好" | ❌ **不可以**（除非单步反事实支持） | §11A.7 判定纪律 |
| 回退率反映 AI 的内省能力 | ❌ **不可以** | 阈值是人设的，须同时报灵敏度 |
| AI 知道自己什么时候不可靠 | ⚠️ 只能说"提供了可用的不确定度信号" | 需以 rescue/harm 与高风险满足率为证据 |
| 存在学习型干扰机对手 | ❌ **不可以** | P3 未实现（§13A.1） |
| 支持轨迹级反事实 | ❌ **不可以** | P4 未实现，目前只有单步反事实 |
| 支持自动压力测试/课程学习 | ❌ **不可以** | P5 未实现 |
| — 大阶段一（资源管理）新增 — | | |
| 资源管理问题已经定义清楚并且能运行 | ✅ 可以 | §11H：单位/时钟/两阶段校验/账本 + 冻结契约 + 阶段验收 5/5 通过 |
| 多节点执行真实生效 | ✅ 可以，且附账本证据 | §11H.8 验收第 1 项：≥2 个节点既有已规划任务又有真实消耗 |
| 资源不超支 | ✅ 可以，且附守恒残差 | §11H.8 验收第 2 项：全部策略×种子×节点×单位残差为 0 |
| 调度器不越权（不读真值/未来/隐藏状态） | ✅ 可以，且有**运行时对照** | §11H.8 验收第 3 项：改掉未来故障，故障窗口之前的规划**逐位相同** |
| 优化参考是"最优解/理论上界" | ❌ **不可以** | §11H.6：只有**完全枚举**的单 tick 小问题才谈精确最优，且只对该问题成立 |
| 优化参考每个 tick 都完全枚举 | ❌ **不可以** | 实测 24 次规划中只有 6~11 次完全枚举，其余降级束搜索（`claim_kind` 逐次记录） |
| 优化参考比规则调度更好 | ❌ **不可以** | §11H.8：`base` 场景**互不支配**——规则在通信与耗时上最优，优化参考在及时性上最优 |
| 完成率能说明方法优劣 | ❌ **不可以** | 完成度由服务上限封顶（需求/能力 ≈ 6.3），三方法几乎同值 |
| 调度改善了估计质量 | ❌ **不可以** | 该维度三方法完全相同（摘要无条件发布）——是**建模缺口**，见 §13A.6 |
| 规则调度把每类服务都服务到了 | ❌ **不可以** | `rule` 把 48 个名额全给了 `estimate_update`，`share` 完成 0/42（§11H.8 第 5 条） |
| 冻结契约可以随手改 | ❌ **不可以** | 改一处即 `verify_frozen()` 失败，须升版本号 + 同步文档 + 重跑验收 |
| 已接入学习算法 | ❌ **不可以** | 闸门见 §11H.8；通过后也**不要求**学习方法必须胜出 |
| — 真闭环与学习协议新增 — | | |
| 调度结果反向控制了感知链（闭环成立） | ✅ 可以，**但必须说明是新路径** | §11K：新路径下 EDF 与轮询/规则在测量数/融合/消息/估计质量/误差上全部不同；旧路径下三者完全相同 |
| 旧路径下调度能改善航迹质量 | ❌ **不可以** | §11K.1：旧路径每 tick 无条件扫描与融合，三策略估计质量完全相同 |
| 可以跨模式比较完成率/估计质量/误差 | ❌ **不可以** | 新路径任务由实际运行状态派生、测量量按调度给；旧路径等于每节点每 tick 免费测量（§11K.3 第 2、3 条） |
| 真闭环让系统"更好" | ❌ **不可以** | 新路径下估计误差由 155.68 m 变为 1000 m 量级——它把此前被隐式补上的测量量显式化了，不是改进 |
| 通信预算是软降级 | ❌ **不可以** | §11K.3 第 1 条：预算是**硬门控**；通信预算耗尽后节点会停住并进入纯预测 |
| 多雷达联合功率控制已实现 | ❌ **不可以** | 本次接通的是**资源调度的多节点感知/通信/融合**闭环；`radars[0]` 联合功率动作仍未实现（§4.1） |
| 终止/截断语义已与 Gymnasium 一致 | ✅ 可以 | §11I.1：任务自然终点 = `terminated`（关 bootstrap），外部步数上限 = `truncated`（保留 bootstrap），由 `info.bootstrap_allowed` 审计 |
| 拉格朗日负结果可以直接归因于方法能力 | ❌ **不可以** | §11I.1：旧实现的双 critic **不是同一个估计对象**；修复未重训，归因降级为"校核前历史观察" |
| 研究分支的假设已被支持 | ❌ **不可以** | §11J.3：组合特征降低平均等待 0.227 s 但**最差完成率退化**，当前证据**不支持**完整假设 |
| 研究分支的 RMSE 相同说明特征无效 | ❌ **不可以** | 当时调度还不会改变感知链（该限制已在 §11K 解除，需重测） |
| 来源一致性分数是"出错概率" | ❌ **不可以** | §11J.2：它是**未经概率校准的相对指示量** |
| 测试集已用于调参 | ❌ **不可以** | §11I.3：`config/learning_splits_v1.json` 封存，`get_split("test")` 默认抛 `SealedTestSplitError`；本轮未解封 |
| — 集中式学习基线（§11L）新增 — | | |
| 学习型调度器已接入并跑通 | ✅ 可以 | §11L：中央 Actor-Critic 输出标准 `ExecutionPlan`，经 `UnifiedExecutor`+`RuntimeExecutor` 执行 |
| 学习基线资源守恒 | ✅ 可以，且逐 episode 校验 | 24 次更新每次守恒；代价对账误差 2.7e-10；执行器拒绝率 0.0000 |
| 学习基线优于规则/优化参考 | ❌ **不可以** | 回报是**自定的学习信号**，规则基线不优化它；同口径比较**未做**（§11L.9 第 1 条） |
| 学习基线有统计显著性 | ❌ **不可以** | 训练 2 个种子起步，validation 3 个种子，未做多种子统计 |
| 智能体学会了规避非法动作 | ❌ **不可以** | §11L.3：不带 mask 的 argmax 非法率 **100%**——mask 是承重的，部署必须带 mask |
| 未加 mask 也能安全部署 | ❌ **不可以** | 同上；非法动作率必须分"带 mask 的 0 / 执行器拒绝率 / 不带 mask 的反事实"三个数一起报 |
| 学习环境没有偷看真值 | ✅ 可以，且有运行时证据 | AST 扫描 + **全 idle 动作下测量/融合/消息恒为 0** |
| 学习基线与规则基线用同一环境 | ✅ 可以 | 世界构造共用 `closed_loop._build_world`；旧路径逐位复现 |
| 可以用学习环境的回报与规则基线排名 | ❌ **不可以** | 两种方法的目标函数不同；跨方法比较必须换同一评价向量重跑 |
| `load_multiplier` 表示"任务数量倍数" | ❌ **不可以** | §11L.5：闭环候选类型上限为 3，它实现为**截止余量缩放**（服务压力） |
| 训练用了 `expose_all` 门控但数值仍可与旧路径比 | ❌ **不可以** | 候选构成变了（实测任务数 24 → 56），完成率不可直接比较 |

---

## 16. 后续工作

### 16.1 v4.0 尚未完成的优先级（P3–P5，按原计划继续）

1. **P3 学习型干扰机（独立研究分支）**：保留现有规则自适应干扰机作为**固定基线**不动，
   另起一个学习型干扰机分支（同样的 4 个动作、独立的能量预算）。
   先冻结雷达策略、单独训练干扰机，并预留交替训练/自博弈接口。
   必须严格区分「规则型对手」与「学习型对手」，
   最终给出固定 / 规则自适应 / 学习型 的三方多种子对比。
2. **P4 轨迹级反事实**：保存仿真状态快照，支持从任意早期步重新分支，
   回答「第 12 步如果不用 50 W，是否会避免第 48 步能量不足」、
   「哪些早期动作导致了最终暴露过高」，输出关键决策时刻、累计贡献与备选轨迹。
   现有 `explain/counterfactual.py` 只做**单步**试算，是其子集。
3. **P5 AI 层扩展与自动课程**：
   * `/api/generate_scenario`、`/api/stress_test`：自然语言 → 结构化 JSON，
     必须先过 Schema 校验 + 参数范围白名单 + 物理一致性检查才允许进入仿真；
   * 自动困难场景搜索（最弱目标位置 / 观测误差 / 干扰时机 / 能量预算组合），
     形成「发现薄弱场景 → 加入训练 → 重新评测」的自动课程闭环。

### 16.2 其他

4. **把 POMDP 臂训练到与主 DQN 同预算（4500 episode）**：v4.0 交付的是
   1200 episode 的匹配预算三方对照（这是能干净归因的对比），
   但主 DQN 是 4500 episode。补齐同预算的长训练可以让两条结论直接对齐。
5. **召回/序贯记忆分支**：把历史窗口换成真正的 DRQN/LSTM（序列回放缓冲），
   在丢测率更高的场景下检验时序记忆是否比 frame-stacking 更有价值。
6. **约束 MDP 进一步完善**：CPO、多约束（同时约束暴露与违反率）。
   注意 §11.4 的拉格朗日负结果必须**保留**，不得通过反复调 λ 掩盖。
7. **建模升级**：Marcum Q 的 ROC、Swerling 起伏、杂波与大气衰减、相控阵扫描驻留。
8. **工程补齐**：`--adaptive-jammer` 下的多种子与敏感性实验；
   `evaluate_pomdp.py` 的前瞻对照臂（当前 `--no-lookahead` 是默认的快速档）；
   Git 初始化与 CI。

### 16.3 v4.5 的完成情况（按用户给定顺序）

v4.5 的三步顺序是 **(1) AI 认知诊断接口 → (2) 多目标压力场景 → (3) 多种子统计**。

9. ✅ **(1) 已完成**：AI 证据链诊断接口见 §11F；
10. ✅ **(2) 已完成**：`multi_target_stress/` 四类压力场景、
    关联层审计（`association_candidate_tracks` / 门限距离 / 最终选择 / 拒绝原因）、
    十项多目标指标、单雷达 / 双雷达不共享 / 理想共享 / 受限共享四路对照，见 §11G。
    NN + 卡尔曼基线**保持不动**（只修了 miss/coasting/删除的判定 bug）；
    压力测试**确认触发误关联告警阈值**（S1/S2/S4），因此 JPDA-lite 的依据已成立，
    但**分支仍未实现**（§11G.6）；
11. **(3) 只留接口**：统一 `--seeds`（`multi_target_stress` 与
    `evaluate_cooperative_sensing.py` 都支持）、聚合 CSV、均值±标准差，
    **不**在本轮跑 20–30 种子，也**没有**做显著性检验。
    注意「接口存在」≠「统计已做」——§11G.4 的数字全部是**描述性**的。

### 16.4 v4.5 P2 之后建议的下一步（按优先级）

12. **引入 JPDA-lite 关联分支**：压力测试已给出充分依据（§11G.6）。
    实现时必须 ① 保持 NN + 卡尔曼为可复现基线；
    ② 报告它改善了**哪些失效模式**（预期：S2 的航迹合并与 S4 的虚警夺轨）；
    ③ 报告**算力代价**（每帧关联的计算量、与 NN 的耗时比）；
    ④ 若改善不明显，同样要如实记录负结果；
13. **补多种子 + 显著性检验**：接口已就绪，把 §11G.4 的描述性差值升级为
    带置信区间的结论（S2 的"共享反而更差"尤其需要多种子确认——
    如果它只是单种子的波动，结论要改）；
14. **机动目标压力**：常速度模型在转弯/加速目标上预测与关联会同时退化，
    这是当前压力测试**完全没覆盖**的失效模式；
15. **打通融合航迹 → RL 观测**：把 `FusionCenter` 的航迹变成一种
    `observation_mode`，让规则策略与 DQN 在"融合航迹输入"下重新评测
    （§11E.5 的遗留项，也是 v4.5 唯一"说好了但没做"的观测侧工作）。

### 16.5 大阶段一之后的下一步（资源管理线）

大阶段一的闸门是「阶段验收 6 项」，**已全部通过**（§11H.8）。
因此下面这条线可以开始，但**不要求**学习方法必须胜出，也**不要求**
规则方法必须失败：

1. **集中式学习基线（当前闸门）**：按 `docs/learning_evaluation_checklist.md`
   与 `docs/resource_rl_baseline.md` 迭代。**第一步已完成**（§11L）：
   中央 Actor-Critic 输出标准 `ExecutionPlan`、真闭环训练、资源守恒、
   train/validation/test 划分严格执行。下一步：
   * **同口径对比**：让规则基线、优化参考与学习策略在**同一评价向量**上重跑，
     否则"谁更好"无从判断（当前学习回报是自定信号，规则基线不优化它）；
   * **多种子 + 显著性**：训练与评测都扩到 10–30 个种子，报均值±标准差；
   * **让策略内化合法性**：当前 mask 是承重的（§11L.3），可尝试
     对非法动作加惩罚或改用"动作可行性预测"头，减少对 mask 的依赖；
   * 打开概率检测/虚警或随机化场景，使多种子**真的产生差异**，
     否则任何"提升"都无法与噪声区分（§11H.8 第 3 条）；
   * 保留三个规则基线与两个优化参考为**可复现对照**，且所有方法继续共用
     同一观测/队列/执行器；对照也要跑在**同一条闭环路径**上。
2. **重测 §11J 的四组消融**：该分支当时受"调度不影响感知链"限制
   （四组 RMSE 完全相同）。限制现已解除，重测后结论可能改变；
3. **给"从未被服务的服务类别"单独加项**：修掉规则调度饿死 `share` 的问题
   （§11H.8 第 5 条）。这不是为了让规则"赢"，而是当前规则本身有明确缺陷；
4. **跨 tick 资源预留**：放开"执行器只支持立即执行"的限制
   （`reserved` 机制已就位），使每节点每 tick 可以排多条任务；
   这同时会改变"完成度由服务上限封顶"的现状与闭环响应带宽，必须重跑基线与契约验收；
5. **把真闭环的门控改成软降级**：通信预算耗尽时允许节点继续本地采样、
   只是不发（§11K.3 第 1 条），避免"资源耗尽即停摆"；
6. **跨节点航迹关联**：把各节点各自上报的同一真值航迹合并成全局 ID，
   需要先解决信息边界问题（§11H.11 第 5 条）；
7. **统计意义上的方法比较**：在 (1) 的前置条件满足后，
   用 20–30 个种子给六维向量配置信区间，把描述性差值升级为结论。

### 16.6 第三方（Codex）后续阶段与本次补完

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 工程全量阅读与 Git 初始化/推送 | ✅ 完成 | `task_plan.md` / `findings.md` / `progress.md`；`.gitignore` |
| 学习问题与评测协议 v1（§11I） | ✅ 完成 | `learning_env.py` / `exact_cases.py` / `learning_protocol.py` / `config/learning_splits_v1.json` / `docs/learning_protocol.md` |
| 拉格朗日 critic 语义修复（§11I.1） | ✅ 代码完成，**未重训** | `rl/lagrangian_agent.py` / `tests/test_lagrangian_semantics.py` |
| 信息新鲜度研究分支（§11J） | ✅ 机制检查完成，**主假设未获支持** | `information_research.py` / `config/information_research_v1.json` / `tools/evaluate_information_research.py` |
| 多雷达资源—感知真闭环（§11K） | ✅ 本轮补完 | `RuntimeExecutor` / `runtime_mode` 开关 / `tests/test_runtime_feedback.py` / 验收第 6 项 / `verify_v4.py` §19 |
| Preference-Conditioned PPO 探索（v1–v3） | 🟡 已完成探索，机制验证未通过，当前暂停 | `docs/preference_ppo_archive.md` / `tools/verify_preference_ppo_archive.py`；test-v5 封存 |

**本轮（接手后）做的事**：

1. **发现并修掉一处回归**：反馈路径的门控代码被误留在旧路径里，
   `runtime` 在旧路径未定义 → 每 tick 抛 `UnboundLocalError` → 被
   `except Exception` 吞掉并记为"任务创建失败" → **闭环跑完全程但零任务**
   （`n_tasks=0`，全量测试 10 项失败）；
   同时把该 `except` **收窄为领域异常**，禁止编程错误伪装成软失败（§11K.4）；
2. **把运行时指标从硬编码常数改为真实统计**
   （重复执行数 = 运行时任务数 − 唯一去重键数；真值隔离由实际载荷与
   中央快照的键扫描统计），并新增逐节点传感器扫描数与事件分类计数；
3. **补齐闭环级端到端证据**：停采样 → 测量下降/σ 单调上升、恢复 → 重新采样、
   share 门控（预算置零 → 0 消息）、融合只消费已到达测量、
   旧路径保留且"调度对航迹质量零影响"作为闭环判据；
4. **把"真闭环"提升为第 6 项阶段验收条件**，并修正窗口设计
   （窗口不能贴到运行末尾，否则"恢复"会因步数不够而假失败）；
5. 更新 `docs/data_contract.md` §4.5、`docs/resource_management.md` §13 与本 README。

### 16.7 下一主线：多雷达 Global Track / Track-to-Track Fusion

Preference-Conditioned PPO 探索已正式暂停，后续主线切换为 **多雷达 Global Track /
Track-to-Track Fusion**，不再创建 preference PPO v4/v5。该工作应从独立协议开始：先冻结
全局航迹 ID、跨节点关联/去重、track-to-track 融合输入的信息边界、协方差与来源语义、
回归场景及评测指标；再在不覆盖本分支负结果的前提下实现与验证。未来若资源允许重新研究
多目标/Pareto RL、层次化策略、偏好课程学习、更长训练预算或更多训练 seed，也必须另立
新协议和新的封存测试集。
