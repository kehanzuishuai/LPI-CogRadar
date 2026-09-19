"""v4.0 验收脚本：一次性检查全部新增能力与向后兼容。

用法::

    python verify_v4.py                    # 仿真层 + AI 层检查（零依赖，base 环境可跑）
    D:\\anaconda\\envs\\pytorch_env\\python.exe verify_v4.py   # 含集成/回退检查（需 torch）

它和 `tests/` 下的单元测试是互补关系：
* `tests/` 用 unittest 钉住**单个模块的不变式**（可单独定位失败点）；
* 本脚本做一次**端到端交付验收**——新模块能否导入、旧实验是否仍逐位复现、
  观测模式是否真的不改变物理、AI 诊断是否真的产出新发现码、
  以及所有交付产物是否都在位。

为什么需要它：v4.0 改动横跨仿真层/学习层/认知层，
单跑某个测试文件看不出「跨层契约」有没有被破坏。
"""

from __future__ import annotations

import os
import sys

#: Windows 控制台默认 GBK，本脚本会打印 ⇒ / ⚠ / ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")

FAILURES = []
SKIPPED = []


def check(name, ok, detail=""):
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("===== 1. 模块可导入（仿真层 + AI 层，零依赖）=====")
    try:
        from engine.observation_model import ObservationModel, ObservationNoiseConfig  # noqa: F401
        from experiment_config import (  # noqa: F401
            DEFAULT_HISTORY_LEN, OBSERVATION_PRESETS, belief_policy_specs,
            make_env, make_pomdp_env, observation_preset,
        )
        from strategy.belief_policy import BeliefPolicy, build_belief_simulator  # noqa: F401
        from strategy.uncertainty_policy import (  # noqa: F401
            MODE_AI, MODE_FALLBACK, MODE_SHIELD, FallbackConfig, UncertaintyAwarePolicy,
        )
        check("engine/strategy 新模块导入", True)
    except Exception as exc:  # noqa: BLE001
        check("engine/strategy 新模块导入", False, repr(exc))
        return 1

    print("===== 2. 向后兼容：full 模式观测维度仍为 12 =====")
    env_full = make_env(observation_mode="full")
    obs, _info = env_full.reset(seed=42)
    check("full 模式观测维度 = 12", len(obs) == 12, "得到 %d" % len(obs))
    check("full 模式关闭噪声", env_full.observation_model.enabled is False)
    check("full 模式观测质量恒为 1", env_full.observation_quality() == 1.0)

    print("===== 3. POMDP 模式 =====")
    env_p = make_pomdp_env(preset="moderate")
    obs, info = env_p.reset(seed=42)
    check("pomdp 观测维度 = 16", len(obs) == 16, "得到 %d" % len(obs))
    check("pomdp reset 不泄露真值", "observations" not in info)
    _, _, _, _, info = env_p.step(5)
    check("pomdp 提供观测估计", "observations" in info)
    check("观测质量在合理区间", 0.3 < info["observation_quality"] <= 1.0,
          "%.3f" % info["observation_quality"])
    check("无真值后门（info 里不含 truth）",
          all("truth" not in r for r in info["observations"].values()))

    env_h = make_pomdp_env(preset="moderate", history_len=DEFAULT_HISTORY_LEN)
    obs_h, _ = env_h.reset(seed=42)
    check("历史窗口维度 = 64", len(obs_h) == 64, "得到 %d" % len(obs_h))

    print("===== 4. 核心不变式：观测模式不改变物理与奖励 =====")

    def rollout(env, actions):
        obs, _ = env.reset(seed=42)
        trace = []
        for action in actions:
            obs, reward, terminated, truncated, i = env.step(action)
            trace.append((
                round(i["tx_power_w"], 9), round(i["pd_min"], 12),
                round(i["intercept_prob"], 12), round(i["exposure_next"], 12),
                round(i["cumulative_energy_j"], 9), round(reward, 12),
            ))
            if terminated or truncated:
                break
        return trace

    actions = [5, 2, 6, 10, 0, 1, 7, 3, 9, 4] * 3
    check(
        "同动作串在 full / pomdp 下真实轨迹与奖励完全一致",
        rollout(make_env(observation_mode="full"), actions)
        == rollout(make_pomdp_env(preset="severe"), actions),
    )

    print("===== 5. 信念桥接复现观测值 =====")
    env_b = make_pomdp_env(preset="moderate")
    env_b.reset(seed=42)
    _, _, _, _, info_b = env_b.step(6)
    belief = build_belief_simulator(env_b, info_b["observations"])
    check("信念剩余能量 = 估计值",
          abs(belief.remaining_energy_j
              - info_b["observations"]["remaining_energy"]["value"]) < 1e-6)
    check("信念干扰比 = 估计值",
          abs(belief.preview(belief.power_levels_w[0]).jam_noise_ratio
              - info_b["observations"]["jam_ratio"]["value"]) < 1e-4)

    print("===== 6. 观测量的物理取值域 =====")
    env_neg = make_pomdp_env(preset="moderate", jam_ratio_sigma=5.0)
    env_neg.reset(seed=1)
    negative = None
    for action in actions:
        _, _, terminated, truncated, i = env_neg.step(action)
        value = i["observations"]["jam_ratio"]["value"]
        if value < 0:
            negative = value
        if terminated or truncated:
            break
    check("J/N 估计不会为负", negative is None, ("出现 %s" % negative) if negative else "")

    print("===== 7. AI 诊断层：新增观测/可信决策发现 =====")
    from ai.context import observability_from_env
    from ai.rule_provider import RuleProvider
    from ai.schema import StateSnapshot, TrustState

    obs_state = observability_from_env(env_b, info_b)
    check("ObservabilityState 生成", obs_state.mode == "pomdp")

    snapshot = StateSnapshot(
        scenario="test", step_index=3, pd_min=0.5, required_pd=0.8,
        task_violated=True, observability=obs_state,
        trust=TrustState(
            uncertainty_source="ensemble", ensemble_size=5,
            q_std_max=0.9, q_std_at_best=0.8, disagreement=0.6,
            ood_score=5.0, q_margin=0.01,
            decision_mode="fallback_rule", decision_mode_cn="完全回退到规则策略",
            reason_code="observation_severely_degraded",
            reason_cn="观测严重退化（丢测/延迟过多），AI 输入不可信",
            triggered=["observation_severely_degraded"],
            fallback_rate=0.3,
        ),
    )
    result = RuleProvider().diagnose(snapshot)
    codes = {f.code for f in result.findings}
    for code in ("AI_HIGH_UNCERTAINTY", "AI_OOD_INPUT", "AI_SMALL_Q_MARGIN",
                 "AI_FALLBACK_TRIGGERED", "ESM_POSITION_UNKNOWN"):
        check("诊断产出 %s" % code, code in codes)
    check("回退解释含诚实提示（不保证更好）",
          any("不保证" in f.message for f in result.findings))

    print("===== 8. 回退策略契约（需要 torch）=====")
    try:
        from rl.dqn_agent import DQNConfig
        from rl.ensemble_agent import EnsembleConfig, EnsembleDQNAgent

        agent = EnsembleDQNAgent(
            DQNConfig(obs_dim=16, n_actions=11, seed=0),
            EnsembleConfig(ensemble_size=3), device="cpu",
        )
        policy = UncertaintyAwarePolicy(agent, FallbackConfig(fallback_mode="shield"))
        env_r = make_pomdp_env(preset="severe")
        env_r.reset(seed=3)
        policy.reset()
        obs_r, _ = env_r.reset(seed=42)
        shield_ok = True
        for _ in range(20):
            action, record = policy.select_action(obs_r, env_r)
            if action < record.ai_action:
                shield_ok = False
                check("护盾不降低功率", False,
                      "action=%d < ai_action=%d" % (action, record.ai_action))
                break
            obs_r, _r, terminated, truncated, _i = env_r.step(action)
            if terminated or truncated:
                break
        if shield_ok:
            check("护盾不降低功率", True)
        summary = policy.summary()
        check("自主率+护盾率+回退率 = 1",
              abs(summary["ai_autonomy_rate"] + summary["shield_rate"]
                  + summary["fallback_rate"] - 1.0) < 1e-9)
    except ImportError as exc:
        print("  [SKIP] 跳过（未安装 torch）: %s" % exc)
        SKIPPED.append("回退策略契约")

    print("===== 9. 交付产物存在性 =====")
    for path in (
        "output/pomdp/pomdp_comparison.csv",
        "output/pomdp/pomdp_ablation.csv",
        "output/pomdp/pomdp_report.html",
        "output/uncertainty/uncertainty_comparison.csv",
        "output/uncertainty/uncertainty_threshold_sweep.csv",
        "output/uncertainty/uncertainty_signal_ablation.csv",
        "output/uncertainty/uncertainty_report.html",
        "output/rl_pomdp_1200/dqn_agent_best.pt",
        "output/rl_pomdp_hist_1200/dqn_agent_best.pt",
        "output/rl_full_1200/dqn_agent_best.pt",
        "output/rl_ensemble/ensemble_best.pt",
    ):
        check(path, os.path.exists(path))

    print("===== 10. 多平台几何与场景导出（v4.1）=====")
    from engine.geometry import (
        Attitude, Pose, TimeSyncError, Vec3, enu_range, enu_range_2d,
        enu_to_spherical, spherical_to_enu, relation,
    )
    from engine.simulator import Simulator
    from models.entity import KIND_RADAR, KIND_TARGET

    # --- 距离对称性（逐位）---
    a, b = Vec3(-1234.5, 6789.0, 12.0), Vec3(9876.5, -4321.0, -7.0)
    check("三维距离逐位对称", enu_range(a, b) == enu_range(b, a))
    check("二维距离逐位对称", enu_range_2d(0.0, 0.0, 6000.0, 0.0)
          == enu_range_2d(6000.0, 0.0, 0.0, 0.0))
    # 注意：6000/3000 的斜距并不等于 6000 的水平距离，所以这里逐项比对
    # 若干 z=0 的样本，而不是拿一个"看起来像"的等式凑。
    flat = all(
        enu_range_2d(ax, ay, bx, by) == enu_range(Vec3(ax, ay, 0.0), Vec3(bx, by, 0.0))
        for ax, ay, bx, by in [(0.0, 0.0, 6000.0, 0.0), (8000.0, 6000.0, 0.0, 0.0),
                               (-1234.5, 6789.0, 9876.5, -4321.0)]
    )
    check("z=0 时三维/二维距离逐位一致", flat)

    # --- 坐标变换往返 ---
    vector = Vec3(6000.0, -3000.0, 2500.0)
    check("ENU↔球坐标往返", vector.is_close(
        spherical_to_enu(enu_to_spherical(vector)), tol=1e-6))
    attitude = Attitude(35.0, -12.0, 8.0)
    check("机体↔ENU 往返", vector.is_close(
        attitude.body_to_enu(attitude.enu_to_body(vector)), tol=1e-9))
    forward, right, up = attitude.body_basis()
    check("机体三轴正交归一",
          abs(forward.norm() - 1.0) < 1e-12 and abs(forward.dot(right)) < 1e-12
          and abs(right.dot(up)) < 1e-12)

    # --- 时间同步 ---
    try:
        relation("A", Pose(position=Vec3(), time_s=0.0),
                 "B", Pose(position=Vec3(1.0, 0.0, 0.0), time_s=5.0))
        check("时间不同步必须报错", False, "没有抛 TimeSyncError")
    except TimeSyncError:
        check("时间不同步必须报错", True)

    # --- 多平台场景 ---
    multi = "config/multi_platform_scenario.json"
    if os.path.exists(multi):
        sim_multi = Simulator(multi)
        sim_multi.load_config()
        sim_multi.reset(seed=42)
        counts = sim_multi.scene.count_by_kind()
        shape_ok = (counts[KIND_RADAR] == 2 and counts[KIND_TARGET] == 3
                    and counts["interceptor"] == 2 and counts["jammer"] == 2)
        check("多平台场景构成 2/3/2/2", shape_ok, str(counts))
        sim_multi.assert_scene_time_synchronized()
        check("多平台场景时间同步", True)
        check("雷达→目标关系数 = 2×3",
              len(sim_multi.radar_target_relations()) == 6)
        # 索引与旧列表必须指向同一对象
        check("索引与旧列表共享同一对象",
              sim_multi.scene.by_id("RADAR_A") is sim_multi.radar
              and sim_multi.scene.by_id("TGT_ESCORT") is sim_multi.targets[2])
        for _ in range(5):
            sim_multi.step(6)
        sim_multi.assert_scene_time_synchronized()
        check("推进后非主雷达也在运动",
              sim_multi.radars[1].x < 25000.0)
    else:
        check("多平台场景配置存在", False, multi)

    # --- v4.5：AI 认知诊断接口 ---
    print("\n===== 11. AI 证据链诊断接口（v4.5）=====")
    import ai.schema as ai_schema
    from ai.api import LPI_API, REQUEST_SCHEMA, ROUTES
    from ai.context import snapshot_from_simulator
    from ai.evidence_check import validate_text_against_context
    from ai.rule_provider import RuleProvider

    required_codes = [
        "TARGET_OUT_OF_FOV", "TARGET_BEYOND_RANGE", "TARGET_OCCLUDED",
        "SENSOR_NOT_UPDATED", "MISSED_DETECTION", "SENSOR_UNAVAILABLE",
        "COMM_PACKET_LOST", "COMM_MESSAGE_EXPIRED", "COMM_LINK_DOWN",
        "COMM_QUEUE_FULL", "REMOTE_MEASUREMENT_DELAYED", "TRACK_COASTING",
        "TRACK_FRAGMENTED", "TRACK_UNCERTAIN", "ASSOCIATION_AMBIGUOUS",
        "REMOTE_SENSOR_CONTRIBUTION", "COOPERATIVE_TRACK_RECOVERED",
    ]
    missing = [c for c in required_codes if c not in ai_schema.FINDING_CODES]
    check("v4.5 新增发现码 ≥17 全部登记", not missing, str(missing))
    check("发现码总数", len(ai_schema.FINDING_CODES) >= 40,
          str(len(ai_schema.FINDING_CODES)))

    # 旧接口保持兼容
    for capability in ("diagnose", "explain_decision", "compare_policies",
                       "generate_report"):
        check("旧能力仍在：%s" % capability,
              capability in ROUTES and capability in REQUEST_SCHEMA)
    for capability in ("explain_track", "explain_cooperation"):
        check("新只读能力已接线：%s" % capability,
              capability in ROUTES and capability in REQUEST_SCHEMA
              and hasattr(LPI_API(), capability))

    # 上下文节与"无真值"纪律
    from communication import SHARE_CONSTRAINED, CommBus, CommConfig
    import experiment_config as _ec
    env45 = _ec.make_env(observation_mode="realistic")
    env45.reset(seed=42)
    env45.attach_comm_bus(CommBus(["RADAR1", "RADAR2"],
                                  CommConfig(policy=SHARE_CONSTRAINED, seed=42)))
    for _ in range(6):
        env45.step(6)
    snap45 = snapshot_from_simulator(env45.sim, result=env45.sim.results[-1],
                                     env=env45)
    payload45 = snap45.to_dict()
    for section in ("measurement_state", "communication_state", "fusion_state",
                    "cooperation_state"):
        check("上下文含 %s" % section, payload45.get(section) is not None)
    leaked = []

    def _scan(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.startswith(("truth", "err_")) or key == "is_false_alarm":
                    leaked.append(key)
                _scan(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _scan(item)

    _scan(payload45)
    check("AI 上下文不含真值字段", not leaked, str(sorted(set(leaked))))

    # 缺失原因真的被区分开
    codes45 = {f.code for f in RuleProvider().diagnose(snap45).findings}
    check("诊断产出测量/通信类发现码",
          bool(codes45 & set(required_codes[:11])), str(sorted(codes45)))

    # 证据校验：编造的必须拦下，真实的不能误伤
    bad = validate_text_against_context(
        "延迟 1.2 s，发现码 FOO_BAR_BAZ，误差 999.0 m", {"latency_mean_s": 1.2})
    check("证据校验拦下编造内容", not bad.passed, str(bad.violations[:2]))
    good = validate_text_against_context("延迟 1.2 s", {"latency_mean_s": 1.2})
    check("证据校验不误伤真实内容", good.passed, str(good.violations[:2]))

    # 两个新接口必须真的能调通（不是只登记了名字）
    api45 = LPI_API()
    r_track = api45.dispatch("explain_track", {"track": {
        "track_id": "T1", "status": "coasting", "hits": 3, "misses": 2,
        "sigma_position": {"x": 150.0, "y": 150.0, "z": 0.0}}})
    check("/api/explain_track 可调用", r_track.get("status") != "error",
          str(r_track.get("error", "")))
    r_coop = api45.dispatch("explain_cooperation", {"communication": {
        "policy": "constrained_share", "n_links": 2, "delivery_rate": 0.83}})
    check("/api/explain_cooperation 可调用", r_coop.get("status") != "error",
          str(r_coop.get("error", "")))

    # --- v4.5：版本号统一 ---
    print("\n===== 12. 版本号统一（v4.5.0）=====")
    version = getattr(_ec, "PROJECT_VERSION", None)
    check("PROJECT_VERSION == 4.5.0", version == "4.5.0", str(version))
    readme = "README.md"
    if os.path.exists(readme):
        with open(readme, "r", encoding="utf-8") as handle:
            head = handle.read(4000)
        check("README 顶部声明 v4.5", "v4.5" in head or "4.5.0" in head)
    else:
        check("README 存在", False, readme)

    # --- v4.5 P2：多目标压力测试 ---
    print("\n===== 13. 多目标压力测试（v4.5 P2）=====")
    from multi_target_stress.metrics import (
        ASSOC_GATE_M,
        MultiTargetMetrics,
        assign_tracks_to_truth,
    )
    from multi_target_stress.report import DEFAULT_OUT_DIR
    from multi_target_stress.runner import STRESS_POLICIES, run_case
    from multi_target_stress.scenarios import SCENARIO_IDS, get_scenario

    check("四类压力场景 + 长遮挡子场景", len(SCENARIO_IDS) == 5,
          str(list(SCENARIO_IDS)))
    check("四路对照齐备（单雷达/不共享/理想/受限）",
          list(STRESS_POLICIES) == ["single", "no_share", "ideal_share",
                                    "constrained_share"],
          str(list(STRESS_POLICIES)))
    check("离线评测门限为显式常数", ASSOC_GATE_M > 0.0, f"{ASSOC_GATE_M:g} m")

    # 关联层审计：候选航迹 / 门限距离 / 最终选择 / 拒绝原因
    s1 = run_case("S1", "single", seed=42)
    audited = [t for t in s1["lifecycle"].traces if t.association_candidate_tracks]
    check("S1 留下关联层审计", bool(audited), f"{len(audited)} 条测量")
    check("S1 记录最终选择",
          any(t.chosen_track_id for t in audited))
    check("S1 记录门限拒绝原因",
          any(c.reject_reason for t in audited
              for c in t.association_candidate_tracks))
    check("S1 交叉产生关联歧义（场景有检验力）",
          s1["metrics"]["ambiguous_rate"] > 0.05,
          f"歧义率 {s1['metrics']['ambiguous_rate']:.4f}")
    check("S1 交叉确实造成 ID 换号",
          s1["metrics"]["id_switch_count"] >= 1,
          f"{s1['metrics']['id_switch_count']} 次")
    s1_shared = run_case("S1", "ideal_share", seed=42)
    check("S1 理想共享减少 ID 换号",
          s1_shared["metrics"]["id_switch_count"]
          < s1["metrics"]["id_switch_count"],
          f"{s1['metrics']['id_switch_count']} → "
          f"{s1_shared['metrics']['id_switch_count']}")

    # **跟踪器 miss/coasting/删除必须真的生效**（v4.5 修的严重 bug）
    s3l = run_case("S3L", "single", seed=42)
    check("长遮挡确实删除航迹（miss 分支生效）",
          s3l["center"].stats["dropped"] >= 1,
          f"dropped={s3l['center'].stats['dropped']}")
    coasting = {t[3] for frame in s3l["frames"] for t in frame["tracks"]}
    check("遮挡期间出现 coasting 状态", "coasting" in coasting,
          str(sorted(coasting)))
    s3l_shared = run_case("S3L", "ideal_share", seed=42)
    check("长遮挡下远端补盲避免碎裂",
          s3l["metrics"]["track_fragmentation_count"] > 0
          and s3l_shared["metrics"]["track_fragmentation_count"] == 0,
          f"{s3l['metrics']['track_fragmentation_count']} → "
          f"{s3l_shared['metrics']['track_fragmentation_count']}")

    # 用户要求的十项多目标指标必须齐全
    required_metrics = (
        "id_switch_count", "track_fragmentation_count", "false_track_rate",
        "missed_track_rate", "association_accuracy", "track_purity",
        "track_completeness", "position_rmse_m", "velocity_rmse_mps",
        "continuity_rate", "duplicate_track_count",
    )
    missing_metrics = [m for m in required_metrics if m not in s1["metrics"]]
    check("十项多目标指标齐备", not missing_metrics, str(missing_metrics))

    # 一对一分配与"重复 vs 假航迹"口径
    _t2tr, _ = assign_tracks_to_truth(
        [("T1", Vec3(0.0, 0.0, 0.0)), ("T2", Vec3(50.0, 0.0, 0.0))],
        [("A", Vec3(0.0, 0.0, 0.0))], gate_m=1000.0)
    check("离线分配是一对一", len(_t2tr) == 1)
    _probe = MultiTargetMetrics()
    _probe.add_frame({"time_s": 0.0, "truth": [
        ("A", Vec3(0.0, 0.0, 0.0), Vec3(), True)],
        "tracks": [("T1", Vec3(0.0, 0.0, 0.0), Vec3(), "confirmed"),
                   ("T2", Vec3(0.0, 40.0, 0.0), Vec3(), "confirmed")],
        "associations": [], "measurement_truth": {}})
    check("多余的近目标航迹记为重复航迹",
          _probe.result()["duplicate_track_count"] == 1)

    # 真值隔离（AST 级：文档字符串里"不读 truth_id"的声明不算违规）
    from validation.checks import check_truth_isolation
    _iso = check_truth_isolation()
    check("fusion/ 跟踪器不读真值字段（AST 级）",
          all(r.passed for r in _iso),
          "；".join(r.detail for r in _iso if not r.passed))

    # 产物
    for name in ("stress_metrics.csv", "stress_metrics.json", "stress_report.md"):
        path = os.path.join(DEFAULT_OUT_DIR, name)
        check(f"{path}", os.path.exists(path))
    if os.path.exists(os.path.join(DEFAULT_OUT_DIR, "stress_report.md")):
        with open(os.path.join(DEFAULT_OUT_DIR, "stress_report.md"),
                  "r", encoding="utf-8") as handle:
            body = handle.read()
        check("压力报告写明未做多种子统计",
              "没有做 20–30 种子" in body or "未做 20–30 种子" in body
              or "没有**做 20–30 种子" in body)
        check("压力报告不宣称统计显著", "显著" not in body.split("## 诚实的边界")[0]
              or "显著性检验" in body)

    # --- v4.5 P2 第二阶段：四类系统级压力场景 ---
    print("\n===== 14. 系统级压力场景（S5–S8）=====")
    import inspect as _inspect

    import fusion.center as _fusion_center

    from ai.rule_provider import RuleProvider as _RP
    from ai.schema import FINDING_CODES as _CODES
    from ai.schema import StateSnapshot as _Snap
    from ai.system_stress import system_stress_state as _stress_state
    from multi_target_stress.runner import run_case as _run_case
    from multi_target_stress.system_scenarios import (
        SYSTEM_SCENARIO_IDS,
        handover_overlap_s,
        handover_windows,
    )
    from multi_target_stress.timing import (
        DROP_STALE,
        OOSM_POLICIES,
        REORDER_BUFFER,
    )

    check("系统级场景 S5–S8 齐备", len(SYSTEM_SCENARIO_IDS) == 4,
          str(list(SYSTEM_SCENARIO_IDS)))
    check("压力测试共 4 核心 + 4 系统级",
          len(SCENARIO_IDS) == 5 and len(SYSTEM_SCENARIO_IDS) == 4,
          f"核心 {list(SCENARIO_IDS)}，系统 {list(SYSTEM_SCENARIO_IDS)}")

    # S5：机动失配（以同场景匀速目标作对照）
    s5 = _run_case("S5", "single", seed=42)["system"]["maneuver"]
    check("S5 机动失配高于同场景匀速对照",
          s5["maneuvered_residual_ratio"] > s5["reference_residual_ratio"],
          "%.3f vs %.3f" % (s5["maneuvered_residual_ratio"],
                            s5["reference_residual_ratio"]))
    check("S5 记录创新峰值 / 门限拒绝 / 恢复时间",
          all(k in s5 for k in ("maneuvered_innovation_max",
                                "maneuvered_gate_rejected",
                                "maneuvered_recovery_time_s")))

    # S6：交接几何 + 不共享失败 / 共享保持覆盖
    windows, overlap = handover_windows(), handover_overlap_s()
    check("S6 交接窗口由几何精确反解", overlap[1] > overlap[0],
          "A=%s B=%s" % (tuple(round(v, 2) for v in windows["A"]),
                         tuple(round(v, 2) for v in windows["B"])))
    ho_single = _run_case("S6", "single", seed=42)["system"]["handover"]
    ho_ideal = _run_case("S6", "ideal_share", seed=42)["system"]["handover"]
    check("S6 不共享时交接失败（本地包线外无航迹）",
          ho_single["handover_continuity_after_local"] < 0.95,
          "%.3f" % ho_single["handover_continuity_after_local"])
    check("S6 共享后目标覆盖不断且远端有贡献",
          ho_ideal["handover_continuity_after_local"] >= 0.95
          and ho_ideal["remote_contribution_ratio"] > 0.2,
          "连续性 %.3f，远端贡献 %.3f" % (
              ho_ideal["handover_continuity_after_local"],
              ho_ideal["remote_contribution_ratio"]))
    check("S6 区分「身份接力」与「覆盖接力」",
          "within_track_handover_achieved" in ho_ideal,
          "within_track=%s" % ho_ideal.get("within_track_handover_achieved"))

    # S7：三种 OOSM 策略 + 代价一起测到
    check("S7 提供三种 OOSM 处理策略", len(OOSM_POLICIES) == 3,
          str(list(OOSM_POLICIES)))
    t_control = _run_case("S7", DROP_STALE, seed=42)["system"]["timing"]
    t_treat = _run_case("S7", REORDER_BUFFER, seed=42)["system"]["timing"]
    check("S7 重排降低乱序率",
          t_treat["out_of_order_rate"] < t_control["out_of_order_rate"],
          "%.4f → %.4f" % (t_control["out_of_order_rate"],
                           t_treat["out_of_order_rate"]))
    check("S7 重排的延迟/时效代价被一起测到",
          t_treat["mean_time_in_system_s"] > t_control["mean_time_in_system_s"]
          and t_treat["stale_rejection_rate"] > t_control["stale_rejection_rate"],
          "在途 %.3f → %.3f s；时效拒绝 %.4f → %.4f" % (
              t_control["mean_time_in_system_s"],
              t_treat["mean_time_in_system_s"],
              t_control["stale_rejection_rate"],
              t_treat["stale_rejection_rate"]))
    check("S7 观测到突发丢包与链路中断",
          t_control["n_bursts"] > 0 and t_control["n_outage_frames"] > 0,
          "突发 %d 段，中断覆盖 %d 帧" % (t_control["n_bursts"],
                                          t_control["n_outage_frames"]))

    # S8：偏差只动测量 + 加第二部雷达反而更差 + 不自动剔除
    s8_single = _run_case("S8", "single", seed=42)["metrics"]
    s8_ideal = _run_case("S8", "ideal_share", seed=42)["metrics"]
    s8_biased = _run_case("S8", "biased_share", seed=42)["metrics"]
    check("S8 无偏共享改善精度",
          s8_ideal["position_rmse_m"] < s8_single["position_rmse_m"],
          "%.2f → %.2f m" % (s8_single["position_rmse_m"],
                             s8_ideal["position_rmse_m"]))
    check("S8 有偏共享比单雷达更差（加第二部雷达反而更差）",
          s8_biased["position_rmse_m"] > s8_single["position_rmse_m"],
          "%.2f vs %.2f m" % (s8_biased["position_rmse_m"],
                              s8_single["position_rmse_m"]))
    bias_state = _run_case("S8", "biased_share", seed=42)["system"]["bias"]
    check("S8 有偏路仍在使用远端测量（不自动剔除）",
          bias_state["remote_contribution_ratio"] > 0.1,
          "%.3f" % bias_state["remote_contribution_ratio"])
    check("健康分只作诊断，未进入跟踪器",
          "sensor_health" not in _inspect.getsource(_fusion_center))

    # 传感器偏差的位级兼容性
    from sensor.sensor import SensorConfig as _SensorConfig
    _clean = _SensorConfig(sensor_id="X", mounting_id="R", max_range_m=1000.0)
    check("传感器偏差字段默认不生效",
          (not _clean.has_bias())
          and _clean.noise_underreport_factor == 1.0
          and _clean.clock_offset_s == 0.0)

    # AI 证据链：新发现码与上下文节
    _required = ("MANEUVER_MODEL_MISMATCH", "TRACK_HANDOVER_IN_PROGRESS",
                 "TRACK_HANDOVER_COMPLETED", "TRACK_HANDOVER_FAILED",
                 "MEASUREMENT_OUT_OF_ORDER", "COMM_LINK_OUTAGE",
                 "COMM_BURST_LOSS", "COMM_RECOVERY_CONGESTION",
                 "SENSOR_BIAS_SUSPECTED", "SENSOR_NOISE_UNDERREPORTED",
                 "INNOVATION_INCONSISTENT")
    _missing = [c for c in _required if c not in _CODES]
    check("系统级压力发现码 11 个全部登记", not _missing, str(_missing))
    check("发现码总数 ≥ 51", len(_CODES) >= 51, str(len(_CODES)))
    _state = _stress_state(
        tracks=_run_case("S5", "single", seed=42)["center"].tracks,
        local_sensor_ids=["SENSOR_LOCAL"])
    check("system_stress_state 含机动/交接/健康三节",
          all(k in _state for k in ("maneuver", "handover", "sensor_health")),
          str(sorted(_state)))
    _snap = _Snap()
    _snap.system_stress_state = _state
    _codes = {f.code for f in _RP().diagnose(_snap).findings}
    check("S5 上 AI 报出机动失配", "MANEUVER_MODEL_MISMATCH" in _codes,
          str(sorted(_codes)))

    # 报告产物（v4.5：产物按 run_id 隔离，落在 output/runs/<run_id>/）
    from run_manifest import latest_run_dir
    # 按工具过滤：output/runs 下有多个工具各自写入的运行目录，
    # 直接取"最新"可能取到别的工具的那次（实测踩过）
    _run_dir = latest_run_dir(".", tool="multi_target_stress")
    check("存在运行隔离目录 output/runs/<run_id>/", bool(_run_dir), str(_run_dir))
    if _run_dir:
        for name in ("stress_metrics.csv", "stress_system_metrics.csv",
                     "stress_report.html", "manifest.json"):
            path = os.path.join(_run_dir, name)
            check(f"run 产物 {name}", os.path.exists(path))
        import json as _json

        with open(os.path.join(_run_dir, "manifest.json"), "r",
                  encoding="utf-8") as handle:
            _manifest = _json.load(handle)
        check("manifest 含 run_id / 配置摘要 / 源码摘要",
              all(k in _manifest for k in ("run_id", "config_digest",
                                           "source_digest")))
        check("manifest 登记了产物清单（含 sha256）",
              bool(_manifest.get("artifacts"))
              and all("sha256" in item for item in _manifest["artifacts"]),
              f"{len(_manifest.get('artifacts', []))} 个产物")

    # --- v4.5：教学仿真资源管理（独立模块，不改历史入口）---
    print("\n===== 15. 教学仿真资源管理（resource_management）=====")
    import json as _json2

    from resource_management import (
        BUDGET_UNITS as _RM_UNITS,
        GlobalClock as _RMClock,
        NodeState as _RMNode,
        PlanStatus as _RMStatus,
        ResourceBudget as _RMBudget,
        ResourceUnit as _RMUnit,
        TaskKind as _RMKind,
        TaskRequest as _RMTask,
        ExecutionPlan as _RMPlan,
        UnifiedExecutor as _RMExec,
    )
    from resource_management.units import (
        UNIT_CN as _RM_CN,
        UNIT_MEANING as _RM_MEANING,
    )

    check("资源单位显式声明（含中文名与含义）",
          all(_RM_CN[u] and _RM_MEANING[u] for u in _RM_UNITS),
          "、".join(u.value for u in _RM_UNITS))

    def _rm_world(slots_b: float = 10.0):
        ex = _RMExec(_RMClock(now_s=0.0))
        for node_id, slots in (("A", 10.0), ("B", slots_b)):
            ex.register_node(_RMNode(node_id=node_id, update_period_s=1.0,
                                     budget=_RMBudget(capacity={
                                         _RMUnit.SAMPLE_SLOT: slots,
                                         _RMUnit.PROCESSING_OP: 10.0,
                                         _RMUnit.COMM_BYTE: 4096.0,
                                     })))
        return ex

    _ex = _rm_world()
    _tick_before = _ex.clock.step_count
    _ex.advance(1.0, "tick")
    _res = _ex.submit(_RMPlan(plan_id="V1", submit_time_s=1.0, tasks=[
        _RMTask(task_id="ta", node_id="A", kind=_RMKind.SAMPLE, start_s=1.0,
                entities=("T1",)),
        _RMTask(task_id="tb", node_id="B", kind=_RMKind.SHARE, start_s=1.0),
    ]))
    check("一份计划可同时驱动两个节点的不同任务",
          _res.status is _RMStatus.APPLIED and _res.n_applied == 2,
          f"{_res.n_applied} 个任务")
    check("一次 tick **只推进一次**全局时钟",
          _ex.clock.step_count - _tick_before == 1,
          f"step_count +{_ex.clock.step_count - _tick_before}")

    # 计划级致命 → 零扣费、零状态污染
    _before = _json2.dumps(_ex.snapshot(), sort_keys=True, ensure_ascii=False)
    _bad = _ex.submit(_RMPlan(plan_id="V2", submit_time_s=1.0, tasks=[
        _RMTask(task_id="x", node_id="NODE_X", kind=_RMKind.SAMPLE,
                start_s=1.0)]))
    _after = _json2.dumps(_ex.snapshot(), sort_keys=True, ensure_ascii=False)
    check("计划级致命：整份不执行且状态逐位不变",
          _bad.status is _RMStatus.REJECTED and _before == _after)

    # 节点局部资源不足 → 只影响该节点
    _ex2 = _rm_world(slots_b=1.0)
    _ex2.advance(1.0)
    _ex2.submit(_RMPlan(plan_id="V3", submit_time_s=1.0, tasks=[
        _RMTask(task_id="b1", node_id="B", kind=_RMKind.SAMPLE, start_s=1.0,
                entities=("T1",))]))
    _ex2.advance(1.0)
    _partial = _ex2.submit(_RMPlan(plan_id="V4", submit_time_s=2.0, tasks=[
        _RMTask(task_id="a2", node_id="A", kind=_RMKind.SAMPLE, start_s=2.0,
                entities=("T1",)),
        _RMTask(task_id="b2", node_id="B", kind=_RMKind.SAMPLE, start_s=2.0,
                entities=("T1",)),
    ]))
    check("节点资源不足只拒该节点，其余节点照常执行",
          _partial.status is _RMStatus.PARTIAL
          and _ex2.node("A").stats["applied"] == 1
          and _ex2.node("B").stats["applied"] == 1,
          f"A={_ex2.node('A').stats['applied']} B={_ex2.node('B').stats['applied']}")

    # 空闲：零成本、零采样报告、信息年龄增长
    _age_before = _ex2.node("A").max_information_age_s(_ex2.clock.now_s)
    _consumed_before = dict(_ex2.node("A").budget.consumed)
    _ex2.advance(3.0)
    _ex2.submit(_RMPlan(plan_id="V5", submit_time_s=5.0, tasks=[
        _RMTask(task_id="a-idle", node_id="A", kind=_RMKind.IDLE, start_s=5.0)]))
    check("空闲零成本且不产生采样报告",
          _ex2.node("A").budget.consumed == _consumed_before)
    check("空闲期间信息年龄必须增加",
          _ex2.node("A").max_information_age_s(_ex2.clock.now_s)
          > _age_before + 2.9,
          f"{_age_before:.1f} → "
          f"{_ex2.node('A').max_information_age_s(_ex2.clock.now_s):.1f} s")

    # 资源守恒
    check("逐节点资源守恒（容量 = 消耗 + 预留 + 剩余）",
          _ex.conservation_report()["all_conserved"]
          and _ex2.conservation_report()["all_conserved"])

    # 账本可追溯
    _rejections = _ex2.ledger.rejections("B")
    check("账本能回答「为什么没执行」",
          bool(_rejections) and all(e.reason_code and e.reason
                                   for e in _rejections),
          f"{len(_rejections)} 条拒绝记录")

    # 独立模块：历史入口不得引用
    _root = os.path.dirname(os.path.abspath(__file__))
    _coupled = []
    for _name in ("main.py", "train_dqn.py", "evaluate_dqn.py",
                  "evaluate_observation_modes.py",
                  "evaluate_cooperative_sensing.py"):
        _p = os.path.join(_root, _name)
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _h:
                if "resource_management" in _h.read():
                    _coupled.append(_name)
    check("历史实验入口未引用 resource_management", not _coupled, str(_coupled))

    # --- v4.5：融合结果 → 资源调度适配层 ---
    print("\n===== 16. 融合结果 → 资源调度适配层 =====")
    from communication import CommBus as _OBSBus
    from communication import CommConfig as _OBSConfig
    from communication import SHARE_IDEAL as _OBSIDEAL
    from fusion import FusionCenter as _OBSFusion
    from fusion import FusionConfig as _OBSFusionConfig
    from resource_management import (
        BUDGET_UNITS as _OBS_UNITS,
        CentralObservationStore as _OBSStore,
        FixedLengthAdapter as _OBSAdapter,
        LEGACY_OBSERVATION_MODES as _OBS_LEGACY,
        NodeObservation as _OBSNodeObs,
        NodeState as _OBSNode,
        QueueTaskKind as _OBSQKind,
        ResourceBudget as _OBSBudget,
        ResourceUnit as _OBSUnit,
        SCHEMA_VERSION as _OBS_SCHEMA,
        TaskQueue as _OBSQueue,
        UnknownObjectError as _OBSUnknown,
        field_metadata as _OBS_fields,
        node_observation_from_fusion as _OBS_from_fusion,
        observation_payload as _OBS_payload,
        observation_truth_violations as _OBS_violations,
        publish_node_observation as _OBS_publish,
    )

    check("适配层使用独立 schema_version",
          bool(_OBS_SCHEMA) and _OBS_SCHEMA not in _OBS_LEGACY,
          _OBS_SCHEMA)
    check("旧观测路径已明确标记并保留",
          set(_OBS_LEGACY) == {"full", "pomdp", "ideal", "realistic"},
          str(list(_OBS_LEGACY)))
    _meta = _OBS_fields()
    check("每个观测字段都标注单位/坐标系/可见范围/来源",
          bool(_meta) and all(
              m["unit"] and m["frame"] and m["visibility"] and m["provenance"]
              for m in _meta.values()),
          f"{len(_meta)} 个字段")

    class _M:
        def __init__(self, sid, cid, t, rng):
            self.sensor_id = sid
            self.candidate_id = cid
            self.sensor_kind = "radar"
            self.time_s = t
            self.range_m = rng
            self.azimuth_deg = 90.0
            self.elevation_deg = 0.0
            self.range_rate_mps = 0.0
            self.std_range_m = 20.0
            self.std_az_deg = 0.3
            self.std_el_deg = 0.3
            self.confidence = 1.0
            self.msg_id = ""
            self.platform_id = ""

    _bus = _OBSBus(["LOCAL", "REMOTE"], _OBSConfig(policy=_OBSIDEAL, seed=7))
    _fusion = _OBSFusion("REMOTE", _OBSFusionConfig(), own_sensor_ids={"S_B"})
    _node = _OBSNode(node_id="REMOTE", update_period_s=1.0,
                     budget=_OBSBudget(capacity={
                         _OBSUnit.SAMPLE_SLOT: 5.0,
                         _OBSUnit.PROCESSING_OP: 5.0,
                         _OBSUnit.COMM_BYTE: 4096.0}))
    _store = _OBSStore(["LOCAL", "REMOTE"])
    _origin = Vec3(0.0, 0.0, 0.0)
    _positions = {}
    for _t in (1.0, 2.0, 3.0, 4.0):
        _fusion.predict_to(_t)
        _fusion.update([_M("S_B", "C_B", _t, 3000.0 + 60.0 * _t)], _t,
                       {"S_B": _origin})
        _obs = _OBS_from_fusion(_fusion, _node, now_s=_t)
        _OBS_publish(_obs, _bus, "REMOTE", now_s=_t)
        _store.ingest_arrived(_bus, "LOCAL", _t)
        if _obs.tracks:
            _positions[_t] = _obs.tracks[0].position[0]

    _view_online = _store.observe(4.0)
    check("中央只读已到达的节点摘要",
          _view_online.node_valid_mask[1] and bool(_view_online.all_track_ids()),
          f"tracks={_view_online.all_track_ids()}")
    check("适配层输出不含真值通道",
          _OBS_violations(_json2.dumps(_view_online.to_dict(),
                                       ensure_ascii=False)) == [])

    # **验收核心**：断开链路后中央不得继续知道远端的新信息
    _frozen = _positions[4.0]
    for _t in (5.0, 6.0, 7.0, 8.0):
        _fusion.predict_to(_t)
        _fusion.update([_M("S_B", "C_B", _t, 3000.0 + 60.0 * _t)], _t,
                       {"S_B": _origin})
    _view_off = _store.observe(8.0)
    _after = _view_off.nodes[1].tracks[0].position[0]
    check("断开远端消息后中央视图**冻结**",
          abs(_after - _frozen) < 1e-9,
          f"{_frozen:.2f} → {_after:.2f} m")
    check("断开后中央**没有**跟随底层真值",
          3000.0 + 60.0 * 8.0 - _after > 200.0,
          f"真值 {3000.0 + 60.0 * 8.0:.1f} vs 中央 {_after:.1f} m")
    check("断开后节点摘要年龄随时间增长",
          abs(float(_view_off.node_information_age_s[1]) - 4.0) < 1e-6,
          f"age={_view_off.node_information_age_s[1]}")

    # 任务队列：未知对象不得提前建任务
    _empty = _OBSNodeObs(node_id="A", observed_at_s=1.0)
    _queue = _OBSQueue()
    _created = _queue.create_from_observation(_empty, 1.0)
    check("空观测只产生与目标无关的采样任务",
          [task.kind for task in _created] == [_OBSQKind.PREDEFINED_SAMPLE],
          str([task.kind.value for task in _created]))
    _blocked = False
    try:
        _queue.enqueue_for_observation(_empty, _OBSQKind.PROCESS, "p1",
                                       targets=("TGT1",))
    except _OBSUnknown:
        _blocked = True
    check("未知对象（如真值 ID）不得提前创建任务", _blocked)

    # 定长适配器：拒绝旧 checkpoint 维度
    _adapter = _OBSAdapter(slots=4)
    _rejected = []
    for _dim in (12, 16, 53):
        try:
            _adapter.check_checkpoint_dim(_dim)
        except ValueError:
            _rejected.append(_dim)
    check("定长适配器拒绝旧路径维度（不截取、不硬塞）",
          _rejected == [12, 16, 53],
          f"output_dim={_adapter.output_dim}，已拦 {_rejected}")

    print("\n===== 17. 非学习型资源调度闭环 =====")
    import json as _SCH_json
    from resource_management.closed_loop import (
        close_tick as _SCH_close_tick,
        run_closed_loop as _SCH_run,
    )
    from resource_management.scheduling import (
        SchedulingConfig as _SCHConfig,
        SchedulerPolicy as _SCHPolicy,
        build_scheduler as _SCH_build,
    )
    from resource_management.tasks import (
        QueueTaskKind as _SCHKind,
        TaskQueue as _SCHQueue,
        TaskStatus as _SCHStatus,
        QueuedTask as _SCHTask,
    )

    _sch = {policy.value: _SCH_run(policy, seed=42, steps=24)
            for policy in _SCHPolicy}
    _planned = {name: sum(bucket["n_planned"]
                          for bucket in result.metrics["per_node"].values())
                for name, result in _sch.items()}
    _kinds = {name: {node: bucket["kinds_planned"]
                     for node, bucket in result.metrics["per_node"].items()}
              for name, result in _sch.items()}
    check("三个基线共用同一观测/队列/执行器（只有 build_scheduler 不同）",
          set(_planned) == {policy.value for policy in _SCHPolicy},
          _SCH_json.dumps(_planned, ensure_ascii=False))
    check("三策略产生**不同分工**（验收核心）",
          len({_SCH_json.dumps(v, sort_keys=True, ensure_ascii=False)
               for v in _kinds.values()}) > 1,
          _SCH_json.dumps(_kinds, ensure_ascii=False))

    _cap = _sch["rule"].metrics["service_capacity"]
    check("完成率与服务上限分开记录（供需 vs 策略）",
          _cap["max_serviceable_tasks"]
          == _cap["n_nodes"] * _cap["steps"]
          * _cap["max_tasks_per_node_per_tick"]
          and _cap["demand_over_capacity"] > 1.0,
          f"需求/能力={_cap['demand_over_capacity']:.2f}，"
          f"利用率={_cap['service_utilization']:.3f}")

    _bad_cap = False
    try:
        _SCHConfig(max_tasks_per_node_per_tick=2).validate()
    except ValueError:
        _bad_cap = True
    check("每节点每 tick 限额 >1 被拒绝（否则整份计划被 PLAN_FATAL 拒）",
          _bad_cap)

    _kinds_tbl = _sch["rule"].queue_summary["by_kind_status"]
    check("逐任务类型 × 状态分布已记录（可看出哪类服务被跳过）",
          set(_kinds_tbl) == {kind.value for kind in _SCHKind},
          _SCH_json.dumps(
              {k: v["completed"] for k, v in _kinds_tbl.items()},
              ensure_ascii=False))

    _q = _SCHQueue()
    for _tid, _dl in (("with_dl", 5.0), ("no_dl", None)):
        _q.enqueue(_SCHTask(
            task_id=_tid, kind=_SCHKind.PROCESS, node_id="A",
            release_time_s=1.0, deadline_s=_dl, targets=("T1",)))
    _starved: set = set()
    _SCH_close_tick(_q, 10.0, 8.0, _starved)
    _by_id = {task.task_id: task for task in _q.tasks}
    check("长期未获服务与过期**互斥**（不重复计入完成率分母）",
          sorted(_starved) == ["no_dl"]
          and _by_id["with_dl"].status is _SCHStatus.EXPIRED,
          f"starved={sorted(_starved)}")

    _rule_rationale = [row for row in _sch["rule"].decisions
                       if row.get("decision") == "planned" and row.get("reasons")]
    check("每次分工都带可读理由 + 结构化证据",
          bool(_rule_rationale) and all(row["reasons"] and row["evidence"]
                                        for row in _rule_rationale),
          f"{len(_rule_rationale)} 条已规划决策")

    _tl = _sch["rule"].timelines
    check("逐节点任务时间线已落盘（分工可逐条追溯）",
          bool(_tl) and all(rows for rows in _tl.values())
          and all({"time_s", "task_id", "kind", "decision", "reason"} <= set(row)
                  for rows in _tl.values() for row in rows),
          f"节点 {sorted(_tl)}")
    check("资源守恒在闭环中成立",
          all(result.metrics.get("conservation_all") for result in _sch.values()))

    _ages = _sch["rule"].node_ages
    check("逐 tick 记录到达年龄/内容年龄/可见航迹数",
          bool(_ages) and all({"ages", "content_ages", "n_tracks"} <= set(row)
                              for row in _ages))
    check("覆盖交接：至少一个节点的可见目标数随时间变化",
          any(len({row["n_tracks"][index] for row in _ages}) > 1
              for index in range(len(_ages[0]["n_tracks"]))
              if all(row["n_tracks"][index] is not None for row in _ages)),
          _SCH_json.dumps(_ages[0]["n_tracks"]))

    print("\n===== 18. 非学习优化参考与阶段冻结 =====")
    from resource_management.acceptance import run_acceptance as _OPT_accept
    from resource_management.contract_v1 import (
        CONTRACT_VERSION as _OPT_CV,
        contract_digest as _OPT_digest,
        verify_frozen as _OPT_frozen,
    )
    from resource_management.optimization import (
        CANONICAL_CLAIMS as _OPT_CLAIMS,
        CLAIM_KIND_EXACT as _OPT_EXACT_KIND,
        METRIC_KEYS as _OPT_KEYS,
        PREDICTION_ASSUMPTIONS as _OPT_ASSUMPTIONS,
        build_optimizer as _OPT_build,
        default_optimizer_config as _OPT_default_cfg,
        OptimizerKind as _OPT_Kind,
    )

    _frozen = _OPT_frozen(strict=False)
    check("冻结契约摘要与当前口径一致",
          _frozen["ok"], f"{_OPT_CV} / {_OPT_digest()}")
    check("评价为 6 维向量（不含单一不透明综合分）",
          len(_OPT_KEYS) == 6
          and set(_OPT_KEYS) == {"service_completion", "task_timeliness",
                                 "estimate_quality", "resource_consumption",
                                 "communication_overhead", "compute_time"},
          str(list(_OPT_KEYS)))
    check("预测模型显式声明「不预测未来故障」",
          any(item.name == "no_future_events" for item in _OPT_ASSUMPTIONS),
          f"{len(_OPT_ASSUMPTIONS)} 条假设随结果落盘")
    check("精确最优声明只用于完全枚举，且自带否定",
          ("精确最优" in _OPT_CLAIMS[_OPT_EXACT_KIND]
           and "不是" in _OPT_CLAIMS[_OPT_EXACT_KIND]),
          f"{len(_OPT_CLAIMS)} 种规范声明")

    _enum_cfg = _OPT_default_cfg(_OPT_Kind.ENUMERATION)
    _roll_cfg = _OPT_default_cfg(_OPT_Kind.ROLLING_HORIZON)
    check("两种优化参考在语义上确实不同（时域 1 vs >1）",
          _enum_cfg.horizon_ticks == 1 and _roll_cfg.horizon_ticks > 1,
          f"enumeration h={_enum_cfg.horizon_ticks}，"
          f"rolling h={_roll_cfg.horizon_ticks}")

    _opt_result = _SCH_run(_SCHPolicy.ROLLING_HORIZON, seed=42, steps=16)
    _opt_summary = _opt_result.optimizer_summary or {}
    _audit = _opt_summary.get("claim_audit") or {}
    check("优化参考的声明自审全部通过（规范文本/类型一致/枚举有据）",
          bool(_audit) and all(value for key, value in _audit.items()
                               if isinstance(value, bool)),
          _SCH_json.dumps(_opt_summary.get("claim_kinds", []),
                          ensure_ascii=False))
    check("计算预算生效且超支被如实记录",
          _opt_summary.get("n_plans", 0) > 0
          and "max_overrun_s" in _opt_summary,
          f"规划 {_opt_summary.get('n_plans')} 次，"
          f"完全枚举 {_opt_summary.get('n_exact')} 次，"
          f"触发预算 {_opt_summary.get('n_budget_exhausted')} 次")

    _vector = _opt_result.metrics["evaluation_vector"]
    check("实测向量与预测向量分开标注来源",
          _vector["provenance"] == "measured"
          and _opt_result.optimizer_outcome is not None
          and _opt_result.optimizer_outcome["vector"]["provenance"]
          == "predicted",
          "measured vs predicted")

    _accept = _OPT_accept(seeds=(42,), steps=12, quick=True)
    check("阶段验收 5 项条件全部通过（未通过则不得进入学习算法阶段）",
          _accept["all_passed"],
          f"{_accept['n_passed']}/{_accept['n_checks']}：" +
          "、".join(c["name_cn"] for c in _accept["checks"] if c["ok"]))
    _boundary = [c for c in _accept["checks"]
                 if c["key"] == "information_boundary"][0]
    check("信息不越权有**运行时**对照（改未来故障不改当前规划）",
          _boundary["ok"]
          and all(_boundary["evidence"]["identical_before_cut"])
          and _boundary["evidence"]["differs_inside_window"],
          "故障窗口前逐位相同，窗口内确实不同")

    print()
    if SKIPPED:
        print("跳过项：%s（非失败）" % "、".join(SKIPPED))
    if FAILURES:
        print("===== 验收失败 %d 项 =====" % len(FAILURES))
        for name in FAILURES:
            print("  -", name)
        return 1
    print("===== 全部验收检查通过 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
