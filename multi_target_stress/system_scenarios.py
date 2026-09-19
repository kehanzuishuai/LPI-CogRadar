"""四类**系统级**压力场景（v4.5 P2 第二阶段）。

与核心关联场景（S1–S4，见 `scenarios.py`）的分工
-------------------------------------------------
核心场景问的是"关联本身会不会错"（换号 / 合并 / 碎裂 / 虚警夺轨）。
系统级场景问的是**整条链条**在更真实的不确定条件下还稳不稳：

| 场景 | 打击的目标 | 要验证的命题 |
| --- | --- | --- |
| **S5 机动目标模型失配** | 常速度卡尔曼的**运动模型** | 目标突然转弯/加减速/爬升时，预测失配会不会导致误关联或丢轨？ |
| **S6 多雷达覆盖交接** | **跨传感器航迹接力** | 目标从"只有 A 看得见"走到"只有 B 看得见"，Track ID 保得住吗？还是两个雷达各建各的航迹？ |
| **S7 通信时序压力** | **时间语义** | 突发丢包 / 链路中断 / 恢复拥塞 / 乱序到达下，历史旧包会不会污染当前航迹？ |
| **S8 多传感器系统偏差** | **融合的加权** | 一部有偏的雷达会不会把融合结果拉偏？会不会出现"加第二部雷达反而更差"？ |

四条纪律（与核心场景一致，外加两条）
------------------------------------
1. 物理参数仍从基础配置复制，只改几何/速度/传感器包线/偏差——见
   `scenarios.py` 的同名说明；
2. 检测统一设为确定性，把失效归因到"机动/交接/时序/偏差"本身；
3. **机动只改真值运动**（`maneuvers.py`），不改任何物理公式；
4. **偏差只加在传感器测量上**（`sensor.SensorConfig` 的偏差字段），
   真值一个字节都不动，且偏差对算法**不可见**。

⚠️ 本模块**不引入** IMM / JPDA / 复杂运动模型。用户要求：
先把当前 NN + Kalman 基线的失效模式**完整记录**下来，
再决定下一阶段是否有必要升级。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from multi_target_stress.maneuvers import (
    ManeuverInjector,
    ManeuverInjector as _ManeuverInjector,  # noqa: F401 - 便于外部按名引用
    climb_step,
    speed_step,
    turn_step,
)
from multi_target_stress.scenarios import (
    BASE_CONFIG_PATH,
    StressScenario,
    _load_base,
    _radar,
    _sensor,
    _target,
)
from multi_target_stress.timing import (
    DELAYED_UPDATE,
    DROP_STALE,
    REORDER_BUFFER,
)

#: 系统级场景 ID
SYSTEM_SCENARIO_IDS: Tuple[str, ...] = ("S5", "S6", "S7", "S8")

#: S6 交接场景的几何常数（本地雷达 A 在 -y 侧、远端雷达 B 在 +y 侧）
HANDOVER_A_Y = -8000.0
HANDOVER_B_Y = 8000.0
HANDOVER_SENSOR_RANGE_M = 11000.0
HANDOVER_CROSS_X = 1500.0
HANDOVER_SPEED_MPS = 400.0
HANDOVER_Y0 = -6000.0
HANDOVER_FOV_DEG = 45.0

#: S7 的链路中断窗口
TIMING_OUTAGE_WINDOWS: Tuple[Tuple[float, float], ...] = ((8.0, 12.0),)

#: S8 注入到远端雷达上的系统偏差
S8_BIAS: Dict[str, Any] = {
    "range_bias_m": 150.0,
    "az_bias_deg": 0.8,
    "clock_offset_s": 0.8,
    "noise_underreport_factor": 0.4,
}


def _sensor_range_window(
    sensor_y: float, x_target: float, half_fov_deg: float, max_range_m: float,
) -> Tuple[float, float]:
    """在给定包线下，雷达能看到航线上目标的**y 区间**。

    目标沿 `x = x_target` 直线运动，雷达在 `(0, sensor_y)`。
    两件事会切断观测，取更严的那个：

    * **作用距离**：`(y - sensor_y)² ≤ R² − x_target²`
    * **视场**：目标方位与雷达机头（±y）的夹角不超过 `half_fov_deg`；
      由于雷达机头指向目标所在的一侧，等价于
      `|y - sensor_y| ≥ x_target / tan(half_fov)`

    返回 `(y_min, y_max)`；两者不重叠时返回空区间（`y_min > y_max`）。
    """
    span = math.sqrt(max(0.0, max_range_m ** 2 - x_target ** 2))
    range_limit = (sensor_y - span, sensor_y + span)
    # 视场：目标必须在机头前方，且偏离不超过半视场角
    fov_min_offset = x_target / math.tan(math.radians(half_fov_deg))
    direction = 1.0 if sensor_y < 0 else -1.0     # A 朝 +y，B 朝 -y
    if direction > 0:
        fov_limit = (sensor_y + fov_min_offset, float("inf"))
    else:
        fov_limit = (float("-inf"), sensor_y - fov_min_offset)
    return (max(range_limit[0], fov_limit[0]),
            min(range_limit[1], fov_limit[1]))


def handover_windows() -> Dict[str, Tuple[float, float]]:
    """S6 交接场景的**解析可见窗口**（秒），由几何反解而不是估。"""
    a_window = _sensor_range_window(HANDOVER_A_Y, HANDOVER_CROSS_X,
                                    HANDOVER_FOV_DEG, HANDOVER_SENSOR_RANGE_M)
    b_window = _sensor_range_window(HANDOVER_B_Y, HANDOVER_CROSS_X,
                                    HANDOVER_FOV_DEG, HANDOVER_SENSOR_RANGE_M)

    def to_time(y_value: float) -> float:
        return (y_value - HANDOVER_Y0) / HANDOVER_SPEED_MPS

    return {
        "A": (max(0.0, to_time(a_window[0])), to_time(a_window[1])),
        "B": (max(0.0, to_time(b_window[0])), to_time(b_window[1])),
    }


def handover_overlap_s() -> Tuple[float, float]:
    """A、B **同时可见**的时间窗（交接必须在这个窗口里完成）。"""
    windows = handover_windows()
    return (max(windows["A"][0], windows["B"][0]),
            min(windows["A"][1], windows["B"][1]))


# ----------------------------------------------------------------------
# S5：机动目标模型失配
# ----------------------------------------------------------------------


def _s5_maneuvers() -> Tuple[Any, ...]:
    """构造 S5 的机动时刻表。

    被机动目标 `TGT_MAN` 起始在 `(7000, -1500)` 以 120 m/s 沿 +y 飞行：

    * `t = 12 s`：水平转弯 +70°（速度方向从 +y 转到接近 +x）
    * `t = 22 s`：再加速 +100 m/s（速率 120 → 220）

    两次都是**速度阶跃**——最坏的模型失配。
    `TGT_REF` 全程匀速（**对照目标**），两者最小间距 > 2 km，
    因此关联层不会互相干扰，指标差异可以干净地归因到"机动"。
    """
    initial = (0.0, 120.0, 0.0)
    turn = turn_step(12.0, "TGT_MAN", _vec(initial), heading_change_deg=110.0)
    after_turn = _vec((initial[0] + turn.delta_v[0],
                       initial[1] + turn.delta_v[1],
                       initial[2] + turn.delta_v[2]))
    accelerate = speed_step(22.0, "TGT_MAN", after_turn, delta_speed_mps=100.0)
    return (turn, accelerate)


def _vec(values: Tuple[float, float, float]):
    from engine.geometry import Vec3

    return Vec3(*values)


def _build_s5() -> StressScenario:
    base = _load_base()
    radar_base, target_base = base["radar"], base["targets"][0]

    def R(radar_id, x, y, heading):  # noqa: N802
        return _radar(radar_id, x, y, heading, base=radar_base)

    def T(target_id, x, y, vx, vy):  # noqa: N802
        return _target(target_id, x, y, vx, vy, base=target_base)

    return StressScenario(
        scenario_id="S5",
        title_cn="机动目标模型失配",
        question="目标突然转弯/加速导致常速度模型失配时，"
                 "预测误差、创新量、门限拒绝会怎样变化？会不会误关联或丢轨？"
                 "航迹要多久才能重新收敛？",
        expected_failure="机动瞬间预测位置与量测严重不符：创新量（马氏距离）"
                         "骤增 → 门限拒绝或错误关联 → 若持续更久则丢轨并重新起始。",
        verification_goal="记录**机动前后**的预测误差、创新量、gate rejection、"
                          "coasting、碎裂与**恢复时间**，用同场景的匀速对照目标"
                          "做基线，判断失配是否越过了关联门限。",
        overrides={
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [
                # 会被机动的目标
                T("TGT_MAN", 7000.0, -1500.0, 0.0, 120.0),
                # 全程匀速的对照目标（与机动目标最小间距 > 2 km）
                T("TGT_REF", 5000.0, 3000.0, 0.0, -60.0),
            ],
            # 本地雷达**慢扫描**（2 s 一帧）、远端 1 s：
            # 更新周期越长，常速度模型的预测窗口越长，机动失配越容易越门限。
            # 共享路因此多了一层意义：远端能在本地"跟不上"时补上新鲜观测。
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0,
                                update_period_s=2.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0,
                                update_period_s=1.0)],
            "occluders": [],
        },
        steps=30,
        n_targets=2,
        group="system",
        maneuvers=_s5_maneuvers(),
        notes=[
            "TGT_MAN：t=12 s 水平转弯 +110°，t=22 s 再加速 +100 m/s（两次速度阶跃）",
            "本地雷达**慢扫描**（2 s 一帧）、远端 1 s：更新周期越长，"
            "常速度模型的预测窗口越长，机动失配越容易越过关联门限",
            "TGT_REF：全程匀速，作为**同场景对照**，用于区分"
            "「机动造成的失配」与「场景本身的误差水平」",
            "跟踪器仍是常速度卡尔曼，**没有**换 IMM / 机动检测",
            "两目标最小间距 > 2 km，关联层不会互相干扰",
        ],
    )


# ----------------------------------------------------------------------
# S6：多雷达覆盖交接（handover）
# ----------------------------------------------------------------------


def _build_s6() -> StressScenario:
    base = _load_base()
    radar_base, target_base = base["radar"], base["targets"][0]

    def R(radar_id, x, y, heading):  # noqa: N802
        return _radar(radar_id, x, y, heading, base=radar_base)

    def T(target_id, x, y, vx, vy):  # noqa: N802
        return _target(target_id, x, y, vx, vy, base=target_base)

    return StressScenario(
        scenario_id="S6",
        title_cn="多雷达覆盖交接（handover）",
        question="目标从「只有 A 看得见」走到「只有 B 看得见」，"
                 "Track ID 保得住吗？会重复建轨/短时双轨/身份跳变吗？"
                 "共享与不共享下交接延迟差多少？",
        expected_failure="不共享时航迹在 A 的包线边缘直接消失；"
                         "共享时若交接不及时，会出现「A 的旧航迹 + B 的新航迹」"
                         "并存（短时双轨 → 重复航迹计数上升），"
                         "或身份跳变（ID switch）。",
        verification_goal="验证「多雷达协同**真的实现航迹接力**」，"
                          "而不是两个雷达各自生成互不相干的航迹："
                          "比较 handover continuity / handover delay / "
                          "remote contribution ratio。",
        overrides={
            "radars": [R("RADAR_LOCAL", 0.0, HANDOVER_A_Y, 0.0),
                       R("RADAR_REMOTE", 0.0, HANDOVER_B_Y, 180.0)],
            "targets": [T("TGT_1", HANDOVER_CROSS_X, HANDOVER_Y0, 0.0,
                          HANDOVER_SPEED_MPS)],
            "sensors": [
                _sensor("SENSOR_LOCAL", "RADAR_LOCAL", HANDOVER_SENSOR_RANGE_M,
                        az_fov_deg=HANDOVER_FOV_DEG),
                _sensor("SENSOR_REMOTE", "RADAR_REMOTE", HANDOVER_SENSOR_RANGE_M,
                        az_fov_deg=HANDOVER_FOV_DEG),
            ],
            "occluders": [],
        },
        steps=30,
        n_targets=1,
        group="system",
        incoming_sensor_id="SENSOR_REMOTE",
        handover_target_id="TGT_1",
        notes=[
            "A 在 y=-8000、机头 +y；B 在 y=+8000、机头 -y；两者作用距离 11 km、"
            "半视场 45°（**覆盖范围故意只部分重叠**）",
            "目标沿 x=1500 从 y=-6000 以 400 m/s 飞到 y=+6000",
            "可见窗口由几何**精确反解**（距离门限 ∧ 视场门限），不是估的",
        ],
    )


# ----------------------------------------------------------------------
# S7：通信时序压力
# ----------------------------------------------------------------------


def _build_s7() -> StressScenario:
    base = _load_base()
    radar_base, target_base = base["radar"], base["targets"][0]

    def R(radar_id, x, y, heading):  # noqa: N802
        return _radar(radar_id, x, y, heading, base=radar_base)

    def T(target_id, x, y, vx, vy):  # noqa: N802
        return _target(target_id, x, y, vx, vy, base=target_base)

    return StressScenario(
        scenario_id="S7",
        title_cn="通信时序压力（突发丢包 / 中断 / 恢复拥塞 / 乱序）",
        question="突发丢包、链路中断、恢复后拥塞、乱序到达下，"
                 "历史旧包会不会在链路恢复后**重新污染**当前航迹？"
                 "重排缓冲能不能改善，代价是什么？",
        expected_failure="乱序到达让「先用新信息、再用旧信息」更新同一条航迹，"
                         "旧包把状态往回拽 → 航迹抖动、创新异常；"
                         "中断恢复后一批积压旧包集中到达，污染更明显。",
        verification_goal="显式区分 measurement / send / arrival / fusion 四个时刻；"
                          "对比 `drop_stale`（对照，旧包照常更新）与 "
                          "`reorder_buffer` / `delayed_update`（重排与简化回溯）"
                          "在**中断期间与恢复后**的航迹连续率与误差，"
                          "并给出重排的**延迟代价**。",
        overrides={
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [T("TGT_1", 7000.0, -800.0, 0.0, 90.0)],
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0)],
            "occluders": [],
        },
        steps=30,
        n_targets=1,
        group="system",
        axis="oosm",
        axis_values=(DROP_STALE, REORDER_BUFFER, DELAYED_UPDATE),
        fixed_policy="constrained_share",
        outage_windows=TIMING_OUTAGE_WINDOWS,
        comm_extras={
            "base_delay_s": 1.2,
            "jitter_s": 0.4,
            "loss_prob": 0.10,
            "expiry_s": 4.0,
            "burst_loss_prob": 0.10,
            "burst_length": 3,
            "outage_windows": TIMING_OUTAGE_WINDOWS,
            "recovery_congestion_s": 5.0,
            "recovery_extra_delay_s": 1.5,
            "recovery_loss_prob": 0.3,
            "reorder_prob": 0.35,
            "reorder_extra_delay_s": 3.0,
        },
        notes=[
            "远端链路：基延迟 1.2 s + 抖动 0.4 s + 独立丢包 10%",
            "**突发丢包**：突发起始概率 0.10、突发长度 3（上下抖动 ±50%）",
            f"**链路中断**：t={TIMING_OUTAGE_WINDOWS[0][0]:g}~"
            f"{TIMING_OUTAGE_WINDOWS[0][1]:g} s 完全不可用",
            "**恢复拥塞**：恢复后 5 s 内额外丢包（峰值 30%）+ 额外延迟（峰值 1.5 s），线性衰减",
            "**乱序到达**：乱序概率 0.35，命中消息额外延迟 1.5~4.5 s",
            "对照轴是 **OOSM 处理策略**，共享策略固定为受限共享（`constrained_share`）",
            "丢包强度刻意压到「远端仍然贡献可观」的量级——"
            "否则场景会退化成「远端基本缺席」，测不出时序处理策略的差别",
        ],
    )


# ----------------------------------------------------------------------
# S8：多传感器系统偏差
# ----------------------------------------------------------------------


def _build_s8() -> StressScenario:
    base = _load_base()
    radar_base, target_base = base["radar"], base["targets"][0]

    def R(radar_id, x, y, heading):  # noqa: N802
        return _radar(radar_id, x, y, heading, base=radar_base)

    def T(target_id, x, y, vx, vy):  # noqa: N802
        return _target(target_id, x, y, vx, vy, base=target_base)

    return StressScenario(
        scenario_id="S8",
        title_cn="多传感器系统偏差 / 不一致",
        question="一部远端雷达带固定距离/方位偏差、时钟偏移与**噪声低估**时，"
                 "会不会把融合结果拉偏？会不会出现「加第二部雷达反而更差」？"
                 "残差一致性检查能不能把它指出来？",
        expected_failure="有偏传感器误差是**系统性**的，不会被多帧平均掉："
                         "它持续把航迹往同一方向拉；"
                         "**噪声低估**更危险——融合按上报 σ 加权，"
                         "于是把更大的权重交给误差更大的传感器。",
        verification_goal="四组对照（本地单雷达 / 不共享双雷达 / 理想共享 / "
                          "**有偏传感器共享**）下比较融合 RMSE、航迹协方差、"
                          "单传感器残差、创新一致性与远端贡献比例；"
                          "给出只做**诊断**的 sensor health score，"
                          "**默认不自动剔除**坏传感器。",
        overrides={
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [T("TGT_1", 7000.0, 0.0, 0.0, 60.0)],
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0)],
            "occluders": [],
        },
        steps=30,
        n_targets=1,
        group="system",
        axis="policy",
        axis_values=("single", "no_share", "ideal_share", "biased_share"),
        bias_variant="biased_share",
        bias_overrides={"SENSOR_REMOTE": dict(S8_BIAS)},
        notes=[
            "偏差**只加在传感器测量层**：真值一个字节都不动，"
            "且算法看不到「这条测量有偏」（载荷白名单里也没有该字段）",
            "注入项：距离偏差 +150 m、方位偏差 +0.8°、"
            "时钟偏移 +0.8 s、**噪声低估系数 0.4**（上报 σ 只有真实值的 40%）",
            "四组：本地单雷达 / 不共享双雷达 / 理想共享 / 有偏传感器共享",
            "sensor health score 只作**诊断分支**，不自动剔除坏传感器",
        ],
    )


_BUILDERS = {
    "S5": _build_s5,
    "S6": _build_s6,
    "S7": _build_s7,
    "S8": _build_s8,
}


def get_system_scenario(scenario_id: str) -> StressScenario:
    """按 ID 取系统级场景。"""
    builder = _BUILDERS.get(scenario_id)
    if builder is None:
        raise ValueError(
            f"未知系统级场景 {scenario_id!r}，只支持 {list(SYSTEM_SCENARIO_IDS)}"
        )
    return builder()


#: 全部系统级场景
SYSTEM_SCENARIOS: List[StressScenario] = [
    get_system_scenario(s) for s in SYSTEM_SCENARIO_IDS
]


def build_maneuver_injector(scenario: StressScenario) -> ManeuverInjector:
    """按场景的机动时刻表构造注入器（无机动时返回空注入器）。"""
    return ManeuverInjector(scenario.maneuvers or ())
