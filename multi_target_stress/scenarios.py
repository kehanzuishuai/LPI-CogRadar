"""四类多目标压力场景（v4.5 P2）。

设计原则
--------
1. **物理参数一律从基础场景配置复制**（`config/radar_scenario_v1.json`），
   只覆盖**位置、朝向、速度、传感器作用距离/视场/更新周期、虚警率、遮挡体**。
   这样"压力"来自几何与测量条件，而不是来自改物理公式——
   基础雷达的发射功率、增益、噪声系数、门限等一个字节都没动。
2. **场景是显式定义的**，不用随机搜索。每个场景都写明它要暴露的失效模式，
   跑之前就能说清"如果关联没问题，应该看到什么"。
3. 每个场景都配一部**远端雷达**，使 单雷达 / 不共享 / 理想共享 / 受限共享
   四路可以在**同一几何**下对比（"单雷达"路只是不建远端传感器）。

坐标系提醒：ENU，`heading_deg=90` 表示机头指 +x（东）；
方位自 +y（北）顺时针为正。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: 基础场景配置（物理参数的唯一来源）
BASE_CONFIG_PATH = os.path.join("config", "radar_scenario_v1.json")

#: 四类场景 + 一个附加的"长遮挡"子场景
SCENARIO_IDS = ("S1", "S2", "S3", "S3L", "S4")


def _load_base() -> Dict[str, Any]:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _heading_of(vx: float, vy: float) -> float:
    """速度方向 → 航向角（方位自 +y 顺时针为正，与全工程一致）。"""
    return math.degrees(math.atan2(vx, vy)) % 360.0


def _radar(
    radar_id: str, x: float, y: float, heading_deg: float = 0.0,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """复制基础雷达的**全部物理参数**，只改标识、位置与姿态。"""
    base = base if base is not None else _load_base()["radar"]
    cfg = dict(base)
    cfg.update({
        "radar_id": radar_id,
        "x": x, "y": y, "z": 0.0,
        "heading_deg": heading_deg, "pitch_deg": 0.0, "roll_deg": 0.0,
        "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
        "timestamp_s": 0.0, "platform_id": radar_id, "is_active": True,
    })
    return cfg


def _target(
    target_id: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
    rcs_m2: Optional[float] = None, base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """复制基础目标的物理参数（含 RCS），只改标识、位置与速度。"""
    base = base if base is not None else _load_base()["targets"][0]
    cfg = dict(base)
    cfg.update({
        "target_id": target_id,
        "x": x, "y": y, "z": 0.0,
        "vx": vx, "vy": vy, "vz": 0.0,
        "heading_deg": _heading_of(vx, vy), "pitch_deg": 0.0, "roll_deg": 0.0,
        "timestamp_s": 0.0, "platform_id": target_id, "is_active": True,
    })
    if rcs_m2 is not None:
        cfg["rcs_m2"] = rcs_m2
    return cfg


def _sensor(
    sensor_id: str, mounting_id: str, max_range_m: float,
    update_period_s: float = 1.0,
    false_alarm_rate: float = 0.0,
    false_alarm_near_target_m: Optional[float] = None,
    az_fov_deg: float = 60.0,
) -> Dict[str, Any]:
    """构造一个雷达传感器配置段。

    作用距离显式给到 30 km：目标在 6~9 km，默认反解值比目标距离小一个量级，
    那属于配置错误而不是"压力"（v4.4 已经踩过一次，见 README §11E.1）。

    **`force_detection=True`（本模块四个场景统一使用）**：几何可见即检测到，
    不做概率漏检。理由：本模块要归因的是**关联/遮挡/虚警/通信**这四类失效，
    而 7 km、RCS 2 m² 下基线检测概率约 0.5（实测逐帧命中在 0/1/2 之间跳），
    随机的漏检会把"被遮挡"和"这一帧没探到"混在一起，让遮挡场景失去区分度。
    概率漏检本身已在 v4.2 的测量层评测里单独量化，不在这里重复。
    """
    cfg: Dict[str, Any] = {
        "sensor_id": sensor_id, "mounting_id": mounting_id, "sensor_kind": "radar",
        "max_range_m": max_range_m, "min_range_m": 0.0,
        "az_fov_deg": az_fov_deg, "el_fov_deg": 30.0,
        "update_period_s": update_period_s,
        "range_sigma_rel": 0.01, "range_sigma_abs_m": 5.0,
        "az_sigma_deg": 0.5, "el_sigma_deg": 0.5, "range_rate_sigma_mps": 1.0,
        "snr50_db": 6.0, "pd_slope_db": 2.0,
        "false_alarm_rate": false_alarm_rate,
        "tx_power_w": 18.0, "peak_gain_db": 30.0, "wavelength_m": 0.1,
        "bandwidth_hz": 1.0e6, "noise_figure_db": 3.0, "system_loss_db": 3.0,
        "temperature_k": 290.0, "observes_kind": "target", "provides_range": True,
        "force_detection": True,
        "seed": 42,
    }
    if false_alarm_near_target_m is not None:
        cfg["false_alarm_near_target_m"] = false_alarm_near_target_m
    return cfg


# ----------------------------------------------------------------------
# 场景定义
# ----------------------------------------------------------------------


@dataclass
class StressScenario:
    """一个压力场景。

    v4.5 P2 起分两组：`group="core"`（核心关联场景 S1–S4）与
    `group="system"`（系统级场景 S5–S8）。系统级场景多出的字段
    （机动表 / 对照轴 / OOSM 策略 / 偏差注入 / 中断窗口）**全部带默认值**，
    因此核心场景的构造代码一行都不用改，旧行为逐位不变。
    """

    scenario_id: str
    title_cn: str
    #: 这个场景要回答的问题
    question: str
    #: 预期暴露的失效模式（跑之前先声明，避免"看到什么就解释什么"）
    expected_failure: str
    #: 配置覆盖（radars / targets / sensors / occluders）
    overrides: Dict[str, Any] = field(default_factory=dict)
    #: 仿真步数（每步 1 s）
    steps: int = 30
    #: 真值目标数
    n_targets: int = 2
    #: 该场景里"本地雷达"的传感器 ID
    local_sensor_id: str = "SENSOR_LOCAL"
    #: 该场景里"远端雷达"的传感器 ID（单雷达路会把它剔除）
    remote_sensor_id: str = "SENSOR_REMOTE"
    notes: List[str] = field(default_factory=list)
    #: 遮挡窗口（秒，仅 S3 系列有）
    occlusion_window_s: Optional[Tuple[float, float]] = None

    # --- v4.5 系统级场景 ---
    #: "core" | "system"
    group: str = "core"
    #: 对照轴：`"policy"`（共享策略）或 `"oosm"`（乱序测量处理策略）
    axis: str = "policy"
    #: 该轴上的取值（留空即四路共享对照）
    axis_values: Tuple[str, ...] = ()
    #: `axis="oosm"` 时固定使用的共享策略
    fixed_policy: str = "constrained_share"
    #: 机动时刻表（S5）
    maneuvers: Tuple[Any, ...] = ()
    #: 后到的传感器 ID（S6 交接：航迹应当从它这里接上）
    incoming_sensor_id: str = ""
    #: 交接评测用的目标 ID（单目标场景）
    handover_target_id: str = ""
    #: 链路中断窗口（S7，与 comm 配置一致，用于算"中断期间/恢复后"）
    outage_windows: Tuple[Tuple[float, float], ...] = ()
    #: 额外的通信配置项（S7：突发丢包 / 恢复拥塞 / 乱序 / 差速延迟）
    comm_extras: Dict[str, Any] = field(default_factory=dict)
    #: 系统偏差注入：`{sensor_id: {字段: 值}}`（仅 `bias_variant` 变体生效）
    bias_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: 应用偏差注入的变体名（其余变体完全不碰传感器）
    bias_variant: str = "biased_share"
    #: 验证目标（与"预期失效模式"分开写：这条是**要验证的正面命题**）
    verification_goal: str = ""

    @property
    def label(self) -> str:
        return f"{self.scenario_id} {self.title_cn}"

    @property
    def variants(self) -> Tuple[str, ...]:
        """该场景沿对照轴要跑的变体列表。"""
        if self.axis_values:
            return tuple(self.axis_values)
        return ("single", "no_share", "ideal_share", "constrained_share")


def build_scenario_overrides(scenario_id: str) -> Dict[str, Any]:
    """返回场景的配置覆盖（场景几何的**唯一**来源）。"""
    base = _load_base()
    radar_base = base["radar"]
    target_base = base["targets"][0]

    def R(radar_id, x, y, heading):  # noqa: N802 - 局部简写，缩短几何表
        return _radar(radar_id, x, y, heading, base=radar_base)

    def T(target_id, x, y, vx, vy):  # noqa: N802
        return _target(target_id, x, y, vx, vy, base=target_base)

    if scenario_id == "S1":
        # 两目标相向而行，在本地雷达正前方约 7 km 处**交叉通过**。
        # 交叉点两者间距 → 0，最近邻关联的唯一依据就是位置残差，
        # 一旦预测位置偏一点就会换号。
        return {
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [T("TGT_A", 7000.0, -900.0, 0.0, 60.0),
                        T("TGT_B", 7000.0, 900.0, 0.0, -60.0)],
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0)],
            "occluders": [],
        }
    if scenario_id == "S2":
        # 三目标密集编队：横向间距 200 m，而 7 km 处量测标准差约 93 m
        # （距离 70 m + 横向 R·σθ ≈ 61 m），间距只有约 2.1σ。
        return {
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [T("TGT_1", 7000.0, -200.0, 0.0, 30.0),
                        T("TGT_2", 7000.0, 0.0, 0.0, 30.0),
                        T("TGT_3", 7000.0, 200.0, 0.0, 30.0)],
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0)],
            "occluders": [],
        }
    if scenario_id in ("S3", "S3L"):
        # 目标横向穿过一个球形遮挡区：本地雷达被切断数秒，远端雷达不受影响。
        #
        # 半径**由目标遮挡时长反解**，不用远场近似猜：
        #   点到直线距离 = x_o·|y| / sqrt(x_t² + y²)，令其 = r 解出阴影半宽
        #       |y| = r·x_t / sqrt(x_o² − r²)
        # 于是遮挡时长 = 2·|y| / v。S3 取 4.0 s、S3L 取 9.4 s。
        # （第一版按 asin(r/x_o) 的远场锥近似给出 301 m，而球离雷达只有
        #   4 km、目标在 7 km，实际阴影半宽是 710 m——近似在这里不成立。）
        seconds = 4.0 if scenario_id == "S3" else 9.4
        radius = _radius_for_occlusion_seconds(seconds)
        return {
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", 4000.0, -9000.0, 18.435)],
            "targets": [T("TGT_1", 7000.0, -1200.0, 0.0, 150.0)],
            "sensors": [_sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0),
                        _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0)],
            "occluders": [{"occluder_id": "BLOCK",
                           "shape": "sphere",
                           "center_x": OCCLUDER_X, "center_y": 0.0,
                           "center_z": 0.0,
                           "radius_m": round(radius, 3)}],
        }
    if scenario_id == "S4":
        # 真实目标附近的可控虚警 + 第二雷达**异步**（2.5 s 周期）+ 通信延迟。
        return {
            "radars": [R("RADAR_LOCAL", 0.0, 0.0, 90.0),
                       R("RADAR_REMOTE", -6000.0, 0.0, 90.0)],
            "targets": [T("TGT_1", 7000.0, -600.0, 0.0, 20.0),
                        T("TGT_2", 7000.0, 600.0, 0.0, -20.0)],
            "sensors": [
                _sensor("SENSOR_LOCAL", "RADAR_LOCAL", 30000.0,
                        update_period_s=1.0, false_alarm_rate=0.6,
                        false_alarm_near_target_m=300.0),
                _sensor("SENSOR_REMOTE", "RADAR_REMOTE", 30000.0,
                        update_period_s=2.5, false_alarm_rate=0.3,
                        false_alarm_near_target_m=300.0),
            ],
            "occluders": [],
        }
    raise ValueError(f"未知场景 {scenario_id!r}，只支持 {SCENARIO_IDS}")


#: 遮挡场景的几何常数（本地雷达在原点、目标沿 x=7000 横向穿过）
OCCLUDER_X = 4000.0
TARGET_CROSS_X = 7000.0
TARGET_CROSS_SPEED_MPS = 150.0
TARGET_CROSS_Y0 = 1200.0


def _shadow_half_width_m(radius_m: float) -> float:
    """球形遮挡体在目标航线上造成的**阴影半宽**（精确解）。

    雷达在原点、目标沿 x = TARGET_CROSS_X 横向运动时，
    球心 (OCCLUDER_X, 0) 到视线的距离为

        d(y) = OCCLUDER_X·|y| / sqrt(TARGET_CROSS_X² + y²)

    令 d(y) = r 解出 |y| = r·x_t / sqrt(x_o² − r²)。
    （远场锥近似 |y| ≈ (x_t − x_o)·tan(asin(r/x_o)) 在这里**不成立**：
     球离雷达只有 4 km 而目标在 7 km，两者分别是 171 m 与 710 m。）
    """
    denominator = OCCLUDER_X ** 2 - radius_m ** 2
    if denominator <= 0.0:
        return float("inf")
    return radius_m * TARGET_CROSS_X / math.sqrt(denominator)


def _radius_for_occlusion_seconds(seconds: float) -> float:
    """反解：要让目标被遮挡 `seconds` 秒，球半径该取多少。

    遮挡时长 = 2·|y_half| / v，且 |y_half| = r·x_t / sqrt(x_o² − r²)，
    解出 r = x_o·|y_half| / sqrt(x_t² + y_half²)。
    """
    half_width = seconds * TARGET_CROSS_SPEED_MPS / 2.0
    return (OCCLUDER_X * half_width /
            math.sqrt(TARGET_CROSS_X ** 2 + half_width ** 2))


def _occlusion_window(scenario_id: str) -> Optional[Tuple[float, float]]:
    """**由几何反解**的本地遮挡窗口（秒），不是估的。"""
    if scenario_id not in ("S3", "S3L"):
        return None
    occluder = build_scenario_overrides(scenario_id)["occluders"][0]
    half_width = _shadow_half_width_m(float(occluder["radius_m"]))
    enter = (TARGET_CROSS_Y0 - half_width) / TARGET_CROSS_SPEED_MPS
    leave = (TARGET_CROSS_Y0 + half_width) / TARGET_CROSS_SPEED_MPS
    return (max(0.0, enter), leave)


def _describe(scenario_id: str) -> Tuple[str, str, str, List[str]]:
    table = {
        "S1": (
            "两目标交叉",
            "两条航迹在交叉点附近会不会换号（ID switch）？会不会出现重复航迹？",
            "ID switch / 重复航迹：交叉时两者位置残差几乎相等，"
            "最近邻按最小残差选边，一旦选错就换号。",
            ["两目标以 120 m/s 相对速度交叉，交叉点在本地雷达正前方 7 km",
             "远端雷达在本地雷达另一侧，视角不同，可用于消歧"],
        ),
        "S2": (
            "密集编队（3 目标）",
            "间距只有约 2.1 倍量测标准差时，航迹会不会合并、互换或重复？",
            "航迹合并（measurement 全被一条航迹吃掉）/ 互换 / 重复航迹。",
            ["横向间距 200 m，7 km 处量测标准差约 93 m",
             "编队以 30 m/s 整体平移"],
        ),
        "S3": (
            "短时遮挡后重现（约 4 s）",
            "遮挡 4 s 后重现，Track ID 是否保持？会不会碎裂或新建重复航迹？",
            "遮挡 4 帧 < drop_after_misses=5，因此**应该**靠外推维持同一 "
            "Track ID（coasting）而不删除；若仍出现碎裂，说明外推/门限有问题。",
            ["球形遮挡体半径由目标遮挡时长反解（几何精确解，非远场近似）",
             "目标 150 m/s 穿过，本地遮挡 4.0 s（< drop_after_misses=5）",
             "远端雷达视线不经遮挡区，应能补盲"],
        ),
        "S3L": (
            "长时遮挡后重现（约 9.4 s）",
            "遮挡超过 drop_after_misses 时，重现后是同一 Track ID 还是新航迹？",
            "9.4 s ≈ 9 帧 > drop_after_misses=5，按基线配置应**删除**旧航迹、"
            "重现时**重新起始**，即 1 次碎裂 + 1 条新 Track ID。"
            "这是对基线外推能力的**上界刻画**，不是缺陷。",
            ["球形遮挡体半径由目标遮挡时长反解（几何精确解）",
             "遮挡 9.4 s（> drop_after_misses=5）",
             "drop_after_misses=5、max_measurement_age_s=3 s 均为基线默认值，未改"],
        ),
        "S4": (
            "目标附近虚警 + 异步远端 + 通信延迟",
            "虚警会不会夺走真实航迹？远端异步观测能否帮忙消歧？",
            "虚警被关联进真实航迹（关联准确率下降）或形成假航迹；"
            "远端测量因延迟/异步/过期而无法及时消歧。",
            ["本地虚警率 0.6/帧、远端 0.3/帧，均集中在真实检测附近 σ=300 m",
             "远端更新周期 2.5 s（异步），受限共享含 1.2 s 基延迟与 25% 丢包",
             "**只共享检测，不共享虚警**（否则虚警压力会同时来自两个平台，无法归因）"],
        ),
    }
    title, question, failure, notes = table[scenario_id]
    return title, question, failure, notes


def get_scenario(scenario_id: str) -> StressScenario:
    """按 ID 取场景对象。"""
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"未知场景 {scenario_id!r}，只支持 {SCENARIO_IDS}")
    title, question, failure, notes = _describe(scenario_id)
    overrides = build_scenario_overrides(scenario_id)
    return StressScenario(
        scenario_id=scenario_id,
        title_cn=title,
        question=question,
        expected_failure=failure,
        overrides=overrides,
        steps=30,
        n_targets=len(overrides["targets"]),
        notes=notes,
        occlusion_window_s=_occlusion_window(scenario_id),
    )


#: 全部场景（构造顺序即报告顺序）
STRESS_SCENARIOS: List[StressScenario] = [get_scenario(s) for s in SCENARIO_IDS]
