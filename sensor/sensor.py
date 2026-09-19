"""传感器模型：作用范围 / 视场 / 扫描周期 / 误差模型 / 可用状态（v4.2）。

分层位置
--------
```
真值世界 Scene ──► SensorSuite.observe(scene, t) ──► SensorReport / MeasurementRecord ──► fusion ──► 观测向量
    唯一真值              只读，不修改真值              带时间戳/协方差/置信度         只保留候选ID
```

**本模块只读真值，绝不修改真值。** 因此"噪声只改变看到什么、不改变真实轨迹与奖励"
这条不变式在结构上就成立，而不是靠自觉。`tests/test_sensor_layer.py` 有逐位断言。

两种传感器，物理上本来就不同
----------------------------
* `RadarSensor`：**主动**。发射→接收回波，可测**距离、方位、俯仰、径向速度**。
  检测概率由雷达方程算出的 SNR 经 ROC 得到。
* `EsmSensor`：**被动**。只接收雷达的泄漏辐射，**没有距离量测**
  （被动测距需要多站或机动，单站做不到）。因此它的测量是**方位/俯仰两维**，
  距离维标准差为 `inf`。把它做成"也能测距"会让多平台侦察场景失真。

每个传感器自己的属性（用户明确要求逐项独立）
--------------------------------------------
| 属性 | 含义 |
| --- | --- |
| `max_range_m` / `min_range_m` | 作用距离（远界/近界盲区） |
| `az_fov_deg` / `el_fov_deg` | 视场半角（相对自身机头/天线法向） |
| `update_period_s` | 扫描/更新周期；不是整数倍步长时按 `floor(t/period)` 判更新 |
| `range_sigma_*` / `az_sigma_deg` / ... | 测量误差模型 |
| `available` | 可用状态（关机/故障） |
| `false_alarm_rate` | 每帧虚警概率 |

判定顺序（**有严格先后，且被测试钉住**）
--------------------------------------
```
1. SENSOR_UNAVAILABLE   传感器关机/故障
2. NOT_UPDATED          未到更新时刻（时间维度，与空间维度正交）
3. BEYOND_RANGE         超出作用距离或落在近界盲区
4. OUT_OF_FOV           方位/俯仰超出视场
5. OCCLUDED             视线被遮挡体切断
6. MISSED_DETECTION     通过全部检查但本帧检测概率未命中
7. （否则）DETECTED
```
为什么是这个顺序：它是**代价从低到高、且互不遮蔽**的顺序。
先判可用性（最便宜、且关闭的传感器不该产生任何几何判断）；
再判时间（未更新时**根本不该用当前真值去算几何**，否则会产生
"没更新却报出精确距离"的矛盾）；然后按空间→遮挡→概率。
如果把 MISSED_DETECTION 放在前面，就会出现"目标在视场外却报成丢测"，
统计上把"指向问题"错记成"概率问题"，消融实验直接失效。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from engine.equations import (
    db2lin,
    lin2db,
    logistic_prob,
    radar_echo_power_w,
    thermal_noise_w,
)
from engine.geometry import (
    Attitude,
    Pose,
    Vec3,
    enu_to_spherical,
    normalize_angle_deg,
    relation,
)
if TYPE_CHECKING:  # pragma: no cover - 仅供类型注解
    #: 见 `sensor/config.py::_occlusion_model` 里的"导入环"说明：
    #: `sensor.occlusion` 需要 `engine.geometry`，而 `engine/__init__` 会导入
    #: `engine.env`，后者又导入 `sensor.config` → `sensor.sensor`。
    #: 因此这里只能用**类型检查期**导入，运行期在真正需要时再导入。
    from sensor.occlusion import OcclusionModel
from sensor.record import (
    MeasurementRecord,
    NoDataReason,
    SensorReport,
    SuiteReport,
    TargetOutcome,
    diagonal_covariance,
    infinite_range_covariance,
)

#: 传感器类型
KIND_RADAR_SENSOR = "radar"
KIND_ESM_SENSOR = "esm"


@dataclass
class SensorConfig:
    """单个传感器的配置。四类属性全部**逐传感器独立**。"""

    sensor_id: str
    #: 该传感器装在哪个实体上（对应 `SceneEntity.entity_id`）
    mounting_id: str
    sensor_kind: str = KIND_RADAR_SENSOR

    # --- 1) 作用范围 ---
    max_range_m: float = 1.0e9
    min_range_m: float = 0.0

    # --- 2) 视场（相对自身机头的半角，度）---
    az_fov_deg: float = 180.0
    el_fov_deg: float = 90.0

    # --- 3) 扫描 / 更新周期 ---
    update_period_s: float = 1.0

    # --- 4) 测量误差模型 ---
    #: 距离误差 = range * range_sigma_rel + range_sigma_abs
    range_sigma_rel: float = 0.0
    range_sigma_abs_m: float = 0.0
    az_sigma_deg: float = 0.0
    el_sigma_deg: float = 0.0
    range_rate_sigma_mps: float = 0.0

    # --- 检测模型（ROC）---
    snr50_db: float = 6.0
    pd_slope_db: float = 2.0
    #: 每帧虚警概率
    false_alarm_rate: float = 0.0
    #: **虚警空间集中度**（v4.5 多目标压力测试用）。
    #:
    #: * `None`（默认）：虚警在视场/作用距离内**均匀随机**——旧行为，逐位不变；
    #: * 给定正数 `σ_m`：本帧若已有真实检测，则虚警以**随机挑一条真实检测**为
    #:   中心、按各轴独立高斯 `σ_m` 偏移生成。
    #:
    #: 为什么需要它：均匀虚警几乎总落在空域里，只会形成一看就假的孤立点；
    #: 而"真实目标附近的虚警"才会**与真实航迹竞争关联**，
    #: 这正是虚警能否夺取真实航迹的关键工况。
    false_alarm_near_target_m: Optional[float] = None

    # --- 其他 ---
    available: bool = True
    #: 是否产生距离量测（被动传感器为 False）
    provides_range: bool = True
    #: 强制"只要几何上可见就一定检测到"（用于 ideal-measurement 对照组）。
    #: 打开时检测器变成**确定性**的：不做概率漏检、置信度恒为 1。
    #: 这样 ideal 与 realistic 两组共享**完全相同**的可见性判定代码路径，
    #: 差异只有噪声/虚警/概率漏检三项。
    force_detection: bool = False
    #: 该传感器**观测哪类实体**（雷达观测"target"，ESM 观测"radar"）
    observes_kind: str = "target"
    #: 雷达方程所需的自身参数（仅主动传感器使用）
    tx_power_w: float = 0.0
    peak_gain_db: float = 30.0
    wavelength_m: float = 0.1
    bandwidth_hz: float = 1.0e6
    noise_figure_db: float = 3.0
    system_loss_db: float = 3.0
    temperature_k: float = 290.0
    seed: int = 0

    # --- v4.5 系统级压力测试：传感器系统偏差 ---
    #: 这些字段**默认全为零 / 1**，因此不配置时测量与旧版**逐位一致**。
    #: 偏差只加在**该传感器自己产出的测量**上，真值一个字节都不动。
    #: 算法侧看不到"这条测量有偏"——偏差是传感器缺陷，不是可用信息。

    #: 距离固定偏差（米）
    range_bias_m: float = 0.0
    #: 方位/俯仰固定偏差（度）
    az_bias_deg: float = 0.0
    el_bias_deg: float = 0.0
    #: 缓慢漂移速率（每秒），从 `bias_start_s` 起线性累积
    range_bias_drift_mps: float = 0.0
    az_bias_drift_degps: float = 0.0
    #: 时钟偏移（秒）：**上报时刻 = 真实时刻 + clock_offset_s**。
    #: 正值会让测量"看起来更新"（年龄被低估），负值反之。
    clock_offset_s: float = 0.0
    #: **噪声低估系数**：上报的 σ 乘以该系数（<1 表示"谎报精度"）。
    #: 实际加噪仍用真实 σ，只是上报得更小——这直接骗过融合的加权。
    noise_underreport_factor: float = 1.0
    #: 偏差生效时刻（秒）：之前无偏，便于做"偏差突然出现"的对照
    bias_start_s: float = 0.0

    def has_bias(self) -> bool:
        """是否配置了任何非零偏差（默认 False，即旧行为）。"""
        return bool(
            self.range_bias_m or self.az_bias_deg or self.el_bias_deg
            or self.range_bias_drift_mps or self.az_bias_drift_degps
            or self.clock_offset_s
            or self.noise_underreport_factor != 1.0
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorConfig":
        known = set(cls.__dataclass_fields__)
        unknown = [k for k in data if k not in known]
        if unknown:
            raise ValueError(
                f"传感器配置出现未知字段 {unknown}；支持的字段为 {sorted(known)}"
            )
        return cls(**{k: v for k, v in data.items()})

    def validate(self) -> None:
        if self.max_range_m <= 0:
            raise ValueError(f"[{self.sensor_id}] max_range_m 必须为正")
        if self.min_range_m < 0 or self.min_range_m >= self.max_range_m:
            raise ValueError(
                f"[{self.sensor_id}] min_range_m 必须落在 [0, max_range_m) 内"
            )
        if not 0.0 < self.az_fov_deg <= 180.0:
            raise ValueError(f"[{self.sensor_id}] az_fov_deg 必须落在 (0, 180]")
        if not 0.0 < self.el_fov_deg <= 90.0:
            raise ValueError(f"[{self.sensor_id}] el_fov_deg 必须落在 (0, 90]")
        if self.update_period_s <= 0:
            raise ValueError(f"[{self.sensor_id}] update_period_s 必须为正")
        if not 0.0 <= self.false_alarm_rate <= 1.0:
            raise ValueError(f"[{self.sensor_id}] false_alarm_rate 必须落在 [0, 1]")
        if self.false_alarm_near_target_m is not None:
            if self.false_alarm_near_target_m <= 0.0:
                raise ValueError(
                    f"[{self.sensor_id}] false_alarm_near_target_m 必须为正"
                    "（或留空表示均匀虚警）"
                )
        if self.pd_slope_db <= 0:
            raise ValueError(f"[{self.sensor_id}] pd_slope_db 必须为正")
        if self.noise_underreport_factor <= 0.0:
            raise ValueError(
                f"[{self.sensor_id}] noise_underreport_factor 必须为正"
                "（<1 表示谎报精度，1 表示如实上报）"
            )
        if self.bias_start_s < 0.0:
            raise ValueError(f"[{self.sensor_id}] bias_start_s 不能为负")


class Sensor:
    """传感器基类。

    子类实现 `_detection_evidence()`（给出 SNR 等检测依据）与
    `_measure(entity_pose, truth_geometry)`（产生测量）。

    ⚠️ 本类**只读**真值：所有方法都不得写 `scene` / 实体的任何字段。
    """

    def __init__(self, config: SensorConfig) -> None:
        config.validate()
        self.config = config
        self.sensor_id = config.sensor_id
        self.sensor_kind = config.sensor_kind
        #: 逐传感器独立随机流（按 传感器ID 派生，互不干扰，便于消融）
        self._rng = random.Random("%s:%s" % (config.seed, config.sensor_id))
        #: 候选编号注册表：真值 ID -> 本传感器自己的候选编号（**内部使用**）
        self._candidate_of_truth: Dict[str, str] = {}
        self._candidate_counter = 0
        self._false_alarm_counter = 0
        #: 上一次更新时刻（用于 NOT_UPDATED 与沿用值）
        self._last_update_time: Optional[float] = None
        #: 上一次扫描**真正测到**的候选记录（供"未到更新时刻"时沿用）
        self._held_records: Dict[str, MeasurementRecord] = {}
        #: 实际发生扫描的时刻序列（**与是否检测到目标无关**）。
        #: 单独记录它是为了把"传感器节奏"与"输出节奏"分开：
        #: 用测量时刻算间隔会把漏检误算成周期抖动（实测周期 1 s 的传感器
        #: 因偶发漏检被算出 1.21±0.43 s 的间隔），那是统计口径错误。
        self.scan_times: List[float] = []
        #: 观测上下文（干扰功率、发射功率、波束增益等由调用方注入）
        self._context: Dict[str, Any] = {}
        #: 统计
        self.stats: Dict[str, int] = {"scans": 0, "detections": 0, "false_alarms": 0}

    # ------------------------------------------------------------------
    # 候选编号（算法只看得到候选，看不到真值）
    # ------------------------------------------------------------------

    def candidate_of(self, truth_id: str) -> str:
        """取/分配某真值实体的候选编号。**分配过程对算法不可见**。"""
        label = self._candidate_of_truth.get(truth_id)
        if label is None:
            self._candidate_counter += 1
            label = f"{self.sensor_id}-C{self._candidate_counter}"
            self._candidate_of_truth[truth_id] = label
        return label

    def truth_of_candidate(self, candidate_id: str) -> Optional[str]:
        """反查候选编号对应的真值 ID。

        ⚠️ **仅供评测使用**（算误差、算关联正确率）。
        决策算法**不得**调用本方法——那等于把真值喂给了算法。
        `sensor/fusion.py` 打包观测时不调用它，并有单元测试断言这一点。
        """
        for truth_id, label in self._candidate_of_truth.items():
            if label == candidate_id:
                return truth_id
        return None

    @property
    def candidate_count(self) -> int:
        return self._candidate_counter

    def reset(self) -> None:
        self._rng = random.Random("%s:%s" % (self.config.seed, self.sensor_id))
        self._candidate_of_truth.clear()
        self._candidate_counter = 0
        self._false_alarm_counter = 0
        self._last_update_time = None
        self._held_records.clear()
        self._context = {}
        self.scan_times = []
        for key in self.stats:
            self.stats[key] = 0

    # ------------------------------------------------------------------
    # 时间 / 可用性
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.config.available)

    def is_update_time(self, time_s: float) -> bool:
        """本时刻是否到了更新点。

        用 `floor(t / period)` 判据，而不是"累计步数取模"：
        这样即使仿真步长不是周期的整数倍（例如 period=2.5 s、dt=1 s），
        更新节奏依旧正确，且与"步"这个离散概念解耦。
        """
        index = math.floor(time_s / self.config.update_period_s + 1e-9)
        if self._last_update_time is None:
            return True
        last_index = math.floor(self._last_update_time / self.config.update_period_s + 1e-9)
        return index > last_index

    # ------------------------------------------------------------------
    # 可见性判定（**顺序即语义**）
    # ------------------------------------------------------------------

    def _range_ok(self, distance: float) -> bool:
        return self.config.min_range_m <= distance <= self.config.max_range_m

    def _in_fov(self, sensor_pose: Pose, target_pose: Pose) -> Tuple[bool, float, float]:
        """目标是否在视场内。返回 (是否在视场, 机体方位, 机体俯仰)。

        视场以**传感器自身机头**为轴（`attitude.enu_to_body` 已经实现了
        姿态→机体系的转换，因此雷达天线一转，视场跟着转）。
        """
        delta = target_pose.position - sensor_pose.position
        body = sensor_pose.attitude.enu_to_body(delta)
        distance = delta.norm()
        if distance <= 0.0:
            return True, 0.0, 0.0
        bearing = normalize_angle_deg(math.degrees(math.atan2(body.y, body.x)))
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, body.z / distance))))
        in_fov = (
            abs(bearing) <= self.config.az_fov_deg
            and abs(elevation) <= self.config.el_fov_deg
        )
        return in_fov, bearing, elevation

    # ------------------------------------------------------------------
    # 测量产生（子类实现）
    # ------------------------------------------------------------------

    def _detection_probability(self, distance: float, entity: Any) -> float:
        """本帧的检测概率。基类返回 1（子类覆盖）。"""
        return 1.0

    def _measure(
        self, sensor_pose: Pose, entity_pose: Pose, true_range: float,
        true_az: float, true_el: float, true_vr: float, entity: Any,
        detection_prob: float,
    ) -> MeasurementRecord:
        """产生一条测量（加噪）。基类不含任何噪声，子类覆盖。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def observe(
        self,
        scene: Any,
        time_s: float,
        occlusion: Optional[OcclusionModel] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> SensorReport:
        """对场景做一次观测，返回本传感器的报告（**只读真值**）。"""
        report = SensorReport(
            sensor_id=self.sensor_id,
            sensor_kind=self.sensor_kind,
            time_s=time_s,
            updated=False,
        )
        self._context = dict(extra_context or {})

        # ---- 1) 可用性 ----
        # 传感器关机/故障时**必须清空沿用记忆**，不能继续输出上一次的测量。
        # 理由：一个已经不可用的传感器还在"提供数据"，会让算法以为它仍在工作——
        # 这是安全语义问题，不是实现细节。因此这里既不发 held，也清空缓存。
        if not self.available:
            self._held_records.clear()
            for entity in self._observable_entities(scene):
                report.outcomes.append(
                    self._no_data(entity, time_s, NoDataReason.SENSOR_UNAVAILABLE,
                                  scene, occlusion)
                )
            return report

        # ---- 2) 更新时刻 ----
        # 未到更新时刻：**不拿当前真值去算几何**（那会产生"没更新却报出精确距离"
        # 的矛盾），只沿用上一次真正测到的记录。沿用值来自传感器自己的候选
        # 登记表，生成过程不读任何真值。
        if not self.is_update_time(time_s):
            for entity in self._observable_entities(scene):
                report.outcomes.append(
                    self._no_data(entity, time_s, NoDataReason.NOT_UPDATED,
                                  scene, occlusion)
                )
            report.held = self._held_for_time(time_s)
            return report

        report.updated = True
        self.stats["scans"] += 1
        self.scan_times.append(time_s)
        self._last_update_time = time_s

        sensor_pose = self.mount_pose(scene)

        # ---- 3~7) 逐个可观测实体 ----
        for entity in self._observable_entities(scene):
            report.outcomes.append(
                self._observe_entity(scene, entity, sensor_pose, time_s,
                                     occlusion, extra_context, report)
            )

        # ---- 虚警（与真实目标无关的假测量）----
        if self.config.false_alarm_rate > 0.0:
            if self._rng.random() < self.config.false_alarm_rate:
                near = None
                if (self.config.false_alarm_near_target_m is not None
                        and report.detections):
                    # 挑一条本帧真实检测作为"附近"的锚点（评测侧才知道它是真的，
                    # 传感器自己并不把 truth_id 传出去）
                    near = self._rng.choice(list(report.detections))
                report.false_alarms.append(
                    self._make_false_alarm(sensor_pose, time_s, near=near)
                )
                self.stats["false_alarms"] += 1

        # 扫描帧作废上一帧的记忆，只保留本帧真正测到的候选。
        # 好处：绝不会出现"目标已经离开视场，算法还在用旧航迹"的幻影航迹。
        # 代价：目标被漏检一次就彻底丢失，没有航迹外推（见 README 局限说明）。
        self._held_records = {
            m.candidate_id: m for m in report.detections
        }
        return report

    def apply_bias(self, record: MeasurementRecord, time_s: float) -> None:
        """把该传感器的**系统偏差**加到一条刚产出的测量上（就地修改）。

        ⚠️ 三条纪律：

        1. **只动测量，不动真值**。`record.truth_*` 由调用方在此之前写入，
           本方法一个都不碰；偏差因此是"传感器学会的错"，不是"世界变了"。
        2. **算法不可见**。偏差不写任何标记字段（消息载荷白名单里也没有），
           接收方无从得知这条测量有偏——这正是要测的东西。
        3. **默认零偏差时不改变任何数值**：`has_bias()` 为假就直接返回，
           因此旧实验的测量逐位不变。

        `time_s` 为**真实时刻**（用于算漂移量与生效判定）。
        """
        cfg = self.config
        if not cfg.has_bias():
            return
        elapsed = time_s - cfg.bias_start_s
        if elapsed <= 0.0:
            return

        drift_range = cfg.range_bias_drift_mps * elapsed
        drift_az = cfg.az_bias_drift_degps * elapsed
        d_range = cfg.range_bias_m + drift_range
        d_az = cfg.az_bias_deg + drift_az
        d_el = cfg.el_bias_deg

        if record.range_m is not None and d_range:
            record.range_m = max(0.0, float(record.range_m) + d_range)
        if record.azimuth_deg is not None and d_az:
            record.azimuth_deg = normalize_angle_deg(
                float(record.azimuth_deg) + d_az
            )
        if record.elevation_deg is not None and d_el:
            record.elevation_deg = float(record.elevation_deg) + d_el

        # 噪声低估：**横向上报**变小，实际误差不变
        factor = cfg.noise_underreport_factor
        if factor != 1.0:
            for field in ("std_range_m", "std_az_deg", "std_el_deg"):
                value = getattr(record, field, None)
                if value is not None:
                    setattr(record, field, float(value) * factor)
            covariance = getattr(record, "covariance", None)
            if isinstance(covariance, list):
                # 3×3 对角协方差：σ → σ·f 意味着方差 → 方差·f²
                scaled: List[List[float]] = []
                for row_index, row in enumerate(covariance):
                    new_row = []
                    for col_index, value in enumerate(row):
                        if row_index == col_index and value != float("inf"):
                            new_row.append(float(value) * factor * factor)
                        else:
                            new_row.append(value)
                    scaled.append(new_row)
                record.covariance = scaled

        # 时钟偏移：上报时刻整体平移
        if cfg.clock_offset_s and getattr(record, "time_s", None) is not None:
            record.time_s = float(record.time_s) + cfg.clock_offset_s

    def _held_for_time(self, time_s: float) -> List[MeasurementRecord]:
        """把上一次扫描的记录"沿用"到当前时刻（更新 `age_s`）。

        复制而非原地修改：原始记录保留了它自己的真实测量时刻，
        这样导出出来的 CSV 里"测量时刻"与"年龄"不会互相矛盾。
        """
        held: List[MeasurementRecord] = []
        for record in self._held_records.values():
            clone = replace(
                record,
                is_fresh=False,
                age_s=max(0.0, time_s - record.time_s),
            )
            held.append(clone)
        return held

    def mount_pose(self, scene: Any) -> Pose:
        """传感器所在平台的位姿（视场与误差都相对它定义）。"""
        return scene.by_id(self.config.mounting_id).pose

    def _observable_entities(self, scene: Any) -> List[Any]:
        return scene.of_kind(self.config.observes_kind, active_only=True)

    def _no_data(
        self, entity: Any, time_s: float, reason: NoDataReason,
        scene: Any, occlusion: Optional[OcclusionModel],
    ) -> TargetOutcome:
        truth_range = None
        try:
            truth_range = scene.by_id(self.config.mounting_id).range_to_entity(entity)
        except Exception:  # noqa: BLE001 - 诊断字段，失败不影响主流程
            truth_range = None
        return TargetOutcome(
            sensor_id=self.sensor_id,
            time_s=time_s,
            status="no_data",
            reason=reason.value,
            candidate_id="",
            record=None,
            truth_id=entity.entity_id,
            truth_range_m=truth_range,
        )

    def _observe_entity(
        self, scene: Any, entity: Any, sensor_pose: Pose, time_s: float,
        occlusion: Optional[OcclusionModel],
        extra_context: Optional[Dict[str, Any]],
        report: SensorReport,
    ) -> TargetOutcome:
        entity_pose = entity.pose
        delta = entity_pose.position - sensor_pose.position
        distance = delta.norm()
        spherical = enu_to_spherical(delta)
        true_az, true_el = spherical.azimuth_deg, spherical.elevation_deg

        # ---- 3) 作用距离 ----
        if not self._range_ok(distance):
            return self._no_data(entity, time_s, NoDataReason.BEYOND_RANGE,
                                 scene, occlusion)

        # ---- 4) 视场 ----
        in_fov, bearing, elevation = self._in_fov(sensor_pose, entity_pose)
        if not in_fov:
            return self._no_data(entity, time_s, NoDataReason.OUT_OF_FOV,
                                 scene, occlusion)

        # ---- 5) 遮挡 ----
        if occlusion is not None and occlusion.is_occluded(
            sensor_pose.position, entity_pose.position
        ):
            return self._no_data(entity, time_s, NoDataReason.OCCLUDED,
                                 scene, occlusion)

        # ---- 6) 检测概率 ----
        pd_value = self._detection_probability(distance, entity)
        pd_value = max(0.0, min(1.0, pd_value))
        if (
            not self.config.force_detection
            and pd_value < 1.0
            and self._rng.random() > pd_value
        ):
            return self._no_data(entity, time_s, NoDataReason.MISSED_DETECTION,
                                 scene, occlusion)

        # ---- 7) 检测成功：产生测量 ----
        los = delta / distance if distance > 0.0 else Vec3()
        true_vr = (entity_pose.velocity - sensor_pose.velocity).dot(los)

        record = self._measure(
            sensor_pose, entity_pose, distance, true_az, true_el, true_vr,
            entity, pd_value,
        )
        record.time_s = time_s
        record.sensor_id = self.sensor_id
        record.sensor_kind = self.sensor_kind
        record.candidate_id = self.candidate_of(entity.entity_id)
        record.confidence = 1.0 if self.config.force_detection else pd_value
        record.is_fresh = True
        record.age_s = 0.0
        # 真值字段：**只用于评测**，不会进入算法输入
        record.truth_id = entity.entity_id
        record.truth_range_m = distance
        record.truth_azimuth_deg = true_az
        record.truth_elevation_deg = true_el
        record.truth_range_rate_mps = true_vr

        # ---- 8) 系统偏差（v4.5）：只污染测量，不动上面写入的真值 ----
        self.apply_bias(record, time_s)

        self.stats["detections"] += 1
        report.detections.append(record)
        return TargetOutcome(
            sensor_id=self.sensor_id,
            time_s=time_s,
            status="detected",
            reason=NoDataReason.NONE.value,
            candidate_id=record.candidate_id,
            record=record,
            truth_id=entity.entity_id,
            truth_range_m=distance,
        )

    def _make_false_alarm(
        self, sensor_pose: Pose, time_s: float,
        near: Optional[MeasurementRecord] = None,
    ) -> MeasurementRecord:
        """在视场/作用距离内随机造一条**假测量**。

        虚警没有任何真值 ID（`truth_id=None`），因此算法完全无法把它与
        真实目标区分开——这正是虚警的意义。评测侧也只能通过
        `is_false_alarm` 标记来统计它（该标记**不进算法输入**）。

        `near` 给定时（见 `SensorConfig.false_alarm_near_target_m`），
        虚警以这条**真实检测**为中心做高斯偏移，用于压力测试
        "目标附近的虚警能否夺取真实航迹"。
        """
        self._false_alarm_counter += 1
        label = f"{self.sensor_id}-FA{self._false_alarm_counter}"
        cfg = self.config
        rng = self._rng
        if near is not None and near.range_m is not None:
            sigma = float(cfg.false_alarm_near_target_m or 0.0)
            distance = max(cfg.min_range_m,
                           float(near.range_m) + rng.gauss(0.0, sigma))
            distance = min(distance, cfg.max_range_m)
            # 方位/俯仰的偏移按"横向偏移量"换算成角度，保证 σ 的空间含义一致
            lateral_az = rng.gauss(0.0, sigma)
            lateral_el = rng.gauss(0.0, sigma)
            bearing = math.degrees(
                math.atan2(lateral_az, max(float(near.range_m), 1.0))
            )
            elevation = math.degrees(
                math.atan2(lateral_el, max(float(near.range_m), 1.0))
            )
            # 偏移是**叠加在真实检测的方位上**的，否则"附近"就无从谈起
            bearing = float(near.azimuth_deg or 0.0) + bearing
            elevation = float(near.elevation_deg or 0.0) + elevation
        else:
            bearing = rng.uniform(-cfg.az_fov_deg, cfg.az_fov_deg)
            elevation = rng.uniform(-cfg.el_fov_deg, cfg.el_fov_deg)
            distance = rng.uniform(max(cfg.min_range_m, 1.0), cfg.max_range_m)

        forward, right, up = sensor_pose.attitude.body_basis()
        direction = (
            forward + right * math.tan(math.radians(bearing))
            + up * math.tan(math.radians(elevation))
        )
        los = direction.normalized()
        # 假测量没有真值可依，误差按该传感器的名义标准差给
        std_range = self._range_sigma(distance)
        record = MeasurementRecord(
            sensor_id=self.sensor_id,
            candidate_id=label,
            time_s=time_s,
            sensor_kind=self.sensor_kind,
            range_m=distance if cfg.provides_range else None,
            azimuth_deg=None,
            elevation_deg=None,
            range_rate_mps=None,
            std_range_m=std_range if cfg.provides_range else None,
            std_az_deg=cfg.az_sigma_deg,
            std_el_deg=cfg.el_sigma_deg,
            covariance=(
                diagonal_covariance(std_range, cfg.az_sigma_deg, cfg.el_sigma_deg)
                if cfg.provides_range
                else infinite_range_covariance(cfg.az_sigma_deg, cfg.el_sigma_deg)
            ),
            confidence=0.0,
            is_false_alarm=True,
            truth_id=None,
        )
        # 绝坐标：由机体系方向 + 平台姿态反解
        absolute = sensor_pose.attitude.body_to_enu(los)
        az_el = enu_to_spherical(absolute)
        record.azimuth_deg = az_el.azimuth_deg
        record.elevation_deg = az_el.elevation_deg
        # 有偏传感器产生的虚警同样带偏（虚警与真实检测走同一套测角/测距硬件）
        self.apply_bias(record, time_s)
        return record

    def _range_sigma(self, distance: float) -> float:
        return abs(distance) * self.config.range_sigma_rel + self.config.range_sigma_abs_m

    # ------------------------------------------------------------------

    def describe(self) -> str:
        cfg = self.config
        fov = (f"视场 ±{cfg.az_fov_deg:g}°×±{cfg.el_fov_deg:g}°"
               if cfg.az_fov_deg < 180.0 else "视场 全向")
        return (
            f"{cfg.sensor_id}[{cfg.sensor_kind}] 装在 {cfg.mounting_id}｜"
            f"作用距离 {cfg.min_range_m:g}~{cfg.max_range_m:g} m｜{fov}｜"
            f"周期 {cfg.update_period_s:g}s｜"
            f"σr={cfg.range_sigma_rel:g}rel+{cfg.range_sigma_abs_m:g}m "
            f"σaz={cfg.az_sigma_deg:g}° σel={cfg.el_sigma_deg:g}°｜"
            f"虚警率 {cfg.false_alarm_rate:g}｜"
            f"{'可用' if cfg.available else '不可用'}"
        )

    def to_dict(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "sensor_id": cfg.sensor_id,
            "sensor_kind": cfg.sensor_kind,
            "mounting_id": cfg.mounting_id,
            "min_range_m": cfg.min_range_m,
            "max_range_m": cfg.max_range_m,
            "az_fov_deg": cfg.az_fov_deg,
            "el_fov_deg": cfg.el_fov_deg,
            "update_period_s": cfg.update_period_s,
            "range_sigma_rel": cfg.range_sigma_rel,
            "range_sigma_abs_m": cfg.range_sigma_abs_m,
            "az_sigma_deg": cfg.az_sigma_deg,
            "el_sigma_deg": cfg.el_sigma_deg,
            "range_rate_sigma_mps": cfg.range_rate_sigma_mps,
            "false_alarm_rate": cfg.false_alarm_rate,
            "provides_range": cfg.provides_range,
            "observes_kind": cfg.observes_kind,
            "available": cfg.available,
        }


# ----------------------------------------------------------------------
# 主动雷达传感器
# ----------------------------------------------------------------------


class RadarSensor(Sensor):
    """主动雷达传感器：可测距离/方位/俯仰/径向速度。

    检测概率来自**雷达方程 + ROC**（复用 `engine/equations.py`，
    不改动任何物理公式）：

        S   = Pt·G²·λ²·σ / ((4π)³·R⁴·L)
        N   = k·T·B·F
        SNR = S / (N + J_eff)
        Pd  = 1 / (1 + exp(-(SNR_dB - snr50_dB)/slope))

    其中 `J_eff` 由调用方以"传感器测得的噪声基底抬升"形式传入
    （`extra_context["interference_w"]`），这样传感器层不必知道干扰模型细节。
    """

    def _detection_probability(self, distance: float, entity: Any) -> float:
        cfg = self.config
        rcs = float(getattr(entity, "rcs_m2", 1.0))
        echo_w = radar_echo_power_w(
            pt_w=self.effective_tx_power_w(),
            gain_db=cfg.peak_gain_db,
            wavelength_m=cfg.wavelength_m,
            rcs_m2=rcs,
            range_m=distance,
            system_loss_db=cfg.system_loss_db,
        )
        noise_w = thermal_noise_w(cfg.bandwidth_hz, cfg.noise_figure_db, cfg.temperature_k)
        return logistic_prob(
            self._snr_db(echo_w, noise_w), cfg.snr50_db, cfg.pd_slope_db
        )

    def effective_tx_power_w(self) -> float:
        """本帧**实际使用**的发射功率（瓦）。

        为什么需要这个方法：`cfg.tx_power_w` 是构造时的静态配置，
        而 LPI 场景里发射功率是**每步都在变的动作**。若主动传感器的检测
        仍按静态配置算，就会出现"仿真器按新动作记账、传感器按初始化配置
        生成测量"的不一致——同一时刻、同一节点、同一次动作对不上。

        取值优先级：
        1. `_context["tx_power_w"]`：由环境为**受控雷达**注入的本步执行功率
           （`engine/env.py::_sensor_context`，来源是 `sim.step()` 实际执行的档位）；
        2. 否则退回 `cfg.tx_power_w`：非受控平台（例如旁观的第二部雷达）
           的功率不受本环境的动作控制，保持它自己的配置值。

        这条规则保证：**动作 → 主仿真 → 传感器测量**是同一个数，
        而不是两个各自演化的数。
        """
        injected = self._context.get("tx_power_w")
        if injected is None:
            return float(self.config.tx_power_w)
        return float(injected)

    def _snr_db(self, echo_w: float, noise_w: float) -> float:
        interference_w = float(self._context.get("interference_w", 0.0))
        total_noise = noise_w + interference_w
        if total_noise <= 0.0:
            return float("-inf")
        return lin2db(echo_w / total_noise)

    def observe(self, scene: Any, time_s: float,
                occlusion: Optional[OcclusionModel] = None,
                extra_context: Optional[Dict[str, Any]] = None) -> SensorReport:
        self._context = dict(extra_context or {})
        return super().observe(scene, time_s, occlusion, extra_context)

    def _measure(
        self, sensor_pose: Pose, entity_pose: Pose, true_range: float,
        true_az: float, true_el: float, true_vr: float, entity: Any,
        detection_prob: float,
    ) -> MeasurementRecord:
        cfg = self.config
        rng = self._rng

        std_range = self._range_sigma(true_range)
        measured_range = true_range + (rng.gauss(0.0, std_range) if std_range > 0 else 0.0)
        measured_az = true_az + (rng.gauss(0.0, cfg.az_sigma_deg)
                                 if cfg.az_sigma_deg > 0 else 0.0)
        measured_el = true_el + (rng.gauss(0.0, cfg.el_sigma_deg)
                                 if cfg.el_sigma_deg > 0 else 0.0)
        measured_vr = true_vr + (rng.gauss(0.0, cfg.range_rate_sigma_mps)
                                 if cfg.range_rate_sigma_mps > 0 else 0.0)
        measured_range = max(0.0, measured_range)

        # --- 传感器从原始量测反推的两个派生量（真实雷达确实这么做）---
        noise_w = thermal_noise_w(cfg.bandwidth_hz, cfg.noise_figure_db, cfg.temperature_k)
        interference_w = float(self._context.get("interference_w", 0.0))
        # ⚠️ 三处必须用**同一个** Pt：正演回波功率、反演 RCS、以及检测概率。
        # 若正演用执行功率、反演仍用静态配置，σ̂ 会被系统性缩放
        # （Pt_cfg/Pt_actual 倍）：实测档位 0（0.5 W）配 18 W 的配置时，
        # RCS 估计会偏小 36 倍——这类误差不会报错，只会静静地把结论带偏。
        pt_w = self.effective_tx_power_w()
        echo_w = radar_echo_power_w(
            pt_w=pt_w, gain_db=cfg.peak_gain_db,
            wavelength_m=cfg.wavelength_m, rcs_m2=float(getattr(entity, "rcs_m2", 1.0)),
            range_m=measured_range, system_loss_db=cfg.system_loss_db,
        )
        snr_measured_db = self._snr_db(echo_w, noise_w)
        # 由回波功率反推 RCS。推导：
        #     echo = Pt·G²·λ²·σ / ((4π)³·R⁴·L)
        #  => σ̂ = echo · ((4π)³·R̂⁴·L) / (Pt·G²·λ²)
        # ⚠️ 这里必须用**回波功率本身**，不能先用 `echo/SNR` 换成噪声功率再乘——
        # 那会漏掉一个 SNR 线性因子，得到 σ̂ ≈ σ/SNR（本场景下偏小约 10 倍）。
        # 这个错误当时让规则策略以为目标 RCS 比实际小一个量级，
        # 于是永远选最大功率（三组对照里规则策略退化成 70 W 恒功率）。
        # 用测量距离 R̂ 代入，因此距离误差会通过 (R_true/R̂)⁴ 耦合进 RCS 估计，
        # 这正是雷达方程本身的性质。
        gain = db2lin(cfg.peak_gain_db)
        numerator = ((4.0 * math.pi) ** 3) * (measured_range ** 4) * db2lin(cfg.system_loss_db)
        denominator = pt_w * gain * gain * (cfg.wavelength_m ** 2)
        rcs_est = (
            echo_w * numerator / denominator
            if denominator > 0.0 and measured_range > 0.0 else None
        )
        jam_ratio_est = (interference_w / noise_w) if noise_w > 0 else 0.0

        return MeasurementRecord(
            range_m=measured_range,
            azimuth_deg=normalize_angle_deg(measured_az),
            elevation_deg=measured_el,
            range_rate_mps=measured_vr,
            std_range_m=std_range,
            std_az_deg=cfg.az_sigma_deg,
            std_el_deg=cfg.el_sigma_deg,
            std_range_rate_mps=cfg.range_rate_sigma_mps,
            covariance=diagonal_covariance(std_range, cfg.az_sigma_deg, cfg.el_sigma_deg),
            snr_db=snr_measured_db,
            rcs_est_m2=rcs_est,
            jam_ratio_est=jam_ratio_est,
        )

def noise_from_echo_and_snr(echo_w: float, snr_db_value: float) -> float:
    """由回波功率与 SNR 反解噪声功率（`snr_db` 的逆）。

    单独抽出来是为了避免在 `RadarSensor` 里重复写 `10**(snr/10)`——
    工程里凡是重复出现的公式都应该是唯一实现。
    """
    if snr_db_value == float("-inf"):
        return 0.0
    return echo_w / db2lin(snr_db_value)


# ----------------------------------------------------------------------
# 被动 ESM 传感器
# ----------------------------------------------------------------------


class EsmSensor(Sensor):
    """被动侦察传感器：**方位/俯仰两维，没有距离量测**。

    为什么不做距离：单站被动测距在物理上做不到（只能测到方向），
    需要多站时差/相位差或平台机动才能解算距离。
    本工程只有单站 ESM，因此距离维标准差为 `inf`，
    融合层必须能正确处理"有些测量没有距离"的情况。
    """

    def _detection_probability(self, distance: float, entity: Any) -> float:
        """被动截获概率：单程链路 + ESM 接收机噪声。

        复用现有单程链路方程与 ROC，不改动物理公式。
        注意这里观测的是**雷达辐射**，因此距离是"雷达到 ESM"的单程距离，
        且增益取雷达在该方向上的发射增益（由调用方通过 `beam_gain_db` 传入，
        因为它取决于雷达波束指向，属于雷达侧信息）。
        """
        from engine.equations import one_way_power_w

        cfg = self.config
        tx_gain_db = float(self._context.get("beam_gain_db", cfg.peak_gain_db))
        tx_power_w = float(self._context.get("emitter_power_w", 0.0))
        if tx_power_w <= 0.0:
            return 0.0
        received_w = one_way_power_w(
            pt_w=tx_power_w,
            tx_gain_db=tx_gain_db,
            rx_gain_db=cfg.peak_gain_db,
            wavelength_m=cfg.wavelength_m,
            range_m=max(distance, 1e-6),
            system_loss_db=cfg.system_loss_db,
        )
        noise_w = thermal_noise_w(cfg.bandwidth_hz, cfg.noise_figure_db, cfg.temperature_k)
        if noise_w <= 0.0:
            return 1.0
        return logistic_prob(lin2db(received_w / noise_w), cfg.snr50_db, cfg.pd_slope_db)

    def observe(self, scene: Any, time_s: float,
                occlusion: Optional[OcclusionModel] = None,
                extra_context: Optional[Dict[str, Any]] = None) -> SensorReport:
        self._context = dict(extra_context or {})
        return super().observe(scene, time_s, occlusion, extra_context)

    def _measure(
        self, sensor_pose: Pose, entity_pose: Pose, true_range: float,
        true_az: float, true_el: float, true_vr: float, entity: Any,
        detection_prob: float,
    ) -> MeasurementRecord:
        cfg = self.config
        rng = self._rng
        measured_az = true_az + (rng.gauss(0.0, cfg.az_sigma_deg)
                                 if cfg.az_sigma_deg > 0 else 0.0)
        measured_el = true_el + (rng.gauss(0.0, cfg.el_sigma_deg)
                                 if cfg.el_sigma_deg > 0 else 0.0)
        # 被动传感器：不给距离，也不给径向速度
        return MeasurementRecord(
            range_m=None,
            azimuth_deg=normalize_angle_deg(measured_az),
            elevation_deg=measured_el,
            range_rate_mps=None,
            std_range_m=None,
            std_az_deg=cfg.az_sigma_deg,
            std_el_deg=cfg.el_sigma_deg,
            std_range_rate_mps=None,
            covariance=infinite_range_covariance(cfg.az_sigma_deg, cfg.el_sigma_deg),
            snr_db=None,
        )


# ----------------------------------------------------------------------
# 传感器套件
# ----------------------------------------------------------------------


class SensorSuite:
    """一组传感器；统一观测并汇总报告。"""

    def __init__(
        self,
        sensors: Sequence[Sensor],
        occlusion: Optional[OcclusionModel] = None,
    ) -> None:
        self.sensors: List[Sensor] = list(sensors)
        if occlusion is None:
            from sensor.occlusion import OcclusionModel as _OcclusionModel

            occlusion = _OcclusionModel()
        self.occlusion = occlusion
        seen = set()
        for sensor in self.sensors:
            if sensor.sensor_id in seen:
                raise ValueError(f"传感器 ID 重复：{sensor.sensor_id}")
            seen.add(sensor.sensor_id)

    def reset(self) -> None:
        for sensor in self.sensors:
            sensor.reset()

    def observe(
        self, scene: Any, time_s: float,
        context_by_sensor: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SuiteReport:
        """对场景做一次全传感器观测（**只读**）。"""
        suite = SuiteReport(time_s=time_s)
        for sensor in self.sensors:
            context = (context_by_sensor or {}).get(sensor.sensor_id, {})
            suite.reports.append(
                sensor.observe(scene, time_s, self.occlusion, context)
            )
        return suite

    def by_id(self, sensor_id: str) -> Sensor:
        for sensor in self.sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        raise KeyError(f"没有传感器 {sensor_id!r}")

    def set_available(self, sensor_id: str, available: bool) -> None:
        self.by_id(sensor_id).config.available = bool(available)

    def describe(self) -> str:
        lines = [f"传感器套件：{len(self.sensors)} 个传感器，"
                 f"{len(self.occlusion)} 个遮挡体"]
        lines.extend("  " + s.describe() for s in self.sensors)
        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.sensors]
