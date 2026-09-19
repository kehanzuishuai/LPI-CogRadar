"""传感器测量记录与「没有数据」的显式分类（v4.2 分层测量升级）。

为什么必须把「没有数据」拆成多种原因
------------------------------------
v4.0 的 POMDP 层用一个 `dropout_prob` 代表全部观测缺失。这在工程上是**错的**，
因为不同的缺失原因对应完全不同的对策与完全不同的物理含义：

* **目标不在视场**：传感器根本不知道那边有东西 —— 这是**几何/指向**问题，
  对策是调整扫描/指向；而且它对"目标是否存在"不提供任何信息。
* **超出作用距离**：目标在视场里，但回波低于检测门限 —— 同样是"看不见"，
  但**增大发射功率或降低门限**可能救回来，与视场问题对策完全不同。
* **被遮挡**：目标在视场、在距离内，但视线被障碍物切断 ——
  这是**环境**问题，任何功率都救不回来（除非改变位置）。
* **尚未到更新时刻**：传感器**周期**问题 —— 上一次测量仍然有效，
  只是"新"，属于**时间**维度，与前三种（空间）正交。
* **检测遗漏 / 丢测**：几何、距离、遮挡全部通过，但这一帧刚好没检测到
  （回波起伏、虚警门限、处理损耗）—— 这是**概率**问题，下一帧可能就有了。

把这五种混成一个 `dropout_prob`，会导致：
1. 智能体无法区分"那里真的没有目标"与"我这一帧运气不好"；
2. 消融实验失去意义（关掉 dropout 到底关掉了什么？）；
3. 回退/保守策略无法针对性设计（该转雷达还是该提功率？）。

因此本模块用 `NoDataReason` 把它们**逐项分开**，并且每种都进入统计与导出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# 「没有数据」的原因
# ----------------------------------------------------------------------


class NoDataReason(str, Enum):
    """为什么这一步没有可用的测量。

    取值刻意用字符串枚举，便于直接写进 CSV/JSON 而无需二次映射。
    """

    #: 有数据（不是缺失）
    NONE = "none"

    #: 目标不在传感器视场内（方位或俯仰超出视场）—— 几何/指向问题
    OUT_OF_FOV = "out_of_fov"

    #: 超出传感器作用距离 —— 能量问题，理论上可被更高功率/更低门限救回
    BEYOND_RANGE = "beyond_range"

    #: 视线被障碍物/遮挡区切断 —— 环境问题，功率无用
    OCCLUDED = "occluded"

    #: 当前时刻尚未到该传感器的更新时刻 —— 时间问题（上一次测量仍然有效）
    NOT_UPDATED = "not_updated"

    #: 通过了全部几何与距离检查，但本帧检测概率未命中 —— 概率问题
    MISSED_DETECTION = "missed_detection"

    #: 传感器本身不可用（关机 / 故障 / 被指令禁用）
    SENSOR_UNAVAILABLE = "sensor_unavailable"


#: 原因 -> 中文说明（AI 诊断层与报告直接引用，保证措辞一致）
REASON_CN: Dict[str, str] = {
    NoDataReason.NONE.value: "有可用测量",
    NoDataReason.OUT_OF_FOV.value: "目标不在传感器视场内",
    NoDataReason.BEYOND_RANGE.value: "超出传感器作用距离",
    NoDataReason.OCCLUDED.value: "视线被遮挡",
    NoDataReason.NOT_UPDATED.value: "尚未到该传感器的更新时刻",
    NoDataReason.MISSED_DETECTION.value: "本帧检测遗漏（概率性丢测）",
    NoDataReason.SENSOR_UNAVAILABLE.value: "传感器不可用",
}

#: 原因 -> 所属维度（空间 / 时间 / 概率 / 可用性），用于统计与消融分组
REASON_DIMENSION: Dict[str, str] = {
    NoDataReason.NONE.value: "ok",
    NoDataReason.OUT_OF_FOV.value: "空间-指向",
    NoDataReason.BEYOND_RANGE.value: "空间-能量",
    NoDataReason.OCCLUDED.value: "空间-环境",
    NoDataReason.NOT_UPDATED.value: "时间",
    NoDataReason.MISSED_DETECTION.value: "概率",
    NoDataReason.SENSOR_UNAVAILABLE.value: "可用性",
}


#: 逐缺失原因的**信息来源**（v4.5 信息边界）。
#:
#: 这三类的可信度完全不同，混在一起就会把"推测"当成"事实"：
#:
#: * `device_known`：设备自己就能确定——传感器下线、本帧未到更新时刻；
#: * `measurement_inferred`：由**自身状态 + 几何**推断，可能与事实不符
#:   （例如远处确实没有目标，而不是"目标超距离"）；
#: * `evaluation_only`：**只有仿真器知道**——"漏检"的定义是"本来有个目标
#:   但没探到"，而没有真值目标清单就无从区分"漏检"与"这里什么都没有"。
#:
#: ⚠️ 在线快照只能以**确定值**上报前两类，且必须标明推断属性；
#: 第三类只能进离线评测通道。
REASON_PROVENANCE: Dict[str, str] = {
    NoDataReason.SENSOR_UNAVAILABLE.value: "device_known",
    NoDataReason.NOT_UPDATED.value: "device_known",
    NoDataReason.BEYOND_RANGE.value: "measurement_inferred",
    NoDataReason.OUT_OF_FOV.value: "measurement_inferred",
    NoDataReason.OCCLUDED.value: "measurement_inferred",
    NoDataReason.MISSED_DETECTION.value: "evaluation_only",
}

#: 全部「没有数据」的原因（不含 NONE），即需要被逐项统计的集合
NO_DATA_REASONS: Tuple[str, ...] = (
    NoDataReason.SENSOR_UNAVAILABLE.value,
    NoDataReason.BEYOND_RANGE.value,
    NoDataReason.OUT_OF_FOV.value,
    NoDataReason.OCCLUDED.value,
    NoDataReason.NOT_UPDATED.value,
    NoDataReason.MISSED_DETECTION.value,
)


# ----------------------------------------------------------------------
# 测量记录
# ----------------------------------------------------------------------


@dataclass
class MeasurementRecord:
    """一条**传感器测量**。

    ⚠️ 这是本工程里"传感器看到的东西"，与"世界真值"是两个不同的东西，
    必须严格区分（见 README §11C 的三层数据字典）：

    * **仿真真值**：`Scene` 里实体的真实位置/速度/姿态，只有仿真器与**评测**可以读；
    * **传感器测量**（本类）：带噪声、带时间戳、带协方差、可能根本不存在；
    * **算法可见输入**：由若干测量**汇聚**成的定长观测向量（`sensor/fusion.py`）。

    字段设计要点
    ------------
    * `candidate_id` 是传感器**自己的候选编号**（如 `RADAR1-C2`），
      跨时间稳定，使智能体能对同一候选做时序推理。它**不是真值 ID**。
    * `truth_id` / `truth_*` 字段**只用于评测**（算误差、算漏检率），
      **绝不允许进入算法可见的观测向量**。`sensor/fusion.py` 在打包时会
      断言丢弃这些字段，并有单元测试钉住。
    * `covariance` 是 (距离, 方位, 俯仰) 三者的 3×3 协方差（对角阵，
      因为三者的误差源相互独立）。被动传感器（ESM）没有距离量测，
      距离行/列填 `inf`。
    """

    # --- 标识与时间 ---
    # 这三个字段由 `Sensor.observe()` 在测量产生后统一回填，
    # 因此这里给默认值，让子类的 `_measure()` 只需关心"测量值本身"。
    sensor_id: str = ""
    candidate_id: str = ""
    time_s: float = 0.0
    sensor_kind: str = "radar"  # radar | esm

    # --- 测量量（极坐标：距离 + 角度，是雷达/ESM 的原生量测）---
    range_m: Optional[float] = None
    azimuth_deg: Optional[float] = None  # 绝对方位（自正北顺时针）
    elevation_deg: Optional[float] = None
    range_rate_mps: Optional[float] = None  # 径向速度（多普勒）

    # --- 不确定度 ---
    std_range_m: Optional[float] = None
    std_az_deg: Optional[float] = None
    std_el_deg: Optional[float] = None
    std_range_rate_mps: Optional[float] = None
    covariance: Optional[List[List[float]]] = None  # (range, az, el) 的 3×3

    # --- 置信度与来源 ---
    confidence: float = 0.0  # 检测置信度 0~1
    snr_db: Optional[float] = None
    is_false_alarm: bool = False
    #: 该测量是否为本步**新产生**（False = 沿用上一次，`age_s` > 0）
    is_fresh: bool = True
    age_s: float = 0.0

    # --- 派生的估计量（传感器从原始量测反推）---
    #: 雷达可由 SNR 与距离反推目标 RCS（真实雷达确实这么做）
    rcs_est_m2: Optional[float] = None
    #: 雷达可测噪声基底，从而估计 J/N
    jam_ratio_est: Optional[float] = None

    # --- 仅评测用的真值（**禁止进入算法输入**）---
    truth_id: Optional[str] = None
    truth_range_m: Optional[float] = None
    truth_azimuth_deg: Optional[float] = None
    truth_elevation_deg: Optional[float] = None
    truth_range_rate_mps: Optional[float] = None

    # ------------------------------------------------------------------

    @property
    def has_range(self) -> bool:
        return self.range_m is not None

    def range_error_m(self) -> Optional[float]:
        """距离测量误差（评测专用）。"""
        if self.range_m is None or self.truth_range_m is None:
            return None
        return self.range_m - self.truth_range_m

    def azimuth_error_deg(self) -> Optional[float]:
        if self.azimuth_deg is None or self.truth_azimuth_deg is None:
            return None
        delta = self.azimuth_deg - self.truth_azimuth_deg
        # 角度误差要跨 ±180° 归一化，否则 179° 与 -179° 会被算成 358° 的大误差
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    def range_rate_error_mps(self) -> Optional[float]:
        if self.range_rate_mps is None or self.truth_range_rate_mps is None:
            return None
        return self.range_rate_mps - self.truth_range_rate_mps

    def to_dict(self, include_truth: bool = False) -> Dict[str, Any]:
        """导出为字典。

        `include_truth=False`（默认）时**不含任何真值字段、也不含虚警标记**——
        这样默认导出的测量记录可以直接放进算法调试链路与通信链路而不会泄题。
        评测脚本显式传 `include_truth=True`。

        ⚠️ `is_false_alarm` 属于**评测标记**，不是测量量：真实的接收机
        不可能在自己的测量报告里标注"这条是虚警"。它早期被无条件写进导出结果，
        等于给消费者一个 oracle 提示（"这条别信"）。现在它只在
        `include_truth=True` 时出现。
        （这个缺陷是被 `communication/message.py` 的**字段白名单**抓出来的——
        早期单元测试只查了 `truth*` 前缀，漏掉了它。）
        """
        payload: Dict[str, Any] = {
            "sensor_id": self.sensor_id,
            "candidate_id": self.candidate_id,
            "sensor_kind": self.sensor_kind,
            "time_s": self.time_s,
            "is_fresh": self.is_fresh,
            "age_s": self.age_s,
            "range_m": self.range_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "range_rate_mps": self.range_rate_mps,
            "std_range_m": self.std_range_m,
            "std_az_deg": self.std_az_deg,
            "std_el_deg": self.std_el_deg,
            "std_range_rate_mps": self.std_range_rate_mps,
            "confidence": self.confidence,
            "snr_db": self.snr_db,
            "rcs_est_m2": self.rcs_est_m2,
            "jam_ratio_est": self.jam_ratio_est,
        }
        if self.covariance is not None:
            payload["cov_xx"] = self.covariance[0][0]
            payload["cov_yy"] = self.covariance[1][1]
            payload["cov_zz"] = self.covariance[2][2]
        if include_truth:
            payload.update({
                "is_false_alarm": self.is_false_alarm,
                "truth_id": self.truth_id,
                "truth_range_m": self.truth_range_m,
                "truth_azimuth_deg": self.truth_azimuth_deg,
                "truth_elevation_deg": self.truth_elevation_deg,
                "truth_range_rate_mps": self.truth_range_rate_mps,
                "err_range_m": self.range_error_m(),
                "err_azimuth_deg": self.azimuth_error_deg(),
                "err_range_rate_mps": self.range_rate_error_mps(),
            })
        return payload


# ----------------------------------------------------------------------
# 逐 (传感器, 目标) 的结果
# ----------------------------------------------------------------------


@dataclass
class TargetOutcome:
    """某个传感器对某个**真实实体**（或虚警）在某一时刻的结果。

    不论有没有数据，都产生一条 `TargetOutcome`——这正是"五种缺失原因
    必须可区分"的落地方式：算法侧只看到有没有 `record`，
    而统计与诊断侧能看到 `reason` 到底是什么。
    """

    sensor_id: str
    time_s: float
    status: str  # "detected" | "no_data"
    reason: str = NoDataReason.NONE.value

    #: 传感器自己的候选编号（无数据时为空串）
    candidate_id: str = ""
    record: Optional[MeasurementRecord] = None

    #: 仅评测用：对应的真值实体 ID
    truth_id: Optional[str] = None

    #: 该实体到传感器的真实距离（评测用；无数据时也能填，便于分析"多远开始看不见"）
    truth_range_m: Optional[float] = None

    @property
    def detected(self) -> bool:
        return self.status == "detected"

    @property
    def reason_cn(self) -> str:
        return REASON_CN.get(self.reason, self.reason)

    @property
    def reason_dimension(self) -> str:
        return REASON_DIMENSION.get(self.reason, "")

    def to_dict(self, include_truth: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "sensor_id": self.sensor_id,
            "time_s": self.time_s,
            "status": self.status,
            "reason": self.reason,
            "reason_cn": self.reason_cn,
            "reason_dimension": self.reason_dimension,
            "candidate_id": self.candidate_id,
            "detected": self.detected,
        }
        if self.record is not None:
            payload.update({
                "range_m": self.record.range_m,
                "azimuth_deg": self.record.azimuth_deg,
                "elevation_deg": self.record.elevation_deg,
                "range_rate_mps": self.record.range_rate_mps,
                "std_range_m": self.record.std_range_m,
                "confidence": self.record.confidence,
                "is_false_alarm": self.record.is_false_alarm,
                "is_fresh": self.record.is_fresh,
                "age_s": self.record.age_s,
            })
        if include_truth:
            payload["truth_id"] = self.truth_id
            payload["truth_range_m"] = self.truth_range_m
        return payload


@dataclass
class SensorReport:
    """单个传感器在某一时刻的完整报告。

    三类测量要分清（这是"延迟/未更新"语义的正确落地方式）：

    * `detections`：本帧**新产生**的检测（`is_fresh=True`, `age_s=0`）；
    * `held`：本帧**未到更新时刻**时沿用的上一次测量（`is_fresh=False`，
      `age_s = 当前时刻 − 原测量时刻`）。它们来自传感器**自己的候选登记表**，
      因此不需要任何真值即可生成——这点很重要，否则"沿用旧值"这条路径
      会变成把真值喂进算法链路的漏洞；
    * `false_alarms`：本帧虚警。

    而 `outcomes` 是**逐实体**的判定结果（含全部「没有数据」的原因），
    它带 `truth_id`，属于**评测通道**：用于统计"某个真值目标为什么没被看到"，
    **不进入**算法输入。两者刻意分开存放。

    为什么不在"更新帧"沿用旧测量（不做航迹外推）：一旦扫描到了，就以本帧为准；
    本帧没检测到就是没数据，不拿旧值顶替。这样**不会产生幻影航迹**，
    代价是失去"目标短暂丢失仍能推算位置"的能力——这是刻意的保守取舍，
    见 README §11C 的局限说明。
    """

    sensor_id: str
    sensor_kind: str
    time_s: float
    updated: bool  # 本步是否到了更新时刻
    outcomes: List[TargetOutcome] = field(default_factory=list)
    detections: List[MeasurementRecord] = field(default_factory=list)
    held: List[MeasurementRecord] = field(default_factory=list)
    false_alarms: List[MeasurementRecord] = field(default_factory=list)

    @property
    def measurements(self) -> List[MeasurementRecord]:
        """本报告里**可供算法使用**的测量（新检测 + 沿用值 + 虚警）。

        注意三点：
        1. 虚警也在这里——算法无从分辨，这正是"更真实观测"的一部分；
        2. 不含任何真值字段（`fusion` 打包时还会再断言一次）；
        3. `outcomes` 里的 `truth_id` **不在**这里。
        """
        out = list(self.detections)
        out.extend(self.held)
        out.extend(self.false_alarms)
        return out

    @property
    def fresh_measurements(self) -> List[MeasurementRecord]:
        return [m for m in self.measurements if m.is_fresh]

    def reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
        return counts

    def to_dict(self, include_truth: bool = False) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_kind": self.sensor_kind,
            "time_s": self.time_s,
            "updated": self.updated,
            "n_detections": len(self.detections),
            "n_held": len(self.held),
            "n_false_alarms": len(self.false_alarms),
            "reason_counts": self.reason_counts(),
            "outcomes": [o.to_dict(include_truth=include_truth) for o in self.outcomes],
            "measurements": [m.to_dict(include_truth=include_truth)
                             for m in self.measurements],
        }


@dataclass
class SuiteReport:
    """全部传感器在某一时刻的报告汇总。"""

    time_s: float
    reports: List[SensorReport] = field(default_factory=list)

    # --- 跨传感器展平的便捷访问器 ---
    # 调用方（融合层、统计、测试）通常只关心"这一步总共看到/没看到什么"，
    # 不希望自己写三层嵌套循环。这些属性把 `reports` 展平，
    # 但**不改变**语义：仍然逐传感器保留 `reports` 供诊断。

    @property
    def outcomes(self) -> List[TargetOutcome]:
        out: List[TargetOutcome] = []
        for report in self.reports:
            out.extend(report.outcomes)
        return out

    @property
    def detections(self) -> List[MeasurementRecord]:
        out: List[MeasurementRecord] = []
        for report in self.reports:
            out.extend(report.detections)
        return out

    @property
    def held(self) -> List[MeasurementRecord]:
        out: List[MeasurementRecord] = []
        for report in self.reports:
            out.extend(report.held)
        return out

    @property
    def false_alarms(self) -> List[MeasurementRecord]:
        out: List[MeasurementRecord] = []
        for report in self.reports:
            out.extend(report.false_alarms)
        return out

    @property
    def measurements(self) -> List[MeasurementRecord]:
        out: List[MeasurementRecord] = []
        for report in self.reports:
            out.extend(report.measurements)
        return out

    @property
    def fresh_measurements(self) -> List[MeasurementRecord]:
        return [m for m in self.measurements if m.is_fresh]

    @property
    def updated_sensors(self) -> List[str]:
        return [r.sensor_id for r in self.reports if r.updated]

    @property
    def updated(self) -> bool:
        """本步是否有**任一**传感器到了更新时刻。

        注意语义：这是"整个套件"的判据（any），不是单个传感器的。
        要判断某个具体传感器，请用 `sensor_report(sid).updated`——
        两者在多传感器套件里完全不同（例如雷达每秒扫、ESM 每 2 秒扫，
        在奇数秒 `updated` 为 True 而 ESM 的 `updated` 为 False）。
        """
        return any(r.updated for r in self.reports)

    def sensor_report(self, sensor_id: str) -> Optional[SensorReport]:
        for report in self.reports:
            if report.sensor_id == sensor_id:
                return report
        return None

    def reason_counts(self) -> Dict[str, int]:
        by_reason: Dict[str, int] = {}
        for outcome in self.outcomes:
            by_reason[outcome.reason] = by_reason.get(outcome.reason, 0) + 1
        return by_reason

    def reason_counts_by_dimension(self) -> Dict[str, int]:
        by_dimension: Dict[str, int] = {}
        for outcome in self.outcomes:
            dimension = outcome.reason_dimension
            by_dimension[dimension] = by_dimension.get(dimension, 0) + 1
        return by_dimension

    def to_dict(self, include_truth: bool = False) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "n_sensors": len(self.reports),
            "updated_sensors": self.updated_sensors,
            "n_measurements": len(self.measurements),
            "n_fresh": len(self.fresh_measurements),
            "reason_counts": self.reason_counts(),
            "reason_by_dimension": self.reason_counts_by_dimension(),
            "reports": [r.to_dict(include_truth=include_truth) for r in self.reports],
        }


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------


def diagonal_covariance(sr: float, sa_deg: float, se_deg: float) -> List[List[float]]:
    """由距离/方位/俯仰标准差构造 3×3 对角协方差。

    为什么是对角阵：三个量的误差源相互独立（距离来自时延估计，
    角度来自波束/相位差估计）。真实系统里方位-俯仰可能有耦合
    （单脉冲测角的交叉耦合），但本阶段不建模，**如实说明**而不是假装有。
    """
    return [
        [float(sr) ** 2, 0.0, 0.0],
        [0.0, float(sa_deg) ** 2, 0.0],
        [0.0, 0.0, float(se_deg) ** 2],
    ]


def infinite_range_covariance(sa_deg: float, se_deg: float) -> List[List[float]]:
    """被动传感器（无距离量测）的协方差：距离维填 inf。"""
    return [
        [math.inf, 0.0, 0.0],
        [0.0, float(sa_deg) ** 2, 0.0],
        [0.0, 0.0, float(se_deg) ** 2],
    ]
