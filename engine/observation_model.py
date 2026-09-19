"""部分可观测（POMDP）测量模型。

为什么单独建一个模块
--------------------
第三/四版的环境给智能体的是**真值**：目标距离、干扰强度、剩余能量、ESM 位置、
Pint、累计暴露都是精确已知的。真实雷达系统里这些量都要靠测量与估计得到，
必然带噪声、延迟甚至丢测。本模块把「真值 -> 可观测量」这一步独立出来，
好处有三：

1. 观测模型与仿真内核解耦——仿真照旧演进真值，只有观测被污染；
2. 可以**逐项**配置噪声（距离/干扰/能量/RCS/暴露/侦察机方位），
   并支持延迟与丢测，便于做消融实验；
3. `enabled=False` 时精确退化为全可观（full_observable），旧实验逐位可复现。

关键设计：给智能体的不是「一个数」，而是「估计值 + 不确定度 + 是否新鲜」
--------------------------------------------------------------
每个观测量都被包成 `ObservationEstimate(value, sigma, observed, stale, confident)`：

* `sigma` 是估计标准差（噪声模型自己知道，可以如实上报，也可以故意低估——
  由配置决定，用来研究"智能体高估自己精度"这种真实失效）；
* `observed=False` 表示本步丢测，`stale=True` 表示仍在沿用旧值（保持上次测量）；
* 智能体拿到的是估计值与置信区间，**真实值、真实 ESM 位置、真实 Pint
  与真实暴露在 POMDP 模式下不再进入观测向量**。

本模块只在 env 层被调用，**不改变任何物理参数与真值演进**。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

@dataclass
class ObservationNoiseConfig:
    """测量噪声 / 延迟 / 丢测配置。

    所有 sigma 都是 1-sigma 标准差；`enabled=False` 时全部噪声关闭，
    行为与全可观模式一致。
    """

    enabled: bool = False

    # --- 逐项测量噪声（1 sigma）---
    range_sigma_m: float = 120.0  # 目标距离
    rcs_sigma_m2: float = 0.12  # 目标 RCS
    jam_ratio_sigma: float = 0.10  # 干扰 J/N
    energy_sigma_j: float = 15.0  # 剩余能量
    interceptor_range_sigma_m: float = 4000.0  # 侦察机距离（真值本就不可知）
    pint_sigma: float = 0.05  # 截获概率估计
    exposure_sigma: float = 0.05  # 累计暴露估计
    pd_sigma: float = 0.03  # 探测概率估计

    # --- 延迟与丢测 ---
    delay_steps: int = 0  # 观测延迟（步）
    dropout_prob: float = 0.0  # 每一步每个量独立丢测概率
    hold_last_on_dropout: bool = True  # 丢测时沿用上一次测量（否则用先验均值）

    # --- 隐藏真值（POMDP 的核心）---
    hide_interceptor_truth: bool = True  # 侦察机真实位置不可直接观测
    hide_pint_truth: bool = True  # 真实 Pint 不可直接观测
    hide_exposure_truth: bool = True  # 真实累计暴露不可直接观测

    # --- 置信度上报策略 ---
    report_honest_sigma: bool = True  # False = 低报自身不确定度（研究过度自信）
    sigma_underreport_factor: float = 0.5  # report_honest_sigma=False 时乘的系数

    seed: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ObservationNoiseConfig":
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> None:
        for name in (
            "range_sigma_m", "rcs_sigma_m2", "jam_ratio_sigma", "energy_sigma_j",
            "interceptor_range_sigma_m", "pint_sigma", "exposure_sigma", "pd_sigma",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"观测噪声 {name} 不能为负")
        if self.delay_steps < 0:
            raise ValueError("delay_steps 不能为负")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob 必须落在 [0, 1) 内")
        if not 0.0 < self.sigma_underreport_factor <= 1.0:
            raise ValueError("sigma_underreport_factor 必须落在 (0, 1] 内")


# ----------------------------------------------------------------------
# 单条观测量
# ----------------------------------------------------------------------

@dataclass
class ObservationEstimate:
    """一条带不确定度的观测量。"""

    name: str
    value: float
    sigma: float = 0.0
    observed: bool = True  # 本步是否真的测到了
    stale: bool = False  # 是否在沿用旧测量（延迟/丢测）
    truth: Optional[float] = None  # 仅用于诊断统计，**不会**进入观测向量

    @property
    def reported_sigma(self) -> float:
        return float(self.sigma)

    @property
    def confident(self) -> bool:
        return self.observed and not self.stale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "sigma": round(self.sigma, 6),
            "observed": self.observed,
            "stale": self.stale,
        }


# ----------------------------------------------------------------------
# 观测模型
# ----------------------------------------------------------------------

class ObservationModel:
    """把仿真真值转换成带噪声、可能延迟/丢测的观测量集合。"""

    #: 需要逐项测量噪声的量（名称 -> 配置字段）
    NOISY_FIELDS: Tuple[Tuple[str, str], ...] = (
        ("target_range", "range_sigma_m"),
        ("target_rcs", "rcs_sigma_m2"),
        ("jam_ratio", "jam_ratio_sigma"),
        ("remaining_energy", "energy_sigma_j"),
        ("interceptor_range", "interceptor_range_sigma_m"),
        ("pint_eff", "pint_sigma"),
        ("exposure", "exposure_sigma"),
        ("pd_min", "pd_sigma"),
    )

    def __init__(self, config: Optional[ObservationNoiseConfig] = None) -> None:
        self.config = config or ObservationNoiseConfig()
        self.config.validate()
        self._rng = random.Random(self.config.seed)
        # 延迟缓冲区：存最近若干步的「干净测量」
        self._delay_buffer: List[Dict[str, float]] = []
        # 丢测时保持的上次测量
        self._last_measurement: Dict[str, float] = {}
        self._last_step_index: int = -1
        self.stats: Dict[str, Any] = {
            "steps": 0,
            "dropped": 0,
            "dropout_slots": 0,
            "total_slots": 0,
            "delayed": 0,
        }

    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        base = int(seed) if seed is not None else int(self.config.seed)
        self._base_seed = base
        self._rng = random.Random(base)
        #: 逐量独立的随机流（懒创建）。见 `_field_rng` 的说明。
        self._field_rngs = {}
        self._delay_buffer.clear()
        self._last_measurement.clear()
        self._last_step_index = -1
        for key in self.stats:
            self.stats[key] = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _field_rng(self, stream: str) -> random.Random:
        """取得某个「随机流」的独立 RNG。

        为什么**必须**逐量独立
        --------------------
        初版所有观测量共用一条 `self._rng`。这在消融实验里会出问题：
        把某一项噪声关掉（sigma=0）之后，它就不再消耗随机数，
        于是**后面所有量**抽到的噪声序列整体错位——
        「只关距离噪声」得到的差异里，混着「干扰/能量/暴露噪声全部换了一组实现」
        的影响。这样测出来的消融效应无法归因到被关掉的那一路。

        改成按 `(种子, 量名)` 派生独立流之后，任何一路的开关都不再影响其他路，
        消融才真正是「单变量」的。用字符串种子（`random.Random("42:target_range")`）
        而不是 `hash()`，因为后者对 str 是按进程随机化的，无法跨运行复现。
        """
        rng = self._field_rngs.get(stream)
        if rng is None:
            rng = random.Random("%d:%s" % (getattr(self, "_base_seed", 0), stream))
            self._field_rngs[stream] = rng
        return rng

    # ------------------------------------------------------------------

    #: hide_* 开关 -> 对应观测量名（False 表示该量被设为"可直接观测真值"）
    HIDE_FLAGS: Tuple[Tuple[str, str], ...] = (
        ("interceptor_range", "hide_interceptor_truth"),
        ("pint_eff", "hide_pint_truth"),
        ("exposure", "hide_exposure_truth"),
    )

    def _directly_observable(self, name: str) -> bool:
        """该量是否被配置为"直接可观测真值"（跳过噪声/延迟/丢测）。"""
        for field_name, flag_name in self.HIDE_FLAGS:
            if field_name == name and not bool(getattr(self.config, flag_name)):
                return True
        return False

    def _sample_noise(self, field_name: str, stream: str = "") -> float:
        """按配置字段抽一个零均值高斯噪声。

        `stream` 指定独立随机流（通常传观测量名），使各路噪声互不干扰。
        """
        sigma = float(getattr(self.config, field_name))
        if sigma <= 0.0:
            return 0.0
        return self._field_rng(stream or field_name).gauss(0.0, sigma)

    def _clean_measurement(self, truth: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """对真值做一次「无延迟、无丢测」的测量，返回 {名称: (测量值, sigma)}。"""
        result: Dict[str, Tuple[float, float]] = {}
        for name, sigma_field in self.NOISY_FIELDS:
            if name not in truth:
                continue
            if self._directly_observable(name):
                # 消融对照：该量不做任何隐藏，直接等于真值
                result[name] = (float(truth[name]), 0.0)
                continue
            sigma = float(getattr(self.config, sigma_field))
            measured = self._apply_bounds(
                name, float(truth[name]) + self._sample_noise(sigma_field, name)
            )
            reported_sigma = (
                sigma
                if self.config.report_honest_sigma
                else sigma * self.config.sigma_underreport_factor
            )
            result[name] = (measured, reported_sigma)
        return result

    def observe(
        self, truth: Dict[str, float], step_index: int
    ) -> Dict[str, ObservationEstimate]:
        """给出一组带不确定度的观测量。

        参数
        ----
        truth     : {量名: 真值}，由调用方（env）从仿真里读出
        step_index: 当前步号，用于延迟缓冲

        返回 {量名: ObservationEstimate}
        """
        cfg = self.config

        # --- 全可观模式：直接把真值当成零不确定度的观测 ---
        if not cfg.enabled:
            out: Dict[str, ObservationEstimate] = {}
            for name in truth:
                out[name] = ObservationEstimate(
                    name=name, value=float(truth[name]), sigma=0.0,
                    observed=True, stale=False, truth=float(truth[name]),
                )
            return out

        self.stats["steps"] += 1
        self._last_step_index = step_index

        # --- 延迟：把本步的干净测量压入缓冲，取 delay_steps 之前的那一份 ---
        clean = self._clean_measurement(truth)
        self._delay_buffer.append({k: v[0] for k, v in clean.items()})
        sigma_map = {k: v[1] for k, v in clean.items()}
        if cfg.delay_steps > 0:
            self.stats["delayed"] += 1
            index = max(0, len(self._delay_buffer) - 1 - cfg.delay_steps)
            delayed = self._delay_buffer[index]
        else:
            delayed = self._delay_buffer[-1]
        # 缓冲只保留必要长度
        max_len = cfg.delay_steps + 2
        if len(self._delay_buffer) > max_len:
            self._delay_buffer = self._delay_buffer[-max_len:]

        # --- 丢测：逐量独立伯努利 ---
        out = {}
        for name in truth:
            # 消融对照：被配置为"直接可观测"的量跳过延迟与丢测
            if self._directly_observable(name):
                out[name] = ObservationEstimate(
                    name=name, value=float(truth[name]), sigma=0.0,
                    observed=True, stale=False, truth=float(truth[name]),
                )
                continue

            self.stats["total_slots"] += 1
            measured_value = delayed.get(name, float(truth[name]))
            reported_sigma = sigma_map.get(name, 0.0)

            dropped = (
                cfg.dropout_prob > 0.0
                and self._field_rng(name + "/dropout").random() < cfg.dropout_prob
            )
            if dropped:
                self.stats["dropout_slots"] += 1
                if cfg.hold_last_on_dropout and name in self._last_measurement:
                    value = self._last_measurement[name]
                    stale = True
                else:
                    # 丢测且**没有任何历史测量**（只可能发生在第 0 步）：
                    # 这里**绝不能**退化成"用真值占位"。早先版本就是这么写的，
                    # 等于在观测里开了一个真值后门——策略只要读 value 就能拿到真值，
                    # 部分可观测的前提当场失效。
                    # 正确做法：用本步本来要给出的测量值（已带噪、已裁剪），
                    # 但如实标记 observed=False / stale=True，让它被当作不可信值处理。
                    value = self._apply_bounds(name, float(measured_value))
                    stale = True
                observed = False
            else:
                value = measured_value
                stale = cfg.delay_steps > 0
                observed = True
                self._last_measurement[name] = value

            if dropped:
                # 丢测时不确定度应放大（残留在旧值上，真实误差更大）
                reported_sigma = reported_sigma * 2.0 + 1e-6

            out[name] = ObservationEstimate(
                name=name,
                value=float(value),
                sigma=float(reported_sigma),
                observed=observed,
                stale=stale,
                truth=float(truth[name]) if name in truth else None,
            )

        self.stats["dropped"] += sum(1 for e in out.values() if not e.observed)
        return out

    # ------------------------------------------------------------------

    #: 各观测量的「参考尺度」，用于把量纲不同的 sigma 归一化后再算观测质量。
    #: 为什么必须有这张表：早期版本直接用 `1/(1+sigma/0.1)` 计算质量，
    #: 而侦察机距离的 sigma 是 4000 m，代入后得分 2.5e-5，
    #: 把整个观测质量拉到接近 0，于是「观测严重退化」在任何一步都成立、
    #: 回退触发条件形同虚设（实测高风险步占比恒为 1.000）。
    #: 归一化后 4000/150000 ≈ 0.027，得分约 0.97，
    #: 这才符合「4 km 误差在 150 km 量程上并不算大」的直觉。
    SIGMA_REFERENCE: Dict[str, float] = {
        "target_range": 20000.0,
        "target_rcs": 5.0,
        "max_range": 20000.0,
        "interceptor_range": 150000.0,
        "jam_ratio": 5.0,
        "remaining_energy": 1400.0,
        "pd_min": 1.0,
        "pint_eff": 1.0,
        "exposure": 1.0,
        "previous_power": 80.0,
    }
    DEFAULT_SIGMA_REFERENCE: float = 1.0

    #: 各观测量的**物理取值域**。测量噪声可能把估计值推到物理上不可能的区域
    #: （例如 J/N 本应 ≥ 0，σ=0.10 的噪声会把估计推到 −0.05）。
    #: 真实测量链会输出被物理约束截断的值，因此这里显式裁剪，
    #: 而不是把"负的干扰噪声比"这种无意义的值交给策略与信念桥接。
    #: 注意：裁剪**不消耗随机数**，因此不会破坏逐量独立随机流（见 `_field_rng`）。
    PHYSICAL_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        "target_range": (0.0, None),
        "max_range": (0.0, None),
        "target_rcs": (0.0, None),
        "jam_ratio": (0.0, None),
        "remaining_energy": (0.0, None),
        "interceptor_range": (0.0, None),
        "previous_power": (0.0, None),
        "pd_min": (0.0, 1.0),
        "pint_eff": (0.0, 1.0),
        "exposure": (0.0, 1.0),
    }

    @classmethod
    def _apply_bounds(cls, name: str, value: float) -> float:
        bounds = cls.PHYSICAL_BOUNDS.get(name)
        if bounds is None:
            return float(value)
        low, high = bounds
        if low is not None and value < low:
            value = low
        if high is not None and value > high:
            value = high
        return float(value)

    def observation_quality(self, estimates: Dict[str, ObservationEstimate]) -> float:
        """把观测质量压成一个 0~1 的标量（1 = 全新鲜全精确）。

        两个**独立**因素相乘：
        * **新鲜度**：丢测 ×0.35，延迟 ×0.7；
        * **精度**：`1 / (1 + sigma/参考尺度)`，参考尺度见 `SIGMA_REFERENCE`。

        注意：这里用的是观测模型**上报**的 sigma（受 `report_honest_sigma` 控制），
        因此当智能体低报自身不确定度时，它也会"觉得自己看得更清楚"。
        这正是我们想要研究的过度自信失效模式。
        """
        if not estimates:
            return 0.0
        scores = []
        for name, estimate in estimates.items():
            score = 1.0
            if not estimate.observed:
                score *= 0.35
            elif estimate.stale:
                score *= 0.7
            reference = self.SIGMA_REFERENCE.get(name, self.DEFAULT_SIGMA_REFERENCE)
            if reference > 0.0:
                score *= 1.0 / (1.0 + estimate.sigma / reference)
            scores.append(score)
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "delay_steps": self.config.delay_steps,
            "dropout_prob": self.config.dropout_prob,
            "hide_interceptor_truth": self.config.hide_interceptor_truth,
            "hide_pint_truth": self.config.hide_pint_truth,
            "hide_exposure_truth": self.config.hide_exposure_truth,
            "report_honest_sigma": self.config.report_honest_sigma,
            "stats": dict(self.stats),
        }
