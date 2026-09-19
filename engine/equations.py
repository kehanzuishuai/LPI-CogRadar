"""简化雷达方程 / 侦察链路方程与概率模型（低截获雷达功率调控仿真 第一版）。

设计原则
--------
1. 本模块是**叶子模块**：只用标准库，不 import 工程内任何其他模块，
   纯函数、无状态。这样 models / engine / strategy / metrics 都可以自由引用它，
   不会产生循环依赖，也便于单独校验。
2. 雷达方程与侦察方程保留物理量纲（W、m、Hz、K），
   把难以精确建模的因素（脉压增益、相参积累、旁瓣对消、极化损失等）
   合并为**可配置的等效因子**，在 JSON 配置里显式给出，便于审计与调参。
3. 探测概率 Pd / 截获概率 Pint 采用 ROC 曲线的 logistic 简化：

       P = 1 / (1 + exp(-(SNR_dB - SNR_ref_dB) / slope_dB))

   概率在参考 SNR 处恰为 0.5，slope 控制曲线陡峭度。该形式单调、可微、
   无需查表，适合作为强化学习奖励的来源；其局限（未区分虚警概率 Pf、
   未建模 Swerling 起伏）在 README「模型局限」一节说明。
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# 物理常数
# --------------------------------------------------------------------------

BOLTZMANN = 1.380649e-23  # 玻尔兹曼常数 J/K
T0_K = 290.0  # 标准参考噪声温度 K


# --------------------------------------------------------------------------
# 单位换算
# --------------------------------------------------------------------------

def db2lin(value_db: float) -> float:
    """dB -> 线性倍数。"""
    return 10.0 ** (value_db / 10.0)


def lin2db(value: float) -> float:
    """线性倍数 -> dB。非正值返回 -inf，避免 log(0) 抛异常。"""
    if value <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(value)


# --------------------------------------------------------------------------
# 噪声
# --------------------------------------------------------------------------

def thermal_noise_w(
    bandwidth_hz: float,
    noise_figure_db: float,
    temperature_k: float = T0_K,
) -> float:
    """接收机热噪声功率 N = k * T * B * F [W]。"""
    return BOLTZMANN * temperature_k * bandwidth_hz * db2lin(noise_figure_db)


# --------------------------------------------------------------------------
# 雷达方程（双程）
# --------------------------------------------------------------------------

def radar_echo_power_w(
    pt_w: float,
    gain_db: float,
    wavelength_m: float,
    rcs_m2: float,
    range_m: float,
    system_loss_db: float,
) -> float:
    """雷达接收回波功率：

        S = Pt * Gt * Gr * lambda^2 * sigma / ( (4*pi)^3 * R^4 * L )

    收发共用同一天线，故 Gt = Gr = G。
    """
    if range_m <= 0.0 or pt_w <= 0.0:
        return 0.0

    gain = db2lin(gain_db)
    numerator = pt_w * gain * gain * (wavelength_m ** 2) * rcs_m2
    denominator = ((4.0 * math.pi) ** 3) * (range_m ** 4) * db2lin(system_loss_db)
    return numerator / denominator


# --------------------------------------------------------------------------
# 单程链路
# --------------------------------------------------------------------------

def one_way_power_w(
    pt_w: float,
    tx_gain_db: float,
    rx_gain_db: float,
    wavelength_m: float,
    range_m: float,
    system_loss_db: float,
) -> float:
    """单程接收功率：

        Pr = Pt * Gt * Gr * lambda^2 / ( (4*pi)^2 * R^2 * L )

    同时用于「雷达 -> 侦察接收机」的截获链路和「干扰机 -> 雷达」的干扰链路。
    """
    if range_m <= 0.0 or pt_w <= 0.0:
        return 0.0

    numerator = pt_w * db2lin(tx_gain_db) * db2lin(rx_gain_db) * (wavelength_m ** 2)
    denominator = ((4.0 * math.pi) ** 2) * (range_m ** 2) * db2lin(system_loss_db)
    return numerator / denominator


# --------------------------------------------------------------------------
# 信噪比 / 信干噪比
# --------------------------------------------------------------------------

def snr_db(signal_w: float, noise_w: float) -> float:
    """S/N，单位 dB。"""
    return lin2db(signal_w / noise_w) if noise_w > 0.0 else float("inf")


def sinr_db(signal_w: float, noise_w: float, interference_w: float = 0.0) -> float:
    """S/(N+J)，单位 dB。干扰通过抬高噪声基底起作用。"""
    total = noise_w + interference_w
    return lin2db(signal_w / total) if total > 0.0 else float("inf")


# --------------------------------------------------------------------------
# 概率模型（ROC 的 logistic 简化）
# --------------------------------------------------------------------------

def logistic_prob(snr_db_value: float, ref_snr_db: float, slope_db: float) -> float:
    """由 SNR 得到概率，SNR = ref_snr_db 时概率恰为 0.5。

    slope_db 越小曲线越陡。指数做截断以避免数值溢出。
    """
    if slope_db <= 0.0:
        raise ValueError("slope_db 必须为正数")

    if math.isinf(snr_db_value):
        return 1.0 if snr_db_value > 0 else 0.0

    z = (snr_db_value - ref_snr_db) / slope_db
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def snr_db_for_prob(prob: float, ref_snr_db: float, slope_db: float) -> float:
    """logistic_prob 的反函数：为使概率达到 prob 所需的 SNR（dB）。

    用于把「任务要求 Pd >= required_pd」换算成 SNR 门限，
    规则策略据此挑选最低可用功率档位。
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob 必须落在 (0, 1) 开区间内")
    return ref_snr_db + slope_db * math.log(prob / (1.0 - prob))


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------

def angular_offset_deg(
    origin_x: float,
    origin_y: float,
    ref_x: float,
    ref_y: float,
    other_x: float,
    other_y: float,
) -> float:
    """以 origin 为顶点，参考点(ref) 与另一点(other) 之间的夹角，范围 [0, 180]。

    用于判断侦察接收机是否落在雷达主瓣内：
    雷达波束指向主目标，若侦察机与主目标方向夹角小于半波束宽度，
    则侦察机被主瓣照射，截获概率显著升高。
    """
    ref_angle = math.atan2(ref_y - origin_y, ref_x - origin_x)
    other_angle = math.atan2(other_y - origin_y, other_x - origin_x)

    diff = abs(ref_angle - other_angle) % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return math.degrees(diff)
