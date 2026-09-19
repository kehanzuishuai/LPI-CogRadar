"""常速度 Kalman 滤波器（v4.4）。

升级前的"融合"只是按 1/σ² 加权平均位置，**没有状态预测**，
因此：漏检时航迹不会外推、关联只能用当前位置硬门限、
速度估计靠相邻位置差分（对噪声极敏感）。本模块提供真正的滤波：

    状态  x = [x, y, z, vx, vy, vz]ᵀ          （ENU 笛卡尔，米 / 米每秒）
    预测  x⁻ = F(dt)·x,   P⁻ = F·P·Fᵀ + Q(dt)
    更新  K = P⁻Hᵀ(H P⁻ Hᵀ + R)⁻¹
          x = x⁻ + K(z − H x⁻),  P = (I − K H)P⁻
    观测  z = [x, y, z]ᵀ,  H = [I₃ | 0₃]

过程噪声用**连续白噪声加速度模型**（标准做法）：

    Q(dt) = q · [[dt⁴/4·I₃, dt³/2·I₃], [dt³/2·I₃, dt²·I₃]]

其中 q 是加速度功率谱密度（m²/s³），是唯一的整定参数，
物理含义清楚：目标机动越强，q 越大。

为什么用笛卡尔而不是极坐标：量测是极坐标（距离/方位/俯仰），
用它做滤波需要扩展卡尔曼（EKF）与雅可比。本阶段**先把闭环跑通**，
采用"极坐标→笛卡尔转换 + 协方差一阶传播"，是工程上常见的折中。
**局限**：转换会引入偏差（尤其远距离横向误差），已在 README 写明。

⚠️ 本模块不接触任何真值，只吃测量转换后的位置与协方差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from engine.geometry import Vec3


def _mat3_identity() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mat3_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def _mat3_scale(a: List[List[float]], s: float) -> List[List[float]]:
    return [[a[i][j] * s for j in range(3)] for i in range(3)]


@dataclass
class KalmanConfig:
    """滤波器整定参数。"""

    #: 加速度功率谱密度（m²/s³）。越大越"信任测量、不信任模型"。
    process_noise_accel: float = 2.0
    #: 观测噪声下限（m²）：即使测量报的 σ 很小也不低于它，避免滤波器过度自信
    min_measurement_variance: float = 1.0
    #: 初始速度不确定度（m/s）
    initial_velocity_sigma: float = 50.0

    def validate(self) -> None:
        if self.process_noise_accel < 0:
            raise ValueError("process_noise_accel 不能为负")
        if self.min_measurement_variance <= 0:
            raise ValueError("min_measurement_variance 必须为正")


class ConstantVelocityKalmanFilter:
    """常速度模型 Kalman 滤波器（6 维状态 / 3 维观测）。

    实现刻意只用**对角近似**的 3×3 子块运算：
    状态协方差 P 是 6×6，但过程噪声与观测噪声都各向同性，
    因此 P 的三个 3×3 子块（PP / PV / VP / VV）始终是对称满阵，
    这里用朴素矩阵乘法直接算，规模小、可读性好，
    也便于单元测试逐项核对。
    """

    def __init__(
        self,
        position: Vec3,
        position_sigma: Vec3,
        config: Optional[KalmanConfig] = None,
        velocity: Optional[Vec3] = None,
    ) -> None:
        self.config = config or KalmanConfig()
        self.config.validate()

        self.position = position
        self.velocity = velocity or Vec3()
        # P 的四个 3×3 子块
        self.P_pp = [
            [max(position_sigma.x, 1e-6) ** 2, 0.0, 0.0],
            [0.0, max(position_sigma.y, 1e-6) ** 2, 0.0],
            [0.0, 0.0, max(position_sigma.z, 1e-6) ** 2],
        ]
        self.P_pv = [[0.0] * 3 for _ in range(3)]
        self.P_vv = _mat3_scale(
            _mat3_identity(), self.config.initial_velocity_sigma ** 2
        )

    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """时间更新：x⁻ = F·x，P⁻ = F·P·Fᵀ + Q。"""
        if dt <= 0.0:
            return
        # x⁻
        self.position = self.position + self.velocity * dt

        # P⁻ = F P Fᵀ + Q，其中 F = [[I, dt·I], [0, I]]
        # PP⁻ = PP + dt·(PV + PVᵀ) + dt²·VV
        new_pp = [
            [self.P_pp[i][j] + dt * (self.P_pv[i][j] + self.P_pv[j][i])
             + dt * dt * self.P_vv[i][j] for j in range(3)]
            for i in range(3)
        ]
        # PV⁻ = PV + dt·VV
        new_pv = [
            [self.P_pv[i][j] + dt * self.P_vv[i][j] for j in range(3)]
            for i in range(3)
        ]
        # VV⁻ = VV
        new_vv = [row[:] for row in self.P_vv]

        # Q（连续白噪声加速度模型）
        q = self.config.process_noise_accel
        q_pp = dt ** 4 / 4.0 * q
        q_pv = dt ** 3 / 2.0 * q
        q_vv = dt ** 2 * q
        for i in range(3):
            new_pp[i][i] += q_pp
            new_pv[i][i] += q_pv
            new_vv[i][i] += q_vv

        self.P_pp, self.P_pv, self.P_vv = new_pp, new_pv, new_vv

    def update(
        self, measurement: Vec3, sigma: Vec3, dt_hint: float = 0.0
    ) -> float:
        """量测更新。返回位置残差范数（用于门限/诊断）。"""
        # 创新 z − H x⁻
        residual = measurement - self.position
        residual_norm = residual.norm()

        # S = H P⁻ Hᵀ + R = PP + R（对角）
        min_var = self.config.min_measurement_variance
        s_diag = [
            self.P_pp[0][0] + max(sigma.x ** 2, min_var),
            self.P_pp[1][1] + max(sigma.y ** 2, min_var),
            self.P_pp[2][2] + max(sigma.z ** 2, min_var),
        ]
        # 观测只作用在位置上，且 R 对角 => 增益按轴独立计算最简
        gain = [self.P_pp[i][i] / s_diag[i] for i in range(3)]

        # x = x⁻ + K·residual
        self.position = self.position + Vec3(
            gain[0] * residual.x, gain[1] * residual.y, gain[2] * residual.z
        )
        # 速度也通过 P_pv 得到修正：v += (PVᵀ/ S) · residual
        # （对角 S 下即为逐轴 PV[i][i]/S[i]）
        v_correction = Vec3(
            self.P_pv[0][0] / s_diag[0] * residual.x,
            self.P_pv[1][1] / s_diag[1] * residual.y,
            self.P_pv[2][2] / s_diag[2] * residual.z,
        )
        self.velocity = self.velocity + v_correction

        # P = (I − K H) P⁻ ：对位置与速度子块分别收缩
        for i in range(3):
            shrink = 1.0 - gain[i]
            for j in range(3):
                self.P_pp[i][j] *= shrink
                self.P_pv[i][j] *= shrink
        # 对称化（数值上保持对称）
        for i in range(3):
            for j in range(i + 1, 3):
                avg = 0.5 * (self.P_pp[i][j] + self.P_pp[j][i])
                self.P_pp[i][j] = self.P_pp[j][i] = avg

        return residual_norm

    # ------------------------------------------------------------------

    def position_sigma(self) -> Vec3:
        """位置各轴标准差（非负）。"""
        return Vec3(
            math.sqrt(max(self.P_pp[0][0], 0.0)),
            math.sqrt(max(self.P_pp[1][1], 0.0)),
            math.sqrt(max(self.P_pp[2][2], 0.0)),
        )

    def velocity_sigma(self) -> Vec3:
        return Vec3(
            math.sqrt(max(self.P_vv[0][0], 0.0)),
            math.sqrt(max(self.P_vv[1][1], 0.0)),
            math.sqrt(max(self.P_vv[2][2], 0.0)),
        )

    def mahalanobis_sq(self, measurement: Vec3, sigma: Vec3) -> float:
        """预测到量测的马氏距离平方（用于门限关联）。

        这是**正确的门限**做法：用预测协方差 + 量测协方差一起归一化，
        而不是用固定米数。固定门限要么在远距离过松、要么在近距离过紧。
        """
        residual = measurement - self.position
        min_var = self.config.min_measurement_variance
        total = 0.0
        for value, p_diag, s in (
            (residual.x, self.P_pp[0][0], sigma.x),
            (residual.y, self.P_pp[1][1], sigma.y),
            (residual.z, self.P_pp[2][2], sigma.z),
        ):
            denom = p_diag + max(s ** 2, min_var)
            if denom > 0:
                total += value * value / denom
        return total

    def to_dict(self) -> dict:
        return {
            "x": self.position.x, "y": self.position.y, "z": self.position.z,
            "vx": self.velocity.x, "vy": self.velocity.y, "vz": self.velocity.z,
            "sigma_x": self.position_sigma().x,
            "sigma_y": self.position_sigma().y,
            "sigma_z": self.position_sigma().z,
            "sigma_vx": self.velocity_sigma().x,
            "sigma_vy": self.velocity_sigma().y,
            "sigma_vz": self.velocity_sigma().z,
        }
