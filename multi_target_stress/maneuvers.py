"""机动目标注入（v4.5 S5：模型失配压力）。

为什么改"真值运动"是合法的、而改"物理公式"不是
------------------------------------------------
跟踪器里用的是**常速度（CV）卡尔曼**。要检验"模型失配会怎样"，
就必须让**目标真的机动**——也就是改真值世界的运动，而不是去改雷达方程、
检测门限或量测噪声。目标机动属于**场景**，不属于物理模型。

本模块在运行期按时刻表给目标**阶跃式**改速度：

    t < t0          匀速（跟踪器的 CV 模型完全正确）
    t ≥ t0          速度突变 → 预测与量测瞬间失配

为什么用"速度阶跃"而不是连续加速度：阶跃是**最坏的**模型失配，
它把"跟踪器能多快重新收敛"这件事压成一个可测的时间常数；
连续加速度只会得到一个被平均掉的、看不出恢复过程的结果。

⚠️ 本模块只写**真值实体**（`sim.targets`），不产生也不修改任何测量；
传感器随后照常观测，跟踪器\算法对这些机动一无所知。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.geometry import Attitude, Vec3


@dataclass(frozen=True)
class VelocityStep:
    """一次速度阶跃（机动）。"""

    time_s: float
    target_id: str
    #: 速度**增量**（ENU，米/秒）——阶跃就是在原速度上加这个量
    delta_v: Tuple[float, float, float]
    label: str

    def describe(self) -> str:
        dx, dy, dz = self.delta_v
        return (
            f"t={self.time_s:g}s {self.target_id} {self.label}"
            f"（Δv=({dx:+.0f}, {dy:+.0f}, {dz:+.0f}) m/s）"
        )


def turn_step(time_s: float, target_id: str, current_velocity: Vec3,
              heading_change_deg: float, label: str = "") -> VelocityStep:
    """构造一个"水平转弯"：保持速率、把速度方向旋转给定角度。

    旋转角自**当前速度方向**起算（正 = 顺时针，与工程内方位约定一致）。
    """
    speed = current_velocity.norm()
    if speed <= 0.0:
        raise ValueError("转弯机动要求目标当前速度非零")
    vx, vy = current_velocity.x, current_velocity.y
    angle = math.radians(heading_change_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # 方位自 +y 顺时针为正：绕 z 轴顺时针旋转
    new_vx = vx * cos_a + vy * sin_a
    new_vy = -vx * sin_a + vy * cos_a
    return VelocityStep(
        time_s=time_s, target_id=target_id,
        delta_v=(new_vx - vx, new_vy - vy, 0.0),
        label=label or f"水平转弯 {heading_change_deg:+.0f}°",
    )


def speed_step(time_s: float, target_id: str, current_velocity: Vec3,
               delta_speed_mps: float, label: str = "") -> VelocityStep:
    """构造一个"加/减速"：沿当前速度方向改变速率。"""
    speed = current_velocity.norm()
    if speed <= 0.0:
        raise ValueError("加减速机动要求目标当前速度非零")
    scale = (speed + delta_speed_mps) / speed
    if scale < 0.0:
        raise ValueError("减速不能把速度变成负的（目标会倒着飞）")
    vx, vy, vz = current_velocity.x * scale, current_velocity.y * scale, current_velocity.z * scale
    return VelocityStep(
        time_s=time_s, target_id=target_id,
        delta_v=(vx - current_velocity.x, vy - current_velocity.y,
                 vz - current_velocity.z),
        label=label or f"{'加速' if delta_speed_mps >= 0 else '减速'}"
                       f" {delta_speed_mps:+.0f} m/s",
    )


def climb_step(time_s: float, target_id: str, climb_rate_mps: float,
               label: str = "") -> VelocityStep:
    """构造一个"爬升/下降"：给 z 方向一个速度增量。"""
    return VelocityStep(
        time_s=time_s, target_id=target_id,
        delta_v=(0.0, 0.0, climb_rate_mps),
        label=label or f"{'爬升' if climb_rate_mps >= 0 else '下降'}"
                       f" {abs(climb_rate_mps):.0f} m/s",
    )


class ManeuverInjector:
    """按时刻表注入机动，并记录"哪一步触发了哪次机动"。"""

    def __init__(self, steps: Sequence[VelocityStep]) -> None:
        self.steps: List[VelocityStep] = sorted(
            steps, key=lambda s: (s.time_s, s.target_id)
        )
        self._applied: List[int] = []
        #: 已触发的机动事件（供报告与 AI 证据链使用；**不含真值位置**）
        self.events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._applied.clear()
        self.events.clear()

    @property
    def schedule(self) -> List[str]:
        return [step.describe() for step in self.steps]

    def apply(self, sim: Any, now: float) -> List[Dict[str, Any]]:
        """把 `now` 时刻该发生的机动写进真值实体；返回本步新触发的事件。"""
        fired: List[Dict[str, Any]] = []
        for index, step in enumerate(self.steps):
            if index in self._applied or step.time_s > now + 1e-12:
                continue
            entity = _find_target(sim, step.target_id)
            self._applied.append(index)
            if entity is None:
                continue
            current = entity.velocity
            new_velocity = Vec3(
                current.x + step.delta_v[0],
                current.y + step.delta_v[1],
                current.z + step.delta_v[2],
            )
            speed = new_velocity.norm()
            entity.set_pose(
                velocity=new_velocity,
                # 航向随速度方向走（方位自 +y 顺时针为正）
                attitude=Attitude(
                    heading_deg=(math.degrees(math.atan2(new_velocity.x,
                                                         new_velocity.y)) % 360.0
                                 if speed > 0 else entity.heading_deg),
                    pitch_deg=(math.degrees(math.asin(
                        max(-1.0, min(1.0, new_velocity.z / speed))
                    )) if speed > 0 else 0.0),
                    roll_deg=0.0,
                ),
            )
            event = {
                "time_s": float(step.time_s),
                "applied_at_s": float(now),
                "target_id": step.target_id,
                "label": step.label,
                "delta_v": tuple(float(v) for v in step.delta_v),
                # 机动后的速率是**观测不到的真值**，只用于报告，
                # 不进算法、不进 AI 上下文
                "speed_after_mps": float(speed),
            }
            self.events.append(event)
            fired.append(event)
        return fired


def _find_target(sim: Any, target_id: str) -> Optional[Any]:
    for target in getattr(sim, "targets", []) or []:
        if getattr(target, "target_id", None) == target_id:
            return target
    scene = getattr(sim, "scene", None)
    if scene is not None:
        try:
            return scene.by_id(target_id)
        except Exception:  # noqa: BLE001 - 场景里没有这个 ID 时视为未命中
            return None
    return None
