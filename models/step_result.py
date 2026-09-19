"""单步仿真结果记录。

替代原通信抗干扰阶段的 LinkState：LinkState 描述「一条通信链路的质量」，
StepResult 描述「雷达一步的探测 / 截获 / 干扰 / 能量 / 收益全貌」。

所有字段都是普通标量或纯数据 dataclass，可安全序列化到 CSV / JSON / HTML。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .receive_record import InterceptionRecord


@dataclass
class TargetDetection:
    """单个目标在该步的探测结果。"""

    target_id: str
    range_m: float
    rcs_m2: float
    echo_power_w: float
    snr_db: float
    pd: float
    satisfied: bool


@dataclass
class StepResult:
    """雷达一步的完整结果。"""

    step_index: int
    time: float

    # --- 动作 ---
    power_level: int
    tx_power_w: float

    # --- 探测 ---
    detections: List[TargetDetection] = field(default_factory=list)
    pd_min: float = 0.0
    snr_radar_db_min: float = float("-inf")
    required_snr_db: float = 0.0
    task_satisfied: bool = False
    detection_term: float = 0.0

    # --- 截获 ---
    interceptions: List[InterceptionRecord] = field(default_factory=list)
    intercept_prob: float = 0.0  # 有效截获概率 Pint_eff（含累计暴露，用于奖励与主指标）
    intercept_prob_instant: float = 0.0  # 本步瞬时截获概率 Pint_inst（暴露量的来源）
    intercept_snr_db: float = float("-inf")

    # --- 累计暴露 / 侦察证据 ---
    exposure: float = 0.0  # 本步开始时的暴露量（参与本步 Pint_eff）
    exposure_next: float = 0.0  # 本步结束后的暴露量（供下一步使用）

    # --- 干扰 ---
    jammer_active: bool = False
    jammer_mode: str = ""  # 自适应干扰机当前模式；固定模式下为空串
    jammer_mode_cn: str = ""
    jam_noise_ratio: float = 0.0  # J / N
    interference_power_w: float = 0.0
    noise_power_w: float = 0.0

    # --- 能量 ---
    step_energy_j: float = 0.0
    cumulative_energy_j: float = 0.0
    remaining_energy_j: float = 0.0
    energy_fraction: float = 0.0  # 累计能耗 / 能量预算
    energy_exhausted: bool = False

    # --- 任务与收益 ---
    task_violated: bool = False  # 本步 Pd_min < required_pd
    reward: float = 0.0
    terminal_penalty: float = 0.0  # 能量耗尽时的一次性终端惩罚（仅终止那一步非 0）

    def to_row(self) -> Dict[str, object]:
        """展平为一行，供 CSV / 表格使用。"""
        return {
            "time": self.time,
            "step_index": self.step_index,
            "power_level": self.power_level,
            "tx_power_w": self.tx_power_w,
            "pd_min": self.pd_min,
            "snr_radar_db_min": self.snr_radar_db_min,
            "task_satisfied": int(self.task_satisfied),
            "task_violated": int(self.task_violated),
            "intercept_prob": self.intercept_prob,
            "intercept_prob_instant": self.intercept_prob_instant,
            "intercept_snr_db": self.intercept_snr_db,
            "exposure": self.exposure,
            "exposure_next": self.exposure_next,
            "jammer_active": int(self.jammer_active),
            "jammer_mode": self.jammer_mode,
            "jam_noise_ratio": self.jam_noise_ratio,
            "step_energy_j": self.step_energy_j,
            "cumulative_energy_j": self.cumulative_energy_j,
            "remaining_energy_j": self.remaining_energy_j,
            "energy_fraction": self.energy_fraction,
            "energy_exhausted": int(self.energy_exhausted),
            "reward": self.reward,
            "terminal_penalty": self.terminal_penalty,
        }
