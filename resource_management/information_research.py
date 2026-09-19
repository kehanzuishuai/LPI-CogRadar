"""“新鲜度 + 不确定度感知调度”的唯一研究分支。

本模块不定义新网络，只定义：

1. 一句可被实验否证的假设；
2. 主基线 / 仅新鲜度 / 仅不确定度 / 两者共用四组消融；
3. 四组完全同形的中央视角学习特征；
4. 规则基线的同信息门控配置。

禁止读取真值偏差、真实关联正误或虚警标签。来源一致性分数只是
由已到达残差证据推导的未校准指示量，不是出错概率。
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

from resource_management.observation import CentralObservation, TrackObservation
from resource_management.scheduling import SchedulingConfig
from resource_management.units import BUDGET_UNITS


RESEARCH_HYPOTHESIS = (
    "在环境、感知处理链、资源模型、训练预算和编码结构固定时，"
    "当已到达数据存在延迟、缺失或来源不一致时，显式使用信息年龄、"
    "估计协方差和基于可观测残差的来源一致性指示量，相比不使用它们，"
    "能否降低教学服务任务完成率的最差场景退化，且不增加资源超支。"
)


class AblationArm(str, Enum):
    MAIN_BASELINE = "main_baseline"
    FRESHNESS_ONLY = "freshness_only"
    UNCERTAINTY_ONLY = "uncertainty_only"
    FRESHNESS_UNCERTAINTY = "freshness_uncertainty"

    @property
    def uses_freshness(self) -> bool:
        return self in (self.FRESHNESS_ONLY, self.FRESHNESS_UNCERTAINTY)

    @property
    def uses_uncertainty(self) -> bool:
        return self in (self.UNCERTAINTY_ONLY, self.FRESHNESS_UNCERTAINTY)


ABLATION_ARMS: Tuple[AblationArm, ...] = tuple(AblationArm)


def rule_config_for_arm(
    arm: AblationArm,
    base: SchedulingConfig | None = None,
) -> SchedulingConfig:
    """为规则基线开启与学习分支完全相同的信息门。"""

    config = base or SchedulingConfig()
    return replace(
        config,
        use_information_age=arm.uses_freshness,
        use_estimate_uncertainty=arm.uses_uncertainty,
        use_source_consistency=arm.uses_uncertainty,
    )


class CentralResearchFeatureAdapter:
    """四组同形的中央视角特征适配器。

    四组的特征名、顺序、维度和槽位都一样；被消融的特征只置零。
    因此后续可使用完全相同的 MLP 结构，不把“输入更宽”误归因为
    “信息更有用”。
    """

    NODE_FIELDS: Tuple[str, ...] = (
        "node_mask", "available", "remaining_sample", "remaining_processing",
        "remaining_comm", "summary_arrival_age_norm", "summary_content_age_norm",
    )
    TRACK_FIELDS: Tuple[str, ...] = (
        "track_mask", "x_norm", "y_norm", "z_norm", "vx_norm", "vy_norm",
        "vz_norm", "track_age_norm", "sigma_position_norm",
        "source_inconsistency_indicator",
    )
    SCALES: Dict[str, float] = {
        "position_m": 30000.0,
        "velocity_mps": 300.0,
        "age_s": 30.0,
        "sigma_m": 1000.0,
    }

    def __init__(
        self,
        expected_node_ids: Sequence[str],
        tracks_per_node: int = 4,
    ) -> None:
        if not expected_node_ids:
            raise ValueError("expected_node_ids 不能为空")
        if len(set(expected_node_ids)) != len(expected_node_ids):
            raise ValueError("expected_node_ids 不能重复")
        if tracks_per_node <= 0:
            raise ValueError("tracks_per_node 必须为正")
        self.expected_node_ids = tuple(str(item) for item in expected_node_ids)
        self.tracks_per_node = int(tracks_per_node)
        self.schema_version = (
            f"rm-central-research-1.0+n{len(self.expected_node_ids)}"
            f"t{self.tracks_per_node}"
        )
        self.output_dim = len(self.feature_names())

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for node_id in self.expected_node_ids:
            names.extend(f"{node_id}_{name}" for name in self.NODE_FIELDS)
            for slot in range(self.tracks_per_node):
                names.extend(
                    f"{node_id}_track{slot}_{name}" for name in self.TRACK_FIELDS
                )
        return names

    def encode(
        self,
        observation: CentralObservation,
        arm: AblationArm,
    ) -> List[float]:
        nodes = {node.node_id: (index, node) for index, node in enumerate(observation.nodes)}
        vector: List[float] = []
        for node_id in self.expected_node_ids:
            entry = nodes.get(node_id)
            if entry is None:
                vector.extend([0.0] * (
                    len(self.NODE_FIELDS)
                    + self.tracks_per_node * len(self.TRACK_FIELDS)
                ))
                continue
            index, node = entry
            valid = (
                index < len(observation.node_valid_mask)
                and bool(observation.node_valid_mask[index])
            )
            arrival_age = self._indexed_age(
                observation.node_information_age_s, index
            )
            content_age = self._indexed_age(observation.node_content_age_s, index)
            vector.extend([
                1.0 if valid else 0.0,
                1.0 if valid and node.available else 0.0,
                self._remaining_ratio(node, 0) if valid else 0.0,
                self._remaining_ratio(node, 1) if valid else 0.0,
                self._remaining_ratio(node, 2) if valid else 0.0,
                self._age_norm(arrival_age) if arm.uses_freshness and valid else 0.0,
                self._age_norm(content_age) if arm.uses_freshness and valid else 0.0,
            ])
            tracks = sorted(
                (track for track, flag in zip(node.tracks, node.track_valid_mask)
                 if flag),
                key=lambda track: track.track_id,
            )
            for slot in range(self.tracks_per_node):
                if not valid or slot >= len(tracks):
                    vector.extend([0.0] * len(self.TRACK_FIELDS))
                    continue
                vector.extend(self._encode_track(tracks[slot], arm))
        if len(vector) != self.output_dim:
            raise RuntimeError(
                f"研究特征维度错误：{len(vector)} != {self.output_dim}"
            )
        return vector

    def describe_arm(self, arm: AblationArm) -> Dict[str, Any]:
        return {
            "arm": arm.value,
            "schema_version": self.schema_version,
            "output_dim": self.output_dim,
            "feature_names": self.feature_names(),
            "freshness_enabled": arm.uses_freshness,
            "uncertainty_enabled": arm.uses_uncertainty,
            "disabled_feature_handling": "same slot set to zero",
            "source_consistency_calibrated_probability": False,
        }

    def _encode_track(
        self, track: TrackObservation, arm: AblationArm
    ) -> List[float]:
        position_scale = self.SCALES["position_m"]
        velocity_scale = self.SCALES["velocity_mps"]
        sigma = max(float(value) for value in track.sigma_position)
        consistency = track.source_consistency_indicator
        inconsistency = (
            max(0.0, min(1.0, 1.0 - float(consistency)))
            if consistency is not None else 0.0
        )
        return [
            1.0,
            track.position[0] / position_scale,
            track.position[1] / position_scale,
            track.position[2] / position_scale,
            track.velocity[0] / velocity_scale,
            track.velocity[1] / velocity_scale,
            track.velocity[2] / velocity_scale,
            self._age_norm(track.information_age_s)
            if arm.uses_freshness else 0.0,
            min(1.0, max(0.0, sigma / self.SCALES["sigma_m"]))
            if arm.uses_uncertainty else 0.0,
            inconsistency if arm.uses_uncertainty else 0.0,
        ]

    @staticmethod
    def _indexed_age(values: Sequence[float | None], index: int) -> float | None:
        return values[index] if index < len(values) else None

    def _age_norm(self, value: float | None) -> float:
        if value is None or value == float("inf"):
            return 1.0 if value == float("inf") else 0.0
        return min(1.0, max(0.0, float(value) / self.SCALES["age_s"]))

    @staticmethod
    def _remaining_ratio(node: Any, unit_index: int) -> float:
        unit = BUDGET_UNITS[unit_index]
        capacity = float(node.capacity.get(unit.value, 0.0) or 0.0)
        remaining = float(node.remaining.get(unit.value, 0.0) or 0.0)
        return max(0.0, min(1.0, remaining / capacity)) if capacity > 0.0 else 0.0
