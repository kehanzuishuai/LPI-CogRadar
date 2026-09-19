"""[遗留模块] 通信抗干扰阶段的跳频选择策略。

低截获雷达功率调控改造后，本模块已**弱化**：
main.py、engine、metrics 都不再引用它，仅保留以复现通信抗干扰阶段的
历史结果（历史引擎见 backup_original/simulator.py）。
新策略见 strategy/power_policy.py。

注意：原实现返回语句里的 `(0.0 if node.current_freq == node.current_freq else
self.switch_cost)` 条件恒为真，switch_cost 分支是死代码，导致该策略在配置的
switch_cost=0.25 下从不切换频率、与纯干扰组结果逐字节相同。
此处保持原样不动，以保证历史结果可复现。
"""

import math
from typing import List

from models import Disturbance, Node


class AntiJamStrategy:
    def __init__(self, disturbances: List[Disturbance], switch_cost: float = 0.25) -> None:
        self.disturbances = disturbances
        self.switch_cost = switch_cost

    def choose_frequency(self, node: Node, current_time: int) -> str:
        candidate_scores = []

        for freq in node.freq_list:
            base_score = self._expected_disturbance_for_freq(node, current_time, freq)
            cost = 0.0 if freq == node.current_freq else self.switch_cost
            candidate_scores.append((base_score + cost, cost, freq))

        candidate_scores.sort(key=lambda item: (item[0], item[1], node.freq_list.index(item[2])))
        best_score, best_cost, best_freq = candidate_scores[0]

        return best_freq if best_score < self._expected_disturbance_for_freq(node, current_time, node.current_freq) + (0.0 if node.current_freq == node.current_freq else self.switch_cost) else node.current_freq

    def _expected_disturbance_for_freq(self, node: Node, current_time: int, freq: str) -> float:
        total_effect = 0.0

        for disturbance in self.disturbances:
            if not disturbance.is_active(current_time):
                continue

            if freq not in disturbance.freq_range:
                continue

            distance_to_disturb = math.hypot(
                node.x - disturbance.center_x,
                node.y - disturbance.center_y,
            )
            if distance_to_disturb > disturbance.radius:
                continue

            spatial_factor = 1.0 - (distance_to_disturb / disturbance.radius)
            spatial_factor = max(0.0, min(1.0, spatial_factor))
            total_effect += disturbance.intensity * spatial_factor * disturbance.duty_cycle

        return min(total_effect, 1.0)
