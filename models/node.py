from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    node_id: str
    x: float
    y: float
    vx: float
    vy: float
    tx_power: float
    rx_threshold: float
    freq_list: List[str]
    current_freq: str
    bandwidth: float
    rate_level: str
    is_active: bool = True

    def update_position(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt