from dataclasses import dataclass


@dataclass
class LinkState:
    tx: str
    rx: str
    distance: float
    path_loss: float
    noise: float
    disturb_effect: float
    quality_score: float
    available: bool