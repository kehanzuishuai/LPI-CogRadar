from dataclasses import dataclass


@dataclass
class Disturbance:
    disturb_id: str
    center_x: float
    center_y: float
    radius: float
    start_time: int
    end_time: int
    freq_range: list[str]
    intensity: float
    duty_cycle: float

    def is_active(self, current_time: int) -> bool:
        return self.start_time <= current_time <= self.end_time