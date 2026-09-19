from dataclasses import dataclass


@dataclass
class Flow:
    flow_id: str
    src: str
    dst: str
    priority: int