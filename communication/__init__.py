"""通信层（v4.3）。

把"某平台现在测到了什么"变成"另一个平台在某个时刻之后才能看到什么"。

```python
from communication import CommBus, CommConfig, SHARE_CONSTRAINED

bus = CommBus(["RADAR_A", "RADAR_B", "ESM_NORTH"],
              CommConfig(policy=SHARE_CONSTRAINED, base_delay_s=0.5,
                         jitter_s=0.2, loss_prob=0.1, expiry_s=2.0))
bus.publish("RADAR_A", "SENSOR_RADAR_A", measurements, now=t)

visible = bus.arrived(t)          # 只含 arrived_at <= t 的消息
stats = bus.statistics()          # 延迟 / 丢包 / 逐链路统计
rows = bus.message_log_rows()     # 逐消息日志（CSV/JSON）
```

三种策略：`no_share` / `ideal_share` / `constrained_share`，
含义与用途见 `communication/bus.py` 的模块文档。
"""

from communication.bus import (  # noqa: F401
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    SHARE_POLICIES,
    SHARE_POLICY_CN,
    CommBus,
    CommConfig,
    TimeBoundaryViolation,
)
from communication.message import (  # noqa: F401
    ALLOWED_PAYLOAD_FIELDS,
    FORBIDDEN_PAYLOAD_PREFIXES,
    CommLink,
    LinkConfig,
    MeasurementMessage,
    PayloadViolation,
    assert_payload_clean,
)

__all__ = [
    "ALLOWED_PAYLOAD_FIELDS", "CommBus", "CommConfig", "CommLink",
    "FORBIDDEN_PAYLOAD_PREFIXES", "LinkConfig", "MeasurementMessage",
    "PayloadViolation", "SHARE_CONSTRAINED", "SHARE_IDEAL", "SHARE_NONE",
    "SHARE_POLICIES", "SHARE_POLICY_CN", "TimeBoundaryViolation",
    "assert_payload_clean",
]
