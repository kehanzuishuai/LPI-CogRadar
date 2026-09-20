"""独立的 Global Track / Track-to-Track Fusion v1 层。

它不替换 ``fusion.FusionCenter``：节点继续维护自己的 local track，
本包只消费已通过 CommBus 实际到达的 ``global-track-v1`` TrackMessage。
"""

from global_fusion.manager import (  # noqa: F401
    GLOBAL_TRACK_ENDPOINT,
    GLOBAL_TRACK_MODE_OFF,
    GLOBAL_TRACK_MODE_TRACK_FUSION,
    GLOBAL_TRACK_MODES,
    GlobalTrack,
    GlobalTrackConfig,
    GlobalTrackManager,
)
from global_fusion.observation import (  # noqa: F401
    GLOBAL_OBSERVATION_SCHEMA_VERSION,
    GlobalObservation,
    GlobalTrackObservation,
    global_observation_diagnostics,
)
from global_fusion.sharing import (  # noqa: F401
    EventTriggeredTrackShareConfig,
    EventTriggeredTrackSharePolicy,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_MEASUREMENT_AND_TRACK,
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_SHARE_MODES,
    TrackShareDecision,
)

__all__ = [
    "GLOBAL_TRACK_ENDPOINT", "GLOBAL_TRACK_MODE_OFF",
    "GLOBAL_TRACK_MODE_TRACK_FUSION", "GLOBAL_TRACK_MODES", "GlobalTrack",
    "GlobalTrackConfig", "GlobalTrackManager",
    "GLOBAL_OBSERVATION_SCHEMA_VERSION", "GlobalObservation",
    "GlobalTrackObservation", "global_observation_diagnostics",
    "EventTriggeredTrackShareConfig", "EventTriggeredTrackSharePolicy",
    "GLOBAL_SHARE_MODE_EVENT_TRACK", "GLOBAL_SHARE_MODE_MEASUREMENT",
    "GLOBAL_SHARE_MODE_MEASUREMENT_AND_TRACK", "GLOBAL_SHARE_MODE_NO_SHARE",
    "GLOBAL_SHARE_MODE_TRACK", "GLOBAL_SHARE_MODES", "TrackShareDecision",
]
