"""多源目标关联与融合（v4.3）。

数据生成链里的位置
------------------
```
真实状态 → 可见性 → 测量 → 通信 → **融合** → 决策
```

融合层把"一堆来自不同传感器、不同时刻、可能还经过通信延迟的测量"
变成"若干条航迹"，并且**每条航迹都能追溯到它用了哪些传感器、哪些时刻的测量**。

本阶段刻意从**基础算法**做起（用户明确要求"不需要一开始实现复杂算法"）：

* **关联**：门限 + 最近邻（贪心指派），用极坐标一致性判定；
* **融合**：**信息形式加权**（按 1/σ² 加权）的笛卡尔状态估计；
* **航迹管理**：起始 / 外推 / 删除。

⚠️ 真值隔离
------------
融合算法**只吃测量**。`fusion/` 全部模块不得访问 `Scene` / 实体真值。
真值只在 `validation/` 与评测脚本里使用（算误差、算丢轨/重复轨迹），
这是"离线评测"通道。

局限（README §11D 会写明）
--------------------------
* 没有 JPDA / MHT，遮挡与密集目标下关联会出错；
* 协方差按对角处理，未建模方位-俯仰与距离的交叉项；
* **没有航迹外推的机动模型**（只有匀速直线）；
* 被动（无距离）测量**不参与位置更新**——单站被动测距物理上做不到，
  用角度去更新笛卡尔位置需要不同的滤波器（如只测角跟踪），本阶段不实现；
* 时间对齐只做"按测量自身的 `time_s` 排序 + 简单匀速回推"，
  没有做完整的状态转移矩阵推演。
"""

from fusion.track import Track, TrackSource, TracksSnapshot  # noqa: F401
from fusion.association import (  # noqa: F401
    AssociationConfig,
    AssociationResult,
    associate,
)
from fusion.center import (  # noqa: F401
    FusionCenter,
    FusionConfig,
    measurement_to_cartesian,
)

__all__ = [
    "AssociationConfig", "AssociationResult", "FusionCenter", "FusionConfig",
    "Track", "TrackSource", "TracksSnapshot", "associate",
    "measurement_to_cartesian",
]
