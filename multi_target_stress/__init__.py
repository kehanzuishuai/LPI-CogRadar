"""多目标压力测试（v4.5 P2）。

要回答的问题
------------
v4.4 的协同感知结论是在**单目标**下得到的。单目标时最近邻关联几乎不会出错，
所以"协同有效"这个结论**没有经过关联层的压力考验**：

* 两个目标交叉时，最近邻会不会把它们**换号**（ID switch）？
* 密集编队里，几条航迹会不会**合并**、**互换**或**重复**？
* 短时遮挡后重现，Track ID 保得住吗？会不会**碎裂**或**新建重复航迹**？
* 真实目标附近的**虚警**会不会夺走真实航迹？远端能不能帮忙消歧？

本包提供四类压力场景、关联层审计与一组多目标指标，并把
**单雷达 / 双雷达不共享 / 理想共享 / 受限共享** 四路放在同一张表里。

纪律
----
* **跟踪器与 AI 都不读真值 ID**：`fusion/` 只吃测量对象，
  AI 上下文由 `ai/context.py` 从结构化证据构造；
  真值只在 `multi_target_stress/metrics.py` 里用于**离线**算指标。
* **不改动 NN + 卡尔曼基线**：本包不修改 `FusionConfig` 的任何默认值，
  也不改 `fusion/` 的关联与滤波数学。默认配置跑出来的结果就是基线结果。
* **不通过改指标制造压力**：四类场景的"困难"来自**几何布置、目标间距、
  虚警率与通信条件**，不是来自调参把门限放宽或收紧。
* 只有压力测试**确实显示出明显误关联**时，才考虑引入 JPDA-lite 分支，
  并同时报告"改善了哪些失效模式"与"算力代价"。
"""

from multi_target_stress.metrics import (
    ASSOC_GATE_M,
    DUPLICATE_GATE_M,
    MultiTargetMetrics,
    assign_tracks_to_truth,
)
from multi_target_stress.scenarios import (
    SCENARIO_IDS,
    STRESS_SCENARIOS,
    StressScenario,
    build_scenario_overrides,
    get_scenario,
)

__all__ = [
    "ASSOC_GATE_M",
    "DUPLICATE_GATE_M",
    "MultiTargetMetrics",
    "SCENARIO_IDS",
    "STRESS_SCENARIOS",
    "StressScenario",
    "assign_tracks_to_truth",
    "build_scenario_overrides",
    "get_scenario",
]
