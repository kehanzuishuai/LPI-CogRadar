"""传感器测量层（v4.2）。

分层位置与数据字典
------------------
```
真值世界           Scene / Simulator        实体的真实位置、速度、姿态
   ↓ 只读
传感器可见性       sensor.sensor            作用范围 / 视场 / 遮挡 / 更新周期 / 可用性
   ↓ 产生
测量报告           sensor.record            MeasurementRecord（带时间戳与协方差，可能不存在）
   ↓ 汇聚（剥掉真值）
算法可见输入       sensor.fusion            FusedObservation.vector（定长，只有候选 ID 与估计值）
```

**三层数据必须分清**，混用会导致"用真值评测算法却以为在评测传感器"这类错误：

| 层 | 内容 | 谁能读 |
| --- | --- | --- |
| 仿真真值 | `Scene` 实体状态、`StepResult` 的真实 Pd/暴露/能耗 | 仿真器、指标、**评测** |
| 传感器测量 | `MeasurementRecord` 的测量字段 | 传感器、融合层、**评测**（算误差） |
| 算法可见输入 | `FusedObservation.vector` | **决策算法** |

`MeasurementRecord.truth_*` 与 `Sensor.truth_of_candidate()` 属于评测专用通道，
`fusion` 打包时绝不读取它们（有 `assert_no_truth_leak` + 单元测试保证）。

⚠️ **导入顺序不能随便调，环的打断点在 `sensor/config.py`**：
`engine/__init__.py` 会导入 `engine.env`，`engine.env` 又导入 `sensor.config`，
而 `sensor.config` 需要 `sensor.occlusion`，后者又需要 `engine.geometry`。
若 `sensor.config` 在**模块顶层**导入 `OcclusionModel`，就会出现
`engine/__init__ ↔ sensor.config ↔ sensor.occlusion`
的环（实测报 `ImportError: cannot import name 'OcclusionModel'`，
且只在"先 import sensor"这种入口顺序下才暴露）。
因此 `sensor/config.py` 把该导入**放进函数内部**，等调用时 `engine` 包已解析完。

模块导航
--------
* `sensor.record`     测量记录、五种以上「没有数据」的显式原因
* `sensor.occlusion`  简化遮挡模型（AABB / 球，线段相交）
* `sensor.sensor`     传感器本体（RadarSensor 主动 / EsmSensor 被动）与套件
* `sensor.fusion`     测量汇聚为定长观测向量
* `sensor.reporting`  CSV / JSON 导出与测量级统计
"""

from sensor.fusion import (  # noqa: F401
    FusedObservation,
    SLOT_FIELDS,
    TrackTableConfig,
    assert_no_truth_leak,
    fuse_measurements,
)
from sensor.occlusion import (  # noqa: F401
    BoxOccluder,
    OcclusionModel,
    SphereOccluder,
    build_occluder,
)
from sensor.sensor import (  # noqa: F401
    KIND_ESM_SENSOR,
    KIND_RADAR_SENSOR,
    EsmSensor,
    RadarSensor,
    Sensor,
    SensorConfig,
    SensorSuite,
)

__all__ = [
    "BoxOccluder", "EsmSensor", "FusedObservation", "KIND_ESM_SENSOR",
    "KIND_RADAR_SENSOR", "MeasurementRecord", "NO_DATA_REASONS", "NoDataReason",
    "OcclusionModel", "REASON_CN", "REASON_DIMENSION", "REASON_PROVENANCE", "RadarSensor", "SLOT_FIELDS",
    "Sensor", "SensorConfig", "SensorReport", "SensorSuite", "SphereOccluder",
    "SuiteReport", "TargetOutcome", "TrackTableConfig", "assert_no_truth_leak",
    "build_occluder", "diagonal_covariance", "fuse_measurements",
    "infinite_range_covariance",
]
