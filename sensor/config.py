"""从场景配置构建传感器套件（v4.2）。

配置格式
--------
在场景 JSON 里增加两个可选段落：

```json
"sensors": [
  {
    "sensor_id": "RS1", "mounting_id": "RADAR1", "sensor_kind": "radar",
    "max_range_m": 20000.0, "min_range_m": 0.0,
    "az_fov_deg": 60.0, "el_fov_deg": 30.0,
    "update_period_s": 1.0,
    "range_sigma_rel": 0.01, "range_sigma_abs_m": 5.0,
    "az_sigma_deg": 0.5, "el_sigma_deg": 0.5, "range_rate_sigma_mps": 1.0,
    "snr50_db": 6.0, "pd_slope_db": 2.0,
    "false_alarm_rate": 0.02,
    "tx_power_w": 18.0, "peak_gain_db": 30.0, "wavelength_m": 0.1,
    "bandwidth_hz": 1000000.0, "noise_figure_db": 3.0,
    "system_loss_db": 3.0, "temperature_k": 290.0,
    "available": true, "seed": 42
  }
],
"occluders": [
  {"occluder_id": "HILL1", "shape": "sphere",
   "center_x": 3000.0, "center_y": 2000.0, "center_z": 0.0, "radius_m": 800.0}
]
```

**没有 `sensors` 段时的默认行为**：为每一部雷达自动生成一个 `RadarSensor`，
为每一个侦察机构建一个 `EsmSensor`，参数取该平台自身的物理量
（作用距离按雷达方程在最低功率档下的探测距离给一个保守默认值）。
这样旧场景不需要改配置也能直接使用测量模式，同时**旧观测模式完全不受影响**——
传感器层只在 `observation_mode` 为 `ideal` / `realistic` 时被构造与调用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from engine.equations import (
    lin2db,
    logistic_prob,
    radar_echo_power_w,
    snr_db_for_prob,
    thermal_noise_w,
)

from sensor.sensor import (
    KIND_ESM_SENSOR,
    KIND_RADAR_SENSOR,
    EsmSensor,
    RadarSensor,
    Sensor,
    SensorConfig,
    SensorSuite,
)


def _occlusion_model(occluders):
    """**延迟导入** `OcclusionModel`（见下方"导入环"说明）。

    为什么不在模块顶层导入：`sensor.occlusion` 需要 `engine.geometry`，
    而 `engine/__init__.py` 会导入 `engine.env`，`engine.env` 又导入本模块——
    顶层导入会构成 `engine/__init__ ↔ sensor.config ↔ sensor.occlusion`
    的环，表现为 `import sensor` / `from sensor.config import ...` 直接抛
    `ImportError: cannot import name 'OcclusionModel'`
    （只在"先 import sensor"这种入口顺序下触发，所以长期没被发现）。
    放到函数内导入就安全了：调用本模块时 `engine` 包必然已解析完。
    """
    from sensor.occlusion import OcclusionModel

    return OcclusionModel.from_config(occluders)

#: 默认的量测噪声（在没有任何配置时使用；刻意偏乐观，避免旧场景被"莫名其妙地看不见"）
DEFAULT_RADAR_SIGMA_REL = 0.01
DEFAULT_RADAR_SIGMA_ABS_M = 5.0
DEFAULT_RADAR_AZ_SIGMA_DEG = 0.5
DEFAULT_RADAR_EL_SIGMA_DEG = 0.5
DEFAULT_RADAR_VR_SIGMA_MPS = 1.0
DEFAULT_ESM_AZ_SIGMA_DEG = 1.5
DEFAULT_ESM_EL_SIGMA_DEG = 1.5


def radar_detection_range_m(radar: Any, rcs_m2: float, snr_db: Optional[float] = None) -> float:
    """由雷达方程反解"在给定 RCS 下能探测多远"。

    推导：S ∝ Pt·G²·λ²·σ /((4π)³R⁴L)，要 S/N = snr_req，解出

        R = [ Pt·G²·λ²·σ / ((4π)³·L·N·snr_req_lin) ]^(1/4)

    用途：在没有显式配置作用距离时，给传感器一个**有物理依据**的默认值，
    而不是随手写一个数。注意这**没有**改动任何物理公式，只是把已有方程反解。
    """
    if snr_db is None:
        snr_db = snr_db_for_prob(
            float(getattr(radar, "required_pd", 0.8)),
            float(radar.snr50_db),
            float(radar.pd_slope_db),
        )
    noise_w = thermal_noise_w(
        float(radar.bandwidth_hz), float(radar.noise_figure_db),
        float(radar.temperature_k),
    )
    gain = 10.0 ** (float(radar.peak_gain_db) / 10.0)
    loss = 10.0 ** (float(radar.system_loss_db) / 10.0)
    snr_lin = 10.0 ** (snr_db / 10.0)
    numerator = float(radar.tx_power_w) * gain * gain * (float(radar.wavelength_m) ** 2) * rcs_m2
    denominator = ((4.0 * math.pi) ** 3) * loss * noise_w * snr_lin
    if denominator <= 0.0 or numerator <= 0.0:
        return 10000.0
    return (numerator / denominator) ** 0.25


def _radar_default_max_range(radar: Any, targets: Sequence[Any]) -> float:
    """取场景中最难与最易探测的目标之间的一档，作为默认作用距离。"""
    rcs_list = [float(getattr(t, "rcs_m2", 1.0)) for t in targets] or [1.0]
    typical = sorted(rcs_list)[len(rcs_list) // 2]  # 中位 RCS
    return float(radar_detection_range_m(radar, typical))


def build_default_suite(
    scene: Any, seed: int = 0, occluders: Optional[Sequence[Dict[str, Any]]] = None
) -> SensorSuite:
    """按场景里的雷达/侦察机自动生成一套传感器（无配置时的默认）。"""
    sensors: List[Sensor] = []

    for radar in scene.radars:
        range_m = _radar_default_max_range(radar, scene.targets)
        sensors.append(RadarSensor(SensorConfig(
            sensor_id=f"SENSOR_{radar.radar_id}",
            mounting_id=radar.radar_id,
            sensor_kind=KIND_RADAR_SENSOR,
            min_range_m=0.0,
            max_range_m=range_m,
            az_fov_deg=60.0,
            el_fov_deg=30.0,
            update_period_s=1.0,
            range_sigma_rel=DEFAULT_RADAR_SIGMA_REL,
            range_sigma_abs_m=DEFAULT_RADAR_SIGMA_ABS_M,
            az_sigma_deg=DEFAULT_RADAR_AZ_SIGMA_DEG,
            el_sigma_deg=DEFAULT_RADAR_EL_SIGMA_DEG,
            range_rate_sigma_mps=DEFAULT_RADAR_VR_SIGMA_MPS,
            snr50_db=float(radar.snr50_db),
            pd_slope_db=float(radar.pd_slope_db),
            false_alarm_rate=0.0,
            tx_power_w=float(radar.tx_power_w),
            peak_gain_db=float(radar.peak_gain_db),
            wavelength_m=float(radar.wavelength_m),
            bandwidth_hz=float(radar.bandwidth_hz),
            noise_figure_db=float(radar.noise_figure_db),
            system_loss_db=float(radar.system_loss_db),
            temperature_k=float(radar.temperature_k),
            observes_kind="target",
            provides_range=True,
            seed=seed,
        )))

    for esm in scene.interceptors:
        sensors.append(EsmSensor(SensorConfig(
            sensor_id=f"SENSOR_{esm.interceptor_id}",
            mounting_id=esm.interceptor_id,
            sensor_kind=KIND_ESM_SENSOR,
            min_range_m=0.0,
            max_range_m=5.0e5,
            az_fov_deg=180.0,
            el_fov_deg=90.0,
            update_period_s=1.0,
            az_sigma_deg=DEFAULT_ESM_AZ_SIGMA_DEG,
            el_sigma_deg=DEFAULT_ESM_EL_SIGMA_DEG,
            snr50_db=float(esm.snr50_db),
            pd_slope_db=float(esm.pint_slope_db),
            false_alarm_rate=0.0,
            peak_gain_db=float(esm.gain_db),   # ESM 的接收增益
            wavelength_m=0.1,
            bandwidth_hz=float(esm.bandwidth_hz),
            noise_figure_db=float(esm.noise_figure_db),
            system_loss_db=float(esm.system_loss_db),
            temperature_k=float(esm.temperature_k),
            observes_kind="radar",             # 被动传感器观测的是**雷达辐射**
            provides_range=False,              # 单站被动：没有距离量测
            seed=seed,
        )))

    return SensorSuite(sensors, _occlusion_model(occluders))


def build_suite_from_config(
    scene: Any,
    data: Dict[str, Any],
    seed: int = 0,
    noise_scale: float = 1.0,
) -> SensorSuite:
    """按配置构建传感器套件。

    `noise_scale`：统一缩放全部量测噪声（0 = 理想测量）。
    这是 "ideal-measurement / realistic-measurement" 两组对照的实现方式：
    **同一套可见性约束**（作用距离/视场/遮挡/周期），只把噪声与虚警关掉。
    """
    raw_sensors = data.get("sensors")
    occluders = data.get("occluders")

    if not raw_sensors:
        suite = build_default_suite(scene, seed=seed, occluders=occluders)
    else:
        sensors: List[Sensor] = []
        for item in raw_sensors:
            item = dict(item)
            item.setdefault("seed", seed)
            config = SensorConfig.from_dict(item)
            cls = EsmSensor if config.sensor_kind == KIND_ESM_SENSOR else RadarSensor
            if not config.provides_range and cls is RadarSensor:
                # 配置里没写 provides_range 时，按类型给默认值
                config.provides_range = config.sensor_kind != KIND_ESM_SENSOR
            sensors.append(cls(config))
        suite = SensorSuite(sensors, _occlusion_model(occluders))

    if noise_scale != 1.0:
        apply_noise_scale(suite, noise_scale)
    return suite


def apply_noise_scale(suite: SensorSuite, scale: float) -> None:
    """统一缩放套件内全部量测噪声与虚警率；`scale=0` 同时关闭概率漏检。

    为什么放在传感器层而不是环境层：这样 "理想测量" 与 "真实测量" 共享
    **完全相同** 的可见性判定代码路径（作用距离/视场/遮挡/更新周期），
    差异**只有**噪声、虚警与概率漏检这三项。
    如果改成"另写一个无噪声的传感器类"，两条路径迟早会漂移，
    对照实验就失去意义。

    `scale == 0` 时把 `force_detection` 也打开：理想传感器**只要几何上看得见
    就一定检测得到**，否则"理想测量"里还混着概率漏检，三组对照就不是
    干净的信息阶梯（全真值 → 仅可见性受限 → 再叠加测量不完美）。
    """
    for sensor in suite.sensors:
        cfg = sensor.config
        cfg.range_sigma_rel *= scale
        cfg.range_sigma_abs_m *= scale
        cfg.az_sigma_deg *= scale
        cfg.el_sigma_deg *= scale
        cfg.range_rate_sigma_mps *= scale
        cfg.false_alarm_rate *= scale
        if scale <= 0.0:
            cfg.force_detection = True
