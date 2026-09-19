"""观测空间契约测试（v4.5 一致性验收）。

钉住两件事：

1. **有符号量必须保留符号**。`bearing_norm` / `elevation_norm` /
   `range_rate_norm` 在物理上有正负（目标可在机头左右任一侧、可接近可远离）。
   早先把整个观测空间声明成 `Box([0]*D, [1]*D)` 并逐步 `clip`，
   于是负值被**静默抹成 0**：不报错、不告警，只是把一个真实的方向量
   变成常数。默认场景里目标恰好都在机头正侧，所以这个缺陷长期没暴露。
2. **观测空间的上下界与编码器契约一致**：编码器按 `[-1, 1]` 输出，
   空间就必须按 `[-1, 1]` 接受，且 `clip` 不得破坏它。

不需要 torch。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec  # noqa: E402
from engine.env import (  # noqa: E402
    MEASUREMENT_MODES,
    OBSERVATION_FEATURES,
    SIGNED_OBSERVATION_FIELDS,
    observation_bounds,
)
from sensor.fusion import SLOT_FIELDS, fuse_measurements  # noqa: E402


class _Measurement:
    """最小测量对象（供 `fuse_measurements` 打包）。"""

    def __init__(self, bearing_deg: float, elevation_deg: float = 0.0,
                 range_rate_mps: float = 0.0, range_m: float = 5000.0) -> None:
        self.sensor_id = "S1"
        self.candidate_id = "C1"
        self.sensor_kind = "radar"
        self.time_s = 0.0
        self.range_m = range_m
        self.azimuth_deg = bearing_deg
        self.elevation_deg = elevation_deg
        self.range_rate_mps = range_rate_mps
        self.std_range_m = 50.0
        self.std_az_deg = 0.5
        self.std_el_deg = 0.5
        self.confidence = 1.0
        self.is_fresh = True
        self.age_s = 0.0


class TestSignedFieldDeclaration(unittest.TestCase):
    def test_signed_fields_are_declared(self) -> None:
        self.assertIn("bearing_norm", SIGNED_OBSERVATION_FIELDS)
        self.assertIn("elevation_norm", SIGNED_OBSERVATION_FIELDS)
        self.assertIn("range_rate_norm", SIGNED_OBSERVATION_FIELDS)

    def test_every_declared_field_exists_in_encoder(self) -> None:
        """登记的有符号维必须真的是编码器输出的槽位（防写错名字）。"""
        for name in SIGNED_OBSERVATION_FIELDS:
            self.assertIn(name, SLOT_FIELDS,
                          f"{name} 不在 sensor.fusion.SLOT_FIELDS 里")

    def test_bounds_are_per_field_not_uniform(self) -> None:
        low, high = observation_bounds(["present", "bearing_norm", "confidence"])
        self.assertEqual(low, [0.0, -1.0, 0.0])
        self.assertEqual(high, [1.0, 1.0, 1.0])

    def test_full_mode_has_no_signed_dims(self) -> None:
        """full 模式的 12 维全为非负量，旧契约不受影响。"""
        low, high = observation_bounds(OBSERVATION_FEATURES)
        self.assertEqual(low, [0.0] * len(OBSERVATION_FEATURES))
        self.assertEqual(high, [1.0] * len(OBSERVATION_FEATURES))


class TestEncoderProducesSignedValues(unittest.TestCase):
    def test_negative_bearing_is_encoded_as_negative(self) -> None:
        """目标在机头**负侧**时，编码器的方位槽必须是负值。"""
        fused = fuse_measurements([_Measurement(bearing_deg=-90.0)])
        vector = list(fused.vector)
        # 槽位顺序：present, range_norm, has_range, bearing_norm, ...
        self.assertLess(vector[3], 0.0,
                        "方位槽没有保留负号——编码器契约被破坏")
        self.assertAlmostEqual(vector[3], -0.5, places=6)

    def test_negative_range_rate_and_elevation(self) -> None:
        fused = fuse_measurements([_Measurement(
            bearing_deg=0.0, elevation_deg=-45.0, range_rate_mps=-100.0)])
        vector = list(fused.vector)
        self.assertAlmostEqual(vector[4], -0.5, places=6)   # 俯仰 / 90
        self.assertAlmostEqual(vector[5], -0.5, places=6)   # 径向速度 / 200


class TestObservationSpaceRoundTrip(unittest.TestCase):
    """正负值往返：编码 → 空间 clip → 取值，符号必须活下来。"""

    def _env(self, mode: str = "realistic"):
        env = ec.make_env(observation_mode=mode)
        env.reset(seed=0)
        return env

    def test_space_accepts_and_preserves_negative_values(self) -> None:
        env = self._env()
        space = env.observation_space
        names = env.observation_features
        signed = [i for i, name in enumerate(names)
                  if name in SIGNED_OBSERVATION_FIELDS]
        self.assertEqual(len(signed), 12, "4 个航迹槽 × 3 个有符号维")
        for index in signed:
            self.assertEqual(space.low[index], -1.0)
            self.assertEqual(space.high[index], 1.0)
        # 往返：-0.42 进去，-0.42 出来
        vector = [0.0] * len(names)
        for index in signed:
            vector[index] = -0.42
        clipped = space.clip(vector)
        for index in signed:
            self.assertAlmostEqual(clipped[index], -0.42, places=9)
        self.assertTrue(space.contains(vector))

    def test_encoder_output_survives_the_space(self) -> None:
        """把一个真实的负方位测量喂进环境，观测向量里符号必须还在。"""
        env = self._env()
        fused = fuse_measurements([_Measurement(bearing_deg=-90.0)])
        raw = list(fused.vector)
        padded = raw + [0.0] * (len(env.observation_features) - len(raw))
        clipped = env.observation_space.clip(padded)
        self.assertLess(clipped[3], 0.0,
                        "观测空间把负方位裁掉了——有符号量被破坏")

    def test_clipping_still_works_for_out_of_range(self) -> None:
        """保留符号不等于放弃裁剪：越界值仍要被夹到界内。"""
        env = self._env()
        space = env.observation_space
        names = env.observation_features
        signed = [i for i, name in enumerate(names)
                  if name in SIGNED_OBSERVATION_FIELDS]
        vector = [5.0] * len(names)
        for index in signed:
            vector[index] = -7.0
        clipped = space.clip(vector)
        self.assertEqual(clipped[0], 1.0)
        for index in signed:
            self.assertEqual(clipped[index], -1.0)

    def test_all_measurement_modes_have_signed_dims(self) -> None:
        for mode in MEASUREMENT_MODES:
            env = self._env(mode)
            negative = sum(1 for value in env.observation_space.low if value < 0)
            self.assertEqual(negative, 12, f"{mode} 模式的有符号维数不对")


if __name__ == "__main__":
    unittest.main(verbosity=2)
