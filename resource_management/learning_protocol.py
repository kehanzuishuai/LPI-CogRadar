"""训练/验证/测试划分的读取、校验与测试集封存。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple


DEFAULT_SPLIT_PATH = os.path.join("config", "learning_splits_v1.json")
DEFAULT_DIGEST_PATH = os.path.join("config", "learning_splits_v1.sha256")


class SealedTestSplitError(PermissionError):
    """在最终评测闸门打开前请求读取测试集。"""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    scenarios: Tuple[str, ...]
    seeds: Tuple[int, ...]


def split_digest(path: str = DEFAULT_SPLIT_PATH) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def verify_split_digest(
    path: str = DEFAULT_SPLIT_PATH,
    digest_path: str = DEFAULT_DIGEST_PATH,
) -> str:
    """校验数据划分文件未在实验过程中被静默改写，并返回摘要。"""
    with open(digest_path, "r", encoding="ascii") as handle:
        expected = handle.read().strip().lower()
    actual = split_digest(path)
    if actual != expected:
        raise ValueError(
            f"学习数据划分摘要不一致：expected={expected}, actual={actual}"
        )
    return actual


def load_registry(path: str = DEFAULT_SPLIT_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    return registry


def validate_registry(registry: Dict[str, Any]) -> None:
    splits = registry.get("splits") or {}
    catalog = registry.get("scenario_catalog") or {}
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"splits 必须恰为 {sorted(required)}")
    seed_sets = {name: set(value.get("seeds") or []) for name, value in splits.items()}
    scenario_sets = {
        name: set(value.get("scenarios") or []) for name, value in splits.items()
    }
    for name in required:
        if not seed_sets[name] or not scenario_sets[name]:
            raise ValueError(f"{name} 的场景与种子均不能为空")
        if len(seed_sets[name]) != len(splits[name]["seeds"]):
            raise ValueError(f"{name} 内部种子重复")
    referenced_scenarios = set().union(*scenario_sets.values())
    if set(catalog) != referenced_scenarios:
        missing = sorted(referenced_scenarios - set(catalog))
        extra = sorted(set(catalog) - referenced_scenarios)
        raise ValueError(f"场景目录与划分不一致：missing={missing}, extra={extra}")
    required_scenario_fields = {
        "load_multiplier", "budget_multiplier", "comm_delay_ticks",
        "comm_drop_probability", "node_outage_windows",
        "sensor_bias_sigma_multiplier",
    }
    for scenario_name, spec in catalog.items():
        if set(spec) != required_scenario_fields:
            raise ValueError(f"{scenario_name} 的场景字段不完整")
        if float(spec["load_multiplier"]) <= 0.0:
            raise ValueError(f"{scenario_name}: load_multiplier 必须为正")
        if float(spec["budget_multiplier"]) <= 0.0:
            raise ValueError(f"{scenario_name}: budget_multiplier 必须为正")
        if int(spec["comm_delay_ticks"]) < 0:
            raise ValueError(f"{scenario_name}: comm_delay_ticks 不能为负")
        drop = float(spec["comm_drop_probability"])
        if not 0.0 <= drop <= 1.0:
            raise ValueError(f"{scenario_name}: comm_drop_probability 越界")
        if float(spec["sensor_bias_sigma_multiplier"]) < 0.0:
            raise ValueError(f"{scenario_name}: sensor bias 不能为负")
        for window in spec["node_outage_windows"]:
            if len(window) != 2 or int(window[0]) < 0 or int(window[1]) <= int(window[0]):
                raise ValueError(f"{scenario_name}: 节点中断窗口非法")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if seed_sets[left] & seed_sets[right]:
            raise ValueError(f"{left}/{right} 种子重叠")
        if scenario_sets[left] & scenario_sets[right]:
            raise ValueError(f"{left}/{right} 场景重叠")
    if not bool(registry.get("test_sealed", False)):
        raise ValueError("test_sealed 必须为 true")


def get_split(
    name: str,
    *,
    path: str = DEFAULT_SPLIT_PATH,
    release_test: bool = False,
) -> SplitSpec:
    registry = load_registry(path)
    if name == "test" and not release_test:
        raise SealedTestSplitError(
            "测试集已封存：训练、调参与 checkpoint 选择阶段不得读取；"
            "仅最终评测显式传 release_test=True"
        )
    if name not in registry["splits"]:
        raise KeyError(f"未知 split {name!r}")
    raw = registry["splits"][name]
    return SplitSpec(
        name=name,
        scenarios=tuple(str(item) for item in raw["scenarios"]),
        seeds=tuple(int(item) for item in raw["seeds"]),
    )
