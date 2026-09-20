"""Preference-Conditioned PPO v1--v3 的只读归档完整性检查。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCHIVES = {
    "v1": {
        "config": "config/preference_ppo_v1.json",
        "sidecar": "config/preference_ppo_v1.sha256",
        "output": "output/rl_resource/preference_ppo",
        "seeds": (907, 911, 919), "sealed_key": "test_v3",
    },
    "v2": {
        "config": "config/preference_ppo_v2.json",
        "sidecar": "config/preference_ppo_v2.sha256",
        "output": "output/rl_resource/preference_ppo_v2",
        "seeds": (1009, 1013, 1019), "sealed_key": "test_v4",
    },
    "v3": {
        "config": "config/preference_ppo_v3.json",
        "sidecar": "config/preference_ppo_v3.sha256",
        "output": "output/rl_resource/preference_ppo_v3",
        "seeds": (1103, 1109, 1117), "sealed_key": "test_v5",
    },
}


def _path(relative: str) -> str:
    return os.path.join(ROOT, relative)


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def resolve_checkpoint(metadata_path: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """解析历史 checkpoint 路径，兼容已搬迁的 Windows 绝对路径。

    metadata 中已记录的路径仍优先；只有该路径不存在时，才从同一 seed 的
    metadata 目录寻找固定文件名 ``policy.pt``。回退路径随后仍必须通过 metadata
    内的既有 SHA-256 校验，故不会重新选择 checkpoint 或改变归档结论。
    """
    recorded = str(metadata.get("checkpoint", ""))
    if recorded and os.path.isfile(recorded):
        return {"path": recorded, "resolution": "metadata_recorded_path"}
    fallback = os.path.join(os.path.dirname(os.path.abspath(metadata_path)), "policy.pt")
    if os.path.isfile(fallback):
        return {
            "path": fallback,
            "resolution": "metadata_seed_directory_policy_pt_fallback",
        }
    raise RuntimeError(
        "checkpoint 不存在：metadata 路径 "
        f"{recorded!r}，seed 目录回退 {fallback!r} 也不存在"
    )


def verify_archive() -> Dict[str, Any]:
    """验证冻结配置、每个 checkpoint、负结果与基础 PPO 正结论都未被改写。"""
    result: Dict[str, Any] = {"versions": {}, "test_v5": "sealed"}
    for version, spec in ARCHIVES.items():
        config_path, sidecar_path = _path(spec["config"]), _path(spec["sidecar"])
        if _digest(config_path) != _text(sidecar_path):
            raise RuntimeError(f"{version} protocol SHA 不匹配")
        output = _path(spec["output"])
        freeze = _load(os.path.join(output, "training_freeze.json"))
        report = _load(os.path.join(output, "validation_report.json"))
        if freeze.get("protocol_sha256") != _digest(config_path):
            raise RuntimeError(f"{version} training freeze 未绑定当前协议")
        if report.get("mechanism_gate", {}).get("passed") is not False:
            raise RuntimeError(f"{version} 负结果被改写为通过")
        sealed = str(freeze.get(spec["sealed_key"], report.get(spec["sealed_key"], "")))
        if "sealed" not in sealed:
            raise RuntimeError(f"{version} 的测试封存状态不正确：{sealed!r}")
        checkpoints = {}
        checkpoint_resolution = {}
        for seed in spec["seeds"]:
            metadata_path = os.path.join(output, f"seed_{seed}", "metadata.json")
            metadata = _load(metadata_path)
            resolved = resolve_checkpoint(metadata_path, metadata)
            checkpoint = resolved["path"]
            if not os.path.isfile(checkpoint) or _digest(checkpoint) != metadata["sha256"]:
                raise RuntimeError(f"{version} seed {seed} checkpoint SHA 不匹配")
            checkpoints[str(seed)] = metadata["sha256"]
            checkpoint_resolution[str(seed)] = resolved["resolution"]
        result["versions"][version] = {
            "protocol_sha256": _digest(config_path), "checkpoint_hashes": checkpoints,
            "checkpoint_resolution": checkpoint_resolution,
            "mechanism_gate": "failed", "test_status": sealed,
        }

    split = _load(_path("config/preference_ppo_v3_splits.json"))
    v3 = _load(os.path.join(_path(ARCHIVES["v3"]["output"]), "validation_report.json"))
    if not split.get("test_v5_sealed") or v3.get("scope", {}).get("test_v5_read") is not False:
        raise RuntimeError("test-v5 被读取或不再封存")
    baseline = _load(_path("output/rl_resource/multiseed_v2/final_release.json"))
    conclusion = baseline.get("conclusions", {}).get("baseline_completion_vs_rule", "")
    if "reproduced" not in conclusion or "positive" not in conclusion:
        raise RuntimeError("多训练 seed 的基础 PPO 完成率结论被改写")
    result["baseline_ppo"] = {
        "status": "preserved",
        "conclusion": conclusion,
        "release": "output/rl_resource/multiseed_v2/final_release.json",
    }
    return result


def main() -> int:
    print(json.dumps(verify_archive(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
