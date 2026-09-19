"""v2 Preference-Conditioned PPO：先冻结，后训练/validation；没有 test-v4 入口。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from resource_management.contract_v1 import contract_digest, verify_frozen
from resource_management.learning_protocol import load_registry, split_digest
from rl_resource.ablation import FrozenBaseline, assert_frozen
from rl_resource.policy import ActorCritic
from rl_resource.train import TrainConfig, evaluate, train

CFG = os.path.join(ROOT, "config", "preference_ppo_v2.json")
CFG_SHA = CFG.replace(".json", ".sha256")
SPLIT = os.path.join(ROOT, "config", "preference_ppo_v2_splits.json")
SPLIT_SHA = os.path.join(ROOT, "config", "preference_ppo_v2_splits.sha256")
BALANCED = os.path.join(ROOT, "config", "balanced_protocol_v1.json")
OUT = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2")


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _freeze_path() -> str:
    return os.path.join(OUT, "training_freeze.json")


def _checked() -> tuple[Dict[str, Any], Dict[str, Any]]:
    protocol, registry = _load(CFG), _load(SPLIT)
    if _digest(CFG) != open(CFG_SHA, encoding="utf-8").read().strip():
        raise RuntimeError("preference_ppo_v2 协议摘要不匹配")
    if _digest(SPLIT) != open(SPLIT_SHA, encoding="utf-8").read().strip():
        raise RuntimeError("test-v4 划分摘要不匹配")
    if not registry.get("test_v4_sealed"):
        raise RuntimeError("test-v4 必须先封存")
    if not verify_frozen(strict=False)["ok"]:
        raise RuntimeError("resource-contract-v1 未冻结")
    for name, preference in protocol["preferences"].items():
        if (len(preference) != 5 or any(value < 0.0 for value in preference)
                or abs(sum(preference) - 1.0) > 1e-9):
            raise ValueError(f"{name} 不是五维 simplex")
    return protocol, registry


def prepare() -> None:
    protocol, registry = _checked()
    _dump(_freeze_path(), {
        "protocol_sha256": _digest(CFG), "split_sha256": _digest(SPLIT),
        "balanced_protocol_sha256": _digest(BALANCED),
        "resource_contract": contract_digest(), "split_digest": split_digest(SPLIT),
        "preferences": protocol["preferences"], "training_seeds": protocol["training_seeds"],
        "train": registry["splits"]["train"], "validation": registry["splits"]["validation"],
        "test_v4": "sealed; not read or released", "share_candidate_requires_track": False,
        "reward_mode": "preference_v2",
    })
    print(_freeze_path())


def _config(seed: int, preference: Sequence[float] | None = None) -> TrainConfig:
    freeze = _load(_freeze_path())
    if seed not in freeze["training_seeds"]:
        raise ValueError("训练 seed 不在冻结清单")
    baseline = FrozenBaseline(train_seed=seed)
    prefs = tuple(tuple(value) for value in freeze["preferences"].values())
    selected = tuple(preference or freeze["preferences"]["balanced"])
    config = TrainConfig(
        scenarios=tuple(freeze["train"]["scenarios"]), seeds=tuple(freeze["train"]["seeds"]),
        episodes=baseline.episodes_per_update * baseline.updates, steps=baseline.steps,
        rollout_episodes=baseline.episodes_per_update, updates=baseline.updates,
        max_nodes=baseline.max_nodes, ppo=baseline.ppo(), policy=baseline.policy(0),
        out_dir=OUT, tag=f"seed_{seed}", arm="main_baseline", reward_mode="preference_v2",
        preference_conditioned=True, preference_set=prefs, evaluation_preference=selected,
        eval_scenarios=tuple(freeze["validation"]["scenarios"]),
        eval_seeds=tuple(freeze["validation"]["seeds"]), seed=seed,
        scenario_registry_path=SPLIT, share_candidate_requires_track=False)
    # v2 仍沿用 v1 的网络/PPO/预算；仅 reward、候选语义和 registry 被协议允许改变。
    assert_frozen(config, baseline)
    return config


def train_one(seed: int) -> None:
    freeze = _load(_freeze_path())
    summary = train(_config(seed), quiet=True)
    _dump(os.path.join(OUT, f"seed_{seed}", "metadata.json"), {
        "seed": seed, "checkpoint": summary["policy_path"],
        "sha256": _digest(summary["policy_path"]),
        "selected_update": summary["selected_update"],
        "freeze_sha256": _digest(_freeze_path()), "selection": "validation only",
        "test_v4": "sealed; not read or released",
    })
    print(seed)


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def report() -> None:
    freeze = _load(_freeze_path())
    rows = []
    checkpoint_hashes = {}
    for seed in freeze["training_seeds"]:
        metadata = _load(os.path.join(OUT, f"seed_{seed}", "metadata.json"))
        if _digest(metadata["checkpoint"]) != metadata["sha256"]:
            raise RuntimeError(f"seed {seed} checkpoint 已改变")
        checkpoint_hashes[str(seed)] = metadata["sha256"]
        model, _ = ActorCritic.load(metadata["checkpoint"], map_location="cpu")
        for name, preference in freeze["preferences"].items():
            summary = evaluate(model, _config(seed, preference), torch.device("cpu"),
                               freeze["validation"]["scenarios"], freeze["validation"]["seeds"])
            rows.extend({"training_seed": seed, "preference": name, **row}
                        for row in summary["rows"])

    table: Dict[str, Dict[str, Any]] = {}
    for name in freeze["preferences"]:
        group = [row for row in rows if row["preference"] == name]
        table[name] = {key: _mean(group, key) for key in (
            "completion_rate", "timeliness", "estimate_quality", "mean_information_age_s",
            "mean_sigma_m", "resource_consumption", "comm_overhead_bytes", "mean_waiting_s", "n_expired")}
        table[name]["actions"] = {action: sum(row["action_composition"][action] for row in group)
                                  for action in ("idle", "sample", "process", "share")}
        table[name]["information_age_state"] = (
            "observed" if any(row.get("information_age_state") == "observed" for row in group)
            else "not_applicable")
        table[name]["sigma_state"] = (
            "observed" if any(row.get("sigma_state") == "observed" for row in group)
            else "not_applicable")
    per_seed = {}
    for seed in freeze["training_seeds"]:
        per_seed[str(seed)] = {name: sum(row["action_composition"]["share"]
                                          for row in rows if row["training_seed"] == seed
                                          and row["preference"] == name)
                               for name in freeze["preferences"]}
    handover = {name: sum(row["action_composition"]["share"] for row in rows
                          if row["preference"] == name
                          and row["scenario"] == "rm_validation_handover")
                for name in freeze["preferences"]}
    gate = {
        "each_seed_has_legal_share_argmax": all(any(count > 0 for count in values.values())
                                                  for values in per_seed.values()),
        "communication_direction": (table["communication_saving"]["actions"]["share"]
                                      <= table["estimate_quality"]["actions"]["share"]
                                      and table["communication_saving"]["comm_overhead_bytes"]
                                      <= table["estimate_quality"]["comm_overhead_bytes"]),
        "handover_quality_direction": handover["estimate_quality"] > handover["communication_saving"],
    }
    gate["passed"] = all(gate.values())
    def dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        high = ("completion_rate", "timeliness", "estimate_quality")
        low = ("resource_consumption", "comm_overhead_bytes")
        return (all(left[key] >= right[key] for key in high)
                and all(left[key] <= right[key] for key in low)
                and any(left[key] != right[key] for key in high + low))
    pareto = [name for name, value in table.items()
              if not any(dominates(other, value) for other_name, other in table.items()
                         if other_name != name)]
    result = {"freeze": freeze, "checkpoint_hashes": checkpoint_hashes,
              "rows": rows, "preference_table": table,
              "per_seed_share_argmax": per_seed, "handover_share_actions": handover,
              "mechanism_gate": gate, "validation_pareto_nondominated": pareto,
              "test_v4": "sealed; not read or released",
              "metric_semantics": {
                  "mean_information_age_s": "null + not_applicable means no observed track; it is never best",
                  "mean_sigma_m": "null + not_applicable means no observed covariance; it is never best"},
              "boundaries": ["No single score/ranking.", "v1 failure is retained unchanged.",
                             "If gate fails, do not release test-v4 or tune v2."]}
    _dump(os.path.join(OUT, "validation_report.json"), result)
    fields = ["training_seed", "preference", "scenario", "seed", "completion_rate", "timeliness",
              "estimate_quality", "mean_information_age_s", "mean_sigma_m", "resource_consumption",
              "comm_overhead_bytes", "mean_waiting_s", "n_expired"]
    with open(os.path.join(OUT, "validation_raw.csv"), "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    print(json.dumps({"rows": len(rows), "mechanism_gate": gate}, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结的 preference PPO v2；无 test-v4 命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train_parser = sub.add_parser("train"); train_parser.add_argument("--seed", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "prepare": prepare()
    elif args.command == "train": train_one(args.seed)
    else: report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
