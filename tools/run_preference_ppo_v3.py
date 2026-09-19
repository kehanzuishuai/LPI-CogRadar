"""v3 Preference-Conditioned PPO：只验证 FiLM 表示，不提供 test-v5 入口。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from resource_management.contract_v1 import contract_digest, verify_frozen
from resource_management.learning_protocol import split_digest
from rl_resource.ablation import FrozenBaseline, assert_frozen
from rl_resource.actions import ACTION_NAMES
from rl_resource.env import CentralizedResourceSchedulingEnv, EnvConfig
from rl_resource.policy import ActorCritic, PolicyConfig
from rl_resource.train import TrainConfig, evaluate, train

CFG = os.path.join(ROOT, "config", "preference_ppo_v3.json")
CFG_SHA = CFG.replace(".json", ".sha256")
SPLIT = os.path.join(ROOT, "config", "preference_ppo_v3_splits.json")
SPLIT_SHA = SPLIT.replace(".json", ".sha256")
AUDIT = os.path.join(ROOT, "config", "preference_controllability_audit_v1.json")
V2_REPORT = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2",
                         "validation_report.json")
OUT = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v3")
ACTION_BY_NAME = {name: index for index, name in enumerate(ACTION_NAMES)}


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def _dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _freeze_path() -> str:
    return os.path.join(OUT, "training_freeze.json")


def _checked() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    protocol, registry, audit = _load(CFG), _load(SPLIT), _load(AUDIT)
    if _digest(CFG) != _read_text(CFG_SHA):
        raise RuntimeError("preference_ppo_v3 协议摘要不匹配")
    if _digest(SPLIT) != _read_text(SPLIT_SHA):
        raise RuntimeError("preference_ppo_v3 分区摘要不匹配")
    if _digest(AUDIT) != protocol["controllability_audit"]["sha256"]:
        raise RuntimeError("固定 controllability audit 摘要不匹配")
    if not registry.get("test_sealed") or not registry.get("test_v5_sealed"):
        raise RuntimeError("test-v5 必须封存")
    if not verify_frozen(strict=False)["ok"]:
        raise RuntimeError("resource-contract-v1 未冻结")
    for name, preference in protocol["preferences"].items():
        if (len(preference) != 5 or any(value < 0.0 for value in preference)
                or abs(sum(preference) - 1.0) > 1e-9):
            raise ValueError(f"{name} 不是五维 simplex")
    return protocol, registry, audit


def prepare() -> None:
    protocol, registry, audit = _checked()
    v2 = _load(V2_REPORT)
    v2_hashes = dict(v2.get("checkpoint_hashes") or {})
    if set(v2_hashes) != {"1009", "1013", "1019"}:
        raise RuntimeError("严格 concat-v2 基线 checkpoint 清单不完整")
    for seed, expected in v2_hashes.items():
        checkpoint = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2",
                                  f"seed_{seed}", "policy.pt")
        if _digest(checkpoint) != expected:
            raise RuntimeError(f"冻结 concat-v2 checkpoint {seed} 摘要不匹配")
    _dump(_freeze_path(), {
        "protocol_sha256": _digest(CFG), "split_sha256": _digest(SPLIT),
        "audit_sha256": _digest(AUDIT), "resource_contract": contract_digest(),
        "split_digest": split_digest(SPLIT), "preferences": protocol["preferences"],
        "training_seeds": protocol["training_seeds"], "train": registry["splits"]["train"],
        "validation": registry["splits"]["validation"], "test_v5": "sealed; not read or released",
        "strict_concat_v2_baseline_hashes": v2_hashes,
        "representation": protocol["conditioning"], "audit_states": audit["states"],
        "reward_mode": "preference_v2", "share_candidate_requires_track": False,
        "checkpoint_selection": protocol["checkpoint_selection"],
    })
    print(_freeze_path())


def _config(seed: int, preference: Sequence[float] | None = None) -> TrainConfig:
    freeze = _load(_freeze_path())
    if seed not in freeze["training_seeds"]:
        raise ValueError("训练 seed 不在冻结清单")
    baseline = FrozenBaseline(train_seed=seed)
    prefs = tuple(tuple(value) for value in freeze["preferences"].values())
    cfg = TrainConfig(
        scenarios=tuple(freeze["train"]["scenarios"]), seeds=tuple(freeze["train"]["seeds"]),
        episodes=baseline.episodes_per_update * baseline.updates, steps=baseline.steps,
        rollout_episodes=baseline.episodes_per_update, updates=baseline.updates,
        max_nodes=baseline.max_nodes, ppo=baseline.ppo(),
        policy=PolicyConfig(max_nodes=baseline.max_nodes, hidden_sizes=baseline.hidden_sizes,
                            activation=baseline.activation, seed=seed,
                            conditioning="film", preference_hidden_size=8),
        out_dir=OUT, tag=f"seed_{seed}", arm="main_baseline", reward_mode="preference_v2",
        preference_conditioned=True, preference_set=prefs,
        evaluation_preference=tuple(preference or freeze["preferences"]["balanced"]),
        eval_scenarios=tuple(freeze["validation"]["scenarios"]),
        eval_seeds=tuple(freeze["validation"]["seeds"]), seed=seed,
        scenario_registry_path=SPLIT, share_candidate_requires_track=False)
    assert_frozen(cfg, baseline)
    return cfg


def train_one(seed: int) -> None:
    summary = train(_config(seed), quiet=True)
    policy_path = summary["policy_path"]
    _dump(os.path.join(OUT, f"seed_{seed}", "metadata.json"), {
        "seed": seed, "checkpoint": policy_path, "sha256": _digest(policy_path),
        "selected_update": summary["selected_update"], "selection": "validation only",
        "policy": summary["policy"], "parameter_count": sum(
            value.numel() for value in ActorCritic.load(policy_path)[0].parameters()),
        "freeze_sha256": _digest(_freeze_path()), "test_v5": "sealed; not read or released",
    })
    print(seed)


def _make_env(state: Dict[str, Any], preference: Sequence[float]) -> CentralizedResourceSchedulingEnv:
    env = CentralizedResourceSchedulingEnv(EnvConfig(
        scenario=state["scenario"], seed=int(state["seed"]), steps=24,
        arm="main_baseline", reward_mode="preference_v2", preference=tuple(preference),
        preference_conditioned=True, scenario_registry_path=SPLIT,
        share_candidate_requires_track=False, keep_trace=False))
    _obs, _info = env.reset(seed=int(state["seed"]))
    for joint_action in state["pre_actions"]:
        _obs, _reward, terminated, truncated, _info = env.step(joint_action)
        if terminated or truncated:
            raise RuntimeError(f"固定状态 {state['id']} 在前置动作中提前结束")
    return env


def _policy_rows(models: Dict[str, ActorCritic], states: Iterable[Dict[str, Any]],
                 preferences: Dict[str, Sequence[float]], arm: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for seed, model in models.items():
        model.eval()
        for state in states:
            for name, preference in preferences.items():
                env = _make_env(state, preference)
                target = env.node_ids.index(state["target_node"])
                obs = torch.tensor([env._observation], dtype=torch.float32)
                logits, _value = model(obs)
                raw = logits[0, target]
                mask = torch.tensor([[env._mask[target]]], dtype=torch.bool)
                masked = model.masked_logits(logits[:, target:target + 1, :], mask)[0, 0]
                probabilities = torch.softmax(masked, dim=-1)
                rows.append({"arm": arm, "training_seed": seed, "state": state["id"],
                             "preference": name,
                             "legal_actions": [name for i, name in enumerate(ACTION_NAMES)
                                               if env._mask[target][i]],
                             "raw_logits": {name: float(raw[i]) for i, name in enumerate(ACTION_NAMES)},
                             "masked_probabilities": {name: (float(probabilities[i])
                                                               if env._mask[target][i] else None)
                                                      for i, name in enumerate(ACTION_NAMES)},
                             "masked_argmax": ACTION_NAMES[int(torch.argmax(masked))]})
    return rows


def _lookup(rows: Sequence[Dict[str, Any]], seed: str, state: str,
            preference: str) -> Dict[str, Any]:
    return next(row for row in rows if (row["training_seed"] == seed
                                        and row["state"] == state
                                        and row["preference"] == preference))


def _probability(row: Dict[str, Any], action: str) -> float:
    return float(row["masked_probabilities"].get(action) or 0.0)


def _sensitivity(rows: Sequence[Dict[str, Any]], seeds: Sequence[str]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for seed in seeds:
        remote_quality = _lookup(rows, seed, "stale_local_fresh_remote", "estimate_quality")
        remote_comm = _lookup(rows, seed, "stale_local_fresh_remote", "communication_saving")
        pressure_resource = _lookup(rows, seed, "resource_pressure", "resource_saving")
        pressure_completion = _lookup(rows, seed, "resource_pressure", "completion")
        sample_resource = _lookup(rows, seed, "sample_legal", "resource_saving")
        sample_completion = _lookup(rows, seed, "sample_legal", "completion")
        remote_collab_q = _probability(remote_quality, "share") + _probability(remote_quality, "process")
        remote_collab_c = _probability(remote_comm, "share") + _probability(remote_comm, "process")
        logit_l1 = sum(abs(remote_quality["raw_logits"][name] - remote_comm["raw_logits"][name])
                       for name in ACTION_NAMES)
        checks[seed] = {
            "remote_quality_collaboration_probability": remote_collab_q,
            "remote_communication_collaboration_probability": remote_collab_c,
            "remote_quality_share_probability": _probability(remote_quality, "share"),
            "remote_communication_share_probability": _probability(remote_comm, "share"),
            "resource_sample_probability": _probability(pressure_resource, "sample"),
            "completion_sample_probability": _probability(pressure_completion, "sample"),
            "resource_service_probability": sum(_probability(sample_resource, action)
                                                for action in ("sample", "process", "share")),
            "completion_service_probability": sum(_probability(sample_completion, action)
                                                  for action in ("sample", "process", "share")),
            "remote_quality_vs_communication_raw_logit_l1": logit_l1,
            "quality_collaboration_increases": remote_collab_q > remote_collab_c,
            "communication_share_decreases": (_probability(remote_comm, "share")
                                                < _probability(remote_quality, "share")),
            "resource_sample_decreases": (_probability(pressure_resource, "sample")
                                          < _probability(pressure_completion, "sample")),
            "completion_service_increases": (sum(_probability(sample_completion, action)
                                                 for action in ("sample", "process", "share"))
                                           > sum(_probability(sample_resource, action)
                                                 for action in ("sample", "process", "share"))),
            "nonzero_logit_response": logit_l1 > 1e-6,
        }
    required = ("quality_collaboration_increases", "communication_share_decreases",
                "resource_sample_decreases", "completion_service_increases",
                "nonzero_logit_response")
    return {"per_seed": checks, "requirements": list(required),
            "passed": all(all(values[key] for key in required) for values in checks.values())}


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _validation_rows(models: Dict[str, ActorCritic], freeze: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for seed, model in models.items():
        for name, preference in freeze["preferences"].items():
            summary = evaluate(model, _config(int(seed), preference), torch.device("cpu"),
                               freeze["validation"]["scenarios"], freeze["validation"]["seeds"])
            rows.extend({"training_seed": seed, "preference": name, **row}
                        for row in summary["rows"])
    return rows


def report() -> None:
    freeze = _load(_freeze_path())
    models: Dict[str, ActorCritic] = {}
    hashes: Dict[str, str] = {}
    parameters: Dict[str, int] = {}
    for seed in freeze["training_seeds"]:
        metadata = _load(os.path.join(OUT, f"seed_{seed}", "metadata.json"))
        if _digest(metadata["checkpoint"]) != metadata["sha256"]:
            raise RuntimeError(f"FiLM checkpoint {seed} 摘要不匹配")
        model, _extra = ActorCritic.load(metadata["checkpoint"], map_location="cpu")
        if model.config.conditioning != "film":
            raise RuntimeError("v3 checkpoint 不是 FiLM 条件化策略")
        models[str(seed)] = model; hashes[str(seed)] = metadata["sha256"]
        parameters[str(seed)] = int(metadata["parameter_count"])
    v2_models: Dict[str, ActorCritic] = {}
    for seed, expected in freeze["strict_concat_v2_baseline_hashes"].items():
        path = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2", f"seed_{seed}", "policy.pt")
        if _digest(path) != expected:
            raise RuntimeError(f"concat-v2 checkpoint {seed} 摘要不匹配")
        v2_models[str(seed)], _extra = ActorCritic.load(path, map_location="cpu")
    film_policy = _policy_rows(models, freeze["audit_states"], freeze["preferences"], "film_v3")
    concat_policy = _policy_rows(v2_models, freeze["audit_states"], freeze["preferences"], "concat_v2")
    sensitivity = _sensitivity(film_policy, sorted(models))
    validation = _validation_rows(models, freeze)
    table = {}
    for preference in freeze["preferences"]:
        group = [row for row in validation if row["preference"] == preference]
        table[preference] = {key: _mean(group, key) for key in (
            "completion_rate", "timeliness", "estimate_quality", "resource_consumption",
            "comm_overhead_bytes", "mean_information_age_s", "mean_sigma_m")}
        table[preference]["actions"] = {action: sum(row["action_composition"][action]
                                                     for row in group)
                                        for action in ACTION_NAMES}
    result = {
        "protocol": "preference-conditioned-ppo-v3", "freeze": freeze,
        "scope": {"representation_only": True, "v2_reward_changed": False,
                  "runtime_executor_changed": False, "test_v5_read": False,
                  "test_v5_unsealed": False}, "film_checkpoint_hashes": hashes,
        "film_parameter_count": parameters,
        "strict_concat_v2_baseline_hashes": freeze["strict_concat_v2_baseline_hashes"],
        "policy_response_matrix": {"film_v3": film_policy, "concat_v2": concat_policy},
        "logit_and_action_sensitivity": sensitivity, "validation_rows": validation,
        "validation_preference_table": table,
        "mechanism_gate": {**sensitivity, "test_v5_action": (
            "sealed; no test-v5 read because gate failed" if not sensitivity["passed"]
            else "gate passed; a future separately authorized stage may request release")},
        "boundaries": ["No reward, action, mask, RuntimeExecutor, physical, or PPO-core change.",
                       "concat-v2 is a frozen strict baseline, not silently replaced.",
                       "No formal test or Pareto claim in this mechanism-only stage."],
    }
    _dump(os.path.join(OUT, "validation_report.json"), result)
    with open(os.path.join(OUT, "validation_raw.csv"), "w", encoding="utf-8-sig", newline="") as handle:
        fields = ["training_seed", "preference", "scenario", "seed", "completion_rate", "timeliness",
                  "estimate_quality", "resource_consumption", "comm_overhead_bytes",
                  "mean_information_age_s", "mean_sigma_m"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in validation])
    _dump(os.path.join(OUT, "policy_response_matrix.json"), result["policy_response_matrix"])
    _dump(os.path.join(OUT, "sensitivity.json"), sensitivity)
    print(json.dumps({"mechanism_gate": result["mechanism_gate"], "test_v5": "sealed"}, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 preference PPO v3；无 test-v5 命令")
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
