"""只读 Preference Controllability Audit：不训练、不改 v2、绝不访问 test-v4。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from resource_management.units import BUDGET_UNITS
from rl_resource.actions import ACTION_IDLE, ACTION_NAMES, ACTION_PROCESS, ACTION_SAMPLE, ACTION_SHARE
from rl_resource.env import CentralizedResourceSchedulingEnv, EnvConfig
from rl_resource.policy import ActorCritic

AUDIT = os.path.join(ROOT, "config", "preference_controllability_audit_v1.json")
V2 = os.path.join(ROOT, "config", "preference_ppo_v2.json")
SPLIT = os.path.join(ROOT, "config", "preference_ppo_v2_splits.json")
CHECKPOINT_REPORT = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2", "validation_report.json")
OUT = os.path.join(ROOT, "output", "preference_controllability_audit")
ACTION_BY_NAME = {name: index for index, name in enumerate(ACTION_NAMES)}
UTILITY_NAMES = ("completion", "timeliness", "estimate_quality", "resource_saving", "communication_saving")
BALANCED = (0.2, 0.2, 0.2, 0.2, 0.2)


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


def _assert_frozen(audit: Dict[str, Any]) -> Dict[str, Any]:
    if _digest(V2) != audit["v2_protocol_sha256"]:
        raise RuntimeError("v2 协议摘要不匹配；审计拒绝继续")
    if _digest(SPLIT) != audit["v2_split_sha256"]:
        raise RuntimeError("v2 划分摘要不匹配；审计拒绝继续")
    registry = _load(SPLIT)
    if not registry.get("test_v4_sealed") or audit.get("test_v4") != "sealed; prohibited":
        raise RuntimeError("test-v4 必须封存且审计禁止读取")
    return _load(V2)


def _make_env(state: Dict[str, Any], preference: Sequence[float]) -> CentralizedResourceSchedulingEnv:
    env = CentralizedResourceSchedulingEnv(EnvConfig(
        scenario=state["scenario"], seed=int(state["seed"]), steps=24,
        arm="main_baseline", reward_mode="preference_v2",
        preference=tuple(preference), preference_conditioned=True,
        scenario_registry_path=SPLIT, share_candidate_requires_track=False,
        keep_trace=False))
    _obs, info = env.reset(seed=int(state["seed"]))
    for joint_action in state["pre_actions"]:
        _obs, _reward, terminated, truncated, info = env.step(joint_action)
        if terminated or truncated:
            raise RuntimeError(f"固定状态 {state['id']} 在前置动作中提前结束")
    return env


def _target_index(env: CentralizedResourceSchedulingEnv, state: Dict[str, Any]) -> int:
    return env.node_ids.index(state["target_node"])


def _legal_actions(env: CentralizedResourceSchedulingEnv, state: Dict[str, Any]) -> list[str]:
    index = _target_index(env, state)
    return [name for action, name in enumerate(ACTION_NAMES) if env._mask[index][action]]


def _state_snapshot(env: CentralizedResourceSchedulingEnv, state: Dict[str, Any]) -> Dict[str, Any]:
    index = _target_index(env, state)
    context = env.action_context()
    node_id = state["target_node"]
    remaining = context.remaining[node_id]
    capacity = env._executor.node(node_id).budget.capacity
    ratios = [float(remaining[unit]) / float(capacity[unit])
              for unit in BUDGET_UNITS if float(capacity[unit]) > 0]
    now = env._driver.current_time_s
    node_information = {}
    for current_node, center in env._runtime.centers.items():
        ages = [max(0.0, now - track.last_measurement_time) for track in center.tracks]
        node_information[current_node] = {
            "n_tracks": len(center.tracks),
            "mean_information_age_s": sum(ages) / len(ages) if ages else None,
            "information_age_state": "observed" if ages else "not_applicable"}
    target_information = node_information[node_id]
    return {"node_id": node_id, "time_s": now, "legal_actions": _legal_actions(env, state),
            "shareable_outbox": len(env._runtime.share_outbox[node_id]),
            "processable": context.processable[node_id],
            "min_remaining_resource_ratio": min(ratios) if ratios else None,
            "n_tracks": target_information["n_tracks"],
            "mean_information_age_s": target_information["mean_information_age_s"],
            "information_age_state": target_information["information_age_state"],
            "node_information": node_information}


def _followup_action(env: CentralizedResourceSchedulingEnv) -> list[int]:
    # 固定且与偏好/model 无关，避免把后续策略差异误归给首动作。
    priority = (ACTION_PROCESS, ACTION_SHARE, ACTION_SAMPLE, ACTION_IDLE)
    return [next(action for action in priority if row[action])
            for row in env._mask[:len(env.node_ids)]]


def _raw_utilities(components: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(components.get("preference_v2_" + name, 0.0)) / 0.2
            for name in UTILITY_NAMES}


def _rollout(state: Dict[str, Any], preference: Sequence[float], first_action: int,
             horizon: int) -> Dict[str, Any]:
    env = _make_env(state, preference)
    target = _target_index(env, state)
    if not env._mask[target][first_action]:
        return {"status": "not_applicable", "reason": "action_illegal_in_fixed_state"}
    joint = [ACTION_IDLE] * len(env.node_ids)
    joint[target] = first_action
    rewards, raw_steps, action_trace = [], [], []
    for index in range(horizon):
        action = joint if index == 0 else _followup_action(env)
        _obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        action_trace.append([ACTION_NAMES[value] for value in action])
        if tuple(preference) == BALANCED:
            raw_steps.append(_raw_utilities(info["reward_components"]))
        if terminated or truncated:
            break
    gamma = env.config.gamma
    return {"status": "ok", "actions": action_trace, "step_rewards": rewards,
            "discounted_return": sum((gamma ** i) * value for i, value in enumerate(rewards)),
            "raw_utility_steps": raw_steps,
            "raw_utility_discounted": {name: sum((gamma ** i) * row[name]
                                                  for i, row in enumerate(raw_steps))
                                        for name in UTILITY_NAMES} if raw_steps else None,
            "conservation_ok": env._executor.conservation_report()["all_conserved"],
            "test_v4_read": False}


def _policy_matrix(states: Iterable[Dict[str, Any]], preferences: Dict[str, Sequence[float]],
                   checkpoint_hashes: Dict[str, str]) -> Dict[str, Any]:
    rows = []
    for seed, expected_hash in checkpoint_hashes.items():
        checkpoint = os.path.join(ROOT, "output", "rl_resource", "preference_ppo_v2",
                                  f"seed_{seed}", "policy.pt")
        if _digest(checkpoint) != expected_hash:
            raise RuntimeError(f"checkpoint {seed} 摘要不匹配")
        model, _ = ActorCritic.load(checkpoint, map_location="cpu")
        model.eval()
        for state in states:
            for name, preference in preferences.items():
                env = _make_env(state, preference)
                target = _target_index(env, state)
                values = torch.tensor([env._observation], dtype=torch.float32)
                logits, _value = model(values)
                raw = logits[0, target]
                masked = model.masked_logits(logits[:, target:target + 1, :],
                                             torch.tensor([[env._mask[target]]], dtype=torch.bool))[0, 0]
                probabilities = torch.softmax(masked, dim=-1)
                rows.append({"training_seed": int(seed), "state": state["id"],
                             "preference": name, "legal_actions": _legal_actions(env, state),
                             "raw_logits": {action: float(raw[index]) for index, action in enumerate(ACTION_NAMES)},
                             "masked_probabilities": {action: (float(probabilities[index])
                                                             if env._mask[target][index] else None)
                                                      for index, action in enumerate(ACTION_NAMES)},
                             "masked_argmax": ACTION_NAMES[int(torch.argmax(masked))]})
    return {"rows": rows}


def _direction_checks(matrix: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    def value(state: str, action: str, preference: str) -> float | None:
        row = matrix[state][action][preference]
        return row.get("discounted_return") if row.get("status") == "ok" else None
    def delta(state: str, action: str, preference: str) -> float | None:
        left, idle = value(state, action, preference), value(state, "idle", preference)
        return None if left is None or idle is None else left - idle
    remote = "stale_local_fresh_remote"
    checks = {
        "reward_communication_penalizes_share": {
            "communication_saving_share_minus_idle": delta(remote, "share", "communication_saving"),
            "estimate_quality_share_minus_idle": delta(remote, "share", "estimate_quality")},
        "reward_quality_values_remote_actions": {
            "estimate_quality_share_minus_idle": delta(remote, "share", "estimate_quality"),
            "communication_saving_share_minus_idle": delta(remote, "share", "communication_saving"),
            "estimate_quality_process_minus_idle": delta(remote, "process", "estimate_quality")},
        "reward_resource_penalizes_cost": {
            "resource_sample_minus_idle": delta("resource_pressure", "sample", "resource_saving"),
            "completion_sample_minus_idle": delta("resource_pressure", "sample", "completion"),
            "resource_share_minus_idle": delta("resource_pressure", "share", "resource_saving")},
    }
    checks["reward_direction_supported"] = (
        checks["reward_communication_penalizes_share"]["communication_saving_share_minus_idle"]
        < checks["reward_communication_penalizes_share"]["estimate_quality_share_minus_idle"]
        and checks["reward_quality_values_remote_actions"]["estimate_quality_share_minus_idle"]
        > checks["reward_quality_values_remote_actions"]["communication_saving_share_minus_idle"]
        and checks["reward_resource_penalizes_cost"]["resource_sample_minus_idle"]
        < checks["reward_resource_penalizes_cost"]["completion_sample_minus_idle"])
    remote_policy = [row for row in policy["rows"] if row["state"] == remote]
    checks["policy_response"] = {
        "mean_share_probability_estimate_quality": _mean_probability(remote_policy, "estimate_quality", "share"),
        "mean_share_probability_communication_saving": _mean_probability(remote_policy, "communication_saving", "share"),
        "argmax_counts": {name: sum(row["masked_argmax"] == "share" for row in remote_policy
                                      if row["preference"] == name)
                          for name in ("estimate_quality", "communication_saving")}}
    checks["policy_direction_supported"] = (
        checks["policy_response"]["mean_share_probability_estimate_quality"]
        > checks["policy_response"]["mean_share_probability_communication_saving"])
    return checks


def _mean_probability(rows: Sequence[Dict[str, Any]], preference: str, action: str) -> float | None:
    values = [row["masked_probabilities"][action] for row in rows
              if row["preference"] == preference and row["masked_probabilities"][action] is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    audit = _load(AUDIT)
    protocol = _assert_frozen(audit)
    checkpoints = _load(CHECKPOINT_REPORT)["checkpoint_hashes"]
    states = audit["states"]
    matrix: Dict[str, Any] = {}
    snapshots = {}
    for state in states:
        reference = _make_env(state, BALANCED)
        snapshot = _state_snapshot(reference, state)
        missing = [name for name in state["required_legal"] if name not in snapshot["legal_actions"]]
        if missing:
            raise RuntimeError(f"固定状态 {state['id']} 缺少要求的合法动作：{missing}")
        snapshots[state["id"]] = snapshot
        matrix[state["id"]] = {}
        for action_name, action in ACTION_BY_NAME.items():
            # equal-weight reference records the unweighted five-utility trajectory once.
            reference_run = _rollout(state, BALANCED, action, int(audit["horizon_steps"]))
            matrix[state["id"]][action_name] = {}
            for preference_name, preference in protocol["preferences"].items():
                run = _rollout(state, preference, action, int(audit["horizon_steps"]))
                if run["status"] == "ok" and reference_run["status"] == "ok":
                    run["raw_utility_steps"] = reference_run["raw_utility_steps"]
                    run["raw_utility_discounted"] = reference_run["raw_utility_discounted"]
                matrix[state["id"]][action_name][preference_name] = run
    policy = _policy_matrix(states, protocol["preferences"], checkpoints)
    checks = _direction_checks(matrix, policy)
    _dump(os.path.join(OUT, "sensitivity_matrix.json"), {"snapshots": snapshots, "matrix": matrix})
    _dump(os.path.join(OUT, "policy_response_matrix.json"), policy)
    _dump(os.path.join(OUT, "audit_report.json"), {
        "protocol": audit["protocol_version"], "scope": {"no_training": True,
        "v2_reward_changed": False, "test_v4_read": False, "test_v4_unsealed": False},
        "empty_sample_semantics": {"reporting": "null + not_applicable, never numeric zero-as-best",
        "reward": "v2 age_quality is 0 with no visible tracks, so no empty-set best-reward loophole"},
        "state_snapshots": snapshots, "direction_checks": checks,
        "root_cause": ("reward direction correct but policy failed to learn it"
                       if checks["reward_direction_supported"] and not checks["policy_direction_supported"]
                       else "reward direction is not fully supported by fixed-state counterfactuals"),
        "checkpoint_hashes": checkpoints})
    rows = []
    for state, actions in matrix.items():
        for action, preferences in actions.items():
            for preference, value in preferences.items():
                rows.append({"state": state, "action": action, "preference": preference,
                             "status": value["status"], "discounted_return": value.get("discounted_return"),
                             "raw_utilities": json.dumps(value.get("raw_utility_discounted"), ensure_ascii=False)})
    with open(os.path.join(OUT, "sensitivity_matrix.csv"), "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"direction_checks": checks, "test_v4": "sealed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
