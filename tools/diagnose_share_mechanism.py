"""Share 机制的只读诊断：不训练、不改 v1、也绝不读取 test-v3。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from resource_management.closed_loop import _build_feedback_world
from resource_management.model import ExecutionPlan, TaskRequest
from resource_management.units import ResourceUnit, TaskKind
from rl_resource.actions import ACTION_IDLE, ACTION_PROCESS, ACTION_SAMPLE, ACTION_SHARE
from rl_resource.env import BALANCED_NORMALIZATION, CentralizedResourceSchedulingEnv, EnvConfig
from rl_resource.policy import ActorCritic

OUT = os.path.join(ROOT, "output", "share_mechanism_diagnosis")
PREFERENCE_CONFIG = os.path.join(ROOT, "config", "preference_ppo_v1.json")
PREFERENCES = json.load(open(PREFERENCE_CONFIG, encoding="utf-8"))["preferences"]
TRAIN_SCENES = ("rm_train_base", "rm_train_light_load", "rm_train_high_arrival", "rm_train_constrained_comm")
VALIDATION_SCENES = ("rm_validation_node_outage", "rm_validation_handover", "rm_validation_sensor_bias")
TRAIN_SEEDS = (101, 103)
VALIDATION_SEEDS = (211, 223, 227)
CHECKPOINT_SEEDS = (907, 911, 919)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(name: str, value: Any) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _empty_funnel() -> Dict[str, Any]:
    return {"node_steps": 0, "share_candidate": 0, "share_legal": 0,
            "share_masked": 0, "share_argmax": 0,
            "raw_share_probability_mass": 0.0,
            "legal_raw_share_probability_mass": 0.0,
            "legal_masked_share_probability_mass": 0.0,
            "action_counts": {"idle": 0, "sample": 0, "process": 0, "share": 0}}


def _funnel_rates(funnel: Dict[str, Any]) -> Dict[str, Any]:
    steps, candidates, legal = (int(funnel[key]) for key in
                                ("node_steps", "share_candidate", "share_legal"))
    return {**funnel,
            "candidate_rate": funnel["share_candidate"] / steps if steps else 0.0,
            "legal_rate_given_candidate": funnel["share_legal"] / candidates if candidates else 0.0,
            "masked_rate_given_candidate": funnel["share_masked"] / candidates if candidates else 0.0,
            "raw_share_probability_mean": funnel["raw_share_probability_mass"] / steps if steps else 0.0,
            "raw_share_probability_mean_when_legal": funnel["legal_raw_share_probability_mass"] / legal if legal else 0.0,
            "masked_share_probability_mean_when_legal": funnel["legal_masked_share_probability_mass"] / legal if legal else 0.0,
            "argmax_rate_given_legal": funnel["share_argmax"] / legal if legal else 0.0}


def _add_funnel(total: Dict[str, Any], item: Dict[str, Any]) -> None:
    for key in ("node_steps", "share_candidate", "share_legal", "share_masked",
                "share_argmax", "raw_share_probability_mass",
                "legal_raw_share_probability_mass", "legal_masked_share_probability_mass"):
        total[key] += item[key]
    for action, count in item["action_counts"].items():
        total["action_counts"][action] += count


def _run_policy_funnel(model: ActorCritic, preference: Sequence[float],
                       scenes: Iterable[str], seeds: Iterable[int]) -> Dict[str, Any]:
    funnel = _empty_funnel()
    for scene in scenes:
        for seed in seeds:
            env = CentralizedResourceSchedulingEnv(EnvConfig(
                scenario=scene, seed=seed, steps=24, arm="main_baseline",
                reward_mode="balanced", preference=tuple(preference),
                preference_conditioned=True, keep_trace=False))
            observation, info = env.reset(seed=seed)
            finished = False
            while not finished:
                values = torch.tensor([observation], dtype=torch.float32)
                logits, _value = model(values)
                raw = torch.softmax(logits[0], dim=-1)
                context = env.action_context()
                selected = []
                for index, row in enumerate(info["mask"][:len(env.node_ids)]):
                    node_id = env.node_ids[index]
                    candidate = ACTION_SHARE in context.candidates.get(node_id, {})
                    legal = bool(row[ACTION_SHARE])
                    funnel["node_steps"] += 1
                    funnel["share_candidate"] += int(candidate)
                    funnel["share_legal"] += int(legal)
                    funnel["share_masked"] += int(candidate and not legal)
                    funnel["raw_share_probability_mass"] += float(raw[index, ACTION_SHARE])
                    if legal:
                        funnel["legal_raw_share_probability_mass"] += float(raw[index, ACTION_SHARE])
                    masked_logits = model.masked_logits(
                        logits[:, index:index + 1, :],
                        torch.tensor([[row]], dtype=torch.bool))[0, 0]
                    masked_probability = torch.softmax(masked_logits, dim=-1)
                    if legal:
                        funnel["legal_masked_share_probability_mass"] += float(masked_probability[ACTION_SHARE])
                    action = int(torch.argmax(masked_logits))
                    selected.append(action)
                    funnel["share_argmax"] += int(action == ACTION_SHARE)
                    funnel["action_counts"][("idle", "sample", "process", "share")[action]] += 1
                observation, _reward, terminated, truncated, info = env.step(selected)
                finished = terminated or truncated
    return _funnel_rates(funnel)


def _total_comm(env: CentralizedResourceSchedulingEnv) -> float:
    return sum(float(node.budget.consumed.get(ResourceUnit.COMM_BYTE, 0.0))
               for node in env._executor.nodes.values())


def _track_summary(env: CentralizedResourceSchedulingEnv, node_id: str) -> Dict[str, Any]:
    now = env._driver.current_time_s
    tracks = list(env._runtime.centers[node_id].tracks)
    ages = [max(0.0, now - float(track.last_measurement_time)) for track in tracks]
    sigmas = [max(track.sigma_position.x, track.sigma_position.y, track.sigma_position.z)
              for track in tracks]
    return {"n_tracks": len(tracks),
            "mean_information_age_s": sum(ages) / len(ages) if ages else None,
            "mean_estimate_quality": (sum(1.0 / (1.0 + age) for age in ages) / len(ages) if ages else 0.0),
            "mean_sigma_max_m": sum(sigmas) / len(sigmas) if sigmas else None,
            "local_updates": sum(int(track.local_updates) for track in tracks),
            "remote_updates": sum(int(track.remote_updates) for track in tracks)}


def _prepared_legal_share_env() -> CentralizedResourceSchedulingEnv:
    """推进到真正的 share 合法状态：B sample → B process → B share。"""
    env = CentralizedResourceSchedulingEnv(EnvConfig(
        scenario="rm_validation_handover", seed=211, steps=24, arm="main_baseline",
        reward_mode="balanced", preference=(0.2,) * 5,
        preference_conditioned=True, keep_trace=True))
    env.reset(seed=211)
    env.step([ACTION_IDLE, ACTION_SAMPLE])
    env.step([ACTION_IDLE, ACTION_PROCESS])
    if not env._mask[1][ACTION_SHARE]:
        raise RuntimeError("诊断前置序列没有到达 share 合法状态")
    return env


def _counterfactual(force_share: bool) -> Dict[str, Any]:
    env = _prepared_legal_share_env()
    before = {"share_candidate": ACTION_SHARE in env.action_context().candidates["NODE_B"],
              "share_legal": bool(env._mask[1][ACTION_SHARE]),
              "outbox_measurements": len(env._runtime.share_outbox["NODE_B"]),
              "comm_bytes": _total_comm(env), "node_a": _track_summary(env, "NODE_A"),
              "node_b": _track_summary(env, "NODE_B")}
    _obs, immediate_reward, _term, _trunc, immediate_info = env.step(
        [ACTION_IDLE, ACTION_SHARE if force_share else ACTION_IDLE])
    after_decision = {"chosen": immediate_info["chosen"], "immediate_reward": immediate_reward,
                      "five_utility_contributions": immediate_info["reward_components"],
                      "comm_bytes": _total_comm(env), "bus_messages": len(env._world["bus"].log),
                      "outbox_measurements": len(env._runtime.share_outbox["NODE_B"]),
                      "node_a_process_legal_next": bool(env._mask[0][ACTION_PROCESS]),
                      "node_a_sample_legal_next": bool(env._mask[0][ACTION_SAMPLE])}
    action_a = ACTION_PROCESS if env._mask[0][ACTION_PROCESS] else ACTION_IDLE
    _obs, later_reward, _term, _trunc, later_info = env.step([action_a, ACTION_IDLE])
    last_process = next((row for row in reversed(env._runtime.event_log)
                         if row.get("event") == "process" and row.get("node_id") == "NODE_A"), None)
    return {"before": before, "after_decision": after_decision,
            "next_tick": {"chosen": later_info["chosen"], "reward": later_reward,
                          "five_utility_contributions": later_info["reward_components"],
                          "node_a_process_event": last_process,
                          "node_a": _track_summary(env, "NODE_A")}}


def _manual_task(world: Dict[str, Any], node_id: str, kind: TaskKind, index: int) -> None:
    now = world["clock"].now_s
    task = TaskRequest(task_id=f"share-diagnosis-{index}-{node_id}-{kind.value}",
                       node_id=node_id, kind=kind, start_s=now,
                       idempotency_key=f"share-diagnosis:{index}:{node_id}:{kind.value}")
    plan = ExecutionPlan(plan_id=f"share-diagnosis-plan-{index}", submit_time_s=now,
                         tasks=[task])
    execution = world["runtime"].submit(plan)
    if execution.n_applied != 1:
        raise RuntimeError(f"最小场景任务没有执行：{execution.to_dict()}")


def _manual_tick(world: Dict[str, Any], label: str) -> None:
    world["clock"].advance(1.0, label)
    world["sim"].scene.advance_all(1.0)
    world["runtime"].begin_tick(world["clock"].now_s)


def _world_track_summary(world: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    now = world["clock"].now_s
    tracks = list(world["centers"][node_id].tracks)
    ages = [max(0.0, now - float(track.last_measurement_time)) for track in tracks]
    sigmas = [max(track.sigma_position.x, track.sigma_position.y, track.sigma_position.z)
              for track in tracks]
    return {"n_tracks": len(tracks),
            "mean_information_age_s": sum(ages) / len(ages) if ages else None,
            "mean_estimate_quality": (sum(1.0 / (1.0 + age) for age in ages) / len(ages) if ages else 0.0),
            "mean_sigma_max_m": sum(sigmas) / len(sigmas) if sigmas else None,
            "local_updates": sum(int(track.local_updates) for track in tracks),
            "remote_updates": sum(int(track.remote_updates) for track in tracks)}


def _minimal_scripted_case(send_share: bool) -> Dict[str, Any]:
    """最小真闭环因果案例：A 的本地航迹变旧，B 保存更新的真实测量。"""
    world = _build_feedback_world(seed=211, steps=8, keep_decisions=False)
    _manual_tick(world, "diagnostic t1")
    _manual_task(world, "NODE_A", TaskKind.SAMPLE, 1)
    _manual_tick(world, "diagnostic t2")
    _manual_task(world, "NODE_A", TaskKind.PROCESS, 2)
    _manual_tick(world, "diagnostic t3")
    _manual_task(world, "NODE_B", TaskKind.SAMPLE, 3)
    before_share = {"node_a_stale": _world_track_summary(world, "NODE_A"),
                    "node_b_outbox_measurements": len(world["runtime"].share_outbox["NODE_B"]),
                    "node_b_has_fresh_measurement": bool(world["runtime"].share_outbox["NODE_B"])}
    _manual_tick(world, "diagnostic t4")
    if send_share:
        _manual_task(world, "NODE_B", TaskKind.SHARE, 4)
    after_share = {"bus_messages": len(world["bus"].log),
                   "sent_comm_bytes": sum(float(row.get("sent_comm_bytes", 0.0))
                                          for row in world["runtime"].event_log if row.get("event") == "share"),
                   "accounted_comm_bytes": sum(float(row.get("accounted_comm_bytes", 0.0))
                                               for row in world["runtime"].event_log if row.get("event") == "share")}
    _manual_tick(world, "diagnostic t5")
    process_available = world["runtime"].has_processable("NODE_A", world["clock"].now_s)
    process_event = None
    if process_available:
        _manual_task(world, "NODE_A", TaskKind.PROCESS, 5)
        process_event = next((row for row in reversed(world["runtime"].event_log)
                              if row.get("event") == "process"
                              and row.get("node_id") == "NODE_A"
                              and row.get("time_s") == 5.0), None)
    return {"script": "A: sample→process→predict-only; B: sample→share-or-idle; A: process iff message arrived",
            "before_share": before_share, "after_share": after_share,
            "node_a_process_available": process_available, "node_a_process_event": process_event,
            "node_a_after": _world_track_summary(world, "NODE_A"),
            "runtime_invariants": {"resource_conserved": world["executor"].conservation_report()["all_conserved"],
                                   "duplicate_runtime_tasks": max(0, len([row for row in world["runtime"].event_log if row.get("event") in ("sample", "process", "share")]) - world["runtime"].unique_task_keys()),
                                   "truth_payload_violations": sum(len(row.get("payload_truth_fields") or []) for row in world["runtime"].event_log)}}


def main() -> None:
    detailed: Dict[str, Any] = {}
    aggregate = {"train": _empty_funnel(), "validation": _empty_funnel()}
    checkpoints: Dict[str, str] = {}
    for seed in CHECKPOINT_SEEDS:
        checkpoint = os.path.join(ROOT, "output", "rl_resource", "preference_ppo", f"seed_{seed}", "policy.pt")
        checkpoints[str(seed)] = _sha256(checkpoint)
        model, _metadata = ActorCritic.load(checkpoint, map_location="cpu")
        model.eval()
        for name, preference in PREFERENCES.items():
            train = _run_policy_funnel(model, preference, TRAIN_SCENES, TRAIN_SEEDS)
            validation = _run_policy_funnel(model, preference, VALIDATION_SCENES, VALIDATION_SEEDS)
            detailed[f"seed{seed}/{name}"] = {"train": train, "validation": validation}
            _add_funnel(aggregate["train"], train)
            _add_funnel(aggregate["validation"], validation)
    result = {"protocol": "share_mechanism_diagnosis-v1",
              "scope": {"no_training": True, "v1_config_unchanged": True,
                        "test_v3_read": False, "test_v3_unsealed": False,
                        "checkpoint_hashes": checkpoints,
                        "preference_config_sha256": _sha256(PREFERENCE_CONFIG)},
              "funnel": detailed,
              "aggregate_funnel": {key: _funnel_rates(value) for key, value in aggregate.items()},
              "counterfactual": {"policy_idle": _counterfactual(False),
                                 "forced_legal_share": _counterfactual(True)},
              "minimal_controlled": {"no_share": _minimal_scripted_case(False),
                                     "share": _minimal_scripted_case(True)},
              "frozen_reward_scale": {"protocol": "balanced_protocol_v1 / preference_ppo_v1",
                                      "weights": "preference simplex; only preference weights vary",
                                      "normalization": BALANCED_NORMALIZATION,
                                      "share_comm_cost_bytes": 128.0,
                                      "immediate_caveat": "share sends now; fusion benefit only appears after arrival plus receiver process"}}
    _json_dump("share_diagnosis.json", result)
    print(os.path.join(OUT, "share_diagnosis.json"))


if __name__ == "__main__":
    main()
