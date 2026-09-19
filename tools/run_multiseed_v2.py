"""PPO 两 arm 的多训练种子复现，以及封存 test-v2 的一次性最终评测。

阶段顺序受命令强制：prepare -> train（十次）-> freeze -> test。
train 子命令只读 training_contract（不含 test-v2 场景/种子），因此没有读取
test-v2 的路径；test 子命令必须验证所有 checkpoint 的 SHA-256。
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, statistics, sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from resource_management.contract_v1 import contract_digest, verify_frozen
from resource_management.learning_protocol import get_split, load_registry, split_digest
from resource_management.unified_evaluation import FrozenEvaluation, evaluate_episode, summary, validate_records
from rl_resource.ablation import FrozenBaseline, assert_frozen
from rl_resource.train import TrainConfig, train

REGISTRY = os.path.join(ROOT, "config", "learning_splits_v2.json")
DIGEST = os.path.join(ROOT, "config", "learning_splits_v2.sha256")
OUT = os.path.join(ROOT, "output", "rl_resource", "multiseed_v2")
ARMS = ("main_baseline", "freshness_uncertainty")
TRAINING_SEEDS = (701, 703, 709, 719, 727)
METHODS = ("rule", "rolling_horizon", "ppo_baseline", "ppo_freshness_uncertainty")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def contract_path() -> str:
    return os.path.join(OUT, "training_contract.json")


def test_manifest_path() -> str:
    return os.path.join(OUT, "test_v2_freeze_manifest.json")


def _verify_registry() -> Dict[str, Any]:
    actual = split_digest(REGISTRY)
    with open(DIGEST, encoding="ascii") as handle:
        expected = handle.read().strip()
    if actual != expected:
        raise RuntimeError("learning_splits_v2 SHA-256 不匹配；拒绝开始")
    registry = load_registry(REGISTRY)
    if not registry.get("test_v2_sealed"):
        raise RuntimeError("test-v2 必须显式 sealed")
    return registry


def prepare() -> None:
    registry = _verify_registry()
    train_split, validation = registry["splits"]["train"], registry["splits"]["validation"]
    frozen = FrozenBaseline()
    # 只让多 seed 改初始化随机性，其他训练契约逐项保持原样。
    invariant = frozen.to_dict()
    invariant.pop("train_seed")
    payload = {
        "protocol": "ppo-multiseed-test-v2-v1",
        "test_v2_registry_sha256": split_digest(REGISTRY),
        "resource_contract_digest": contract_digest(),
        "training_seeds": list(TRAINING_SEEDS), "arms": list(ARMS),
        "train": train_split, "validation": validation,
        "invariant_training_contract": invariant,
        "checkpoint_selection": frozen.selection_rule,
        "test_v2_status": "sealed; training commands receive only this opaque registry SHA",
        "old_test_status": registry["old_test_status"],
    }
    if not verify_frozen(strict=False)["ok"]:
        raise RuntimeError("resource-contract-v1 不匹配")
    dump(contract_path(), payload)
    print(json.dumps({"prepared": contract_path(), "sha256": sha(contract_path())}, ensure_ascii=False))


def train_one(arm: str, seed: int) -> None:
    if arm not in ARMS or seed not in TRAINING_SEEDS:
        raise ValueError("arm 或训练 seed 不在预冻结清单")
    contract = load(contract_path())  # deliberately contains no test-v2 rows
    # 此阶段只使用 prepare 阶段写入的不可逆摘要；不要再打开 test-v2 登记表。
    if not contract.get("test_v2_registry_sha256"):
        raise RuntimeError("缺少预冻结 test-v2 摘要；拒绝训练")
    frozen = FrozenBaseline(train_seed=seed)
    cfg = TrainConfig(
        scenarios=tuple(contract["train"]["scenarios"]), seeds=tuple(contract["train"]["seeds"]),
        episodes=frozen.episodes_per_update * frozen.updates, steps=frozen.steps,
        rollout_episodes=frozen.episodes_per_update, updates=frozen.updates,
        max_nodes=frozen.max_nodes, ppo=frozen.ppo(), policy=frozen.policy(0),
        out_dir=OUT, tag=f"training/{arm}/seed_{seed}", arm=arm,
        eval_scenarios=tuple(contract["validation"]["scenarios"]),
        eval_seeds=tuple(contract["validation"]["seeds"]), seed=seed)
    assert_frozen(cfg, frozen)
    result = train(cfg, quiet=True)
    policy = result["policy_path"]
    dump(os.path.join(OUT, "training", arm, f"seed_{seed}", "multiseed_metadata.json"), {
        "training_seed": seed, "arm": arm, "training_contract_sha256": sha(contract_path()),
        "selected_update": result["selected_update"], "checkpoint_sha256": sha(policy),
        "validation_of_selected": result["validation_of_selected"],
        "validation_without_mask": result["validation_without_mask"],
    })
    print(json.dumps({"arm": arm, "training_seed": seed, "checkpoint": policy,
                      "selected_update": result["selected_update"]}, ensure_ascii=False))


def freeze_test() -> None:
    registry = _verify_registry()
    contract = load(contract_path())
    if contract["test_v2_registry_sha256"] != split_digest(REGISTRY):
        raise RuntimeError("registry changed after prepare")
    checkpoints: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        checkpoints[arm] = {}
        for seed in TRAINING_SEEDS:
            base = os.path.join(OUT, "training", arm, f"seed_{seed}")
            policy, meta = os.path.join(base, "policy.pt"), os.path.join(base, "multiseed_metadata.json")
            if not os.path.isfile(policy) or not os.path.isfile(meta):
                raise RuntimeError(f"缺少 {arm}/seed_{seed} 训练产物，拒绝解封")
            item = load(meta)
            if item["checkpoint_sha256"] != sha(policy):
                raise RuntimeError(f"{arm}/seed_{seed} checkpoint 已改变")
            checkpoints[arm][str(seed)] = {"path": policy, "sha256": sha(policy),
                                           "selected_update": item["selected_update"]}
    manifest = {
        "protocol": "ppo-multiseed-test-v2-v1", "test_v2_released": False,
        "training_contract_sha256": sha(contract_path()),
        "test_v2_registry_sha256": split_digest(REGISTRY),
        "resource_contract_digest": contract_digest(), "checkpoints": checkpoints,
        "methods": list(METHODS), "runtime_mode": "plan_controlled_feedback",
        "task_gating": "expose_all", "test_v2": registry["splits"]["test"],
        "selection_after_test_forbidden": True,
    }
    dump(test_manifest_path(), manifest)
    print(json.dumps({"frozen": test_manifest_path(), "sha256": sha(test_manifest_path())}, ensure_ascii=False))


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _seed_means(rows: Iterable[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["training_seed"]), []).append(float(row[metric]))
    means = {seed: statistics.fmean(values) for seed, values in grouped.items()}
    return {"per_training_seed": means, "between_training_seed": summary(list(means.values())),
            "within_environment": {seed: summary(values) for seed, values in grouped.items()}}


def test_v2() -> None:
    manifest = load(test_manifest_path())
    registry = _verify_registry()
    if manifest["test_v2_registry_sha256"] != split_digest(REGISTRY):
        raise RuntimeError("test-v2 registry changed after freeze")
    test = get_split("test", path=REGISTRY, release_test=True)
    rows: List[Dict[str, Any]] = []
    # 规则方法与所有 PPO 在相同 scenario × environment-seed 格子评测。
    reference = FrozenEvaluation(scenario_path=REGISTRY)
    for method in ("rule", "rolling_horizon"):
        for scenario in test.scenarios:
            for env_seed in test.seeds:
                row = evaluate_episode(method, scenario, env_seed, reference); row["training_seed"] = "rule_fixed"; rows.append(row)
    for arm, method in (("main_baseline", "ppo_baseline"), ("freshness_uncertainty", "ppo_freshness_uncertainty")):
        for seed in TRAINING_SEEDS:
            path = manifest["checkpoints"][arm][str(seed)]["path"]
            if sha(path) != manifest["checkpoints"][arm][str(seed)]["sha256"]:
                raise RuntimeError(f"{arm}/seed_{seed} checkpoint changed")
            frozen = FrozenEvaluation(checkpoints={"ppo_baseline": path, "ppo_freshness_uncertainty": path}, scenario_path=REGISTRY)
            for scenario in test.scenarios:
                for env_seed in test.seeds:
                    row = evaluate_episode(method, scenario, env_seed, frozen); row["training_seed"] = seed; rows.append(row)
    errors = validate_records(rows)
    if errors: raise RuntimeError("test-v2 integrity failure: " + "; ".join(errors))
    metrics = ("completion_rate", "estimate_quality", "communication_overhead", "unmasked_argmax_invalid_rate")
    paired_metrics = ("completion_rate", "estimate_quality", "communication_overhead")
    ppo_rows = [row for row in rows if str(row["method"]).startswith("ppo_")]
    stats = {method: {metric: _seed_means([r for r in ppo_rows if r["method"] == method], metric)
                      for metric in metrics} for method in ("ppo_baseline", "ppo_freshness_uncertainty")}
    refs = {(r["method"], r["scenario"], r["environment_seed"]): r for r in rows if r["method"] in ("rule", "rolling_horizon")}
    paired: Dict[str, Any] = {}
    for method in ("ppo_baseline", "ppo_freshness_uncertainty"):
        paired[method] = {}
        for ref in ("rule", "rolling_horizon"):
            paired[method][ref] = {metric: _seed_means([
                {**r, metric: float(r[metric]) - float(refs[(ref, r["scenario"], r["environment_seed"])][metric])}
                for r in ppo_rows if r["method"] == method], metric) for metric in paired_metrics}
    manifest["test_v2_released"] = True
    result = {"manifest": manifest, "records": rows, "statistics": stats, "paired_differences": paired,
              "boundaries": ["Old test is historical single-checkpoint evidence only.", "Training seed and environment seed variation are reported separately.", "No seed/checkpoint/hyperparameter selection after test-v2 release.", "Masked legality is not claimed as learned legality."]}
    dump(os.path.join(OUT, "test_v2_report.json"), result); _write_csv(os.path.join(OUT, "test_v2_raw.csv"), rows)
    dump(test_manifest_path(), manifest)
    print(json.dumps({"test_v2_records": len(rows), "report": os.path.join(OUT, "test_v2_report.json")}, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare"); p_train = sub.add_parser("train"); p_train.add_argument("--arm", required=True); p_train.add_argument("--seed", type=int, required=True)
    sub.add_parser("freeze"); sub.add_parser("test")
    args = parser.parse_args(argv)
    if args.command == "prepare": prepare()
    elif args.command == "train": train_one(args.arm, args.seed)
    elif args.command == "freeze": freeze_test()
    else: test_v2()
    return 0

if __name__ == "__main__": raise SystemExit(main())
