"""新鲜度/不确定度调度的小样本开发机制检查。

这不是大规模训练，不读取封存测试集，也不做显著性声明。它使用同一
规则调度器和同一闭环，只切换四组特征门，检查方向是否有继续训练的机制证据。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication import SHARE_CONSTRAINED, SHARE_IDEAL  # noqa: E402
from multi_target_stress.system_scenarios import (  # noqa: E402
    S8_BIAS,
    get_system_scenario,
)
from resource_management.closed_loop import run_closed_loop  # noqa: E402
from resource_management.information_research import (  # noqa: E402
    ABLATION_ARMS,
    RESEARCH_HYPOTHESIS,
    rule_config_for_arm,
)
from resource_management.scheduling import SchedulerPolicy  # noqa: E402


DEFAULT_CONFIG = os.path.join("config", "information_research_v1.json")
DEFAULT_DIGEST = os.path.join("config", "information_research_v1.sha256")
DEFAULT_OUT = os.path.join("output", "development", "information_research")


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_contract(path: str = DEFAULT_CONFIG,
                  digest_path: str = DEFAULT_DIGEST) -> Dict[str, Any]:
    with open(digest_path, "r", encoding="ascii") as handle:
        expected = handle.read().strip().lower()
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"研究契约摘要不一致：expected={expected}, actual={actual}"
        )
    with open(path, "r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("hypothesis") != RESEARCH_HYPOTHESIS:
        raise ValueError("研究假设与代码常量不一致")
    if set(contract.get("ablation_arms") or {}) != {
        arm.value for arm in ABLATION_ARMS
    }:
        raise ValueError("四组消融名称与代码不一致")
    if contract["training_budget"].get("test_release") is not False:
        raise ValueError("开发阶段不得解封测试集")
    return contract


def _scenario_mechanisms(name: str) -> Dict[str, Any]:
    if name == "handover":
        get_system_scenario("S6")  # 显式校验固定场景仍存在
        return {
            "share_policy": SHARE_IDEAL,
            "information_research_extension": True,
        }
    if name == "communication_timing":
        scenario = get_system_scenario("S7")
        return {
            "share_policy": SHARE_CONSTRAINED,
            "comm": dict(scenario.comm_extras),
            "information_research_extension": True,
        }
    if name == "system_bias":
        get_system_scenario("S8")
        return {
            "share_policy": SHARE_IDEAL,
            "information_research_extension": True,
            # 偏差标签只在真值编排层注入；调度观测无此字段。
            "bias": {"NODE_B": dict(S8_BIAS)},
        }
    raise KeyError(name)


def run_development_check(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    dev = contract["development_mechanism_check"]
    rows: List[Dict[str, Any]] = []
    for scenario_name in dev["scenarios"]:
        mechanisms = _scenario_mechanisms(scenario_name)
        for seed in dev["seeds"]:
            for arm in ABLATION_ARMS:
                result = run_closed_loop(
                    SchedulerPolicy.RULE,
                    seed=int(seed),
                    steps=int(dev["steps"]),
                    scheduling_config=rule_config_for_arm(arm),
                    mechanisms=mechanisms,
                    keep_decisions=False,
                )
                metrics = result.metrics
                vector = metrics["evaluation_vector"]["values"]
                rows.append({
                    "scenario": scenario_name,
                    "stress_reference": {
                        "handover": "S6", "communication_timing": "S7",
                        "system_bias": "S8",
                    }[scenario_name],
                    "seed": int(seed),
                    "arm": arm.value,
                    "completion_rate": metrics["completion_rate"],
                    "n_completed": metrics["n_completed"],
                    "n_expired": metrics["n_expired"],
                    "mean_waiting_s": metrics["mean_waiting_s"],
                    "resource_consumption": vector["resource_consumption"],
                    "position_error_rmse_m": metrics["position_error_rmse_m"],
                    "compute_time_s": metrics["compute_time_s"],
                    "conservation_all": metrics["conservation_all"],
                })
    return rows


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for arm in ABLATION_ARMS:
        selected = [row for row in rows if row["arm"] == arm.value]
        completions = [float(row["completion_rate"]) for row in selected]
        output.append({
            "arm": arm.value,
            "n_cells": len(selected),
            "completion_mean": _mean(completions),
            "completion_min": min(completions) if completions else 0.0,
            "completion_range": (
                max(completions) - min(completions) if completions else 0.0
            ),
            "expired_mean": _mean(float(row["n_expired"]) for row in selected),
            "waiting_mean_s": _mean(
                float(row["mean_waiting_s"]) for row in selected
            ),
            "resource_consumption_mean": _mean(
                float(row["resource_consumption"]) for row in selected
            ),
            "position_error_rmse_mean_m": _mean(
                float(row["position_error_rmse_m"]) for row in selected
            ),
            "compute_time_mean_s": _mean(
                float(row["compute_time_s"]) for row in selected
            ),
            "all_resources_conserved": all(
                bool(row["conservation_all"]) for row in selected
            ),
        })
    return output


def interpretation(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = {row["arm"]: row for row in summary}
    baseline = by_arm["main_baseline"]
    combined = by_arm["freshness_uncertainty"]
    delta_worst = combined["completion_min"] - baseline["completion_min"]
    delta_expired = combined["expired_mean"] - baseline["expired_mean"]
    delta_resource = (
        combined["resource_consumption_mean"]
        - baseline["resource_consumption_mean"]
    )
    mechanism_support = (
        delta_worst > 1e-12
        and delta_resource <= 1e-12
        and combined["all_resources_conserved"]
    )
    return {
        "hypothesis": RESEARCH_HYPOTHESIS,
        "mechanism_support_in_small_development_check": mechanism_support,
        "combined_minus_baseline_worst_completion": delta_worst,
        "combined_minus_baseline_mean_expired": delta_expired,
        "combined_minus_baseline_resource_consumption": delta_resource,
        "claim": (
            "小样本机制结果与假设方向一致，但不构成统计显著性证据。"
            if mechanism_support else
            "小样本机制检查未支持“最差完成率改善且不增加资源消耗”的完整假设；"
            "保留退化和无差异结果，不扩大结论。"
        ),
        "known_causal_limit": (
            "当前资源任务的执行尚未反向控制传感器或融合更新，"
            "因此估计误差可能对四组不变；这是环境接线限制，不是算法结论。"
        ),
        "statistical_significance_claimed": False,
        "test_split_opened": False,
    }


def write_outputs(out_dir: str, contract: Dict[str, Any],
                  rows: List[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    summary = aggregate(rows)
    conclusion = interpretation(summary)
    payload = {
        "protocol_version": contract["protocol_version"],
        "contract_sha256": _sha256(DEFAULT_CONFIG),
        "hypothesis": RESEARCH_HYPOTHESIS,
        "development_only": True,
        "rows": rows,
        "summary": summary,
        "interpretation": conclusion,
    }
    json_path = os.path.join(out_dir, "development_evidence.json")
    csv_path = os.path.join(out_dir, "development_cells.csv")
    report_path = os.path.join(out_dir, "development_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# 新鲜度与不确定度调度：开发机制报告\n\n")
        handle.write(f"> {RESEARCH_HYPOTHESIS}\n\n")
        handle.write("本轮仅使用少量开发种子，没有打开测试集，不宣称统计显著性。\n\n")
        handle.write("| 消融组 | 完成率均值 | 最差完成率 | 跨格子极差 | 平均过期 | 平均等待(s) | 资源消耗 | RMSE(m) | 计算(s) |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary:
            handle.write(
                f"| {row['arm']} | {row['completion_mean']:.4f} | "
                f"{row['completion_min']:.4f} | {row['completion_range']:.4f} | "
                f"{row['expired_mean']:.2f} | {row['waiting_mean_s']:.3f} | "
                f"{row['resource_consumption_mean']:.4f} | "
                f"{row['position_error_rmse_mean_m']:.2f} | "
                f"{row['compute_time_mean_s']:.6f} |\n"
            )
        handle.write("\n## 结论边界\n\n")
        handle.write(conclusion["claim"] + "\n\n")
        handle.write(conclusion["known_causal_limit"] + "\n\n")
        handle.write("来源一致性分数未经概率校准，不得称为出错概率。\n")
    return {"json": json_path, "csv": csv_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--digest", default=DEFAULT_DIGEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    contract = load_contract(args.config, args.digest)
    rows = run_development_check(contract)
    paths = write_outputs(args.out_dir, contract, rows)
    print(json.dumps({"n_cells": len(rows), "outputs": paths,
                      "interpretation": interpretation(aggregate(rows))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
