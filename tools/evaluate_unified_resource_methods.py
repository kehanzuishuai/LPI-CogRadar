"""统一公平评测入口：validation 预检后一次性显式释放 test。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# 允许直接以 ``python tools/...py`` 运行，而不要求用户事先设置 PYTHONPATH。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_management.unified_evaluation import (
    FrozenEvaluation, aggregate, evaluate_split, freeze_manifest, validate_records,
)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _render_html(report: Dict[str, Any]) -> str:
    rows = report["records"]
    columns = ("method", "scenario", "environment_seed", "service_completion",
               "task_timeliness", "estimate_quality", "resource_consumption",
               "communication_overhead", "compute_time_s", "completion_rate",
               "mean_waiting_s", "n_expired", "executor_rejection_rate")
    table = ["<table><thead><tr>"] + [f"<th>{html.escape(key)}</th>" for key in columns] + ["</tr></thead><tbody>"]
    for row in rows:
        table.append("<tr>" + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in columns) + "</tr>")
    table.append("</tbody></table>")
    payload = html.escape(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return """<!doctype html><meta charset='utf-8'><title>统一资源调度评测</title>
<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ccc;padding:4px}th{background:#eee}pre{white-space:pre-wrap}</style>
<h1>统一资源调度公平评测</h1><p>无综合分；只报告同口径多维取舍。PPO 的 mask 依赖在 CSV/JSON 中单列。</p>""" + "".join(table) + "<h2>统计与配对差异</h2><pre>" + payload + "</pre>"


def _write_report(out_dir: str, split: str, frozen_manifest: Dict[str, Any],
                  records: List[Dict[str, Any]], release_test: bool) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    errors = validate_records(records)
    if errors:
        raise RuntimeError("评测一致性失败：\n" + "\n".join(errors))
    report = {
        "report_version": "unified-resource-evaluation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "test_released": bool(release_test and split == "test"),
        "freeze_manifest": frozen_manifest,
        "records": records,
        "aggregate": aggregate(records, frozen_manifest["frozen"]["reference_method"]),
        "validation_checks": {"ok": True, "errors": []},
        "honest_boundaries": [
            "不使用单一综合分；六维评价和附加指标按方向分别解释。",
            "PPO 带 mask 的合法率为构造保证，不表示模型已内化约束；同时报告去 mask 诊断。",
            "每个 PPO 方法当前只有一个冻结 checkpoint，训练随机性 n=1，不可估计；CI 仅反映环境种子变化。",
            "test 解封后，结果不得用于 checkpoint、奖励、超参、种子、mask 或场景选择。",
            "估计误差/质量的真值对齐仅为离线评测，不进入任何方法的输入。",
        ],
    }
    json_path = os.path.join(out_dir, f"{split}_report.json")
    csv_path = os.path.join(out_dir, f"{split}_per_episode.csv")
    html_path = os.path.join(out_dir, f"{split}_report.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_csv(csv_path, records)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(_render_html(report))
    return {"json": json_path, "csv": csv_path, "html": html_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="七方法统一公平资源调度评测")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--release-test", action="store_true",
                        help="一次性显式解封封存 test；须先完成 validation 与冻结")
    parser.add_argument("--out-dir", default=os.path.join("output", "unified_resource_evaluation"))
    args = parser.parse_args(argv)
    if args.split == "test" and not args.release_test:
        parser.error("test 仍封存；必须显式传 --release-test")
    frozen = FrozenEvaluation()
    manifest = freeze_manifest(frozen)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "freeze_manifest.json")
    # 只有 validation **成功结束后**才形成正式冻结清单。此前允许修评测
    # 实现；test 则必须拿既有清单逐项比对，禁止悄悄换输入。
    if args.split == "test":
        if not os.path.exists(manifest_path):
            raise RuntimeError("尚未完成 validation 冻结，拒绝解封 test")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise RuntimeError("已有冻结清单与当前代码/配置/checkpoint 指纹不一致，拒绝评测")
    records = evaluate_split(args.split, frozen, release_test=args.release_test)
    written = _write_report(args.out_dir, args.split, manifest, records, args.release_test)
    if args.split == "validation":
        # 仅通过 `validate_records` 的 validation 才能落成最终冻结清单。
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"split": args.split, "n_records": len(records), "artifacts": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
