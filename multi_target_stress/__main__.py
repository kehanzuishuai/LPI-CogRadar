"""多目标压力测试入口（v4.5 P2：4 核心关联场景 + 4 系统级场景）。

用法
----
    python -m multi_target_stress                    # 全部场景（含 S3L 子场景）
    python -m multi_target_stress --group core       # 只跑核心关联场景 S1–S4
    python -m multi_target_stress --group system     # 只跑系统级场景 S5–S8
    python -m multi_target_stress --scenario S5 S6   # 指定场景
    python -m multi_target_stress --seeds 42 7 13    # 多种子（接口就绪，默认单种子）
    python -m multi_target_stress --no-audit         # 不导出逐测量审计 CSV
"""

from __future__ import annotations

import argparse
import os
import sys

#: Windows 控制台默认 GBK，报告里有 → / ⚠ 等符号，不强制 UTF-8 会在中途崩
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_target_stress.metrics import METRIC_ORDER  # noqa: E402
from multi_target_stress.report import (  # noqa: E402
    DEFAULT_OUT_DIR,
    VARIANT_CN,
    write_all,
)
from multi_target_stress.scenarios import SCENARIO_IDS  # noqa: E402
from multi_target_stress.system_scenarios import SYSTEM_SCENARIO_IDS  # noqa: E402

#: 全部场景 ID（核心 + 系统级）
ALL_IDS = tuple(SCENARIO_IDS) + tuple(SYSTEM_SCENARIO_IDS)

#: 系统级场景的表格列：关联层前 11 列对系统场景意义有限，只留最相关的
SYSTEM_COLUMNS = (
    ("id_switch_count", "ID换号", 0),
    ("track_fragmentation_count", "碎裂", 0),
    ("duplicate_track_count", "重复航迹", 0),
    ("missed_track_rate", "漏跟率", 3),
    ("association_accuracy", "关联准确率", 3),
    ("position_rmse_m", "位置RMSE(m)", 1),
    ("continuity_rate", "连续性", 3),
    ("remote_utilization", "远端利用率", 3),
    ("delivery_rate", "送达率", 3),
    ("tracks_dropped", "删除航迹", 0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m multi_target_stress",
        description="多目标压力测试：4 核心关联场景（S1–S4）+ 4 系统级场景（S5–S8）",
    )
    parser.add_argument("--scenario", nargs="*", default=None, choices=list(ALL_IDS),
                        help=f"只跑指定场景（默认全部：{list(ALL_IDS)}）")
    parser.add_argument("--group", choices=("core", "system", "all"),
                        default="all",
                        help="按分组筛选（core=S1–S4，system=S5–S8）")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42],
                        help="种子列表（默认 [42]；多种子接口已就绪，"
                             "本轮不做 20–30 种子统计）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-audit", action="store_true",
                        help="不导出逐测量的关联审计 / 生命周期 CSV")
    return parser


def _select_scenarios(args) -> list:
    if args.scenario:
        return list(args.scenario)
    if args.group == "core":
        return list(SCENARIO_IDS)
    if args.group == "system":
        return list(SYSTEM_SCENARIO_IDS)
    return list(ALL_IDS)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ids = _select_scenarios(args)
    print("=" * 100)
    print("多目标压力测试（v4.5 P2）")
    print(f"场景 {ids}")
    print(f"种子 {args.seeds}｜跟踪器：NN + 常速度卡尔曼（基线，关联与滤波数学未改动）")
    print("=" * 100)

    payload = write_all(
        out_dir=args.out_dir, seeds=args.seeds,
        scenario_ids=ids, dump_audit=not args.no_audit,
    )

    for scenario_id, item in payload["per_scenario"].items():
        scenario = item["scenario"]
        columns = (METRIC_ORDER[:11] if scenario.group == "core"
                   else SYSTEM_COLUMNS)
        print()
        print(f"--- {scenario_id} {scenario.title_cn}"
              f"（{'核心关联' if scenario.group == 'core' else '系统级'}）---")
        print(f"  问题：{scenario.question}")
        print("%-24s " % "变体" + " ".join("%12s" % t for _k, t, _d in columns))
        for variant, entry in item["summary"].items():
            cells = " ".join(
                "%12.*f" % (digits, entry.get(f"{key}_mean", 0.0))
                for key, _t, digits in columns
            )
            label = f"{VARIANT_CN.get(variant, variant)}"
            print("%-24s %s" % (label[:24], cells))
        print("  结论：")
        for line in item["judgement"]:
            print("   *", line)

    print()
    print("=" * 100)
    print(f"指标 CSV        ：{payload['csv']}")
    print(f"系统级指标 CSV  ：{payload['system_csv']}")
    print(f"汇总 JSON       ：{payload['json']}")
    print(f"HTML 报告       ：{payload['html']}")
    print(f"run_id          ：{payload['run_id']}")
    print(f"运行清单        ：{payload['manifest']}")
    print("⚠️ 默认单种子，属描述性统计；未做 20–30 种子与显著性检验。")
    print("⚠️ 未引入 JPDA / IMM：本轮只记录 NN + Kalman 基线的失效模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
