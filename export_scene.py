"""场景快照导出（v4.1 多平台几何升级）。

把场景在**每一个时间步**的实体状态与实体间几何关系写成 CSV/JSON，
供论文作图、外部工具校验与人工核对。

为什么要导出"关系"而不只是"实体"
--------------------------------
升级前能拿到的场景真值是一堆聚合量（最近目标距离、最远目标距离、最小 RCS、
最近侦察机距离）。这些数字**脱离了"谁对谁、哪一刻"就无法解释**：
在多雷达场景里，"最近目标距离 = 4001.8 m" 到底是谁量到的？
`scene_relations.csv` 把每一对实体的距离、方位、俯仰、机体方位、径向速度
逐条写清楚，`observer_id` / `target_id` / `time_s` 三列就是语义本身。

输出文件
--------
`scene_entities.csv`   每行 = 某个时刻的某个实体（位置/速度/姿态/平台归属）
`scene_relations.csv`  每行 = 某个时刻的某一对**有向**关系
`scene_initial.json`   t=0 的完整快照（实体 + 关系 + 坐标体系声明）
`scene_final.json`     最后一个时间步的完整快照

用法
----
    python export_scene.py                                   # 单雷达旧场景，规则策略
    python export_scene.py --config config/multi_platform_scenario.json
    python export_scene.py --config config/multi_platform_scenario.json \\
        --policy fixed --out-dir output/scene_multi --max-steps 10
    python export_scene.py --pairs "RADAR_A>TGT_HIGH_FAST,RADAR_B>TGT_ESCORT"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.scene import Scene  # noqa: E402
from engine.simulator import Simulator  # noqa: E402
from strategy.power_policy import (  # noqa: E402
    FixedPowerPolicy,
    RandomPowerPolicy,
    RuleBasedPowerPolicy,
)

DEFAULT_OUT_DIR = "output/scene"
DEFAULT_MULTI_CONFIG = "config/multi_platform_scenario.json"

POLICIES = {
    "fixed": FixedPowerPolicy,
    "rule": RuleBasedPowerPolicy,
    "random": RandomPowerPolicy,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出场景快照（实体 + 有向几何关系）到 CSV/JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/radar_scenario_v1.json")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="rule")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="最多导出多少步（0 = 跑完整段仿真）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", default="",
                        help="只导出指定有向对，形如 'A>B,C>D'；留空表示导出全部对")
    parser.add_argument("--no-relations", action="store_true",
                        help="只导出实体快照，不导出关系（实体很多时可显著减小文件）")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_pairs(text: str) -> Optional[List[Tuple[str, str]]]:
    """把 'A>B,C>D' 解析成 [(A,B),(C,D)]。"""
    if not text.strip():
        return None
    pairs: List[Tuple[str, str]] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ">" not in item:
            raise ValueError(f"关系对 {item!r} 格式不对，应为 '观察者>目标'")
        observer, target = item.split(">", 1)
        pairs.append((observer.strip(), target.strip()))
    return pairs


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sim = Simulator(args.config)
    sim.load_config()
    sim.reset(seed=args.seed)

    scene: Scene = sim.scene
    pairs = parse_pairs(args.pairs)
    if pairs:
        for observer, target in pairs:
            scene.by_id(observer)  # 提前校验，写错 ID 立刻报错而不是写出空文件
            scene.by_id(target)

    policy = POLICIES[args.policy]()
    policy.reset()

    print("======== 场景快照导出 ========")
    print(f"配置        : {args.config}")
    print(f"策略        : {policy.describe()}")
    print(scene.describe())
    if pairs:
        print(f"关系子集    : {len(pairs)} 对（{args.pairs}）")
    print(f"输出目录    : {args.out_dir}\n")

    entities_csv = os.path.join(args.out_dir, "scene_entities.csv")
    relations_csv = os.path.join(args.out_dir, "scene_relations.csv")

    # --- t=0 快照 ---
    scene.write_json(os.path.join(args.out_dir, "scene_initial.json"))
    scene.write_entities_csv(entities_csv, append=False)
    if not args.no_relations:
        scene.write_relations_csv(relations_csv, pairs=pairs, append=False)

    step_limit = args.max_steps if args.max_steps > 0 else None
    steps_done = 0
    while not sim.is_done:
        if step_limit is not None and steps_done >= step_limit:
            break
        level = policy.select_level(sim)
        sim.step(int(level))
        sim.assert_scene_time_synchronized()  # 每步都校验时间同步
        scene.write_entities_csv(entities_csv, append=True)
        if not args.no_relations:
            scene.write_relations_csv(relations_csv, pairs=pairs, append=True)
        steps_done += 1
        if not args.quiet and steps_done % 10 == 0:
            print(f"  ...已导出 {steps_done} 步（t={sim.current_time:g}s）")

    final_json = scene.write_json(os.path.join(args.out_dir, "scene_final.json"))
    print(f"\n导出完成：{steps_done} 步")
    print(f"  → {entities_csv}")
    if not args.no_relations:
        print(f"  → {relations_csv}")
    print(f"  → {os.path.join(args.out_dir, 'scene_initial.json')}")
    print(f"  → {final_json}")

    # --- 回读校验：确认写出的 CSV 行数与实体×步数一致 ---
    expected_entity_rows = len(scene.entities) * (steps_done + 1)
    with open(entities_csv, "r", encoding="utf-8-sig") as handle:
        actual = sum(1 for _ in handle) - 1
    status = "OK" if actual == expected_entity_rows else "不一致！"
    print(f"\n回读校验：实体 CSV {actual} 行（期望 {expected_entity_rows}）→ {status}")
    if actual != expected_entity_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
