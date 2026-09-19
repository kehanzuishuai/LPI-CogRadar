"""集中式学习调度基线的评测入口。

数据纪律（与 `docs/learning_protocol.md` §5 一致）
--------------------------------------------------
* 默认只评 **validation 分区**（checkpoint 选择用的就是它）；
* **test 分区封存**：必须显式 `--release-test` 才会调用
  `get_split("test", release_test=True)`，且打开前应已冻结代码、超参、
  checkpoint 选择规则与指标。本脚本**不做**任何按 test 结果回调的机制。

用法
----
    D:\\anaconda\\envs\\pytorch_env\\python.exe -m rl_resource.evaluate \\
        --policy output/rl_resource/baseline/policy.pt
    # 最终评测（确认冻结后）：
    python -m rl_resource.evaluate --policy ... --release-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import ensure_utf8_console  # noqa: E402

import torch  # noqa: E402

from resource_management.learning_protocol import (  # noqa: E402
    SealedTestSplitError, get_split,
)
from rl_resource.policy import ActorCritic  # noqa: E402
from rl_resource.train import TrainConfig, evaluate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(description="评测集中式学习调度基线")
    parser.add_argument("--policy", required=True, help="policy.pt 路径")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--release-test", action="store_true",
                        help="显式解封测试分区（仅在冻结所有选择后使用）")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=4)
    parser.add_argument("--out", default="", help="把结果写到该 JSON 路径")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.release_test:
        print("测试分区已封存：如确需评测，请显式传 --release-test")
        return 2
    try:
        split = get_split(args.split, release_test=args.release_test)
    except SealedTestSplitError as exc:
        print(str(exc))
        return 2

    model, extra = ActorCritic.load(args.policy)
    cfg = TrainConfig(steps=args.steps, max_nodes=args.max_nodes,
                      scenarios=split.scenarios, seeds=split.seeds,
                      eval_scenarios=split.scenarios, eval_seeds=split.seeds)
    device = torch.device("cpu")

    masked = evaluate(model, cfg, device, split.scenarios, split.seeds)
    unmasked = evaluate(model, cfg, device, split.scenarios, split.seeds,
                        use_mask=False)

    report: Dict[str, Any] = {
        "policy": os.path.abspath(args.policy),
        "split": args.split,
        "test_released": bool(args.split == "test" and args.release_test),
        "scenarios": list(split.scenarios),
        "seeds": list(split.seeds),
        "checkpoint_extra": extra,
        "with_mask": {key: value for key, value in masked.items()
                      if key != "rows"},
        "without_mask": {key: value for key, value in unmasked.items()
                         if key != "rows"},
        "per_episode": masked["rows"],
    }
    print("策略：%s" % args.policy)
    print("分区：%s（场景 %s，种子 %s）"
          % (args.split, list(split.scenarios), list(split.seeds)))
    print("带 mask ：返回 %+.3f | 资源消耗 %.4f | 守恒 %s | 执行器拒绝率 %.4f "
          "| 对账最大误差 %.2e"
          % (masked["mean_return"], masked["mean_resource_cost"],
             masked["conservation_all_ok"], masked["executor_rejection_rate"],
             masked["max_cost_reconciliation_error"]))
    print("动作合法性覆盖率：%s" % {
        key: round(value, 4) for key, value
        in masked["legal_action_rates"].items()})
    print("不带 mask：argmax 非法率 %.4f | 非法动作概率质量 %.4f"
          % (unmasked["illegal_action_rate"],
             unmasked["illegal_probability_mass"]))
    print("提示：**部署必须带 mask**。带 mask 时非法率为 0 是构造保证，")
    print("      不带 mask 的数字说明策略本身并没有学会规避非法动作。")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2,
                      default=str)
        print("已写出：%s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
