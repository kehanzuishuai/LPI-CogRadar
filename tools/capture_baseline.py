"""冻结"修复前"基线：用于修复后逐项对照，说明结果变化的原因。

关键认识：**基线必须在改代码之前采集**。E3（功率不进主动传感器）
与 E4（有符号观测被裁剪）一旦修复，ideal/realistic 两组观测必然变化，
DQN 评测数字也会跟着变。没有事前基线，就无法区分
"修复导致的变化" 与 "随机波动/别的 bug"。

产物：`docs/baseline_pre_fix.json`
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: 参与源码摘要的包（只算"会改变结果"的代码，不算 output/docs）
#: **必须与 `run_manifest.SOURCE_PACKAGES` 一致**，否则同一份基线会出现两个摘要。
SOURCE_PACKAGES = ("engine", "sensor", "communication", "fusion", "ai", "rl",
                   "strategy", "models", "metrics", "explain", "validation",
                   "multi_target_stress", "resource_management", "tools")
SOURCE_FILES = ("experiment_config.py", "logging_utils.py", "main.py",
                "train_dqn.py", "evaluate_dqn.py",
                "evaluate_observation_modes.py",
                "evaluate_cooperative_sensing.py", "verify_v4.py",
                "run_manifest.py")


def source_digest() -> dict:
    """源码摘要：逐文件 sha256 + 汇总摘要（无 git 时的版本标识）。"""
    entries = {}
    for package in SOURCE_PACKAGES:
        base = os.path.join(ROOT, package)
        for directory, _dirs, files in os.walk(base):
            if "__pycache__" in directory:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, ROOT).replace("\\", "/")
                with open(path, "rb") as handle:
                    entries[relative] = hashlib.sha256(handle.read()).hexdigest()
    for name in SOURCE_FILES:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                entries[name] = hashlib.sha256(handle.read()).hexdigest()
    combined = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(entries.items())).encode()
    ).hexdigest()
    return {"n_files": len(entries), "digest": combined, "files": entries}


def run(command: list) -> dict:
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="冻结基线（修复前后各跑一次，用于逐项对照）")
    parser.add_argument("--out", default="baseline_pre_fix.json",
                        help="输出文件名（落在 docs/ 下）")
    parser.add_argument("--skip-ladder", action="store_true",
                        help="跳过观测模式天梯（它最慢，且需要 torch）")
    args = parser.parse_args()

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": args.out.replace(".json", ""),
        "note": ("E3（功率不进主动传感器）/E4（有符号观测被裁剪）等修复后，"
                 "ideal/realistic 相关数字允许变化，"
                 "但必须用本文件逐项对照说明变化原因。"),
        "python": sys.version,
        "source": source_digest(),
        "runs": {},
    }

    # 1) 单元测试
    snapshot["runs"]["unittest"] = run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"])

    # 2) 端到端验收
    snapshot["runs"]["verify_v4"] = run([sys.executable, "verify_v4.py"])

    # 3) 旧实验控制组：full 模式必须逐位不变（对照锚点）
    if not args.skip_ladder:
        snapshot["runs"]["observation_modes"] = run(
            [sys.executable, "evaluate_observation_modes.py",
             "--seeds", "42", "7", "13", "21", "33",
             "--out-dir", "output/observation_modes"])
    snapshot["artifacts"] = {}
    for relative in ("output/observation_modes/observation_mode_ladder.csv",
                     "output/cooperative/cooperative_comparison.csv",
                     "output/stress/stress_metrics.csv"):
        path = os.path.join(ROOT, relative)
        snapshot["artifacts"][relative] = (
            io.open(path, encoding="utf-8-sig").read() if os.path.exists(path)
            else None
        )

    out_dir = os.path.join(ROOT, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)
    print("baseline written:", out_path)
    print("source digest:", snapshot["source"]["digest"],
          f"({snapshot['source']['n_files']} files)")
    for name, result in snapshot["runs"].items():
        print(f"  {name}: returncode={result['returncode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
