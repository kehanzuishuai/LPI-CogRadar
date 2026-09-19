"""运行清单（run manifest）：可追溯 + **禁止互相覆盖**。

为什么需要
----------
同一份汇总报告被不同实验覆盖，会让"这个数字是哪次跑出来的"无从查证：
外部复核时发现 `output/stress/stress_metrics.csv` 里只剩一次 S2 的输出，
而别的审计文件还在——正是覆盖造成的。只要报告路径固定、又允许重跑，
这种事故迟早发生。

本模块给出最小可用的运行清单：

    output/runs/<run_id>/            每次运行独占一个目录
        manifest.json                运行标识 + 配置摘要 + 源码摘要 + 产物清单
        <产物文件...>                 该次运行的 CSV / JSON / HTML

    output/runs/index.json           追加式索引（只增不改）

`run_id` 由「UTC 时间戳 + 配置摘要前 8 位 + 首个种子」构成，
**确定性可读**且不会重名。配置摘要是对全部实验参数做规范化 JSON 后的 sha256，
源码摘要覆盖所有会改变结果的 `.py`。

两条纪律
--------
1. **同一目录不允许被第二次运行写入**（`RunManifest.claim` 会拒绝）；
2. 结论只允许引用 manifest 里登记过的产物——清单外的数字视为不可追溯。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

#: 会改变实验结果的源码范围（与 `tools/capture_baseline.py` 保持一致）
#:
#: `resource_management` 与 `tools` 是 v4.5 之后新增的**会产出实验产物**的范围：
#: 前者出 `output/runs/<run_id>/resource_management/`，后者出
#: `.../scheduler_baselines/`。不把它们纳入摘要就会出现"产物清单齐了、
#: 却答不出这份产物是哪版代码跑出来的"——正是引入 run_manifest 要防的事。
SOURCE_PACKAGES = ("engine", "sensor", "communication", "fusion", "ai", "rl",
                   "strategy", "models", "metrics", "explain", "validation",
                   "multi_target_stress", "resource_management", "tools")
SOURCE_FILES = ("experiment_config.py", "logging_utils.py", "main.py",
                "train_dqn.py", "evaluate_dqn.py",
                "evaluate_observation_modes.py",
                "evaluate_cooperative_sensing.py", "verify_v4.py",
                "run_manifest.py")

RUNS_SUBDIR = os.path.join("output", "runs")


def source_digest(root: str) -> Dict[str, Any]:
    """源码摘要：逐文件 sha256 + 汇总摘要（本工程尚无 git，用它当版本标识）。"""
    entries: Dict[str, str] = {}
    for package in SOURCE_PACKAGES:
        base = os.path.join(root, package)
        for directory, _dirs, files in os.walk(base):
            if "__pycache__" in directory:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, root).replace("\\", "/")
                with open(path, "rb") as handle:
                    entries[relative] = hashlib.sha256(handle.read()).hexdigest()
    for name in SOURCE_FILES:
        path = os.path.join(root, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                entries[name] = hashlib.sha256(handle.read()).hexdigest()
    combined = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(entries.items())).encode()
    ).hexdigest()
    return {"n_files": len(entries), "digest": combined, "files": entries}


def config_digest(config: Dict[str, Any]) -> str:
    """配置摘要：规范化 JSON（键排序、无空格）后的 sha256。"""
    normalized = json.dumps(config, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def latest_run_dir(root: str = ".", subdir: str = RUNS_SUBDIR,
                   tool: Optional[str] = None) -> Optional[str]:
    """最近一次运行的目录（按 `index.json` 里的 created_at，回退到目录名排序）。

    自检与报告需要"看最新一次运行"，但**不允许**把不同运行混在一份汇总里；
    因此这里只返回**单个**目录，而不是去聚合多个运行。
    """
    index_path = os.path.join(root, subdir, "index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, ValueError):
            index = []
        if tool:
            index = [row for row in index if row.get("tool") == tool]
        if index:
            newest = max(index, key=lambda row: str(row.get("created_at", "")))
            candidate = os.path.join(root, subdir, str(newest.get("run_id", "")))
            if os.path.isdir(candidate):
                return candidate
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        return None
    # 回退：索引不可用时按目录名排序；指定了 tool 就逐个读 manifest 过滤
    directories = sorted(
        (name for name in os.listdir(base)
         if os.path.isdir(os.path.join(base, name))),
        reverse=True,
    )
    for name in directories:
        candidate = os.path.join(base, name)
        if not tool:
            return candidate
        manifest_path = os.path.join(candidate, "manifest.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                if json.load(handle).get("tool") == tool:
                    return candidate
        except (OSError, ValueError):
            continue
    return None if tool else None


@dataclass
class RunManifest:
    """一次运行的清单。"""

    command: str
    config: Dict[str, Any] = field(default_factory=dict)
    seeds: Sequence[int] = field(default_factory=tuple)
    root: str = "."
    #: 产出这次运行的工具名（如 `multi_target_stress` / `resource_management`）。
    #: 没有它就无法回答"最新一次运行为什么是这个目录"——
    #: 多个工具各自写 `output/runs/<run_id>/`，取"最新"会取错工具的那次。
    tool: str = ""
    created_at: str = ""
    run_id: str = ""
    config_hash: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    _dir: str = ""

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.config_hash = config_digest(self.config)
        if not self.source:
            self.source = source_digest(self.root)
        if not self.run_id:
            stamp = self.created_at.replace(":", "").replace("-", "")
            stamp = stamp.split(".")[0]          # 20260919T031500
            seed = str(self.seeds[0]) if self.seeds else "noseed"
            self.run_id = f"{stamp}_{self.config_hash[:8]}_s{seed}"

    # ------------------------------------------------------------------

    def run_dir(self, create: bool = True) -> str:
        """本次运行的独占目录。"""
        if not self._dir:
            self._dir = os.path.join(self.root, RUNS_SUBDIR, self.run_id)
        if create:
            os.makedirs(self._dir, exist_ok=True)
        return self._dir

    def artifact_path(self, name: str) -> str:
        """产物路径（落在本次运行的目录内，**不会**覆盖别次运行）。"""
        return os.path.join(self.run_dir(), name)

    def claim(self, path: str, allow_overwrite: bool = False) -> str:
        """声明要写一个产物；若该文件属于**另一次运行**则拒绝。

        判定依据是同目录下已有的 `manifest.json` 里的 `run_id`：
        同一运行内重写自己的文件是允许的，写入别人的运行目录则报错。
        """
        directory = os.path.dirname(os.path.abspath(path))
        existing = os.path.join(directory, "manifest.json")
        if os.path.exists(existing) and os.path.exists(path):
            try:
                with open(existing, "r", encoding="utf-8") as handle:
                    other = json.load(handle)
            except (OSError, ValueError):
                other = {}
            other_id = other.get("run_id")
            if other_id and other_id != self.run_id and not allow_overwrite:
                raise RuntimeError(
                    f"拒绝覆盖：{path} 属于另一次运行 run_id={other_id}，"
                    f"本次为 {self.run_id}。不同实验不得写入同一份汇总报告——"
                    "请使用 `RunManifest.artifact_path` 让产物落在运行目录内。"
                )
        return path

    def record(self, path: str) -> Dict[str, Any]:
        """登记一个已写出的产物（路径 + sha256 + 字节数）。"""
        absolute = os.path.abspath(path)
        entry = {
            "path": os.path.relpath(absolute, os.path.abspath(self.root))
            .replace("\\", "/"),
            "bytes": os.path.getsize(absolute),
            "sha256": file_sha256(absolute),
        }
        self.artifacts.append(entry)
        return entry

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "tool": self.tool,
            "command": self.command,
            "python": sys.version,
            "cwd": os.path.abspath(self.root),
            "config": self.config,
            "config_digest": self.config_hash,
            "source_digest": self.source.get("digest", ""),
            "source_files": self.source.get("n_files", 0),
            "seeds": list(self.seeds),
            "artifacts": list(self.artifacts),
            "note": ("产物清单之外的数字不可追溯；不同运行写入各自的 "
                     "output/runs/<run_id>/ 目录，不互相覆盖。"),
        }

    def write(self) -> str:
        """写出 manifest.json 并追加到全局索引（索引只增不改）。"""
        directory = self.run_dir()
        path = os.path.join(directory, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

        index_path = os.path.join(self.root, RUNS_SUBDIR, "index.json")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        index: List[Dict[str, Any]] = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as handle:
                    index = json.load(handle)
            except (OSError, ValueError):
                index = []
        entry = {
            "run_id": self.run_id,
            "tool": self.tool,
            "created_at": self.created_at,
            "command": self.command,
            "config_digest": self.config_hash,
            "source_digest": self.source.get("digest", ""),
            "n_artifacts": len(self.artifacts),
        }
        index = [row for row in index if row.get("run_id") != self.run_id]
        index.append(entry)
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
        return path
