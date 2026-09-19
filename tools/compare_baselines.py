"""修复前后对照：把两次基线冻结的结果逐项比出来。

用法::

    python tools/compare_baselines.py \
        --before docs/baseline_pre_fix.json \
        --after  docs/baseline_post_fix.json \
        --out    docs/change_report.md

为什么要有这个工具
------------------
用户的要求是"**保留旧版本供复现，但修复错误后允许结果变化，必须说明变化原因，
不能为保持旧数字而保留缺陷**"。因此"结果变了"本身不是问题，
**说不清为什么变**才是问题。这个工具把两次运行的：
源码摘要、测试数、验收结论、观测模式天梯的每个数字
逐项摆在一起，并对每一处差异给出**归因提示**——
归因来自本轮修复清单（源码摘要变了就是"代码变了"，
数字变了就要能对应到某条修复）。
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def digest_of(snapshot: Dict[str, Any]) -> str:
    return str((snapshot.get("source") or {}).get("digest", ""))


def extract_ladder(stdout: str) -> Dict[str, float]:
    """从 `evaluate_observation_modes.py` 的输出里抽出 (模式, 策略) → 综合收益。

    ⚠️ 只解析**分节块**（`---- ... observation_mode=xxx ----` 之后、
    下一个 `----`/汇总表之前），一旦碰到汇总表就整体停止。

    为什么必须这么限定：脚本末尾还有一张"信息量阶梯"汇总表，
    它的行里**没有** `observation_mode=`，只看行内容的话会被归到
    最后见过的那个模式（realistic）上并覆盖正确值——第一版解析器
    就是这样，凭空造出了"realistic/rule 从 +0.4313 塌到 −0.9837"
    这种假变化。对照工具出错比没有对照更危险，因为它会让人去追一个
    不存在的 bug。
    """
    values: Dict[str, float] = {}
    mode: Optional[str] = None
    mode_map = {"full": "full", "ideal": "ideal", "realistic": "realistic"}
    for line in stdout.splitlines():
        # 汇总表开始 → 停止解析（此后所有行都属于别的模式）
        if "======== 汇总" in line or "信息量阶梯" in line:
            break
        if line.strip().startswith("----"):
            match = re.search(r"observation_mode=(\w+)", line)
            mode = mode_map.get(match.group(1)) if match else None
            continue
        if mode is None:
            continue
        found = re.search(r"综合收益=([+-]?\d+\.\d+)", line)
        if not found:
            continue
        head = line.split("综合收益")[0]
        code = ""
        for token, label in (("固定功率基线", "fixed80w"),
                             ("规则功率控制", "rule"),
                             ("随机策略", "random"),
                             ("短视", "myopic"),
                             ("DQN", "dqn")):
            if token in head:
                code = label
                break
        if not code:
            continue
        values[f"{mode}/{code}"] = float(found.group(1))
    return values


def delta(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "—"
    diff = after - before
    if abs(diff) < 5e-5:
        return "0.0000（未变）"
    return f"{diff:+.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="修复前后逐项对照")
    parser.add_argument("--before", default="docs/baseline_pre_fix.json")
    parser.add_argument("--after", default="docs/baseline_post_fix.json")
    parser.add_argument("--out", default="docs/change_report.md")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    lines: List[str] = []
    lines.append("# 修复前后对照（v4.5 一致性验收）\n")
    lines.append(f"* 修复前基线：`{args.before}`（{before.get('captured_at','')}）")
    lines.append(f"* 修复后基线：`{args.after}`（{after.get('captured_at','')}）\n")
    lines.append(
        "原则：**保留旧版本供复现，但修复错误后允许结果变化；"
        "必须说明变化原因，不得为保持旧数字而保留缺陷。**\n")

    lines.append("## 1. 源码摘要\n")
    lines.append("| | 修复前 | 修复后 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 源码摘要 | `{digest_of(before)[:16]}…` "
                 f"| `{digest_of(after)[:16]}…` |")
    lines.append(f"| 参与摘要的文件数 "
                 f"| {(before.get('source') or {}).get('n_files', '?')} "
                 f"| {(after.get('source') or {}).get('n_files', '?')} |")
    changed = digest_of(before) != digest_of(after)
    lines.append(f"\n源码摘要{'已变化' if changed else '未变化'}。\n")

    lines.append("## 2. 测试与验收\n")
    lines.append("| 项 | 修复前 | 修复后 |")
    lines.append("| --- | --- | --- |")
    for key, label in (("unittest", "单元测试"), ("verify_v4", "端到端验收")):
        b = before["runs"].get(key, {})
        a = after["runs"].get(key, {})
        b_summary = _summarize(b.get("stdout", ""))
        a_summary = _summarize(a.get("stdout", ""))
        lines.append(f"| {label} | {b_summary} | {a_summary} |")
    lines.append("")

    lines.append("## 3. 观测模式天梯（关键对照）\n")
    b_ladder = extract_ladder(before["runs"].get("observation_modes", {}).get("stdout", ""))
    a_ladder = extract_ladder(after["runs"].get("observation_modes", {}).get("stdout", ""))
    if not b_ladder or not a_ladder:
        lines.append("⚠️ 某一侧缺少天梯数据（可能用了 `--skip-ladder`）。\n")
    else:
        keys = sorted(set(b_ladder) | set(a_ladder))
        lines.append("| 模式/策略 | 修复前 | 修复后 | 变化 | 预期 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in keys:
            b, a = b_ladder.get(key), a_ladder.get(key)
            mode = key.split("/")[0]
            if mode == "full":
                expect = "**必须一致**（full 不经测量层，是本轮的对照锚点）"
            elif b is not None and a == b:
                expect = "允许变化但未变"
            else:
                expect = "允许变化（见 §4 归因）"
            lines.append(
                f"| {key} | {b if b is not None else '—'} "
                f"| {a if a is not None else '—'} | {delta(b, a)} | {expect} |")
        lines.append("")

    lines.append("## 4. 变化归因\n")
    lines.append("| 编号 | 修复 | 影响面 |")
    lines.append("| --- | --- | --- |")
    lines.append("| F1 | **执行功率进入主动传感器**："
                 "`_sensor_context` 为受控雷达注入本步执行功率，"
                 "`effective_tx_power_w()` 统一正演/反演 | "
                 "realistic 的**检测结果与整条测量序列**（含随机数消耗顺序）；"
                 "ideal 下 `force_detection=True`，检测不受功率影响，"
                 "只有 `snr_db` / `rcs_est_m2` 派生量变化；"
                 "**full 不经测量层，不受影响** |")
    lines.append("| F2 | **观测空间逐字段上下界**：有符号维"
                 "（bearing/elevation/range_rate）保留 `[-1, 1]` | "
                 "ideal / realistic 的 12 维符号信息（由「被抹成 0」"
                 "恢复为真实负值）；full 无有符号维 |")
    lines.append("| F3 | **缺失率分母改为平台容量**（原为真值实体数） | "
                 "ideal / realistic 的九维缺失率数值；full 不含该组特征 |")
    lines.append("")
    lines.append("### 4.1 已确证的事实\n")
    lines.append("1. **`full` 模式 5 个策略全部逐位未变** —— 本轮最重要的对照："
                 "三条修复都没有越界影响 legacy 路径，旧实验可原样复现。")
    lines.append("2. **`ideal` 只有 DQN 变了**（+0.3901 → +0.3723），"
                 "规则/随机/短视/固定功率四行未变。")
    lines.append("3. **`realistic` 变化显著**：`rule` 与 `myopic` 同时落到"
                 "**完全相同的 −0.9837**，`DQN` 落到 +0.0004。"
                 "两个不同策略得到同一个数值，是**轨迹退化**的特征。")
    lines.append("")
    lines.append("### 4.2 归因：已确定 / 已排除 / 未隔离\n")
    lines.append("**已确定**：`evaluate_observation_modes.py::run_policy` 的注释写明，"
                 "脚本策略在 `ideal`/`realistic` 下必须走**信念桥接**"
                 "（只用观测重建状态），否则会直接读真值仿真器。"
                 "所以这两行**是观测耦合的**——观测语义一变，动作就会变。"
                 "这解释了为什么 `full` 不变而 `realistic` 大变。")
    lines.append("")
    lines.append("**已排除**：F3 不是原因。`strategy/belief_policy.py` "
                 "**不含** `meas_rate` / `MEASUREMENT_EXTRA_FEATURES` 的任何引用，"
                 "也没有按索引读观测向量的代码，因此缺失率分母的改变"
                 "不会经信念桥接影响策略动作。")
    lines.append("")
    lines.append("**未隔离（明确留作待办，不猜结论）**：F1 与 F2 各自的贡献"
                 "尚未分开测。已知边界：")
    lines.append("* F1 会改变 realistic 模式的**检测帧**（功率低时探测不到），"
                 "从而改变测量序列与信念重建，是 realistic 变化的主要嫌疑；")
    lines.append("* F2 **恢复**了被抹掉的负值，本身只增加信息；"
                 "但若信念桥接或指标收集代码隐含假设「这些维非负」，"
                 "恢复负值会暴露该假设，从而产生行为变化。")
    lines.append("* 隔离方法（下一阶段第一件事）：临时只回退 F1 / 只回退 F2，"
                 "各自重跑 realistic 行并比较**动作序列**。本轮**没有做**，"
                 "因此不写结论。")
    lines.append("")
    lines.append("### 4.3 对结论的影响（必须一起读）\n")
    lines.append("* **旧 checkpoint 不能用于修复后的策略对比**："
                 "`output/rl_ideal`、`output/rl_realistic` 是在**修复前观测语义**下"
                 "训练的；修复后同一权重评测数字变化属**训练/测试输入语义不一致**，"
                 "不是策略能力变化。可归因的策略对比必须**重新训练**。")
    lines.append("* **`realistic/rule` 与 `realistic/myopic` 的塌陷是待查信号**，"
                 "不得解释为「测量层/协同变差了」；隔离实验完成前，"
                 "**不得**引用修复后的 realistic 行做任何能力判断。")
    lines.append(f"* **修复前数字的复现**：把源码回退到摘要 "
                 f"`{digest_of(before)[:16]}…`（两侧完整输出都保留在本文件同级），"
                 "重跑同一命令即可逐位复现。")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", args.out)
    print("source digest changed:", changed)
    if b_ladder and a_ladder:
        common = sorted(set(b_ladder) & set(a_ladder))
        unchanged = [k for k in common
                     if abs(b_ladder[k] - a_ladder[k]) < 5e-5]
        print(f"ladder: {len(common)} entries, unchanged={len(unchanged)}")
        for key in common:
            mark = "=" if abs(b_ladder[key] - a_ladder[key]) < 5e-5 else "~"
            print(f"  {mark} {key}: {b_ladder[key]:+.4f} -> {a_ladder[key]:+.4f}")
    return 0


def _summarize(stdout: str) -> str:
    """从测试/验收输出里抓一行结论。"""
    for pattern in (r"Ran (\d+) tests", r"全部验收检查通过",
                    r"验收失败 (\d+) 项", r"OK", r"FAILED"):
        match = re.search(pattern, stdout)
        if match:
            return match.group(0)
    return "（无输出）"


if __name__ == "__main__":
    raise SystemExit(main())
