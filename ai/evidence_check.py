"""AI 输出的证据校验（v4.5）。

为什么需要它
------------
v4.2 起 README 把"AI 解释不会编造"收紧成了
「AI 解释被限制在结构化证据范围内，并通过规则检查」。
**"通过规则检查"必须真的有一个检查器**，否则那句话只是措辞。

本模块是那道检查：远端大模型（或任何 provider）产出自然语言后，
逐项核对它引用的东西**是否真的存在于上下文里**：

1. **发现码白名单**：文本里出现的全大写代码必须属于 `FINDING_CODES`；
2. **标识符存在性**：提到的 `sensor_id` / `track_id` / `candidate_id`
   必须出现在上下文中（不存在即为编造）；
3. **数值可溯**：文本里的数值必须能在上下文里找到（含四舍五入容差），
   否则标记为不可溯。

校验失败时的行为
----------------
**标记 `evidence_check_failed` 并回退本地 rule provider**（见 `ai/service.py`）。
不回退的话，一次幻觉就会直接进入诊断结论。

诚实边界
--------
这是**规则检查，不是证明**：
* 它只能发现"引用了不存在的东西"，不能发现"引用了真实数值但结论错误"；
* 数值匹配用容差与舍入，存在误报/漏报可能；
* 因此正确表述始终是"被限制在结构化证据范围内并通过规则检查"，
  **不是**"不会编造"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ai.schema import FINDING_CODES

#: 形如全大写代码的词（用于发现码白名单校验）
_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")

#: 数值（含小数与负号）
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

#: 常见小整数/年份等无需溯源的数值（计数、比例说明等）
_IGNORABLE_NUMBERS: Set[str] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "24", "60", "100", "1000",
}

#: 形如 SENSOR_xxx / RADAR_xxx / xxx-T1 / xxxx-C1 的标识符
_ID_PATTERN = re.compile(
    r"\b(?:SENSOR_[A-Za-z0-9_]+|[A-Z][A-Za-z0-9_]*-\s?[TCFA]\d+|[A-Z][A-Z0-9_]{2,})\b"
)


@dataclass
class EvidenceCheckResult:
    """校验结果。"""

    passed: bool = True
    violations: List[str] = field(default_factory=list)
    checked_numbers: int = 0
    checked_codes: int = 0
    checked_ids: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": not self.passed,
            "violations": list(self.violations),
            "checked_numbers": self.checked_numbers,
            "checked_codes": self.checked_codes,
            "checked_ids": self.checked_ids,
        }


def collect_numbers(payload: Any) -> Set[float]:
    """递归收集结构化上下文里的全部数值（用于数值可溯校验）。"""
    out: Set[float] = set()
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, bool):
            continue
        if isinstance(node, (int, float)):
            out.add(float(node))
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return out


def collect_identifiers(payload: Any) -> Set[str]:
    """递归收集上下文里出现过的字符串（用于标识符存在性校验）。"""
    out: Set[str] = set()
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            out.add(node)
            # 连字符形式也拆开收集，便于匹配 "RADAR_A-T1"
            out.update(part for part in re.split(r"[,\s]+", node) if part)
        elif isinstance(node, dict):
            stack.extend(node.keys())
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return out


def _number_is_traceable(value: float, known: Set[float],
                         rel_tol: float = 0.02, abs_tol: float = 0.05) -> bool:
    """数值是否能在上下文里找到（含舍入容差）。"""
    for candidate in known:
        if abs(value - candidate) <= max(abs_tol, abs(candidate) * rel_tol):
            return True
        # 上下文里常见"已四舍五入"的值（round(x, 3)），放宽一次
        if abs(round(candidate, 2) - round(value, 2)) <= max(abs_tol, abs(candidate) * rel_tol):
            return True
    return False


def validate_text_against_context(
    text: str,
    context: Dict[str, Any],
    allowed_codes: Optional[Iterable[str]] = None,
    extra_identifiers: Optional[Iterable[str]] = None,
) -> EvidenceCheckResult:
    """校验自然语言是否只引用了上下文里真实存在的东西。

    参数
    ----
    text       : 待校验的自然语言（provider 输出的 summary / findings 文本）
    context    : 结构化上下文（例如 `StateSnapshot.to_dict()`）
    allowed_codes     : 允许出现的发现码集合；默认 `FINDING_CODES`
    extra_identifiers : 额外允许的标识符（例如本轮诊断自身产出的 track_id）
    """
    result = EvidenceCheckResult()
    if not text:
        return result

    codes_allowed = set(allowed_codes) if allowed_codes is not None else set(FINDING_CODES)
    known_numbers = collect_numbers(context)
    known_ids = collect_identifiers(context)
    if extra_identifiers:
        known_ids.update(str(x) for x in extra_identifiers)

    # --- 1) 发现码白名单 ---
    for token in set(_CODE_PATTERN.findall(text)):
        if token in codes_allowed:
            result.checked_codes += 1
            continue
        # 允许数字/单位类的全大写词（如 Hz / W / dB / SNR）不视为发现码
        if token in _UNIT_LIKE:
            continue
        # ⚠️ 上下文里**真实存在**的标识符（如 SENSOR_A）也是全大写，
        # 不能当成"编造的发现码"。这个假阳性会让合法诊断被误判为幻觉，
        # 进而错误触发回退 —— 必须在这里放行。
        if token in known_ids:
            result.checked_ids += 1
            continue
        result.violations.append(
            f"发现码不在白名单：{token}（可能是编造的代码）"
        )
        result.checked_codes += 1

    # --- 2) 数值可溯 ---
    for raw in set(_NUMBER_PATTERN.findall(text)):
        if raw in _IGNORABLE_NUMBERS:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        result.checked_numbers += 1
        if not _number_is_traceable(value, known_numbers):
            result.violations.append(f"数值无法在上下文中溯源：{raw}")

    # --- 3) 标识符存在性 ---
    for token in set(_ID_PATTERN.findall(text)):
        if token in codes_allowed or token in _UNIT_LIKE:
            continue
        if token in known_ids:
            result.checked_ids += 1
            continue
        # 允许常见短语/单位（全大写但非标识符）
        if token in _NON_IDENTIFIER_WORDS:
            continue
        # 只有"看起来像 ID"的才报（含下划线或 -T/-C 后缀）
        if "_" in token or re.search(r"-[TCFA]\d+$", token):
            result.violations.append(f"引用了上下文中不存在的标识符：{token}")
            result.checked_ids += 1

    result.passed = not result.violations
    return result


#: 单位/量纲类的全大写词，不当作发现码
_UNIT_LIKE: Set[str] = {
    "W", "KW", "DB", "DBM", "HZ", "KHZ", "MHZ", "GHZ", "J", "KJ", "M", "KM",
    "S", "MS", "MPS", "SNR", "SINR", "PD", "PINT", "RCS", "FOV", "LOS", "ID",
    "RMSE", "ENU", "AABB", "KF", "CV", "FN", "FP", "AI", "JSON", "CSV", "HTML",
    "LPI", "ESM", "RWR", "TGT", "JAM", "RADAR", "FAQ", "NOTE",
}

#: 全大写但属于普通词汇/缩写，不当作标识符
_NON_IDENTIFIER_WORDS: Set[str] = {
    "AI", "RMSE", "SNR", "PD", "FOV", "ID", "OK", "N/A",
}
