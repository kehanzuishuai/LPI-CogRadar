"""可解释决策包。

    counterfactual.py  对可行动作做反事实试算，产出结构化证据（不做自然语言解释）

自然语言由 `ai/` 认知诊断层生成，且只允许引用本模块给出的证据代码与数值。
"""

from .counterfactual import (
    DEFAULT_FOCUS_POWERS_W,
    EVIDENCE_CODES,
    CounterfactualOutcome,
    CounterfactualReport,
    build_counterfactual_report,
    build_key_moment_reports,
)

__all__ = [
    "DEFAULT_FOCUS_POWERS_W",
    "EVIDENCE_CODES",
    "CounterfactualOutcome",
    "CounterfactualReport",
    "build_counterfactual_report",
    "build_key_moment_reports",
]
