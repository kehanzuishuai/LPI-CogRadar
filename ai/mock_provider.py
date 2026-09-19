"""MockProvider：最小可用的确定性桩实现。

用途
----
1. **单元测试 / 离线演示**：不依赖任何规则逻辑，返回固定结构，
   用来验证「AI 层挂了也不会影响主仿真」这条架构约束；
2. **接口契约验证**：新接入一个远程 provider 时，可以先用 mock 跑通调用链；
3. **可注入故障**：`fail_rate=1.0` 时四个能力全部返回 `status="error"`，
   用来测试 service 层的降级路径。

它不是给生产用的——默认 provider 是 `rule`。
"""

from __future__ import annotations

from typing import Any, Dict

from .provider import AIProvider
from .schema import (
    CompareResult,
    DiagnosisResult,
    ExplainResult,
    Finding,
    ProviderInfo,
    ReportResult,
    STATUS_ERROR,
    STATUS_OK,
    StateSnapshot,
)


class MockProvider(AIProvider):
    """固定响应桩，可选注入故障。"""

    name = "mock"
    model = "mock-v1"

    def __init__(self, fail_rate: float = 0.0, raise_on_call: bool = False) -> None:
        self.fail_rate = float(fail_rate)
        self.raise_on_call = bool(raise_on_call)
        self.calls: int = 0

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            model=self.model,
            kind="local",
            requires_api_key=False,
            supports=["diagnose", "explain_decision", "compare_policies", "generate_report"],
            notes="确定性桩实现，用于测试与验证降级路径；fail_rate=1.0 可注入故障",
        )

    # ------------------------------------------------------------------

    def _should_fail(self) -> bool:
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("MockProvider 被配置为必然抛错（用于测试降级）")
        if self.fail_rate >= 1.0:
            return True
        if self.fail_rate > 0.0:
            return (self.calls % max(1, int(1.0 / self.fail_rate))) == 0
        return False

    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        if self._should_fail():
            return DiagnosisResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error="mock failure injected",
            )
        return DiagnosisResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            severity="info",
            summary=f"[mock] step={snapshot.step_index} pd={snapshot.pd_min:.3f}",
            findings=[
                Finding(
                    code="DETECTION_WELL_WITHIN_MARGIN",
                    severity="info",
                    title="[mock] 固定响应",
                    message="本响应由 MockProvider 生成，仅用于接口验证。",
                    evidence={"step_index": snapshot.step_index, "pd_min": snapshot.pd_min},
                )
            ],
            recommendations=["[mock] 维持当前策略"],
            confidence=0.1,
        )

    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        if self._should_fail():
            return ExplainResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error="mock failure injected",
            )
        report = payload.get("counterfactual") or payload
        return ExplainResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            step_index=int(report.get("step_index", 0)),
            direction=str(report.get("direction", "hold")),
            explanation="[mock] 固定解释文本，仅用于接口验证。",
            verdict_code=str((report.get("verdict") or {}).get("code", "")),
            evidence_codes=list((report.get("verdict") or {}).get("evidence_codes", []) or []),
            confidence=0.1,
        )

    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        if self._should_fail():
            return CompareResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error="mock failure injected",
            )
        labels = [str(r.get("label", "?")) for r in (payload.get("summaries") or [])]
        return CompareResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            ranking=labels,
            highlights=[f"[mock] 收到 {len(labels)} 个策略"],
            confidence=0.1,
        )

    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        if self._should_fail():
            return ReportResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error="mock failure injected",
            )
        return ReportResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            title=str(payload.get("title") or "[mock] 报告"),
            sections=[{"heading": "mock", "body": "固定内容，仅用于接口验证。"}],
            conclusion="[mock] 固定结论。",
            confidence=0.1,
        )
