"""`AIProvider` 统一接口与工厂。

设计约束（架构级，不是风格偏好）
--------------------------------
1. **与具体大模型厂商解耦**：上层只依赖 `AIProvider` 的四个方法，
   换 OpenAI / DeepSeek / 通义 / 本地模型只需要换 provider 实现或改一行配置。
2. **AI 层不得控制雷达动作**：接口里没有任何「执行动作」的方法；
   `StateSnapshot` 里也没有动作指令字段。AI 只能描述、解释、建议。
3. **失败不影响主仿真**：provider 抛出的任何异常都由 `AIDiagnosisService`
   兜住并降级，仿真与训练流程完全不受影响。
4. **无 Key 也能跑**：默认 provider 是 `rule`（纯规则+模板，确定性、零依赖）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .schema import (
    CompareResult,
    DiagnosisResult,
    ExplainResult,
    ProviderInfo,
    ReportResult,
    StateSnapshot,
)

#: 四个能力的名字（与 HTTP 路由 /api/xxx 一致）
CAPABILITIES: List[str] = [
    "diagnose",
    "explain_decision",
    "compare_policies",
    "generate_report",
    # v4.5：证据链只读接口（实现了才能被 _call 调用；未实现则自动回退规则 provider）
    "explain_track",
    "explain_cooperation",
]


class AIProvider(ABC):
    """AI 认知诊断 provider 的统一接口。"""

    name: str = "base"
    model: str = ""

    # ------------------------------------------------------------------

    @abstractmethod
    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        """实时态势诊断：给出现场判断、异常原因与建议。"""

    @abstractmethod
    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        """策略解释：把反事实证据转成自然语言（不得自造事实）。"""

    @abstractmethod
    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        """多策略对比总结。"""

    @abstractmethod
    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        """实验总结报告。"""

    # ------------------------------------------------------------------
    # v4.5：证据链只读接口。默认返回 error，让 AIDiagnosisService._call
    # 自动降级到规则 provider（本地实现永远可用），而不是抛 AttributeError。

    def explain_track(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """单条航迹的证据链解释；本 provider 未实现。"""
        return DiagnosisResult(
            status=STATUS_ERROR,
            provider=self.name,
            summary="",
            error=f"{type(self).__name__} 未实现 explain_track",
        )

    def explain_cooperation(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """协同感知的证据链解释；本 provider 未实现。"""
        return DiagnosisResult(
            status=STATUS_ERROR,
            provider=self.name,
            summary="",
            error=f"{type(self).__name__} 未实现 explain_cooperation",
        )

    # ------------------------------------------------------------------

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            model=self.model,
            kind="local",
            requires_api_key=False,
            supports=list(CAPABILITIES),
        )

    def health(self) -> Dict[str, Any]:
        """健康检查；远程 provider 应在此真正探活。"""
        return {"provider": self.name, "model": self.model, "healthy": True}


# ----------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------

#: provider 别名 -> 规范名
PROVIDER_ALIASES: Dict[str, str] = {
    "rule": "rule",
    "rules": "rule",
    "local": "rule",
    "mock": "mock",
    "stub": "mock",
    "openai": "openai",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "dashscope": "qwen",
    "tongyi": "qwen",
    "http": "http",
    "remote": "http",
}


def available_providers() -> List[str]:
    """规范 provider 名列表（供 CLI `--ai-provider` 的 help 使用）。"""
    return ["rule", "mock", "openai", "deepseek", "qwen", "http"]


def _filtered_kwargs(factory: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """按构造函数签名过滤 kwargs。

    上层（CLI / service）会统一传 api_key/base_url/model 等远程参数，
    本地 provider 没有这些形参，直接透传会 TypeError。
    这里按签名过滤，既能共用一套调用代码，又不会静默吞掉拼错的参数
    （拼错的参数会体现在过滤后为空、由 provider 自己报错）。
    """
    import inspect

    if not kwargs:
        return {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def create_provider(name: str = "rule", **kwargs: Any) -> AIProvider:
    """按名字创建 provider。

    无 Key 也能跑的只有 rule / mock；其余为远程 provider，
    缺少 api_key 时会**抛错**（由 service 层兜住并降级），
    这是刻意的：远程 provider 不可用时不应静默变成别的东西。
    """
    canonical = PROVIDER_ALIASES.get((name or "rule").strip().lower(), name)

    if canonical == "rule":
        from .rule_provider import RuleProvider

        return RuleProvider(**_filtered_kwargs(RuleProvider, kwargs))
    if canonical == "mock":
        from .mock_provider import MockProvider

        return MockProvider(**_filtered_kwargs(MockProvider, kwargs))
    if canonical in ("openai", "deepseek", "qwen", "http"):
        from .http_provider import RemoteHTTPProvider

        return RemoteHTTPProvider(
            vendor=canonical, **_filtered_kwargs(RemoteHTTPProvider, {**kwargs, "vendor": canonical})
        )

    raise ValueError(
        f"未知的 AI provider：{name!r}。可用：{available_providers()}"
    )


class NullProvider(AIProvider):
    """空 provider：始终返回 degraded，用于「关掉 AI 层」的场景。"""

    name = "null"
    model = "none"

    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        return DiagnosisResult(
            provider=self.name, status="degraded", summary="AI 诊断层已关闭"
        )

    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        return ExplainResult(
            provider=self.name, status="degraded", explanation="AI 诊断层已关闭"
        )

    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        return CompareResult(provider=self.name, status="degraded")

    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        return ReportResult(
            provider=self.name, status="degraded", title="AI 诊断层已关闭"
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name, model="none", kind="local",
            requires_api_key=False, supports=[], notes="AI 层关闭时的占位实现",
        )
