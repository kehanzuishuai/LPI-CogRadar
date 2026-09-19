"""AIDiagnosisService：四个能力的统一门面，并保证 **AI 失败绝不影响主仿真**。

三层保护
--------
1. **异常隔离**：provider 的任何异常都在这里被捕获，绝不向上抛到仿真/训练循环；
2. **自动降级**：主 provider 失败时按 `fallback` 链切换（默认回退到本地 `rule`）；
3. **耗时与缓存**：`timeout_s` 只是记录（Python 线程无法强杀），
   但 `max_calls` 与缓存能防止误用导致训练被拖慢。

用法
----
    service = AIDiagnosisService.create("rule")
    result = service.diagnose(snapshot)          # 永远返回 DiagnosisResult
    result = service.explain_decision(payload)
    result = service.compare_policies(payload)
    result = service.generate_report(payload)

    # 关掉 AI 层
    service = AIDiagnosisService.disabled()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from logging_utils import utc_timestamp

from .provider import AIProvider, NullProvider, create_provider
from .schema import (
    CompareResult,
    DiagnosisResult,
    ExplainResult,
    ReportResult,
    STATUS_DEGRADED,
    STATUS_ERROR,
    StateSnapshot,
)

logger = logging.getLogger("lpi.ai.service")


#: 需要做证据校验并可能回退的 provider（本地规则 provider 不需要）
REMOTE_PROVIDER_NAMES = {"deepseek", "openai", "http", "remote",
                         "qwen", "mock_remote", "RemoteHTTPProvider"}


class AIDiagnosisService:
    """AI 认知诊断服务（四个能力的门面）。"""

    def __init__(
        self,
        provider: Optional[AIProvider] = None,
        fallback: Optional[Sequence[AIProvider]] = None,
        cache_size: int = 256,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.provider = provider if (provider is not None and self.enabled) else NullProvider()
        self.fallback: List[AIProvider] = list(fallback or [])
        self.cache_size = int(cache_size)
        self._cache: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {
            "calls": 0,
            "ok": 0,
            "degraded": 0,
            "error": 0,
            "fallback_used": 0,
            "total_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        provider_name: str = "rule",
        enabled: bool = True,
        fallback_to_rule: bool = True,
        **provider_kwargs: Any,
    ) -> "AIDiagnosisService":
        """按名字创建服务；主 provider 建不起来时自动降级到 rule。"""
        if not enabled:
            return cls.disabled()

        primary: AIProvider
        degraded_note = ""
        try:
            primary = create_provider(provider_name, **provider_kwargs)
        except Exception as exc:  # noqa: BLE001 - 构造失败也要能跑
            degraded_note = f"provider {provider_name!r} 创建失败（{exc}），已降级到 rule"
            logger.warning("AI provider 创建失败，降级到 rule：%s", exc)
            primary = create_provider("rule")

        fallbacks: List[AIProvider] = []
        if fallback_to_rule:
            try:
                rule = create_provider("rule")
                if getattr(primary, "name", "") != getattr(rule, "name", ""):
                    fallbacks.append(rule)
            except Exception:  # noqa: BLE001
                pass

        service = cls(provider=primary, fallback=fallbacks)
        if degraded_note:
            service.stats["degraded"] += 1
            service.stats["note"] = degraded_note
        return service

    @classmethod
    def disabled(cls) -> "AIDiagnosisService":
        """返回一个关闭状态的服务（所有能力都返回 degraded，不产生任何副作用）。"""
        return cls(provider=None, enabled=False)

    # ------------------------------------------------------------------
    # 内部：带降级的调用
    # ------------------------------------------------------------------

    def _call(self, capability: str, arg: Any, cache_key: Optional[str] = None) -> Any:
        self.stats["calls"] += 1

        if not self.enabled:
            return getattr(NullProvider(), capability)(arg)

        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        started = time.perf_counter()
        attempts: List[AIProvider] = [self.provider] + self.fallback
        last_result: Any = None
        last_error = ""

        for index, provider in enumerate(attempts):
            try:
                result = getattr(provider, capability)(arg)
            except Exception as exc:  # noqa: BLE001 - 这里是最后一道防线
                last_error = f"{type(provider).__name__}.{capability}: {exc}"
                logger.warning("AI 调用异常（已隔离，不影响仿真）：%s", last_error)
                continue

            status = getattr(result, "status", STATUS_ERROR)
            if status == STATUS_ERROR:
                last_error = getattr(result, "error", "") or "provider 返回 error"
                last_result = result
                continue

            # 成功（或 provider 自己声明的 degraded）
            if index > 0:
                self.stats["fallback_used"] += 1
                setattr(result, "status", STATUS_DEGRADED)
                note = f"（主 provider 失败，已降级到 {provider.name}）"
                for attr in ("summary", "explanation", "conclusion"):
                    if hasattr(result, attr) and getattr(result, attr):
                        setattr(result, attr, getattr(result, attr) + note)
                        break
                if index == 1 and last_error:
                    setattr(result, "error", last_error)

            latency = (time.perf_counter() - started) * 1000.0
            if hasattr(result, "latency_ms"):
                result.latency_ms = round(latency, 3)
            if hasattr(result, "generated_at"):
                result.generated_at = utc_timestamp()

            self.stats["ok" if status != STATUS_DEGRADED else "degraded"] += 1
            self.stats["total_latency_ms"] += latency

            if cache_key is not None:
                if len(self._cache) >= self.cache_size:
                    self._cache.clear()
                self._cache[cache_key] = result
            return result

        # 全部失败
        self.stats["error"] += 1
        latency = (time.perf_counter() - started) * 1000.0
        self.stats["total_latency_ms"] += latency
        logger.warning("AI 全部 provider 失败：%s", last_error)

        if last_result is not None:
            return last_result

        result_cls = {
            "diagnose": DiagnosisResult,
            "explain_decision": ExplainResult,
            "compare_policies": CompareResult,
            "generate_report": ReportResult,
        }[capability]
        result = result_cls(status=STATUS_ERROR, provider="none", error=last_error)
        result.latency_ms = round(latency, 3)
        if hasattr(result, "generated_at"):
            result.generated_at = utc_timestamp()
        return result

    # ------------------------------------------------------------------
    # 四个能力（与 /api/* 一一对应）
    # ------------------------------------------------------------------

    def _verify_evidence(self, result: Any) -> Any:
        """对 provider 输出做证据校验（见 ai/evidence_check.py）。

        这是 README 里「AI 解释被限制在结构化证据范围内，**并通过规则检查**」
        中"规则检查"那一半的可执行实现：文本里引用的数值/传感器/航迹/原因码
        必须真的存在于这次诊断的结构化上下文里。
        """
        context = getattr(self, "_last_context", {}) or {}
        parts: List[str] = []
        for attr in ("summary", "error"):
            value = getattr(result, attr, "")
            if isinstance(value, str) and value:
                parts.append(value)
        for finding in getattr(result, "findings", []) or []:
            for attr in ("title", "message", "code"):
                value = getattr(finding, attr, "")
                if isinstance(value, str) and value:
                    parts.append(value)
        for rec in getattr(result, "recommendations", []) or []:
            if isinstance(rec, str):
                parts.append(rec)
        if not parts or not context:
            return result
        check = validate_text_against_context("\n".join(parts), context)
        result.evidence_check_passed = check.passed
        result.evidence_check_failed = not check.passed
        result.evidence_violations = list(check.violations)
        return result

    def _fallback_rule_provider(self) -> Any:
        from ai.rule_provider import RuleProvider

        return RuleProvider()

    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        key = f"diagnose:{snapshot.scenario}:{snapshot.step_index}"
        # 记录本次结构化上下文，供证据校验使用
        try:
            self._last_context = snapshot.to_dict()
        except Exception:  # noqa: BLE001 - 诊断不该因上下文序列化失败而中断
            self._last_context = {}

        result = self._call("diagnose", snapshot, cache_key=key)
        result = self._verify_evidence(result)

        # 仅对**远端** provider 做回退：本地 rule/mock 的文本本来就是
        # 由结构化证据拼出来的，校验它没有意义。
        if result.evidence_check_failed and str(
            getattr(result, "provider", "")
        ) in REMOTE_PROVIDER_NAMES:
            fallback = self._fallback_rule_provider().diagnose(snapshot)
            fallback.evidence_check_passed = False
            fallback.evidence_check_failed = True
            fallback.evidence_violations = list(result.evidence_violations)
            fallback.fell_back_to_rule = True
            fallback.error = "远端 provider 输出未通过证据校验，已回退本地规则诊断"
            return fallback
        return result

    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        step = int(payload.get("step_index", (payload.get("counterfactual") or {}).get("step_index", -1)))
        key = f"explain:{step}:{hash(str(payload.get('counterfactual', {})))}"
        return self._call("explain_decision", payload, cache_key=key)

    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        labels = ",".join(str(r.get("label")) for r in (payload.get("summaries") or []))
        key = f"compare:{labels}"
        return self._call("compare_policies", payload, cache_key=key)

    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        labels = ",".join(str(r.get("label")) for r in (payload.get("summaries") or []))
        key = f"report:{labels}:{payload.get('title', '')}"
        return self._call("generate_report", payload, cache_key=key)

    # ------------------------------------------------------------------

    def explain_track(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """解释**单条航迹**：形成、更新、外推与不确定性（只读，不控制）。

        输入（全部来自结构化证据，**不含真值**）：
            {"track": {...}, "fusion": {...}, "measurement": {...},
             "communication": {...}}
        `track` 为 `ai.schema.TrackState` 形状的字典。
        """
        track_id = str((payload.get("track") or {}).get("track_id", ""))
        return self._call("explain_track", payload, cache_key=f"track:{track_id}")

    def explain_cooperation(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """解释**协同感知**：远端信息在哪些时刻帮到了本地，哪些被通信吃掉。

        ⚠️ 收益大小（RMSE 改善等）属离线评测通道，本接口只解释结构性事实。
        """
        return self._call("explain_cooperation", payload,
                          cache_key=f"coop:{len(payload.get('cases') or [])}")

    def info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider.info().to_dict(),
            "fallback": [p.info().to_dict() for p in self.fallback],
            "cache_size": self.cache_size,
            "stats": dict(self.stats),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
