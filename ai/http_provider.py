"""RemoteHTTPProvider：与厂商解耦的远程大模型 provider（预留实现）。

特点
----
* **只依赖标准库 `urllib`**，不需要 openai / requests 等第三方包；
* 采用 OpenAI 兼容的 `/chat/completions` 协议，因此 OpenAI、DeepSeek、
  通义（DashScope 兼容模式）、以及任何自建兼容网关都能直接用，
  只需要改 `base_url` 与 `model`；
* **必须有 api_key 才会工作**；缺失时构造阶段就抛错，
  由 `AIDiagnosisService` 兜住并降级到 `rule` provider；
* 强制模型输出 JSON（`response_format` + 提示词双重约束），
  解析失败时同样降级——**绝不让主仿真依赖远程服务的可用性**。

安全与合规
----------
* 只发送 `StateSnapshot` 这类**结构化、非敏感**的仿真状态；
* 不发送任何文件内容或本机信息；
* api_key 只从参数或环境变量读取，不写入任何日志与产物。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .provider import AIProvider
from .schema import (
    CompareResult,
    DiagnosisResult,
    ExplainResult,
    ProviderInfo,
    ReportResult,
    STATUS_ERROR,
    STATUS_OK,
    StateSnapshot,
)

#: 各厂商默认端点与推荐模型（都走 OpenAI 兼容协议）
VENDOR_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "http": {
        "base_url": "",
        "model": "",
        "api_key_env": "LPI_AI_API_KEY",
    },
}

#: 系统提示词：把「不许编造、只解释给定证据」写死
SYSTEM_PROMPT = """你是低截获雷达功率调控仿真平台 LPI-CogRadar 的认知诊断助手。
你只做三件事：解读结构化状态、解释已有证据、总结实验结果。
硬性规则：
1. 只允许使用输入 JSON 里出现过的数值与证据代码，禁止引入任何外部知识或猜测；
2. 不得给出与输入数据矛盾的结论；数据不足时明确说"数据不足"；
3. 你没有任何控制权限，不得声称自己调整了功率或干扰；
4. 只输出 JSON，不要输出 markdown 代码块以外的解释文字。
"""


class RemoteHTTPProvider(AIProvider):
    """OpenAI 兼容协议的远程 provider。"""

    def __init__(
        self,
        vendor: str = "http",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 20.0,
        max_retries: int = 1,
        temperature: float = 0.2,
        verbose: bool = False,
    ) -> None:
        defaults = VENDOR_DEFAULTS.get(vendor, VENDOR_DEFAULTS["http"])
        self.vendor = vendor
        self.name = f"http:{vendor}"
        self.api_key = api_key or os.environ.get(defaults["api_key_env"], "")
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")
        self.model = model or defaults["model"]
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self.verbose = bool(verbose)

        if not self.api_key:
            raise ValueError(
                f"远程 AI provider {self.name} 需要 API Key："
                f"请通过参数传入，或设置环境变量 {defaults['api_key_env']}。"
                f"（本项目默认使用 rule provider，无需 Key 即可运行）"
            )
        if not self.base_url:
            raise ValueError(f"远程 AI provider {self.name} 需要 base_url")

    # ------------------------------------------------------------------

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            model=self.model,
            kind="remote",
            requires_api_key=True,
            supports=["diagnose", "explain_decision", "compare_policies", "generate_report"],
            notes=f"OpenAI 兼容协议，base_url={self.base_url}",
        )

    def health(self) -> Dict[str, Any]:
        """用一个极小的请求探活；不抛错，只回报状态。"""
        try:
            self._chat("回复 JSON {\"ok\": true}", max_tokens=16)
            return {"provider": self.name, "model": self.model, "healthy": True}
        except Exception as exc:  # noqa: BLE001 - 健康检查不应抛出
            return {
                "provider": self.name,
                "model": self.model,
                "healthy": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # HTTP 基础设施
    # ------------------------------------------------------------------

    def _chat(self, user_prompt: str, max_tokens: int = 900) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - 统一重试与降级
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"远程 AI 调用失败：{type(last_error).__name__}: {last_error}")

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """从模型输出里抠出 JSON（兼容 ```json 代码块）。"""
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def _ask_json(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"任务：{task}\n"
            f"输入 JSON：\n{json.dumps(payload, ensure_ascii=False)}\n"
            "请只输出 JSON。"
        )
        return self._parse_json(self._chat(prompt))

    # ------------------------------------------------------------------
    # 四个能力
    # ------------------------------------------------------------------

    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        started = time.perf_counter()
        try:
            data = self._ask_json(
                "根据状态快照给出态势诊断。输出 JSON 结构："
                '{"severity":"info|warning|critical","summary":"...",'
                '"findings":[{"code":"...","severity":"...","title":"...",'
                '"message":"...","evidence":{}}],"recommendations":["..."],'
                '"confidence":0.0~1.0}。'
                "code 只能从输入里出现的候选代码中选择，evidence 只能填输入里已有的数值。",
                snapshot.to_dict(),
            )
            from .schema import Finding

            findings = [
                Finding(
                    code=str(f.get("code", "UNKNOWN")),
                    severity=str(f.get("severity", "info")),
                    title=str(f.get("title", "")),
                    message=str(f.get("message", "")),
                    evidence=dict(f.get("evidence", {}) or {}),
                )
                for f in (data.get("findings") or [])
            ]
            return DiagnosisResult(
                provider=self.name,
                model=self.model,
                status=STATUS_OK,
                severity=str(data.get("severity", "info")),
                summary=str(data.get("summary", "")),
                findings=findings,
                recommendations=[str(r) for r in (data.get("recommendations") or [])],
                confidence=float(data.get("confidence", 0.6)),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - 由 service 兜底，这里先如实回报
            return DiagnosisResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        started = time.perf_counter()
        report = payload.get("counterfactual") or payload
        try:
            data = self._ask_json(
                "把反事实证据翻译成一段中文解释，说明本步为什么升/降/维持功率。"
                "只允许引用输入里的数值与 evidence_codes，禁止编造。输出 JSON："
                '{"explanation":"...","confidence":0.0~1.0}',
                report,
            )
            return ExplainResult(
                provider=self.name,
                model=self.model,
                status=STATUS_OK,
                step_index=int(report.get("step_index", 0)),
                direction=str(report.get("direction", "hold")),
                explanation=str(data.get("explanation", "")),
                verdict_code=str((report.get("verdict") or {}).get("code", "")),
                evidence_codes=list((report.get("verdict") or {}).get("evidence_codes", []) or []),
                confidence=float(data.get("confidence", 0.6)),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return ExplainResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        started = time.perf_counter()
        try:
            data = self._ask_json(
                "对比这些策略的汇总指标，输出 JSON："
                '{"ranking":["..."],"highlights":["..."],"tradeoffs":["..."],'
                '"confidence":0.0~1.0}。只允许引用输入里的数值。',
                {
                    "summaries": payload.get("summaries", []),
                    "metric_order": payload.get("metric_order", []),
                },
            )
            return CompareResult(
                provider=self.name,
                model=self.model,
                status=STATUS_OK,
                metric_order=list(payload.get("metric_order", []) or []),
                ranking=[str(x) for x in (data.get("ranking") or [])],
                highlights=[str(x) for x in (data.get("highlights") or [])],
                tradeoffs=[str(x) for x in (data.get("tradeoffs") or [])],
                confidence=float(data.get("confidence", 0.6)),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return CompareResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        started = time.perf_counter()
        try:
            data = self._ask_json(
                "根据实验汇总生成中文总结报告，输出 JSON："
                '{"title":"...","sections":[{"heading":"...","body":"..."}],'
                '"conclusion":"...","confidence":0.0~1.0}。只允许引用输入里的数值。',
                {
                    "title": payload.get("title", ""),
                    "context": payload.get("context", {}),
                    "summaries": payload.get("summaries", []),
                },
            )
            return ReportResult(
                provider=self.name,
                model=self.model,
                status=STATUS_OK,
                title=str(data.get("title") or payload.get("title") or ""),
                sections=[
                    {"heading": str(s.get("heading", "")), "body": str(s.get("body", ""))}
                    for s in (data.get("sections") or [])
                ],
                conclusion=str(data.get("conclusion", "")),
                confidence=float(data.get("confidence", 0.6)),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return ReportResult(
                provider=self.name, model=self.model, status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
