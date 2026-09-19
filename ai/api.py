"""AI 认知诊断 API 能力层 + 可选的零依赖 HTTP 服务。

两层结构
--------
1. **能力层（本文件主体）**：`LPI_API` 提供四个与 HTTP 路由同名的方法，
   可以直接在 Python 里调用，不需要起服务：

       api.diagnose(snapshot)               -> /api/diagnose
       api.explain_decision(payload)        -> /api/explain_decision
       api.compare_policies(payload)        -> /api/compare_policies
       api.generate_report(payload)         -> /api/generate_report

2. **HTTP 层（`serve` / `ai_server.py`）**：用标准库 `http.server` 暴露
   `POST /api/diagnose` 等路由，**不依赖 Flask/FastAPI**。
   默认只绑定 127.0.0.1，避免误暴露。

架构约束（必须保持）
--------------------
* AI 层**没有任何控制接口**：四个能力都是只读诊断，不存在 "set_power" 之类路由；
* 任何失败都返回结构化错误 JSON（HTTP 200 + `status: "error"`），
  不抛 5xx、不中断调用方，更不影响主仿真。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .context import COMPARE_METRICS, summaries_to_payload
from .service import AIDiagnosisService

logger = logging.getLogger("lpi.ai.api")

#: 能力名 -> HTTP 路由
ROUTES: Dict[str, str] = {
    "diagnose": "/api/diagnose",
    "explain_decision": "/api/explain_decision",
    "compare_policies": "/api/compare_policies",
    "generate_report": "/api/generate_report",
    # v4.5：只读诊断接口，不新增控制通道
    "explain_track": "/api/explain_track",
    "explain_cooperation": "/api/explain_cooperation",
}

#: 每个能力的入参说明（`GET /api` 自我描述时返回）
REQUEST_SCHEMA: Dict[str, Dict[str, Any]] = {
    "diagnose": {
        "desc": "实时态势诊断",
        "body": "StateSnapshot（见 ai/schema.py）或其 to_dict() 结果",
        "required": ["step_index", "pd_min", "required_pd"],
    },
    "explain_decision": {
        "desc": "策略解释（把反事实证据转成自然语言）",
        "body": '{"counterfactual": <CounterfactualReport.to_dict()>}',
        "required": ["counterfactual"],
    },
    "compare_policies": {
        "desc": "多策略对比总结",
        "body": '{"summaries": [{label, ...metrics}], "metric_order": [...], "context": {...}}',
        "required": ["summaries"],
    },
    "generate_report": {
        "desc": "实验总结报告",
        "body": '{"title": "...", "summaries": [...], "context": {...}}',
        "required": ["summaries"],
    },
    "explain_track": {
        "desc": "单条航迹的证据链解释（v4.5，只读）",
        "body": '{"track": {track_id, status, hits, misses, local_updates, '
                'remote_updates, freshness, sigma_position, source_sensors, ...}}',
        "required": ["track"],
    },
    "explain_cooperation": {
        "desc": "协同状态的证据链解释（v4.5，只读；只讲结构不宣称收益）",
        "body": '{"communication": {...}, "fusion": {...}}',
        "required": [],
    },
}


class LPI_API:
    """四个 AI 能力的直接调用门面（不经过 HTTP）。"""

    def __init__(self, service: Optional[AIDiagnosisService] = None) -> None:
        self.service = service or AIDiagnosisService.create("rule")

    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """自我描述：路由、入参、provider 信息（对应 `GET /api`）。"""
        return {
            "project": "LPI-CogRadar",
            "api_version": "1.0",
            "note": "AI 层为只读诊断，不具备任何雷达控制能力",
            "routes": ROUTES,
            "schemas": REQUEST_SCHEMA,
            "provider": self.service.info(),
        }

    # ------------------------------------------------------------------
    # 四个能力
    # ------------------------------------------------------------------

    def diagnose(self, payload: Any) -> Dict[str, Any]:
        """POST /api/diagnose

        payload 可以是 `StateSnapshot`，也可以是它的 `to_dict()` 结果。
        """
        snapshot = _coerce_snapshot(payload)
        if snapshot is None:
            return _error("diagnose", "入参无法解析为 StateSnapshot（需要 step_index/pd_min 等字段）")
        return self.service.diagnose(snapshot).to_dict()

    def explain_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/explain_decision

        payload: {"counterfactual": <CounterfactualReport.to_dict()>}
        """
        if not isinstance(payload, dict):
            return _error("explain_decision", "入参必须是 JSON 对象")
        report = payload.get("counterfactual")
        if report is None:
            return _error("explain_decision", "缺少 counterfactual 字段")
        return self.service.explain_decision({"counterfactual": report}).to_dict()

    def compare_policies(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/compare_policies

        payload: {"summaries": [...], "metric_order": [...], "context": {...}}
        """
        if not isinstance(payload, dict):
            return _error("compare_policies", "入参必须是 JSON 对象")
        summaries = payload.get("summaries")
        if not isinstance(summaries, list) or not summaries:
            return _error("compare_policies", "缺少非空 summaries 列表")
        request = summaries_to_payload(
            summaries,
            metric_order=payload.get("metric_order") or COMPARE_METRICS,
            context=payload.get("context"),
        )
        result = self.service.compare_policies(request)
        return result.to_dict()

    def generate_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/generate_report"""
        if not isinstance(payload, dict):
            return _error("generate_report", "入参必须是 JSON 对象")
        summaries = payload.get("summaries")
        if not isinstance(summaries, list) or not summaries:
            return _error("generate_report", "缺少非空 summaries 列表")
        request = summaries_to_payload(
            summaries,
            metric_order=payload.get("metric_order") or COMPARE_METRICS,
            context=payload.get("context"),
        )
        request["title"] = payload.get("title", "")
        result = self.service.generate_report(request)
        return result.to_dict()

    def explain_track(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """v4.5：单条航迹的证据链解释（只读）。"""
        result = self.service.explain_track(payload or {})
        return result.to_dict()

    def explain_cooperation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """v4.5：协同状态的证据链解释（只读，只讲结构不宣称收益）。"""
        result = self.service.explain_cooperation(payload or {})
        return result.to_dict()

    # ------------------------------------------------------------------

    def dispatch(self, capability: str, payload: Any) -> Dict[str, Any]:
        """按能力名分发（HTTP 层与 CLI 共用）。

        能力表直接从 :data:`ROUTES` 派生：登记了路由却没有实现方法时，
        这里会立刻失败（而不是等到运行时才报"未知能力"）。
        """
        missing = [name for name in ROUTES if not callable(getattr(self, name, None))]
        if missing:
            raise RuntimeError(f"ROUTES 登记了未实现的能力：{missing}")
        handler = getattr(self, capability, None)
        if not callable(handler) or capability not in ROUTES:
            return _error(capability, f"未知能力：{capability!r}；可用：{list(ROUTES)}")
        try:
            return handler(payload)
        except Exception as exc:  # noqa: BLE001 - API 层不得把异常抛给调用方
            logger.warning("API 调用 %s 失败（已隔离）：%s", capability, exc)
            return _error(capability, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------

def _error(capability: str, message: str) -> Dict[str, Any]:
    """统一的错误响应体（HTTP 200，status=error，不打断调用方）。"""
    return {
        "provider": "none",
        "status": "error",
        "capability": capability,
        "error": message,
        "summary": "",
        "findings": [],
        "recommendations": [],
    }


def _coerce_snapshot(payload: Any) -> Any | None:
    """把 dict / StateSnapshot 统一成 StateSnapshot。

    HTTP 路径下入参必然是 dict（`StateSnapshot.to_dict()` 的结果），
    因此这里必须做**完整**重建——漏掉 power/energy 会让诊断退化成
    "功率未知、能量未知"，把最有用的证据丢掉。
    """
    from .schema import (
        AgentState,
        EnergyState,
        InterceptorState,
        JammerState,
        PowerState,
        StateSnapshot,
        TargetState,
    )

    if isinstance(payload, StateSnapshot):
        return payload
    if not isinstance(payload, dict):
        return None

    # 至少要有一个有意义的字段，否则视为垃圾入参
    if not any(k in payload for k in ("step_index", "pd_min", "detection", "power")):
        return None

    try:
        detection = payload.get("detection") or {}
        interception = payload.get("interception") or {}
        recent = payload.get("recent") or {}

        targets = [
            TargetState(
                target_id=str(t.get("target_id", "?")),
                range_m=float(t.get("range_m", 0.0)),
                rcs_m2=float(t.get("rcs_m2", 0.0)),
                snr_db=float(t.get("snr_db", 0.0)),
                pd=float(t.get("pd", 0.0)),
            )
            for t in (payload.get("targets") or [])
        ]
        interceptors = [
            InterceptorState(
                interceptor_id=str(i.get("interceptor_id", "?")),
                range_m=float(i.get("range_m", 0.0)),
                beam=str(i.get("beam", "")),
                snr_db=float(i.get("snr_db", 0.0)),
                pint_inst=float(i.get("pint_inst", 0.0)),
            )
            for i in (payload.get("interceptors") or [])
        ]
        jammers = [
            JammerState(
                jammer_id=str(j.get("jammer_id", "?")),
                active=bool(j.get("active", False)),
                mode=str(j.get("mode", "fixed")),
                action=str(j.get("action", "")),
                action_cn=str(j.get("action_cn", "")),
                jam_noise_ratio=float(j.get("jam_noise_ratio", 0.0)),
                threat=float(j.get("threat", 0.0)),
            )
            for j in (payload.get("jammers") or [])
        ]

        power_payload = payload.get("power")
        power = (
            PowerState(
                level=int(power_payload.get("level", 0)),
                tx_power_w=float(power_payload.get("tx_power_w", 0.0)),
                power_levels_w=[float(v) for v in (power_payload.get("power_levels_w") or [])],
                feasible_levels=[int(v) for v in (power_payload.get("feasible_levels") or [])],
                action_mask=[bool(v) for v in (power_payload.get("action_mask") or [])],
                previous_level=int(power_payload.get("previous_level", -1)),
                switched=bool(power_payload.get("switched", False)),
            )
            if isinstance(power_payload, dict)
            else None
        )

        energy_payload = payload.get("energy")
        energy = (
            EnergyState(
                budget_j=float(energy_payload.get("budget_j", 0.0)),
                remaining_j=float(energy_payload.get("remaining_j", 0.0)),
                cumulative_j=float(energy_payload.get("cumulative_j", 0.0)),
                fraction_used=float(energy_payload.get("fraction_used", 0.0)),
                min_step_energy_j=float(energy_payload.get("min_step_energy_j", 0.0)),
            )
            if isinstance(energy_payload, dict)
            else None
        )

        agent_payload = payload.get("agent")
        agent = (
            AgentState(
                kind=str(agent_payload.get("kind", "unknown")),
                action=agent_payload.get("action"),
                tx_power_w=agent_payload.get("tx_power_w"),
                q_values=[float(v) for v in (agent_payload.get("q_values") or [])],
                q_margin=agent_payload.get("q_margin"),
                epsilon=agent_payload.get("epsilon"),
                lambda_cost=agent_payload.get("lambda_cost"),
                cost_rate=agent_payload.get("cost_rate"),
            )
            if isinstance(agent_payload, dict)
            else None
        )

        return StateSnapshot(
            project=str(payload.get("project", "LPI-CogRadar")),
            schema_version=str(payload.get("schema_version", "1.0")),
            scenario=str(payload.get("scenario", "")),
            step_index=int(payload.get("step_index", 0)),
            time=float(payload.get("time", 0.0)),
            horizon_steps=int(payload.get("horizon_steps", 0)),
            targets=targets,
            interceptors=interceptors,
            jammers=jammers,
            pd_min=float(payload.get("pd_min", detection.get("pd_min", 0.0))),
            required_pd=float(payload.get("required_pd", detection.get("required_pd", 0.8))),
            task_satisfied=bool(
                payload.get("task_satisfied", detection.get("task_satisfied", False))
            ),
            task_violated=bool(
                payload.get("task_violated", detection.get("task_violated", False))
            ),
            pint_eff=float(payload.get("pint_eff", interception.get("pint_eff", 0.0))),
            pint_inst=float(payload.get("pint_inst", interception.get("pint_inst", 0.0))),
            exposure=float(payload.get("exposure", interception.get("exposure", 0.0))),
            power=power,
            energy=energy,
            agent=agent,
            reward=payload.get("reward"),
            recent_violation_rate=recent.get("violation_rate"),
            recent_avg_power_w=recent.get("avg_power_w"),
        )
    except Exception:  # noqa: BLE001 - 解析失败就当作无效入参
        return None


# ----------------------------------------------------------------------
# 可选 HTTP 服务（标准库实现，零第三方依赖）
# ----------------------------------------------------------------------

def make_http_handler(api: LPI_API, allow_remote: bool = False):
    """构造一个 `http.server` 的请求处理器类。"""
    from http.server import BaseHTTPRequestHandler

    allowed_hosts = {"127.0.0.1", "localhost", "::1"}

    class LPIRequestHandler(BaseHTTPRequestHandler):
        server_version = "LPI-CogRadar-AI/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            logger.info("HTTP %s", fmt % args)

        # --- 工具 ---
        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _client_allowed(self) -> bool:
            if allow_remote:
                return True
            host = (self.client_address[0] if self.client_address else "") or ""
            return host in allowed_hosts

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}

        # --- 路由 ---
        def do_GET(self) -> None:  # noqa: N802
            if not self._client_allowed():
                self._send_json(403, _error("http", "默认只允许本机访问；如需远程请显式开启"))
                return
            if self.path.rstrip("/") in ("", "/api"):
                self._send_json(200, api.describe())
                return
            self._send_json(404, _error("http", f"未知路径：{self.path}"))

        def do_POST(self) -> None:  # noqa: N802
            if not self._client_allowed():
                self._send_json(403, _error("http", "默认只允许本机访问"))
                return

            path = self.path.rstrip("/")
            capability = next(
                (name for name, route in ROUTES.items() if route.rstrip("/") == path),
                None,
            )
            if capability is None:
                self._send_json(404, _error("http", f"未知路径：{self.path}"))
                return

            try:
                payload = self._read_json()
            except Exception as exc:  # noqa: BLE001
                self._send_json(200, _error(capability, f"请求体不是合法 JSON：{exc}"))
                return

            # 注意：始终返回 200 + 结构化状态，避免调用方因网络层错误中断
            self._send_json(200, api.dispatch(capability, payload))

    return LPIRequestHandler


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    provider: str = "rule",
    allow_remote: bool = False,
    **provider_kwargs: Any,
) -> None:
    """启动 AI 诊断 HTTP 服务（阻塞）。"""
    from http.server import ThreadingHTTPServer

    service = AIDiagnosisService.create(provider, **provider_kwargs)
    api = LPI_API(service)
    handler = make_http_handler(api, allow_remote=allow_remote)
    server = ThreadingHTTPServer((host, port), handler)

    logger.info("LPI-CogRadar AI 诊断服务已启动：http://%s:%d", host, port)
    logger.info("可用路由：%s", ", ".join(ROUTES.values()))
    logger.info("provider：%s", service.info()["provider"]["name"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断，服务停止")
    finally:
        server.server_close()
