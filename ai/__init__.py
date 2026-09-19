"""LPI-CogRadar 的 AI 认知诊断层。

    LPI-CogRadar: An AI-Driven Cognitive Radar Power Control
                  and Electromagnetic Adversarial Simulation Platform

模块结构
--------
    schema.py        结构化 JSON 输入输出协议（只读状态快照 + 四类结果）
    provider.py      AIProvider 统一接口与工厂（与厂商解耦）
    rule_provider.py 默认 provider：规则+模板，**无需 API Key**，离线确定
    mock_provider.py 确定性桩（测试与降级路径验证，可注入故障）
    http_provider.py 远程 provider（OpenAI/DeepSeek/通义，OpenAI 兼容协议，预留）
    context.py       仿真侧 -> 只读快照 的转换（边界层）
    service.py       四个能力的门面 + 异常隔离 + 自动降级
    api.py           /api/diagnose 等四个能力 + 零依赖 HTTP 服务

架构约束（不可违反）
--------------------
1. AI 层**没有控制接口**，只读状态、只给诊断与建议；
2. AI 层失败**绝不影响主仿真**：全部异常在 `AIDiagnosisService` 内被隔离并降级；
3. 无 Key 即可运行：默认 provider 是本地 `rule`；
4. 解释必须可追溯：自然语言只能引用结构化证据里的代码与数值。
"""

from .api import ROUTES, LPI_API, serve
from .context import (
    COMPARE_METRICS,
    agent_state_from_dqn,
    snapshot_from_simulator,
    summaries_to_payload,
)
from .provider import AIProvider, available_providers, create_provider
from .schema import (
    AgentState,
    CompareResult,
    DiagnosisResult,
    EnergyState,
    ExplainResult,
    Finding,
    InterceptorState,
    JammerState,
    PowerState,
    ProviderInfo,
    ReportResult,
    StateSnapshot,
    TargetState,
)
from .service import AIDiagnosisService

__all__ = [
    # 协议
    "StateSnapshot",
    "TargetState",
    "InterceptorState",
    "JammerState",
    "PowerState",
    "EnergyState",
    "AgentState",
    "Finding",
    "DiagnosisResult",
    "ExplainResult",
    "CompareResult",
    "ReportResult",
    "ProviderInfo",
    # provider
    "AIProvider",
    "create_provider",
    "available_providers",
    # 上下文与服务
    "snapshot_from_simulator",
    "agent_state_from_dqn",
    "summaries_to_payload",
    "COMPARE_METRICS",
    "AIDiagnosisService",
    # API
    "LPI_API",
    "ROUTES",
    "serve",
]
