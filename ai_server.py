"""LPI-CogRadar AI 认知诊断服务（HTTP）。

    python ai_server.py                                  # 规则 provider，无 Key
    python ai_server.py --provider deepseek              # 需要 DEEPSEEK_API_KEY
    python ai_server.py --provider openai --port 9000
    python ai_server.py --allow-remote                   # 默认只允许本机访问

四个路由（POST，JSON 请求体）：

    /api/diagnose           实时态势诊断          body: StateSnapshot
    /api/explain_decision   策略解释（引证据）    body: {"counterfactual": {...}}
    /api/compare_policies   多策略对比            body: {"summaries": [...], ...}
    /api/generate_report    实验总结报告          body: {"title":..., "summaries":[...]}

`GET /api` 返回自我描述（路由、入参 schema、provider 信息）。

零第三方依赖（标准库 http.server）。任何失败都返回 HTTP 200 + `status:"error"`，
**不会**抛 5xx，也不会影响任何仿真进程。
"""

from __future__ import annotations

import argparse
import logging

from ai import available_providers, serve
from experiment_config import CONFIG_PATH, PROJECT_NAME, PROJECT_TITLE, PROJECT_VERSION
from logging_utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} AI 认知诊断 HTTP 服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", default="rule",
                        choices=available_providers(),
                        help="AI provider：rule/mock 无需 Key；远程 provider 需要 API Key")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--allow-remote", action="store_true",
                        help="允许非本机访问（默认只允许 127.0.0.1/localhost）")
    parser.add_argument("--api-key", default=None, help="远程 provider 的 API Key")
    parser.add_argument("--base-url", default=None, help="远程 provider 的 base_url（兼容网关）")
    parser.add_argument("--model", default=None, help="远程 provider 的模型名")
    parser.add_argument("--timeout", type=float, default=20.0, help="远程调用超时（秒）")
    parser.add_argument("--log-dir", default="output/logs", help="日志目录")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging("ai_server", log_dir=args.log_dir)
    print(f"{PROJECT_TITLE}")
    print(f"版本：{PROJECT_VERSION}   场景配置：{CONFIG_PATH}")
    print()

    serve(
        host=args.host,
        port=args.port,
        provider=args.provider,
        allow_remote=args.allow_remote,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
