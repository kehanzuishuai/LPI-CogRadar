"""AI 认知诊断 + 可解释决策 CLI。

    # 规则策略 + 本地 rule provider（无需任何 API Key）
    python diagnose_ai.py

    # 换用 DQN，并指定反事实关注档位
    python diagnose_ai.py --policy dqn --model output/rl/dqn_agent.pt --focus-powers 18 25 35

    # 换成远程大模型（同一套协议，只影响文字表达）
    python diagnose_ai.py --provider deepseek

    # 自适应干扰机场景 + 把 AI 诊断结果落盘
    python diagnose_ai.py --adaptive-jammer --out-dir output/ai_diagnosis

做三件事
--------
1. **态势诊断**：跑一遍 episode，在关键时刻（档位切换 / 探测未达标 / 能量警戒）
   逐步调用 `/api/diagnose`；
2. **决策解释**：对同一时刻做**反事实试算**（默认试算 18/25/35 W 以及相邻档位），
   把结构化证据交给 `/api/explain_decision` 转成自然语言
   —— 大模型只允许引用证据里的代码与数值，不能凭空解释；
3. **对比与总结**：对多策略汇总调用 `/api/compare_policies` 与 `/api/generate_report`。

产物：`ai_report.json`（完整结构化结果） + `ai_report.html`（可读报告） + 统一日志。
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import experiment_config as ec
from ai import (
    AIDiagnosisService,
    LPI_API,
    agent_state_from_dqn,
    available_providers,
    snapshot_from_simulator,
    summaries_to_payload,
)
from explain import build_counterfactual_report, build_key_moment_reports
from logging_utils import ensure_utf8_console, get_logger, setup_logging
from metrics import write_curves_html

#: Windows 控制台默认 GBK，本脚本会打印 ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
ensure_utf8_console()

logger = get_logger("lpi.ai.diagnose")

DEFAULT_OUT_DIR = os.path.join("output", "ai_diagnosis")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{ec.PROJECT_NAME} 认知诊断与可解释决策",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--policy", default="rule", choices=["rule", "dqn"],
                        help="被诊断的策略")
    parser.add_argument("--model", default=os.path.join("output", "rl", "dqn_agent.pt"),
                        help="--policy dqn 时的模型路径")
    parser.add_argument("--seed", type=int, default=ec.DEFAULT_SEED, help="场景种子")
    parser.add_argument("--adaptive-jammer", action="store_true",
                        help="切换到规则自适应智能干扰机")
    parser.add_argument("--provider", default="rule", choices=available_providers(),
                        help="AI provider：rule/mock 无 Key；远程需 API Key")
    parser.add_argument("--api-key", default=None, help="远程 provider 的 API Key")
    parser.add_argument("--base-url", default=None, help="远程 provider 的 base_url")
    parser.add_argument("--model-name", default=None, help="远程 provider 的模型名")
    parser.add_argument("--no-ai", action="store_true",
                        help="关闭 AI 层（验证『关掉 AI 也能完整运行』）")
    parser.add_argument("--max-moments", type=int, default=5, help="最多解释几个关键时刻")
    parser.add_argument("--focus-powers", type=float, nargs="+", default=[18.0, 25.0, 35.0],
                        help="反事实试算关注的功率（W）")
    parser.add_argument("--log-dir", default=os.path.join("output", "logs"))
    parser.add_argument("--device", default="cpu")
    return parser


# ----------------------------------------------------------------------

def run_and_collect(args: argparse.Namespace) -> tuple[Any, List[Any], List[int]]:
    """跑一遍 episode，返回 (env, results, actions)。"""
    env = ec.make_env(args.config)
    if args.adaptive_jammer:
        env.sim.apply_overrides(adaptive_jammer=True)

    if args.policy == "dqn":
        from rl import DQNAgent, silence_numpy_bridge_warning

        silence_numpy_bridge_warning()
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"未找到模型 {args.model}，请先训练或用 --policy rule")
        agent = DQNAgent.load(args.model, device=args.device)
        results = ec.run_dqn_episode(env, agent, args.seed)
    else:
        policy = ec.scripted_policy_specs(int(env.sim.scenario.num_steps))[1].factory()
        results = ec.run_scripted_episode(env, policy, args.seed)

    actions = [int(r.power_level) for r in results]
    return env, results, actions


def replay_with_moments(
    args: argparse.Namespace,
    actions: Sequence[int],
    moments: Sequence[Dict[str, Any]],
    api: LPI_API,
) -> List[Dict[str, Any]]:
    """按记录下来的动作序列重放，在关键时刻产出诊断与解释。

    为什么重放：`StepResult` 不带状态，无法回到某一时刻。
    重放同一串动作即可精确复现状态，从而在该时刻做反事实试算。
    """
    moment_map = {int(m["step_index"]): m for m in moments}
    env = ec.make_env(args.config)
    if args.adaptive_jammer:
        env.sim.apply_overrides(adaptive_jammer=True)
    env.reset(seed=args.seed)

    outputs: List[Dict[str, Any]] = []
    for level in actions:
        if env.sim.is_done:
            break
        index = int(env.sim.step_index)

        if index in moment_map:
            # 1) 反事实结构化证据（纯函数试算，不改变状态）
            cf = build_counterfactual_report(
                env.sim, chosen_level=level, focus_powers_w=args.focus_powers
            )
            cf_dict = cf.to_dict()

            # 2) 当前状态快照 -> /api/diagnose
            snapshot = snapshot_from_simulator(env.sim)
            diagnosis = api.diagnose(snapshot)

            # 3) 结构化证据 -> /api/explain_decision
            explanation = api.explain_decision({"counterfactual": cf_dict})

            outputs.append(
                {
                    "step_index": index,
                    "time": float(env.sim.current_time),
                    "moment_reasons": moment_map[index].get("reasons", []),
                    "diagnose": diagnosis,
                    "explain": explanation,
                    "counterfactual": cf_dict,
                }
            )
            logger.info(
                "关键时刻 t=%.1fs：%s | %s",
                env.sim.current_time,
                explanation.get("verdict_code", ""),
                ", ".join(moment_map[index].get("reasons", [])),
            )

        env.step(level)
    return outputs


def build_policy_comparison(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """跑几个基线策略，供 /api/compare_policies 与 /api/generate_report 使用。"""
    rows: List[Dict[str, Any]] = []
    num_steps = ec.scenario_horizon(args.config)
    specs = ec.scripted_policy_specs(
        num_steps, include_myopic=True, include_lookahead=True
    )
    env = ec.make_env(args.config)
    if args.adaptive_jammer:
        env.sim.apply_overrides(adaptive_jammer=True)

    for spec in specs:
        policy = spec.factory()
        results = ec.run_scripted_episode(env, policy, args.seed)
        rows.append(ec.summarize_episode(env, results, spec.label, policy.describe()))
        logger.info("对比基线 %s 完成", spec.label)
    return rows


# ----------------------------------------------------------------------

def write_html_report(payload: Dict[str, Any], path: str) -> None:
    """把 AI 诊断结果渲染成可读 HTML（纯表格 + 文本，不依赖任何前端框架）。"""
    parts: List[str] = []
    parts.append(
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{html_mod.escape(payload.get('title', 'AI 诊断报告'))}</title>"
        "<style>"
        "body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:26px auto;max-width:1080px;"
        "line-height:1.6;color:#222}"
        "h1{font-size:21px;border-bottom:2px solid #d8dee9;padding-bottom:8px}"
        "h2{font-size:16px;margin-top:24px}"
        "table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px}"
        "th,td{border:1px solid #d8dee9;padding:6px 9px;text-align:left}"
        "th{background:#eef2f7}"
        ".meta{color:#667;font-size:13px}"
        ".crit{color:#b3261e;font-weight:600}.warn{color:#a15c00;font-weight:600}"
        ".info{color:#2f6f3e}"
        ".card{border:1px solid #d8dee9;border-radius:6px;padding:12px 16px;background:#fafbfc;margin:12px 0}"
        "code{background:#f0f2f5;padding:1px 5px;border-radius:3px}"
        "</style></head><body>"
    )
    parts.append(f"<h1>{html_mod.escape(payload.get('title', 'AI 诊断报告'))}</h1>")
    parts.append(
        f"<p class='meta'>项目：{html_mod.escape(payload.get('project', ''))} "
        f"| provider：<code>{html_mod.escape(payload.get('provider', ''))}</code>"
        f"{'（已降级）' if payload.get('provider_status') == 'degraded' else ''} "
        f"| 生成时间：{html_mod.escape(payload.get('generated_at', ''))}</p>"
    )

    for section in payload.get("sections", []):
        parts.append(f"<h2>{html_mod.escape(section['heading'])}</h2>")
        body = section["body"]
        if section.get("html"):
            parts.append(body)
        else:
            parts.append(
                "<p>" + html_mod.escape(body).replace("\n", "<br>") + "</p>"
            )

    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging("ai_diagnosis", log_dir=args.log_dir)

    print(f"{ec.PROJECT_TITLE}")
    print(f"版本 {ec.PROJECT_VERSION} | provider={args.provider} | "
          f"policy={args.policy} | 自适应干扰机={args.adaptive_jammer}")
    print()

    logger.info("第一步：跑一遍 episode 并收集动作序列")
    env, results, actions = run_and_collect(args)
    horizon = int(env.sim.scenario.num_steps)
    summary = ec.summarize_episode(env, results, f"{args.policy}(诊断对象)", args.policy)
    logger.info(
        "episode 完成：%d 步，满足率 %.4f，违反率 %.4f，能耗 %.1f J",
        len(results), summary["horizon_satisfaction_rate"],
        summary["violation_rate"], summary["cumulative_energy_j"],
    )

    logger.info("第二步：挑选关键时刻")
    moments = build_key_moment_reports(env.sim, results, max_moments=args.max_moments)
    logger.info("共挑出 %d 个关键时刻", len(moments))

    logger.info("第三步：初始化 AI 服务")
    service = (
        AIDiagnosisService.disabled()
        if args.no_ai
        else AIDiagnosisService.create(
            args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model_name,
        )
    )
    api = LPI_API(service)
    logger.info("provider 信息：%s", service.info()["provider"])

    logger.info("第四步：关键时刻的诊断与解释（重放同一动作序列）")
    moment_outputs = replay_with_moments(args, actions, moments, api)

    logger.info("第五步：多策略对比与总结报告")
    comparison_rows = build_policy_comparison(args)
    compare_payload = summaries_to_payload(comparison_rows)
    compare_result = api.compare_policies(compare_payload)
    report_result = api.generate_report(
        {
            "title": f"{ec.PROJECT_NAME} 实验总结",
            "summaries": compare_payload["summaries"],
            "context": {
                "场景": os.path.basename(args.config),
                "干扰机": "规则自适应智能干扰机" if args.adaptive_jammer else "固定时间窗",
                "种子": args.seed,
                "被诊断策略": args.policy,
            },
        }
    )

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    payload: Dict[str, Any] = {
        "title": f"{ec.PROJECT_NAME} AI 认知诊断报告",
        "project": ec.PROJECT_NAME,
        "version": ec.PROJECT_VERSION,
        "provider": service.provider.name,
        "provider_status": "degraded" if not service.enabled else "ok",
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint_self_description": api.describe(),
        "episode_summary": {
            k: v for k, v in summary.items()
            if k in ("horizon_satisfaction_rate", "violation_rate", "avg_tx_power_w",
                     "cumulative_energy_j", "avg_intercept_prob", "avg_exposure",
                     "composite_reward", "steps", "jammer_modes")
        },
        "key_moments": moment_outputs,
        "compare_policies": compare_result,
        "generate_report": report_result,
    }

    json_path = os.path.join(args.out_dir, "ai_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    # --- HTML 报告 ---
    sections: List[Dict[str, str]] = []

    rows_html = [
        "<table><tr><th>指标</th><th>数值</th></tr>"
    ]
    for key, value in payload["episode_summary"].items():
        text = f"{value:.4f}" if isinstance(value, float) else str(value)
        rows_html.append(f"<tr><td>{key}</td><td>{text}</td></tr>")
    rows_html.append("</table>")
    sections.append(
        {"heading": "一、被诊断 episode 概况", "body": "".join(rows_html), "html": True}
    )

    if moment_outputs:
        for item in moment_outputs:
            diag = item["diagnose"]
            expl = item["explain"]
            cf = item["counterfactual"]
            inner = [
                f"<p><b>时刻</b>：t={item['time']:.1f}s（第 {item['step_index']} 步）"
                f"　<b>触发原因</b>：{'、'.join(item['moment_reasons'])}</p>",
                f"<div class='card'><b>态势诊断</b>"
                f"（severity=<span class='{html_mod.escape(diag.get('severity','info'))}'>"
                f"{html_mod.escape(diag.get('severity',''))}</span>）："
                f"{html_mod.escape(diag.get('summary',''))}</div>",
            ]
            findings = diag.get("findings") or []
            if findings:
                inner.append("<table><tr><th>级别</th><th>结论</th><th>说明</th></tr>")
                for f in findings:
                    sev = html_mod.escape(str(f.get("severity", "")))
                    inner.append(
                        f"<tr><td class='{sev}'>{sev}</td>"
                        f"<td>{html_mod.escape(str(f.get('title','')))}</td>"
                        f"<td>{html_mod.escape(str(f.get('message','')))}</td></tr>"
                    )
                inner.append("</table>")
            inner.append(
                f"<div class='card'><b>决策解释</b>"
                f"（direction=<code>{html_mod.escape(str(expl.get('direction','')))}</code>，"
                f"证据代码：<code>{html_mod.escape('、'.join(expl.get('evidence_codes') or []))}</code>）<br>"
                f"{html_mod.escape(str(expl.get('explanation','')))}</div>"
            )
            cands = cf.get("counterfactuals") or []
            if cands:
                inner.append(
                    "<table><tr><th>功率(W)</th><th>可行</th><th>Pd</th><th>Pint_inst</th>"
                    "<th>Pint_eff</th><th>暴露(下一步)</th><th>单步能耗(J)</th>"
                    "<th>剩余(J)</th><th>单步收益</th><th>相对执行档</th></tr>"
                )
                for c in cands:
                    mark = " ✔" if c.get("is_chosen") else ""
                    inner.append(
                        f"<tr><td>{c['tx_power_w']}{mark}</td>"
                        f"<td>{'是' if c['feasible'] else '否'}</td>"
                        f"<td>{c['pd_min']:.3f}</td><td>{c['pint_inst']:.3f}</td>"
                        f"<td>{c['pint_eff']:.3f}</td><td>{c['exposure_next']:.4f}</td>"
                        f"<td>{c['step_energy_j']:.1f}</td>"
                        f"<td>{c['remaining_energy_after_j']:.1f}</td>"
                        f"<td>{c['immediate_reward']:+.4f}</td>"
                        f"<td>{c['delta_reward_vs_chosen']:+.4f}</td></tr>"
                    )
                inner.append("</table>")
            sections.append(
                {
                    "heading": f"关键时刻 {item['step_index']}（t={item['time']:.1f}s）",
                    "body": "".join(inner),
                    "html": True,
                }
            )
    else:
        sections.append({"heading": "关键时刻", "body": "本次没有触发关键时刻。"})

    cmp_table = ["<table><tr><th>策略</th><th>满足率</th><th>违反率</th><th>平均功率</th>"
                 "<th>能耗(J)</th><th>平均Pint</th><th>平均暴露</th><th>综合收益</th></tr>"]
    for row in compare_result.get("table", []):
        cmp_table.append(
            f"<tr><td>{html_mod.escape(str(row.get('label','')))}</td>"
            f"<td>{(row.get('horizon_satisfaction_rate') or 0):.4f}</td>"
            f"<td>{(row.get('violation_rate') or 0):.4f}</td>"
            f"<td>{(row.get('avg_tx_power_w') or 0):.2f}</td>"
            f"<td>{(row.get('cumulative_energy_j') or 0):.1f}</td>"
            f"<td>{(row.get('avg_intercept_prob') or 0):.4f}</td>"
            f"<td>{(row.get('avg_exposure') or 0):.4f}</td>"
            f"<td>{(row.get('composite_reward') or 0):+.4f}</td></tr>"
        )
    cmp_table.append("</table>")
    if compare_result.get("tradeoffs"):
        cmp_table.append(
            "<div class='card'><b>关键权衡</b><br>"
            + "<br>".join(html_mod.escape(t) for t in compare_result["tradeoffs"])
            + "</div>"
        )
    sections.append(
        {"heading": "多策略对比（/api/compare_policies）",
         "body": "".join(cmp_table), "html": True}
    )

    report_html = ["<div class='card'>"]
    for section in report_result.get("sections", []):
        report_html.append(f"<b>{html_mod.escape(section['heading'])}</b><br>")
        report_html.append(
            html_mod.escape(section["body"]).replace("\n", "<br>") + "<br><br>"
        )
    report_html.append("</div>")
    sections.append(
        {"heading": "实验总结（/api/generate_report）",
         "body": "".join(report_html), "html": True}
    )

    html_payload = dict(payload)
    html_payload["sections"] = sections
    html_path = os.path.join(args.out_dir, "ai_report.html")
    write_html_report(html_payload, html_path)

    # --- 控制台摘要 ---
    print("\n======== 关键时刻诊断摘要 ========")
    for item in moment_outputs:
        diag = item["diagnose"]
        expl = item["explain"]
        print(f"  t={item['time']:5.1f}s [{diag.get('severity','')}] "
              f"{diag.get('summary','')[:90]}")
        print(f"     解释（{expl.get('direction','')}/{expl.get('verdict_code','')}）："
              f"{str(expl.get('explanation',''))[:150]}")

    print("\n======== 多策略对比（AI 总结）========")
    for line in compare_result.get("highlights", []):
        print(f"  - {line}")
    for line in compare_result.get("tradeoffs", []):
        print(f"  ! {line}")

    print("\n======== 输出文件 ========")
    for path in (json_path, html_path):
        print(f"  {path}")
    print(f"\nAI 服务统计：{service.info()['stats']}")


if __name__ == "__main__":
    main()
