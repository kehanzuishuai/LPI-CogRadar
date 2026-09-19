"""HTML 实验结果报告（保留原工程 output/experiment_results.html 的输出角色）。

生成一个自包含的 HTML：汇总对照表 + 相对固定功率基线的量化结论 +
四张折线图（发射功率 / 探测概率 / 截获概率 / 累计能耗）。

图表用 Chart.js CDN；即使离线拿不到 CDN，表格与结论仍然是纯 HTML，
报告依然可读。
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

_CHART_CDN = "https://cdn.jsdelivr.net/npm/chart.js"

_SUMMARY_ROWS = [
    ("horizon_satisfaction_rate", "探测任务满足率（按完整任务步数，主指标）", "{:.4f}"),
    ("detection_task_satisfaction_rate", "探测任务满足率（按实际执行步数）", "{:.4f}"),
    ("steps", "实际执行步数", "{:.0f}"),
    ("horizon_steps", "完整任务步数", "{:.0f}"),
    ("terminated_early", "是否提前终止（能量耗尽）", "{:.0f}"),
    ("avg_tx_power_w", "平均发射功率 (W)", "{:.2f}"),
    ("peak_tx_power_w", "峰值发射功率 (W)", "{:.2f}"),
    ("cumulative_energy_j", "累计能耗 (J)", "{:.1f}"),
    ("remaining_energy_j", "剩余能量 (J)", "{:.1f}"),
    ("energy_utilization", "能量预算占用率", "{:.4f}"),
    ("avg_intercept_prob", "平均截获概率 Pint_eff", "{:.4f}"),
    ("avg_instant_intercept_prob", "平均瞬时截获概率 Pint_inst", "{:.4f}"),
    ("cumulative_intercept_prob", "累计被截获概率", "{:.4f}"),
    ("avg_exposure", "平均累计暴露量", "{:.4f}"),
    ("final_exposure", "结束时暴露量", "{:.4f}"),
    ("cumulative_exposure", "累计暴露（Σ Pint_inst）", "{:.4f}"),
    ("lpi_compliant_rate", "低截获达标率 (Pint ≤ 阈值)", "{:.4f}"),
    ("first_intercept_time", "首次被截获时间 (s，-1 表示全程未超阈值)", "{:.1f}"),
    ("violation_steps", "探测未达标步数", "{:.0f}"),
    ("avg_pd", "平均探测概率 Pd", "{:.4f}"),
    ("min_pd", "最小探测概率 Pd", "{:.4f}"),
    ("avg_intercept_snr_db", "平均截获 SNR (dB)", "{:.2f}"),
    ("composite_reward", "综合收益（单步均值）", "{:.4f}"),
    ("power_switch_count", "功率档位切换次数", "{:.0f}"),
    ("jammed_steps", "受干扰步数", "{:.0f}"),
    ("avg_jam_noise_ratio", "平均 J/N", "{:.4f}"),
]


def _series(results: Sequence[Any]) -> Dict[str, List[float]]:
    return {
        "time": [round(r.time, 3) for r in results],
        "tx_power_w": [round(r.tx_power_w, 4) for r in results],
        "pd_min": [round(r.pd_min, 5) for r in results],
        "intercept_prob": [round(r.intercept_prob, 5) for r in results],
        "exposure": [round(r.exposure, 5) for r in results],
        "cumulative_energy_j": [round(r.cumulative_energy_j, 3) for r in results],
        "task_satisfied": [1 if r.task_satisfied else 0 for r in results],
    }


def _build_conclusions(summaries: Sequence[Dict[str, Any]]) -> List[str]:
    """以第一个（固定功率）实验为基准，导出量化对比结论。"""
    if len(summaries) < 2:
        return []

    base = summaries[0]
    conclusions: List[str] = []

    for summary in summaries[1:]:
        label = html.escape(str(summary["label"]))
        parts: List[str] = []

        energy_delta = _relative_change(
            base["cumulative_energy_j"], summary["cumulative_energy_j"]
        )
        if energy_delta is not None:
            parts.append(f"累计能耗 {energy_delta}")

        pint_delta = _relative_change(
            base["avg_intercept_prob"], summary["avg_intercept_prob"]
        )
        if pint_delta is not None:
            parts.append(f"平均截获概率 {pint_delta}")

        sat_delta = summary["horizon_satisfaction_rate"] - base[
            "horizon_satisfaction_rate"
        ]
        parts.append(f"探测任务满足率 {sat_delta:+.4f}")

        reward_delta = summary["composite_reward"] - base["composite_reward"]
        parts.append(f"综合收益 {reward_delta:+.4f}")

        conclusions.append(
            f"相对「{html.escape(str(base['label']))}」，<b>{label}</b>：" + "，".join(parts)
        )

    return conclusions


def _relative_change(baseline: float, value: float) -> str | None:
    if abs(baseline) < 1e-12:
        return None
    change = (value - baseline) / abs(baseline)
    return f"{change:+.1%}"


def build_html_report(
    summaries: Sequence[Dict[str, Any]],
    runs: Dict[str, Sequence[Any]],
    scenario_info: Dict[str, Any] | None = None,
) -> str:
    """生成 HTML 报告文本。"""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenario_info = scenario_info or {}

    # ---- 汇总对照表 ----
    summary_header = "".join(
        f"<th>{html.escape(str(s['label']))}</th>" for s in summaries
    )
    summary_rows = []
    for key, title, fmt in _SUMMARY_ROWS:
        cells = "".join(
            f"<td>{fmt.format(s.get(key, 0.0)) if isinstance(s.get(key, 0.0), (int, float)) else html.escape(str(s.get(key, '')))}</td>"
            for s in summaries
        )
        summary_rows.append(f"<tr><th class=\"rowhead\">{title}</th>{cells}</tr>")
    summary_table = (
        f"<table><thead><tr><th>指标</th>{summary_header}</tr></thead>"
        f"<tbody>{''.join(summary_rows)}</tbody></table>"
    )

    # ---- 场景信息 ----
    info_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in scenario_info.items()
    )
    info_table = f"<table class=\"info\"><tbody>{info_rows}</tbody></table>"

    # ---- 结论 ----
    conclusions = _build_conclusions(summaries)
    conclusions_html = (
        "<ul>" + "".join(f"<li>{c}</li>" for c in conclusions) + "</ul>"
        if conclusions
        else "<p>只有一个实验组，无对照结论。</p>"
    )

    # ---- 图表数据 ----
    chart_data = {label: _series(results) for label, results in runs.items()}
    chart_json = json.dumps(chart_data, ensure_ascii=False)
    labels_json = json.dumps(list(runs.keys()), ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>低截获雷达智能功率调控仿真 实验结果</title>
<script src="{_CHART_CDN}"></script>
<style>
  :root {{ --line: #d8dee9; --head: #eef2f7; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
         margin: 28px auto; max-width: 1180px; color: #222; line-height: 1.55; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid var(--line); padding-bottom: 10px; }}
  h2 {{ font-size: 17px; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 9px; text-align: center; }}
  th {{ background: var(--head); font-weight: 600; }}
  th.rowhead {{ text-align: left; white-space: nowrap; }}
  table.info {{ width: auto; min-width: 460px; }}
  table.info th {{ text-align: left; }}
  .meta {{ color: #667; font-size: 13px; }}
  .card {{ border: 1px solid var(--line); border-radius: 6px; padding: 14px 18px;
           background: #fafbfc; margin: 14px 0; }}
  .chart {{ margin: 10px 0 26px; }}
  canvas {{ max-width: 100%; }}
  ul {{ margin: 8px 0 0 18px; }}
</style>
</head>
<body>
<h1>低截获雷达智能功率调控仿真 实验结果</h1>
<p class="meta">生成时间：{generated_at}</p>

<h2>1. 场景与链路预算</h2>
{info_table}

<h2>2. 核心指标对照</h2>
{summary_table}

<h2>3. 量化结论</h2>
<div class="card">{conclusions_html}</div>

<h2>4. 逐步曲线</h2>
<div class="chart"><canvas id="chartPower" height="110"></canvas></div>
<div class="chart"><canvas id="chartPd" height="110"></canvas></div>
<div class="chart"><canvas id="chartPint" height="110"></canvas></div>
<div class="chart"><canvas id="chartExposure" height="110"></canvas></div>
<div class="chart"><canvas id="chartEnergy" height="110"></canvas></div>

<script>
const RUNS = {chart_json};
const LABELS = {labels_json};
const COLORS = ["#d64545", "#2f7ed8", "#3aa76d", "#c07c1f", "#7a52cc"];

function buildChart(canvasId, seriesKey, title, yTitle) {{
  const datasets = LABELS.map((label, i) => ({{
    label: label,
    data: RUNS[label].time.map((t, k) => ({{ x: t, y: RUNS[label][seriesKey][k] }})),
    borderColor: COLORS[i % COLORS.length],
    backgroundColor: COLORS[i % COLORS.length],
    borderWidth: 1.8,
    pointRadius: 0,
    tension: 0.15,
  }}));
  new Chart(document.getElementById(canvasId), {{
    type: "line",
    data: {{ datasets: datasets }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{ title: {{ display: true, text: title }}, legend: {{ position: "bottom" }} }},
      scales: {{
        x: {{ type: "linear", title: {{ display: true, text: "时间 t (s)" }} }},
        y: {{ title: {{ display: true, text: yTitle }} }},
      }},
    }},
  }});
}}

buildChart("chartPower", "tx_power_w", "发射功率 vs 时间", "Pt (W)");
buildChart("chartPd", "pd_min", "最小探测概率 Pd vs 时间", "Pd");
buildChart("chartPint", "intercept_prob", "有效截获概率 Pint_eff vs 时间", "Pint_eff");
buildChart("chartExposure", "exposure", "累计暴露量 vs 时间", "exposure");
buildChart("chartEnergy", "cumulative_energy_j", "累计能耗 vs 时间", "E (J)");
</script>
</body>
</html>
"""


def write_html_report(
    summaries: Sequence[Dict[str, Any]],
    runs: Dict[str, Sequence[Any]],
    output_path: str,
    scenario_info: Dict[str, Any] | None = None,
) -> None:
    """生成并写出 HTML 报告。"""
    content = build_html_report(summaries, runs, scenario_info)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


# ----------------------------------------------------------------------
# 通用曲线页（供 DQN 训练曲线复用）
# ----------------------------------------------------------------------

def build_curves_html(
    curves: Sequence[Dict[str, Any]],
    page_title: str,
    subtitle: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    """生成多张折线图的 HTML 页面。

    curves 中每一项形如：

        {
          "title": "Episode Reward",
          "x_label": "episode",
          "y_label": "reward",
          "series": {"训练": {"x": [...], "y": [...]}, "评测": {...}},
        }
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in (metadata or {}).items()
    )
    meta_table = (
        f"<table class=\"info\"><tbody>{meta_rows}</tbody></table>" if meta_rows else ""
    )

    blocks = []
    scripts = []
    for i, curve in enumerate(curves):
        canvas_id = f"curve{i}"
        payload = {
            name: {"x": list(series["x"]), "y": list(series["y"])}
            for name, series in curve.get("series", {}).items()
        }
        blocks.append(
            f"<h2>{html.escape(str(curve.get('title', f'曲线 {i + 1}')))}</h2>"
            f'<div class="chart"><canvas id="{canvas_id}" height="110"></canvas></div>'
        )
        scripts.append(
            f"""
(function() {{
  const DATA = {json.dumps(payload, ensure_ascii=False)};
  const names = Object.keys(DATA);
  const datasets = names.map((name, k) => ({{
    label: name,
    data: DATA[name].x.map((x, j) => ({{ x: x, y: DATA[name].y[j] }})),
    borderColor: COLORS[k % COLORS.length],
    backgroundColor: COLORS[k % COLORS.length],
    borderWidth: 1.8,
    pointRadius: 0,
    tension: 0.15,
  }}));
  new Chart(document.getElementById("{canvas_id}"), {{
    type: "line",
    data: {{ datasets: datasets }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{ legend: {{ position: "bottom" }} }},
      scales: {{
        x: {{ type: "linear", title: {{ display: true, text: {json.dumps(str(curve.get('x_label', 'x')), ensure_ascii=False)} }} }},
        y: {{ title: {{ display: true, text: {json.dumps(str(curve.get('y_label', 'y')), ensure_ascii=False)} }} }},
      }},
    }},
  }});
}})();
"""
        )

    subtitle_html = f"<p class=\"meta\">{html.escape(subtitle)}</p>" if subtitle else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page_title)}</title>
<script src="{_CHART_CDN}"></script>
<style>
  :root {{ --line: #d8dee9; --head: #eef2f7; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
         margin: 28px auto; max-width: 1080px; color: #222; line-height: 1.55; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid var(--line); padding-bottom: 10px; }}
  h2 {{ font-size: 16px; margin-top: 26px; }}
  table {{ border-collapse: collapse; margin: 12px 0 20px; font-size: 13px; }}
  table.info {{ width: auto; min-width: 460px; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 9px; }}
  th {{ background: var(--head); text-align: left; }}
  .meta {{ color: #667; font-size: 13px; }}
  .chart {{ margin: 8px 0 22px; }}
  canvas {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>{html.escape(page_title)}</h1>
<p class="meta">生成时间：{generated_at}</p>
{subtitle_html}
{meta_table}
{''.join(blocks)}
<script>
const COLORS = ["#2f7ed8", "#d64545", "#3aa76d", "#c07c1f", "#7a52cc"];
{''.join(scripts)}
</script>
</body>
</html>
"""


def write_curves_html(
    curves: Sequence[Dict[str, Any]],
    output_path: str,
    page_title: str,
    subtitle: str = "",
    metadata: Dict[str, Any] | None = None,
) -> None:
    """生成并写出曲线页。"""
    content = build_curves_html(curves, page_title, subtitle, metadata)
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
