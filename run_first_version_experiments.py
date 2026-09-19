import importlib.util
import json
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
BACKUP_SIM_PATH = ROOT / "backup_original" / "simulator.py"
CONFIG_PATH = ROOT / "config" / "scenario_v1.json"
OUTPUT_HTML = ROOT / "output" / "experiment_results.html"


def load_backup_simulator() -> Any:
    spec = importlib.util.spec_from_file_location("backup_original_simulator", BACKUP_SIM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_weighted_simulator(weights: Dict[str, float]) -> Any:
    module = load_backup_simulator()
    SimulatorOriginal = module.Simulator
    LinkStateClass = module.LinkState
    sim = SimulatorOriginal(str(CONFIG_PATH))
    sim.load_config()

    def calc_base_quality(self, node_a, node_b, distance):
        assert self.scenario is not None
        avg_power_factor = (node_a.tx_power + node_b.tx_power) / 20.0
        path_loss = self.calc_path_loss(distance) * weights["path_loss_weight"]
        noise = self.scenario.background_noise * weights["noise_weight"]
        quality = avg_power_factor - path_loss - noise
        return max(0.0, min(1.0, quality))

    def calc_disturb_effect(self, node_a, node_b, current_time):
        effect = 0.0
        for d in self.disturbances:
            if not d.is_active(current_time):
                continue
            if node_a.current_freq not in d.freq_range or node_b.current_freq not in d.freq_range:
                continue

            da = ((node_a.x - d.center_x) ** 2 + (node_a.y - d.center_y) ** 2) ** 0.5
            db = ((node_b.x - d.center_x) ** 2 + (node_b.y - d.center_y) ** 2) ** 0.5
            min_dist = min(da, db)
            if min_dist > d.radius:
                continue

            spatial_factor = 1.0 - (min_dist / d.radius)
            spatial_factor = max(0.0, min(1.0, spatial_factor))
            effect += d.intensity * weights["disturb_weight"] * spatial_factor * d.duty_cycle

        return min(effect, 1.0)

    def build_links(self, current_time: int, use_disturbance: bool = False):
        assert self.scenario is not None
        links = []
        active_nodes = [n for n in self.nodes.values() if n.is_active]

        for node_a, node_b in __import__("itertools").combinations(active_nodes, 2):
            distance = self.calc_distance(node_a, node_b)
            if distance > self.scenario.max_comm_distance:
                continue

            path_loss = self.calc_path_loss(distance)
            base_quality = self.calc_base_quality(node_a, node_b, distance)
            disturb_effect = self.calc_disturb_effect(node_a, node_b, current_time) if use_disturbance else 0.0
            final_quality = max(0.0, base_quality - disturb_effect)
            threshold = max(node_a.rx_threshold, node_b.rx_threshold) * weights["threshold_scale"]
            available = final_quality >= threshold

            links.append(LinkStateClass(
                tx=node_a.node_id,
                rx=node_b.node_id,
                distance=distance,
                path_loss=path_loss,
                noise=self.scenario.background_noise,
                disturb_effect=disturb_effect,
                quality_score=final_quality,
                available=available,
            ))

        return links

    sim.calc_base_quality = MethodType(calc_base_quality, sim)
    sim.calc_disturb_effect = MethodType(calc_disturb_effect, sim)
    sim.build_links = MethodType(build_links, sim)
    return sim


def summarize_run(results: List[Dict[str, Any]]) -> Dict[str, float]:
    link_count = len(results[0]["links"]) if results else 0
    total_links = link_count if link_count else 1
    available_links = [l for step in results for l in step["links"] if l.available]
    quality_scores = [l.quality_score for step in results for l in step["links"]]
    disturb_scores = [l.disturb_effect for step in results for l in step["links"]]
    reachable_flows = sum(1 for step in results for f in step["flows"] if f["reachable"])
    total_flows = sum(len(step["flows"]) for step in results)
    step_count = len(results)

    return {
        "avg_available_links": sum(len([l for l in step["links"] if l.available]) for step in results) / step_count,
        "avg_quality": sum(quality_scores) / (len(quality_scores) or 1),
        "avg_disturb": sum(disturb_scores) / (len(disturb_scores) or 1),
        "reachable_ratio": reachable_flows / (total_flows or 1),
        "efficiency": (
            0.5 * (reachable_flows / (total_flows or 1))
            + 0.25 * (sum(len([l for l in step["links"] if l.available]) for step in results) / (step_count * total_links))
            + 0.25 * (sum(quality_scores) / (len(quality_scores) or 1))
        ),
    }


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_row][i]) < 1e-12:
            raise ValueError("Singular matrix")
        M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        M[i] = [x / pivot for x in M[i]]
        for j in range(n):
            if j != i:
                factor = M[j][i]
                M[j] = [Mj - factor * Mi for Mj, Mi in zip(M[j], M[i])]
    return [row[-1] for row in M]


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def linear_regression(X: List[List[float]], y: List[float]) -> Dict[str, Any]:
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = [sum(x_ij * yi for x_ij, yi in zip(x_i, y)) for x_i in Xt]
    beta = solve_linear_system(XtX, Xty)
    y_mean = sum(y) / len(y)
    y_pred = [sum(xij * bij for xij, bij in zip(x, beta)) for x, beta in zip(X, [beta] * len(X))]
    ssr = sum((yp - y_mean) ** 2 for yp in y_pred)
    sst = sum((yi - y_mean) ** 2 for yi in y)
    r2 = ssr / sst if sst else 1.0
    return {"coefficients": beta, "r2": r2}


def build_html(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> str:
    now = datetime.now().isoformat()
    headings = [
        "id",
        "path_loss_weight",
        "noise_weight",
        "disturb_weight",
        "threshold_scale",
        "baseline_reachable_ratio",
        "disturb_reachable_ratio",
        "disturb_efficiency",
    ]
    best_rows = sorted(rows, key=lambda r: r["disturb_efficiency"], reverse=True)[:5]
    data_json = json.dumps(rows)
    coefficients = model["coefficients"]
    r2 = model["r2"]

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>干扰-通信链路效率模型实验结果</title>
<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1,h2 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th,td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
th {{ background: #f4f4f4; }}
.summary-box {{ padding: 16px; background: #f9f9f9; border: 1px solid #ddd; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>干扰-通信链路效率模型实验结果</h1>
<p>生成时间：{now}</p>
<div class=\"summary-box\">
<h2>统计摘要</h2>
<p>实验次数：{len(rows)}</p>
<p>最佳实验(disturb_efficiency)：{best_rows[0]["disturb_efficiency"]:.4f}</p>
</div>
<h2>模型系数</h2>
<table>
<tr><th>变量</th><th>系数</th></tr>
<tr><td>截距</td><td>{coefficients[0]:.6f}</td></tr>
<tr><td>path_loss_weight</td><td>{coefficients[1]:.6f}</td></tr>
<tr><td>noise_weight</td><td>{coefficients[2]:.6f}</td></tr>
<tr><td>disturb_weight</td><td>{coefficients[3]:.6f}</td></tr>
<tr><td>threshold_scale</td><td>{coefficients[4]:.6f}</td></tr>
</table>
<p>R² = {r2:.4f}</p>
<h2>前五优实验</h2>
<table>
<tr>{''.join(f'<th>{h}</th>' for h in headings)}</tr>
{''.join('<tr>' + ''.join(f'<td>{row[h]:.4f}</td>' if isinstance(row[h], float) else f'<td>{row[h]}</td>' for h in headings) + '</tr>' for row in best_rows)}
</table>
<h2>全部实验结果</h2>
<table>
<tr>{''.join(f'<th>{h}</th>' for h in headings)}</tr>
{''.join('<tr>' + ''.join(f'<td>{row[h]:.4f}</td>' if isinstance(row[h], float) else f'<td>{row[h]}</td>' for h in headings) + '</tr>' for row in rows)}
</table>
<h2>权重与效率关系图</h2>
<canvas id=\"chart1\" width=800 height=320></canvas>
<canvas id=\"chart2\" width=800 height=320></canvas>
<canvas id=\"chart3\" width=800 height=320></canvas>
<canvas id=\"chart4\" width=800 height=320></canvas>
<script>
const rows = {data_json};
function buildDataset(field) {{
  return rows.map(r => r[field]);
}}
const config1 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs path_loss_weight', data: rows.map(r => {{x: r.path_loss_weight, y: r.disturb_efficiency}}), backgroundColor: 'rgba(75, 192, 192, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: 'path_loss_weight'}}}}, y: {{title: {{display: true, text: 'disturb_efficiency'}}}}}}}} }};
const config2 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs noise_weight', data: rows.map(r => {{x: r.noise_weight, y: r.disturb_efficiency}}), backgroundColor: 'rgba(153, 102, 255, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: 'noise_weight'}}}}, y: {{title: {{display: true, text: 'disturb_efficiency'}}}}}}}} }};
const config3 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs disturb_weight', data: rows.map(r => {{x: r.disturb_weight, y: r.disturb_efficiency}}), backgroundColor: 'rgba(255, 159, 64, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: 'disturb_weight'}}}}, y: {{title: {{display: true, text: 'disturb_efficiency'}}}}}}}} }};
const config4 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs threshold_scale', data: rows.map(r => {{x: r.threshold_scale, y: r.disturb_efficiency}}), backgroundColor: 'rgba(54, 162, 235, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: 'threshold_scale'}}}}, y: {{title: {{display: true, text: 'disturb_efficiency'}}}}}}}} }};
new Chart(document.getElementById('chart1'), config1);
new Chart(document.getElementById('chart2'), config2);
new Chart(document.getElementById('chart3'), config3);
new Chart(document.getElementById('chart4'), config4);
</script>
</body>
</html>
"""


def run_experiments(num_experiments: int = 20) -> None:
    random.seed(42)
    rows: List[Dict[str, Any]] = []

    param_ranges = {
        "path_loss_weight": (0.7, 1.5),
        "noise_weight": (0.5, 1.5),
        "disturb_weight": (0.5, 2.0),
        "threshold_scale": (0.8, 1.2),
    }

    for i in range(num_experiments):
        weights = {
            key: (param_ranges[key][0] if i == 0 else random.uniform(*param_ranges[key]))
            for key in param_ranges
        }
        if i == 0:
            weights = {"path_loss_weight": 1.0, "noise_weight": 1.0, "disturb_weight": 1.0, "threshold_scale": 1.0}

        sim = make_weighted_simulator(weights)
        baseline_results = sim.run_baseline()

        sim = make_weighted_simulator(weights)
        disturbed_results = sim.run_with_disturbance()

        baseline_summary = summarize_run(baseline_results)
        disturbed_summary = summarize_run(disturbed_results)
        row = OrderedDict(
            id=i + 1,
            path_loss_weight=weights["path_loss_weight"],
            noise_weight=weights["noise_weight"],
            disturb_weight=weights["disturb_weight"],
            threshold_scale=weights["threshold_scale"],
            baseline_reachable_ratio=baseline_summary["reachable_ratio"],
            baseline_avg_available_links=baseline_summary["avg_available_links"],
            baseline_avg_quality=baseline_summary["avg_quality"],
            baseline_avg_disturb=baseline_summary["avg_disturb"],
            disturb_reachable_ratio=disturbed_summary["reachable_ratio"],
            disturb_avg_available_links=disturbed_summary["avg_available_links"],
            disturb_avg_quality=disturbed_summary["avg_quality"],
            disturb_avg_disturb=disturbed_summary["avg_disturb"],
            disturb_efficiency=disturbed_summary["efficiency"],
        )
        rows.append(row)
        print(f"Experiment {i + 1}/{num_experiments}: {row}")

    X = [[1.0, r["path_loss_weight"], r["noise_weight"], r["disturb_weight"], r["threshold_scale"]] for r in rows]
    y = [r["disturb_efficiency"] for r in rows]
    model = linear_regression(X, y)

    os.makedirs(OUTPUT_HTML.parent, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(build_html(rows, model))

    print(f"Experiment completed. HTML report written to {OUTPUT_HTML}")


if __name__ == "__main__":
    run_experiments(20)
