from docx import Document

# 读取run_first_version_experiments.py的内容
with open('run_first_version_experiments.py', 'r', encoding='utf-8') as f:
    python_code = f.read()

# 创建文档
doc = Document()

# 添加标题
doc.add_heading('EP', level=1)

# 添加原始代码
doc.add_heading('原始代码', level=2)
doc.add_paragraph(python_code)

# 添加中文翻译
doc.add_heading('中文翻译', level=2)

chinese_code = '''import importlib.util
import json
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Tuple


# 项目根目录
项目根目录 = Path(__file__).resolve().parent
# 备份模拟器路径
备份模拟器路径 = 项目根目录 / "backup_original" / "simulator.py"
# 配置文件路径
配置文件路径 = 项目根目录 / "config" / "scenario_v1.json"
# 输出HTML路径
输出HTML路径 = 项目根目录 / "output" / "experiment_results.html"


def 加载备份模拟器() -> Any:
    """加载备份的模拟器模块"""
    spec = importlib.util.spec_from_file_location("backup_original_simulator", 备份模拟器路径)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def 创建加权模拟器(权重: Dict[str, float]) -> Any:
    """创建带有权重的模拟器"""
    module = 加载备份模拟器()
    原始模拟器 = module.Simulator
    链路状态类 = module.LinkState
    模拟器 = 原始模拟器(str(配置文件路径))
    模拟器.load_config()

    def 计算基础质量(self, 节点A, 节点B, 距离):
        """计算基础通信质量"""
        assert self.scenario is not None
        平均功率因子 = (节点A.tx_power + 节点B.tx_power) / 20.0
        路径损耗 = self.calc_path_loss(距离) * 权重["path_loss_weight"]
        噪声 = self.scenario.background_noise * 权重["noise_weight"]
        质量 = 平均功率因子 - 路径损耗 - 噪声
        return max(0.0, min(1.0, 质量))

    def 计算干扰效果(self, 节点A, 节点B, 当前时间):
        """计算干扰效果"""
        效果 = 0.0
        for d in self.disturbances:
            if not d.is_active(当前时间):
                continue
            if 节点A.current_freq not in d.freq_range or 节点B.current_freq not in d.freq_range:
                continue

            距离A = ((节点A.x - d.center_x) ** 2 + (节点A.y - d.center_y) ** 2) ** 0.5
            距离B = ((节点B.x - d.center_x) ** 2 + (节点B.y - d.center_y) ** 2) ** 0.5
            最小距离 = min(距离A, 距离B)
            if 最小距离 > d.radius:
                continue

            空间因子 = 1.0 - (最小距离 / d.radius)
            空间因子 = max(0.0, min(1.0, 空间因子))
            效果 += d.intensity * 权重["disturb_weight"] * 空间因子 * d.duty_cycle

        return min(效果, 1.0)

    def 构建链路(self, 当前时间: int, 使用干扰: bool = False):
        """构建通信链路"""
        assert self.scenario is not None
        链路列表 = []
        活跃节点 = [n for n in self.nodes.values() if n.is_active]

        for 节点A, 节点B in __import__("itertools").combinations(活跃节点, 2):
            距离 = self.calc_distance(节点A, 节点B)
            if 距离 > self.scenario.max_comm_distance:
                continue

            路径损耗 = self.calc_path_loss(距离)
            基础质量 = self.计算基础质量(节点A, 节点B, 距离)
            干扰效果 = self.计算干扰效果(节点A, 节点B, 当前时间) if 使用干扰 else 0.0
            最终质量 = max(0.0, 基础质量 - 干扰效果)
            阈值 = max(节点A.rx_threshold, 节点B.rx_threshold) * 权重["threshold_scale"]
            可用 = 最终质量 >= 阈值

            链路列表.append(链路状态类(
                tx=节点A.node_id,
                rx=节点B.node_id,
                distance=距离,
                path_loss=路径损耗,
                noise=self.scenario.background_noise,
                disturb_effect=干扰效果,
                quality_score=最终质量,
                available=可用,
            ))

        return 链路列表

    模拟器.计算基础质量 = MethodType(计算基础质量, 模拟器)
    模拟器.计算干扰效果 = MethodType(计算干扰效果, 模拟器)
    模拟器.构建链路 = MethodType(构建链路, 模拟器)
    return 模拟器


def 总结运行结果(结果: List[Dict[str, Any]]) -> Dict[str, float]:
    """总结运行结果"""
    链路数量 = len(结果[0]["links"]) if 结果 else 0
    总链路数 = 链路数量 if 链路数量 else 1
    可用链路 = [l for step in 结果 for l in step["links"] if l.available]
    质量分数 = [l.quality_score for step in 结果 for l in step["links"]]
    干扰分数 = [l.disturb_effect for step in 结果 for l in step["links"]]
    可达流 = sum(1 for step in 结果 for f in step["flows"] if f["reachable"])
    总流 = sum(len(step["flows"]) for step in 结果)
    步骤数 = len(结果)

    return {
        "平均可用链路数": sum(len([l for l in step["links"] if l.available]) for step in 结果) / 步骤数,
        "平均质量": sum(质量分数) / (len(质量分数) or 1),
        "平均干扰": sum(干扰分数) / (len(干扰分数) or 1),
        "可达率": 可达流 / (总流 or 1),
        "效率": (
            0.5 * (可达流 / (总流 or 1))
            + 0.25 * (sum(len([l for l in step["links"] if l.available]) for step in 结果) / (步骤数 * 总链路数))
            + 0.25 * (sum(质量分数) / (len(质量分数) or 1))
        ),
    }


def 解线性方程组(A: List[List[float]], b: List[float]) -> List[float]:
    """解线性方程组"""
    n = len(b)
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]
    for i in range(n):
        主元行 = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[主元行][i]) < 1e-12:
            raise ValueError("奇异矩阵")
        M[i], M[主元行] = M[主元行], M[i]
        主元 = M[i][i]
        M[i] = [x / 主元 for x in M[i]]
        for j in range(n):
            if j != i:
                因子 = M[j][i]
                M[j] = [Mj - 因子 * Mi for Mj, Mi in zip(M[j], M[i])]
    return [row[-1] for row in M]


def 矩阵转置(矩阵: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    return [list(row) for row in zip(*矩阵)]


def 矩阵乘法(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def 线性回归(X: List[List[float]], y: List[float]) -> Dict[str, Any]:
    """线性回归分析"""
    Xt = 矩阵转置(X)
    XtX = 矩阵乘法(Xt, X)
    Xty = [sum(x_ij * yi for x_ij, yi in zip(x_i, y)) for x_i in Xt]
    beta = 解线性方程组(XtX, Xty)
    y_mean = sum(y) / len(y)
    y_pred = [sum(xij * bij for xij, bij in zip(x, beta)) for x, beta in zip(X, [beta] * len(X))]
    ssr = sum((yp - y_mean) ** 2 for yp in y_pred)
    sst = sum((yi - y_mean) ** 2 for yi in y)
    r2 = ssr / sst if sst else 1.0
    return {"系数": beta, "r2": r2}


def 构建HTML(行数据: List[Dict[str, Any]], 模型: Dict[str, Any]) -> str:
    """构建HTML报告"""
    现在 = datetime.now().isoformat()
    标题 = [
        "id",
        "路径损耗权重",
        "噪声权重",
        "干扰权重",
        "阈值缩放",
        "基准可达率",
        "干扰可达率",
        "干扰效率",
    ]
    最佳行 = sorted(行数据, key=lambda r: r["干扰效率"], reverse=True)[:5]
    数据JSON = json.dumps(行数据)
    系数 = 模型["系数"]
    r2 = 模型["r2"]

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>干扰-通信链路效率模型实验结果</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
<p>生成时间：{现在}</p>
<div class="summary-box">
<h2>统计摘要</h2>
<p>实验次数：{len(行数据)}</p>
<p>最佳实验(干扰效率)：{最佳行[0]["干扰效率"]:.4f}</p>
</div>
<h2>模型系数</h2>
<table>
<tr><th>变量</th><th>系数</th></tr>
<tr><td>截距</td><td>{系数[0]:.6f}</td></tr>
<tr><td>路径损耗权重</td><td>{系数[1]:.6f}</td></tr>
<tr><td>噪声权重</td><td>{系数[2]:.6f}</td></tr>
<tr><td>干扰权重</td><td>{系数[3]:.6f}</td></tr>
<tr><td>阈值缩放</td><td>{系数[4]:.6f}</td></tr>
</table>
<p>R² = {r2:.4f}</p>
<h2>前五优实验</h2>
<table>
<tr>{''.join(f'<th>{h}</th>' for h in 标题)}</tr>
{''.join('<tr>' + ''.join(f'<td>{row[h]:.4f}</td>' if isinstance(row[h], float) else f'<td>{row[h]}</td>' for h in 标题) + '</tr>' for row in 最佳行)}
</table>
<h2>全部实验结果</h2>
<table>
<tr>{''.join(f'<th>{h}</th>' for h in 标题)}</tr>
{''.join('<tr>' + ''.join(f'<td>{row[h]:.4f}</td>' if isinstance(row[h], float) else f'<td>{row[h]}</td>' for h in 标题) + '</tr>' for row in 行数据)}
</table>
<h2>权重与效率关系图</h2>
<canvas id="chart1" width=800 height=320></canvas>
<canvas id="chart2" width=800 height=320></canvas>
<canvas id="chart3" width=800 height=320></canvas>
<canvas id="chart4" width=800 height=320></canvas>
<script>
const rows = {数据JSON};
function buildDataset(field) {{
  return rows.map(r => r[field]);
}}
const config1 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs 路径损耗权重', data: rows.map(r => {{x: r.路径损耗权重, y: r.干扰效率}}), backgroundColor: 'rgba(75, 192, 192, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: '路径损耗权重'}}}}, y: {{title: {{display: true, text: '干扰效率'}}}}}}}} }};
const config2 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs 噪声权重', data: rows.map(r => {{x: r.噪声权重, y: r.干扰效率}}), backgroundColor: 'rgba(153, 102, 255, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: '噪声权重'}}}}, y: {{title: {{display: true, text: '干扰效率'}}}}}}}} }};
const config3 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs 干扰权重', data: rows.map(r => {{x: r.干扰权重, y: r.干扰效率}}), backgroundColor: 'rgba(255, 159, 64, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: '干扰权重'}}}}, y: {{title: {{display: true, text: '干扰效率'}}}}}}}} }};
const config4 = {{ type: 'scatter', data: {{ datasets: [{{ label: '效率 vs 阈值缩放', data: rows.map(r => {{x: r.阈值缩放, y: r.干扰效率}}), backgroundColor: 'rgba(54, 162, 235, 0.7)'}}] }}, options: {{scales: {{x: {{type: 'linear', position: 'bottom', title: {{display: true, text: '阈值缩放'}}}}, y: {{title: {{display: true, text: '干扰效率'}}}}}}}} }};
new Chart(document.getElementById('chart1'), config1);
new Chart(document.getElementById('chart2'), config2);
new Chart(document.getElementById('chart3'), config3);
new Chart(document.getElementById('chart4'), config4);
</script>
</body>
</html>
"""


def 运行实验(实验次数: int = 20) -> None:
    """运行实验"""
    random.seed(42)
    行数据: List[Dict[str, Any]] = []

    参数范围 = {
        "path_loss_weight": (0.7, 1.5),
        "noise_weight": (0.5, 1.5),
        "disturb_weight": (0.5, 2.0),
        "threshold_scale": (0.8, 1.2),
    }

    for i in range(实验次数):
        权重 = {
            key: (参数范围[key][0] if i == 0 else random.uniform(*参数范围[key]))
            for key in 参数范围
        }
        if i == 0:
            权重 = {"path_loss_weight": 1.0, "noise_weight": 1.0, "disturb_weight": 1.0, "threshold_scale": 1.0}

        模拟器 = 创建加权模拟器(权重)
        基准结果 = 模拟器.run_baseline()

        模拟器 = 创建加权模拟器(权重)
        干扰结果 = 模拟器.run_with_disturbance()

        基准摘要 = 总结运行结果(基准结果)
        干扰摘要 = 总结运行结果(干扰结果)
        行 = OrderedDict(
            id=i + 1,
            路径损耗权重=权重["path_loss_weight"],
            噪声权重=权重["noise_weight"],
            干扰权重=权重["disturb_weight"],
            阈值缩放=权重["threshold_scale"],
            基准可达率=基准摘要["可达率"],
            基准平均可用链路数=基准摘要["平均可用链路数"],
            基准平均质量=基准摘要["平均质量"],
            基准平均干扰=基准摘要["平均干扰"],
            干扰可达率=干扰摘要["可达率"],
            干扰平均可用链路数=干扰摘要["平均可用链路数"],
            干扰平均质量=干扰摘要["平均质量"],
            干扰平均干扰=干扰摘要["平均干扰"],
            干扰效率=干扰摘要["效率"],
        )
        行数据.append(行)
        print(f"实验 {i + 1}/{实验次数}: {行}")

    X = [[1.0, r["路径损耗权重"], r["噪声权重"], r["干扰权重"], r["阈值缩放"]] for r in 行数据]
    y = [r["干扰效率"] for r in 行数据]
    模型 = 线性回归(X, y)

    os.makedirs(输出HTML路径.parent, exist_ok=True)
    with open(输出HTML路径, "w", encoding="utf-8") as f:
        f.write(构建HTML(行数据, 模型))

    print(f"实验完成。HTML报告已写入 {输出HTML路径}")


if __name__ == "__main__":
    运行实验(20)
'''

# 添加中文翻译代码
doc.add_paragraph(chinese_code)

# 保存文档
doc.save('EP_cn.docx')
print('Word document created: EP_cn.docx')
