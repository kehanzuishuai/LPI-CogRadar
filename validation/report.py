"""环境可信度报告（v4.3）。

报告结构（用户明确要求的四类）
------------------------------
1. **已验证模块**：由 `validation/checks.py` 的**实测结果**推导，不是手写声明；
2. **尚未验证的假设**：明确列出"我们没测过、但结论依赖它"的东西；
3. **可支持的结论**：只有在已有证据（校核结果 + 实验数据）支持时才列出；
4. **不可支持的结论**：明确禁止的表述，防止后续被误用/夸大。

设计原则
--------
**已验证 / 未验证不能手写**。如果把"已验证"写成一份静态清单，
它迟早会和实际校核结果脱节（测试挂了但报告还写着通过）。
因此本模块的 `verified_modules` 完全由 `check_report` 推导：
哪一层的检查全过，哪一层才算"已验证"；
有任何一条 FAIL，该层就降级为"部分验证"，并列出失败的检查 ID。

公开工具 / 公开数据的使用边界
-----------------------------
本阶段允许对**非敏感**模块做有限外部校核，但必须写明性质：

* 允许比较的是：通用雷达量测误差随距离的关系、异步多传感器跟踪的
  融合误差量级、多雷达观测几何（可见性/覆盖）的结构性结论；
* **不得**据此声称验证了真实装备、真实战场性能、真实电子战效能。
  本仿真里的雷达/侦察/干扰参数是**教学与算法研究用的等效参数**，
  不与任何具体型号对应。
这一条同时写进报告正文与 README，避免只在代码注释里。
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: 各层的中文名（报告里用）
LAYER_CN: Dict[str, str] = {
    "geometry": "几何层",
    "measurement": "测量层",
    "communication": "通信层",
    "fusion": "融合层",
}

#: 数据生成链（每份报告都带上，提醒结论的追溯关系）
DATA_CHAIN: List[str] = [
    "真实状态（Scene / Simulator，唯一真值）",
    "可见性（sensor：作用距离 / 视场 / 遮挡 / 更新周期 / 可用状态）",
    "测量（MeasurementRecord：带噪声、协方差、时间戳、候选 ID）",
    "通信（communication：生成/发送/到达时刻、延迟、丢包、带宽、过期）",
    "融合（fusion：关联 + 加权估计 + 溯源）",
    "决策（规则 / DQN / 集成 / AI 诊断，只读已到达的融合结果）",
]

#: **尚未验证的假设**——这些是"结论依赖它、但我们没有实测"的东西。
UNVERIFIED_ASSUMPTIONS: List[Dict[str, str]] = [
    {"id": "ASM-01", "assumption": "各量测误差相互独立且零均值高斯",
     "impact": "加权融合的最小方差性质、卡方式门限都依赖它；真实测量可能有偏与相关"},
    {"id": "ASM-02", "assumption": "协方差按对角处理，忽略方位-俯仰-距离交叉项",
     "impact": "远距离横向误差被各向同性地近似，融合权重不最优"},
    {"id": "ASM-03", "assumption": "目标做匀速直线运动，外推无机动模型",
     "impact": "机动目标会导致关联失败与丢轨，本阶段未测机动场景"},
    {"id": "ASM-04", "assumption": "遮挡是 AABB / 球的简单几何，无地形高程与地球曲率",
     "impact": "复杂地形下的通视结论不适用"},
    {"id": "ASM-05", "assumption": "虚拟告警是独立伯努利事件，无 CFAR 门限自适应",
     "impact": "密集杂波环境下的虚警率结论不适用"},
    {"id": "ASM-06", "assumption": "通信链路的延迟/丢包独立同分布，无突发错误与重传",
     "impact": "真实数据链的突发丢包会让协同性能比本模型更差"},
    {"id": "ASM-07", "assumption": "单站被动传感器无距离量测，且不参与位置更新",
     "impact": "多站被动定位的协同增益未建模，本阶段低估被动传感器价值"},
    {"id": "ASM-08", "assumption": "融合未做 JPDA / MHT，密集目标下关联会串",
     "impact": "近距多目标场景的丢轨/重复轨迹会比本报告更严重"},
    {"id": "ASM-09", "assumption": "仿真参数为教学/研究等效参数，不对应具体型号",
     "impact": "任何绝对性能数字都不可外推到真实装备"},
]

#: **不可支持的结论**——明确禁止的表述。
UNSUPPORTED_CONCLUSIONS: List[Dict[str, str]] = [
    {"claim": "本仿真验证了真实雷达/侦察/干扰装备的性能",
     "why": "参数是等效参数，无实测标定；也没有真实战场环境"},
    {"claim": "本仿真预测了真实电子战条件下的作战效能",
     "why": "无地形、无多径、无真实信号环境、无对抗战术建模"},
    {"claim": "融合算法在密集目标下同样有效",
     "why": "只用最近邻关联，未做 JPDA/MHT，也未做密集目标压力测试"},
    {"claim": "通信受限下的协同收益已被充分刻画",
     "why": "只做了三档共享策略对照，未扫全延迟-丢包组合空间"},
    {"claim": "被动传感器已产生协同增益",
     "why": "单站被动无距离量测，本阶段不参与位置更新"},
    {"claim": "AI 诊断不会编造",
     "why": "正确表述是「AI 解释被限制在结构化证据范围内，并通过规则检查」——"
            "它被约束而不是被证明不会错"},
    {"claim": "DQN 在测量级环境下优于规则策略",
     "why": "需在完全相同信息权限下重新评测；此前 DQN 与规则的差异不显著（t≈1.03）"},
]


@dataclass
class TrustReport:
    """环境可信度报告。"""

    check_report: Dict[str, Any]
    verified_modules: List[str] = field(default_factory=list)
    partially_verified_modules: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    verified_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    supported_conclusions: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_from": {
                "config_path": self.check_report.get("config_path"),
                "seeds": self.check_report.get("seeds"),
                "measurement_steps": self.check_report.get("measurement_steps"),
                "fusion_steps": self.check_report.get("fusion_steps"),
            },
            "data_generation_chain": DATA_CHAIN,
            "verified_modules": self.verified_modules,
            "partially_verified_modules": self.partially_verified_modules,
            "failed_checks": self.failed_checks,
            "verified_details": self.verified_details,
            "unverified_assumptions": UNVERIFIED_ASSUMPTIONS,
            "supported_conclusions": self.supported_conclusions,
            "unsupported_conclusions": UNSUPPORTED_CONCLUSIONS,
            "external_validation_policy": {
                "allowed": [
                    "通用雷达量测误差随距离的关系（趋势与量级）",
                    "异步多传感器跟踪的融合误差量级",
                    "多雷达观测几何的结构性结论（覆盖/可见性）",
                ],
                "forbidden": [
                    "声称验证了真实装备性能",
                    "声称验证了真实战场/电子战效能",
                    "把等效参数当作具体型号参数引用",
                ],
                "note": "外部对比只能用于『实现是否符合通用物理与统计规律』，"
                        "不能用于证明装备能力。",
            },
            "all_passed": self.check_report.get("all_passed", False),
        }

    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# LPI-CogRadar 环境可信度报告")
        lines.append("")
        lines.append("> 本报告由 `run_validation.py` 从**实际校核结果**自动生成，"
                     "不是手写声明。")
        lines.append("")
        lines.append("## 0. 数据生成链")
        lines.append("")
        lines.append("任何实验结果都应能追溯到这条链上的明确环节：")
        lines.append("")
        for index, stage in enumerate(DATA_CHAIN, 1):
            lines.append(f"{index}. {stage}")
        lines.append("")
        src = self.check_report
        lines.append(f"本次校核来源：`{src.get('config_path')}`，"
                     f"种子 {src.get('seeds')}，"
                     f"测量层 {src.get('measurement_steps')} 步，"
                     f"融合层 {src.get('fusion_steps')} 步。")
        lines.append("")

        lines.append("## 1. 已验证模块")
        lines.append("")
        if self.verified_modules:
            lines.append("| 层 | 检查项数 | 全部通过 |")
            lines.append("| --- | --- | --- |")
            for layer in self.verified_modules:
                detail = self.verified_details.get(layer, {})
                lines.append(f"| {LAYER_CN.get(layer, layer)} | "
                             f"{detail.get('total', 0)} | ✅ |")
        else:
            lines.append("（没有任何一层通过全部检查）")
        if self.partially_verified_modules:
            lines.append("")
            lines.append("**部分验证（存在失败项）**：")
            for layer in self.partially_verified_modules:
                detail = self.verified_details.get(layer, {})
                lines.append(f"* {LAYER_CN.get(layer, layer)}："
                             f"通过 {detail.get('passed', 0)}/{detail.get('total', 0)}，"
                             f"失败项 {detail.get('failed', [])}")
        if self.failed_checks:
            lines.append("")
            lines.append(f"**失败检查**：{self.failed_checks}")
        lines.append("")

        lines.append("## 2. 尚未验证的假设")
        lines.append("")
        lines.append("这些是**结论依赖、但本阶段没有实测**的东西。"
                     "引用任何结论时必须连带说明它们。")
        lines.append("")
        lines.append("| 编号 | 假设 | 影响 |")
        lines.append("| --- | --- | --- |")
        for item in UNVERIFIED_ASSUMPTIONS:
            lines.append(f"| {item['id']} | {item['assumption']} | {item['impact']} |")
        lines.append("")

        lines.append("## 3. 可支持的结论")
        lines.append("")
        if self.supported_conclusions:
            lines.append("| 结论 | 依据 |")
            lines.append("| --- | --- |")
            for item in self.supported_conclusions:
                lines.append(f"| {item['claim']} | {item['evidence']} |")
        else:
            lines.append("（本阶段没有可列出的结论）")
        lines.append("")

        lines.append("## 4. 不可支持的结论")
        lines.append("")
        lines.append("以下表述**不允许**出现在论文、答辩或对外材料中：")
        lines.append("")
        lines.append("| 禁止的表述 | 原因 |")
        lines.append("| --- | --- |")
        for item in UNSUPPORTED_CONCLUSIONS:
            lines.append(f"| {item['claim']} | {item['why']} |")
        lines.append("")

        lines.append("## 5. 外部校核的边界")
        lines.append("")
        lines.append("**允许**（用于检查实现是否符合通用物理与统计规律）：")
        for item in self.to_dict()["external_validation_policy"]["allowed"]:
            lines.append(f"* {item}")
        lines.append("")
        lines.append("**禁止**：")
        for item in self.to_dict()["external_validation_policy"]["forbidden"]:
            lines.append(f"* {item}")
        lines.append("")
        lines.append("> 本仿真的雷达 / 侦察 / 干扰参数是**教学与算法研究用的等效参数**，"
                     "不与任何具体型号对应。")
        return "\n".join(lines)

    def write(self, out_dir: str, stem: str = "trust_report") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{stem}.json")
        md_path = os.path.join(out_dir, f"{stem}.md")
        with io.open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        with io.open(md_path, "w", encoding="utf-8") as handle:
            handle.write(self.to_markdown())
        return {"json": json_path, "markdown": md_path}


def build_trust_report(check_report: Dict[str, Any]) -> TrustReport:
    """从校核结果推导可信度报告。

    ⚠️ 「已验证」完全由 `check_report` 推导：
    某层全部检查通过才进 `verified_modules`；有失败项则进
    `partially_verified_modules` 并列出失败 ID。
    **不允许**手写"已验证"清单——那样它迟早会和实际测试脱节。
    """
    summary: Dict[str, Dict[str, Any]] = check_report.get("summary", {})
    verified: List[str] = []
    partial: List[str] = []
    failed: List[str] = []
    details: Dict[str, Dict[str, Any]] = {}

    for layer, info in summary.items():
        total = int(info.get("total", 0))
        passed = int(info.get("passed", 0))
        layer_failed: List[str] = list(info.get("failed", []) or [])
        details[layer] = {
            "total": total, "passed": passed, "failed": layer_failed,
        }
        failed.extend(layer_failed)
        if total > 0 and passed == total:
            verified.append(layer)
        else:
            partial.append(layer)

    # --- 可支持的结论：只在对应层已验证时才列出 ---
    supported: List[Dict[str, str]] = []
    if "geometry" in verified:
        supported.append({
            "claim": "几何层满足数学恒等式（距离逐位对称、方位/俯仰定义正确、"
                     "运动学与闭式解一致、姿态正交归一）",
            "evidence": "validation/checks.py::check_geometry 全部通过",
        })
    if "measurement" in verified:
        supported.append({
            "claim": "测量层按配置工作（误差无偏、虚警率与周期符合配置，"
                     "且多种缺失原因可区分）",
            "evidence": "validation/checks.py::check_measurement 全部通过",
        })
    if "communication" in verified:
        supported.append({
            "claim": "通信层按配置工作（延迟均值等于固定延迟、投递率≈1−丢包率、"
                     "过期按时丢弃），且决策只能读到已到达消息",
            "evidence": "validation/checks.py::check_communication 全部通过",
        })
    if "fusion" in verified:
        supported.append({
            "claim": "融合层能产出可溯源航迹（每条结果都能回答"
                     "「用了哪个传感器、哪个时刻的测量」）",
            "evidence": "validation/checks.py::check_fusion 全部通过",
        })
    supported.append({
        "claim": "旧简化环境（full 真值观测）的全部既有数值在升级后保持逐位可复现",
        "evidence": "main.py / evaluate_dqn.py 回归 + tests 里的逐位断言",
    })
    supported.append({
        "claim": "信念桥接不再携带真值的未来干扰起伏与他平台位置（行为级检验）",
        "evidence": "tests/test_belief_policy.py::TestNoFutureLeak（9 项）",
    })

    return TrustReport(
        check_report=check_report,
        verified_modules=verified,
        partially_verified_modules=partial,
        failed_checks=failed,
        verified_details=details,
        supported_conclusions=supported,
    )
