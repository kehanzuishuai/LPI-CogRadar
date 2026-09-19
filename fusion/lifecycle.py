"""测量生命周期追踪（v4.4）。

要回答的问题
------------
"某条远端测量最终在哪里被丢弃、为什么没有进入航迹？"

升级前无法回答这个问题：测量进了融合中心就消失了，没有留下任何痕迹。
因此实测"收到数千条共享测量但航迹指标不变"时，只能靠猜。

本模块给每条测量一个**贯穿全链路的身份**，逐步记录它的去向：

    generated → sent → arrived → stale_rejected / gate_rejected / kind_rejected
              → associated → track_created / track_updated

每一步都记录：`sensor_id`、`source_platform`、`candidate_id`、各阶段时刻、
`reject_reason`、以及最终落到哪个 `track_id`。

设计要点
--------
* **漏斗统计**（`funnel()`）把"哪一步拦掉了多少"一眼看清，
  这是诊断协同收益的关键工具；
* **不读真值**。生命周期记录的是"测量走到哪了"，与真值无关；
  真值只在离线评测里用于算误差/召回。
* 生命周期是**可关闭的**（`enabled=False`）：全量记录会给大场景带来
  明显开销，正式长跑时可以关掉，诊断时再开。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: 生命周期阶段（顺序即流水线顺序）
STAGE_GENERATED = "generated"
STAGE_SENT = "sent"
STAGE_ARRIVED = "arrived"
STAGE_STALE_REJECTED = "stale_rejected"
STAGE_KIND_REJECTED = "kind_rejected"
STAGE_GATE_REJECTED = "gate_rejected"
STAGE_ASSOCIATED = "associated"
STAGE_TRACK_CREATED = "track_created"
STAGE_TRACK_UPDATED = "track_updated"

#: 全部"被拒绝"的阶段（漏斗里需要逐项统计）
REJECT_STAGES = (
    STAGE_STALE_REJECTED,
    STAGE_KIND_REJECTED,
    STAGE_GATE_REJECTED,
)

#: 进入航迹的阶段
ACCEPT_STAGES = (STAGE_TRACK_CREATED, STAGE_TRACK_UPDATED)

STAGE_CN: Dict[str, str] = {
    STAGE_GENERATED: "已生成（传感器产出测量）",
    STAGE_SENT: "已发送（进入通信链路）",
    STAGE_ARRIVED: "已到达（决策侧可见）",
    STAGE_STALE_REJECTED: "被拒：超过时效",
    STAGE_KIND_REJECTED: "被拒：观测对象不匹配",
    STAGE_GATE_REJECTED: "被拒：未通过关联门限",
    STAGE_ASSOCIATED: "已关联到航迹",
    STAGE_TRACK_CREATED: "新建航迹",
    STAGE_TRACK_UPDATED: "更新航迹",
}


#: 关联层拒绝原因（与发现码 `ASSOCIATION_AMBIGUOUS` 配合）
REJECT_MAHALANOBIS = "mahalanobis_gate"
REJECT_AZIMUTH = "azimuth_gate"
REJECT_ELEVATION = "elevation_gate"
REJECT_NO_TRACK = "no_track"
REJECT_DROPPED = "track_dropped"


@dataclass
class AssociationCandidate:
    """**一条测量 vs 一条航迹**的关联评估记录（关联层审计的核心）。

    用户要求"关联可审计"：不能只留"最终关联到 T3"，
    还必须留下"当时比较过哪几条航迹、各自的门限距离是多少、
    为什么没选它们"。否则 ID switch / 误关联只能靠指标反推，
    无法回答"这一帧为什么关联错了"。

    ⚠️ 这里**只有航迹 ID 与几何量**，没有真值 ID ——
    关联层的审计记录本身也不得携带真值。
    """

    track_id: str
    #: 门限距离 = 马氏距离平方（用预测协方差归一化后的残差²）
    mahalanobis_sq: float
    #: 当时生效的门限
    gate_mahalanobis_sq: float
    #: 该测量到**预测位置**的残差范数（米）。v4.5 加：被门限拒绝的测量同样要留残差
    #: ——"为什么拒"比"拒了几条"更有信息量，而且目标机动时**恰恰是被拒的那条**
    #: 最能说明运动模型失配。
    residual_m: float = 0.0
    #: 与航迹预测方位的夹角（度）；无角度量测时为 None
    az_diff_deg: Optional[float] = None
    el_diff_deg: Optional[float] = None
    #: 是否通过全部门限
    passed: bool = False
    #: 未通过时的原因（`REJECT_*`），通过时为空串
    reject_reason: str = ""
    #: 是否被最终选中
    chosen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "mahalanobis_sq": round(self.mahalanobis_sq, 6),
            "gate_mahalanobis_sq": round(self.gate_mahalanobis_sq, 6),
            "residual_m": round(self.residual_m, 6),
            "az_diff_deg": (None if self.az_diff_deg is None
                            else round(self.az_diff_deg, 6)),
            "el_diff_deg": (None if self.el_diff_deg is None
                            else round(self.el_diff_deg, 6)),
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "chosen": self.chosen,
        }

    def brief(self) -> str:
        mark = "✔" if self.chosen else ("○" if self.passed else "✘")
        reason = f"({self.reject_reason})" if self.reject_reason else ""
        return f"{self.track_id}:{self.mahalanobis_sq:.2f}{mark}{reason}"


@dataclass
class MeasurementTrace:
    """一条测量的完整生命周期记录。"""

    trace_id: str
    candidate_id: str = ""
    sensor_id: str = ""
    sensor_kind: str = ""
    source_platform: str = ""
    msg_id: str = ""
    #: 是否为经通信到达的远端测量
    is_remote: bool = False

    generated_at: Optional[float] = None
    sent_at: Optional[float] = None
    arrived_at: Optional[float] = None
    measured_at: Optional[float] = None
    #: 决策侧使用它的时刻
    consumed_at: Optional[float] = None

    stages: List[str] = field(default_factory=list)
    reject_reason: str = ""
    track_id: str = ""
    #: 关联代价（通过了关联时记录，便于分析门限是否过紧）
    association_cost: Optional[float] = None
    #: 备注（例如门限残差明细）
    note: str = ""

    # --- 关联层审计（v4.5）---
    #: 本帧评估过的全部候选航迹（含被门限拒绝的）
    association_candidate_tracks: List[AssociationCandidate] = field(
        default_factory=list
    )
    #: 通过门限的候选航迹数（≥2 即"关联存在歧义"）
    n_tracks_in_gate: int = 0
    #: 最终选中的航迹（未关联上时为空串）
    chosen_track_id: str = ""
    #: 关联是否存在歧义（多个候选都过门限）—— 对应发现码 ASSOCIATION_AMBIGUOUS
    ambiguous: bool = False

    # ------------------------------------------------------------------

    def mark(self, stage: str) -> None:
        if stage not in self.stages:
            self.stages.append(stage)

    def reject(self, stage: str, reason: str = "") -> None:
        self.mark(stage)
        self.reject_reason = reason or stage

    @property
    def final_stage(self) -> str:
        return self.stages[-1] if self.stages else ""

    @property
    def reached_track(self) -> bool:
        return any(s in ACCEPT_STAGES for s in self.stages)

    @property
    def rejected(self) -> bool:
        return any(s in REJECT_STAGES for s in self.stages)

    def age_at_consume_s(self) -> Optional[float]:
        if self.consumed_at is None or self.measured_at is None:
            return None
        return max(0.0, self.consumed_at - self.measured_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "candidate_id": self.candidate_id,
            "sensor_id": self.sensor_id,
            "sensor_kind": self.sensor_kind,
            "source_platform": self.source_platform,
            "msg_id": self.msg_id,
            "is_remote": self.is_remote,
            "generated_at": self.generated_at,
            "sent_at": self.sent_at,
            "arrived_at": self.arrived_at,
            "measured_at": self.measured_at,
            "consumed_at": self.consumed_at,
            "age_at_consume_s": self.age_at_consume_s(),
            "stages": "→".join(self.stages),
            "final_stage": self.final_stage,
            "reject_reason": self.reject_reason,
            "track_id": self.track_id,
            "association_cost": self.association_cost,
            "reached_track": self.reached_track,
            "note": self.note,
            # --- 关联层审计 ---
            "n_candidate_tracks": len(self.association_candidate_tracks),
            "n_tracks_in_gate": self.n_tracks_in_gate,
            "chosen_track_id": self.chosen_track_id,
            "ambiguous": self.ambiguous,
            "candidates_brief": ";".join(
                c.brief() for c in self.association_candidate_tracks
            ),
        }

    def association_rows(self) -> List[Dict[str, Any]]:
        """展开成"一行 = 一个 (测量, 候选航迹) 对"，供 CSV 逐条审计。"""
        rows: List[Dict[str, Any]] = []
        base = {
            "trace_id": self.trace_id,
            "candidate_id": self.candidate_id,
            "sensor_id": self.sensor_id,
            "is_remote": self.is_remote,
            "measured_at": self.measured_at,
            "n_tracks_in_gate": self.n_tracks_in_gate,
            "ambiguous": self.ambiguous,
            "chosen_track_id": self.chosen_track_id,
            "final_stage": self.final_stage,
            "reject_reason": self.reject_reason,
        }
        if not self.association_candidate_tracks:
            rows.append(dict(base, track_id="", mahalanobis_sq=None,
                             gate_mahalanobis_sq=None, az_diff_deg=None,
                             el_diff_deg=None, passed=None, chosen=False,
                             gate_reject_reason=""))
            return rows
        for candidate in self.association_candidate_tracks:
            row = dict(base)
            row.update(candidate.to_dict())
            row["gate_reject_reason"] = candidate.reject_reason
            rows.append(row)
        return rows


class LifecycleLog:
    """生命周期日志：逐条测量追踪 + 漏斗统计 + 导出。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.traces: List[MeasurementTrace] = []
        self._by_id: Dict[str, MeasurementTrace] = {}
        self._counter = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.traces.clear()
        self._by_id.clear()
        self._counter = 0

    def open_trace(
        self,
        candidate_id: str = "",
        sensor_id: str = "",
        sensor_kind: str = "",
        source_platform: str = "",
        msg_id: str = "",
        is_remote: bool = False,
        measured_at: Optional[float] = None,
        generated_at: Optional[float] = None,
    ) -> Optional[MeasurementTrace]:
        """开一条追踪。`enabled=False` 时返回 None（调用方需容忍 None）。"""
        if not self.enabled:
            return None
        self._counter += 1
        trace = MeasurementTrace(
            trace_id=f"TR{self._counter:06d}",
            candidate_id=candidate_id, sensor_id=sensor_id,
            sensor_kind=sensor_kind, source_platform=source_platform,
            msg_id=msg_id, is_remote=is_remote,
            measured_at=measured_at, generated_at=generated_at,
        )
        trace.mark(STAGE_GENERATED)
        self.traces.append(trace)
        self._by_id[trace.trace_id] = trace
        return trace

    def mark(self, trace: Optional[MeasurementTrace], stage: str) -> None:
        """给某个阶段打标（`trace` 为 None 时静默跳过）。

        为什么放在日志对象上而不是让调用方直接调 `trace.mark()`：
        融合中心里到处都是 `if trace is not None:` 会让主流程被审计代码淹没。
        把"容忍 None"收敛到这几个方法里，主流程保持干净。
        """
        if trace is None:
            return
        trace.mark(stage)

    def reject(
        self, trace: Optional[MeasurementTrace], stage: str, reason: str = ""
    ) -> None:
        """记录一次拒绝。`trace` 为 None 时静默跳过（生命周期可关闭）。"""
        if trace is None:
            return
        trace.reject(stage, reason)

    def mark_sent(self, trace: Optional[MeasurementTrace], at: Optional[float]) -> None:
        if trace is None:
            return
        trace.sent_at = at
        trace.mark(STAGE_SENT)

    def record_association(
        self,
        trace: Optional[MeasurementTrace],
        candidates: Sequence[AssociationCandidate],
        chosen_track_id: str = "",
    ) -> None:
        """落盘关联层审计：全部候选 + 门限距离 + 拒绝原因 + 最终选择。

        `trace` 为 None（生命周期关闭）时静默跳过。
        """
        if trace is None:
            return
        trace.association_candidate_tracks = list(candidates)
        trace.n_tracks_in_gate = sum(1 for c in candidates if c.passed)
        trace.chosen_track_id = chosen_track_id
        trace.ambiguous = trace.n_tracks_in_gate >= 2

    def association_ambiguity_stats(self) -> Dict[str, Any]:
        """关联歧义统计（供诊断：多少比例的测量面对多个可关联航迹）。"""
        evaluated = [t for t in self.traces if t.association_candidate_tracks]
        ambiguous = [t for t in evaluated if t.ambiguous]
        in_gate = [t for t in evaluated if t.n_tracks_in_gate >= 1]
        return {
            "n_evaluated": len(evaluated),
            "n_with_candidate": len(in_gate),
            "n_ambiguous": len(ambiguous),
            "ambiguous_rate": (len(ambiguous) / len(evaluated)) if evaluated else 0.0,
            "mean_candidate_tracks": (
                sum(len(t.association_candidate_tracks) for t in evaluated)
                / len(evaluated)
            ) if evaluated else 0.0,
        }


    def mark_arrived(self, trace: Optional[MeasurementTrace], at: Optional[float]) -> None:
        if trace is None:
            return
        trace.arrived_at = at
        trace.mark(STAGE_ARRIVED)

    def get(self, trace_id: str) -> Optional[MeasurementTrace]:
        return self._by_id.get(trace_id)

    def find_by_msg(self, msg_id: str) -> List[MeasurementTrace]:
        return [t for t in self.traces if t.msg_id == msg_id]

    # ------------------------------------------------------------------

    def funnel(self) -> Dict[str, Any]:
        """漏斗：每一步有多少测量通过、多少被拦、原因是什么。

        这是回答"数千条共享测量为什么没进航迹"的**主工具**：
        看哪一列的计数把总量吃掉了即可。
        """
        total = len(self.traces)
        by_final: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        per_sensor: Dict[str, Dict[str, int]] = {}
        remote_total = 0
        remote_in_track = 0
        local_total = 0
        local_in_track = 0

        for trace in self.traces:
            by_final[trace.final_stage] = by_final.get(trace.final_stage, 0) + 1
            if trace.reject_reason:
                by_reason[trace.reject_reason] = by_reason.get(trace.reject_reason, 0) + 1
            bucket = per_sensor.setdefault(trace.sensor_id, {})
            key = trace.final_stage or "?"
            bucket[key] = bucket.get(key, 0) + 1
            if trace.is_remote:
                remote_total += 1
                remote_in_track += 1 if trace.reached_track else 0
            else:
                local_total += 1
                local_in_track += 1 if trace.reached_track else 0

        return {
            "total": total,
            "by_final_stage": by_final,
            "by_reject_reason": by_reason,
            "per_sensor": per_sensor,
            "local": {
                "total": local_total,
                "in_track": local_in_track,
                "utilization": (local_in_track / local_total) if local_total else 0.0,
            },
            "remote": {
                "total": remote_total,
                "in_track": remote_in_track,
                #: **远端测量利用率**：到达的远端测量里有多少真的进了航迹
                "utilization": (remote_in_track / remote_total) if remote_total else 0.0,
            },
        }

    # ------------------------------------------------------------------

    def write_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        rows = [t.to_dict() for t in self.traces]
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def write_association_csv(self, path: str) -> str:
        """导出关联层审计：一行 = 一个 (测量, 候选航迹) 对。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for trace in self.traces:
            rows.extend(trace.association_rows())
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def format_association_audit(self) -> str:
        """关联层审计摘要：歧义比例 + 门限拒绝原因分布。"""
        stats = self.association_ambiguity_stats()
        reasons: Dict[str, int] = {}
        for trace in self.traces:
            for candidate in trace.association_candidate_tracks:
                if candidate.reject_reason:
                    reasons[candidate.reject_reason] = (
                        reasons.get(candidate.reject_reason, 0) + 1
                    )
        lines = ["======== 关联层审计 ========"]
        lines.append(
            f"参与关联评估的测量 {stats['n_evaluated']} 条，"
            f"平均候选航迹数 {stats['mean_candidate_tracks']:.2f}"
        )
        lines.append(
            f"**关联歧义**（≥2 条候选都过门限）：{stats['n_ambiguous']} 条 "
            f"= {stats['ambiguous_rate']:.4f}"
        )
        if reasons:
            lines.append("门限拒绝分布：")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {reason:<24s} {count:>6d}")
        return "\n".join(lines)

    def format_funnel(self) -> str:
        funnel = self.funnel()
        lines = ["======== 测量生命周期漏斗 ========"]
        lines.append(f"追踪总数 {funnel['total']}"
                     f"（本地 {funnel['local']['total']}，远端 {funnel['remote']['total']}）")
        lines.append("")
        lines.append("按最终去向：")
        for stage, count in sorted(funnel["by_final_stage"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {STAGE_CN.get(stage, stage):<28s} {count:>6d}")
        if funnel["by_reject_reason"]:
            lines.append("")
            lines.append("按拒绝原因：")
            for reason, count in sorted(funnel["by_reject_reason"].items(),
                                        key=lambda kv: -kv[1]):
                lines.append(f"  {reason:<28s} {count:>6d}")
        lines.append("")
        lines.append(f"**远端测量利用率** = "
                     f"{funnel['remote']['in_track']}/{funnel['remote']['total']} "
                     f"= {funnel['remote']['utilization']:.4f}")
        lines.append(f"本地测量利用率   = "
                     f"{funnel['local']['in_track']}/{funnel['local']['total']} "
                     f"= {funnel['local']['utilization']:.4f}")
        lines.append("")
        lines.append("按传感器：")
        for sensor_id, bucket in sorted(funnel["per_sensor"].items()):
            items = "，".join(f"{STAGE_CN.get(k, k)}={v}" for k, v in
                              sorted(bucket.items(), key=lambda kv: -kv[1]))
            lines.append(f"  {sensor_id:<22s} {items}")
        return "\n".join(lines)
