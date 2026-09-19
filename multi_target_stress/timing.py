"""乱序测量（OOSM）处理策略（v4.5 S7：通信时序压力）。

问题
----
现有链路已经能丢包、能延迟、能过期，但**到达顺序**始终等于**发送顺序**。
真实数据链不是这样：多跳转发、重传、不同链路时延差，都会让"后发的先到"。
一旦发生，跟踪器就会**先用新信息、再用旧信息**更新同一条航迹——
旧信息把状态往回拽，表现为航迹抖动、创新异常、甚至误关联。

四种时间戳必须显式区分（本模块与 `LifecycleLog` 一起保证）：

| 时间 | 含义 | 谁能看到 |
| --- | --- | --- |
| measurement time | 目标被**测量**的时刻（`record.time_s`） | 算法可见 |
| send time | 发送方**发出**消息的时刻（`MeasurementMessage.sent_at`） | 决策侧可见 |
| arrival time | 消息**到达**决策侧的时刻（`arrived_at`） | 决策侧可见 |
| fusion time | 该测量**被用于更新航迹**的时刻（`MeasurementTrace.consumed_at`） | 决策侧可见 |

策略
----
* `drop_stale`（**对照**，也是 v4.3–v4.5 的既有行为）：
  来什么就立刻喂什么，谁过时由跟踪器自己的 `max_measurement_age_s` 拒绝。
  乱序时旧测量会**照常**参与更新（只要没超时效）。
* `reorder_buffer`（本版新增）：
  维护一个**受限延迟**的重排缓冲。到达顺序里"比已经融合过的最新测量还旧"
  的测量先扣住不放，等它老到 `window_s`（或后续没有更早的了）再按
  `measurement time` 升序释放。**代价是这些测量被延迟 `window_s`**，
  这是真实代价，必须在报告里一起给出。
* `delayed_update`（本版新增，**简化实现**）：
  与 `reorder_buffer` 同为缓冲，但释放时**按扣留时长放大该测量的协方差**
  再交给跟踪器——直觉是"这条信息是相对于过去的，它对当前状态的约束力更弱"。
  ⚠️ 这是**简化的回溯处理**，不是严格的 OOSM 滤波器
  （严格做法要做状态回溯/重算，见 Bar-Shalom 的 OOSM 系列工作）。
  本工程**不声称**它等价于任何标准 OOSM 算法。

⚠️ 本模块只决定"哪些测量、什么时刻、以什么不确定度进入融合"，
**不读真值**，也不修改跟踪器内部的关联与滤波数学（基线保持不动）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 不重排，直接喂（对照）
DROP_STALE = "drop_stale"
#: 受限延迟重排缓冲（新增）
REORDER_BUFFER = "reorder_buffer"
#: 重排 + 协方差放大的延迟更新（新增，简化版）
DELAYED_UPDATE = "delayed_update"

OOSM_POLICIES: Tuple[str, ...] = (DROP_STALE, REORDER_BUFFER, DELAYED_UPDATE)

OOSM_POLICY_CN: Dict[str, str] = {
    DROP_STALE: "不重排（旧包照常更新，只靠时效门限拦）",
    REORDER_BUFFER: "受限延迟重排缓冲（按测量时刻升序释放）",
    DELAYED_UPDATE: "重排 + 协方差放大的延迟更新（简化回溯）",
}

#: 决策标签
DECISION_IMMEDIATE = "immediate"
DECISION_HELD = "held"
DECISION_RELEASED_LATE = "released_late"
DECISION_IN_ORDER_RELEASE = "released_in_order"


def _measurement_time(measurement: Any, default: float) -> float:
    value = getattr(measurement, "time_s", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _inflate_covariance(measurement: Any, factor: float) -> Any:
    """返回一个**浅拷贝**，其协方差/标准差按 `factor` 放大（≥1）。

    `drop_stale` 与 `reorder_buffer` 都不调用它；
    只有 `delayed_update` 在释放迟到测量时调用。
    `factor` 由扣留时长决定：扣得越久，越不该主导融合结果。
    """
    if factor <= 1.0:
        return measurement
    clone = copy.copy(measurement)
    for attr in ("std_range_m", "std_az_deg", "std_el_deg", "std_range_rate_mps"):
        value = getattr(clone, attr, None)
        if isinstance(value, (int, float)):
            setattr(clone, attr, float(value) * factor)
    covariance = getattr(clone, "covariance", None)
    if isinstance(covariance, list):
        scaled: List[Any] = []
        for row_index, row in enumerate(covariance):
            if not isinstance(row, list):
                scaled.append(row)
                continue
            new_row = []
            for col_index, value in enumerate(row):
                if row_index == col_index and isinstance(value, (int, float)):
                    new_row.append(float(value) * factor * factor)
                else:
                    new_row.append(value)
            scaled.append(new_row)
        clone.covariance = scaled
    return clone


@dataclass
class OosmDecision:
    """一条测量的时序决策记录（可导出、可审计）。"""

    candidate_id: str
    sensor_id: str
    is_remote: bool
    measurement_time_s: float
    arrival_time_s: float
    fusion_time_s: Optional[float]
    decision: str
    hold_s: float = 0.0
    covariance_inflation: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sensor_id": self.sensor_id,
            "is_remote": self.is_remote,
            "measurement_time_s": self.measurement_time_s,
            "arrival_time_s": self.arrival_time_s,
            "fusion_time_s": self.fusion_time_s,
            "decision": self.decision,
            "hold_s": round(self.hold_s, 6),
            "covariance_inflation": round(self.covariance_inflation, 6),
            "reason": self.reason,
        }


class OosmController:
    """把一个"到达批次"整理成"本步该融合的测量 + 决策记录"。"""

    def __init__(self, policy: str = DROP_STALE, window_s: float = 2.0,
                 inflation_per_s: float = 0.5) -> None:
        if policy not in OOSM_POLICIES:
            raise ValueError(
                f"未知 OOSM 策略 {policy!r}，只支持 {list(OOSM_POLICIES)}"
            )
        if window_s < 0.0:
            raise ValueError("window_s 不能为负")
        self.policy = policy
        self.window_s = float(window_s)
        self.inflation_per_s = float(inflation_per_s)
        self._buffer: List[Dict[str, Any]] = []
        self._newest_fused_time: Optional[float] = None
        self.decisions: List[OosmDecision] = []
        self.stats: Dict[str, int] = {
            "immediate": 0, "held": 0, "released_late": 0,
            "released_in_order": 0, "out_of_order_seen": 0,
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._newest_fused_time = None
        self.decisions.clear()
        for key in self.stats:
            self.stats[key] = 0

    # ------------------------------------------------------------------

    def select(
        self,
        measurements: Sequence[Any],
        flags: Sequence[bool],
        now: float,
    ) -> Tuple[List[Any], List[bool], List[OosmDecision]]:
        """返回本步应融合的 (测量, 是否远端) 与决策记录。

        `flags[i]` 与 `measurements[i]` 一一对应（True = 经通信到达）。
        """
        records: List[OosmDecision] = []
        if self.policy == DROP_STALE:
            for measurement, is_remote in zip(measurements, flags):
                m_time = _measurement_time(measurement, now)
                decision = OosmDecision(
                    candidate_id=str(getattr(measurement, "candidate_id", "")),
                    sensor_id=str(getattr(measurement, "sensor_id", "")),
                    is_remote=bool(is_remote),
                    measurement_time_s=m_time, arrival_time_s=now,
                    fusion_time_s=now, decision=DECISION_IMMEDIATE,
                    reason="不重排策略：立即喂入，时效由跟踪器门限负责",
                )
                records.append(decision)
                self.stats["immediate"] += 1
            self.decisions.extend(records)
            return list(measurements), list(flags), records

        # --- 重排 / 延迟更新：先入缓冲，再按测量时刻升序释放 ---
        for measurement, is_remote in zip(measurements, flags):
            m_time = _measurement_time(measurement, now)
            out_of_order = (
                self._newest_fused_time is not None
                and m_time < self._newest_fused_time - 1e-12
            )
            if out_of_order:
                self.stats["out_of_order_seen"] += 1
            self._buffer.append({
                "measurement": measurement,
                "is_remote": bool(is_remote),
                "measurement_time_s": m_time,
                "arrival_time_s": now,
                "out_of_order": out_of_order,
            })

        release: List[Dict[str, Any]] = []
        keep: List[Dict[str, Any]] = []
        for item in self._buffer:
            held_for = now - item["arrival_time_s"]
            # 释放条件：扣满窗口，或它已经不再"比最新融合时刻旧"
            ready = (
                held_for >= self.window_s
                or self._newest_fused_time is None
                or item["measurement_time_s"] >= self._newest_fused_time - 1e-12
            )
            (release if ready else keep).append(item)
        self._buffer = keep
        release.sort(key=lambda item: (item["measurement_time_s"],
                                       item["arrival_time_s"]))

        out_measurements: List[Any] = []
        out_flags: List[bool] = []
        for item in release:
            m_time = item["measurement_time_s"]
            held_for = max(0.0, now - item["arrival_time_s"])
            late = (
                self._newest_fused_time is not None
                and m_time < self._newest_fused_time - 1e-12
            )
            inflation = 1.0
            measurement = item["measurement"]
            if self.policy == DELAYED_UPDATE and (late or held_for > 0.0):
                # 简化回溯：扣得越久，越放大不确定度，避免旧包主导状态
                inflation = 1.0 + self.inflation_per_s * held_for
                measurement = _inflate_covariance(measurement, inflation)
            decision_name = DECISION_RELEASED_LATE if late else DECISION_IN_ORDER_RELEASE
            records.append(OosmDecision(
                candidate_id=str(getattr(measurement, "candidate_id", "")),
                sensor_id=str(getattr(measurement, "sensor_id", "")),
                is_remote=bool(item["is_remote"]),
                measurement_time_s=m_time, arrival_time_s=item["arrival_time_s"],
                fusion_time_s=now, decision=decision_name,
                hold_s=held_for, covariance_inflation=inflation,
                reason=("乱序测量按测量时刻升序补入（已被扣留 %.3fs）" % held_for)
                       if late else "按序释放",
            ))
            self.stats[decision_name] += 1
            out_measurements.append(measurement)
            out_flags.append(bool(item["is_remote"]))
            self._newest_fused_time = max(
                m_time,
                self._newest_fused_time if self._newest_fused_time is not None else m_time,
            )
        self.stats["held"] = len(self._buffer)
        self.decisions.extend(records)
        return out_measurements, out_flags, records

    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def finalize(self) -> List[OosmDecision]:
        """运行结束时把仍扣在缓冲里的测量标成 `held` 并落一条记录。

        为什么需要它：被扣住、且整轮都没等到释放窗口的测量，
        如果没有任何记录，"有多少信息卡在缓冲里"这件事就消失了。
        """
        records: List[OosmDecision] = []
        for item in self._buffer:
            records.append(OosmDecision(
                candidate_id=str(getattr(item["measurement"], "candidate_id", "")),
                sensor_id=str(getattr(item["measurement"], "sensor_id", "")),
                is_remote=bool(item["is_remote"]),
                measurement_time_s=item["measurement_time_s"],
                arrival_time_s=item["arrival_time_s"],
                fusion_time_s=None,
                decision=DECISION_HELD,
                hold_s=0.0,
                reason="运行结束时仍扣在重排缓冲里，从未被融合",
            ))
        self.decisions.extend(records)
        self._buffer.clear()
        self.stats["held"] = 0
        return records

    def summary(self) -> Dict[str, Any]:
        fused = self.stats["immediate"] + self.stats["released_late"] \
            + self.stats["released_in_order"]
        holds = [d.hold_s for d in self.decisions
                 if d.decision == DECISION_RELEASED_LATE]
        return {
            "policy": self.policy,
            "policy_cn": OOSM_POLICY_CN.get(self.policy, self.policy),
            "n_fused": fused,
            "n_out_of_order_seen": self.stats["out_of_order_seen"],
            "n_released_late": self.stats["released_late"],
            "n_pending_at_end": len(self._buffer),
            "mean_hold_s": (sum(holds) / len(holds)) if holds else 0.0,
            "max_hold_s": max(holds) if holds else 0.0,
            "mean_covariance_inflation": (
                sum(d.covariance_inflation for d in self.decisions
                    if d.decision == DECISION_RELEASED_LATE) / len(holds)
                if holds else 1.0
            ),
        }
