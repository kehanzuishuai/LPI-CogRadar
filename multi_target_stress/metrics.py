"""多目标指标（v4.5 P2）——**唯一允许读真值的地方**。

为什么单目标的指标不能直接用
----------------------------
`evaluate_cooperative_sensing.py` 的指标对每个航迹**各自**找最近的
真值目标。单目标时没问题，多目标时会错：

* 两条航迹可以同时"匹配"到**同一个**真值目标 → 重复航迹被算成两次正确；
* "误关联率"只在同一条 track_id 的最近真值**变了**时才计数，
  丢掉目标再以新 ID 重新捕获（碎裂）根本不会被记到；
* 目标交叉时两者最近真值互换，但那不代表关联真的错了。

所以这里改成**每帧做一次一对一分配**（离线、贪心、按距离升序），
再在分配结果上算指标。分配只用位置距离，与跟踪器内部用的马氏距离
**不是同一套判据**——这一点是有意的：评测侧若复用跟踪器的门限，
就只能证明"跟踪器自洽"，不能证明"它跟真值对得上"。

⚠️ 评测门限（`ASSOC_GATE_M` / `DUPLICATE_GATE_M`）是**离线参数**，
不回流入任何算法。改它们只改"怎么算分"，不改"算法怎么做"。

口径说明（诚实标注）
--------------------
* `missed_track_rate` 是**合并口径**（所有帧所有目标一起算），
  `track_completeness` 是**逐目标平均**（macro）。两者在覆盖均匀时相等，
  不均匀时会分开——这是有意的，不是重复指标。
* `false_track_rate` 用 1000 m 门限：真实目标误差超过 1 km 的航迹会被
  计成假航迹。因此报告同时给出 `position_error_p95_m`，
  便于区分"真有假航迹"与"误差过大被误判"。
* 一对一分配用**贪心**（按距离升序），不是匈牙利最优解。
  目标数 ≤3 时贪心与最优解在本工程场景下一致，但这是**未证明**的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 离线关联门限（米）：航迹与真值目标距离在此以内才可能被分配
ASSOC_GATE_M = 1000.0
#: 重复航迹门限（米）：同一目标此范围内出现多条航迹即计重复
DUPLICATE_GATE_M = 1000.0


def assign_tracks_to_truth(
    track_items: Sequence[Tuple[str, Any]],
    truth_items: Sequence[Tuple[str, Any]],
    gate_m: float = ASSOC_GATE_M,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """离线**一对一**分配：track_id → truth_id 与 truth_id → track_id。

    `track_items` / `truth_items` 为 `(id, position)` 序列，position 需支持
    减法与 `norm()`（即 `engine.geometry.Vec3`）。
    贪心按距离升序，保证一对一（每条航迹、每个目标最多分配一次）。
    """
    pairs: List[Tuple[float, str, str]] = []
    for track_id, position in track_items:
        for truth_id, truth_position in truth_items:
            distance = (position - truth_position).norm()
            if distance <= gate_m:
                pairs.append((distance, track_id, truth_id))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    track_to_truth: Dict[str, str] = {}
    truth_to_track: Dict[str, str] = {}
    for _distance, track_id, truth_id in pairs:
        if track_id in track_to_truth or truth_id in truth_to_track:
            continue
        track_to_truth[track_id] = truth_id
        truth_to_track[truth_id] = track_id
    return track_to_truth, truth_to_track


@dataclass
class MultiTargetMetrics:
    """跨帧累积的多目标指标。"""

    gate_m: float = ASSOC_GATE_M
    duplicate_gate_m: float = DUPLICATE_GATE_M

    # --- 累积量 ---
    n_frames: int = 0
    _pos_sq: List[float] = field(default_factory=list)
    _vel_sq: List[float] = field(default_factory=list)
    _pos_err: List[float] = field(default_factory=list)

    n_track_frames: int = 0
    n_unassigned_track_frames: int = 0
    n_false_track_frames: int = 0
    duplicate_track_count: int = 0
    _duplicate_track_ids: set = field(default_factory=set)

    active_target_frames: int = 0
    missed_target_frames: int = 0

    id_switch_count: int = 0
    track_fragmentation_count: int = 0
    transitions_total: int = 0
    transitions_same: int = 0

    assoc_total: int = 0
    assoc_correct: int = 0
    fa_associated_to_real_track: int = 0
    fa_total: int = 0

    n_tracks_seen: int = 0
    _track_truth_history: Dict[str, List[str]] = field(default_factory=dict)
    _truth_history: Dict[str, List[Optional[str]]] = field(default_factory=dict)
    _track_purity: Dict[str, float] = field(default_factory=dict)

    # --- 内部状态 ---
    _prev_assign: Dict[str, Optional[str]] = field(default_factory=dict)
    _last_track: Dict[str, str] = field(default_factory=dict)
    _in_gap: set = field(default_factory=set)

    # ------------------------------------------------------------------

    def add_frame(self, frame: Dict[str, Any]) -> None:
        """处理一帧。

        `frame` 需含：
        * `tracks`: [(track_id, position, velocity, status), ...]
        * `truth`:  [(target_id, position, velocity, is_active), ...]
        * `associations`: [(sensor_id, candidate_id, chosen_track_id, is_remote), ...]
        * `measurement_truth`: {(sensor_id, candidate_id): truth_id_or_None}
        """
        self.n_frames += 1
        tracks = frame.get("tracks") or []
        truth_all = frame.get("truth") or []
        active = [t for t in truth_all if t[3]]
        measurement_truth: Dict[Tuple[str, str], Optional[str]] = (
            frame.get("measurement_truth") or {}
        )

        track_items = [(t[0], t[1]) for t in tracks]
        truth_items = [(t[0], t[1]) for t in active]
        track_to_truth, truth_to_track = assign_tracks_to_truth(
            track_items, truth_items, self.gate_m
        )

        # --- 精度（只在被分配的 (航迹, 目标) 对上算）---
        by_id = {t[0]: t for t in tracks}
        for track_id, truth_id in track_to_truth.items():
            track = by_id[track_id]
            target = next(t for t in active if t[0] == truth_id)
            error = (track[1] - target[1]).norm()
            self._pos_sq.append(error ** 2)
            self._pos_err.append(error)
            self._vel_sq.append((track[2] - target[2]).norm() ** 2)

        # --- 假航迹 / 重复航迹 ---
        #
        # ⚠️ 这里**必须**用一对一分配的结果来分，不能数"目标附近有几条航迹"：
        # 交叉场景里两个目标本身就互相靠近，那样会把"另一个目标的航迹"
        # 全部误判成重复航迹（第一版就是这样，34 次重复全是假阳性）。
        # 正确口径：未被分配到的多余航迹，再看它身边有没有真值目标——
        #   身边有目标（该目标已被别的航迹占用）→ **重复航迹**
        #   身边没有任何目标                     → **假航迹**
        self.n_track_frames += len(tracks)
        self.n_tracks_seen = max(self.n_tracks_seen, len(tracks))
        for track_id, position, _velocity, _status in tracks:
            if track_id in track_to_truth:
                self._track_truth_history.setdefault(track_id, []).append(
                    track_to_truth[track_id]
                )
                continue
            self.n_unassigned_track_frames += 1
            nearest = (min((position - truth_position).norm()
                           for _truth_id, truth_position in truth_items)
                       if truth_items else float("inf"))
            if nearest <= self.duplicate_gate_m:
                self.duplicate_track_count += 1
                self._duplicate_track_ids.add(track_id)
            else:
                self.n_false_track_frames += 1

        # --- 逐目标的连续性 / 换号 / 碎裂 ---
        for target_id, _position, _velocity, _active in active:
            current = truth_to_track.get(target_id)
            previous = self._prev_assign.get(target_id)
            if current is not None:
                if target_id in self._in_gap:
                    before = self._last_track.get(target_id)
                    if before and before != current:
                        self.track_fragmentation_count += 1
                    self._in_gap.discard(target_id)
                elif previous is not None and previous != current:
                    self.id_switch_count += 1
                if previous is not None:
                    self.transitions_total += 1
                    if previous == current:
                        self.transitions_same += 1
                self._prev_assign[target_id] = current
                self._last_track[target_id] = current
            else:
                if previous is not None:
                    self._in_gap.add(target_id)
                self._prev_assign[target_id] = None
                self.missed_target_frames += 1
            self.active_target_frames += 1
            self._truth_history.setdefault(target_id, []).append(current)

        # --- 关联决策准确率（用测量自己的真值标签，仅评测侧可见）---
        for sensor_id, candidate_id, chosen, _is_remote in frame.get("associations") or []:
            truth_id = measurement_truth.get((sensor_id, candidate_id))
            if truth_id is None:
                self.fa_total += 1
            if not chosen:
                continue
            self.assoc_total += 1
            if truth_id is not None and track_to_truth.get(chosen) == truth_id:
                self.assoc_correct += 1
            elif truth_id is None:
                # 虚警被关联进了一条真实目标的航迹 —— "虚警夺取真实航迹"
                if chosen in track_to_truth:
                    self.fa_associated_to_real_track += 1

    # ------------------------------------------------------------------

    def _purity(self) -> float:
        values: List[float] = []
        for _track_id, history in self._track_truth_history.items():
            if not history:
                continue
            counts: Dict[str, int] = {}
            for truth_id in history:
                counts[truth_id] = counts.get(truth_id, 0) + 1
            values.append(max(counts.values()) / len(history))
        return sum(values) / len(values) if values else 0.0

    def _completeness(self) -> float:
        """逐目标覆盖率（macro）。"""
        values: List[float] = []
        for _target_id, history in self._truth_history.items():
            if not history:
                continue
            covered = sum(1 for item in history if item is not None)
            values.append(covered / len(history))
        return sum(values) / len(values) if values else 0.0

    def result(self) -> Dict[str, Any]:
        def _mean_sq(values: Sequence[float]) -> float:
            return math.sqrt(sum(values) / len(values)) if values else 0.0

        sorted_err = sorted(self._pos_err)
        p95 = (sorted_err[min(len(sorted_err) - 1, int(0.95 * len(sorted_err)))]
               if sorted_err else 0.0)
        return {
            "n_frames": self.n_frames,
            # --- 用户要求的十项 ---
            "id_switch_count": self.id_switch_count,
            "track_fragmentation_count": self.track_fragmentation_count,
            "false_track_rate": (self.n_false_track_frames /
                                 self.n_track_frames) if self.n_track_frames else 0.0,
            "missed_track_rate": (self.missed_target_frames /
                                  self.active_target_frames)
                                 if self.active_target_frames else 0.0,
            "association_accuracy": (self.assoc_correct / self.assoc_total)
                                    if self.assoc_total else 0.0,
            "track_purity": self._purity(),
            "track_completeness": self._completeness(),
            "position_rmse_m": _mean_sq(self._pos_sq),
            "velocity_rmse_mps": _mean_sq(self._vel_sq),
            "continuity_rate": (self.transitions_same / self.transitions_total)
                               if self.transitions_total else 0.0,
            "duplicate_track_count": self.duplicate_track_count,
            "n_duplicate_track_ids": len(self._duplicate_track_ids),
            "n_unassigned_track_frames": self.n_unassigned_track_frames,
            "n_false_track_frames": self.n_false_track_frames,
            # --- 便于解释的辅助量 ---
            "position_error_p95_m": p95,
            "n_track_frames": self.n_track_frames,
            "n_active_target_frames": self.active_target_frames,
            "n_missed_target_frames": self.missed_target_frames,
            "n_associations": self.assoc_total,
            "n_associations_correct": self.assoc_correct,
            "n_false_alarms_consumed": self.fa_total,
            "n_false_alarms_into_real_track": self.fa_associated_to_real_track,
            "n_transitions": self.transitions_total,
            "n_transitions_same": self.transitions_same,
            "max_tracks_in_a_frame": self.n_tracks_seen,
        }


#: 报告里要展示的指标顺序（用户要求的十项在前）
METRIC_ORDER: Tuple[Tuple[str, str, int], ...] = (
    ("id_switch_count", "ID换号次数", 0),
    ("track_fragmentation_count", "航迹碎裂次数", 0),
    ("duplicate_track_count", "重复航迹计数", 0),
    ("false_track_rate", "假航迹率", 4),
    ("missed_track_rate", "漏跟率", 4),
    ("association_accuracy", "关联准确率", 4),
    ("track_purity", "航迹纯度", 4),
    ("track_completeness", "航迹完整度(macro)", 4),
    ("continuity_rate", "连续性", 4),
    ("position_rmse_m", "位置RMSE(m)", 2),
    ("velocity_rmse_mps", "速度RMSE(m/s)", 2),
    ("position_error_p95_m", "位置误差P95(m)", 2),
    ("n_false_alarms_into_real_track", "虚警进入真实航迹", 0),
    ("n_associations", "关联决策总数", 0),
    ("max_tracks_in_a_frame", "单帧最多航迹数", 0),
)


def format_metrics_row(label: str, metrics: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, title, digits in METRIC_ORDER[:11]:
        value = metrics.get(key, 0.0)
        parts.append(f"{title}={value:.{digits}f}")
    return f"{label}: " + "  ".join(parts)
