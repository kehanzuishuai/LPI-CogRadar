"""快照的信息边界（v4.5 一致性验收）。

问题
----
`ai/context.py::snapshot_from_simulator` 一开始就是为**离线评测**写的：
它直接读 `sim.targets` / `sim.interceptors` / `sim.jammers`，
于是快照里带着真值目标 ID、真实距离、真实 RCS、真实 Pd/SNR，
还有只有仿真器才知道的东西（哪些检测是虚警、某个"缺失"的真实原因）。
在离线评测里这没问题——评测本来就要用真值算误差。

但同一个函数也被用来给**在线诊断**喂数据。于是"在线 AI"看到的
信息权限远高于它要诊断的那个算法：它可以"知道"有一个目标叫 TGT1
在 6000 m 处，而在线跟踪器连目标存不存在都只能猜。
这不是精度问题，是**信息权限**问题：拿真值解释决策，解释就不再可信。

本模块把两条路径**显式分开**，并给出可执行的信息边界检查：

| 来源 | 允许内容 | 典型用途 |
| --- | --- | --- |
| `online` | 本平台已知自状态、**已经收到**的测量与估计、可达的消息 | 在线诊断、AI 解释 |
| `offline_evaluation` | 上述全部 + 真值（目标/侦察机/干扰机真值、真实 Pd、真实虚警标签、真值缺失原因） | 离线评测、误差统计 |

⚠️ 检查是**递归的、面向序列化结果**的：不只查新增节，
而是把整个 `to_dict()` 结果走一遍——否则新增字段又会偷偷带进真值。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: 信息来源
SOURCE_ONLINE = "online"
SOURCE_OFFLINE = "offline_evaluation"
SOURCES: Tuple[str, ...] = (SOURCE_ONLINE, SOURCE_OFFLINE)

#: 顶层字段里**只有仿真器/评测端才知道**的（真值派生）。
#: 在线快照不得出现这些键（或必须为空 / 显式标为未知）。
TRUTH_DERIVED_TOP_FIELDS: Tuple[str, ...] = (
    "targets",          # 真值目标清单（含真实距离 / RCS / Pd / SNR）
    "interceptors",     # 真值侦察机清单
    "jammers",          # 真值干扰机清单（含干扰机内部威胁度）
)

#: 递归扫描时**禁止出现在在线快照里的键名**（真值通道）
FORBIDDEN_ONLINE_KEYS: Tuple[str, ...] = (
    "truth_id", "truth_range_m", "truth_azimuth_deg", "truth_elevation_deg",
    "truth_range_rate_mps", "is_false_alarm", "n_false_alarms",
    "in_fov", "is_occluded", "matched_truth",
)

#: 在线快照里允许出现、但**必须标明是估计**的量（不能伪装成确定事实）
ESTIMATE_FIELDS: Tuple[str, ...] = (
    "pd_min", "pint_eff", "pint_inst", "exposure",
)

#: 缺失原因的信息来源（与 `sensor.record.REASON_PROVENANCE` 对应）
PROVENANCE_DEVICE = "device_known"          # 设备自己就能确定
PROVENANCE_INFERRED = "measurement_inferred"  # 由测量/几何推断，可能错
PROVENANCE_EVAL_ONLY = "evaluation_only"      # 只有仿真器知道

#: 在线快照里**必须报为"未知"**的字段（而不是编一个确定值）
ONLINE_UNKNOWN_FIELDS: Tuple[str, ...] = (
    "target_count",
    "target_truth_ids",
    "true_missing_reasons",
    "true_false_alarm_labels",
    "hidden_platform_state",
    "future_messages",
)


def _iter_nodes(node: Any) -> Iterable[Any]:
    """深度优先遍历 dict/list 结构（生成每个容器与标量）。"""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def snapshot_information_violations(
    payload: Dict[str, Any], source: str = SOURCE_ONLINE
) -> List[str]:
    """检查（已序列化的）快照是否符合给定来源的信息权限。

    * `source="online"`：真值派生字段必须为空/缺失，禁止出现真值键；
    * `source="offline_evaluation"`：不做限制（评测本来就要用真值），
      但要求显式声明 `information_boundary`，避免被误当成在线输入。

    返回违规描述列表；空列表 = 通过。
    """
    violations: List[str] = []
    if source not in SOURCES:
        return [f"未知来源 {source!r}，只能是 {list(SOURCES)}"]

    declared = payload.get("information_boundary")
    if declared != source:
        violations.append(
            f"information_boundary={declared!r} 与检查来源 {source!r} 不一致"
            "（快照必须自报来源，否则无法判断它能不能被在线使用）"
        )

    if source == SOURCE_ONLINE:
        for field in TRUTH_DERIVED_TOP_FIELDS:
            value = payload.get(field)
            if value:
                violations.append(
                    f"在线快照含真值派生字段 {field}（{len(value)} 项）"
                )
        forbidden = set(FORBIDDEN_ONLINE_KEYS)
        for node in _iter_nodes(payload):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key not in forbidden:
                        continue
                    if value in (None, False, 0, [], {}, ""):
                        continue   # 显式为"空/否"是允许的（表示不存在）
                    violations.append(f"在线快照出现真值键 {key}={value!r}")
        # 缺失原因：evaluation_only 原因不得以确定值出现
        measurement = payload.get("measurement_state") or {}
        for key in ("reason_counts", "reason_rates"):
            if measurement.get(key):
                violations.append(
                    f"在线快照含 {key}：逐原因计数需要真值实体清单才能算，"
                    "只能进离线评测通道"
                )
        estimated = payload.get("online_unknown_fields") or []
        missing = [f for f in ONLINE_UNKNOWN_FIELDS if f not in estimated]
        if missing:
            violations.append(
                f"在线快照未声明未知项 {missing}（不确定的内容必须显式标为未知）"
            )
    return violations


def split_by_boundary(
    snapshot: Any, source: str
) -> Tuple[Any, List[str]]:
    """就地给快照打上来源标记，并返回 (快照, 违规列表)。

    `snapshot` 需是 `StateSnapshot`（或有同名属性/`to_dict` 的对象）。
    """
    if hasattr(snapshot, "information_boundary"):
        snapshot.information_boundary = source
    elif isinstance(snapshot, dict):
        snapshot["information_boundary"] = source
    payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
    return snapshot, snapshot_information_violations(payload, source)


def strip_eval_only(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把序列化结果里**评测专用**的内容清掉，返回新字典。

    用于"离线快照 → 在线视图"的降级：离线快照可以带真值，
    但若要拿它做在线诊断，必须先过这一道，且降级后必须重新做边界检查。
    """
    import copy

    cleaned = copy.deepcopy(payload)
    for field in TRUTH_DERIVED_TOP_FIELDS:
        if field in cleaned:
            cleaned[field] = []
    measurement = cleaned.get("measurement_state")
    if isinstance(measurement, dict):
        for key in ("reason_counts", "reason_rates", "n_false_alarms"):
            measurement.pop(key, None)
    cleaned["information_boundary"] = SOURCE_ONLINE
    # ⚠️ 必须**赋值**而不是 setdefault：离线快照里已经有
    # `online_unknown_fields=[]`，setdefault 会保留那个空列表，
    # 于是降级后仍然缺"未知项声明"（实测就踩了这一步）。
    cleaned["online_unknown_fields"] = list(ONLINE_UNKNOWN_FIELDS)
    return cleaned
