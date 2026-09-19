"""通信消息与链路模型（v4.3）。

数据生成链里通信层的位置
------------------------
```
真实状态 → 可见性 → 测量 → **通信** → 融合 → 决策
```

通信层要把"某平台**现在**测到了什么"变成"另一个平台**在某个时刻之后**
才能看到什么"。这一层最容易出的错是**时间语义**：

* 消息必须同时携带 **生成时刻 / 发送时刻 / 到达时刻 / 过期时刻**，
  只写一个 `time` 是不够的——决策侧需要知道"我拿到的是多久以前的信息"；
* 决策算法**只能读已经到达的消息**。任何"顺手读一下邻居平台的当前真值"
  都会让协同实验失去意义。

因此本模块把这几件事**显式建模**：固定延迟 + 随机抖动、丢包、
带宽限制（字节/秒 + 队列上限）、消息过期。

⚠️ 隔离纪律：`MeasurementMessage.payload` **只允许装测量量**
（距离/方位/俯仰/径向速度/标准差/置信度/候选编号），
**不允许**装真值、误差、虚警标记。`assert_payload_clean()`
会在构造时检查，`validation/` 里还有行为级检验。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: 消息载荷里**禁止出现**的键前缀（真值/误差/虚警标记）
FORBIDDEN_PAYLOAD_PREFIXES: Tuple[str, ...] = ("truth", "err_", "is_false_alarm")

#: 允许出现在载荷里的测量字段（白名单，比黑名单更安全）
ALLOWED_PAYLOAD_FIELDS: Tuple[str, ...] = (
    "sensor_id", "candidate_id", "sensor_kind", "time_s",
    "range_m", "azimuth_deg", "elevation_deg", "range_rate_mps",
    "std_range_m", "std_az_deg", "std_el_deg", "std_range_rate_mps",
    "confidence", "snr_db", "rcs_est_m2", "jam_ratio_est",
    "is_fresh", "age_s",
    # 协方差对角项属于**测量不确定度**，必须共享，否则接收方无法加权融合
    "cov_xx", "cov_yy", "cov_zz",
)


#: 消息类型
MESSAGE_KIND_MEASUREMENT = "measurement"
MESSAGE_KIND_NODE_OBSERVATION = "node_observation"

#: **节点观测摘要**的载荷字段（v4.5：融合结果 → 资源调度适配层）。
#:
#: 为什么需要单独一张白名单：观测摘要带的是**航迹级**信息
#: （稳定的 track_id、位置/速度估计、协方差、最后测量时刻、是否外推、
#: 来源传感器），与测量消息的字段集完全不同。
#: 把它塞进测量白名单会稀释测量的类型边界；
#: 另开一条绕过白名单的通道则会拆掉真值泄漏的最后一道闸门。
#: 因此这里按类型分别给白名单，**两者都强制检查**。
NODE_OBSERVATION_PAYLOAD_FIELDS: Tuple[str, ...] = (
    "node_id", "observed_at_s", "n_tracks",
    "capacity_sample_slot", "capacity_processing_op", "capacity_comm_byte",
    "remaining_sample_slot", "remaining_processing_op", "remaining_comm_byte",
    "n_messages_arrived", "n_messages_inflight", "missing_note",
)

#: 观测摘要里**逐航迹**的字段名（键形如 `track_0_x`）
NODE_OBSERVATION_TRACK_FIELDS: Tuple[str, ...] = (
    "id", "x", "y", "z", "vx", "vy", "vz",
    "sigma_x", "sigma_y", "sigma_z",
    "last_meas_s", "last_fusion_s", "age_s", "coasting", "n_sources",
    "sensors", "platforms", "source_consistency", "source_evidence_count",
    "normalized_residual_mean",
)

#: `track_<序号>_<字段>` 的键形状
_TRACK_KEY_PATTERN = re.compile(r"^track_(\d+)_([a-z_]+)$")


def _allowed_for_kind(kind: str) -> Tuple[str, ...]:
    if kind == MESSAGE_KIND_NODE_OBSERVATION:
        return NODE_OBSERVATION_PAYLOAD_FIELDS
    return ALLOWED_PAYLOAD_FIELDS


class PayloadViolation(ValueError):
    """消息载荷里出现了不允许的字段（例如真值）。"""


def assert_payload_clean(payload: Dict[str, Any],
                         kind: str = MESSAGE_KIND_MEASUREMENT) -> None:
    """检查消息载荷只含**该消息类型**允许的字段。

    用**白名单**而不是黑名单：新增字段时若忘了登记，会直接报错，
    而不是悄悄把真值带过去。这个方向的选择很重要——
    黑名单的失败模式是"漏掉新加的真值字段"，白名单的失败模式是"多报错"，
    显然后者安全。

    白名单**按消息类型分开**（测量 / 节点观测摘要），
    但 `FORBIDDEN_PAYLOAD_PREFIXES` 对两者一律生效。
    """
    allowed = _allowed_for_kind(kind)
    for key in payload:
        text_key = str(key)
        for prefix in FORBIDDEN_PAYLOAD_PREFIXES:
            if text_key.startswith(prefix):
                raise PayloadViolation(
                    f"消息载荷出现禁止字段 {key!r}："
                    "真值/误差/虚警标记不得进入通信链路"
                )
        if text_key in allowed:
            continue
        if kind == MESSAGE_KIND_NODE_OBSERVATION:
            match = _TRACK_KEY_PATTERN.match(text_key)
            if match and match.group(2) in NODE_OBSERVATION_TRACK_FIELDS:
                continue
            raise PayloadViolation(
                f"观测摘要载荷出现未登记字段 {key!r}；允许的固定字段为 "
                f"{list(NODE_OBSERVATION_PAYLOAD_FIELDS)}，"
                f"逐航迹字段键形如 `track_<序号>_<字段>`，"
                f"字段只能是 {list(NODE_OBSERVATION_TRACK_FIELDS)}。"
            )
        raise PayloadViolation(
            f"消息载荷出现未登记字段 {key!r}；"
            f"允许的字段为 {list(allowed)}。"
            "新增字段必须先登记，避免真值字段被悄悄带过去。"
        )


@dataclass
class MeasurementMessage:
    """一条跨平台传输的测量消息。

    `arrived_at` 在链路确定投递时刻后回填；`dropped` 与 `drop_reason`
    记录被丢/被丢弃的原因，供逐消息日志与延迟统计使用。
    """

    msg_id: str
    src_platform_id: str
    src_sensor_id: str
    seq: int
    generated_at: float
    sent_at: float
    payload: Dict[str, Any]
    size_bytes: float = 128.0
    expires_at: float = float("inf")
    #: 目的平台（v4.5 加）：乱序统计与逐链路日志需要它。
    #: 旧代码不传时为 ""，不影响任何既有数值。
    dst_platform_id: str = ""
    #: 消息类型（v4.5）：`measurement`（默认）或 `node_observation`。
    #: 载荷白名单按它选择，因此**不能**用它绕过检查。
    kind: str = MESSAGE_KIND_MEASUREMENT

    # --- 由链路回填 ---
    arrived_at: Optional[float] = None
    dropped: bool = False
    drop_reason: str = ""

    def __post_init__(self) -> None:
        assert_payload_clean(self.payload, self.kind)

    # ------------------------------------------------------------------

    @property
    def in_flight(self) -> bool:
        return self.arrived_at is None and not self.dropped

    def latency_s(self) -> Optional[float]:
        """端到端延迟（含排队）。未到达则为 None。"""
        if self.arrived_at is None:
            return None
        return self.arrived_at - self.generated_at

    def age_at(self, now: float) -> float:
        """在时刻 `now` 这条消息有多旧（按**生成时刻**算）。"""
        return max(0.0, now - self.generated_at)

    def expired_at(self, now: float) -> bool:
        """若在 `now` 才到达是否会过期。

        过期按**到达时刻**判定：晚到的信息即使内容正确，也可能已经不能用了。
        """
        return now > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "kind": self.kind,
            "src_platform_id": self.src_platform_id,
            "dst_platform_id": self.dst_platform_id,
            "src_sensor_id": self.src_sensor_id,
            "seq": self.seq,
            "generated_at": self.generated_at,
            "sent_at": self.sent_at,
            "arrived_at": self.arrived_at,
            "expires_at": self.expires_at,
            "latency_s": self.latency_s(),
            "size_bytes": self.size_bytes,
            "dropped": self.dropped,
            "drop_reason": self.drop_reason,
            **{f"payload_{k}": v for k, v in self.payload.items()},
        }


@dataclass
class LinkConfig:
    """单向链路参数。两个方向可以不同（真实数据链常常不对称）。"""

    link_id: str
    src_platform_id: str
    dst_platform_id: str

    #: 固定传播/处理延迟（秒）
    base_delay_s: float = 0.0
    #: 随机抖动幅度（秒），在 [-jitter, +jitter] 上均匀分布
    jitter_s: float = 0.0
    #: 丢包概率（0~1）
    loss_prob: float = 0.0
    #: 带宽（字节/秒）；None 或 <=0 表示不限
    bandwidth_bytes_per_s: Optional[float] = None
    #: 发送队列上限（字节）；超过则丢弃新消息
    max_queue_bytes: Optional[float] = None
    #: 过期时限（秒，相对生成时刻）；None 表示不过期
    expiry_s: Optional[float] = None
    available: bool = True
    seed: int = 0

    # --- v4.5 通信时序压力（默认全零，旧行为逐位不变）---
    #: **突发丢包**：每条消息独立判定"是否开启一段突发"
    burst_loss_prob: float = 0.0
    #: 突发长度（连续丢多少条）。实际长度在该值上下抖动（见 transmit）
    burst_length: int = 0
    #: **链路中断窗口** `[(start_s, end_s), ...]`（仿真绝对时刻，秒）。
    #: 窗口内发送的消息一律丢弃，原因 `link_outage`。
    #: 判定依据是**发送时刻**（不是到达时刻），语义单纯、便于复现。
    outage_windows: Tuple[Tuple[float, float], ...] = ()
    #: 中断恢复后的**拥塞窗口**（秒）：窗口内额外丢包 + 额外延迟，线性衰减
    recovery_congestion_s: float = 0.0
    #: 拥塞窗口内的额外延迟峰值（秒）
    recovery_extra_delay_s: float = 0.0
    #: 拥塞窗口内的额外丢包概率峰值
    recovery_loss_prob: float = 0.0
    #: **乱序概率**：命中时该消息额外延迟 `[0.5, 1.5]×reorder_extra_delay_s`，
    #: 从而晚于后发消息到达（制造 out-of-sequence measurement）
    reorder_prob: float = 0.0
    reorder_extra_delay_s: float = 2.0

    def validate(self) -> None:
        if self.base_delay_s < 0:
            raise ValueError(f"[{self.link_id}] base_delay_s 不能为负")
        if self.jitter_s < 0:
            raise ValueError(f"[{self.link_id}] jitter_s 不能为负")
        if not 0.0 <= self.loss_prob <= 1.0:
            raise ValueError(f"[{self.link_id}] loss_prob 必须落在 [0, 1]")
        if self.bandwidth_bytes_per_s is not None and self.bandwidth_bytes_per_s <= 0:
            raise ValueError(f"[{self.link_id}] bandwidth_bytes_per_s 必须为正或 None")
        if self.max_queue_bytes is not None and self.max_queue_bytes <= 0:
            raise ValueError(f"[{self.link_id}] max_queue_bytes 必须为正或 None")
        if self.expiry_s is not None and self.expiry_s <= 0:
            raise ValueError(f"[{self.link_id}] expiry_s 必须为正或 None")
        if not 0.0 <= self.burst_loss_prob <= 1.0:
            raise ValueError(f"[{self.link_id}] burst_loss_prob 必须落在 [0, 1]")
        if self.burst_length < 0:
            raise ValueError(f"[{self.link_id}] burst_length 不能为负")
        if not 0.0 <= self.reorder_prob <= 1.0:
            raise ValueError(f"[{self.link_id}] reorder_prob 必须落在 [0, 1]")
        if self.reorder_extra_delay_s <= 0:
            raise ValueError(f"[{self.link_id}] reorder_extra_delay_s 必须为正")
        if not 0.0 <= self.recovery_loss_prob <= 1.0:
            raise ValueError(f"[{self.link_id}] recovery_loss_prob 必须落在 [0, 1]")
        for window in self.outage_windows:
            if len(window) != 2 or window[1] < window[0]:
                raise ValueError(
                    f"[{self.link_id}] outage_windows 元素必须是 (start, end) 且 end≥start"
                )


class CommLink:
    """一条单向链路：决定每条消息的投递时刻，或把它丢掉。"""

    def __init__(self, config: LinkConfig) -> None:
        config.validate()
        self.config = config
        self._rng = random.Random("%d:%s" % (config.seed, config.link_id))
        #: 队列里待发字节数（用于带宽限制）
        self._queued_bytes = 0.0
        self._last_drain_time: Optional[float] = None
        #: 突发丢包剩余计数（>0 表示正处于一段突发里）
        self._burst_remaining: int = 0
        #: 统计
        self.stats: Dict[str, Any] = {
            "sent": 0, "delivered": 0, "lost": 0, "dropped_queue_full": 0,
            "expired": 0, "link_down": 0, "queued_bytes_peak": 0.0,
            # --- v4.5 通信时序压力 ---
            "burst_lost": 0, "outage_lost": 0, "congestion_lost": 0,
            "reordered": 0,
        }

    def reset(self) -> None:
        self._rng = random.Random("%d:%s" % (self.config.seed, self.config.link_id))
        self._queued_bytes = 0.0
        self._last_drain_time = None
        self._burst_remaining = 0
        for key in self.stats:
            self.stats[key] = 0
        self.stats["queued_bytes_peak"] = 0.0

    # ------------------------------------------------------------------

    def in_outage(self, now: float) -> bool:
        """`now`（发送时刻）是否落在某个中断窗口内。"""
        for start, end in self.config.outage_windows:
            if start <= now <= end:
                return True
        return False

    def congestion_factor(self, now: float) -> float:
        """中断恢复后的拥塞衰减因子（1 = 刚恢复，0 = 拥塞已清空）。

        用于让"恢复瞬间"最拥塞、随后线性缓解——这是真实数据链的行为，
        也是"链路恢复后历史旧包是否会重新污染航迹"这一问题的关键条件。
        """
        cfg = self.config
        if cfg.recovery_congestion_s <= 0.0:
            return 0.0
        factor = 0.0
        for _start, end in cfg.outage_windows:
            elapsed = now - end
            if 0.0 <= elapsed < cfg.recovery_congestion_s:
                factor = max(
                    factor, 1.0 - elapsed / cfg.recovery_congestion_s
                )
        return factor

    # ------------------------------------------------------------------

    def _drain(self, now: float) -> None:
        """按带宽把队列里的字节"发出去"。"""
        cfg = self.config
        if self._last_drain_time is None:
            self._last_drain_time = now
            return
        elapsed = max(0.0, now - self._last_drain_time)
        self._last_drain_time = now
        if cfg.bandwidth_bytes_per_s is None or elapsed <= 0.0:
            return
        self._queued_bytes = max(
            0.0, self._queued_bytes - cfg.bandwidth_bytes_per_s * elapsed
        )

    def transmit(self, message: MeasurementMessage, now: float) -> MeasurementMessage:
        """尝试发送一条消息；回填 `arrived_at` 或标记丢弃。

        丢弃判定的**顺序**是有意义的（先发生的物理原因先判）：
        链路不可用 → **中断窗口** → 突发丢包 → 独立丢包 →
        **恢复后拥塞** → 带宽/队列 → 过期。
        """
        cfg = self.config
        self._drain(now)

        if not cfg.available:
            message.dropped = True
            message.drop_reason = "link_down"
            self.stats["link_down"] += 1
            return message

        # --- 链路中断窗口（v4.5）---
        if self.in_outage(now):
            message.dropped = True
            message.drop_reason = "link_outage"
            self.stats["outage_lost"] += 1
            return message

        self.stats["sent"] += 1

        # --- 突发丢包（v4.5）：一段连续丢，而不是逐条独立丢 ---
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            message.dropped = True
            message.drop_reason = "burst_loss"
            self.stats["burst_lost"] += 1
            return message
        if cfg.burst_loss_prob > 0.0 and self._rng.random() < cfg.burst_loss_prob:
            # 突发长度按 burst_length 上下抖动：真实突发很少是固定整数
            base = max(1, int(cfg.burst_length or 1))
            self._burst_remaining = max(
                1, int(round(base * self._rng.uniform(0.5, 1.5)))
            ) - 1
            message.dropped = True
            message.drop_reason = "burst_loss"
            self.stats["burst_lost"] += 1
            return message

        # --- 丢包 ---
        if cfg.loss_prob > 0.0 and self._rng.random() < cfg.loss_prob:
            message.dropped = True
            message.drop_reason = "lost"
            self.stats["lost"] += 1
            return message

        # --- 中断恢复后的拥塞（v4.5）：额外丢包 + 额外延迟，随时间衰减 ---
        congestion = self.congestion_factor(now)
        if congestion > 0.0 and cfg.recovery_loss_prob > 0.0:
            if self._rng.random() < cfg.recovery_loss_prob * congestion:
                message.dropped = True
                message.drop_reason = "congestion"
                self.stats["congestion_lost"] += 1
                return message

        # --- 带宽 / 队列 ---
        if cfg.bandwidth_bytes_per_s is not None:
            self._queued_bytes += message.size_bytes
            self.stats["queued_bytes_peak"] = max(
                self.stats["queued_bytes_peak"], self._queued_bytes
            )
            # 队列里积压超过上限 -> 丢弃新消息（真实链路常见行为）
            if cfg.max_queue_bytes is not None and self._queued_bytes > cfg.max_queue_bytes:
                self._queued_bytes = max(0.0, self._queued_bytes - message.size_bytes)
                message.dropped = True
                message.drop_reason = "queue_full"
                self.stats["dropped_queue_full"] += 1
                return message
            # 排队时间 = 当前积压 / 带宽
            queue_delay = self._queued_bytes / cfg.bandwidth_bytes_per_s
        else:
            queue_delay = 0.0

        # --- 延迟（固定 + 抖动 + 拥塞 + 乱序）---
        jitter = 0.0
        if cfg.jitter_s > 0.0:
            jitter = self._rng.uniform(-cfg.jitter_s, cfg.jitter_s)
        congestion_delay = cfg.recovery_extra_delay_s * congestion
        reorder_delay = 0.0
        if cfg.reorder_prob > 0.0 and self._rng.random() < cfg.reorder_prob:
            reorder_delay = cfg.reorder_extra_delay_s * self._rng.uniform(0.5, 1.5)
            self.stats["reordered"] += 1
        arrival = now + max(
            0.0, cfg.base_delay_s + jitter + queue_delay + congestion_delay
            + reorder_delay
        )
        message.arrived_at = arrival

        # --- 过期（按到达时刻判定）---
        if cfg.expiry_s is not None:
            message.expires_at = message.generated_at + cfg.expiry_s
            if arrival > message.expires_at:
                message.dropped = True
                message.drop_reason = "expired"
                message.arrived_at = None
                self.stats["expired"] += 1
                return message

        self.stats["delivered"] += 1
        return message
