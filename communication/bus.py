"""通信总线与共享策略（v4.3）。

三种共享策略（对应本轮要求的三组实验）
--------------------------------------
| 策略 | 含义 | 实现 |
| --- | --- | --- |
| `no_share` | 不共享 | 没有任何链路；每个平台只用**自己**的局部测量 |
| `ideal_share` | 理想零延迟共享 | 全连接、0 延迟、0 丢包、不限带宽、不过期 |
| `constrained_share` | 有延迟丢包共享 | 按配置的固定延迟 + 抖动 + 丢包 + 带宽 + 过期 |

**理想共享不是"更好的通信"，而是一条上界参考线**：
它回答"如果通信完全不是瓶颈，协同能带来多少"。
实际能拿到多少取决于 `constrained_share` 的参数，
因此两者必须成对汇报，不能只报理想值。

时间语义（本模块最关键的纪律）
------------------------------
决策算法**只能读已经到达的消息**：

    visible = bus.arrived(now)        # 只含 arrived_at <= now
    pending = bus.in_flight(now)      # 还在路上的（**不得**读取内容）

`arrived()` 的实现里有一条硬断言：任何 `arrived_at > now` 的消息出现即抛错。
这不是注释，是可执行的边界——`tests/test_communication.py` 会构造
"发送于 t、到达于 t+3"的消息，验证在 t+2 时刻它**不可见**。

带宽与过期都在 `CommLink.transmit` 里处理（见 `communication/message.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from communication.message import CommLink, LinkConfig, MeasurementMessage

#: 共享策略名
SHARE_NONE = "no_share"
SHARE_IDEAL = "ideal_share"
SHARE_CONSTRAINED = "constrained_share"
SHARE_POLICIES: Tuple[str, ...] = (SHARE_NONE, SHARE_IDEAL, SHARE_CONSTRAINED)

SHARE_POLICY_CN: Dict[str, str] = {
    SHARE_NONE: "不共享（各平台只用局部测量）",
    SHARE_IDEAL: "理想零延迟共享（全连接 / 无延迟 / 无丢包 / 不过期）",
    SHARE_CONSTRAINED: "受限共享（有延迟、丢包、带宽与过期限制）",
}


class TimeBoundaryViolation(RuntimeError):
    """试图读取尚未到达（或未来）的消息 —— 这是硬边界，必须抛错。"""


@dataclass
class CommConfig:
    """通信配置：策略 + 每个平台的消息大小/过期等公共参数。"""

    policy: str = SHARE_NONE
    message_size_bytes: float = 128.0
    #: 受限策略下的默认链路参数
    base_delay_s: float = 0.0
    jitter_s: float = 0.0
    loss_prob: float = 0.0
    bandwidth_bytes_per_s: Optional[float] = None
    max_queue_bytes: Optional[float] = None
    expiry_s: Optional[float] = None
    seed: int = 0

    # --- v4.5 通信时序压力（默认全零，旧行为逐位不变）---
    burst_loss_prob: float = 0.0
    burst_length: int = 0
    outage_windows: Tuple[Tuple[float, float], ...] = ()
    recovery_congestion_s: float = 0.0
    recovery_extra_delay_s: float = 0.0
    recovery_loss_prob: float = 0.0
    reorder_prob: float = 0.0
    reorder_extra_delay_s: float = 2.0
    #: 逐链路差异：`{link_id: {字段: 值}}`，用于"不同链路延迟"对照。
    #: 键形如 `"A->B"`（与 `LinkConfig.link_id` 一致）。
    per_link_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.policy not in SHARE_POLICIES:
            raise ValueError(
                f"共享策略 {self.policy!r} 非法，只能是 {list(SHARE_POLICIES)}"
            )
        if self.message_size_bytes <= 0:
            raise ValueError("message_size_bytes 必须为正")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CommConfig":
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


class CommBus:
    """多平台通信总线：路由消息、维护到达时间线、产出逐消息日志。

    ⚠️ 总线**不做任何真值访问**。它只把 `MeasurementMessage` 在平台之间搬运。
    """

    def __init__(
        self,
        platform_ids: Sequence[str],
        config: Optional[CommConfig] = None,
    ) -> None:
        self.platform_ids: List[str] = list(platform_ids)
        self.config = config or CommConfig()
        self.config.validate()

        self._links: Dict[Tuple[str, str], CommLink] = {}
        self._build_links()

        #: 全部已发送消息（含被丢的），逐消息日志用
        self.log: List[MeasurementMessage] = []
        self._seq = 0
        #: 每个消费方（目的平台）已投递过的消息 ID 集合。
        #: **没有它就会重复投递历史消息**（见 `consume` 的说明）。
        self._delivered_to: Dict[str, set] = {p: set() for p in self.platform_ids}

    # ------------------------------------------------------------------

    def _build_links(self) -> None:
        cfg = self.config
        if cfg.policy == SHARE_NONE:
            return  # 没有任何链路
        for src in self.platform_ids:
            for dst in self.platform_ids:
                if src == dst:
                    continue
                if cfg.policy == SHARE_IDEAL:
                    link_config = LinkConfig(
                        link_id=f"{src}->{dst}", src_platform_id=src,
                        dst_platform_id=dst, base_delay_s=0.0, jitter_s=0.0,
                        loss_prob=0.0, bandwidth_bytes_per_s=None,
                        max_queue_bytes=None, expiry_s=None, seed=cfg.seed,
                    )
                else:
                    link_config = LinkConfig(
                        link_id=f"{src}->{dst}", src_platform_id=src,
                        dst_platform_id=dst, base_delay_s=cfg.base_delay_s,
                        jitter_s=cfg.jitter_s, loss_prob=cfg.loss_prob,
                        bandwidth_bytes_per_s=cfg.bandwidth_bytes_per_s,
                        max_queue_bytes=cfg.max_queue_bytes,
                        expiry_s=cfg.expiry_s, seed=cfg.seed,
                        burst_loss_prob=cfg.burst_loss_prob,
                        burst_length=cfg.burst_length,
                        outage_windows=tuple(cfg.outage_windows or ()),
                        recovery_congestion_s=cfg.recovery_congestion_s,
                        recovery_extra_delay_s=cfg.recovery_extra_delay_s,
                        recovery_loss_prob=cfg.recovery_loss_prob,
                        reorder_prob=cfg.reorder_prob,
                        reorder_extra_delay_s=cfg.reorder_extra_delay_s,
                    )
                    # 逐链路覆盖（"不同链路延迟"等对照）
                    for key, value in (cfg.per_link_overrides or {}).get(
                        link_config.link_id, {}
                    ).items():
                        if not hasattr(link_config, key):
                            raise ValueError(
                                f"per_link_overrides[{link_config.link_id!r}] "
                                f"含未知字段 {key!r}"
                            )
                        setattr(link_config, key, value)
                self._links[(src, dst)] = CommLink(link_config)

    @property
    def links(self) -> Dict[Tuple[str, str], CommLink]:
        return dict(self._links)

    @property
    def sharing_enabled(self) -> bool:
        return bool(self._links)

    def reset(self) -> None:
        for link in self._links.values():
            link.reset()
        self.log.clear()
        self._seq = 0
        self._delivered_to = {p: set() for p in self.platform_ids}

    # ------------------------------------------------------------------

    def publish(
        self,
        src_platform_id: str,
        src_sensor_id: str,
        measurements: Iterable[Any],
        now: float,
        dst_platform_ids: Optional[Sequence[str]] = None,
    ) -> List[MeasurementMessage]:
        """把一个平台的局部测量打包成消息发出去。

        `measurements` 是 `MeasurementRecord`（或任何有 `to_dict` 的测量对象）。
        只取**白名单内的测量字段**进载荷；`assert_payload_clean` 会再查一遍。
        """
        if not self._links:
            return []

        targets = list(dst_platform_ids) if dst_platform_ids is not None else [
            p for p in self.platform_ids if p != src_platform_id
        ]
        sent: List[MeasurementMessage] = []

        for measurement in measurements:
            raw = measurement.to_dict(include_truth=False) if hasattr(
                measurement, "to_dict"
            ) else dict(measurement)
            self._seq += 1
            for dst in targets:
                link = self._links.get((src_platform_id, dst))
                if link is None:
                    continue
                message = MeasurementMessage(
                    msg_id=f"M{self._seq:06d}",
                    src_platform_id=src_platform_id,
                    src_sensor_id=src_sensor_id,
                    seq=self._seq,
                    generated_at=float(now),
                    sent_at=float(now),
                    payload=dict(raw),
                    size_bytes=float(self.config.message_size_bytes),
                    dst_platform_id=dst,
                )
                link.transmit(message, now)
                self.log.append(message)
                sent.append(message)
        return sent

    # ------------------------------------------------------------------

    def arrived(self, now: float) -> List[MeasurementMessage]:
        """**只返回在 `now` 时刻已经到达**的消息（本模块的核心纪律）。

        语义是**过滤**：在途消息不会出现在返回值里。
        （早期版本在这里对在途消息抛异常，结果正常调用直接失败——
        过滤才是正确语义；越界检测交给 `assert_only_arrived`。）
        """
        out: List[MeasurementMessage] = []
        for message in self.log:
            if message.dropped or message.arrived_at is None:
                continue
            if message.arrived_at <= now + 1e-12:
                out.append(message)
        return out

    def consume(self, dst_platform_id: str, now: float) -> List[MeasurementMessage]:
        """**消费**在 `now` 已到达、且尚未投递给该平台的消息。

        这是决策侧应该用的接口。与 `arrived()` 的区别：

        * `arrived(now)`  —— 纯查询：返回**全部**已到达消息（含以前投递过的）。
          用于诊断与"当前共有多少信息可用"这类统计；
        * `consume(dst, now)` —— 投递一次即记账，**同一消息不会重复交给
          同一消费方**。决策用它。

        为什么必须有这个区别：早期只有 `arrived()`，于是每一步都把
        自 t=0 以来的全部历史重新投递一遍。表现是远端测量堆积、
        绝大多数因"年龄过大"被融合中心判为过期——但那不是链路的
        真实延迟，而是重复投递造成的**假象**。这个 bug 是被
        `fusion/lifecycle.py` 的漏斗诊断抓出来的（拒绝原因出现
        age=4s…28s，而链路延迟配置只有 1.2 s，物理上不可能）。
        """
        already = self._delivered_to.setdefault(dst_platform_id, set())
        out: List[MeasurementMessage] = []
        for message in self.log:
            if message.dropped or message.arrived_at is None:
                continue
            if message.arrived_at > now + 1e-12:
                continue
            if message.msg_id in already:
                continue
            already.add(message.msg_id)
            out.append(message)
        return out

    def deliverable_count(self, dst_platform_id: str, now: float) -> int:
        """当前已到达且尚未投递给该目的的消息数（只读）。"""
        already = self._delivered_to.get(dst_platform_id, set())
        return sum(
            1 for message in self.log
            if not message.dropped
            and message.arrived_at is not None
            and message.arrived_at <= now + 1e-12
            and message.dst_platform_id == dst_platform_id
            and message.msg_id not in already
        )

    def pending_for(self, dst_platform_id: str) -> int:
        """该平台尚在途（已发送未投递）的消息数。"""
        already = self._delivered_to.get(dst_platform_id, set())
        return sum(
            1 for m in self.log
            if not m.dropped and m.arrived_at is not None and m.msg_id not in already
        )

    @staticmethod
    def assert_only_arrived(
        messages: Iterable[MeasurementMessage], now: float
    ) -> None:
        """守卫：确认一批消息**全部**已经到达。

        这是给"直接遍历 `bus.log`"这类代码用的。真实风险正是有人绕过
        `arrived()` 去翻全量日志——那样就会读到在途/未来消息。
        把守卫做成一个**可调用、可测试**的函数，而不是写在注释里。
        """
        for message in messages:
            if message.dropped or message.arrived_at is None:
                continue
            if message.arrived_at > now + 1e-12:
                raise TimeBoundaryViolation(
                    f"消息 {message.msg_id} 的到达时刻 {message.arrived_at:g} "
                    f"晚于当前时刻 {now:g}，不得被读取"
                )

    def arrived_from(
        self, dst_platform_id: str, now: float
    ) -> List[MeasurementMessage]:
        """某个平台在 `now` 时刻已经到达的消息。

        注意：总线记录的是广播副本，因此这里按"非本平台发出"来筛选
        （即这条消息是从别的平台来的）。
        """
        return [
            m for m in self.arrived(now)
            if m.src_platform_id != dst_platform_id
        ]

    def in_flight(self, now: float) -> List[MeasurementMessage]:
        """仍在路上、尚未到达的消息。

        ⚠️ 这些消息**不得**被决策读取内容。本方法只用于统计与日志
        （例如"当前有多少信息在途"），返回的对象在调用方侧应只取元数据。
        """
        return [
            m for m in self.log
            if m.arrived_at is not None and m.arrived_at > now + 1e-12
        ]

    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """延迟 / 丢包统计（逐链路 + 汇总）。"""
        delivered = [m for m in self.log if m.arrived_at is not None]
        latencies = sorted(m.latency_s() for m in delivered)
        drops: Dict[str, int] = {}
        for message in self.log:
            if message.dropped:
                drops[message.drop_reason] = drops.get(message.drop_reason, 0) + 1

        def percentile(values: Sequence[float], q: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
            return float(values[index])

        per_link = {
            link.config.link_id: dict(link.stats) for link in self._links.values()
        }

        # --- 乱序统计（v4.5）---
        #
        # 定义：在同一个 (src, dst) 链路上，一条消息**到达时**，
        # 若此前已经有"生成得更晚"的消息到达过，则算一次乱序到达。
        # 这正是会打到跟踪器的那件事——先到的信息更新了状态，
        # 后到的旧信息又把它往回拽。
        by_route: Dict[Tuple[str, str], List[MeasurementMessage]] = {}
        for message in delivered:
            by_route.setdefault(
                (message.src_platform_id, message.dst_platform_id), []
            ).append(message)
        out_of_order = 0
        max_lag = 0.0
        for route_messages in by_route.values():
            newest_generated = None
            for message in sorted(route_messages,
                                  key=lambda m: (m.arrived_at or 0.0, m.seq)):
                if newest_generated is not None and message.generated_at < newest_generated:
                    out_of_order += 1
                    max_lag = max(max_lag, newest_generated - message.generated_at)
                newest_generated = max(
                    newest_generated if newest_generated is not None else message.generated_at,
                    message.generated_at,
                )

        return {
            "policy": self.config.policy,
            "policy_cn": SHARE_POLICY_CN.get(self.config.policy, ""),
            "n_links": len(self._links),
            "n_messages": len(self.log),
            "n_delivered": len(delivered),
            "n_dropped": len(self.log) - len(delivered),
            "drop_reasons": drops,
            "delivery_rate": (len(delivered) / len(self.log)) if self.log else 0.0,
            "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "latency_p50_s": percentile(latencies, 0.50),
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_max_s": latencies[-1] if latencies else 0.0,
            # --- v4.5 ---
            "n_out_of_order": out_of_order,
            "out_of_order_rate": (out_of_order / len(delivered)) if delivered else 0.0,
            "max_reorder_lag_s": max_lag,
            "per_link": per_link,
        }

    def message_log_rows(self) -> List[Dict[str, Any]]:
        """逐消息日志（CSV/JSON 导出用）。"""
        return [m.to_dict() for m in self.log]

    def describe(self) -> str:
        cfg = self.config
        lines = [
            f"通信总线：策略 {cfg.policy}（{SHARE_POLICY_CN.get(cfg.policy, '')}），"
            f"{len(self._links)} 条链路，{len(self.platform_ids)} 个平台"
        ]
        if cfg.policy == SHARE_CONSTRAINED:
            lines.append(
                f"  链路参数：固定延迟 {cfg.base_delay_s:g}s，抖动 ±{cfg.jitter_s:g}s，"
                f"丢包 {cfg.loss_prob:g}，"
                f"带宽 {cfg.bandwidth_bytes_per_s or '不限'} B/s，"
                f"队列上限 {cfg.max_queue_bytes or '不限'} B，"
                f"过期 {cfg.expiry_s if cfg.expiry_s is not None else '不过期'} s"
            )
        return "\n".join(lines)
