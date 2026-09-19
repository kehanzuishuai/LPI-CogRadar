"""低截获雷达功率调控环境（Gymnasium 风格接口）。

接口契约与 Gymnasium 完全一致，但**不依赖** gymnasium/numpy：

    obs, info                        = env.reset(seed=None, options=None)
    obs, reward, terminated, truncated, info = env.step(action)

    action ∈ {0, 1, ..., 10}      11 档离散发射功率（env.action_space: Discrete(11)）
    obs    ∈ Box(D)               定长浮点向量，D 由 observation_mode 决定

终止语义遵循 Gymnasium 约定：
    terminated=True —— 能量耗尽等真实终止（由配置 terminate_on_energy_exhausted 控制）
    truncated=True  —— 到达场景时长上限（时间截断，不是 MDP 的终止态）

奖励与离线指标（metrics.collector）共用 models.reward.composite_reward，
保证「训练时优化的目标」和「实验报告里汇报的综合收益」是同一个定义。

观测模式（v4.0 新增，默认保持 v3.1 行为）
----------------------------------------
`observation_mode="full"`（默认）
    直接给智能体真值，观测维度 12。**与 v3.1 逐位一致**，旧实验与旧模型可直接复用。

`observation_mode="pomdp"`
    把「真值 -> 可观测量」交给 engine.observation_model.ObservationModel：
    距离/RCS/干扰/能量/侦察机方位/Pint/暴露/上一步 Pd 全都有测量噪声，
    并可选延迟与随机丢测；侦察机真实位置、真实 Pint、真实累计暴露不再进入观测。
    观测维度 16 = 前 12 维（含义与 full 相同，但都是**估计值**）+ 4 维不确定度通道。

`history_len > 1`
    在基础观测之上堆叠最近 K 帧（frame stacking / 历史窗口编码），
    观测维度 = 基础维度 × K。这是给部分可观测条件下做「时序记忆」的 DQN 分支用的，
    不需要改动任何 RL 代码（只是观测维度变大）。

注意：**真值演进与奖励计算完全不受观测模式影响**——噪声只加在「智能体看到什么」上，
物理参数、能量约束、奖励函数一律不动。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from engine.equations import snr_db_for_prob
from engine.observation_model import ObservationModel, ObservationNoiseConfig
from sensor import (
    SLOT_FIELDS as TRACK_SLOT_FIELDS,
    TrackTableConfig,
    fuse_measurements,
)
from communication import SHARE_POLICY_CN, CommBus
from fusion import FusionCenter, FusionConfig
from fusion.lifecycle import LifecycleLog
from sensor.config import build_suite_from_config
from engine.simulator import Simulator
from engine.spaces import Box, Discrete

class _SharedMeasurement:
    """从通信消息载荷解包出的测量对象（供跟踪器消费）。

    刻意不暴露真值字段：接收方拿到的是**消息载荷**而非原始测量对象，
    这正是"跨平台共享"与"本地自用"的区别。
    真值属性恒为 None，只是为了满足跟踪器的接口一致性。
    """

    __slots__ = ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                 "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                 "std_az_deg", "std_el_deg", "confidence", "msg_id", "platform_id",
                 "truth_id", "truth_range_m", "truth_azimuth_deg",
                 "truth_elevation_deg", "truth_range_rate_mps", "is_false_alarm")

    def __init__(self, payload: Dict[str, Any], msg_id: str = "",
                 platform_id: str = "") -> None:
        for key in ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                    "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                    "std_az_deg", "std_el_deg", "confidence"):
            setattr(self, key, payload.get(key))
        self.msg_id = msg_id
        self.platform_id = platform_id
        self.is_false_alarm = False
        self.truth_id = None
        self.truth_range_m = None
        self.truth_azimuth_deg = None
        self.truth_elevation_deg = None
        self.truth_range_rate_mps = None


DEFAULT_CONFIG_PATH = "config/radar_scenario_v1.json"

# 观测向量各维含义（顺序即维度顺序，便于论文与调试时对齐）
OBSERVATION_FEATURES: List[str] = [
    "primary_target_range_norm",  # 主目标距离 / 20 km
    "min_target_rcs_norm",  # 各目标最小 RCS / 5 m^2
    "max_target_range_norm",  # 最远目标距离 / 20 km
    "interceptor_range_norm",  # 侦察机距离 / 150 km
    "jam_noise_ratio_norm",  # min(J/N, 5) / 5，雷达可测量的噪声基底抬升
    "previous_power_norm",  # 上一步发射功率 / 最大档功率
    "remaining_energy_norm",  # 剩余能量比例（第二版为硬约束，见 README）
    "time_norm",  # 归一化时间
    "previous_pd",  # 上一步最小探测概率
    "previous_pint",  # 上一步有效截获概率 Pint_eff
    "required_pd",  # 任务要求的探测概率
    "exposure_norm",  # 第二版新增：累计暴露量 / 侦察证据
]

#: POMDP 模式下追加的不确定度通道（紧跟 OBSERVATION_FEATURES 之后）
POMDP_EXTRA_FEATURES: List[str] = [
    "observation_quality",  # 0~1，观测整体新鲜度/精度（1 = 全新鲜全精确）
    "interceptor_range_sigma_norm",  # 侦察机距离估计标准差 / 150 km
    "exposure_sigma",  # 累计暴露估计标准差
    "pint_sigma",  # 截获概率估计标准差
]

POMDP_OBSERVATION_FEATURES: List[str] = OBSERVATION_FEATURES + POMDP_EXTRA_FEATURES

#: 测量模式下**航迹表之后**追加的特征：
#: 前 4 维是**本平台精确已知的自状态**（不来自传感器），
#: 后 9 维是测量层的缺失统计（逐原因计数 + 观测质量）。
#: 把这两组分开是因为它们的可信度完全不同：自状态没有测量误差，
#: 而缺失统计本身就是"我看到得有多差"的元信息。
MEASUREMENT_EXTRA_FEATURES: List[str] = [
    # --- 自状态（精确已知，非测量）---
    "own_previous_power_norm",
    "own_remaining_energy_norm",
    "own_time_norm",
    "own_required_pd",
    # --- 测量层元信息 ---
    "meas_n_measurements_norm",
    "meas_n_fresh_norm",
    "meas_observation_quality",
    "meas_rate_out_of_fov",
    "meas_rate_beyond_range",
    "meas_rate_occluded",
    "meas_rate_not_updated",
    "meas_rate_missed_detection",
    "meas_rate_sensor_unavailable",
]

OBSERVATION_MODES: Tuple[str, ...] = ("full", "pomdp", "ideal", "realistic")

#: 哪些模式走传感器测量层
MEASUREMENT_MODES: Tuple[str, ...] = ("ideal", "realistic")

#: **有符号观测维**：编码器按设计输出 -1..1（见 `sensor/fusion.py::SLOT_FIELDS`），
#: 因此观测空间的上下界必须是 [-1, 1]，**不得**被统一裁成 [0, 1]。
#:
#: 为什么这条必须显式声明：`bearing_norm` / `elevation_norm` / `range_rate_norm`
#: 三个量在物理上是有符号的（目标可以在机头左右任一侧、可以接近也可以远离）。
#: 早先版本把整个观测空间声明成 `Box([0]*D, [1]*D)` 并在每步 `clip`，
#: 于是"目标在机头负侧"这一信息被**静默抹掉**——不是报错，而是把
#: 一个真实的方向量变成常数 0，模型既学不到也没人会发现。
#: 默认场景里目标恰好都在机头正侧，所以这个缺陷长期没有暴露；
#: 只要存在任一目标位于机头负侧的场景（多目标/编队/交接场景很常见），
#: 它就必然触发。见 `tests/test_observation_contract.py` 的往返测试。
SIGNED_OBSERVATION_FIELDS: FrozenSet[str] = frozenset({
    "bearing_norm",
    "elevation_norm",
    "range_rate_norm",
})


def observation_bounds(features: Sequence[str]) -> Tuple[List[float], List[float]]:
    """按**逐维语义**给出观测的上下界（不再统一 [0, 1]）。

    只有显式登记在 `SIGNED_OBSERVATION_FIELDS` 里的维取 [-1, 1]，
    其余取 [0, 1]。新增有符号维时必须在这里登记，否则会被裁掉。
    """
    low: List[float] = []
    high: List[float] = []
    for name in features:
        if name in SIGNED_OBSERVATION_FIELDS:
            low.append(-1.0)
            high.append(1.0)
        else:
            low.append(0.0)
            high.append(1.0)
    return low, high


class LpiPowerEnv:
    """低截获雷达智能功率调控环境。"""

    metadata = {
        "name": "LpiRadarPower-v1",
        "render_modes": ["ansi"],
        "observation_features": OBSERVATION_FEATURES,
        "pomdp_observation_features": POMDP_OBSERVATION_FEATURES,
        "observation_modes": list(OBSERVATION_MODES),
    }

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        reward_weights: Dict[str, float] | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
        energy_budget_j: float | None = None,
        observation_mode: str = "full",
        history_len: int = 1,
        observation_noise: ObservationNoiseConfig | Dict[str, Any] | None = None,
        terminate_on_energy_exhausted: bool | None = None,
        extra_overrides: Dict[str, Any] | None = None,
        expose_observation_truth: bool = False,
        measurement_max_tracks: int = 4,
        expose_measurement_truth: bool = False,
        enable_tracker: bool = True,
    ) -> None:
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode={observation_mode!r} 非法，只能是 {OBSERVATION_MODES}"
            )
        if history_len < 1:
            raise ValueError("history_len 至少为 1")

        #: 是否在 info 里附带观测量的真值。
        #: 默认 False —— 真值若出现在 info 里，策略代码就有可能顺手读它，
        #: 那部分可观测实验就白做了。只有诊断/消融脚本才显式打开它做统计。
        self.expose_observation_truth = bool(expose_observation_truth)

        self.sim = Simulator(config_path)
        self.sim.load_config()

        # 通过 apply_overrides 覆盖（会写回缓存配置，不会被 reset 抹掉）。
        # 只覆盖任务/评估层面参数，不触碰雷达方程与侦察模型。
        if (
            reward_weights
            or energy_budget_j is not None
            or terminate_on_energy_exhausted is not None
            or extra_overrides
        ):
            self.sim.apply_overrides(
                reward_weights=reward_weights,
                energy_budget_j=energy_budget_j,
                terminate_on_energy_exhausted=terminate_on_energy_exhausted,
                extra=extra_overrides,
            )

        # --- 观测模式 ---
        self.observation_mode = observation_mode
        self.history_len = int(history_len)

        # 航迹表配置必须在构造特征表**之前**定义：特征维度由它决定
        # （max_tracks × 每槽位维度）。顺序写反过一次，直接把维数算错了。
        self._track_config = TrackTableConfig(max_tracks=int(measurement_max_tracks))
        self.measurement_noise_scale = 0.0 if observation_mode == "ideal" else 1.0

        if observation_mode == "full":
            self._base_features = OBSERVATION_FEATURES
        elif observation_mode == "pomdp":
            self._base_features = POMDP_OBSERVATION_FEATURES
        else:
            self._base_features = (
                list(TRACK_SLOT_FIELDS) * self._track_config.max_tracks
                + list(MEASUREMENT_EXTRA_FEATURES)
            )
        self._base_obs_dim = len(self._base_features)
        self.observation_features = list(self._base_features) * self.history_len

        self._noise_config = self._resolve_noise_config(observation_noise)
        self.observation_model = ObservationModel(self._noise_config)
        self._history: deque = deque(maxlen=self.history_len)

        # --- v4.2 传感器测量层（仅在 ideal / realistic 模式下构造与调用）---
        #: 是否在 info 里附带测量记录的真值字段（`truth_id` / 误差）。
        #: ⚠️ 这是**评测通道**：打开后 `info["measurements"]` 会含真值，
        #: 决策算法一律不得读取。默认关闭。
        self.expose_measurement_truth = bool(expose_measurement_truth)
        self.suite: Any = None
        if observation_mode in MEASUREMENT_MODES:
            self.suite = build_suite_from_config(
                self.sim.scene,
                getattr(self.sim, "_raw_config", {}) or {},
                seed=0 if seed is None else int(seed),
                noise_scale=self.measurement_noise_scale,
            )
        self._last_suite_report: Any = None
        self._last_missing_counts: Dict[str, int] = {}
        self._last_fused: Any = None

        # --- v4.5：跟踪器与通信总线（AI 诊断的结构化证据来源）---
        # 只加记账，**不改观测向量**：旧的 full/pomdp/ideal/realistic 维度不变。
        self.platform_id = str(self.sim.radar.radar_id)
        self.enable_tracker = bool(enable_tracker) and observation_mode in MEASUREMENT_MODES
        self.tracker: Any = None
        self.comm_bus: Any = None
        self._fusion_snapshot: Any = None
        self._comm_used_this_step = 0
        self._comm_arrived_this_step = 0
        self._comm_rejected_this_step = 0
        if self.enable_tracker and self.suite is not None:
            own_sensor_ids = [sc.sensor_id for sc in self.suite.sensors
                              if sc.config.mounting_id == self.platform_id]
            self.tracker = FusionCenter(
                self.platform_id,
                FusionConfig(),
                own_sensor_ids=own_sensor_ids,
                lifecycle=LifecycleLog(True),
            )

        self.action_space = Discrete(self.sim.num_levels)
        # 逐字段上下界：有符号维保留负值（见 `observation_bounds` 的说明）。
        # 统一 [0, 1] 会静默抹掉方位/俯仰/径向速度的符号，属于**信息破坏**，
        # 不是"归一化"。
        _obs_low, _obs_high = observation_bounds(self.observation_features)
        self.observation_space = Box(_obs_low, _obs_high)
        self.render_mode = render_mode

        self._last_pd: float = 0.0  # 智能体侧「上一步 Pd」（full 模式为真值）
        self._last_pint: float = 0.0
        self._last_pd_true: float = 0.0  # 真值，仅用于 POMDP 生成观测
        self._last_pint_true: float = 0.0
        self._last_quality: float = 1.0
        self._last_estimates: Dict[str, Any] = {}
        self._obs_step_index: int = 0
        self._seed = seed
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # 观测模式辅助
    # ------------------------------------------------------------------

    def _resolve_noise_config(
        self, observation_noise: ObservationNoiseConfig | Dict[str, Any] | None
    ) -> ObservationNoiseConfig:
        """决定实际使用的噪声配置。

        优先级：显式参数 > 场景配置的 `observation` 段 > 内置默认（开启噪声）。
        full 模式下一律关闭噪声（保持 v3.1 逐位一致）。
        """
        if self.observation_mode == "full":
            return ObservationNoiseConfig(enabled=False)

        if isinstance(observation_noise, ObservationNoiseConfig):
            cfg = observation_noise
        elif isinstance(observation_noise, dict):
            cfg = ObservationNoiseConfig.from_dict(observation_noise)
        else:
            raw_root = getattr(self.sim, "_raw_config", None) or {}
            raw = raw_root.get("observation") if isinstance(raw_root, dict) else None
            cfg = ObservationNoiseConfig.from_dict(raw) if isinstance(raw, dict) else ObservationNoiseConfig()
            if not cfg.enabled:
                # 场景配置没显式开启时，pomdp 模式默认启用噪声
                cfg.enabled = True
        cfg.validate()
        return cfg

    def set_observation_noise(self, **kwargs: Any) -> None:
        """运行时调噪声参数（消融实验用）。会重置观测模型状态。"""
        data = {k: getattr(self._noise_config, k) for k in self._noise_config.__dataclass_fields__}
        data.update(kwargs)
        self._noise_config = ObservationNoiseConfig.from_dict(data)
        self._noise_config.validate()
        self.observation_model = ObservationModel(self._noise_config)

    # ------------------------------------------------------------------
    # Gymnasium 风格接口
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """复位环境，返回 (observation, info)。

        seed 会传给 Simulator.reset()，真正决定干扰强度起伏序列；
        不传则沿用场景配置里的 random_seed。
        """
        if seed is not None:
            self._seed = seed
            self.action_space.seed(seed)

        self.sim.reset(seed=seed)
        self._last_pd = 0.0
        self._last_pint = 0.0
        self._last_pd_true = 0.0
        self._last_pint_true = 0.0
        self._last_quality = 1.0
        self._last_estimates = {}
        self._obs_step_index = 0
        # 观测模型的随机源与场景种子绑定，保证同种子可复现
        obs_seed = int(seed) if seed is not None else int(self._noise_config.seed)
        self.observation_model.config.seed = obs_seed
        self.observation_model.reset(seed=obs_seed)

        if self.suite is not None:
            self.suite.reset()
        if self.tracker is not None:
            self.tracker.reset()
        if self.comm_bus is not None:
            self.comm_bus.reset()
        self._fusion_snapshot = None
        self._comm_used_this_step = 0
        self._comm_arrived_this_step = 0
        self._comm_rejected_this_step = 0
        self._last_suite_report = None
        self._last_missing_counts = {}
        self._history.clear()
        first = self._build_base_observation()
        self._history.append(first)

        info = {
            "scenario": self.sim.scenario.scenario_name,  # type: ignore[union-attr]
            "num_levels": self.sim.num_levels,
            "power_levels_w": list(self.sim.power_levels_w),
            "required_pd": self.sim.radar.required_pd,  # type: ignore[union-attr]
            "energy_budget_j": self.sim.radar.energy_budget_j,  # type: ignore[union-attr]
            "exposure_decay": self.sim.scenario.exposure_decay,  # type: ignore[union-attr]
            "exposure_gain": self.sim.scenario.exposure_gain,  # type: ignore[union-attr]
            "observation_features": self.observation_features,
            "observation_mode": self.observation_mode,
            "history_len": self.history_len,
            "observation_space_dim": len(self.observation_features),
            "observation_quality": 1.0,
        }
        return self._stacked_observation(), info

    def step(
        self, action: int
    ) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """施加一个功率档位动作，返回 (obs, reward, terminated, truncated, info)。

        能量是硬约束：**执行前**先检查本步所需能量 Pt·Δt 是否超过剩余能量。
        若不可行，动作会被裁剪到当前可执行的最高档（`info["action_clipped"]=True`
        记录这一事实），从而保证累计能耗**永不**超过 `energy_budget_j`。
        仿真层对不可行动作是直接抛 `InfeasibleActionError` 的，
        环境层做裁剪是为了让 RL 训练不会因一次越界动作直接崩掉；
        智能体可以用 `env.action_masks()` 自己做 masked argmax，从源头避免越界。
        """
        requested = int(action)
        if not 0 <= requested < self.sim.num_levels:
            raise ValueError(
                f"action={requested} 越界，合法范围 0..{self.sim.num_levels - 1}"
            )

        feasible = self.sim.feasible_levels()
        if not feasible:
            raise RuntimeError(
                "剩余能量已买不起任何一档功率，episode 应当已结束（请先 reset）"
            )

        executed = requested if requested in feasible else feasible[-1]
        action_clipped = executed != requested

        result = self.sim.step(executed)

        self._last_pd_true = result.pd_min
        self._last_pint_true = result.intercept_prob
        # 观测模式下智能体记住的是它「测到」的值，而不是真值
        self._last_pd = result.pd_min
        self._last_pint = result.intercept_prob

        self._obs_step_index += 1
        frame = self._build_base_observation()
        self._history.append(frame)

        energy_terminated = bool(
            self.sim.radar.terminate_on_energy_exhausted  # type: ignore[union-attr]
            and self.sim.terminated_by_energy
        )
        terminated = energy_terminated
        truncated = bool(self.sim.is_done and not terminated)

        info = {
            "step_index": result.step_index,
            "time": result.time,
            "requested_action": requested,
            "executed_action": executed,
            "action_clipped": action_clipped,
            "action_mask": self.action_masks(),
            "tx_power_w": result.tx_power_w,
            "power_level": result.power_level,
            "pd_min": result.pd_min,
            "intercept_prob": result.intercept_prob,
            "intercept_prob_instant": result.intercept_prob_instant,
            "task_satisfied": result.task_satisfied,
            "task_violated": result.task_violated,
            "jammer_active": result.jammer_active,
            "jam_noise_ratio": result.jam_noise_ratio,
            "exposure": result.exposure,
            "exposure_next": result.exposure_next,
            "cumulative_energy_j": result.cumulative_energy_j,
            "remaining_energy_j": result.remaining_energy_j,
            "energy_fraction": result.energy_fraction,
            "energy_exhausted": result.energy_exhausted,
            "terminal_penalty": result.terminal_penalty,
            "required_snr_db": result.required_snr_db,
            "terminated_reason": "energy" if energy_terminated else ("horizon" if truncated else ""),
            "observation_mode": self.observation_mode,
            "observation_quality": self._last_quality,
        }
        if self.observation_mode == "pomdp":
            info["observations"] = self._last_estimates
        return self._stacked_observation(), result.reward, terminated, truncated, info

    def action_masks(self) -> List[bool]:
        """当前可行动作掩码（True = 本步能量买得起）。

        供智能体做 masked argmax / masked ε-greedy，
        也为将来接入标准 masked-RL 实现（如 sb3-contrib MaskablePPO）预留接口。
        """
        return self.sim.action_mask()

    def render(self) -> str | None:
        """文本回放当前步（render_mode='ansi'）。"""
        if not self.sim.results:
            return "episode 尚未 step 过"

        r = self.sim.results[-1]
        line = (
            f"t={r.time:5.1f}s  Pt={r.tx_power_w:6.2f}W  "
            f"Pd_min={r.pd_min:5.3f}{'✓' if r.task_satisfied else '✗'}  "
            f"Pint={r.intercept_prob:5.3f}  "
            f"J/N={r.jam_noise_ratio:5.2f}  "
            f"E={r.cumulative_energy_j:8.1f}J  r={r.reward:+.3f}"
        )
        if self.render_mode == "ansi":
            print(line)
        return line

    def close(self) -> None:
        """无外部资源需要释放，保留以满足 Gymnasium 接口完整性。"""
        return None

    # ------------------------------------------------------------------
    # 观测构造
    # ------------------------------------------------------------------

    def _observation(self) -> List[float]:
        """返回当前观测（含历史窗口堆叠）。"""
        return self._stacked_observation()

    def _stacked_observation(self) -> List[float]:
        if self.history_len == 1:
            return self.observation_space.clip(list(self._history[-1]))
        flat: List[float] = []
        for frame in self._history:
            flat.extend(frame)
        # deque 未满时用最早的一帧补齐，保证长度恒定
        while len(flat) < len(self.observation_features):
            flat = list(self._history[0]) + flat
        return self.observation_space.clip(flat[: len(self.observation_features)])

    def _truth(self) -> Dict[str, float]:
        """从仿真读出「真值」，交给观测模型加噪声。"""
        sim = self.sim
        targets = sim.active_targets()
        primary = sim._primary_target()
        primary_range = primary.range_to(sim.radar.x, sim.radar.y) if primary else 0.0  # type: ignore[union-attr]
        min_rcs = min((t.rcs_m2 for t in targets), default=0.0)
        max_range = max(
            (t.range_to(sim.radar.x, sim.radar.y) for t in targets), default=0.0  # type: ignore[union-attr]
        )
        esm_ranges = [
            e.range_to(sim.radar.x, sim.radar.y) for e in sim.interceptors if e.is_active  # type: ignore[union-attr]
        ]
        esm_range = min(esm_ranges) if esm_ranges else 0.0
        _, _, _, jam_ratio = sim.jam_state()
        return {
            "target_range": primary_range,
            "target_rcs": min_rcs,
            "max_range": max_range,
            "interceptor_range": esm_range,
            "jam_ratio": jam_ratio,
            "remaining_energy": sim.remaining_energy_j,
            "pd_min": self._last_pd_true,
            "pint_eff": self._last_pint_true,
            "exposure": sim.exposure.value,
            "previous_power": (
                sim.scenario.power_levels_w[sim.previous_power_level]  # type: ignore[union-attr]
                if sim.previous_power_level >= 0
                else 0.0
            ),
        }

    def _build_base_observation(self) -> List[float]:
        """构造单帧观测（不含历史堆叠）。

        full 模式：与 v3.1 完全相同的 12 维真值观测（逐位一致）。
        pomdp 模式：同样 12 维但全部是估计值，再追加 4 维不确定度通道。
        """
        if self.observation_mode == "full":
            return self._build_full_observation()
        if self.observation_mode == "pomdp":
            return self._build_pomdp_observation()
        return self._build_measurement_observation()

    def _build_full_observation(self) -> List[float]:
        """v3.1 的原始观测构造，保持逐字不变以保证旧实验可复现。"""
        sim = self.sim
        assert sim.radar is not None and sim.scenario is not None

        targets = sim.active_targets()
        primary = sim._primary_target()
        primary_range = primary.range_to(sim.radar.x, sim.radar.y) if primary else 0.0
        min_rcs = min((t.rcs_m2 for t in targets), default=0.0)
        max_range = max((t.range_to(sim.radar.x, sim.radar.y) for t in targets), default=0.0)

        esm_ranges = [
            e.range_to(sim.radar.x, sim.radar.y) for e in sim.interceptors if e.is_active
        ]
        esm_range = min(esm_ranges) if esm_ranges else 0.0

        _, _, _, jam_ratio = sim.jam_state()

        max_power = sim.scenario.max_power_w
        previous_power = (
            sim.scenario.power_levels_w[sim.previous_power_level]
            if sim.previous_power_level >= 0
            else 0.0
        )

        raw = [
            primary_range / 20000.0,
            min_rcs / 5.0,
            max_range / 20000.0,
            esm_range / 150000.0,
            min(jam_ratio, 5.0) / 5.0,
            previous_power / max_power if max_power > 0 else 0.0,
            sim.remaining_energy_j / sim.radar.energy_budget_j
            if sim.radar.energy_budget_j > 0
            else 0.0,
            min(sim.current_time / sim.scenario.sim_duration, 1.0),
            self._last_pd,
            self._last_pint,
            sim.radar.required_pd,
            sim.exposure.value,
        ]
        return self.observation_space.clip(raw)

    def _build_pomdp_observation(self) -> List[float]:
        """POMDP 观测：全部来自带噪声/延迟/丢测的估计，附 4 维不确定度。

        `hide_interceptor_truth / hide_pint_truth / hide_exposure_truth`
        三个开关由 ObservationModel 内部处理（关闭即等于给真值），
        这里不做任何配置改写。
        """
        sim = self.sim
        assert sim.radar is not None and sim.scenario is not None

        truth = self._truth()
        estimates = self.observation_model.observe(truth, self._obs_step_index)

        self._last_estimates = {k: v.to_dict() for k, v in estimates.items()}
        if self.expose_observation_truth:
            # 仅诊断用：把真值单独放在 truth 字段，供统计观测误差
            for name, estimate in estimates.items():
                if estimate.truth is not None:
                    self._last_estimates[name]["truth"] = float(estimate.truth)
        quality = self.observation_model.observation_quality(estimates)
        self._last_quality = quality

        def value(name: str, default: float = 0.0) -> float:
            est = estimates.get(name)
            return float(est.value) if est is not None else default

        def sigma(name: str, default: float = 0.0) -> float:
            est = estimates.get(name)
            return float(est.sigma) if est is not None else default

        # 智能体"记住"的是它测到的上一步 Pd / Pint，用于下一步观测
        self._last_pd = value("pd_min", self._last_pd)
        self._last_pint = value("pint_eff", self._last_pint)

        max_power = sim.scenario.max_power_w
        previous_power = value("previous_power", 0.0)
        budget = sim.radar.energy_budget_j

        raw = [
            value("target_range") / 20000.0,
            value("target_rcs") / 5.0,
            value("max_range") / 20000.0,
            value("interceptor_range") / 150000.0,
            min(value("jam_ratio"), 5.0) / 5.0,
            previous_power / max_power if max_power > 0 else 0.0,
            value("remaining_energy") / budget if budget > 0 else 0.0,
            min(sim.current_time / sim.scenario.sim_duration, 1.0),
            self._last_pd,
            self._last_pint,
            sim.radar.required_pd,
            value("exposure"),
            # --- 不确定度通道 ---
            quality,
            min(sigma("interceptor_range") / 150000.0, 1.0),
            min(sigma("exposure"), 1.0),
            min(sigma("pint_eff"), 1.0),
        ]
        return self.observation_space.clip(raw)

    # ------------------------------------------------------------------
    # 不确定度 / 观测质量（供不确定性回退策略使用）
    # ------------------------------------------------------------------

    def observation_quality(self) -> float:
        """最近一步的观测质量（0~1）。full 模式恒为 1。"""
        return float(self._last_quality)

    @property
    def last_estimates(self) -> Dict[str, Any]:
        """最近一步的观测量（含估计值与上报标准差）。

        这是智能体**唯一**的状态信息来源；`strategy/belief_policy.py`
        用它构造信念状态，脚本策略不必（也不应）直接读 `env.sim`。
        """
        return self._last_estimates

    def observation_uncertainty(self) -> Dict[str, float]:
        """最近一步各量的上报标准差，便于策略层读取并判断是否回退。"""
        if self.observation_mode != "pomdp":
            return {}
        return {name: float(d.get("sigma", 0.0)) for name, d in self._last_estimates.items()}

    def observation_model_summary(self) -> Dict[str, Any]:
        return {
            "mode": self.observation_mode,
            "history_len": self.history_len,
            "obs_dim": len(self.observation_features),
            **self.observation_model.describe(),
        }

    # ------------------------------------------------------------------
    # 测量层观测（v4.2）
    # ------------------------------------------------------------------

    def _sensor_context(self) -> Dict[str, Dict[str, Any]]:
        """给每个传感器准备观测上下文。

        雷达传感器需要知道**当前**的干扰功率（它测的是噪声基底抬升）；
        ESM 需要知道雷达此刻的发射功率与朝它那个方向的发射增益
        （被动截获强度取决于雷达辐射多强、往哪辐射）。
        这些量由仿真器提供，传感器层本身不实现干扰/天线模型。
        """
        sim = self.sim
        _, interference_w, _, _ = sim.jam_state()
        levels = sim.power_levels_w
        current_power = (
            levels[sim.previous_power_level] if sim.previous_power_level >= 0 else 0.0
        )
        context: Dict[str, Dict[str, Any]] = {}
        controlled_radar_id = str(getattr(sim.radar, "radar_id", ""))
        for sensor in (self.suite.sensors if self.suite is not None else []):
            entry: Dict[str, Any] = {}
            if sensor.sensor_kind == "radar":
                entry["interference_w"] = float(interference_w)
                # --- v4.5 一致性修复（E3）---
                # **受控雷达**的主动传感器必须用本步真正执行的发射功率，
                # 否则会出现"仿真器按新动作记账、传感器仍按初始化配置生成测量"
                # 的双账本：实测档位 0（0.5 W）与档位 10（80 W）下，
                # 主动传感器的配置都是 18 W、测量记录逐项完全相同，
                # 而主仿真的 Pd 从 0.0023 变到 0.993。
                # 时序上是安全的：`LpiPowerEnv.step()` 先 `sim.step(executed)`，
                # 之后才构建观测并调用测量层，因此此刻
                # `sim.previous_power_level` **就是本步执行的档位**。
                # 非受控平台（旁观的第二部雷达）不注入，保持它自己的配置功率。
                if str(getattr(sensor.config, "mounting_id", "")) == controlled_radar_id:
                    entry["tx_power_w"] = float(current_power)
            else:
                # ESM 观测的是雷达辐射：功率 + 该方向上的发射增益
                entry["emitter_power_w"] = float(current_power)
                emitter = sim.scene.maybe_by_id(getattr(sensor.config, "mounting_id", ""))
                target_radar = sim.scene.maybe_by_id(sensor.config.observes_kind == "radar"
                                                     and sim.radar.radar_id or "")
                if target_radar is not None and emitter is not None:
                    gain_db, _ = sim._gain_and_beam_toward(emitter.x, emitter.y)
                    entry["beam_gain_db"] = float(gain_db)
                else:
                    entry["beam_gain_db"] = float(sim.radar.peak_gain_db)
            context[sensor.sensor_id] = entry
        return context

    def _build_measurement_observation(self) -> List[float]:
        """由传感器测量汇聚成定长观测（**不含任何真值**）。

        组成：``[航迹表 max_tracks×10]`` + ``[自状态 4]`` + ``[缺失统计 9]``。

        * 航迹表：本步可用的测量（含沿用值与虚警），按置信度排序取前 K 条；
        * 自状态：自身发射功率、剩余能量、时间、任务要求 —— **本平台精确已知**，
          不是测量，因此与传感器测量分开；
        * 缺失统计：逐原因的比例与观测质量 —— 告诉智能体"它现在看得有多差"。

        ⚠️ 这个函数是整个工程里**唯一**允许为决策算法生产"环境信息"的地方，
        因此它必须只读 `SensorReport.measurements`（候选 ID 与估计值），
        绝不读 `outcomes` 里的 `truth_id`。
        """
        sim = self.sim
        assert sim.radar is not None and sim.scenario is not None

        suite_report = self.suite.observe(sim.scene, sim.current_time, self._sensor_context())
        measurements = suite_report.measurements

        reason_counts = suite_report.reason_counts()
        by_dimension = suite_report.reason_counts_by_dimension()
        # ⚠️ 分母**不能**用真值实体数（`len(r.outcomes)`）：
        # 每个 outcome 对应一条**真值**实体，所以按它归一化的缺失率会
        # 直接暴露"本帧一共有几个目标"——即使一个都没探到也不例外
        # （3 个目标全部视场外 → 三个率都是 1.0，等于告诉算法有 3 个目标）。
        # 这是把真值送进决策输入，属信息边界违规，不是"归一化"。
        # 改用**本平台自己的感知容量**（传感器数 × 跟踪槽位数）作分母：
        # 它完全由平台配置决定，与场景里究竟有多少目标无关。
        n_sensor_scans = sum(1 for r in suite_report.reports if r.updated) or 1
        own_capacity = max(1, n_sensor_scans * self._track_config.max_tracks)
        total_outcomes = own_capacity

        self._last_suite_report = suite_report
        self._last_missing_counts = dict(by_dimension)

        # 驱动跟踪器（只记账，不参与观测向量）
        self._update_tracker(suite_report, sim.current_time)

        # --- 自状态（精确已知）---
        levels = sim.power_levels_w
        max_power = sim.scenario.max_power_w
        previous_power = (
            levels[sim.previous_power_level] if sim.previous_power_level >= 0 else 0.0
        )
        budget = sim.radar.energy_budget_j
        own_scalars = [
            previous_power / max_power if max_power > 0 else 0.0,
            sim.remaining_energy_j / budget if budget > 0 else 0.0,
            min(sim.current_time / sim.scenario.sim_duration, 1.0),
            sim.radar.required_pd,
        ]

        # --- 参考航向：主雷达机头，使"前方/后方"在观测里有物理含义 ---
        reference_heading = float(sim.radar.attitude.heading_deg)

        fused = fuse_measurements(
            measurements,
            config=self._track_config,
            extra_scalars=own_scalars,
            reason_counts=reason_counts,
            reason_by_dimension=by_dimension,
            reference_heading_deg=reference_heading,
            n_fresh=len(suite_report.fresh_measurements),
        )

        # --- 缺失统计（按原因比例，而不是笼统一个丢测率）---
        def rate(reason: str) -> float:
            return float(reason_counts.get(reason, 0)) / total_outcomes

        summary = [
            min(len(measurements) / self._track_config.max_tracks, 1.0),
            min(len(suite_report.fresh_measurements) / self._track_config.max_tracks, 1.0),
            fused.observation_quality,
            rate("out_of_fov"),
            rate("beyond_range"),
            rate("occluded"),
            rate("not_updated"),
            rate("missed_detection"),
            rate("sensor_unavailable"),
        ]

        self._last_fused = fused
        self._last_quality = float(fused.observation_quality)

        # 把测量换成与 POMDP 模式同形的估计字典，使脚本策略（信念桥接）
        # 在测量模式下**无需改动**即可只用观测做决策。
        self._last_estimates = self._estimates_from_measurements(measurements)

        return self.observation_space.clip(list(fused.vector) + summary)

    def _estimates_from_measurements(
        self, measurements: List[Any]
    ) -> Dict[str, Any]:
        """把测量折算成 `{名称: {value, sigma, observed, stale}}` 形式的估计。

        这样做的用途：`strategy/belief_policy.py` 的信念桥接可以直接复用，
        让规则/前瞻策略在测量模式下也只依据观测决策，而不必知道
        "测量"与"全局噪声模型"的区别。

        ⚠️ 这里只读测量的估计字段，不读 `truth_*`。
        """
        sim = self.sim
        assert sim.radar is not None
        radar_sensor_ids = {
            s.sensor_id for s in (self.suite.sensors if self.suite else [])
            if s.sensor_kind == "radar"
        }
        ranged = [
            m for m in measurements
            if m.sensor_id in radar_sensor_ids and m.range_m is not None
        ]
        # 取置信度最高的一条雷达测量作为主目标的估计
        best = max(ranged, key=lambda m: m.confidence, default=None)

        def record(value: float, sigma: float, observed: bool = True,
                   stale: bool = False) -> Dict[str, Any]:
            return {"value": float(value), "sigma": float(sigma),
                    "observed": bool(observed), "stale": bool(stale)}

        estimates: Dict[str, Any] = {}
        if best is not None:
            estimates["target_range"] = record(
                best.range_m, float(best.std_range_m or 0.0), True, not best.is_fresh
            )
            if best.rcs_est_m2 is not None:
                estimates["target_rcs"] = record(best.rcs_est_m2, 0.0)
            estimates["jam_ratio"] = record(float(best.jam_ratio_est or 0.0), 0.0)
        else:
            # 雷达一无所获：给一个"未观测"的占位，并放大不确定度
            estimates["target_range"] = {
                "value": 0.0, "sigma": float(self._track_config.range_scale_m),
                "observed": False, "stale": False,
            }
            estimates["target_rcs"] = {"value": 0.0, "sigma": 1.0,
                                       "observed": False, "stale": False}
            estimates["jam_ratio"] = {"value": 0.0, "sigma": 1.0,
                                      "observed": False, "stale": False}

        # 本平台内部量（不是传感器测量）：能量与暴露由雷达自己递推
        estimates["remaining_energy"] = record(sim.remaining_energy_j, 0.0)
        estimates["exposure"] = record(float(sim.exposure.value), 0.0)
        estimates["pd_min"] = record(self._last_pd, 0.0)
        estimates["pint_eff"] = record(self._last_pint, 0.0)
        estimates["previous_power"] = record(
            sim.scenario.power_levels_w[sim.previous_power_level]
            if sim.previous_power_level >= 0 else 0.0,
            0.0,
        )
        # 侦察机位置：雷达**观测不到**（它没有对 ESM 的定位手段），
        # 因此不给该键 —— 信念桥接会退回默认值，如实反映"不知道"。
        return estimates

    def measurement_records(self, include_truth: bool | None = None) -> List[Dict[str, Any]]:
        """本步测量记录（供导出与诊断）。

        `include_truth` 默认取 `self.expose_measurement_truth`；
        显式传 True 才会带真值与误差列（评测通道）。
        """
        if self._last_suite_report is None:
            return []
        flag = self.expose_measurement_truth if include_truth is None else bool(include_truth)
        return [m.to_dict(include_truth=flag) for m in self._last_suite_report.measurements]

    def suite_report(self) -> Any:
        """最近一步的完整传感器报告（含逐实体的「没有数据」原因）。"""
        return self._last_suite_report

    @property
    def fused_observation(self) -> Any:
        """最近一步的融合结果（定长向量 + 槽位说明 + 缺失统计）。"""
        return self._last_fused

    # ------------------------------------------------------------------
    # v4.5：跟踪器 / 通信总线 驱动与证据访问器
    # ------------------------------------------------------------------

    def attach_comm_bus(self, comm_bus: Any) -> None:
        """挂上通信总线后，本平台会消费**已到达**的远端测量。

        只消费已到达的（`CommBus.consume` 一次投递），
        因此"延迟/丢包/过期"会真实影响融合输入。
        """
        self.comm_bus = comm_bus

    def _local_sensor_ids(self) -> set:
        if self.suite is None:
            return set()
        return {sc.sensor_id for sc in self.suite.sensors
                if sc.config.mounting_id == self.platform_id}

    def _update_tracker(self, suite_report: Any, now: float) -> None:
        """把本步的本平台测量（+ 已到达的远端测量）送进跟踪器。"""
        if self.tracker is None:
            return
        own_ids = self._local_sensor_ids()
        local = [m for sr in suite_report.reports if sr.sensor_id in own_ids
                 for m in sr.detections + sr.held]

        shared: List[Any] = []
        arrived = 0
        if self.comm_bus is not None:
            for message in self.comm_bus.consume(self.platform_id, now):
                arrived += 1
                payload = dict(message.payload)
                shared.append(_SharedMeasurement(payload, message.msg_id,
                                                 message.src_platform_id))

        # 远端测量发布：本平台把自己**新产生**的测量发出去
        if self.comm_bus is not None and self.comm_bus.sharing_enabled:
            fresh_by_sensor: Dict[str, List[Any]] = {}
            for sr in suite_report.reports:
                if sr.sensor_id in own_ids and sr.detections:
                    fresh_by_sensor[sr.sensor_id] = sr.detections
            for sensor_id, detections in fresh_by_sensor.items():
                self.comm_bus.publish(self.platform_id, sensor_id, detections, now=now)

        sensor_positions = {
            sc.sensor_id: self.sim.scene.by_id(sc.config.mounting_id).position
            for sc in self.suite.sensors
        }
        snapshot = self.tracker.update(
            local + shared, now, sensor_positions,
            remote_measurement_flags=[False] * len(local) + [True] * len(shared),
        )
        self._fusion_snapshot = snapshot
        self._comm_arrived_this_step = arrived
        self._comm_used_this_step = snapshot.n_remote_measurements
        self._comm_rejected_this_step = (snapshot.n_kind_rejected
                                        + snapshot.n_stale_rejected)

    @property
    def tracks(self) -> List[Any]:
        """当前航迹（跟踪器内部对象；结构化视图见 `fusion_state_dict()`）。"""
        return list(self.tracker.tracks) if self.tracker is not None else []

    @property
    def fusion_snapshot(self) -> Any:
        return self._fusion_snapshot

    # --- 结构化证据（供 ai/context.py 使用，**不含真值**）---

    def measurement_state_dict(self) -> Dict[str, Any]:
        report = self._last_suite_report
        reason_counts: Dict[str, int] = {}
        reason_rates: Dict[str, float] = {}
        n_measurements = n_fresh = n_held = n_false = 0
        candidates: List[Dict[str, Any]] = []
        if report is not None:
            reason_counts = dict(report.reason_counts())
            total = sum(reason_counts.values()) or 1
            reason_rates = {k: v / total for k, v in reason_counts.items()}
            n_measurements = len(report.measurements)
            n_fresh = len(report.fresh_measurements)
            n_held = len(report.held)
            n_false = len(report.false_alarms)
            for m in report.measurements:
                candidates.append({
                    "candidate_id": m.candidate_id,
                    "sensor_id": m.sensor_id,
                    "sensor_kind": m.sensor_kind,
                    "is_fresh": bool(m.is_fresh),
                    "age_s": round(float(m.age_s), 4),
                    "range_m": None if m.range_m is None else round(m.range_m, 3),
                    "azimuth_deg": None if m.azimuth_deg is None
                    else round(m.azimuth_deg, 4),
                    "elevation_deg": None if m.elevation_deg is None
                    else round(m.elevation_deg, 4),
                    "std_range_m": None if m.std_range_m is None
                    else round(m.std_range_m, 3),
                    "confidence": round(float(m.confidence), 6),
                })
        sensors: List[Dict[str, Any]] = []
        if self.suite is not None:
            for sc in self.suite.sensors:
                sensors.append({
                    "sensor_id": sc.sensor_id,
                    "sensor_kind": sc.sensor_kind,
                    "mounting_id": sc.config.mounting_id,
                    "is_own_platform": sc.config.mounting_id == self.platform_id,
                    "available": bool(sc.config.available),
                    "max_range_m": float(sc.config.max_range_m),
                    "az_fov_deg": float(sc.config.az_fov_deg),
                    "update_period_s": float(sc.config.update_period_s),
                })
        return {
            "mode": self.observation_mode,
            "n_measurements": n_measurements,
            "n_fresh": n_fresh,
            "n_held": n_held,
            "n_false_alarms": n_false,
            "observation_quality": round(float(self._last_quality), 6),
            "reason_counts": reason_counts,
            "reason_rates": reason_rates,
            "sensors": sensors,
            "candidates": candidates,
        }

    def communication_state_dict(self) -> Dict[str, Any]:
        if self.comm_bus is None:
            return {"policy": "none", "policy_cn": "未接入通信总线", "n_links": 0,
                    "n_messages_sent": 0, "n_delivered": 0, "n_dropped": 0,
                    "delivery_rate": 0.0, "drop_reasons": {},
                    "latency_mean_s": 0.0, "latency_p95_s": 0.0,
                    "n_in_flight": 0, "n_arrived_stale": self._comm_rejected_this_step}
        stats = self.comm_bus.statistics()
        return {
            "policy": stats["policy"],
            "policy_cn": SHARE_POLICY_CN.get(stats["policy"], stats["policy"]),
            "n_links": stats["n_links"],
            "n_messages_sent": stats["n_messages"],
            "n_delivered": stats["n_delivered"],
            "n_dropped": stats["n_dropped"],
            "delivery_rate": stats["delivery_rate"],
            "drop_reasons": stats["drop_reasons"],
            "latency_mean_s": stats["latency_mean_s"],
            "latency_p95_s": stats["latency_p95_s"],
            # 只报**数量**，在途消息的内容不得读取
            "n_in_flight": self.comm_bus.pending_for(self.platform_id),
            "n_arrived_stale": self._comm_rejected_this_step,
        }

    def fusion_state_dict(self) -> Dict[str, Any]:
        if self.tracker is None:
            return {"enabled": False, "platform_id": self.platform_id}
        now = self.sim.current_time
        tracks: List[Dict[str, Any]] = []
        n_coasting = n_confirmed = n_tentative = 0
        freshness_sum = 0.0
        for track in self.tracker.tracks:
            if track.status == "coasting":
                n_coasting += 1
            elif track.status == "confirmed":
                n_confirmed += 1
            elif track.status == "tentative":
                n_tentative += 1
            fresh = track.freshness(now)
            freshness_sum += fresh
            recent = [src.to_dict() for src in track.sources[-5:]]
            tracks.append({
                "track_id": track.track_id,
                "status": track.status,
                "x": track.position.x, "y": track.position.y, "z": track.position.z,
                "vx": track.velocity.x, "vy": track.velocity.y,
                "vz": track.velocity.z,
                "sigma_x": track.sigma_position.x,
                "sigma_y": track.sigma_position.y,
                "sigma_z": track.sigma_position.z,
                "sigma_vx": track.filter.velocity_sigma().x if track.filter else 0.0,
                "sigma_vy": track.filter.velocity_sigma().y if track.filter else 0.0,
                "sigma_vz": track.filter.velocity_sigma().z if track.filter else 0.0,
                "hits": track.hits, "misses": track.misses,
                "local_updates": track.local_updates,
                "remote_updates": track.remote_updates,
                "is_local_origin": track.is_local_origin,
                "freshness": fresh,
                "measurement_age_s": track.measurement_age_s(now),
                "n_sources": len(track.sources),
                "platforms": sorted(set(track.platforms)),
                "source_sensors": sorted({s.sensor_id for s in track.sources}),
                "recent_sources": recent,
                "has_remote_contribution": track.remote_updates > 0,
            })
        n = len(self.tracker.tracks)
        snapshot = self._fusion_snapshot
        return {
            "enabled": True,
            "platform_id": self.platform_id,
            "n_tracks": n,
            "n_confirmed": n_confirmed,
            "n_coasting": n_coasting,
            "n_tentative": n_tentative,
            "freshness_mean": (freshness_sum / n) if n else 0.0,
            "n_local_measurements": getattr(snapshot, "n_local_measurements", 0),
            "n_remote_measurements": getattr(snapshot, "n_remote_measurements", 0),
            "n_kind_rejected": getattr(snapshot, "n_kind_rejected", 0),
            "n_stale_rejected": getattr(snapshot, "n_stale_rejected", 0),
            "tracks_initiated": self.tracker.stats["initiated"],
            "tracks_dropped": self.tracker.stats["dropped"],
            "gate_rejected_total": self.tracker.stats["gate_rejected"],
            "tracks": tracks,
        }

    def cooperation_state_dict(self) -> Dict[str, Any]:
        fusion = self.fusion_state_dict()
        if not fusion.get("enabled"):
            return {"sharing_enabled": False, "policy": "none",
                    "notes": ["未启用跟踪器，无法给出协同状态"]}
        comm = self.communication_state_dict()
        with_remote = sum(1 for t in fusion["tracks"] if t["has_remote_contribution"])
        local_only = len(fusion["tracks"]) - with_remote
        remote_only = sum(
            1 for t in fusion["tracks"]
            if t["has_remote_contribution"] and t["local_updates"] == 0
        )
        arrived = self._comm_arrived_this_step
        used = self._comm_used_this_step
        notes: List[str] = []
        if not comm.get("n_links"):
            notes.append("未共享：本平台只使用自己的测量")
        if remote_only:
            notes.append(f"{remote_only} 条航迹**仅靠远端测量**维持（本地无贡献）")
        if comm.get("drop_reasons"):
            notes.append(f"消息丢弃原因：{comm['drop_reasons']}")
        return {
            "sharing_enabled": bool(comm.get("n_links")),
            "policy": comm.get("policy", "none"),
            "n_tracks_with_remote": with_remote,
            "n_tracks_local_only": local_only,
            "remote_measurements_arrived": arrived,
            "remote_measurements_used": used,
            "remote_measurements_rejected": self._comm_rejected_this_step,
            "remote_utilization": (used / arrived) if arrived else 0.0,
            "tracks_supported_remotely_only": remote_only,
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def required_snr_db(self) -> float:
        """当前任务要求对应的探测 SNR 门限，便于与规则策略对齐。"""
        radar = self.sim.radar
        assert radar is not None
        return snr_db_for_prob(radar.required_pd, radar.snr50_db, radar.pd_slope_db)
