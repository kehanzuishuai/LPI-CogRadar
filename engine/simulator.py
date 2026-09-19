"""低截获雷达智能功率调控仿真内核（第一版）。

沿用了原通信抗干扰仿真器的使用方式——「配置 -> 场景 -> 仿真 -> 策略 -> 指标」：

    sim = Simulator("config/radar_scenario_v1.json")
    sim.load_config()
    results = sim.run(RuleBasedPowerPolicy())      # List[StepResult]

与原版的对应关系
----------------
    原（通信抗干扰）                  现（低截获雷达功率调控）
    -----------------------------    ------------------------------------------
    build_links()                     preview()：一步的探测 / 截获 / 干扰全量评估
    calc_base_quality()/calc_disturb  engine/equations.py 的雷达方程与侦察方程
    find_path()/路由                  （删除：与功率调控无关）
    update_nodes()                    _advance()：雷达 / 目标 / 侦察机 / 干扰机运动
    run_baseline/with_disturbance/    run(policy) + 三个便捷入口
    run_with_strategy

关键设计：preview() 是**纯函数**（不修改任何状态），策略可以安全地
「试算」全部 11 个功率档位再决定，这正是规则功率控制基线所需要的。
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from engine.equations import (
    angular_offset_deg,
    db2lin,
    lin2db,
    logistic_prob,
    one_way_power_w,
    radar_echo_power_w,
    sinr_db,
    snr_db,
    snr_db_for_prob,
    thermal_noise_w,
)
from models import (
    DEFAULT_REWARD_WEIGHTS,
    EnemyInterceptor,
    ExposureTracker,
    InterceptionRecord,
    Jammer,
    Radar,
    Scenario,
    StepResult,
    Target,
    TargetDetection,
    composite_reward,
    terminal_energy_penalty,
)
from engine.geometry import GeometricRelation  # noqa: F401  (对外契约的类型标注)
from engine.scene import Scene


class InfeasibleActionError(ValueError):
    """请求的功率档位所需能量超过了当前剩余能量。

    能量是硬约束：**动作在执行前就要被否决**，而不是执行完再发现超支。
    仿真器对不可行动作一律抛错（宁可炸得响，也不要静默把预算花超）；
    环境层（`engine.env.LpiPowerEnv`）则把它裁剪到当前可执行的最高档并记录，
    以保证 RL 训练不会因为一次越界动作直接崩掉。
    """


def _instantiate(cls: Any, data: Dict[str, Any], section: str) -> Any:
    """按配置字典构造实体，并对**未知键**给出可诊断的报错。

    为什么要专门处理未知键：多平台配置格式还在演进，
    配置里写错一个字段名（例如 `velocity_z` 写成 `vel_z`）如果被静默忽略，
    表现出的现象是"这个平台的运动/姿态没生效"，极难定位。
    因此这里显式列出未知键并抛出，而不是让 dataclass 自己报
    `unexpected keyword argument`（那句话不会告诉你哪个 section 出的问题）。
    """
    if not isinstance(data, dict):
        raise ValueError(f"配置段 {section!r} 的每一项都必须是对象，收到 {type(data).__name__}")

    known = getattr(cls, "__dataclass_fields__", {})
    unknown = [k for k in data if k not in known]
    if unknown:
        raise ValueError(
            f"配置段 {section!r} 出现未知字段 {unknown}；"
            f"{cls.__name__} 支持的字段为 {sorted(known)}"
        )
    return cls(**data)


@dataclass
class StepEvaluation:
    """一步的物理评估结果（不含能量与收益，不修改状态）。

    由 preview() 产生，step() 在此基础上补充能量与收益后生成 StepResult。
    """

    tx_power_w: float
    detections: List[TargetDetection] = field(default_factory=list)
    pd_min: float = 0.0
    snr_radar_db_min: float = float("-inf")
    required_snr_db: float = 0.0
    task_satisfied: bool = False
    detection_term: float = 0.0

    interceptions: List[InterceptionRecord] = field(default_factory=list)
    intercept_prob: float = 0.0  # Pint_eff（含累计暴露）
    intercept_prob_instant: float = 0.0  # Pint_inst
    intercept_snr_db: float = float("-inf")

    exposure: float = 0.0  # 本步开始时的暴露量
    exposure_next: float = 0.0  # 本步结束后的暴露量

    jammer_active: bool = False
    interference_power_w: float = 0.0
    noise_power_w: float = 0.0
    jam_noise_ratio: float = 0.0


class Simulator:
    """单雷达 / 多目标 / 多侦察机 / 多干扰源的低截获功率调控仿真内核。"""

    # ------------------------------------------------------------------
    # 构造与配置
    # ------------------------------------------------------------------

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self._raw_config: Dict[str, Any] | None = None

        self.scenario: Scenario | None = None
        self.radar: Radar | None = None
        self.targets: List[Target] = []
        self.interceptors: List[EnemyInterceptor] = []
        self.jammers: List[Jammer] = []
        self.exposure: ExposureTracker = ExposureTracker()

        self.rng: random.Random = random.Random(42)

        # --- episode 状态 ---
        self.step_index: int = 0
        self.cumulative_energy_j: float = 0.0
        self.episode_return: float = 0.0
        self.previous_power_level: int = -1
        self.results: List[StepResult] = []
        self.terminated_by_energy: bool = False

    def load_config(self) -> None:
        """读取 JSON 配置并建立场景对象。"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "radar" not in data and "radars" not in data:
            raise ValueError(
                f"配置文件 {self.config_path} 不是低截获雷达场景："
                "既没有 'radar'（单雷达）也没有 'radars'（多雷达列表）段。\n"
                "单雷达用 config/radar_scenario_v1.json，"
                "多平台用 config/multi_platform_scenario.json"
                "（旧的通信抗干扰场景 config/scenario_v1.json 已不再被主流程读取）。"
            )

        self._raw_config = data
        self._build_from_config(data)

    def _build_from_config(self, data: Dict[str, Any]) -> None:
        """由配置字典构建场景对象并复位 episode 状态（reset 与 load 共用）。"""
        scenario = Scenario(
            scenario_name=data["scenario_name"],
            sim_duration=float(data["sim_duration"]),
            time_step=float(data["time_step"]),
            random_seed=int(data.get("random_seed", 42)),
            description=data.get("description", ""),
            power_levels_w=[float(v) for v in data["power_levels_w"]],
            fixed_power_level=int(data.get("fixed_power_level", -1)),
            lpi_pint_threshold=float(data.get("lpi_pint_threshold", 0.5)),
            exposure_decay=float(data.get("exposure_decay", 0.9)),
            exposure_gain=float(data.get("exposure_gain", 0.2)),
            reward_weights=dict(data.get("reward_weights", DEFAULT_REWARD_WEIGHTS)),
        )
        radar_section = data.get("radar")
        radars_section = data.get("radars")
        if radars_section:
            # 多平台格式：radars 是列表，首部为主雷达（物理评估与功率控制只针对它）。
            # 多雷达**协同决策**不在本阶段范围内。
            radars = [_instantiate(Radar, item, "radars") for item in radars_section]
        elif radar_section:
            radars = [_instantiate(Radar, radar_section, "radar")]
        else:
            raise ValueError("配置必须提供 'radar'（单雷达）或 'radars'（多雷达列表）")

        targets = [_instantiate(Target, t, "targets") for t in data.get("targets", [])]
        interceptors = [
            _instantiate(EnemyInterceptor, e, "interceptors")
            for e in data.get("interceptors", [])
        ]
        jammers = [_instantiate(Jammer, j, "jammers") for j in data.get("jammers", [])]
        radar = radars[0]

        if not targets:
            raise ValueError("场景至少需要一个 target，否则没有探测任务可言")
        if not interceptors:
            raise ValueError("场景至少需要一个 interceptor，否则无法评估低截获性能")

        self.scenario = scenario
        self.radar = radar
        self.radars = radars
        self.targets = targets
        self.interceptors = interceptors
        self.jammers = jammers
        # v4.1：统一实体注册表。它与上面四个列表共享**同一批对象**（不是副本），
        # 因此旧代码与新的几何查询看到的状态永远一致。
        self.scene = Scene(
            radars=radars,
            targets=targets,
            interceptors=interceptors,
            jammers=jammers,
            name=str(data.get("scenario_name", "")),
        )
        self.exposure = ExposureTracker(
            decay=scenario.exposure_decay, gain=scenario.exposure_gain
        )

        # 干扰强度起伏序列：同一种子下完全可复现
        self.rng = random.Random(scenario.random_seed)
        for jammer in self.jammers:
            jammer.prepare(scenario.num_steps, self.rng)

        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.step_index = 0
        self.cumulative_energy_j = 0.0
        self.episode_return = 0.0
        self.previous_power_level = -1
        self.results = []
        self.terminated_by_energy = False
        self.exposure.reset()

    def apply_overrides(
        self,
        reward_weights: Dict[str, float] | None = None,
        energy_budget_j: float | None = None,
        terminate_on_energy_exhausted: bool | None = None,
        power_levels_w: List[float] | None = None,
        adaptive_jammer: bool | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """覆盖部分配置并立即重建场景。

        关键点：覆盖项会**写回 load_config 缓存的原始配置字典**，
        否则下一次 reset()（run() 内部会调用）按缓存重建时会把覆盖悄悄抹掉
        —— 这正是「传了 reward_weights 却毫无效果」这类隐蔽 bug 的成因。

        这里只开放「任务/评估/对手模式层面」的参数（奖励权重、能量预算、终止开关、
        动作档位、干扰机模式），刻意**不允许**修改雷达方程、侦察模型等物理参数，
        保证任何对比实验都建立在同一套物理模型之上。

        adaptive_jammer：True = 把干扰机切换为规则自适应智能干扰机；
        False = 回到固定时间窗模式（默认，保证旧实验逐位可复现）。
        """
        if self._raw_config is None:
            self.load_config()

        data: Dict[str, Any] = copy.deepcopy(self._raw_config)  # type: ignore[arg-type]

        if reward_weights is not None:
            merged = dict(data.get("reward_weights", DEFAULT_REWARD_WEIGHTS))
            merged.update(reward_weights)
            data["reward_weights"] = merged
        if power_levels_w is not None:
            data["power_levels_w"] = [float(v) for v in power_levels_w]

        if adaptive_jammer is not None:
            mode = "adaptive" if adaptive_jammer else "fixed"
            jammers = [dict(j) for j in data.get("jammers", [])]
            for jammer in jammers:
                jammer["jammer_mode"] = mode
            data["jammers"] = jammers

        radar_section = dict(data.get("radar", {}))
        if energy_budget_j is not None:
            radar_section["energy_budget_j"] = float(energy_budget_j)
        if terminate_on_energy_exhausted is not None:
            radar_section["terminate_on_energy_exhausted"] = bool(
                terminate_on_energy_exhausted
            )
        data["radar"] = radar_section

        if extra:
            data.update(extra)

        self._raw_config = data
        self._build_from_config(data)

    def reset(self, seed: int | None = None) -> None:
        """复位到 episode 起点。

        直接从 load_config 时缓存的配置字典重建场景对象，
        因此位置、能量、干扰起伏序列都会被精确还原（等价于重新读文件，
        但不产生磁盘 I/O）。

        seed 不为 None 时覆盖配置里的 random_seed，并写回缓存配置，
        使后续不带参数的 reset 保持一致。训练时需要按 episode 换种子
        以获得多样的干扰起伏序列，评测时则固定为场景种子。
        """
        if self._raw_config is None:
            self.load_config()
            if seed is None:
                return

        if seed is not None:
            self._raw_config = {**self._raw_config, "random_seed": int(seed)}  # type: ignore[dict-item]

        self._build_from_config(self._raw_config)  # type: ignore[arg-type]

    def _ensure_ready(self) -> None:
        if self.scenario is None or self.radar is None:
            raise RuntimeError("请先调用 load_config() 再使用仿真器")

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def power_levels_w(self) -> List[float]:
        self._ensure_ready()
        return self.scenario.power_levels_w  # type: ignore[union-attr]

    @property
    def num_levels(self) -> int:
        return self.scenario.num_levels  # type: ignore[union-attr]

    @property
    def current_time(self) -> float:
        return self.scenario.time_at(self.step_index)  # type: ignore[union-attr]

    @property
    def is_done(self) -> bool:
        """episode 是否结束。

        两个终止条件（第二版新增能量约束）：
        1. 到达场景时长上限（时间截断）；
        2. **剩余能量已经买不起任何一档功率**（连最低档都发不出去）——硬约束。
        放在仿真器里而不是环境里，是为了让 Simulator.run() 与 LpiPowerEnv.step()
        走完全一致的终止逻辑。
        """
        self._ensure_ready()
        if self.step_index >= self.scenario.num_steps:  # type: ignore[union-attr]
            return True
        if self.radar.terminate_on_energy_exhausted and self.energy_exhausted:  # type: ignore[union-attr]
            return True
        return False

    # ------------------------------------------------------------------
    # 能量硬约束：动作可行性
    # ------------------------------------------------------------------

    # 浮点容差：Pt·Δt 与剩余能量比较时允许的误差
    ENERGY_EPS = 1e-9

    def step_energy_j(self, power_level: int) -> float:
        """执行某档位一步所需的能量 Pt·Δt。"""
        assert self.scenario is not None
        return self.scenario.power_levels_w[power_level] * self.scenario.time_step

    @property
    def min_step_energy_j(self) -> float:
        """最低档一步所需的能量（能量耗尽判据用）。"""
        assert self.scenario is not None
        return self.scenario.power_levels_w[0] * self.scenario.time_step

    def is_level_feasible(self, power_level: int) -> bool:
        """该档位所需能量是否不超过剩余能量。"""
        return self.step_energy_j(power_level) <= self.remaining_energy_j + self.ENERGY_EPS

    def feasible_levels(self) -> List[int]:
        """当前可执行的功率档位（功率档位升序，因此必然是一段前缀）。"""
        assert self.scenario is not None
        remaining = self.remaining_energy_j
        dt = self.scenario.time_step
        feasible: List[int] = []
        for level, pt_w in enumerate(self.scenario.power_levels_w):
            if pt_w * dt <= remaining + self.ENERGY_EPS:
                feasible.append(level)
            else:
                break  # 升序，后面只会更贵
        return feasible

    def action_mask(self) -> List[bool]:
        """动作可行性掩码，供 RL 智能体做 masked argmax / 采样。"""
        mask = [False] * self.num_levels
        for level in self.feasible_levels():
            mask[level] = True
        return mask

    def max_feasible_level(self) -> int | None:
        """当前能执行的最高档位；连最低档都买不起时返回 None（episode 应当结束）。"""
        feasible = self.feasible_levels()
        return feasible[-1] if feasible else None

    def clip_to_feasible(self, power_level: int) -> int | None:
        """把动作裁剪到当前可行的最高档；无可行档时返回 None。"""
        if self.is_level_feasible(power_level):
            return power_level
        return self.max_feasible_level()

    @property
    def energy_exhausted(self) -> bool:
        """剩余能量已经买不起**任何**一档功率（不是「累计能耗达到预算」）。"""
        return self.max_feasible_level() is None

    @property
    def remaining_energy_j(self) -> float:
        """剩余能量 E(t)。按 E(t+1) = E(t) − Pt·Δt 递推，非负。"""
        budget = self.radar.energy_budget_j  # type: ignore[union-attr]
        return max(0.0, budget - self.cumulative_energy_j)

    @property
    def remaining_steps(self) -> int:
        """距离任务时长上限还剩多少步（用于能量耗尽的终端惩罚）。"""
        return max(0, self.scenario.num_steps - self.step_index)  # type: ignore[union-attr]

    @property
    def energy_fraction(self) -> float:
        budget = self.radar.energy_budget_j  # type: ignore[union-attr]
        return self.cumulative_energy_j / budget if budget > 0 else 0.0

    def active_targets(self) -> List[Target]:
        return [t for t in self.targets if t.is_active]

    # ------------------------------------------------------------------
    # 几何辅助
    # ------------------------------------------------------------------

    def _primary_target(self) -> Target | None:
        """主目标：波束指向它，也是判断侦察机在主瓣/旁瓣的参考方向。"""
        targets = self.active_targets()
        if not targets:
            return None
        # 取最近的目标作为主目标（工程上通常优先跟踪威胁最大的近距目标）
        return min(targets, key=lambda t: t.range_to(self.radar.x, self.radar.y))  # type: ignore[union-attr]

    def _beam_reference_point(self) -> tuple[float, float]:
        assert self.radar is not None
        target = self._primary_target()
        if target is None:
            return (self.radar.x + 1.0, self.radar.y)
        return (target.x, target.y)

    def _gain_and_beam_toward(self, x: float, y: float) -> tuple[float, str]:
        """雷达朝某方向的发射增益，以及该方向属于主瓣还是旁瓣。"""
        assert self.radar is not None
        ref_x, ref_y = self._beam_reference_point()
        offset = angular_offset_deg(self.radar.x, self.radar.y, ref_x, ref_y, x, y)

        if offset <= self.radar.main_beam_width_deg / 2.0:
            return self.radar.peak_gain_db, "main"
        return self.radar.sidelobe_gain_db, "sidelobe"

    # ------------------------------------------------------------------
    # 核心：一步的物理评估（纯函数，不修改状态）
    # ------------------------------------------------------------------

    def jam_state(
        self, step_index: int | None = None
    ) -> tuple[float, float, bool, float]:
        """当前一步的干扰态势，与发射功率无关（干扰机独立辐射）。

        返回 (雷达热噪声功率 W, 折算到雷达输入端的等效干扰功率 W,
              是否有干扰机在工作, J/N)。
        提取成独立方法后，观测向量可以在不做完整探测评估的前提下拿到 J/N。
        """
        self._ensure_ready()
        assert self.radar is not None and self.scenario is not None

        radar = self.radar
        index = self.step_index if step_index is None else step_index
        current_time = self.scenario.time_at(index)

        noise_w = thermal_noise_w(
            radar.bandwidth_hz, radar.noise_figure_db, radar.temperature_k
        )
        interference_w = 0.0
        jammer_active = False

        for jammer in self.jammers:
            if not jammer.is_jamming(current_time):
                continue
            factor = jammer.factor_at(index)
            if factor <= 0.0:
                # 自适应干扰机在 NO_JAM / 间歇"关"的步不贡献干扰
                continue
            jammer_active = True
            rx_gain_db, _ = self._gain_and_beam_toward(jammer.x, jammer.y)
            received_j = one_way_power_w(
                pt_w=jammer.peak_power_w * factor,
                tx_gain_db=jammer.gain_db,
                rx_gain_db=rx_gain_db,
                wavelength_m=radar.wavelength_m,
                range_m=jammer.range_to(radar.x, radar.y),
                system_loss_db=jammer.system_loss_db,
            )
            # 雷达抗干扰等效处理增益（脉压 / 相参积累 / 旁瓣对消合并建模）
            interference_w += received_j * db2lin(-jammer.suppression_db)

        jam_noise_ratio = interference_w / noise_w if noise_w > 0 else 0.0
        return noise_w, interference_w, jammer_active, jam_noise_ratio

    def preview(self, pt_w: float, step_index: int | None = None) -> StepEvaluation:
        """给定发射功率，计算该步的探测与截获结果，**不修改任何状态**。

        策略可以反复调用它来试算各档功率（规则功率控制基线就依赖这一点）。
        """
        self._ensure_ready()
        assert self.radar is not None and self.scenario is not None

        index = self.step_index if step_index is None else step_index
        radar = self.radar

        # ---------- 噪声与干扰 ----------
        noise_w, interference_w, jammer_active, jam_noise_ratio = self.jam_state(index)

        # ---------- 探测 ----------
        required_snr_db = snr_db_for_prob(
            radar.required_pd, radar.snr50_db, radar.pd_slope_db
        )
        detections: List[TargetDetection] = []

        for target in self.targets:
            if not target.is_active:
                continue
            rng_m = target.range_to(radar.x, radar.y)
            echo_w = radar_echo_power_w(
                pt_w=pt_w,
                gain_db=radar.peak_gain_db,
                wavelength_m=radar.wavelength_m,
                rcs_m2=target.rcs_m2,
                range_m=rng_m,
                system_loss_db=radar.system_loss_db,
            )
            target_snr_db = sinr_db(echo_w, noise_w, interference_w)
            pd = logistic_prob(target_snr_db, radar.snr50_db, radar.pd_slope_db)
            detections.append(
                TargetDetection(
                    target_id=target.target_id,
                    range_m=rng_m,
                    rcs_m2=target.rcs_m2,
                    echo_power_w=echo_w,
                    snr_db=target_snr_db,
                    pd=pd,
                    satisfied=pd >= radar.required_pd,
                )
            )

        # ---------- 截获 ----------
        interceptions: List[InterceptionRecord] = []
        for esm in self.interceptors:
            if not esm.is_active:
                continue
            rng_m = esm.range_to(radar.x, radar.y)
            tx_gain_db, beam = self._gain_and_beam_toward(esm.x, esm.y)
            received_w = one_way_power_w(
                pt_w=pt_w,
                tx_gain_db=tx_gain_db,
                rx_gain_db=esm.gain_db,
                wavelength_m=radar.wavelength_m,
                range_m=rng_m,
                system_loss_db=esm.system_loss_db,
            )
            esm_noise_w = thermal_noise_w(
                esm.bandwidth_hz, esm.noise_figure_db, esm.temperature_k
            )
            esm_snr_db = snr_db(received_w, esm_noise_w)
            pint = logistic_prob(esm_snr_db, esm.snr50_db, esm.pint_slope_db)
            interceptions.append(
                InterceptionRecord(
                    interceptor_id=esm.interceptor_id,
                    range_m=rng_m,
                    beam=beam,
                    tx_gain_db=tx_gain_db,
                    received_power_w=received_w,
                    noise_power_w=esm_noise_w,
                    snr_db=esm_snr_db,
                    pint=pint,
                )
            )

        # 取威胁最大的侦察机作为该步的截获结果
        worst = max(interceptions, key=lambda r: r.pint) if interceptions else None
        intercept_prob_instant = worst.pint if worst else 0.0
        intercept_snr_db = worst.snr_db if worst else float("-inf")

        # ---------- 累计暴露 -> 有效截获概率 ----------
        # 因果顺序：本步的 Pint_eff 用**本步开始前**已累积的暴露量（exposure.value），
        # 而本步动作产生的新证据在 step() 里才写入 exposure，
        # 因此动作只影响未来、不改变本步——这正是时序耦合的来源。
        exposure_before = self.exposure.value
        intercept_prob = ExposureTracker.effective_pint(
            intercept_prob_instant, exposure_before
        )
        exposure_next = self.exposure.next_value(intercept_prob_instant)

        # ---------- 汇总 ----------
        if detections:
            pd_min = min(d.pd for d in detections)
            snr_min = min(d.snr_db for d in detections)
            task_satisfied = all(d.satisfied for d in detections)
            detection_term = min(min(d.pd / radar.required_pd, 1.0) for d in detections)
        else:
            pd_min, snr_min, task_satisfied, detection_term = 0.0, float("-inf"), False, 0.0

        return StepEvaluation(
            tx_power_w=pt_w,
            detections=detections,
            pd_min=pd_min,
            snr_radar_db_min=snr_min,
            required_snr_db=required_snr_db,
            task_satisfied=task_satisfied,
            detection_term=detection_term,
            interceptions=interceptions,
            intercept_prob=intercept_prob,
            intercept_prob_instant=intercept_prob_instant,
            intercept_snr_db=intercept_snr_db,
            exposure=exposure_before,
            exposure_next=exposure_next,
            jammer_active=jammer_active,
            interference_power_w=interference_w,
            noise_power_w=noise_w,
            jam_noise_ratio=jam_noise_ratio,
        )

    # ------------------------------------------------------------------
    # 推进一步
    # ------------------------------------------------------------------

    def step(self, power_level: int) -> StepResult:
        """施加一个功率档位动作，返回该步结果，并把环境推进到下一步。"""
        self._ensure_ready()
        assert self.scenario is not None and self.radar is not None

        if self.is_done:
            raise RuntimeError(
                f"episode 已结束（step_index={self.step_index} >= {self.scenario.num_steps}），"
                "请先 reset()"
            )
        if not 0 <= power_level < self.num_levels:
            raise ValueError(
                f"power_level={power_level} 越界，合法范围 0..{self.num_levels - 1}"
            )

        # --- 能量硬约束：执行前检查本步所需能量是否超过剩余能量 ---
        step_energy_j = self.step_energy_j(power_level)
        if not self.is_level_feasible(power_level):
            raise InfeasibleActionError(
                f"档位 {power_level}（{self.scenario.power_levels_w[power_level]} W）"
                f"本步需要 {step_energy_j:.3f} J，但只剩 {self.remaining_energy_j:.3f} J。"
                f"可用档位：{self.feasible_levels()}"
                "（环境层会先裁剪到可行档，直接调用仿真器则必须自行保证可行）"
            )

        pt_w = self.scenario.power_levels_w[power_level]
        evaluation = self.preview(pt_w, self.step_index)

        # --- 扣减能量：E(t+1) = E(t) − Pt·Δt ---
        self.cumulative_energy_j += step_energy_j

        # 不变式：累计能耗永远不超过预算（可行性检查已保证，这里做兜底断言）
        budget = self.radar.energy_budget_j
        if self.cumulative_energy_j > budget + self.ENERGY_EPS:
            raise RuntimeError(
                f"能量不变式被破坏：累计 {self.cumulative_energy_j:.6f} J "
                f"> 预算 {budget:.6f} J。这是实现缺陷，不应发生。"
            )

        # 扣减后已无任何可执行档位 -> 本步之后立即终止
        if self.energy_exhausted:
            self.terminated_by_energy = True

        # --- 探测任务未达标判定 ---
        violated = not evaluation.task_satisfied

        # --- 收益 ---
        reward = composite_reward(
            detection_term=evaluation.detection_term,
            intercept_prob=evaluation.intercept_prob,
            power_fraction=pt_w / self.scenario.max_power_w,
            weights=self.scenario.reward_weights,
            violation=1.0 if violated else 0.0,
        )

        # --- 能量耗尽的一次性终端惩罚：未执行的任务步按全部失败计 ---
        output_terminal_penalty = 0.0
        if self.terminated_by_energy and self.radar.terminate_on_energy_exhausted:
            output_terminal_penalty = terminal_energy_penalty(
                self.remaining_steps, self.scenario.reward_weights
            )
            reward += output_terminal_penalty

        self.episode_return += reward

        # --- 推进累计暴露（本步证据影响未来）---
        exposure_after = self.exposure.update(evaluation.intercept_prob_instant)

        # --- 自适应干扰机：用本步证据决定**下一步**的干扰动作 ---
        jammer_mode, jammer_mode_cn = self._update_adaptive_jammers(
            exposure_after, pt_w, evaluation.task_satisfied
        )

        result = StepResult(
            step_index=self.step_index,
            time=self.current_time,
            power_level=power_level,
            tx_power_w=pt_w,
            detections=evaluation.detections,
            pd_min=evaluation.pd_min,
            snr_radar_db_min=evaluation.snr_radar_db_min,
            required_snr_db=evaluation.required_snr_db,
            task_satisfied=evaluation.task_satisfied,
            detection_term=evaluation.detection_term,
            interceptions=evaluation.interceptions,
            intercept_prob=evaluation.intercept_prob,
            intercept_prob_instant=evaluation.intercept_prob_instant,
            intercept_snr_db=evaluation.intercept_snr_db,
            exposure=evaluation.exposure,
            exposure_next=exposure_after,
            jammer_active=evaluation.jammer_active,
            jammer_mode=jammer_mode,
            jammer_mode_cn=jammer_mode_cn,
            jam_noise_ratio=evaluation.jam_noise_ratio,
            interference_power_w=evaluation.interference_power_w,
            noise_power_w=evaluation.noise_power_w,
            step_energy_j=step_energy_j,
            cumulative_energy_j=self.cumulative_energy_j,
            remaining_energy_j=self.remaining_energy_j,
            energy_fraction=self.energy_fraction,
            energy_exhausted=self.energy_exhausted,
            task_violated=violated,
            reward=reward,
            terminal_penalty=output_terminal_penalty,
        )

        self.previous_power_level = power_level
        self.results.append(result)
        self._advance()
        return result

    def _update_adaptive_jammers(
        self, exposure: float, radar_power_w: float, task_satisfied: bool
    ) -> tuple[str, str]:
        """让规则自适应干扰机观测本步证据并决定下一步动作。

        返回 (主干扰机模式, 中文名)；固定模式或无自适应干扰机时返回 ("", "")。

        因果顺序：本步的干扰功率早已在 jam_state() 里按**上一步末**决定的模式算完，
        这里看到的是本步产生的新证据，只影响**下一步**。
        """
        primary_mode = ""
        primary_mode_cn = ""
        assert self.scenario is not None and self.radar is not None

        for jammer in self.jammers:
            controller = jammer.controller
            if controller is None:
                continue
            mode = controller.observe(
                exposure=exposure,
                radar_power_w=radar_power_w,
                task_satisfied=task_satisfied,
                max_power_w=self.scenario.max_power_w,
                step_index=self.step_index,
                current_time=self.current_time,
            )
            if not primary_mode:
                primary_mode = mode
                from models.adaptive_jammer import MODE_CN

                primary_mode_cn = MODE_CN.get(mode, mode)
        return primary_mode, primary_mode_cn

    def jammer_action_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """导出所有自适应干扰机的动作轨迹（供日志/CSV/报告使用）。"""
        traces: Dict[str, List[Dict[str, Any]]] = {}
        for jammer in self.jammers:
            if jammer.controller is not None:
                traces[jammer.jammer_id] = list(jammer.controller.trace)
        return traces

    def action_reward(self, evaluation: StepEvaluation, pt_w: float) -> float:
        """给定一步的评估结果，计算其**单步**收益（不含终端惩罚）。

        统一入口：GreedyOraclePolicy、LookaheadPolicy 等需要「试算某档功率收益」的
        策略都调用它，从而与 step() 内部的收益口径严格一致。
        """
        assert self.scenario is not None
        return composite_reward(
            detection_term=evaluation.detection_term,
            intercept_prob=evaluation.intercept_prob,
            power_fraction=pt_w / self.scenario.max_power_w,
            weights=self.scenario.reward_weights,
            violation=0.0 if evaluation.task_satisfied else 1.0,
        )

    def _advance(self) -> None:
        """推进平台运动与时间步（替代原 update_nodes）。

        v4.1：改为经由 `Scene.advance_all()` 统一推进**全部**实体
        （含所有雷达，而不只是主雷达），保证：
        * 每个实体的位置、速度、姿态与 `timestamp_s` 同步前进；
        * 多雷达场景下非主雷达也会运动（否则多平台几何是假的）。
        运动学与旧实现逐字等价（`x += vx*dt`），旧场景数值不变。
        """
        assert self.scenario is not None and self.radar is not None
        self.scene.advance_all(self.scenario.time_step)
        self.step_index += 1

    # ------------------------------------------------------------------
    # 几何查询（v4.1：统一走 Scene，禁止各模块自算）
    # ------------------------------------------------------------------

    def relation(
        self, observer_id: str, target_id: str, time_s: float | None = None
    ) -> GeometricRelation:
        """取两个实体之间的**有向**几何关系（谁相对谁、在哪一时刻）。

        这是升级后获取几何量的唯一推荐入口。旧的 `range_to()` 仍在，
        但它只给距离、没有方向与时刻语义，多平台场景下请用本方法。
        """
        return self.scene.relation(observer_id, target_id, time_s=time_s)

    def relations_between(
        self, observer_kind: str, target_kind: str, time_s: float | None = None
    ) -> List[GeometricRelation]:
        """两类实体之间的全部有向关系（如所有雷达到所有目标）。"""
        return self.scene.relations_between(observer_kind, target_kind, time_s=time_s)

    def radar_target_relations(
        self, time_s: float | None = None
    ) -> List[GeometricRelation]:
        """所有雷达 → 所有目标的显式关系（替代"最近/最远目标距离"这类聚合值）。"""
        from models.entity import KIND_RADAR, KIND_TARGET

        return self.scene.relations_between(KIND_RADAR, KIND_TARGET, time_s=time_s)

    def scene_snapshot(self) -> Dict[str, Any]:
        """当前时刻的完整场景快照（实体 + 关系），供 CSV/JSON 导出。"""
        return self.scene.to_dict()

    def assert_scene_time_synchronized(self) -> None:
        """校验场景内全部实体时间戳一致（多平台几何的前提）。"""
        self.scene.assert_time_synchronized()

    # ------------------------------------------------------------------
    # 整段运行
    # ------------------------------------------------------------------

    def run(self, policy: Any) -> List[StepResult]:
        """用给定策略跑完一个 episode，返回逐步结果。"""
        self.reset()
        if hasattr(policy, "reset"):
            policy.reset()

        results: List[StepResult] = []
        while not self.is_done:
            level = policy.select_level(self)
            results.append(self.step(level))
        return results

    # --- 便捷入口：对应原 run_baseline / run_with_disturbance / run_with_strategy ---
    # 策略在函数内局部导入，避免 engine 与 strategy 在模块加载期互相依赖。

    def run_fixed_power(self) -> List[StepResult]:
        """固定功率基线（默认满功率）。"""
        from strategy.power_policy import FixedPowerPolicy

        assert self.scenario is not None
        return self.run(FixedPowerPolicy(self.scenario.resolve_fixed_power_level()))

    def run_rule_based(self, margin_db: float = 0.0) -> List[StepResult]:
        """规则功率控制基线：选取满足探测要求的最低功率档。"""
        from strategy.power_policy import RuleBasedPowerPolicy

        return self.run(RuleBasedPowerPolicy(margin_db=margin_db))

    def run_random(self, seed: int = 0) -> List[StepResult]:
        """随机策略：仅用于演示 Gymnasium 风格动作空间。"""
        from strategy.power_policy import RandomPowerPolicy

        return self.run(RandomPowerPolicy(seed=seed))

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """输出关键链路预算，便于在控制台核对模型标定是否合理。"""
        self._ensure_ready()
        assert self.radar is not None and self.scenario is not None

        radar = self.radar
        noise_w = thermal_noise_w(
            radar.bandwidth_hz, radar.noise_figure_db, radar.temperature_k
        )
        required_snr_db = snr_db_for_prob(
            radar.required_pd, radar.snr50_db, radar.pd_slope_db
        )
        required_snr_lin = db2lin(required_snr_db)

        primary = self._primary_target()
        info: Dict[str, Any] = {
            "scenario": self.scenario.scenario_name,
            "num_steps": self.scenario.num_steps,
            "num_levels": self.scenario.num_levels,
            "power_levels_w": list(self.scenario.power_levels_w),
            "fixed_power_level": self.scenario.resolve_fixed_power_level(),
            "lpi_pint_threshold": self.scenario.lpi_pint_threshold,
            "exposure_decay": self.scenario.exposure_decay,
            "exposure_gain": self.scenario.exposure_gain,
            "reward_weights": dict(self.scenario.reward_weights),
            "terminate_on_energy_exhausted": radar.terminate_on_energy_exhausted,
            "radar_noise_w": noise_w,
            "required_pd": radar.required_pd,
            "required_snr_db": required_snr_db,
            "energy_budget_j": radar.energy_budget_j,
        }

        # --- 逐目标的链路预算，找出真正「卡脖子」的目标 ---
        per_target: Dict[str, Dict[str, float]] = {}
        for target in self.active_targets():
            rng_m = target.range_to(radar.x, radar.y)
            echo_per_watt = radar_echo_power_w(
                pt_w=1.0,
                gain_db=radar.peak_gain_db,
                wavelength_m=radar.wavelength_m,
                rcs_m2=target.rcs_m2,
                range_m=rng_m,
                system_loss_db=radar.system_loss_db,
            )
            snr_per_watt = echo_per_watt / noise_w if noise_w > 0 else 0.0
            needed_w = required_snr_lin / snr_per_watt if snr_per_watt > 0 else float("inf")
            per_target[target.target_id] = {
                "range_m": rng_m,
                "rcs_m2": target.rcs_m2,
                "snr_db_per_watt": lin2db(snr_per_watt),
                "needed_power_w": needed_w,
            }

        if per_target:
            binding_id = max(per_target, key=lambda k: per_target[k]["needed_power_w"])
            info["per_target_budget"] = per_target
            info["binding_target"] = binding_id
            info["nominal_min_power_w_needed"] = per_target[binding_id]["needed_power_w"]
            info["min_level_for_all_targets"] = self.min_satisfying_level()
            if primary is not None:
                info["primary_target"] = primary.target_id
                info["primary_range_m"] = per_target[primary.target_id]["range_m"]
                info["primary_rcs_m2"] = primary.rcs_m2
                info["radar_snr_db_per_watt"] = per_target[primary.target_id][
                    "snr_db_per_watt"
                ]

        esm = self.interceptors[0]
        rng_m = esm.range_to(radar.x, radar.y)
        _, beam = self._gain_and_beam_toward(esm.x, esm.y)
        jammer_modes = {
            j.jammer_id: j.jammer_mode for j in self.jammers
        }
        info.update(
            {
                "interceptor": esm.interceptor_id,
                "interceptor_range_m": rng_m,
                "interceptor_beam": beam,
                "interceptor_snr50_db": esm.snr50_db,
                "jammer_modes": jammer_modes,
                "adaptive_jammer": any(j.is_adaptive for j in self.jammers),
            }
        )
        return info

    def min_satisfying_level(self, step_index: int | None = None) -> int | None:
        """当前几何下，能满足探测要求的最低功率档位；无解返回 None。"""
        assert self.scenario is not None
        for level, pt_w in enumerate(self.scenario.power_levels_w):
            if self.preview(pt_w, step_index).task_satisfied:
                return level
        return None
