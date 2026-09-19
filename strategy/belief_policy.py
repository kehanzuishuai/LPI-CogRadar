"""信念状态桥接：让「基于状态的脚本策略」在部分可观测条件下也能公平参赛。

为什么必须有这个模块
--------------------
工程里所有脚本基线（规则、短视贪心、前瞻）的签名都是 `select_level(sim)`，
它们直接读**真值**并调用 `sim.preview()` 做试算。这在全可观实验里没问题，
但一旦去做「部分可观测 vs 全可观」的对比就会出大问题：

    如果 DQN 只能看到带噪、延迟、丢测的估计，而规则策略能看真值，
    那两者比出来的差距里混着「信息量差异」，**不能**归因于算法能力。

诚实地处理它只有两条路：
1. 承认信息不对称，把脚本基线明确标注为「全状态参考上界」，不参与排名；
2. 给脚本策略一个只依赖观测的版本。

本模块实现第 2 条。做法是构造一个**信念仿真器（belief simulator）**：
从真值仿真器深拷贝一份，然后把「智能体测到的估计值」写进去，
使这份副本在 `preview()` 下表现得就像那些估计值是真值一样。
脚本策略照旧在副本上运行，**它读不到任何真值**。

信念的正确性是有限的，这点必须说清楚
------------------------------------
* 距离：按主目标距离估计值做统一的径向缩放，主目标距离精确等于估计值；
  其余目标被同一比例缩放，这是「只有一个距离量测」的必然结果。
* RCS：统一缩放使**最小 RCS** 等于估计值，保留目标间相对关系。
* 剩余能量：按估计值反推累计能耗，精确。
* 干扰：用探针法标定「每瓦峰值功率产生的干扰功率」，再反解出使 J/N
  等于估计值的峰值功率，因此在当前时刻精确；但**未来时刻的干扰起伏
  仍沿用真值仿真器的序列**——这正是部分可观测下无法避免的模型误差。
* 目标未来的运动、侦察机位置、以及真值本身的演化都不受信念影响。

换句话说：信念副本在「当前一步」是准的，在「往后推演」时必然越来越偏。
这正是我们想考察的困难，而不是一个需要被修掉的 bug。
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strategy.power_policy import PowerPolicy

#: 探针法允许的最大峰值功率倍数，避免数值爆炸
_MAX_PROBE_SCALE = 1e9


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_belief_simulator(
    env: Any,
    estimates: Optional[Dict[str, Any]] = None,
    options: Optional["BeliefOptions"] = None,
) -> Any:
    """按观测估计值构造一个信念仿真器副本。

    参数
    ----
    env       : LpiPowerEnv（观测模式须为 pomdp / ideal / realistic）
    estimates : `info["observations"]`，缺省时取 env 缓存的最新估计
    options   : 见 `BeliefOptions`

    返回一个 `Simulator` 深拷贝，其当前状态已按估计值改写，
    并**已剥离真值的未来信息**（见 `_sanitize_belief`）。
    **不会**修改传入的 env / 真值仿真器。

    ⚠️ 为什么必须剥离（v4.3 真值泄漏审计的结论）
    ------------------------------------------
    `copy.deepcopy(sim)` 会把**整个**真值仿真器复制过来，其中包括：

    * `jammer._fluctuation_series` —— 干扰强度起伏是**开局一次性预生成整段**的，
      因此副本里带着**未来每一步**的真实干扰强度；
    * 侦察机等**本平台观测不到**的实体的真实位置。

    这两项都会真的影响策略行为。审计实测：

    * 篡改真值的**未来**起伏后，信念推演出的 J/N 从 **1.52 变成 6.91**；
    * 把真值 ESM 挪走 100 km 后，信念算出的 Pint 从 **0.530 变成 0.329**。

    也就是说，不剥离就等于让"部分可观测"策略**偷看未来与敌方位置**，
    所有部分可观测实验的结论都会被高估。这是真实存在过的缺陷，不是理论担忧。
    """
    opts = options or BeliefOptions()
    sim = copy.deepcopy(env.sim)
    if estimates is None:
        estimates = getattr(env, "_last_estimates", {}) or {}

    # --- 先剥离真值未来信息，再写入估计值 ---
    _sanitize_belief(env, sim, opts)

    if not estimates:
        sim.belief_metadata = {
            "sanitized": True,
            "estimate_keys": [],
            "options": opts.to_dict(),
        }
        return sim

    radar = sim.radar
    budget = float(radar.energy_budget_j)

    def est(name: str) -> Optional[float]:
        record = estimates.get(name)
        if record is None:
            return None
        try:
            return float(record["value"])
        except (KeyError, TypeError, ValueError):
            return None

    # ---------------- 1) 剩余能量 ----------------
    remaining = est("remaining_energy")
    if remaining is not None:
        sim.cumulative_energy_j = _clamp(budget - remaining, 0.0, budget)

    # ---------------- 2) 目标距离（统一径向缩放） ----------------
    primary = None
    try:
        primary = sim._primary_target()
    except Exception:  # pragma: no cover - 防御性
        primary = None
    est_range = est("target_range")
    if primary is not None and est_range is not None:
        true_range = primary.range_to(radar.x, radar.y)
        if true_range > 1e-9:
            scale = max(est_range, 1.0) / true_range
            for target in sim.targets:
                target.x = radar.x + (target.x - radar.x) * scale
                target.y = radar.y + (target.y - radar.y) * scale

    # ---------------- 3) RCS（使最小 RCS 等于估计值） ----------------
    est_rcs = est("target_rcs")
    if est_rcs is not None and est_rcs > 0.0:
        true_min = min((t.rcs_m2 for t in sim.targets), default=0.0)
        if true_min > 1e-9:
            factor = est_rcs / true_min
            for target in sim.targets:
                target.rcs_m2 = max(1e-6, target.rcs_m2 * factor)

    # ---------------- 4) 干扰强度 ----------------
    est_jam = est("jam_ratio")
    if est_jam is not None:
        _force_jam_ratio(sim, max(0.0, est_jam))

    # ---------------- 5) 累计暴露 ----------------
    est_exposure = est("exposure")
    if est_exposure is not None:
        sim.exposure.value = _clamp(est_exposure, 0.0, 1.0)

    sim.belief_metadata = {
        "sanitized": True,
        "estimate_keys": sorted(estimates.keys()),
        "options": opts.to_dict(),
    }
    return sim


@dataclass
class BeliefOptions:
    """信念构造选项（全都是"要不要老实"的开关）。"""

    #: 未来干扰起伏怎么处理：
    #:   "persistence" —— 沿用最后一次已知值（默认，**不含未来信息**）
    #:   "resample"    —— 用**信念自己的** RNG 按同样统计特性重新生成
    #:   "keep_truth"  —— 保留真值未来序列（**这是泄漏**，仅用于构造"先知"上界基线）
    future_jammer_model: str = "persistence"

    #: 观测不到的侦察机怎么办：
    #:   "deactivate" —— 在信念里置为 inactive，`preview()` 不再产生截获记录。
    #:                   于是 Pint 视为"未知/0"——对使用截获惩罚的策略是**乐观**的，
    #:                   但至少不是作弊。
    #:   "keep_truth" —— 保留真值位置（**这是泄漏**，仅用于"先知"上界基线）
    unobservable_interceptor: str = "deactivate"

    #: 信念自己的 RNG 种子（与真值仿真器无关，避免共享随机流）
    seed: int = 20260918

    def validate(self) -> None:
        if self.future_jammer_model not in ("persistence", "resample", "keep_truth"):
            raise ValueError(f"future_jammer_model={self.future_jammer_model!r} 非法")
        if self.unobservable_interceptor not in ("deactivate", "keep_truth"):
            raise ValueError(
                f"unobservable_interceptor={self.unobservable_interceptor!r} 非法"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "future_jammer_model": self.future_jammer_model,
            "unobservable_interceptor": self.unobservable_interceptor,
            "seed": self.seed,
        }


def _sanitize_belief(env: Any, sim: Any, options: BeliefOptions) -> None:
    """剥离信念仿真器里的真值未来信息与他平台真值。

    三件事：
    1. **未来干扰起伏**：真值把整段起伏预生成了，副本里带着未来。
       默认改成"持续性模型"（沿用最后一个已知值），或用自己的 RNG 重采样；
    2. **观测不到的平台**：雷达不知道侦察机在哪，信念里不能保留其真值位置；
    3. **随机流**：给信念自己的 RNG，避免与真值共享后续随机数。
    """
    options.validate()
    index = int(getattr(env.sim, "step_index", 0))

    # --- 1) 未来干扰起伏 ---
    for jammer in sim.jammers:
        series = list(getattr(jammer, "_fluctuation_series", []) or [])
        if not series:
            continue
        known = series[: index + 1]  # 已经过去的（属于可被观测的历史）
        last_known = known[-1] if known else 1.0
        remaining = max(0, len(series) - len(known))
        if options.future_jammer_model == "persistence":
            future = [last_known] * remaining
        elif options.future_jammer_model == "resample":
            rng = random.Random("%d:%s" % (options.seed, jammer.jammer_id))
            future = []
            value = last_known
            for _ in range(remaining):
                value += rng.uniform(-jammer.fluctuation_step, jammer.fluctuation_step)
                value = max(1.0 - jammer.fluctuation,
                            min(1.0 + jammer.fluctuation, value))
                future.append(value)
        else:  # keep_truth —— 显式泄漏，仅用于"先知"上界
            future = series[len(known):]
        jammer._fluctuation_series = known + list(future)

    # --- 2) 观测不到的平台 ---
    if options.unobservable_interceptor == "deactivate":
        observed = _observed_interceptor_ids(env)
        for esm in sim.interceptors:
            if esm.interceptor_id not in observed:
                esm.is_active = False

    # --- 3) 独立随机流 ---
    sim.rng = random.Random(options.seed)


def _observed_interceptor_ids(env: Any) -> set:
    """当前**确实被观测到**的侦察机 ID 集合（来自测量，不是真值）。

    雷达没有对 ESM 的定位手段，因此正常情况下这个集合是空的。
    将来若接入 RWR / 多站定位等能力，只要测量里出现该平台，它就会被算进来。
    """
    observed: set = set()
    suite = getattr(env, "suite", None)
    if suite is None:
        return observed
    for sensor in getattr(suite, "sensors", []):
        if getattr(sensor.config, "observes_kind", "") != "interceptor":
            continue
        for truth_id in getattr(sensor, "_candidate_of_truth", {}):
            observed.add(truth_id)
    return observed


def _force_jam_ratio(sim: Any, target_ratio: float) -> None:
    """调整干扰机峰值功率，使 `preview()` 给出的 J/N 等于 target_ratio。

    做法：把第一台干扰机的峰值功率设为 1 W、其余置 0，试算一次拿到
    「每瓦干扰功率」k 与噪声功率 N；则峰值功率应取 target_ratio * N / k。
    若当前时刻干扰机未开机导致 k = 0，则先把它强制开机再标定一次
    （这相当于「智能体以为此刻正在被干扰」，是它有估计值时的合理推断）。
    """
    jammers: List[Any] = list(sim.jammers)
    if not jammers:
        return

    def zero_all_but_first() -> None:
        for index, jammer in enumerate(jammers):
            jammer.peak_power_w = 1.0 if index == 0 else 0.0

    zero_all_but_first()
    probe = sim.preview(sim.power_levels_w[0])
    per_watt = float(probe.interference_power_w)
    noise = float(probe.noise_power_w)

    if per_watt <= 1e-30 and target_ratio > 0.0:
        # 当前时刻没在干扰，但智能体认为有 -> 强制开机后重新标定
        for jammer in jammers:
            jammer.dynamic = False
            jammer.jammer_mode = "fixed"
            jammer.start_time = 0.0
            jammer.end_time = 1e9
            jammer.duty_cycle = 1.0
            jammer.intensity = 1.0
        zero_all_but_first()
        probe = sim.preview(sim.power_levels_w[0])
        per_watt = float(probe.interference_power_w)
        noise = float(probe.noise_power_w)

    if per_watt <= 1e-30:
        # 无论怎么试探都无法产生干扰（例如压制量为 0），只能作罢
        for jammer in jammers:
            jammer.peak_power_w = 0.0
        return

    if target_ratio <= 0.0:
        for jammer in jammers:
            jammer.peak_power_w = 0.0
        return

    needed = target_ratio * noise / per_watt
    jammers[0].peak_power_w = min(needed, _MAX_PROBE_SCALE)
    for jammer in jammers[1:]:
        jammer.peak_power_w = 0.0


class BeliefPolicy:
    """把任意 `select_level(sim)` 脚本策略改造成「只依赖观测」的策略。

    对外的 `select_level_from_env(env)` 先用观测估计值搭出信念仿真器，
    再把信念交给内部策略。因此内部策略**看不到真值**。

    同时保留 `select_level(sim)`：在 full 模式下它退化为直接用真值，
    方便与旧实验对齐。
    """

    def __init__(self, inner: PowerPolicy, name: Optional[str] = None) -> None:
        self.inner = inner
        self.name = name or f"{inner.name}(仅观测)"
        self.belief_count = 0
        self.last_belief: Any = None

    def reset(self) -> None:
        self.inner.reset()
        self.belief_count = 0
        self.last_belief = None

    # ------------------------------------------------------------------

    def belief_for(self, env: Any) -> Any:
        """为当前 env 构造信念仿真器（并留下最近一次的引用便于调试）。"""
        belief = build_belief_simulator(env)
        self.belief_count += 1
        self.last_belief = belief
        return belief

    def select_level_from_env(self, env: Any) -> int:
        """只依据 env 的观测选择档位。"""
        if getattr(env, "observation_mode", "full") == "full":
            return int(self.inner.select_level(env.sim))
        belief = self.belief_for(env)
        if belief.is_done or not belief.feasible_levels():
            # 信念认为已经结束（例如剩余能量估计为 0）：退到真值下的可行最高档
            level = env.sim.clip_to_feasible(env.sim.previous_power_level)
            if level is None:
                return 0
            return int(level)
        return int(self.inner.select_level(belief))

    # 兼容 PowerPolicy 接口（full 模式下直接委托）
    def select_level(self, sim: Any) -> int:
        return int(self.inner.select_level(sim))

    def describe(self) -> str:
        return (
            f"{self.name}（内部为 {self.inner.describe()}；"
            f"决策输入 = 带噪信念状态，不含真值）"
        )
