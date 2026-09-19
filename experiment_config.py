"""统一实验配置与共用运行器。

为什么需要这个模块
------------------
之前在 `diagnose_temporal_coupling.py` 与 `evaluate_dqn.py` 里**各自**构造前瞻策略
（`LookaheadPolicy(horizon=num_steps + 1, discount=1.0)` 写了两遍）。
两处当时恰好一致，但这种重复极易漂移：改了一处、忘了另一处，
就会得到互相矛盾的结论——本项目确实发生过一次「诊断脚本显示前瞻满足率 0.9508、
评测脚本显示 0.9344」的假象（事后核实是引用了改参数前的过期诊断输出，
而非真实不一致）。

现在把下面这些东西**收敛到本模块**，所有脚本一律通过同一工厂获取，
从结构上杜绝这类不一致：

* 场景路径与随机种子（单种子 / 多种子集合）
* 前瞻视野规则（`lookahead_horizon`）
* 策略集合与 CSV 命名
* episode 运行方式（脚本策略 / DQN 策略）

约定
----
所有脚本评价同一个 episode 时，必须用**同一个种子、同一份配置、同一套策略定义**，
这样任何两处结果都能直接逐位对比。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from engine import LpiPowerEnv
from metrics import summarize_run
from strategy.belief_policy import BeliefPolicy
from strategy.power_policy import (
    FixedPowerPolicy,
    GreedyOraclePolicy,
    LookaheadPolicy,
    PowerPolicy,
    RandomPowerPolicy,
    RuleBasedPowerPolicy,
)

# ----------------------------------------------------------------------
# 项目标识
# ----------------------------------------------------------------------

#: 项目英文名（统一使用；出现在报告标题、日志、API 响应与文档中）
PROJECT_NAME = "LPI-CogRadar"

#: 项目全称
PROJECT_TITLE = (
    "LPI-CogRadar: An AI-Driven Cognitive Radar Power Control "
    "and Electromagnetic Adversarial Simulation Platform"
)

#: 当前版本号
PROJECT_VERSION = "4.5.0"

# ----------------------------------------------------------------------
# 实验级常量
# ----------------------------------------------------------------------

CONFIG_PATH = "config/radar_scenario_v1.json"

#: 主实验种子（与 main.py / evaluate_dqn.py 一致）
DEFAULT_SEED = 42

#: 随机策略自身的种子（与场景种子区分开）
RANDOM_POLICY_SEED = 2026

#: 多种子评测默认使用的种子集合（10 个，覆盖正负起伏）
DEFAULT_SEEDS: Tuple[int, ...] = (42, 7, 13, 21, 33, 55, 77, 99, 123, 2024)

#: 能量预算敏感性实验的预算档位
DEFAULT_ENERGY_BUDGETS: Tuple[float, ...] = (1200.0, 1300.0, 1400.0, 1500.0, 1800.0)

#: 前瞻推演用的折扣因子（1.0 = 不做折扣，配合完整视野）
LOOKAHEAD_DISCOUNT = 1.0


# ----------------------------------------------------------------------
# 统一的构造入口
# ----------------------------------------------------------------------

def lookahead_horizon(num_steps: int) -> int:
    """前瞻视野规则：任务步数 + 1。

    +1 保证从**任何**一步出发都能看到 episode 结束（含最后一步），
    因此这是「完整任务视野」的前瞻，而不是固定步数的短前瞻。
    """
    return int(num_steps) + 1


def make_lookahead(num_steps: int, discount: float = LOOKAHEAD_DISCOUNT) -> LookaheadPolicy:
    """统一的前瞻策略工厂。所有脚本必须经由它构造前瞻基线。"""
    return LookaheadPolicy.full_horizon(num_steps, discount=discount)


def make_env(
    config_path: str = CONFIG_PATH,
    energy_budget_j: float | None = None,
    observation_mode: str = "full",
    history_len: int = 1,
    observation_noise: Any = None,
    expose_observation_truth: bool = False,
    measurement_max_tracks: int = 4,
    expose_measurement_truth: bool = False,
    horizon_semantics: str = "finite_task",
) -> LpiPowerEnv:
    """统一的实验环境。energy_budget_j 只覆盖任务约束，不触碰物理参数。

    observation_mode：
      "full"      —— 真值观测（v3.1 行为，默认，逐位可复现）
      "pomdp"     —— v4.0 全局噪声/延迟/丢测模型（保留以复现 v4.0 结果）
      "ideal"     —— **测量层**：作用距离/视场/遮挡/更新周期约束生效，
                     但测量无噪声、无概率漏检、无虚警
      "realistic" —— **测量层**：同上，再叠加量测噪声、概率漏检与虚警

    ideal / realistic 走 `sensor/` 测量层，观测由传感器测量汇聚而成
    （见 README §11C 的三层数据字典）。
    """
    return LpiPowerEnv(
        config_path,
        energy_budget_j=energy_budget_j,
        observation_mode=observation_mode,
        history_len=history_len,
        observation_noise=observation_noise,
        expose_observation_truth=expose_observation_truth,
        measurement_max_tracks=measurement_max_tracks,
        expose_measurement_truth=expose_measurement_truth,
        horizon_semantics=horizon_semantics,
    )


def scenario_horizon(config_path: str = CONFIG_PATH, energy_budget_j: float | None = None) -> int:
    """任务步数（前瞻视野与 horizon_satisfaction_rate 都用它）。"""
    return int(make_env(config_path, energy_budget_j).sim.scenario.num_steps)


# ----------------------------------------------------------------------
# v4.0 部分可观测预设（只调测量噪声，不动物理参数）
# ----------------------------------------------------------------------

#: 各档部分可观测强度：轻度 / 中度（默认）/ 重度
OBSERVATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {
        "enabled": False,
    },
    "mild": {
        "enabled": True,
        "range_sigma_m": 60.0,
        "rcs_sigma_m2": 0.06,
        "jam_ratio_sigma": 0.05,
        "energy_sigma_j": 8.0,
        "interceptor_range_sigma_m": 2000.0,
        "pint_sigma": 0.03,
        "exposure_sigma": 0.03,
        "pd_sigma": 0.02,
        "delay_steps": 1,
        "dropout_prob": 0.03,
    },
    "moderate": {
        "enabled": True,
        "range_sigma_m": 120.0,
        "rcs_sigma_m2": 0.12,
        "jam_ratio_sigma": 0.10,
        "energy_sigma_j": 15.0,
        "interceptor_range_sigma_m": 4000.0,
        "pint_sigma": 0.05,
        "exposure_sigma": 0.05,
        "pd_sigma": 0.03,
        "delay_steps": 1,
        "dropout_prob": 0.10,
    },
    "severe": {
        "enabled": True,
        "range_sigma_m": 250.0,
        "rcs_sigma_m2": 0.25,
        "jam_ratio_sigma": 0.20,
        "energy_sigma_j": 30.0,
        "interceptor_range_sigma_m": 8000.0,
        "pint_sigma": 0.10,
        "exposure_sigma": 0.10,
        "pd_sigma": 0.06,
        "delay_steps": 2,
        "dropout_prob": 0.25,
    },
}

#: 「部分可观测 + 历史窗口」的推荐组合：K 帧堆叠
DEFAULT_HISTORY_LEN = 4


def observation_preset(name: str) -> Dict[str, Any]:
    """取一个预设的噪声配置（返回副本，调用方可自由改写）。"""
    if name not in OBSERVATION_PRESETS:
        raise KeyError(
            f"未知观测预设 {name!r}，可选 {sorted(OBSERVATION_PRESETS)}"
        )
    return dict(OBSERVATION_PRESETS[name])


def make_pomdp_env(
    preset: str = "moderate",
    history_len: int = 1,
    energy_budget_j: float | None = None,
    config_path: str = CONFIG_PATH,
    expose_observation_truth: bool = False,
    **noise_overrides: Any,
) -> LpiPowerEnv:
    """构造一个部分可观测环境（预设 + 逐项覆盖）。

    expose_observation_truth=True 时 `info["observations"][名]["truth"]` 会附带真值，
    仅供诊断脚本统计观测误差；**策略代码不应读取它**，否则部分可观测假设失效。
    """
    noise = observation_preset(preset)
    noise.update(noise_overrides)
    return make_env(
        config_path,
        energy_budget_j=energy_budget_j,
        observation_mode="pomdp",
        history_len=history_len,
        observation_noise=noise,
        expose_observation_truth=expose_observation_truth,
    )


# ----------------------------------------------------------------------
# 多种子鲁棒性评测用的场景扰动（只扰动初始条件，不动物理参数）
# ----------------------------------------------------------------------

#: 扰动幅度（只作用于初始条件与时间表）
JITTER_TARGET_POSITION_M = 300.0
JITTER_JAMMER_POSITION_M = 500.0
JITTER_JAMMER_WINDOW_S = 3.0


def jittered_scenario(
    seed: int,
    config_path: str = CONFIG_PATH,
    target_position_m: float = JITTER_TARGET_POSITION_M,
    jammer_position_m: float = JITTER_JAMMER_POSITION_M,
    jammer_window_s: float = JITTER_JAMMER_WINDOW_S,
) -> Dict[str, Any]:
    """按种子生成一组**初始条件扰动**，供多种子鲁棒性评测使用。

    为什么需要它
    ------------
    本场景里种子只驱动干扰机的强度起伏，而该起伏被夹在 ±10% 边界上反复饱和，
    折算到 J/N 上只有约 0.83 dB 的摆幅——小于 25 W→35 W 的档位间隔（约 1.46 dB）。
    结果是：**规则/短视/前瞻这类确定性策略在不同种子下给出完全相同的决策**，
    多种子评测的标准差恒为 0，说明不了任何问题。

    为了让「不依赖 seed=42」这句话有实际内容，这里对**初始条件**做域随机化：
    目标与干扰机的初始位置、干扰时间窗。**只改场景的初始状态与时间表**，
    不触碰雷达方程、侦察模型、功率档位、RCS、发射功率等任何物理参数。

    返回可直接传给 `Simulator.apply_overrides(extra=...)` 的字典。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(int(seed))

    targets: List[Dict[str, Any]] = []
    for target in data.get("targets", []):
        item = dict(target)
        item["x"] = float(item["x"]) + rng.uniform(-target_position_m, target_position_m)
        item["y"] = float(item["y"]) + rng.uniform(-target_position_m, target_position_m)
        targets.append(item)

    jammers: List[Dict[str, Any]] = []
    for jammer in data.get("jammers", []):
        item = dict(jammer)
        item["x"] = float(item["x"]) + rng.uniform(-jammer_position_m, jammer_position_m)
        item["y"] = float(item["y"]) + rng.uniform(-jammer_position_m, jammer_position_m)
        if "start_time" in item and "end_time" in item:
            start = float(item["start_time"]) + rng.uniform(
                -jammer_window_s, jammer_window_s
            )
            end = float(item["end_time"]) + rng.uniform(-jammer_window_s, jammer_window_s)
            start = max(0.0, start)
            end = max(start + 1.0, end)
            item["start_time"] = round(start, 3)
            item["end_time"] = round(end, 3)
        jammers.append(item)

    overrides: Dict[str, Any] = {}
    if targets:
        overrides["targets"] = targets
    if jammers:
        overrides["jammers"] = jammers
    return overrides


def make_env_for_seed(
    seed: int,
    config_path: str = CONFIG_PATH,
    energy_budget_j: float | None = None,
    jitter: bool = False,
    observation_mode: str = "full",
    history_len: int = 1,
    observation_noise: Any = None,
    measurement_max_tracks: int = 4,
    expose_observation_truth: bool = False,
) -> LpiPowerEnv:
    """按种子构造环境。

    jitter=True 时先做初始条件域随机化（见 `jittered_scenario`），
    用于多种子鲁棒性评测；jitter=False 时保持场景原样，
    只让种子驱动干扰强度起伏（与 main.py / evaluate_dqn.py 完全一致）。

    observation_mode 取 "pomdp" / "ideal" / "realistic" 时分别启用
    全局噪声模型或传感器测量层，观测噪声/测量随机源都绑定到同一种子
    （保证多种子评测可复现）。
    """
    env = make_env(
        config_path,
        energy_budget_j=energy_budget_j,
        observation_mode=observation_mode,
        history_len=history_len,
        observation_noise=observation_noise,
        measurement_max_tracks=measurement_max_tracks,
        expose_observation_truth=expose_observation_truth,
    )
    if jitter:
        env.sim.apply_overrides(extra=jittered_scenario(seed, config_path))
    return env


@dataclass
class PolicySpec:
    """一个基线策略的定义：显示名 + 构造函数 + CSV 文件名。"""

    label: str
    factory: Callable[[], PowerPolicy]
    csv_name: str
    needs_model: bool = False  # True 表示需要完整模型知识（不可部署，仅作参考）


def scripted_policy_specs(
    num_steps: int,
    random_seed: int = RANDOM_POLICY_SEED,
    include_fixed: bool = True,
    include_rule: bool = True,
    include_random: bool = True,
    include_myopic: bool = True,
    include_lookahead: bool = True,
) -> List[PolicySpec]:
    """标准脚本策略集合（每次调用 factory() 得到全新实例，避免状态串味）。"""
    specs: List[PolicySpec] = []
    if include_fixed:
        specs.append(
            PolicySpec("固定功率基线(80W)", lambda: FixedPowerPolicy(), "step_fixed_power.csv")
        )
    if include_rule:
        specs.append(
            PolicySpec("规则功率控制", lambda: RuleBasedPowerPolicy(), "step_rule_based.csv")
        )
    if include_random:
        specs.append(
            PolicySpec(
                "随机策略",
                lambda: RandomPowerPolicy(seed=random_seed),
                "step_random.csv",
            )
        )
    if include_myopic:
        specs.append(
            PolicySpec(
                "逐档贪心(短视)",
                lambda: GreedyOraclePolicy(),
                "step_oracle.csv",
                needs_model=True,
            )
        )
    if include_lookahead:
        specs.append(
            PolicySpec(
                "前瞻规划(非短视)",
                lambda: make_lookahead(num_steps),
                "step_lookahead.csv",
                needs_model=True,
            )
        )
    return specs


def belief_policy_specs(num_steps: int) -> List[PolicySpec]:
    """「只依赖观测」版本的脚本策略集合（部分可观测对比实验用）。

    与 `scripted_policy_specs` 一一对应，但内部策略拿到的是由带噪观测
    搭出的信念状态，而不是真值仿真器。两者成对汇报才能把
    「信息量差异」与「算法能力差异」分开。
    """
    return [
        PolicySpec(
            "规则(仅观测)",
            lambda: BeliefPolicy(RuleBasedPowerPolicy()),
            "step_belief_rule.csv",
        ),
        PolicySpec(
            "逐档贪心(仅观测)",
            lambda: BeliefPolicy(GreedyOraclePolicy()),
            "step_belief_oracle.csv",
            needs_model=True,
        ),
        PolicySpec(
            "前瞻(仅观测)",
            lambda: BeliefPolicy(make_lookahead(num_steps)),
            "step_belief_lookahead.csv",
            needs_model=True,
        ),
    ]


def add_observation_arguments(parser: Any) -> None:
    """给脚本加统一的观测模式 CLI 参数（各脚本共用，避免参数名漂移）。"""
    parser.add_argument(
        "--observation-mode",
        choices=["full", "pomdp", "ideal", "realistic"],
        default="full",
        help="观测模式：full=全真值（v3.1 行为，默认）；pomdp=v4.0 全局噪声模型；"
             "ideal/realistic=v4.2 传感器测量层（后者含噪声/漏检/虚警）",
    )
    parser.add_argument(
        "--observation-preset",
        choices=sorted(OBSERVATION_PRESETS),
        default="moderate",
        help="部分可观测强度预设（仅 pomdp 模式生效）",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=1,
        help="观测历史窗口长度 K（>1 表示堆叠最近 K 帧，用于时序记忆分支）",
    )
    parser.add_argument(
        "--measurement-max-tracks",
        type=int,
        default=4,
        help="测量模式下航迹表槽位数（决定观测维度，仅 ideal/realistic 生效）",
    )


def observation_kwargs_from_args(args: Any) -> Dict[str, Any]:
    """从 argparse 结果里取出观测相关构造参数。"""
    mode = getattr(args, "observation_mode", "full")
    kwargs: Dict[str, Any] = {
        "observation_mode": mode,
        "history_len": int(getattr(args, "history_len", 1) or 1),
    }
    if mode == "pomdp":
        preset = getattr(args, "observation_preset", "moderate")
        kwargs["observation_noise"] = observation_preset(preset)
    if mode in ("ideal", "realistic"):
        kwargs["measurement_max_tracks"] = int(
            getattr(args, "measurement_max_tracks", 4) or 4
        )
    return kwargs


def make_env_from_args(
    args: Any,
    energy_budget_j: float | None = None,
    config_path: str | None = None,
    expose_observation_truth: bool = False,
) -> LpiPowerEnv:
    """按命令行参数构造环境（脚本统一入口，保证训练/评测/诊断用同一套观测设置）。"""
    return make_env(
        config_path or getattr(args, "config", CONFIG_PATH),
        energy_budget_j=energy_budget_j,
        expose_observation_truth=expose_observation_truth,
        horizon_semantics=getattr(args, "horizon_semantics", "finite_task"),
        **observation_kwargs_from_args(args),
    )


def observation_label(args: Any) -> str:
    """人类可读的观测模式标签（写进报告与表头）。"""
    mode = getattr(args, "observation_mode", "full")
    if mode == "full":
        return "全可观"
    preset = getattr(args, "observation_preset", "moderate")
    history = int(getattr(args, "history_len", 1) or 1)
    text = f"部分可观测({preset})"
    if history > 1:
        text += f"+历史窗口K={history}"
    return text


# ----------------------------------------------------------------------
# 统一的 episode 运行器
# ----------------------------------------------------------------------
def run_scripted_episode(env: LpiPowerEnv, policy: PowerPolicy, seed: int) -> List[Any]:
    """用脚本策略跑一个 episode。

    与 DQN 走**同一条复位路径**（env.reset(seed)），保证两者面对完全相同的场景。
    """
    env.reset(seed=seed)
    sim = env.sim
    if hasattr(policy, "reset"):
        policy.reset()
    while not sim.is_done:
        sim.step(policy.select_level(sim))
    return list(sim.results)


def run_dqn_episode(env: LpiPowerEnv, agent: Any, seed: int, greedy: bool = True) -> List[Any]:
    """用 DQN 跑一个 episode（动作选择使用环境观测 + 动作可行性掩码）。"""
    obs, _ = env.reset(seed=seed)
    while True:
        action = agent.select_action(obs, greedy=greedy, action_mask=env.action_masks())
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    return list(env.sim.results)


def run_belief_episode(env: LpiPowerEnv, policy: Any, seed: int) -> List[Any]:
    """用「只依赖观测」的策略跑一个 episode。

    与 `run_scripted_episode` 的区别：脚本策略拿到的是**信念状态**
    （由带噪观测估计值搭出来的仿真器副本），而不是真值仿真器。
    这样「部分可观测下的规则/前瞻」才与 DQN 站在同一信息水平上。

    策略需实现 `select_level_from_env(env)`；带噪策略同时应实现 `reset()`。
    """
    env.reset(seed=seed)
    sim = env.sim
    if hasattr(policy, "reset"):
        policy.reset()
    while not sim.is_done:
        level = policy.select_level_from_env(env)
        env.step(int(level))
    return list(sim.results)


def summarize_episode(
    env: LpiPowerEnv,
    results: Sequence[Any],
    label: str,
    policy_name: str,
) -> Dict[str, Any]:
    """用统一口径汇总一个 episode（指标层口径与 main.py 完全一致）。"""
    scenario = env.sim.scenario
    radar = env.sim.radar
    assert scenario is not None and radar is not None
    return summarize_run(
        list(results),
        label=label,
        policy=policy_name,
        energy_budget_j=radar.energy_budget_j,
        lpi_pint_threshold=scenario.lpi_pint_threshold,
        horizon_steps=scenario.num_steps,
    )


# ----------------------------------------------------------------------
# 统计聚合（多种子评测用）
# ----------------------------------------------------------------------

#: 多种子/敏感性实验需要报告均值与标准差的指标
AGGREGATE_METRICS: Tuple[str, ...] = (
    "horizon_satisfaction_rate",
    "detection_task_satisfaction_rate",
    "violation_rate",
    "explicit_violation_steps",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "remaining_energy_j",
    "energy_utilization",
    "avg_intercept_prob",
    "avg_instant_intercept_prob",
    "avg_exposure",
    "cumulative_exposure",
    "composite_reward",
    "steps",
    "violation_steps",
    "power_switch_count",
)


def aggregate_summaries(
    summaries: Sequence[Dict[str, Any]], label: str, policy_name: str = ""
) -> Dict[str, Any]:
    """把同一策略在多个种子下的汇总结果聚合成「均值 ± 标准差」。

    返回的字典里每个指标同时给出：
        <metric>         均值
        <metric>__std    样本标准差（ddof=1，n<2 时为 0）
        <metric>__min / __max
    """
    if not summaries:
        raise ValueError(f"[{label}] 没有可聚合的结果")

    aggregated: Dict[str, Any] = {
        "label": label,
        "policy": policy_name or summaries[0].get("policy", ""),
        "n_seeds": len(summaries),
    }

    for metric in AGGREGATE_METRICS:
        values = [float(s[metric]) for s in summaries if metric in s]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        aggregated[metric] = mean
        aggregated[f"{metric}__std"] = std
        aggregated[f"{metric}__min"] = min(values)
        aggregated[f"{metric}__max"] = max(values)

    return aggregated


def format_aggregate_table(rows: Sequence[Dict[str, Any]], metrics: Sequence[str] | None = None) -> str:
    """把聚合结果排版成「均值 ± 标准差」文本表。"""
    metrics = metrics or (
        "horizon_satisfaction_rate",
        "avg_tx_power_w",
        "cumulative_energy_j",
        "avg_intercept_prob",
        "avg_exposure",
        "composite_reward",
    )
    header = f"{'策略':<20}" + "".join(f"{m:>22}" for m in metrics)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for metric in metrics:
            mean = float(row.get(metric, 0.0))
            std = float(row.get(f"{metric}__std", 0.0))
            cells.append(f"{mean:>13.4f}±{std:<8.4f}")
        lines.append(f"{row['label']:<20}" + "".join(cells))
    return "\n".join(lines)
