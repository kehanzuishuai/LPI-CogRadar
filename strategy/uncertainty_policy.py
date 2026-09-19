"""不确定度感知的安全回退策略（v4.0 P2）。

要解决的问题
------------
普通 DQN 在任何输入下都会给出一个确定的 argmax。部分可观测条件下，
当观测丢测、延迟严重、或者输入落在训练分布之外时，这个 argmax 可能毫无依据，
但智能体自己**不知道**——它照样输出一个看起来很正常的高功率档。

本模块让策略在「自己不可靠」时主动交出控制权。

三种决策模式（每一步只选一种，并记录原因）
------------------------------------------
1. `ai`           —— 正常使用集成 DQN 的 argmax；
2. `shield`       —— **安全护盾**：仍用 AI 的动作，但若它低于「按估计状态刚好够用」
                     的最低档位，就把它抬到那一档。防的是「AI 选了功率不足的档、
                     任务探测失败」这一类风险；
3. `fallback_rule`—— **完全回退**：把决策整体交给可解释的规则策略（在信念状态上）。

回退触发条件（按优先级，任一命中即回退）
----------------------------------------
| 原因码                        | 条件                                   | 防的是什么           |
|-------------------------------|----------------------------------------|----------------------|
| `observation_severely_degraded`| 观测质量 < `obs_quality_threshold`      | 丢测/延迟太严重      |
| `out_of_distribution`         | OOD 评分 > `ood_threshold`              | 输入陌生             |
| `high_ensemble_disagreement`  | 集成分歧 > 阈值 或 成员投票分歧 > 阈值   | 网络之间互相矛盾     |
| `small_q_margin`              | Q 优势 < `q_margin_threshold`           | 几个动作几乎一样好   |

为什么同时看三个独立信号
------------------------
* 集成分歧（epistemic）在「N 个网络一起错」时**会失效**；
* OOD 评分只看输入分布，不看网络；
* 观测质量只看测量链路，不看网络；
三者互相独立，任何一个都能在另外两个失灵时报警。只用其中一个都会留下盲区。

诚实说明
--------
* 回退**不是**「AI 学会了承认自己不会」。它是一条人工写的阈值规则，
  阈值是人选的，所以「回退率」很大程度上反映的是**阈值设定**，
  而不是智能体的内省能力。报告里必须同时给出多个阈值下的灵敏度，
  不能只报一个好看的数。
* 回退到规则策略**保证不会更差**这件事**不成立**：规则策略本身在部分可观测
  条件下也会错（它用的是带噪信念）。回退的价值在于「可解释 + 有界行为」，
  不是「性能保证」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from strategy.belief_policy import BeliefPolicy
from strategy.power_policy import PowerPolicy, RuleBasedPowerPolicy

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    # ⚠️ 分层纪律：本模块属于**仿真层**（strategy/），必须保持零第三方依赖。
    # 早期版本在运行时 `from rl.ensemble_agent import UncertaintyInfo`，
    # 而 rl/ 依赖 torch —— 于是 `import strategy.uncertainty_policy`
    # 就把 torch 拖进了本该纯标准库的层，
    # base 环境（无 torch）直接 ImportError。
    # 这里改成只在类型检查时导入：运行时不产生任何 rl/torch 依赖。
    # 本模块对智能体只用鸭子类型（调用 agent.uncertainty(...) 并读属性），
    # 因此不需要在运行时引用具体类型。
    from rl.ensemble_agent import UncertaintyInfo

#: 决策模式
MODE_AI = "ai"
MODE_SHIELD = "shield"
MODE_FALLBACK = "fallback_rule"

#: 触发原因码 -> 中文解释（AI 诊断接口直接引用，保证措辞一致）
REASON_CN: Dict[str, str] = {
    "": "不确定度正常，采用 AI 决策",
    "observation_severely_degraded": "观测严重退化（丢测/延迟过多），AI 输入不可信",
    "out_of_distribution": "输入落在训练分布之外（OOD），网络估值无外推依据",
    "high_ensemble_disagreement": "集成各成员对该状态的动作优劣判断互相矛盾",
    "small_q_margin": "最优与次优动作的 Q 值几乎相同，AI 决策缺乏区分度",
}

MODE_CN: Dict[str, str] = {
    MODE_AI: "AI 自主决策",
    MODE_SHIELD: "安全护盾（AI 动作被抬到满足探测要求的最低档）",
    MODE_FALLBACK: "完全回退到规则策略",
}


@dataclass
class FallbackConfig:
    """回退阈值配置。所有阈值都是**人工设定**的超参数。"""

    enabled: bool = True
    fallback_mode: str = "shield"  # "shield" | "fallback_rule"

    # --- 触发阈值 ---
    q_std_threshold: float = 0.35  # 集成分歧（绝对 Q 尺度）
    disagreement_threshold: float = 0.60  # 成员投票分歧比例
    q_margin_threshold: float = 0.05  # Q 优势下限
    ood_threshold: float = 3.0  # 分布外评分
    #: 观测质量下限。这个默认值不是拍脑袋来的：实测三档预设的观测质量分布为
    #: mild 0.656~0.691、moderate 0.580~0.685、severe 0.497~0.671，
    #: 因此 0.60 大致落在「中等偏下的那部分步」上——轻度部分可观测下几乎不触发，
    #: 重度下会稳定触发。阈值本身是人的选择，所以报告里必须同时给出灵敏度扫描。
    obs_quality_threshold: float = 0.60

    # --- 开关：允许逐项关闭，用于消融 ---
    use_ensemble_signal: bool = True
    use_ood_signal: bool = True
    use_observation_signal: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FallbackConfig":
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> None:
        if self.fallback_mode not in ("shield", "fallback_rule"):
            raise ValueError(
                f"fallback_mode={self.fallback_mode!r} 非法，只能是 'shield' 或 'fallback_rule'"
            )
        for name in (
            "q_std_threshold", "disagreement_threshold", "q_margin_threshold",
            "ood_threshold", "obs_quality_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"阈值 {name} 不能为负")


@dataclass
class DecisionRecord:
    """一步决策的完整记录（既进指标统计，也进 AI 诊断解释）。"""

    step_index: int
    action: int
    mode: str
    reason_code: str
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    obs_quality: float = 1.0
    ai_action: int = -1
    shield_action: Optional[int] = None
    rule_action: Optional[int] = None
    triggered: Tuple[str, ...] = ()

    @property
    def reason_cn(self) -> str:
        return REASON_CN.get(self.reason_code, self.reason_code)

    @property
    def mode_cn(self) -> str:
        return MODE_CN.get(self.mode, self.mode)

    @property
    def is_autonomous(self) -> bool:
        return self.mode == MODE_AI

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "mode": self.mode,
            "mode_cn": self.mode_cn,
            "reason_code": self.reason_code,
            "reason_cn": self.reason_cn,
            "triggered": list(self.triggered),
            "obs_quality": round(self.obs_quality, 6),
            "ai_action": self.ai_action,
            "shield_action": self.shield_action,
            "rule_action": self.rule_action,
            "uncertainty": dict(self.uncertainty),
        }


class UncertaintyAwarePolicy:
    """在集成 DQN 之上加一层不确定度监测与安全回退。

    用法：
        policy = UncertaintyAwarePolicy(ensemble_agent)
        obs, _ = env.reset(seed=42)
        policy.reset()
        while not done:
            action, record = policy.select_action(obs, env)
            obs, r, term, trunc, info = env.step(action)
    """

    name = "不确定度感知策略"

    def __init__(
        self,
        agent: Any,
        config: Optional[FallbackConfig] = None,
        fallback_policy: Optional[PowerPolicy] = None,
        conservative_margin_db: float = 0.05,
    ) -> None:
        self.agent = agent
        self.config = config or FallbackConfig()
        self.config.validate()
        # 回退用的规则策略：在**信念状态**上决策，因此它自己也只能看到观测
        self.fallback: Any = fallback_policy or BeliefPolicy(
            RuleBasedPowerPolicy(margin_db=conservative_margin_db)
        )
        self.records: List[DecisionRecord] = []
        self._step_index = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.records = []
        self._step_index = 0
        if hasattr(self.fallback, "reset"):
            self.fallback.reset()

    # ------------------------------------------------------------------

    def _obs_quality(self, env: Any) -> float:
        if hasattr(env, "observation_quality"):
            return float(env.observation_quality())
        return 1.0

    def _evaluate_triggers(
        self, info: UncertaintyInfo, obs_quality: float
    ) -> List[str]:
        """返回命中的原因码列表（已按优先级排序）。"""
        cfg = self.config
        triggered: List[str] = []

        if cfg.use_observation_signal and obs_quality < cfg.obs_quality_threshold:
            triggered.append("observation_severely_degraded")
        if cfg.use_ood_signal and info.ood_score > cfg.ood_threshold:
            triggered.append("out_of_distribution")
        if cfg.use_ensemble_signal and (
            info.q_std_max > cfg.q_std_threshold
            or info.disagreement > cfg.disagreement_threshold
        ):
            triggered.append("high_ensemble_disagreement")
        if (
            cfg.use_ensemble_signal
            and info.n_feasible > 1
            and info.q_margin < cfg.q_margin_threshold
        ):
            triggered.append("small_q_margin")
        return triggered

    def _rule_action(self, env: Any) -> Optional[int]:
        """在信念状态上求规则策略的动作（作为回退/护盾基准）。"""
        try:
            return int(self.fallback.select_level_from_env(env))
        except Exception:
            return None

    # ------------------------------------------------------------------

    def select_action(
        self, obs: Sequence[float], env: Any
    ) -> Tuple[int, DecisionRecord]:
        """返回 (实际执行的动作, 决策记录)。

        模式语义（务必精确，指标全靠它）：
          ai          —— 最终执行的就是 AI 的 argmax（无论是否处于高风险状态）
          shield      —— AI 动作被安全护盾**抬升**了
          fallback_rule —— 整体交给规则策略
        「是否处于高风险状态」由 `record.triggered` 单独表示，与 mode 正交。
        因此「高风险但护盾判断无需干预」会被如实记成 mode=ai + triggered 非空，
        而不会被伪装成一次回退。
        """
        action_mask = env.action_masks()
        info, q_mean = self.agent.uncertainty(obs, action_mask)

        feasible = [i for i, ok in enumerate(action_mask) if ok]
        if not feasible:
            raise RuntimeError("action_mask 全为 False，episode 应当已结束")

        # AI 的原始动作：掩码内 argmax（平局取索引最小者，保证确定性）
        ai_action = feasible[0]
        best_value = float(q_mean[ai_action])
        for level in feasible[1:]:
            value = float(q_mean[level])
            if value > best_value:
                best_value = value
                ai_action = level

        obs_quality = self._obs_quality(env)
        triggered = self._evaluate_triggers(info, obs_quality)
        reason_code = triggered[0] if triggered else ""

        rule_action: Optional[int] = None
        shield_action: Optional[int] = None
        mode = MODE_AI
        action = ai_action

        if triggered and self.config.enabled:
            rule_action = self._rule_action(env)
            if self.config.fallback_mode == "fallback_rule":
                if rule_action is not None:
                    mode = MODE_FALLBACK
                    action = rule_action
                # rule_action 为 None（信念不可用）时保持 AI 动作，mode 仍为 ai
            elif rule_action is not None and rule_action > ai_action:
                # 安全护盾：AI 选的档位低于「按估计状态刚好够用」的最低档 -> 抬升
                mode = MODE_SHIELD
                shield_action = rule_action
                action = rule_action

        clipped = env.sim.clip_to_feasible(action)
        action = int(clipped) if clipped is not None else int(feasible[-1])

        record = DecisionRecord(
            step_index=self._step_index,
            action=action,
            mode=mode,
            reason_code=reason_code,
            uncertainty=info.to_dict(),
            obs_quality=obs_quality,
            ai_action=ai_action,
            shield_action=shield_action,
            rule_action=rule_action,
            triggered=tuple(triggered),
        )
        self.records.append(record)
        self._step_index += 1
        return action, record

    # ------------------------------------------------------------------

    def describe(self) -> str:
        cfg = self.config
        return (
            f"{self.name}（回退模式={cfg.fallback_mode}；"
            f"阈值 q_std>{cfg.q_std_threshold} / 分歧>{cfg.disagreement_threshold} / "
            f"Q优势<{cfg.q_margin_threshold} / OOD>{cfg.ood_threshold} / "
            f"观测质量<{cfg.obs_quality_threshold}）"
        )

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """把记录汇总成 P2 需要的指标。"""
        total = len(self.records)
        if total == 0:
            return {
                "decision_steps": 0,
                "ai_autonomy_rate": 0.0,
                "fallback_rate": 0.0,
                "shield_rate": 0.0,
                "high_risk_rate": 0.0,
                "mode_counts": {},
                "reason_counts": {},
                "mean_obs_quality": 0.0,
                "mean_q_std_max": 0.0,
                "mean_ood_score": 0.0,
            }
        mode_counts: Dict[str, int] = {}
        reason_counts: Dict[str, int] = {}
        for record in self.records:
            mode_counts[record.mode] = mode_counts.get(record.mode, 0) + 1
            if record.reason_code:
                reason_counts[record.reason_code] = reason_counts.get(record.reason_code, 0) + 1
        autonomous = sum(1 for r in self.records if r.is_autonomous)
        shielded = sum(1 for r in self.records if r.mode == MODE_SHIELD)
        fell_back = sum(1 for r in self.records if r.mode == MODE_FALLBACK)
        high_risk = sum(1 for r in self.records if r.triggered)
        return {
            "decision_steps": total,
            "ai_autonomy_rate": autonomous / total,
            "fallback_rate": fell_back / total,
            "shield_rate": shielded / total,
            "high_risk_rate": high_risk / total,
            "mode_counts": mode_counts,
            "reason_counts": reason_counts,
            "mean_obs_quality": sum(r.obs_quality for r in self.records) / total,
            "mean_q_std_max": sum(
                float(r.uncertainty.get("q_std_max", 0.0)) for r in self.records
            ) / total,
            "mean_ood_score": sum(
                float(r.uncertainty.get("ood_score", 0.0)) for r in self.records
            ) / total,
        }
