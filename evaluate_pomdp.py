"""P1 实验：部分可观测（POMDP）条件下的智能功率调控对比。

这个脚本回答 v4.0 的第一个研究问题：

    **在只能看到带噪、延迟、丢测的观测时，AI 策略是否仍然有效？**

对比设计的核心难点（也是本脚本存在的理由）
------------------------------------------
工程里所有脚本基线（规则 / 短视 / 前瞻）原本都直接读**真值仿真器**。
如果拿它们去和「只能看部分观测的 DQN」比，比出来的差距里混着「信息量差异」，
不能归因于算法能力。因此本脚本分成三组：

A. 全状态参考组（读真值）
   固定80W / 规则 / 随机 / 短视 / 前瞻
   —— 它们是**上界参考**，信息量高于任何 POMDP 策略，只用于标定难度，
      不参与「谁更好」的排名。

B. 仅观测组（读带噪信念状态，`strategy/belief_policy.py`）
   规则(仅观测) / 短视(仅观测) / 前瞻(仅观测)
   —— 与 DQN 站在**同一信息水平**，这才是公平对照。

C. 学习组
   DQN(全可观)          —— 在 full 模式训练与评测，作为「观测无损时」的参照
   DQN(部分可观测)      —— 在 pomdp 模式训练与评测
   DQN(部分可观测+历史) —— 在 pomdp 模式 + 历史窗口堆叠训练与评测（时序记忆分支）

消融实验（`--ablation`）
------------------------
逐项关掉某一类噪声（只关距离 / 只关干扰 / 只关战斗部能量 / 只关隐藏真值），
看性能损失主要来自哪一路观测退化。`--ablation` 会额外跑一遍。

复现旧的 v3.1 结果
------------------
`--full-only` 只跑 A 组与 full 模式的 DQN，输出与 evaluate_dqn.py 一致；
不传 `--observation-*` 参数时默认 `full`，因此对旧实验没有任何影响。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_config as ec
from rl.dqn_agent import DQNAgent, silence_numpy_bridge_warning

silence_numpy_bridge_warning()

DEFAULT_OUT_DIR = "output/pomdp"
DEFAULT_FULL_MODEL = os.path.join("output", "rl", "dqn_agent_best.pt")
DEFAULT_POMDP_MODEL = os.path.join("output", "rl_pomdp_1200", "dqn_agent_best.pt")
DEFAULT_POMDP_HIST_MODEL = os.path.join("output", "rl_pomdp_hist_1200", "dqn_agent_best.pt")
DEFAULT_FULL_MODEL_MATCHED = os.path.join("output", "rl_full_1200", "dqn_agent_best.pt")


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P1：部分可观测条件下的策略对比（v4.0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ec.DEFAULT_SEEDS),
                        help="多种子评测的种子列表")
    parser.add_argument("--no-jitter", action="store_true",
                        help="关闭初始条件域随机化（默认开启：多种子才有意义）")
    parser.add_argument("--energy-budget", type=float, default=None,
                        help="覆盖能量预算（任务约束）")
    parser.add_argument("--device", default="cpu", help="torch 设备")

    # 部分可观测设置
    ec.add_observation_arguments(parser)
    parser.add_argument("--full-model", default=DEFAULT_FULL_MODEL,
                        help="full 模式下训练的 DQN checkpoint")
    parser.add_argument("--pomdp-model", default=DEFAULT_POMDP_MODEL,
                        help="pomdp 模式下训练的 DQN checkpoint")
    parser.add_argument("--pomdp-hist-model", default=DEFAULT_POMDP_HIST_MODEL,
                        help="pomdp + 历史窗口训练的 DQN checkpoint")
    parser.add_argument("--hist-model-history-len", type=int,
                        default=ec.DEFAULT_HISTORY_LEN,
                        help="历史窗口模型训练时用的帧数 K。"
                             "它决定该臂评测环境的观测维度（16×K），"
                             "必须与 checkpoint 匹配，否则会被跳过")
    parser.add_argument("--full-model-matched", default=DEFAULT_FULL_MODEL_MATCHED,
                        help="与 POMDP 臂**同训练预算**的全可观 DQN checkpoint。"
                             "用来把「训练预算差异」从「观测模式差异」里剥离出去")

    parser.add_argument("--full-only", action="store_true",
                        help="只评测全可观组（等价于旧的 evaluate_dqn 口径）")
    parser.add_argument("--ablation", action="store_true",
                        help="额外跑逐项噪声消融")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="跳过前瞻规划（计算量大）")
    parser.add_argument("--no-baselines", action="store_true",
                        help="跳过脚本基线，只评测 DQN")
    return parser


# ----------------------------------------------------------------------
# 单次评测
# ----------------------------------------------------------------------

def _load_agent(path: str, device: str) -> Optional[DQNAgent]:
    if not path or not os.path.exists(path):
        return None
    return DQNAgent.load(path, device=device)


def _run_dqn(env: Any, agent: DQNAgent, seed: int) -> List[Any]:
    return ec.run_dqn_episode(env, agent, seed=seed, greedy=True)


def evaluate_arm(
    label: str,
    out_dir: str,
    seeds: Sequence[int],
    config_path: str,
    energy_budget_j: Optional[float],
    jitter: bool,
    kind: str,
    policy_factory: Any = None,
    agent_path: Optional[str] = None,
    observation_mode: str = "full",
    history_len: int = 1,
    observation_noise: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """评测一个实验臂，返回聚合结果（含多种子均值±标准差）。

    kind: "policy"（需要真值）| "belief"（只依赖观测）| "dqn"
    """
    agent = None
    if kind == "dqn":
        agent = _load_agent(agent_path or "", device)
        if agent is None:
            print(f"  [跳过] {label}：找不到 checkpoint {agent_path}")
            return None

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        env = ec.make_env_for_seed(
            seed,
            config_path=config_path,
            energy_budget_j=energy_budget_j,
            jitter=jitter,
            observation_mode=observation_mode,
            history_len=history_len,
            observation_noise=observation_noise,
        )

        if kind == "dqn":
            assert agent is not None
            obs_dim = int(env.observation_space.shape[0])
            if obs_dim != agent.config.obs_dim:
                print(
                    f"  [跳过] {label}：观测维度不匹配"
                    f"（环境 {obs_dim} vs 模型 {agent.config.obs_dim}）"
                )
                return None
            results = _run_dqn(env, agent, seed)
        elif kind == "belief":
            policy = policy_factory()
            results = ec.run_belief_episode(env, policy, seed)
        else:
            policy = policy_factory()
            results = ec.run_scripted_episode(env, policy, seed)

        rows.append(ec.summarize_episode(env, results, label=label, policy_name=label))

    aggregate = ec.aggregate_summaries(rows, label=label)
    aggregate["observation_mode"] = observation_mode
    aggregate["history_len"] = history_len
    aggregate["group"] = {
        "policy": "A_全状态参考",
        "belief": "B_仅观测",
        "dqn": "C_学习",
    }[kind]
    return aggregate


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------

def _write_table_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["group", "label", "observation_mode", "history_len"] + list(ec.AGGREGATE_METRICS)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def format_grouped_table(rows: Sequence[Dict[str, Any]]) -> str:
    """按「信息水平」分组打印，避免把不同信息量的策略混在一起排名。

    注意：同一组必须**一次性**交给 format_aggregate_table 排版，
    否则每行都会重复打印一遍表头。早先逐行调用的写法就是这样，
    输出里表头刷屏、真正的数据行被淹没。
    """
    if not rows:
        return "（无结果）"
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for row in rows:
        group = row.get("group", "")
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(row)

    lines: List[str] = []
    for group in order:
        lines.append("")
        lines.append(f"---------- {group} ----------")
        lines.append(ec.format_aggregate_table(groups[group]).rstrip())
    return "\n".join(lines)


def _write_text_html(path: str, title: str, sections: Sequence[Tuple[str, Sequence[str]]]) -> str:
    """把纯文本段落包成一份自包含 HTML 报告。

    为什么不用 metrics/report.py 的 build_html_report：
    那个函数是给「逐 episode 曲线 + 逐策略汇总」设计的，签名要求传入
    `runs` 明细。本脚本的结论是**多种子聚合**结果，没有等价的 runs 明细，
    硬套会得到一份空壳报告。这里直接输出文本块，虽然朴素但内容完整。
    """
    import html as html_mod

    parts = [        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{html_mod.escape(title)}</title>",
        "<style>",
        "body{font-family:Consolas,'Microsoft YaHei',monospace;margin:24px;"
        "background:#fafafa;color:#222;line-height:1.55}",
        "h1{font-size:20px}h2{font-size:16px;margin-top:28px;"
        "border-left:4px solid #4472c4;padding-left:8px}",
        "pre{background:#fff;border:1px solid #ddd;border-radius:4px;"
        "padding:12px;overflow-x:auto;font-size:12.5px}",
        "</style></head><body>",
        f"<h1>{html_mod.escape(title)}</h1>",
    ]
    for heading, blocks in sections:
        parts.append(f"<h2>{html_mod.escape(heading)}</h2>")
        for block in blocks:
            parts.append(f"<pre>{html_mod.escape(str(block))}</pre>")
    parts.append("</body></html>")
    text = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def build_conclusions(rows: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> List[str]:
    """生成结论文字。**只陈述数据支持的结论**，不夸大。"""
    by_label = {row["label"]: row for row in rows}
    lines: List[str] = [f"评测种子数：{len(seeds)}（{list(seeds)}）", ""]

    def get(label: str, metric: str) -> Optional[float]:
        row = by_label.get(label)
        if row is None:
            return None
        value = row.get(metric)
        return None if value is None else float(value)

    def show(label: str) -> str:
        row = by_label.get(label)
        if row is None:
            return f"{label}：未评测"
        return (
            f"{label}：满足率 {float(row.get('horizon_satisfaction_rate', 0.0)):.4f}"
            f"±{float(row.get('horizon_satisfaction_rate__std', 0.0)):.4f}，"
            f"综合收益 {float(row.get('composite_reward', 0.0)):+.4f}"
            f"±{float(row.get('composite_reward__std', 0.0)):+.4f}，"
            f"平均功率 {float(row.get('avg_tx_power_w', 0.0)):.2f} W"
        )

    # 1) 部分可观测对各类策略的影响
    # 注意：两组的标签由不同的工厂生成，必须显式列出配对，
    # 不能用 f"{name}(全状态)" 去拼——A 组用的是 PolicySpec.label
    # （如「规则功率控制」），B 组用的是「规则」。拼错过一次，结果整节静默为空。
    lines.append("【1】部分可观测的代价（仅观测组 vs 全状态参考组）")
    label_pairs = (
        ("规则", "规则功率控制(全状态)", "规则(仅观测)"),
        ("逐档贪心(短视)", "逐档贪心(短视)(全状态)", "逐档贪心(仅观测)"),
        ("前瞻规划(非短视)", "前瞻规划(非短视)(全状态)", "前瞻(仅观测)"),
    )
    printed_any = False
    for name, full_label, obs_label in label_pairs:
        full_val = get(full_label, "composite_reward")
        obs_val = get(obs_label, "composite_reward")
        if full_val is None or obs_val is None:
            continue
        printed_any = True
        delta = obs_val - full_val
        lines.append(
            f"  {name}：全状态 {full_val:+.4f} → 仅观测 {obs_val:+.4f}"
            f"（Δ {delta:+.4f}）"
        )
        full_sat = get(full_label, "horizon_satisfaction_rate")
        obs_sat = get(obs_label, "horizon_satisfaction_rate")
        if full_sat is not None and obs_sat is not None:
            lines.append(
                f"      满足率 {full_sat:.4f} → {obs_sat:.4f}"
                f"（Δ {obs_sat - full_sat:+.4f}，即 {(obs_sat - full_sat) * 100:+.1f} 个百分点）"
            )
    if not printed_any:
        lines.append("  （两组标签未同时出现，无法配对；请确认是否用了 --full-only 或 --no-baselines）")
    lines.append("")

    # 2) DQN 在部分可观测下是否仍然有效
    lines.append("【2】DQN 在部分可观测下是否仍然有效")
    obs_rule_return = get("规则(仅观测)", "composite_reward")
    dqn_pomdp_return = get("DQN(部分可观测)", "composite_reward")
    dqn_full_return = get("DQN(全可观)", "composite_reward")
    if obs_rule_return is not None and dqn_pomdp_return is not None:
        delta = dqn_pomdp_return - obs_rule_return
        verdict = "优于" if delta > 0 else ("持平" if abs(delta) < 1e-6 else "不如")
        lines.append(
            f"  DQN(部分可观测) {dqn_pomdp_return:+.4f} vs 规则(仅观测) "
            f"{obs_rule_return:+.4f} → {verdict}，Δ {delta:+.4f}"
        )
        lines.append(
            "  注意：这是**同一信息水平**下的对比，可以归因于算法差异。"
        )
    if dqn_full_return is not None and dqn_pomdp_return is not None:
        lines.append(
            f"  DQN 自身（**训练预算不同**，仅供参考）：全可观 {dqn_full_return:+.4f} → "
            f"部分可观测 {dqn_pomdp_return:+.4f}，Δ {dqn_pomdp_return - dqn_full_return:+.4f}"
        )
        lines.append(
            "    主 DQN 训练 4500 episode，POMDP 臂训练 1200 episode，"
            "这个差值里混着训练预算差异，**不能**直接归因于观测模式。"
        )
    matched = get("DQN(全可观,匹配预算)", "composite_reward")
    if matched is not None and dqn_pomdp_return is not None:
        delta = dqn_pomdp_return - matched
        lines.append(
            f"  同预算对照（都 1200 episode）：全可观 {matched:+.4f} → 部分可观测 "
            f"{dqn_pomdp_return:+.4f}，Δ {delta:+.4f}"
        )
        lines.append(
            "    ← **这个差值才是部分可观测带来的真实代价**，因为两边只有观测模式不同。"
        )
    else:
        lines.append(
            "  缺少同预算全可观对照（--full-model-matched），"
            "无法把观测模式的影响与训练预算的影响分离。"
        )
    lines.append("")

    # 3) 历史窗口分支
    lines.append("【3】历史窗口编码是否缓解部分可观测")
    dqn_hist = get("DQN(部分可观测+历史)", "composite_reward")
    if dqn_hist is not None and dqn_pomdp_return is not None:
        delta = dqn_hist - dqn_pomdp_return
        verdict = "有改善" if delta > 0 else ("无改善" if abs(delta) < 1e-6 else "反而变差")
        lines.append(f"  无窗口 {dqn_pomdp_return:+.4f} → 有窗口 {dqn_hist:+.4f} → {verdict}（Δ {delta:+.4f}）")
    else:
        lines.append("  历史窗口分支未评测（缺少 checkpoint，先运行 train_dqn.py --history-len 4）")
    lines.append("")

    lines.append("【逐条明细】")
    for row in rows:
        lines.append("  " + show(row["label"]))
    return lines


# ----------------------------------------------------------------------
# 消融
# ----------------------------------------------------------------------

#: 逐项噪声消融：(名称, 覆盖项)。只关掉一路，其余保持 moderate。
ABLATION_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("全噪声(moderate)", {}),
    ("只关距离噪声", {"range_sigma_m": 0.0, "rcs_sigma_m2": 0.0}),
    ("只关干扰噪声", {"jam_ratio_sigma": 0.0}),
    ("只关能量噪声", {"energy_sigma_j": 0.0}),
    ("只关暴露/Pint噪声", {"exposure_sigma": 0.0, "pint_sigma": 0.0, "pd_sigma": 0.0}),
    ("只关延迟", {"delay_steps": 0}),
    ("只关丢测", {"dropout_prob": 0.0}),
    ("真值全开(仅保留延迟/丢测)", {
        "hide_interceptor_truth": False,
        "hide_pint_truth": False,
        "hide_exposure_truth": False,
    }),
)


def run_ablation(
    args: argparse.Namespace,
    seeds: Sequence[int],
    jitter: bool,
) -> List[Dict[str, Any]]:
    """逐项关掉一路噪声，看仅观测规则策略的性能如何变化。"""
    base = ec.observation_preset("moderate")
    rows: List[Dict[str, Any]] = []
    for name, override in ABLATION_CASES:
        noise = dict(base)
        noise.update(override)
        label = f"仅观测规则｜{name}"
        result = evaluate_arm(
            label=label,
            out_dir=args.out_dir,
            seeds=seeds,
            config_path=args.config,
            energy_budget_j=args.energy_budget,
            jitter=jitter,
            kind="belief",
            policy_factory=lambda: ec.BeliefPolicy(ec.RuleBasedPowerPolicy()),
            observation_mode="pomdp",
            history_len=1,
            observation_noise=noise,
            device=args.device,
        )
        if result:
            rows.append(result)
            print(
                f"  {name:24s} 满足率={result['horizon_satisfaction_rate']:.4f}"
                f"  综合收益={result['composite_reward']:+.4f}"
                f"  平均功率={result['avg_tx_power_w']:.2f}W"
            )
    return rows


# ----------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(args.seeds)
    jitter = not args.no_jitter

    obs_label = ec.observation_label(args)
    print("======== P1：部分可观测条件下的策略对比 ========")
    print(f"场景            : {args.config}")
    print(f"观测模式        : {obs_label}")
    print(f"评测种子        : {len(seeds)} 个 {seeds}")
    print(f"初始条件扰动    : {'开' if jitter else '关'}")
    print(f"输出目录        : {args.out_dir}\n")

    pomdp_kwargs = ec.observation_kwargs_from_args(args)
    if args.full_only:
        pomdp_kwargs = {"observation_mode": "full", "history_len": 1}

    rows: List[Dict[str, Any]] = []

    # ---------------- A 组：全状态参考 ----------------
    if not args.no_baselines:
        print("---- A 组：全状态参考基线（读真值，信息量高于任何 POMDP 策略，仅作上界参考）----")
        full_specs = ec.scripted_policy_specs(
            ec.scenario_horizon(args.config, args.energy_budget),
            include_random=True,
            include_myopic=True,
            include_lookahead=not args.no_lookahead,
        )
        for spec in full_specs:
            label = f"{spec.label}(全状态)"
            result = evaluate_arm(
                label=label,
                out_dir=args.out_dir,
                seeds=seeds,
                config_path=args.config,
                energy_budget_j=args.energy_budget,
                jitter=jitter,
                kind="policy",
                policy_factory=spec.factory,
                observation_mode="full",
                device=args.device,
            )
            if result:
                rows.append(result)
                print(
                    f"  {label:28s} 满足率={result['horizon_satisfaction_rate']:.4f}"
                    f"  综合收益={result['composite_reward']:+.4f}"
                )

    # ---------------- B 组：仅观测 ----------------
    if not args.no_baselines and not args.full_only:
        print("\n---- B 组：仅观测基线（读带噪信念状态，与 DQN 同一信息水平）----")
        for spec in ec.belief_policy_specs(
            ec.scenario_horizon(args.config, args.energy_budget)
        ):
            if args.no_lookahead and "前瞻" in spec.label:
                continue
            result = evaluate_arm(
                label=spec.label,
                out_dir=args.out_dir,
                seeds=seeds,
                config_path=args.config,
                energy_budget_j=args.energy_budget,
                jitter=jitter,
                kind="belief",
                policy_factory=spec.factory,
                **pomdp_kwargs,
                device=args.device,
            )
            if result:
                rows.append(result)
                print(
                    f"  {spec.label:28s} 满足率={result['horizon_satisfaction_rate']:.4f}"
                    f"  综合收益={result['composite_reward']:+.4f}"
                )

    # ---------------- C 组：学习 ----------------
    print("\n---- C 组：学习型策略 ----")
    dqn_arms: List[Tuple[str, str, Dict[str, Any]]] = []
    if args.full_only:
        dqn_arms.append(("DQN(全可观)", args.full_model, {"observation_mode": "full", "history_len": 1}))
    else:
        dqn_arms.append(("DQN(全可观)", args.full_model, {"observation_mode": "full", "history_len": 1}))
        dqn_arms.append((
            "DQN(全可观,匹配预算)",
            args.full_model_matched,
            {"observation_mode": "full", "history_len": 1},
        ))
        dqn_arms.append(("DQN(部分可观测)", args.pomdp_model, {"observation_mode": "pomdp", "history_len": 1}))
        dqn_arms.append((
            "DQN(部分可观测+历史)",
            args.pomdp_hist_model,
            {
                "observation_mode": "pomdp",
                "history_len": int(args.hist_model_history_len),
            },
        ))

    for label, model_path, kwargs in dqn_arms:
        model_kwargs = dict(kwargs)
        if model_kwargs["observation_mode"] == "pomdp":
            model_kwargs["observation_noise"] = ec.observation_preset(
                args.observation_preset
            )
        result = evaluate_arm(
            label=label,
            out_dir=args.out_dir,
            seeds=seeds,
            config_path=args.config,
            energy_budget_j=args.energy_budget,
            jitter=jitter,
            kind="dqn",
            agent_path=model_path,
            device=args.device,
            **model_kwargs,
        )
        if result:
            rows.append(result)
            print(
                f"  {label:28s} 满足率={result['horizon_satisfaction_rate']:.4f}"
                f"  综合收益={result['composite_reward']:+.4f}"
                f"  平均功率={result['avg_tx_power_w']:.2f}W"
            )

    # ---------------- 汇总 ----------------
    print("\n======== 分组汇总（不同信息水平不混排）========")
    print(format_grouped_table(rows))

    print("\n======== 结论 ========")
    conclusions = build_conclusions(rows, seeds)
    for line in conclusions:
        print(line)

    table_csv = os.path.join(args.out_dir, "pomdp_comparison.csv")
    _write_table_csv(table_csv, rows)
    print(f"\n→ {table_csv}")

    # ---------------- 消融 ----------------
    ablation_rows: List[Dict[str, Any]] = []
    if args.ablation and not args.full_only:
        print("\n======== 逐项噪声消融（仅观测规则策略，moderate 基准）========")
        ablation_rows = run_ablation(args, seeds, jitter)
        if ablation_rows:
            ablation_csv = os.path.join(args.out_dir, "pomdp_ablation.csv")
            _write_table_csv(ablation_csv, ablation_rows)
            print(f"→ {ablation_csv}")

    # ---------------- HTML ----------------
    sections = [
        ("1. 说明", [
            "本报告对比「全状态参考组 / 仅观测组 / 学习组」三类策略在部分可观测条件下的表现。",
            "全状态参考组直接读真值仿真器，信息量高于任何 POMDP 策略，因此只作为上界参考，不参与排名。",
            "仅观测组通过 strategy/belief_policy.py 构造的信念状态做决策，与 DQN 站在同一信息水平。",
            f"观测设置：{obs_label}；种子数 {len(seeds)}；初始条件扰动 {'开' if jitter else '关'}。",
        ]),
        ("2. 分组结果", [format_grouped_table(rows)]),
        ("3. 结论", conclusions),
    ]
    if ablation_rows:
        sections.append(("4. 逐项噪声消融", [format_grouped_table(ablation_rows)]))

    html = _write_text_html(
        os.path.join(args.out_dir, "pomdp_report.html"),
        title="LPI-CogRadar v4.0 — P1 部分可观测对比报告",
        sections=sections,
    )
    print(f"→ {html}")


if __name__ == "__main__":
    main()
