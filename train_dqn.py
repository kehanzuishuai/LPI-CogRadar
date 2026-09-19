"""DQN 训练入口。

    python train_dqn.py                        # 默认超参，800 episode
    python train_dqn.py --episodes 300 --out-dir output/rl_smoke

环境、状态、动作、奖励全部复用现有工程，不做任何重写：
    环境  engine/env.py::LpiPowerEnv           （Gymnasium 风格 reset/step）
    状态  11 维观测 observation_space: Box(11)
    动作  11 档离散发射功率 action_space: Discrete(11)
    奖励  models/reward.py::composite_reward   （与 main.py 报告的「综合收益」同源）

产物（默认 output/rl/）
    training_log.csv        每个 episode 的训练指标
    eval_log.csv            周期性贪心评测（固定测试场景）指标
    training_curves.html    训练曲线页（episode reward / 满足率 / Pint / 功率 / 能耗）
    training_curves.png     同上的 PNG 版本（需 matplotlib，缺失时跳过）
    dqn_agent.pt            模型 checkpoint（含超参、优化器、ε、步数）
    dqn_agent.json          checkpoint 侧车 JSON（人类可读，便于复现）
    train_config.json       本次训练的完整配置与场景信息

可复现性
--------
* `--seed` 同时固定 python random / numpy / torch / 环境的干扰起伏序列；
* 训练时每个 episode 用 `seed + episode * seed_stride`，让干扰起伏多样化；
* 评测固定用 `--eval-seed`（默认 42，即场景默认种子），
  与 evaluate_dqn.py、main.py 的对照实验完全同一场景。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from engine import LpiPowerEnv, OBSERVATION_FEATURES
from metrics import summarize_run, write_curves_html
from rl import (
    DQNAgent,
    DQNConfig,
    LagrangianConfig,
    LagrangianDQNAgent,
    silence_numpy_bridge_warning,
)
import experiment_config

# 本工程的 RL 路径只用 torch，不依赖 numpy；该调用抑制 torch numpy 桥接的
# 无害告警（详见 rl/dqn_agent.py 的说明）。
silence_numpy_bridge_warning()


def _mean(values: Sequence[float]) -> float:
    """纯 Python 均值，避免为了一个 mean 引入 numpy 依赖。"""
    return sum(values) / len(values) if values else 0.0


DEFAULT_CONFIG_PATH = "config/radar_scenario_v1.json"
DEFAULT_OUT_DIR = os.path.join("output", "rl")

# 训练指标的列定义
TRAIN_LOG_FIELDS = [
    "episode",
    "env_seed",
    "episode_reward",
    "total_reward",
    "detection_task_satisfaction_rate",
    "avg_intercept_prob",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "avg_pd",
    "epsilon",
    "learning_rate",
    "mean_loss",
    "cost_rate",
    "lambda_cost",
    "env_steps",
    "train_steps",
    "wall_time_s",
]

EVAL_LOG_FIELDS = [
    "episode",
    "env_steps",
    "horizon_satisfaction_rate",
    "violation_rate",
    "avg_intercept_prob",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "composite_reward",
    "avg_pd",
    "avg_exposure",
    "power_switch_count",
]


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 LpiPowerEnv 上训练 DQN（低截获雷达功率调控）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="场景配置路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument(
        "--horizon-semantics",
        choices=("finite_task", "legacy_truncation"),
        default="finite_task",
        help=("任务时域语义：finite_task 是自然终点且不 bootstrap；"
              "legacy_truncation 仅用于复现旧 checkpoint"),
    )

    # 训练规模
    parser.add_argument("--episodes", type=int, default=800, help="训练 episode 数")
    parser.add_argument("--seed", type=int, default=0, help="全局随机种子")
    parser.add_argument("--seed-stride", type=int, default=1,
                        help="每个 episode 的种子步长（训练集多样性来源）")
    parser.add_argument("--jitter", action="store_true",
                        help="训练时做初始条件域随机化（目标/干扰机初始位置、干扰时间窗"
                             "随 episode 种子变化）。**不触碰任何物理参数**。"
                             "不做扰动时，模型会过拟合到固定初始条件，"
                             "在多种子鲁棒性评测中会明显劣于规则/前瞻基线")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="周期性评测与最终评测使用的场景种子")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="每隔多少 episode 做一次贪心评测（0 表示不做）")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="每次评测重复的 episode 数（>1 时取平均）")

    # DQN 超参
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                        help="Q 网络隐藏层宽度")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--buffer-size", type=int, default=50000, help="经验回放容量")
    parser.add_argument("--target-sync", type=int, default=200,
                        help="目标网络同步周期（训练步）")
    parser.add_argument("--learning-starts", type=int, default=500,
                        help="开始训练前先收集多少环境步")
    parser.add_argument("--eps-start", type=float, default=1.0, help="ε 初始值")
    parser.add_argument("--eps-end", type=float, default=0.05, help="ε 终值")
    parser.add_argument("--eps-decay-steps", type=int, default=8000,
                        help="ε 线性衰减步数")
    parser.add_argument("--double-dqn", action="store_true",
                        help="启用 Double DQN（默认关闭，保持标准 DQN）")
    parser.add_argument("--lr-decay-every", type=int, default=0,
                        help="每多少 episode 衰减一次学习率（0 = 不衰减）。"
                             "训练后期策略在最优解附近震荡时，衰减 lr 能让它收敛下来")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5,
                        help="每次学习率衰减的乘数")
    # --- 自适应智能干扰机（规则型对手，默认关闭以保持旧实验可复现）---
    parser.add_argument("--adaptive-jammer", action="store_true",
                        help="把干扰机切换为规则自适应智能干扰机（默认固定时间窗）")
    # --- 安全 RL（拉格朗日约束分支，不替换主 DQN）---
    parser.add_argument("--safe-rl", action="store_true",
                        help="启用拉格朗日约束 DQN 分支（双 critic + 动态 λ）")
    parser.add_argument("--cost-limit", type=float, default=0.07,
                        help="目标约束违反率上限 d。取 0.07 是因为：普通 DQN 实测 0.082、"
                             "前瞻规划实测 0.066，0.07 落在两者之间——"
                             "既可行（约束能被满足）又是激活的（必须牺牲一点收益）")
    parser.add_argument("--lambda-lr", type=float, default=0.5,
                        help="拉格朗日乘子的对偶上升步长 η_λ。"
                             "太小（如 0.05）会让 λ 在整个训练预算内都爬不到有效区间，"
                             "约束形同虚设（本项目实测：λ 只到 0.20，违反率反而升到 0.098）")
    parser.add_argument("--lambda-max", type=float, default=1.5,
                        help="拉格朗日乘子上界（与单步奖励量级 ~1.0 匹配）")
    parser.add_argument("--lambda-init", type=float, default=0.6,
                        help="拉格朗日乘子初值（**热启动**）。"
                             "从 0 开始需要很久才能爬到有效区间，"
                             "0.6 大致相当于固定惩罚版本 w_violation=1.5 的一半强度，"
                             "给对偶上升一个合理起点")
    parser.add_argument("--device", default="cpu", help="torch 设备")

    # --- v4.0 部分可观测（默认 full，保证旧实验逐位可复现）---
    experiment_config.add_observation_arguments(parser)

    parser.add_argument("--quiet", action="store_true", help="减少控制台输出")
    return parser


def make_agent(args: argparse.Namespace, env: LpiPowerEnv) -> Any:
    config = DQNConfig(
        obs_dim=int(env.observation_space.shape[0]),
        n_actions=int(env.action_space.n),
        hidden_sizes=tuple(int(h) for h in args.hidden),
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_sync_steps=args.target_sync,
        learning_starts=args.learning_starts,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay_steps,
        double_dqn=args.double_dqn,
        seed=args.seed,
    )
    if getattr(args, "safe_rl", False):
        # 安全 RL 分支：拉格朗日约束 DQN（双 critic + 动态 λ），不替换主 DQN
        return LagrangianDQNAgent(
            config,
            constraint=LagrangianConfig(
                cost_limit=args.cost_limit,
                lambda_lr=args.lambda_lr,
                lambda_max=args.lambda_max,
                lambda_init=args.lambda_init,
            ),
            device=args.device,
        )
    return DQNAgent(config, device=args.device)


# ----------------------------------------------------------------------
# 单次 rollout
# ----------------------------------------------------------------------

def run_episode(
    env: LpiPowerEnv,
    agent: DQNAgent,
    seed: int,
    train: bool,
) -> Dict[str, Any]:
    """跑一个 episode。

    train=True  ：ε-greedy 探索 + 写入回放池 + 批量训练
    train=False ：贪心策略，仅评测，不写入回放池
    """
    obs, _ = env.reset(seed=seed)

    if hasattr(agent, "reset_episode_cost"):
        agent.reset_episode_cost()
        agent.set_episode_horizon(int(env.sim.scenario.num_steps))

    total_reward = 0.0
    satisfied_steps = 0
    steps = 0
    intercept_probs: List[float] = []
    tx_powers: List[float] = []
    pds: List[float] = []
    losses: List[float] = []
    cumulative_energy = 0.0
    clipped_actions = 0
    violated_flags: List[int] = []

    while True:
        # 能量硬约束：只在可行档位内选动作（掩码同时用于探索与利用）
        action = agent.select_action(obs, greedy=not train, action_mask=env.action_masks())

        if train:
            agent.begin_step()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            # Gymnasium 语义：只有 terminated 切断 bootstrap；外部
            # truncated 仍可 bootstrap。有限任务时域由环境报 terminated。
            # next_action_mask 是**下一状态**的可行性掩码，用于对目标 Q 的 max 做 mask
            extra: Dict[str, Any] = {}
            if hasattr(agent, "reset_episode_cost"):
                # 安全 RL：约束代价 c_t = 1[Pd < required_pd]
                cost = 0.0 if info["task_satisfied"] else 1.0
                if terminated:
                    # 能量耗尽：没执行到的任务步同样算"没完成探测任务"，
                    # 与评测口径（violation_rate 以完整任务步数为分母）保持一致
                    remaining = max(0, env.sim.scenario.num_steps - (info["step_index"] + 1))
                    cost += float(remaining)
                extra["cost"] = cost
            agent.store(
                obs,
                action,
                reward,
                next_obs,
                terminated,
                next_action_mask=env.action_masks(),
                **extra,
            )
            loss = agent.maybe_train()
            if loss is not None:
                losses.append(loss)

        total_reward += reward
        satisfied_steps += 1 if info["task_satisfied"] else 0
        intercept_probs.append(float(info["intercept_prob"]))
        tx_powers.append(float(info["tx_power_w"]))
        pds.append(float(info["pd_min"]))
        cumulative_energy = float(info["cumulative_energy_j"])
        steps += 1
        violated_flags.append(0 if info["task_satisfied"] else 1)
        if info.get("action_clipped"):
            clipped_actions += 1

        obs = next_obs
        if done:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "episode_reward": total_reward / steps if steps else 0.0,
        "detection_task_satisfaction_rate": satisfied_steps / steps if steps else 0.0,
        "avg_intercept_prob": _mean(intercept_probs),
        "avg_tx_power_w": _mean(tx_powers),
        "cumulative_energy_j": cumulative_energy,
        "avg_pd": _mean(pds),
        "mean_loss": _mean(losses),
        "clipped_actions": clipped_actions,
        "cost_rate": (
            sum(violated_flags) / steps if steps and violated_flags else 0.0
        ),
    }


def evaluate_agent(
    env: LpiPowerEnv,
    agent: DQNAgent,
    eval_seed: int,
    episodes: int,
) -> Dict[str, Any]:
    """在固定测试场景上做贪心评测，返回平均指标。

    额外复用 metrics.summarize_run()，因此评测口径与 main.py 完全一致。
    """
    keys = [
        "horizon_satisfaction_rate",
        "violation_rate",
        "avg_intercept_prob",
        "avg_tx_power_w",
        "cumulative_energy_j",
        "composite_reward",
        "avg_pd",
        "avg_exposure",
        "power_switch_count",
    ]
    collected: Dict[str, List[float]] = {k: [] for k in keys}

    for episode in range(max(1, episodes)):
        run_episode(env, agent, seed=eval_seed + episode, train=False)
        summary = summarize_run(
            env.sim.results,
            label=f"dqn_eval_{episode}",
            policy="DQN(greedy)",
            energy_budget_j=env.sim.radar.energy_budget_j,  # type: ignore[union-attr]
            lpi_pint_threshold=env.sim.scenario.lpi_pint_threshold,  # type: ignore[union-attr]
            horizon_steps=env.sim.scenario.num_steps,  # type: ignore[union-attr]
        )
        for key in keys:
            collected[key].append(float(summary[key]))

    return {key: _mean(values) for key, values in collected.items()}


# ----------------------------------------------------------------------
# 训练主流程
# ----------------------------------------------------------------------

def train(args: argparse.Namespace, energy_budget_j: float | None = None) -> None:
    """训练一个 DQN。

    energy_budget_j：可选地覆盖能量预算（任务约束）。供
    `sensitivity_energy_budget.py` 在每个预算下独立训练模型使用；
    它**不会**改动任何雷达/目标/侦察机/干扰机物理参数。
    """
    os.makedirs(args.out_dir, exist_ok=True)

    env = experiment_config.make_env_from_args(args, energy_budget_j=energy_budget_j)
    overrides: Dict[str, Any] = {}
    if getattr(args, "adaptive_jammer", False):
        overrides["adaptive_jammer"] = True
    if getattr(args, "safe_rl", False):
        # 约束版：把**逐步**的「未达标固定惩罚」从奖励里去掉，改由代价 critic + λ 显式约束。
        # 但**保留**能量耗尽的终端惩罚（键 terminal）——否则智能体会发现
        # 「烧光能量提前结束」不再有代价，于是疯狂提功率以规避当前违反，
        # 把能量提前耗尽，反而让满足率崩掉（这是一个真实踩过的坑）。
        overrides["reward_weights"] = {"violation": 0.0, "terminal": 1.5}
    if overrides:
        env.sim.apply_overrides(**overrides)

    agent = make_agent(args, env)
    # 评测专用环境：**不带训练期的域随机化**，但保留同样的对手、奖励与观测设定，
    # 保证评测与训练面对同一个 MDP/POMDP。
    eval_env = experiment_config.make_env_from_args(args, energy_budget_j=energy_budget_j)
    if overrides:
        eval_env.sim.apply_overrides(**overrides)

    print("======== DQN 训练 ========")
    print(f"场景          : {args.config}")
    print(f"环境          : {env.metadata['name']}，"
          f"观测 {env.observation_space.shape[0]} 维，动作 {env.action_space}")
    print(f"能量预算      : {env.sim.radar.energy_budget_j} J（硬约束，动作执行前检查）")
    print(f"策略网络      : {agent.describe()}")
    print(f"全局种子      : {args.seed}（每个 episode 种子 = seed + episode*{args.seed_stride}）")
    print(f"评测场景种子  : {args.eval_seed}")
    print(f"训练规模      : {args.episodes} episode × "
          f"{env.sim.scenario.num_steps} 步 = "  # type: ignore[union-attr]
          f"{args.episodes * env.sim.scenario.num_steps} 环境步"  # type: ignore[union-attr]
          )
    print(f"输出目录      : {args.out_dir}\n")

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    best_eval_reward = float("-inf")
    best_checkpoint = os.path.join(args.out_dir, "dqn_agent_best.pt")

    started = time.time()

    for episode in range(1, args.episodes + 1):
        # 学习率衰减：训练后期在最优解附近震荡时让它逐步收敛
        if (
            args.lr_decay_every > 0
            and episode > 1
            and (episode - 1) % args.lr_decay_every == 0
        ):
            for group in agent.optimizer.param_groups:
                group["lr"] *= args.lr_decay_gamma
            if not args.quiet:
                print(f"  [lr] episode {episode}: lr -> "
                      f"{agent.optimizer.param_groups[0]['lr']:.3e}")

        env_seed = args.seed + episode * args.seed_stride
        if args.jitter:
            # 域随机化：只扰动初始条件（位置/时间窗），不动任何物理参数。
            # apply_overrides 会写回缓存配置，因此随后的 reset(seed) 仍然保持这组初始条件。
            env.sim.apply_overrides(
                extra=experiment_config.jittered_scenario(env_seed, args.config)
            )
        result = run_episode(env, agent, seed=env_seed, train=True)

        # 安全 RL：λ 只由**贪心评测**的违反率更新（见下方 eval 分支）。
        # 这里仅在未启用周期评测时退化为使用训练 rollout 的违反率。
        lambda_info: Dict[str, Any] = {}
        if hasattr(agent, "update_lambda") and args.eval_every <= 0:
            lambda_info = agent.update_lambda(episode)

        row = {
            "episode": episode,
            "env_seed": env_seed,
            "episode_reward": round(result["episode_reward"], 6),
            "total_reward": round(result["total_reward"], 6),
            "detection_task_satisfaction_rate": round(
                result["detection_task_satisfaction_rate"], 6
            ),
            "avg_intercept_prob": round(result["avg_intercept_prob"], 6),
            "avg_tx_power_w": round(result["avg_tx_power_w"], 6),
            "cumulative_energy_j": round(result["cumulative_energy_j"], 4),
            "avg_pd": round(result["avg_pd"], 6),
            "epsilon": round(agent.epsilon, 6),
            "learning_rate": agent.optimizer.param_groups[0]["lr"],
            "mean_loss": round(result["mean_loss"], 6),
            "cost_rate": round(result.get("cost_rate", 0.0), 6),
            "lambda_cost": round(lambda_info.get("lambda_after", 0.0), 6),
            "env_steps": agent.env_steps,
            "train_steps": agent.train_steps,
            "wall_time_s": round(time.time() - started, 2),
        }
        train_rows.append(row)

        # --- 周期性贪心评测（固定测试场景）---
        if args.eval_every > 0 and (
            episode % args.eval_every == 0 or episode == args.episodes
        ):
            evaluation = evaluate_agent(eval_env, agent, args.eval_seed, args.eval_episodes)

            # 安全 RL：λ 由**贪心策略**的违反率更新（不是训练 rollout 的违反率）。
            # 训练 rollout 带 ε 探索噪声，用它做对偶上升会把 λ 顶到上界，
            # 反而让部署策略的违反率升高（详见 LagrangianDQNAgent.update_lambda 的说明）。
            if hasattr(agent, "update_lambda"):
                lambda_info = agent.update_lambda(
                    episode, cost_rate_override=evaluation["violation_rate"]
                )
                row["lambda_cost"] = round(lambda_info["lambda_after"], 6)
                if not args.quiet:
                    print(
                        f"  [λ] ep{episode:>4} 贪心违反率={lambda_info['cost_rate']:.4f} "
                        f"(目标 {lambda_info['cost_limit']}) "
                        f"λ: {lambda_info['lambda_before']:.4f} -> "
                        f"{lambda_info['lambda_after']:.4f}"
                    )

            eval_row = {
                "episode": episode,
                "env_steps": agent.env_steps,
                **{k: round(v, 6) for k, v in evaluation.items()},
            }
            eval_rows.append(eval_row)

            if evaluation["composite_reward"] > best_eval_reward:
                best_eval_reward = evaluation["composite_reward"]
                agent.save(
                    best_checkpoint,
                    metadata=_checkpoint_metadata(args, env, agent, episode, evaluation),
                )

            if not args.quiet:
                print(
                    f"[eval @ep{episode:>4}] 综合收益={evaluation['composite_reward']:+.4f}  "
                    f"满足率={evaluation['horizon_satisfaction_rate']:.4f}  "
                    f"平均Pint={evaluation['avg_intercept_prob']:.4f}  "
                    f"平均功率={evaluation['avg_tx_power_w']:.2f}W  "
                    f"能耗={evaluation['cumulative_energy_j']:.1f}J"
                )

        if not args.quiet and (episode % 20 == 0 or episode == 1):
            print(
                f"[train ep{episode:>4}] reward={row['episode_reward']:+.4f}  "
                f"满足率={row['detection_task_satisfaction_rate']:.3f}  "
                f"ε={agent.epsilon:.3f}  loss={row['mean_loss']:.4f}  "
                f"steps={agent.env_steps}"
            )

    elapsed = time.time() - started

    # --- 最终模型 ---
    final_checkpoint = os.path.join(args.out_dir, "dqn_agent.pt")
    final_eval = evaluate_agent(eval_env, agent, args.eval_seed, args.eval_episodes)
    agent.save(
        final_checkpoint,
        metadata=_checkpoint_metadata(args, env, agent, args.episodes, final_eval),
    )

    # --- 落盘 ---
    train_csv = os.path.join(args.out_dir, "training_log.csv")
    eval_csv = os.path.join(args.out_dir, "eval_log.csv")
    _write_csv(train_rows, TRAIN_LOG_FIELDS, train_csv)
    _write_csv(eval_rows, EVAL_LOG_FIELDS, eval_csv)

    curves_html = os.path.join(args.out_dir, "training_curves.html")
    _write_curves(train_rows, eval_rows, curves_html, args)
    _write_curves_png(train_rows, eval_rows, os.path.join(args.out_dir, "training_curves.png"))

    with open(os.path.join(args.out_dir, "dqn_agent.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": json.loads(agent.config.to_json()),
                "metadata": _checkpoint_metadata(args, env, agent, args.episodes, final_eval),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "energy_budget_j": float(env.sim.radar.energy_budget_j),  # type: ignore[union-attr]
                "observation_features": OBSERVATION_FEATURES,
                "power_levels_w": env.sim.power_levels_w,
                "required_pd": env.sim.radar.required_pd,  # type: ignore[union-attr]
                "reward_weights": env.sim.scenario.reward_weights,  # type: ignore[union-attr]
                "wall_time_s": round(elapsed, 2),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n======== 训练完成 ========")
    print(f"耗时            : {elapsed:.1f} s")
    print(f"环境步 / 训练步 : {agent.env_steps} / {agent.train_steps}")
    print(f"最终 ε          : {agent.epsilon:.4f}")
    print(f"固定测试场景评测: 综合收益={final_eval['composite_reward']:+.4f}  "
          f"满足率={final_eval['horizon_satisfaction_rate']:.4f}  "
          f"平均Pint={final_eval['avg_intercept_prob']:.4f}  "
          f"平均功率={final_eval['avg_tx_power_w']:.2f}W  "
          f"能耗={final_eval['cumulative_energy_j']:.1f}J")
    print(f"最优评测综合收益: {best_eval_reward:+.4f}（{best_checkpoint}）")
    print("\n输出文件：")
    for path in [
        train_csv,
        eval_csv,
        curves_html,
        os.path.join(args.out_dir, "training_curves.png"),
        final_checkpoint,
        best_checkpoint,
        os.path.join(args.out_dir, "dqn_agent.json"),
        os.path.join(args.out_dir, "train_config.json"),
    ]:
        if os.path.exists(path):
            print(f"  {path}")

    print("\n下一步：python evaluate_dqn.py   # 与固定80W / 规则 / 随机 统一对比")


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------

def _checkpoint_metadata(
    args: argparse.Namespace,
    env: LpiPowerEnv,
    agent: DQNAgent,
    episode: int,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """写进 checkpoint 的复现信息。"""
    meta = {
        "scenario": env.sim.scenario.scenario_name,  # type: ignore[union-attr]
        "config_path": os.path.abspath(args.config),
        "train_seed": args.seed,
        "eval_seed": args.eval_seed,
        "episode": episode,
        "env_steps": agent.env_steps,
        "train_steps": agent.train_steps,
        "epsilon": round(agent.epsilon, 6),
        "hidden_sizes": [int(h) for h in agent.config.hidden_sizes],
        "observation_features": OBSERVATION_FEATURES,
        "power_levels_w": env.sim.power_levels_w,
        "eval_metrics": {k: round(float(v), 6) for k, v in evaluation.items()},
        "adaptive_jammer": bool(getattr(args, "adaptive_jammer", False)),
    }
    # 安全 RL 的约束元数据
    if hasattr(agent, "lambda_cost"):
        meta.update(
            {
                "agent_kind": "lagrangian_dqn",
                "cost_limit": float(agent.constraint.cost_limit),
                "lambda_lr": float(agent.constraint.lambda_lr),
                "lambda_max": float(agent.constraint.lambda_max),
                "lambda_cost": round(float(agent.lambda_cost), 6),
            }
        )
    return meta


def _write_csv(rows: List[Dict[str, Any]], fields: List[str], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_curves(
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    path: str,
    args: argparse.Namespace,
) -> None:
    """输出训练曲线页（训练 episode 曲线 + 固定场景贪心评测曲线）。"""
    episodes = [r["episode"] for r in train_rows]
    eval_episodes = [r["episode"] for r in eval_rows]

    def series(train_key: str, eval_key: Optional[str] = None) -> Dict[str, Dict[str, list]]:
        data = {
            "训练（ε-greedy）": {
                "x": episodes,
                "y": [r[train_key] for r in train_rows],
            }
        }
        if eval_key and eval_rows:
            data["评测（贪心，固定场景）"] = {
                "x": eval_episodes,
                "y": [r[eval_key] for r in eval_rows],
            }
        return data

    curves = [
        {
            "title": "Episode 平均收益 (综合收益)",
            "x_label": "episode",
            "y_label": "reward",
            "series": series("episode_reward", "composite_reward"),
        },
        {
            "title": "探测任务满足率（完整任务步数口径）",
            "x_label": "episode",
            "y_label": "满足率",
            "series": series(
                "detection_task_satisfaction_rate", "horizon_satisfaction_rate"
            ),
        },
        {
            "title": "平均截获概率 Pint",
            "x_label": "episode",
            "y_label": "Pint",
            "series": series("avg_intercept_prob", "avg_intercept_prob"),
        },
        {
            "title": "平均发射功率 (W)",
            "x_label": "episode",
            "y_label": "Pt (W)",
            "series": series("avg_tx_power_w", "avg_tx_power_w"),
        },
        {
            "title": "累计能耗 (J)",
            "x_label": "episode",
            "y_label": "E (J)",
            "series": series("cumulative_energy_j", "cumulative_energy_j"),
        },
        {
            "title": "ε 探索率与训练损失",
            "x_label": "episode",
            "y_label": "值",
            "series": {
                "ε": {"x": episodes, "y": [r["epsilon"] for r in train_rows]},
                "mean_loss": {"x": episodes, "y": [r["mean_loss"] for r in train_rows]},
            },
        },
    ]
    # 损失可能量级很小，单独给它一张图更易读
    curves.append(
        {
            "title": "训练损失 (Huber)",
            "x_label": "episode",
            "y_label": "loss",
            "series": {
                "mean_loss": {"x": episodes, "y": [r["mean_loss"] for r in train_rows]}
            },
        }
    )

    write_curves_html(
        curves,
        path,
        page_title="DQN 训练曲线 —— 低截获雷达功率调控",
        subtitle=(
            f"训练 {len(train_rows)} episode；"
            f"评测固定使用场景种子 {args.eval_seed}（与 evaluate_dqn.py / main.py 同一场景）"
        ),
        metadata={
            "全局种子": args.seed,
            "评测种子": args.eval_seed,
            "学习率": args.lr,
            "折扣因子 γ": args.gamma,
            "批大小": args.batch_size,
            "回放容量": args.buffer_size,
            "目标网络同步周期": f"{args.target_sync} 训练步",
            "ε 调度": f"{args.eps_start} → {args.eps_end}（{args.eps_decay_steps} 步内线性衰减）",
            "Q 网络": f"{args.hidden}",
            "Double DQN": args.double_dqn,
        },
    )


def _configure_matplotlib_cjk() -> bool:
    """给 matplotlib 配置中文字体，返回是否配置成功。

    找到中文字体则 PNG 用中文标题；找不到就退回英文标题，
    避免出现「豆腐块」缺字（matplotlib 默认字体不含 CJK 字形）。
    """
    import matplotlib
    from matplotlib import font_manager

    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate),
                fallback_to_default=False,
            )
        except Exception:
            continue
        matplotlib.rcParams["font.sans-serif"] = [candidate] + list(
            matplotlib.rcParams.get("font.sans-serif", [])
        )
        matplotlib.rcParams["axes.unicode_minus"] = False
        return True
    return False


def _write_curves_png(
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    path: str,
) -> None:
    """可选：用 matplotlib 输出 PNG 曲线，便于直接放进论文。缺失则跳过。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    episodes = [r["episode"] for r in train_rows]
    eval_episodes = [r["episode"] for r in eval_rows]
    use_cjk = _configure_matplotlib_cjk()

    # (中文标题, 英文标题, 训练列, 评测列, ylabel)
    panels = [
        ("Episode 平均收益", "Episode mean reward",
         "episode_reward", "composite_reward", "reward"),
        ("探测任务满足率", "Detection task satisfaction rate",
         "detection_task_satisfaction_rate", "horizon_satisfaction_rate", "rate"),
        ("平均截获概率 Pint", "Mean intercept probability",
         "avg_intercept_prob", "avg_intercept_prob", "Pint"),
        ("平均发射功率", "Mean transmit power",
         "avg_tx_power_w", "avg_tx_power_w", "Pt (W)"),
        ("累计能耗", "Cumulative energy",
         "cumulative_energy_j", "cumulative_energy_j", "E (J)"),
        ("ε 探索率", "Epsilon",
         "epsilon", None, "epsilon"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (zh_title, en_title, train_key, eval_key, ylabel) in zip(
        axes.ravel(), panels
    ):
        axis.plot(episodes, [r[train_key] for r in train_rows], lw=1.0,
                  label="训练 train" if use_cjk else "train")
        if eval_key and eval_rows:
            axis.plot(
                eval_episodes,
                [r[eval_key] for r in eval_rows],
                lw=1.6, marker="o", ms=3,
                label="评测 eval (greedy)" if use_cjk else "eval (greedy)",
            )
        axis.set_title(zh_title if use_cjk else en_title, fontsize=11)
        axis.set_xlabel("episode", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        "DQN 训练曲线 —— 低截获雷达功率调控" if use_cjk
        else "DQN training curves - LPI radar power control",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
