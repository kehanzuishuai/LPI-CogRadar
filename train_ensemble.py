"""训练集成 DQN（Bootstrapped Ensemble），供 P2 的不确定度回退使用。

与 train_dqn.py 的关系
----------------------
训练循环、指标口径、日志格式都刻意与 train_dqn.py 保持一致
（同一套 episode 运行方式、同样的 horizon_satisfaction_rate 口径），
差别只在于智能体换成 `EnsembleDQNAgent`：N 个成员共享一个回放缓冲区，
每次更新时各自在自助采样的子批次上做一次梯度下降。

为什么共享缓冲区
----------------
标准 Bootstrapped DQN 让每个成员用**独立**的缓冲区，多样性更好但显存/内存
成本是 N 倍。本项目在单 CPU 上训练，因此选择共享缓冲区 + 自助掩码，
用一个可控的多样性损失换取 N 倍成本下降。
**代价**：成员之间的分歧会偏小，集成分歧信号会偏保守（更容易漏报）。
这一点在报告里必须写明，不能声称「N 个完全独立的模型」。

保守的默认设置
--------------
N=5、bootstrap_prob=0.8。成员 i 的初始化种子 = base_seed + i*1000，
保证初始化确实不同（否则分歧恒为 0，整个不确定度信号失效——
这在 tests/test_uncertainty_fallback.py 里有专门的测试钉住）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_config as ec
from rl.dqn_agent import DQNConfig, silence_numpy_bridge_warning
from rl.ensemble_agent import EnsembleConfig, EnsembleDQNAgent

silence_numpy_bridge_warning()

DEFAULT_OUT_DIR = "output/rl_ensemble"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练集成 DQN（P2 不确定度回退用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=ec.CONFIG_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--jitter", action="store_true",
                        help="训练期初始条件域随机化（推荐开，与主 DQN 对齐）")
    parser.add_argument("--energy-budget", type=float, default=None)
    parser.add_argument("--adaptive-jammer", action="store_true")

    # 集成超参
    parser.add_argument("--ensemble-size", type=int, default=5,
                        help="成员数（v4.0 要求 3~5）")
    parser.add_argument("--bootstrap-prob", type=float, default=0.8,
                        help="每个成员看到每个样本的概率（自助掩码）")
    parser.add_argument("--ood-warmup-steps", type=int, default=2000,
                        help="前多少步不报 OOD 评分（观测统计量尚未稳定）")

    # 网络与优化
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--target-sync", type=int, default=200)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-steps", type=int, default=30000)
    parser.add_argument("--lr-decay-every", type=int, default=0)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--device", default="cpu")

    ec.add_observation_arguments(parser)
    parser.add_argument("--quiet", action="store_true")
    return parser


# ----------------------------------------------------------------------

def make_ensemble(args: argparse.Namespace, env: Any) -> EnsembleDQNAgent:
    config = DQNConfig(
        obs_dim=int(env.observation_space.shape[0]),
        n_actions=int(env.action_space.n),
        hidden_sizes=tuple(args.hidden),
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_sync_steps=args.target_sync,
        learning_starts=args.learning_starts,
        double_dqn=args.double_dqn,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay_steps,
        seed=args.seed,
    )
    ensemble_config = EnsembleConfig(
        ensemble_size=args.ensemble_size,
        bootstrap_prob=args.bootstrap_prob,
        ood_warmup_steps=args.ood_warmup_steps,
    )
    return EnsembleDQNAgent(config, ensemble_config, device=args.device)


def run_episode(
    env: Any, agent: EnsembleDQNAgent, seed: int, train: bool = True
) -> Dict[str, Any]:
    """跑一个 episode。口径与 train_dqn.run_episode 一致。"""
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    satisfied_steps = 0
    steps = 0
    losses: List[float] = []
    intercept_probs: List[float] = []
    tx_powers: List[float] = []
    pds: List[float] = []
    cumulative_energy = 0.0
    violated_flags: List[int] = []

    while True:
        if train:
            agent.begin_step()
            agent.update_epsilon()
        action = agent.select_action(
            obs, greedy=not train, action_mask=env.action_masks()
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            agent.store(
                obs,
                action,
                reward,
                next_obs,
                terminated,
                next_action_mask=env.action_masks(),
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
        violated_flags.append(0 if info["task_satisfied"] else 1)
        steps += 1
        obs = next_obs
        if done:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "episode_reward": total_reward / steps if steps else 0.0,
        "horizon_satisfaction_rate": (
            satisfied_steps / env.sim.scenario.num_steps if steps else 0.0
        ),
        "avg_intercept_prob": _mean(intercept_probs),
        "avg_tx_power_w": _mean(tx_powers),
        "cumulative_energy_j": cumulative_energy,
        "avg_pd": _mean(pds),
        "mean_loss": _mean(losses),
        "cost_rate": _mean([float(v) for v in violated_flags]),
    }


def evaluate_ensemble(
    eval_env: Any, agent: EnsembleDQNAgent, seeds: Sequence[int]
) -> Dict[str, Any]:
    """在若干固定场景种子上做贪心评测（不探索）。"""
    returns: List[float] = []
    satisfactions: List[float] = []
    powers: List[float] = []
    pds: List[float] = []
    q_stds: List[float] = []
    for seed in seeds:
        metrics = run_episode(eval_env, agent, seed, train=False)
        returns.append(metrics["episode_reward"])
        # 用统一口径重算，保证与 evaluate_*.py 一致
        summary = ec.summarize_episode(
            eval_env, list(eval_env.sim.results), label="集成DQN", policy_name="集成DQN"
        )
        satisfactions.append(float(summary["horizon_satisfaction_rate"]))
        powers.append(metrics["avg_tx_power_w"])
        pds.append(metrics["avg_pd"])
        # 评测期也记录不确定度水平，便于观察训练是否让网络更自信
        obs, _ = eval_env.reset(seed=seed)
        while True:
            info, _ = agent.uncertainty(obs, eval_env.action_masks())
            q_stds.append(info.q_std_max)
            action = agent.select_action(obs, greedy=True, action_mask=eval_env.action_masks())
            obs, _r, terminated, truncated, _i = eval_env.step(action)
            if terminated or truncated:
                break
    return {
        "episode_return": _mean(returns),
        "horizon_satisfaction_rate": _mean(satisfactions),
        "avg_tx_power_w": _mean(powers),
        "avg_pd": _mean(pds),
        "mean_q_std_max": _mean(q_stds),
    }


# ----------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    env = ec.make_env_from_args(args, energy_budget_j=args.energy_budget)
    overrides: Dict[str, Any] = {}
    if args.adaptive_jammer:
        overrides["adaptive_jammer"] = True
    if overrides:
        env.sim.apply_overrides(**overrides)

    agent = make_ensemble(args, env)
    eval_env = ec.make_env_from_args(args, energy_budget_j=args.energy_budget)
    if overrides:
        eval_env.sim.apply_overrides(**overrides)

    print("======== 集成 DQN 训练（P2 不确定度回退）========")
    print(f"场景          : {args.config}")
    print(f"观测          : {ec.observation_label(args)}，"
          f"{env.observation_space.shape[0]} 维")
    print(f"集成          : {agent.describe()}")
    print(f"能量预算      : {env.sim.radar.energy_budget_j} J（硬约束）")
    print(f"训练规模      : {args.episodes} episode × {env.sim.scenario.num_steps} 步")
    print(f"输出目录      : {args.out_dir}\n")

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    best_return = float("-inf")
    best_path = os.path.join(args.out_dir, "ensemble_best.pt")

    started = time.time()
    for episode in range(1, args.episodes + 1):
        env_seed = int(args.seed) + (episode - 1) * int(args.seed_stride)
        if args.jitter:
            env.sim.apply_overrides(extra=ec.jittered_scenario(env_seed, args.config))

        metrics = run_episode(env, agent, env_seed, train=True)
        metrics["episode"] = episode
        metrics["epsilon"] = agent.epsilon
        metrics["env_steps"] = agent.env_steps
        train_rows.append(metrics)

        if not args.quiet and (episode % 50 == 0 or episode == 1):
            print(
                f"[train ep {episode:4d}] reward={metrics['episode_reward']:+.4f}  "
                f"满足率={metrics['horizon_satisfaction_rate']:.3f}  "
                f"ε={agent.epsilon:.3f}  loss={metrics['mean_loss']:.4f}  "
                f"steps={metrics['steps']}"
            )

        if args.lr_decay_every > 0 and episode % args.lr_decay_every == 0:
            for optimizer in agent.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] *= args.lr_decay_gamma

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_seeds = [args.eval_seed + i for i in range(args.eval_episodes)]
            ev = evaluate_ensemble(eval_env, agent, eval_seeds)
            ev["episode"] = episode
            ev["wall_time_s"] = round(time.time() - started, 2)
            eval_rows.append(ev)
            print(
                f"  [eval ep {episode:4d}] 综合收益={ev['episode_return']:+.4f}  "
                f"满足率={ev['horizon_satisfaction_rate']:.4f}  "
                f"平均功率={ev['avg_tx_power_w']:.2f}W  "
                f"平均集成分歧={ev['mean_q_std_max']:.4f}"
            )
            if ev["episode_return"] > best_return:
                best_return = ev["episode_return"]
                agent.save(
                    best_path,
                    metadata={
                        "label": "集成DQN(best)",
                        "observation_mode": args.observation_mode,
                        "history_len": args.history_len,
                        "observation_preset": getattr(args, "observation_preset", ""),
                        "eval_return": ev["episode_return"],
                        "episode": episode,
                    },
                )

    agent.save(
        os.path.join(args.out_dir, "ensemble_final.pt"),
        metadata={
            "label": "集成DQN(final)",
            "observation_mode": args.observation_mode,
            "history_len": args.history_len,
        },
    )

    # ---------------- 输出 ----------------
    if train_rows:
        with open(os.path.join(args.out_dir, "training_log.csv"), "w",
                  encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(train_rows[0].keys()))
            writer.writeheader()
            writer.writerows(train_rows)
    if eval_rows:
        with open(os.path.join(args.out_dir, "eval_log.csv"), "w",
                  encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)

    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "obs_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(agent.obs_stats, handle, ensure_ascii=False, indent=2)

    print(f"\n环境步 / 训练步 : {agent.env_steps} / {agent.train_steps}")
    print(f"最优评测综合收益: {best_return:+.4f}（{best_path}）")
    print(f"耗时            : {time.time() - started:.1f} s")
    print(f"\n下一步：python evaluate_uncertainty.py --ensemble-model {best_path}")


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
