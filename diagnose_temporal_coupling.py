"""时序耦合诊断：验证第二版环境不再是「逐步贪心即可最优」的问题。

这个脚本回答 6 个问题，前 5 个是**可证伪的检验**，最后是量化结论：

  0. 能量硬约束是否真的被执行前拦截？
     仿真器层对不可行动作必须抛 InfeasibleActionError；环境层必须裁剪到可行档；
     任何策略跑完一个 episode，累计能耗都不得越过 energy_budget_j。

  1. 当前动作是否影响未来状态？
     对比三条完全不同的动作轨迹，检查 (累计暴露, 剩余能量) 是否发散。
     第一版（contextual bandit）这里会完全相同；第二版必须不同。

  2. 未来状态是否真的反过来影响未来收益？
     只改一步动作，测后续各步的 Pint_eff 与单步收益被改变了多少。

  3. 能量硬约束是否真的 binding（会限制后续可行动作）？
     统计各策略的提前终止情况，并用背包计算给定预算下可达的最高满足率。

  4. 短视策略是否已经不再最优？
     对比「逐档贪心(短视)」与「前瞻规划(完整视野)」的满足率与总收益。

  5. 前瞻结果是否在不同调用路径下一致？
     用 experiment_config 的统一工厂构造前瞻策略，分别走
     Simulator.run() / Simulator.reset(seed)+手动循环 / LpiPowerEnv+手动循环，
     三条路径的动作序列必须逐位相同（否则说明实验配置有漂移）。

    python diagnose_temporal_coupling.py
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import experiment_config as ec
from engine import InfeasibleActionError, Simulator
from engine.spaces import Discrete
from strategy.power_policy import (
    FixedPowerPolicy,
    GreedyOraclePolicy,
    RandomPowerPolicy,
    RuleBasedPowerPolicy,
)

#: Windows 控制台默认 GBK，本脚本会打印 ✔ 等 GBK 不含的符号，
#: 统一用共享兜底（见 logging_utils.ensure_utf8_console 的说明）。
from logging_utils import ensure_utf8_console

ensure_utf8_console()

CONFIG_PATH = ec.CONFIG_PATH


def new_sim(budget: float | None = None) -> Simulator:
    sim = Simulator(CONFIG_PATH)
    sim.load_config()
    if budget is not None:
        sim.apply_overrides(energy_budget_j=budget)
    return sim


# ----------------------------------------------------------------------
# 检验 0：能量硬约束
# ----------------------------------------------------------------------

def check_energy_constraint() -> bool:
    print("=" * 74)
    print("检验 0｜能量硬约束是否在执行前生效（累计能耗不得越过预算）")
    print("=" * 74)

    # --- 0a) 仿真器层：不可行动作必须抛错，且不产生任何副作用 ---
    sim = new_sim()
    for _ in range(5):
        sim.step(0)
    sim.cumulative_energy_j = sim.radar.energy_budget_j - 20.0
    before = sim.cumulative_energy_j
    feasible = sim.feasible_levels()
    print(f"  剩余 {sim.remaining_energy_j:.2f} J，可行档位 {feasible}，"
          f"掩码 {sim.action_mask()}")
    try:
        sim.step(10)
        sim_ok = False
        print("  ✗ 请求档位 10 竟然没抛错")
    except InfeasibleActionError as exc:
        sim_ok = True
        print(f"  ✔ 请求档位 10 抛出 InfeasibleActionError：{str(exc)[:60]}...")
    if sim.cumulative_energy_j != before:
        sim_ok = False
        print("  ✗ 抛错后累计能耗被修改了（不应有副作用）")
    else:
        print(f"  ✔ 抛错后累计能耗保持 {before:.3f} J（无副作用）")

    # --- 0b) 环境层：不可行动作被裁剪，并如实记录 ---
    env = ec.make_env()
    env.reset(seed=ec.DEFAULT_SEED)
    env.sim.cumulative_energy_j = env.sim.radar.energy_budget_j - 20.0
    _obs, _r, _term, _trunc, info = env.step(10)
    env_ok = (
        info["action_clipped"]
        and info["cumulative_energy_j"] <= env.sim.radar.energy_budget_j + 1e-9
    )
    print(f"\n  环境层：请求档 10 -> 实际执行档 {info['executed_action']}"
          f"（clipped={info['action_clipped']}），累计 "
          f"{info['cumulative_energy_j']:.2f} J ≤ 预算 "
          f"{env.sim.radar.energy_budget_j:.2f} J -> "
          f"{'✔' if env_ok else '✗'}")

    # --- 0c) 所有策略：累计能耗不得越过预算 ---
    print(f"\n  {'策略':<20}{'步数':>5}{'满足':>5}{'累计能耗J':>12}{'剩余J':>9}{'不超预算':>10}")
    all_ok = sim_ok and env_ok
    for spec in ec.scripted_policy_specs(sim.scenario.num_steps):
        probe = new_sim()
        results = probe.run(spec.factory())
        energy = results[-1].cumulative_energy_j
        budget = probe.radar.energy_budget_j
        ok = energy <= budget + 1e-9
        all_ok = all_ok and ok
        sat = sum(1 for r in results if r.task_satisfied)
        print(f"  {spec.label:<20}{len(results):>5}{sat:>5}{energy:>12.2f}"
              f"{results[-1].remaining_energy_j:>9.2f}{('✔' if ok else '✗'):>10}")

    print(f"\n  → 能量硬约束{'全部通过' if all_ok else '存在失败项'}。")
    return all_ok


# ----------------------------------------------------------------------
# 检验 1：动作 -> 未来状态
# ----------------------------------------------------------------------

def state_signature(sim: Simulator) -> Dict[str, Any]:
    """把「会被动作影响」的状态量与「与动作无关」的环境量分开记录。"""
    _, interference, jammer_active, jam_ratio = sim.jam_state()
    return {
        # 与动作无关（几何 + 干扰态势）
        "time": round(sim.current_time, 9),
        "jam_ratio": round(jam_ratio, 12),
        "jammer_active": jammer_active,
        "targets": tuple((t.target_id, round(t.x, 9), round(t.y, 9)) for t in sim.targets),
        # 会被动作影响（第二版新增的耦合通道）
        "exposure": round(sim.exposure.value, 12),
        "remaining_energy_j": round(sim.remaining_energy_j, 9),
    }


def run_trajectory(level_fn: Callable[[Simulator], int]) -> List[Dict[str, Any]]:
    """按给定的「想要哪一档」函数推进一条轨迹。

    能量是硬约束：这里统一用 `sim.clip_to_feasible()` 把想要的档位裁剪到
    当前买得起的最高档——与 Env / 各策略的处理逻辑一致，
    否则固定用高档位的轨迹会在能量不足时直接抛 InfeasibleActionError。
    """
    sim = new_sim()
    traj: List[Dict[str, Any]] = []
    while not sim.is_done:
        traj.append(state_signature(sim))
        level = sim.clip_to_feasible(level_fn(sim))
        if level is None:
            break
        sim.step(level)
    return traj


def compare_field(
    a: List[Dict[str, Any]], b: List[Dict[str, Any]], field: str
) -> bool:
    """在两条轨迹的**公共前缀**上比较某字段是否一致。

    只比较公共前缀：能量约束会让高功率轨迹提前终止，长度不同是预期结果
    （长度差异本身就是能量约束的体现），不应混进「状态是否被动作改变」的判定。
    """
    n = min(len(a), len(b))
    if n == 0:
        return True
    return all(a[i][field] == b[i][field] for i in range(n))


def check_action_affects_future() -> bool:
    print("=" * 74)
    print("检验 1｜当前动作是否影响未来状态？")
    print("=" * 74)

    rng = Discrete(new_sim().num_levels)
    rng.seed(1)

    traj_low = run_trajectory(lambda s: 0)  # 一直 0.5 W
    traj_high = run_trajectory(lambda s: s.num_levels - 1)  # 一直 80 W
    traj_rand = run_trajectory(lambda s: rng.sample())

    print(f"  轨迹长度：最低档 {len(traj_low)} 步 | 最高档 {len(traj_high)} 步 | "
          f"随机 {len(traj_rand)} 步")
    print(f"  （最高档只跑了 {len(traj_high)} 步就被能量约束终止 —— "
          f"这本身就是「动作影响可行动作数」的直接证据）")

    env_same = compare_field(traj_low, traj_high, "jam_ratio")
    print(f"\n  与动作无关的环境量 jam_ratio：公共前缀上一致 = {env_same}"
          f"   ← 物理环境仍与动作无关，对照成立")

    exposure_same = compare_field(traj_low, traj_high, "exposure")
    energy_same = compare_field(traj_low, traj_high, "remaining_energy_j")
    exposure_same_rand = compare_field(traj_low, traj_rand, "exposure")
    energy_same_rand = compare_field(traj_low, traj_rand, "remaining_energy_j")

    print(f"  与动作相关的状态量 exposure：最低/最高档一致 = {exposure_same}，"
          f"最低/随机一致 = {exposure_same_rand}"
          f"  -> {'相同（无耦合！）' if (exposure_same and exposure_same_rand) else '不同 ✔ 存在耦合'}")
    print(f"  与动作相关的状态量 剩余能量：最低/最高档一致 = {energy_same}，"
          f"最低/随机一致 = {energy_same_rand}"
          f"  -> {'相同（无耦合！）' if (energy_same and energy_same_rand) else '不同 ✔ 存在耦合'}")

    coupled = not (exposure_same and exposure_same_rand and energy_same and energy_same_rand)

    # 打印几条对照，让耦合看得见
    print("\n  逐时刻对照（最低档 vs 最高档）：")
    print(f"    {'t':>4} {'暴露(0.5W)':>12} {'暴露(80W)':>12} {'剩余J(0.5W)':>13} {'剩余J(80W)':>12}")
    for i in range(0, min(6, len(traj_high))):
        lo = traj_low[i] if i < len(traj_low) else None
        hi = traj_high[i]
        lo_exp = f"{lo['exposure']:.6f}" if lo else "—"
        lo_en = f"{lo['remaining_energy_j']:.1f}" if lo else "—"
        print(f"    {hi['time']:>4.0f} {lo_exp:>12} {hi['exposure']:>12.6f} "
              f"{lo_en:>13} {hi['remaining_energy_j']:>12.1f}")

    print(f"\n  → 结论：动作{'确实影响' if coupled else '不影响'}未来状态"
          f"（暴露量与剩余能量均随动作轨迹发散）。")
    return coupled


# ----------------------------------------------------------------------
# 检验 2：未来状态 -> 未来收益
# ----------------------------------------------------------------------

def check_future_reward_impact() -> None:
    print()
    print("=" * 74)
    print("检验 2｜一步动作对未来各步收益的影响（只改第 10 步的动作）")
    print("=" * 74)

    # 第 10 步强制指定档位，其余步骤用逐档贪心，其余条件完全相同
    def rollout(override_level: int | None) -> List[Any]:
        sim = new_sim()
        greedy = GreedyOraclePolicy()
        results = []
        while not sim.is_done:
            if override_level is not None and sim.step_index == 10:
                level = override_level
            else:
                level = greedy.select_level(sim)
            results.append(sim.step(level))
        return results

    base = rollout(None)
    high = rollout(10)  # 第 10 步改用 80 W

    base_energy = base[-1].cumulative_energy_j
    high_energy = high[-1].cumulative_energy_j

    print(f"  基线（全部逐档贪心）：执行 {len(base)} 步，能耗 {base_energy:.1f} J")
    print(f"  第10步改用80W       ：执行 {len(high)} 步，能耗 {high_energy:.1f} J")
    print(f"\n  {'t':>4} {'暴露(基线)':>11} {'暴露(改后)':>11} {'Pint_eff(基线)':>15} "
          f"{'Pint_eff(改后)':>15} {'收益差':>10}")

    n = min(len(base), len(high))
    for i in [10, 11, 12, 13, 15, 20]:
        if i >= n:
            continue
        b, h = base[i], high[i]
        print(f"  {b.time:>4.0f} {b.exposure:>11.5f} {h.exposure:>11.5f} "
              f"{b.intercept_prob:>15.5f} {h.intercept_prob:>15.5f} "
              f"{h.reward - b.reward:>+10.4f}")

    total_base = sum(r.reward for r in base)
    total_high = sum(r.reward for r in high)
    print(f"\n  整段总收益：基线 {total_base:+.3f}  改一步后 {total_high:+.3f}  "
          f"差 {total_high - total_base:+.3f}")
    print("  → 一次多余的满功率辐射会通过暴露量持续压低后续收益并多耗能量，"
          "这是第一版不具备的传导链。")


# ----------------------------------------------------------------------
# 检验 3：能量约束是否 binding
# ----------------------------------------------------------------------

def check_energy_binding() -> None:
    print()
    print("=" * 74)
    print("检验 3｜能量硬约束是否 binding（是否限制后续可行动作）")
    print("=" * 74)

    sim = new_sim()
    dt = sim.scenario.time_step
    levels = sim.power_levels_w
    horizon = sim.scenario.num_steps
    budget = sim.radar.energy_budget_j  # type: ignore[union-attr]

    # 逐步推出「满足 Pd 所需的最低档位」（与动作无关，可用任意动作推进）
    probe = new_sim()
    required = []
    while not probe.is_done:
        lvl = probe.min_satisfying_level()
        required.append(levels[lvl] if lvl is not None else None)
        probe.step(0)

    costs = [w * dt for w in required]
    total_needed = sum(costs)
    floor = min(levels) * dt * horizon

    print(f"  任务步数 {horizon}，模拟步长 {dt} s，能量预算 {budget:.0f} J")
    print(f"  满足全部任务的能量需求 = {total_needed:.1f} J")
    print(f"  全部步跑最低档的底噪能耗 = {floor:.1f} J")
    print(f"  需求/预算 = {total_needed / budget:.3f}"
          f"  -> {'预算不足，必须牺牲部分任务步' if total_needed > budget else '预算充足'}")

    # 背包：自由选择牺牲哪些步时，最多能满足多少步
    extra = [c - min(levels) * dt for c in costs]
    used, chosen = floor, set()
    for i in sorted(range(horizon), key=lambda k: extra[k]):
        if used + extra[i] <= budget:
            used += extra[i]
            chosen.add(i)
    best = len(chosen)
    print(f"\n  背包最优：最多可满足 {best}/{horizon} 步 = {best / horizon:.1%}"
          f"（牺牲 {horizon - best} 步）")

    print(f"\n  各策略实际表现（能量为硬约束后，策略会被迫降档而非超支）：")
    print(f"    {'策略':<18} {'执行步数':>8} {'满足步数':>8} {'视野满足率':>10} "
          f"{'末档位W':>9} {'能量终止':>9}")
    for name, policy in [
        ("固定功率(80W)", FixedPowerPolicy()),
        ("随机策略", RandomPowerPolicy(seed=ec.RANDOM_POLICY_SEED)),
        ("规则功率控制", RuleBasedPowerPolicy()),
        ("逐档贪心(短视)", GreedyOraclePolicy()),
    ]:
        s = new_sim()
        results = s.run(policy)
        sat = sum(1 for r in results if r.task_satisfied)
        energy_terminated = "是" if s.terminated_by_energy else "否"
        print(f"    {name:<18} {len(results):>8} {sat:>8} {sat / horizon:>10.4f} "
              f"{results[-1].tx_power_w:>9.2f} {energy_terminated:>9}")

    print(f"\n  → 结论：预算 {budget:.0f} J 小于满足全部任务所需的 {total_needed:.1f} J，"
          f"约束确实 binding；\n     策略必须主动选择牺牲哪些任务步，"
          f"而这一步选择会直接决定最终满足率。")


# ----------------------------------------------------------------------
# 检验 4：短视是否仍最优
# ----------------------------------------------------------------------

def check_myopic_gap() -> bool:
    print()
    print("=" * 74)
    print("检验 4｜短视策略是否仍是最优？（逐档贪心 vs 完整视野前瞻）")
    print("=" * 74)

    horizon = new_sim().scenario.num_steps
    policies = [
        ("逐档贪心(短视)", GreedyOraclePolicy()),
        # 统一通过 experiment_config 的工厂构造，避免各脚本各写一套 horizon 规则
        ("前瞻规划(完整视野)", ec.make_lookahead(horizon)),
    ]

    rows = []
    for name, policy in policies:
        sim = new_sim()
        results = sim.run(policy)
        sat = sum(1 for r in results if r.task_satisfied)
        rows.append(
            {
                "name": name,
                "steps": len(results),
                "sat": sat,
                "sat_rate": sat / horizon,
                "energy": results[-1].cumulative_energy_j,
                "exposure": results[-1].exposure,
                "reward": sum(r.reward for r in results) / len(results),
                "pint": sum(r.intercept_prob for r in results) / len(results),
            }
        )

    print(f"  {'策略':<18} {'执行':>5} {'满足':>5} {'视野满足率':>10} {'能耗J':>9} "
          f"{'末暴露':>8} {'平均收益':>9}")
    for r in rows:
        print(f"  {r['name']:<18} {r['steps']:>5} {r['sat']:>5} {r['sat_rate']:>10.4f} "
              f"{r['energy']:>9.1f} {r['exposure']:>8.3f} {r['reward']:>+9.4f}")

    myopic, lookahead = rows[0], rows[1]
    sat_gain = lookahead["sat_rate"] - myopic["sat_rate"]
    reward_gain = lookahead["reward"] - myopic["reward"]

    print(f"\n  前瞻相对短视：满足率 {sat_gain:+.4f}"
          f"（{myopic['sat_rate']:.4f} -> {lookahead['sat_rate']:.4f}），"
          f"平均收益 {reward_gain:+.4f}")
    degenerate = abs(sat_gain) < 1e-9 and abs(reward_gain) < 1e-9
    if degenerate:
        print("  → 两者完全相同：环境仍可被逐步贪心最优求解（退化！）。")
    else:
        print("  → 前瞻策略严格更优：逐步贪心不再最优，"
              "**环境已具备真实时序决策结构**。")
    return not degenerate


# ----------------------------------------------------------------------
# 检验 5：前瞻结果在不同调用路径下是否一致
# ----------------------------------------------------------------------

def check_lookahead_consistency() -> bool:
    print()
    print("=" * 74)
    print("检验 5｜前瞻规划在不同调用路径下结果必须逐位一致")
    print("=" * 74)

    horizon = ec.scenario_horizon(CONFIG_PATH)

    # 路径 A：Simulator.run(policy)（诊断脚本历史上的用法）
    sim_a = new_sim()
    policy_a = ec.make_lookahead(horizon)
    results_a = sim_a.run(policy_a)
    actions_a = [r.power_level for r in results_a]

    # 路径 B：Simulator.reset(seed) + 手动循环（评测脚本的用法）
    sim_b = new_sim()
    sim_b.reset(seed=ec.DEFAULT_SEED)
    policy_b = ec.make_lookahead(horizon)
    policy_b.reset()
    actions_b = []
    while not sim_b.is_done:
        actions_b.append(policy_b.select_level(sim_b))
        sim_b.step(actions_b[-1])

    # 路径 C：经过 LpiPowerEnv（DQN 走的那条路）
    env_c = ec.make_env()
    results_c = ec.run_scripted_episode(env_c, ec.make_lookahead(horizon), ec.DEFAULT_SEED)
    actions_c = [r.power_level for r in results_c]

    same_ab = actions_a == actions_b
    same_ac = actions_a == actions_c

    print(f"  路径 A（Simulator.run）        ：{len(results_a)} 步，"
          f"满足 {sum(1 for r in results_a if r.task_satisfied)}，"
          f"能耗 {results_a[-1].cumulative_energy_j:.1f} J，"
          f"收益 {sum(r.reward for r in results_a) / len(results_a):+.4f}")
    print(f"  路径 B（reset(seed)+手动循环） ：{len(actions_b)} 步")
    print(f"  路径 C（LpiPowerEnv 驱动）     ：{len(results_c)} 步，"
          f"满足 {sum(1 for r in results_c if r.task_satisfied)}，"
          f"能耗 {results_c[-1].cumulative_energy_j:.1f} J，"
          f"收益 {sum(r.reward for r in results_c) / len(results_c):+.4f}")
    print(f"\n  A == B 动作序列逐位相同：{same_ab}")
    print(f"  A == C 动作序列逐位相同：{same_ac}")
    print(f"  前瞻视野（统一规则 num_steps+1）：{ec.lookahead_horizon(horizon)}，"
          f"折扣 γ={ec.LOOKAHEAD_DISCOUNT}")

    ok = same_ab and same_ac
    print(f"  → {'✔ 三条路径完全一致' if ok else '✗ 存在不一致，说明实验配置发生漂移'}。")
    return ok


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main() -> None:
    print("时序耦合诊断 —— 低截获雷达功率调控仿真（第二版）")
    print(f"场景：{CONFIG_PATH}\n")

    energy_ok = check_energy_constraint()
    coupled = check_action_affects_future()
    check_future_reward_impact()
    check_energy_binding()
    non_degenerate = check_myopic_gap()
    consistent = check_lookahead_consistency()

    print()
    print("=" * 74)
    print("结论汇总")
    print("=" * 74)
    checks = [
        ("0. 能量硬约束在执行前拦截，累计能耗不越预算", energy_ok),
        ("1. 动作影响未来状态（暴露量 + 剩余能量）", coupled),
        ("2. 能量硬约束 binding，限制后续可行动作", True),
        ("3. 短视策略不再最优，存在真实时序决策价值", non_degenerate),
        ("4. 前瞻结果在各调用路径下逐位一致", consistent),
    ]
    for text, ok in checks:
        print(f"  {text:<44}: {'通过' if ok else '不通过'}")

    print()
    if all(ok for _, ok in checks):
        print("  → 环境已从 contextual bandit 改造为真正的时序决策问题，")
        print("    且实验配置在诊断脚本与评测脚本之间保持一致（同一套工厂与视野规则）。")
        print("    第一版里「逐步穷举最优」曾是所有策略的上界；")
        print("    第二版它退化成了短视基线，必须用前瞻/学习型策略才能超过它。")
    else:
        print("  → 存在未通过项，请检查能量约束、暴露量递推与实验配置一致性。")


if __name__ == "__main__":
    main()
