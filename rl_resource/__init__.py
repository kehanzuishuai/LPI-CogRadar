"""大阶段二：集中式学习资源调度基线（``rl_resource``）。

定位
----
**这一层的目标不是"超过规则或优化参考"**，而是先建立一条**可信的、可复现的**
集中式学习基线：

1. 一个中央智能体读取**算法可见信息**（`CentralObservation` + 任务队列），
   输出标准的 `ExecutionPlan`；
2. 计划仍由既有 `UnifiedExecutor`（唯一记账点）+ `RuntimeExecutor`（唯一副作点）
   校验与执行——**智能体不得直接改 Sensor / Fusion / CommBus**；
3. 训练必须跑在**真闭环** `runtime_mode="plan_controlled_feedback"` 上
   （否则调度根本影响不了感知链，学到的策略没有意义，见 README §11K）；
4. 严格用 train 分区训练、validation 分区选 checkpoint，**test 分区保持封存**；
5. 只实现现有的 `sample` / `process` / `share` / `idle` 四类动作，
   **不新增任何物理动作**。

与既有模块的边界
----------------
* `resource_management/`：环境与契约（零依赖，**不 import torch**）；
* `rl_resource/`：学习侧（**需要 torch**）。依赖方向单向：
  `rl_resource → resource_management`，反向不成立。
* 不修改 `resource-contract-v1`，不改传感器 / 通信 / 融合 / 任务完成口径。

模块
----
| 文件 | 内容 |
| --- | --- |
| `scenarios.py` | 冻结场景目录 → 真闭环机制的映射（首次实现，含限制说明） |
| `obs.py` | 定长观测编码器（只吃调度器本来就可见的字段） |
| `actions.py` | 结构化动作空间（逐节点分类）+ 合法动作 mask + 动作→`ExecutionPlan` |
| `env.py` | `CentralizedResourceSchedulingEnv`（真闭环，Gymnasium 风格） |
| `policy.py` | Actor-Critic（逐节点分类头 + mask，torch） |
| `ppo.py` | PPO + GAE |
| `train.py` | smoke/正式训练的 CLI（train 分区训练 + validation 选 checkpoint） |
| `evaluate.py` | 评测（默认只允许 validation；test 需显式解封） |
"""

from __future__ import annotations

__all__ = [
    "ACTION_IDLE", "ACTION_KINDS", "ACTION_NAMES", "ACTION_PROCESS",
    "ACTION_SAMPLE", "ACTION_SHARE",
]


def __getattr__(name: str):
    """惰性导出动作常量，避免仅仅 `import rl_resource` 就拖入 torch。"""
    if name in ("ACTION_IDLE", "ACTION_KINDS", "ACTION_NAMES",
                "ACTION_PROCESS", "ACTION_SAMPLE", "ACTION_SHARE"):
        from rl_resource import actions
        return getattr(actions, name)
    raise AttributeError(name)
