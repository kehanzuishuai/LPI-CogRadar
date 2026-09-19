"""场景实体注册表：多平台场景的统一索引与几何查询入口（v4.1）。

它解决什么问题
--------------
升级前，场景真值是 `Simulator` 上几个平行的列表加一堆**聚合值**：
`targets` / `interceptors` / `jammers`，外加"最近目标距离""最远目标距离"
"最小 RCS""最近侦察机距离"。这套表示法有两个硬伤：

1. **没有身份**。聚合值回答不了"是哪个目标最近""哪个干扰源在压制哪部雷达"；
   多雷达场景下连"这个距离属于哪对实体"都无法表达。
2. **没有方向**。距离对称，但方位、俯仰、径向速度、视线方向都是有向的。
   `A 相对 B` 与 `B 相对 A` 是不同的量，必须显式区分。

`Scene` 把场景从"若干列表 + 聚合值"改成**一个带索引的实体集合 + 有向关系图**：

* 所有实体（雷达/目标/侦察机/干扰源）统一进一个注册表，
  `entity_id` **跨类型唯一**，重复直接报错（而不是让后者静默覆盖前者）；
* 任何几何量都必须经由 `relation(observer_id, target_id)` 取得，
  返回的 `GeometricRelation` 自带"谁相对谁、在哪个时刻"的语义；
* 聚合值（最近/最远/最小 RCS 等）降级为**派生视图**，
  由显式关系算出来，而不是场景真值本身。

时间语义
--------
每个实体自带 `timestamp_s`。Scene 的参考时刻取**主雷达**的时间戳。
`relation()` 会校验双方时间戳与查询时刻同步，不同步直接抛 `TimeSyncError`——
"两个不同时刻的平台之间没有良定义的几何关系"，这是硬约束而不是注释。

向后兼容
--------
`Simulator` 仍然保留 `radar` / `targets` / `interceptors` / `jammers` 四个属性，
旧代码与旧脚本一律照旧；`Scene` 是新增加的**同一个底层实体的统一视图**
（不是副本——改 `scene.by_id('TGT1').x` 会同步反映到 `sim.targets`）。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from engine.geometry import (
    DEFAULT_TIME_TOLERANCE_S,
    GeometricRelation,
    Pose,
    relation,
)
from models.entity import (
    KIND_INTERCEPTOR,
    KIND_JAMMER,
    KIND_RADAR,
    KIND_TARGET,
    SceneEntity,
)

#: 实体快照 CSV 的列顺序（固定下来，便于脚本与论文表格对齐）
ENTITY_SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "time_s", "entity_id", "entity_kind", "entity_kind_cn", "platform_id",
    "x", "y", "z", "vx", "vy", "vz", "speed_mps",
    "heading_deg", "pitch_deg", "roll_deg", "is_active",
)

#: 关系快照 CSV 的列顺序
RELATION_SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "time_s", "observer_id", "observer_kind", "target_id", "target_kind",
    "range_m", "azimuth_deg", "elevation_deg", "bearing_deg",
    "elevation_body_deg", "range_rate_mps", "closing",
    "los_x", "los_y", "los_z",
)


class SceneError(ValueError):
    """场景构造或索引有误（ID 重复、未知实体、时间不同步等）。"""


class Scene:
    """多平台场景的实体注册表与几何查询入口。"""

    def __init__(
        self,
        radars: Sequence[Any],
        targets: Sequence[Any],
        interceptors: Sequence[Any],
        jammers: Sequence[Any],
        name: str = "",
    ) -> None:
        if not radars:
            raise SceneError("场景至少需要一部雷达")
        self.name = name
        self.radars: List[Any] = list(radars)
        self.targets: List[Any] = list(targets)
        self.interceptors: List[Any] = list(interceptors)
        self.jammers: List[Any] = list(jammers)

        # --- 建索引：entity_id -> 实体，跨类型唯一 ---
        self._index: Dict[str, SceneEntity] = {}
        for entity in self.entities:
            entity_id = entity.entity_id
            if entity_id in self._index:
                other = self._index[entity_id]
                raise SceneError(
                    f"实体 ID 重复：{entity_id!r} 同时属于 "
                    f"{other.entity_kind_cn} 和 {entity.entity_kind_cn}。"
                    "entity_id 必须在整个场景内唯一，否则按 ID 取几何量会取错对象。"
                )
            if not entity_id:
                raise SceneError(f"{entity.entity_kind_cn} 存在空 entity_id")
            self._index[entity_id] = entity

    # ------------------------------------------------------------------
    # 基本索引
    # ------------------------------------------------------------------

    @property
    def entities(self) -> List[SceneEntity]:
        """全部实体（雷达 → 目标 → 侦察机 → 干扰源 的稳定顺序）。"""
        return [*self.radars, *self.targets, *self.interceptors, *self.jammers]

    @property
    def primary_radar(self) -> Any:
        """主雷达（列表首部）。

        多雷达**协同决策**不在本阶段范围内：物理评估与功率控制仍只针对主雷达。
        但几何查询对**所有**雷达都可用。
        """
        return self.radars[0]

    @property
    def entity_ids(self) -> List[str]:
        return [e.entity_id for e in self.entities]

    def count_by_kind(self) -> Dict[str, int]:
        counts = {KIND_RADAR: 0, KIND_TARGET: 0, KIND_INTERCEPTOR: 0, KIND_JAMMER: 0}
        for entity in self.entities:
            counts[entity.entity_kind] = counts.get(entity.entity_kind, 0) + 1
        return counts

    def of_kind(self, kind: str, active_only: bool = False) -> List[SceneEntity]:
        """按类型取实体。`kind` 用 `models.entity.KIND_*` 常量。"""
        out = [e for e in self.entities if e.entity_kind == kind]
        if active_only:
            out = [e for e in out if e.is_active]
        return out

    def by_id(self, entity_id: str) -> SceneEntity:
        """按 ID 取实体；不存在时抛错（不返回 None，避免错误悄悄传播）。"""
        try:
            return self._index[entity_id]
        except KeyError:
            raise SceneError(
                f"场景中没有实体 {entity_id!r}；现有实体：{sorted(self._index)}"
            ) from None

    def maybe_by_id(self, entity_id: str) -> Optional[SceneEntity]:
        return self._index.get(entity_id)

    # ------------------------------------------------------------------
    # 时间
    # ------------------------------------------------------------------

    @property
    def reference_time(self) -> float:
        """场景参考时刻 = 主雷达的时间戳。"""
        return float(self.primary_radar.timestamp_s)

    def timestamps(self) -> Dict[str, float]:
        return {e.entity_id: float(e.timestamp_s) for e in self.entities}

    def assert_time_synchronized(
        self, tolerance_s: float = DEFAULT_TIME_TOLERANCE_S
    ) -> None:
        """校验全部实体时间戳一致；不一致直接抛 `TimeSyncError`。

        注意：允许"某实体未参与推进"这种设计（例如静止的雷达也可以推进），
        但**不允许**时间戳悄悄漂移——那会让所有几何关系失去意义。
        """
        stamps = self.timestamps()
        if not stamps:
            return
        lo = min(stamps.values())
        hi = max(stamps.values())
        if hi - lo > tolerance_s:
            worst = sorted(stamps.items(), key=lambda kv: kv[1])
            raise SceneError(
                f"场景内实体时间戳不同步：最早 {worst[0][0]}@{worst[0][1]:g}s，"
                f"最晚 {worst[-1][0]}@{worst[-1][1]:g}s，跨度 {hi - lo:g}s。"
                "多平台几何要求所有实体在同一时间基准上求值。"
            )

    def advance_all(self, dt: float) -> None:
        """把全部实体推进 dt（保持时间戳同步）。"""
        for entity in self.entities:
            entity.advance(dt)

    def set_reference_time(self, time_s: float) -> None:
        """把所有实体的时间戳对齐到给定时刻（**不改变位置**）。

        仅用于"把上一段仿真的末端当作新片段的起点"这类场景切换，
        或测试中显式构造同步位姿。位置不会因此移动。
        """
        for entity in self.entities:
            entity.timestamp_s = float(time_s)

    # ------------------------------------------------------------------
    # 几何查询（唯一入口）
    # ------------------------------------------------------------------

    def pose_of(self, entity_id: str) -> Pose:
        return self.by_id(entity_id).pose

    def relation(
        self,
        observer_id: str,
        target_id: str,
        time_s: Optional[float] = None,
        time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    ) -> GeometricRelation:
        """取 `observer_id → target_id` 的有向几何关系。

        `time_s` 缺省取主雷达时间戳；双方时间戳必须与之同步，否则抛
        `TimeSyncError`（"谁相对于谁、在哪一时刻"必须成立才能求几何量）。
        """
        observer = self.by_id(observer_id)
        target = self.by_id(target_id)
        if time_s is None:
            time_s = self.reference_time
        return relation(
            observer_id, observer.pose, target_id, target.pose,
            time_s=time_s, time_tolerance_s=time_tolerance_s,
        )

    def relations_from(
        self, observer_id: str, target_kinds: Optional[Iterable[str]] = None,
        time_s: Optional[float] = None,
    ) -> List[GeometricRelation]:
        """某个观察者到一批实体的关系（默认到全部其它实体）。"""
        kinds = set(target_kinds) if target_kinds is not None else None
        out: List[GeometricRelation] = []
        for target in self.entities:
            if target.entity_id == observer_id:
                continue
            if kinds is not None and target.entity_kind not in kinds:
                continue
            out.append(self.relation(observer_id, target.entity_id, time_s=time_s))
        return out

    def relations_between(
        self, observer_kind: str, target_kind: str, time_s: Optional[float] = None,
        active_only: bool = False,
    ) -> List[GeometricRelation]:
        """两个**类型**之间的全部关系（如所有雷达到所有目标）。

        这是多平台场景的主查询：一次拿到 `N_radar × N_target` 条显式关系，
        而不是一个"最近目标距离"。
        """
        observers = self.of_kind(observer_kind, active_only=active_only)
        targets = self.of_kind(target_kind, active_only=active_only)
        out: List[GeometricRelation] = []
        for observer in observers:
            for target in targets:
                if observer.entity_id == target.entity_id:
                    continue
                out.append(
                    self.relation(observer.entity_id, target.entity_id, time_s=time_s)
                )
        return out

    def all_relations(
        self, time_s: Optional[float] = None, active_only: bool = False
    ) -> List[GeometricRelation]:
        """全部有向关系（观察者 × 目标，去掉自反对）。"""
        entities = [e for e in self.entities if (e.is_active or not active_only)]
        out: List[GeometricRelation] = []
        for observer in entities:
            for target in entities:
                if observer.entity_id == target.entity_id:
                    continue
                out.append(
                    self.relation(observer.entity_id, target.entity_id, time_s=time_s)
                )
        return out

    # ------------------------------------------------------------------
    # 派生视图（聚合值降级为"由显式关系算出来"）
    # ------------------------------------------------------------------

    def nearest(self, observer_id: str, kind: str) -> Optional[GeometricRelation]:
        """离观察者最近的某类实体（派生量，不是场景真值）。"""
        relations = {
            r.target_id: r
            for r in self.relations_from(observer_id, target_kinds=(kind,))
        }
        if not relations:
            return None
        return min(relations.values(), key=lambda r: r.range_m)

    def farthest(self, observer_id: str, kind: str) -> Optional[GeometricRelation]:
        relations = {
            r.target_id: r
            for r in self.relations_from(observer_id, target_kinds=(kind,))
        }
        if not relations:
            return None
        return max(relations.values(), key=lambda r: r.range_m)

    def min_rcs_target(self, observer_id: Optional[str] = None) -> Optional[Any]:
        """最小 RCS 的（活跃）目标。

        保留这个聚合量是因为环境的 12 维观测在用；但它现在是**派生视图**，
        真正的场景真值是 `relations_between(KIND_RADAR, KIND_TARGET)` 的逐条关系。
        """
        targets = [t for t in self.targets if t.is_active]
        if not targets:
            return None
        return min(targets, key=lambda t: t.rcs_m2)

    def distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """全部实体对的对称三维距离矩阵。"""
        out: Dict[Tuple[str, str], float] = {}
        entities = self.entities
        for a in entities:
            for b in entities:
                out[(a.entity_id, b.entity_id)] = a.range_to_entity(b)
        return out

    # ------------------------------------------------------------------
    # 快照与导出
    # ------------------------------------------------------------------

    def entity_records(self) -> List[Dict[str, Any]]:
        """逐实体状态记录（含 time_s 列，便于多步拼接成时序 CSV）。"""
        return [e.to_state_dict() for e in self.entities]

    def relation_records(
        self,
        time_s: Optional[float] = None,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """逐关系记录。`pairs` 给定时只导出指定有序对。"""
        if pairs is None:
            relations = self.all_relations(time_s=time_s)
        else:
            relations = [self.relation(a, b, time_s=time_s) for a, b in pairs]

        kind_of = {e.entity_id: e.entity_kind for e in self.entities}
        records: List[Dict[str, Any]] = []
        for item in relations:
            record = item.to_dict()
            record["observer_kind"] = kind_of.get(item.observer_id, "")
            record["target_kind"] = kind_of.get(item.target_id, "")
            record["los_x"] = item.los_enu.x
            record["los_y"] = item.los_enu.y
            record["los_z"] = item.los_enu.z
            records.append(record)
        return records

    def to_dict(self, time_s: Optional[float] = None) -> Dict[str, Any]:
        """完整场景快照（实体 + 关系 + 索引摘要）。"""
        return {
            "scene_name": self.name,
            "time_s": self.reference_time if time_s is None else time_s,
            "coordinate_system": {
                "frame": "ENU",
                "axes": {"x": "East", "y": "North", "z": "Up"},
                "units": {"position": "m", "velocity": "m/s", "angle": "deg"},
                "azimuth_definition": "自 +y（正北）起顺时针为正",
                "elevation_definition": "水平面以上为正",
                "attitude_definition": "heading 自正北顺时针；pitch 抬头为正；roll 右滚为正",
            },
            "counts": self.count_by_kind(),
            "index": {"entity_ids": self.entity_ids},
            "timestamps": self.timestamps(),
            "entities": self.entity_records(),
            "relations": self.relation_records(time_s=time_s),
        }

    def write_json(self, path: str, time_s: Optional[float] = None) -> str:
        payload = self.to_dict(time_s=time_s)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def write_entities_csv(self, path: str, append: bool = False) -> str:
        return _write_csv(
            path, ENTITY_SNAPSHOT_FIELDS, self.entity_records(), append=append
        )

    def write_relations_csv(
        self,
        path: str,
        time_s: Optional[float] = None,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        append: bool = False,
    ) -> str:
        return _write_csv(
            path, RELATION_SNAPSHOT_FIELDS,
            self.relation_records(time_s=time_s, pairs=pairs), append=append,
        )

    # ------------------------------------------------------------------

    def describe(self) -> str:
        counts = self.count_by_kind()
        return (
            f"场景 {self.name or '(未命名)'} @t={self.reference_time:g}s："
            f"{counts[KIND_RADAR]} 雷达 / {counts[KIND_TARGET]} 目标 / "
            f"{counts[KIND_INTERCEPTOR]} 侦察机 / {counts[KIND_JAMMER]} 干扰源；"
            f"实体总数 {len(self.entities)}，有向关系 {len(self.all_relations())} 条"
        )


def _write_csv(path: str, fields: Sequence[str], rows: Sequence[Dict[str, Any]],
               append: bool = False) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path
