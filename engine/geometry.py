"""统一坐标与几何模块（v4.1 多平台几何升级的基石）。

**这是一个叶子模块**：只用标准库，不 import 工程内任何其他模块
（与 `engine/equations.py`、`engine/spaces.py` 同一约定）。
因此 `models/`、`engine/`、`strategy/`、`ai/` 都可以自由引用它而不产生循环依赖。

为什么必须把它独立出来
----------------------
升级前的工程里有 **4 份各自实现的距离计算**：
`models/target.py`、`models/interceptor.py`、`models/jammer.py` 各写了一遍
`math.hypot(self.x - x, self.y - y)`，`strategy/anti_jam.py` 又直接调了一次 `math.hypot`；
而"最近目标距离""最远目标距离"这类**聚合值**成了场景真值。
在单雷达/双目标下它还能凑合，但一旦进入多雷达、多目标、多侦察机、多干扰源，
这套做法会立刻失效：

* 同一对实体在不同模块里可能算出不同的距离（哪怕只差最后一位，也会让
  「谁被干扰更强」这类判断翻转）；
* 聚合值丢掉了"哪一对实体、在哪个时刻"的语义，无法回答
  「3 号雷达对 2 号目标此刻的视线是否被 1 号干扰源压制」。

因此本模块把几何**收口到一处**，并强制每条几何关系都携带完整语义
（谁相对于谁、在哪个时刻），见 `GeometricRelation`。

坐标系约定（**全工程统一，不得各模块自定**）
-------------------------------------------
* **ENU 直角坐标**：`x` 轴指东（East）、`y` 轴指北（North）、`z` 轴朝天（Up）；
  单位 米；速度单位 米/秒。
  历史配置里的 `x`/`y` 直接沿用该定义，新增的 `z` 默认 0，
  **因此所有旧场景的三维距离与二维距离完全相等**（见 `enu_range_2d` 的说明）。
* **球坐标**：`range`（距离，m）、`azimuth`（方位角，度，自 +y 轴/正北起顺时针为正，
  即 0°=北、90°=东）、`elevation`（俯仰角，度，水平面以上为正）。
* **姿态**：`heading`（航向，度，自正北起顺时针）、`pitch`（俯仰，度，抬头为正）、
  `roll`（横滚，度，右滚为正）。
* **机体坐标**：`forward`（机头）、`right`（右）、`up`（上），单位向量由姿态唯一确定。
  `bearing` 是**相对观察者机头**的方位（0°=正前方，+90°=右），
  `elevation_body` 是相对机体水平面的俯仰。

⚠️ 关于浮点一致性（**动这一行代码前务必先读**）
-----------------------------------------------
历史代码用 `math.hypot(dx, dy)` 算距离。`math.hypot` **不是** `sqrt(dx*dx+dy*dy)`
的等价写法——它用了防溢出的缩放算法，两者的最低位可能不同。
一旦最低位变化，Pd / 能耗 / 奖励都会跟着漂移，旧实验就不再逐位可复现。
所以本模块的距离函数**一律继续使用 `math.hypot`**，且二维路径不绕道三维
（`enu_range_2d` 直接调两参 `hypot`，不补一个 `z=0`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------

#: 浮点比较用的默认容差（米 / 度）
DEFAULT_TOLERANCE = 1e-9

#: 时间同步的默认容差（秒）
DEFAULT_TIME_TOLERANCE_S = 1e-9

#: 方位角/俯仰角的定义域
_AZIMUTH_RANGE = (-180.0, 180.0)
_ELEVATION_RANGE = (-90.0, 90.0)


class GeometryError(ValueError):
    """几何计算的前提被破坏（零长度视线、时间不同步等）。"""


# ----------------------------------------------------------------------
# 基础：三维向量
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Vec3:
    """ENU 三维向量/位置。不可变，便于安全共享与作为字典键。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # --- 运算 ---
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> "Vec3":
        if scalar == 0.0:
            raise ZeroDivisionError("Vec3 不能除以零")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    # --- 度量 ---
    def norm_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        """欧氏长度。与 `enu_range` 保持同一算法（`math.hypot`）。"""
        return math.hypot(self.x, self.y, self.z)

    def normalized(self) -> "Vec3":
        length = self.norm()
        if length <= 0.0:
            raise GeometryError("零向量无法归一化（视线方向未定义）")
        return Vec3(self.x / length, self.y / length, self.z / length)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vec3") -> float:
        """三维距离（对称）。"""
        return enu_range(self, other)

    def is_close(self, other: "Vec3", tol: float = DEFAULT_TOLERANCE) -> bool:
        return (
            abs(self.x - other.x) <= tol
            and abs(self.y - other.y) <= tol
            and abs(self.z - other.z) <= tol
        )

    # --- 转换 ---
    def to_spherical(self) -> "Spherical":
        return enu_to_spherical(self)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vec3":
        return cls(
            float(data.get("x", 0.0)),
            float(data.get("y", 0.0)),
            float(data.get("z", 0.0)),
        )


@dataclass(frozen=True)
class Spherical:
    """ENU 球坐标：距离 + 方位角 + 俯仰角。

    `azimuth_deg`：自 +y（正北）起顺时针为正，取值 (-180, 180]。
    `elevation_deg`：水平面以上为正，取值 [-90, 90]。
    """

    range_m: float
    azimuth_deg: float
    elevation_deg: float

    def to_cartesian(self) -> Vec3:
        return spherical_to_enu(self)

    def to_dict(self) -> Dict[str, float]:
        return {
            "range_m": self.range_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


# ----------------------------------------------------------------------
# 基础：姿态与位姿
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Attitude:
    """平台姿态（度）。默认全零 = 机头指北、水平、无横滚。"""

    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def body_basis(self) -> Tuple[Vec3, Vec3, Vec3]:
        """返回机体三轴在 ENU 下的单位向量 (forward, right, up)。

        定义（右滚为正）::

            forward = ( sinψ·cosθ,  cosψ·cosθ,  sinθ )
            right0  = ( cosψ,      -sinψ,       0    )
            up0     = right0 × forward
            right   =  right0·cosφ + up0·sinφ
            up      = -right0·sinφ + up0·cosφ

        其中 ψ=heading, θ=pitch, φ=roll。
        """
        psi = math.radians(self.heading_deg)
        theta = math.radians(self.pitch_deg)
        phi = math.radians(self.roll_deg)

        sin_p, cos_p = math.sin(psi), math.cos(psi)
        sin_t, cos_t = math.sin(theta), math.cos(theta)

        forward = Vec3(sin_p * cos_t, cos_p * cos_t, sin_t)
        right0 = Vec3(cos_p, -sin_p, 0.0)
        up0 = right0.cross(forward)

        sin_r, cos_r = math.sin(phi), math.cos(phi)
        right = right0 * cos_r + up0 * sin_r
        up = right0 * (-sin_r) + up0 * cos_r
        return forward, right, up

    def enu_to_body(self, vector: Vec3) -> Vec3:
        """把 ENU 向量转换到机体坐标 (forward, right, up)。"""
        forward, right, up = self.body_basis()
        return Vec3(vector.dot(forward), vector.dot(right), vector.dot(up))

    def body_to_enu(self, vector: Vec3) -> Vec3:
        """把机体坐标下的向量转换回 ENU（`enu_to_body` 的逆）。"""
        forward, right, up = self.body_basis()
        return forward * vector.x + right * vector.y + up * vector.z

    def to_dict(self) -> Dict[str, float]:
        return {
            "heading_deg": self.heading_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }


@dataclass(frozen=True)
class Pose:
    """某一时刻的位姿：位置 + 速度 + 姿态 + 时间戳。

    `time_s` 是**该位姿所对应的时刻**，不是"当前仿真时间"。
    混用两者是时间不同步 bug 的主要来源，因此 `relation()` 会校验它。
    """

    position: Vec3
    velocity: Vec3 = Vec3()
    attitude: Attitude = Attitude()
    time_s: float = 0.0
    entity_id: str = ""

    def propagate(self, dt: float) -> "Pose":
        """匀速直线外推得到新位姿（仅用于"在 t+dt 时刻求值"，不驱动仿真）。

        真正的状态推进仍由 `Simulator._advance()` 负责；
        本方法存在的意义是让"哪个时刻"这件事可以被显式构造与测试。
        """
        return Pose(
            position=self.position + self.velocity * dt,
            velocity=self.velocity,
            attitude=self.attitude,
            time_s=self.time_s + dt,
            entity_id=self.entity_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "time_s": self.time_s,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "attitude": self.attitude.to_dict(),
        }


# ----------------------------------------------------------------------
# 距离与方向
# ----------------------------------------------------------------------

def enu_range(a: Vec3, b: Vec3) -> float:
    """两点三维距离。

    **必须**用 `math.hypot`：它与 `sqrt(dx²+dy²+dz²)` 的最低几位可能不同，
    而最低位变化会顺着 Pd / 能耗 / 奖励一路传播，破坏旧实验的逐位可复现性。
    对称性：`enu_range(a, b) == enu_range(b, a)`（`hypot` 对符号不敏感，
    且平方后符号信息丢失，因此是**逐位相等**，不是近似相等）。
    """
    return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def enu_range_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """二维水平距离。

    ⚠️ 这里是**刻意不补 `z=0` 再走三维路径**的：虽然 `hypot(dx,dy,0.0)`
    在数学上等于 `hypot(dx,dy)`，但为了把"旧场景数值不变"这件事变成
    不依赖浮点实现细节的硬保证，二维路径直接调用两参 `math.hypot`，
    与升级前的 `models/*.range_to()` 写法**逐字一致**。
    """
    return math.hypot(ax - bx, ay - by)


def los_unit(observer: Vec3, target: Vec3) -> Vec3:
    """视线单位向量，方向为 **observer → target**。

    零距离时抛 `GeometryError`（方向未定义），而不是返回一个假的默认方向。
    """
    return (target - observer).normalized()


def normalize_angle_deg(angle_deg: float) -> float:
    """把角度归一化到 (-180, 180]。"""
    value = math.fmod(angle_deg, 360.0)
    if value <= -180.0:
        value += 360.0
    elif value > 180.0:
        value -= 360.0
    return value


def enu_to_spherical(vector: Vec3) -> Spherical:
    """ENU 向量 -> (距离, 方位角, 俯仰角)。

    方位角自 +y（正北）起顺时针为正；俯仰角水平面以上为正。
    """
    rng = vector.norm()
    if rng <= 0.0:
        return Spherical(0.0, 0.0, 0.0)
    azimuth = math.degrees(math.atan2(vector.x, vector.y))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, vector.z / rng))))
    return Spherical(rng, normalize_angle_deg(azimuth), elevation)


def spherical_to_enu(spherical: Spherical) -> Vec3:
    """(距离, 方位角, 俯仰角) -> ENU 向量（`enu_to_spherical` 的逆）。"""
    rng = spherical.range_m
    az = math.radians(spherical.azimuth_deg)
    el = math.radians(spherical.elevation_deg)
    horizontal = rng * math.cos(el)
    return Vec3(horizontal * math.sin(az), horizontal * math.cos(az), rng * math.sin(el))


def azimuth_elevation(observer: Vec3, target: Vec3) -> Tuple[float, float]:
    """observer → target 的绝对方位角与俯仰角（度）。"""
    spherical = enu_to_spherical(target - observer)
    return spherical.azimuth_deg, spherical.elevation_deg


def relative_bearing_deg(azimuth_deg: float, heading_deg: float) -> float:
    """把绝对方位角换算成相对观察者航向的方位角（0°=正前方，+90°=右）。"""
    return normalize_angle_deg(azimuth_deg - heading_deg)


def angular_separation_deg(a: Vec3, b: Vec3) -> float:
    """两个 ENU 向量之间的夹角（度）。零向量时抛错。"""
    na, nb = a.norm(), b.norm()
    if na <= 0.0 or nb <= 0.0:
        raise GeometryError("零向量之间没有夹角")
    cosine = max(-1.0, min(1.0, a.dot(b) / (na * nb)))
    return math.degrees(math.acos(cosine))


def radial_velocity_mps(
    observer_velocity: Vec3, target_velocity: Vec3, los: Vec3
) -> float:
    """径向速度 **d|r|/dt**，`los` 为 observer → target 的单位视线。

    符号约定（必须记住）：**正值 = 两者正在远离（距离增大）**，
    负值 = 正在接近。定义式为

        v_r = (v_target − v_observer) · u_los

    即"目标相对观察者的速度在视线方向上的投影"。

    ⚠️ **本量在交换观察者与目标时不变**（对称）：
    反向时 `(v_observer − v_target) · (−u_los)` 与原式恒等。
    这一点经常被写错成"符号相反"——有向的是相对速度**向量**，
    而径向速度是它在连线上的投影，连线翻转时投影值不变。
    """
    return (target_velocity - observer_velocity).dot(los)


# ----------------------------------------------------------------------
# 任意实体对之间的几何关系
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class GeometricRelation:
    """**有向**几何关系：`observer` 观测（或作用于）`target`，时刻为 `time_s`。

    为什么要有向：**角度量**天生不对称。`A 相对 B` 与 `B 相对 A` 的方位角相差 180°、
    俯仰角符号相反、视线向量互为反向，机体方位还各自依赖自己的航向。
    升级前工程里到处是"目标到雷达的距离"这种无向聚合值，
    一旦出现多雷达就分不清是谁对谁的量。因此这里把方向写进类型本身。

    ⚠️ 但也有**一对是天生对称**的量，必须说清楚，否则很容易写出错误断言：

    | 量 | 方向性 | 说明 |
    | --- | --- | --- |
    | `range_m` | **对称** | 距离本身无方向 |
    | `range_rate_mps` | **对称** | 它是**距离变化率** d\|R\|/dt（雷达多普勒量的定义），属于"这一对实体"，不属于某个观察者 |
    | `closing` | **对称** | 同上，是"两者在接近/远离"这一对事件的属性 |
    | `los_enu` | 反向 | `A→B` 与 `B→A` 互为相反向量 |
    | `azimuth_deg` | 反向 | 相差 180° |
    | `elevation_deg` | 反向 | 符号相反 |
    | `bearing_deg` | 各自独立 | 相对各自机头，两者之间没有固定关系 |
    | `elevation_body_deg` | 反向 | 符号相反 |

    之所以 `range_rate_mps` 对称，是因为按定义

        v_r = (v_target − v_observer) · u(observer→target)

    反向时 `(v_observer − v_target) · (−u) = (v_target − v_observer) · u`，两者恒等。
    这与"相对速度是有向的"并不矛盾——有向的是**相对速度向量**，
    而径向速度是它在一对实体连线上的投影，连线本身翻转时投影值不变。

    其余约定：
    * `los_enu` 是 **observer → target** 的单位向量；
    * `azimuth_deg` / `elevation_deg` 是**绝对**角（相对 ENU）；
    * `bearing_deg` / `elevation_body_deg` 是**相对 observer 机体**的角；
    * `range_rate_mps` > 0 表示距离在增大（远离，`closing=False`）。
    """

    observer_id: str
    target_id: str
    time_s: float

    range_m: float
    los_enu: Vec3
    azimuth_deg: float
    elevation_deg: float
    bearing_deg: float
    elevation_body_deg: float
    range_rate_mps: float
    closing: bool

    observer_position: Vec3
    target_position: Vec3

    @property
    def is_colocated(self) -> bool:
        """两者是否重合（此时方向量无定义）。"""
        return self.range_m <= 0.0

    def describe(self) -> str:
        direction = "接近" if self.closing else "远离"
        return (
            f"[{self.observer_id} → {self.target_id}] @t={self.time_s:g}s  "
            f"距离 {self.range_m:.1f} m  "
            f"方位 {self.azimuth_deg:+.2f}°(绝对) / {self.bearing_deg:+.2f}°(机体)  "
            f"俯仰 {self.elevation_deg:+.2f}°  "
            f"径向速度 {self.range_rate_mps:+.2f} m/s（{direction}）"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "target_id": self.target_id,
            "time_s": self.time_s,
            "range_m": self.range_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "bearing_deg": self.bearing_deg,
            "elevation_body_deg": self.elevation_body_deg,
            "range_rate_mps": self.range_rate_mps,
            "closing": self.closing,
            "los_enu": self.los_enu.to_dict(),
            "observer_position": self.observer_position.to_dict(),
            "target_position": self.target_position.to_dict(),
        }


def relation(
    observer_id: str,
    observer_pose: Pose,
    target_id: str,
    target_pose: Pose,
    time_s: Optional[float] = None,
    time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
) -> GeometricRelation:
    """构造 `observer → target` 在 `time_s` 时刻的完整几何关系。

    参数
    ----
    time_s : 查询时刻。给定时会**校验**两个位姿的时间戳与之同步
             （超出 `time_tolerance_s` 直接抛 `TimeSyncError`），
             这是"时间同步"从口号变成可测行为的关键；
             不给定时取 observer 位姿的时间戳，并在两者不一致时同样报错。

    异常
    ----
    TimeSyncError : 两个位姿的时间戳不同步（"不同时刻的两个平台没有几何关系"）。
    """
    if time_s is None:
        time_s = observer_pose.time_s

    if abs(observer_pose.time_s - target_pose.time_s) > time_tolerance_s:
        raise TimeSyncError(
            f"位姿时间不同步：{observer_id}@{observer_pose.time_s:g}s 与 "
            f"{target_id}@{target_pose.time_s:g}s 相差 "
            f"{abs(observer_pose.time_s - target_pose.time_s):g}s，"
            "不同时刻的两个平台之间没有良定义的几何关系"
        )
    for pose, name in ((observer_pose, observer_id), (target_pose, target_id)):
        if abs(pose.time_s - time_s) > time_tolerance_s:
            raise TimeSyncError(
                f"{name} 的位姿时间戳 {pose.time_s:g}s 与查询时刻 {time_s:g}s 不一致"
            )

    delta = target_pose.position - observer_pose.position
    distance = delta.norm()

    if distance <= 0.0:
        # 重合：方向未定义。不编造方向，明确返回零向量并且角量为 0。
        los = Vec3()
        azimuth = elevation = bearing = elevation_body = 0.0
        range_rate = 0.0
    else:
        los = delta / distance
        spherical = enu_to_spherical(delta)
        azimuth, elevation = spherical.azimuth_deg, spherical.elevation_deg
        body = observer_pose.attitude.enu_to_body(delta)
        # bearing：机头方向为 0，右为正
        bearing = normalize_angle_deg(math.degrees(math.atan2(body.y, body.x)))
        elevation_body = math.degrees(
            math.asin(max(-1.0, min(1.0, body.z / distance)))
        )
        range_rate = radial_velocity_mps(
            observer_pose.velocity, target_pose.velocity, los
        )

    return GeometricRelation(
        observer_id=observer_id,
        target_id=target_id,
        time_s=time_s,
        range_m=distance,
        los_enu=los,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        bearing_deg=bearing,
        elevation_body_deg=elevation_body,
        range_rate_mps=range_rate,
        closing=range_rate < 0.0,
        observer_position=observer_pose.position,
        target_position=target_pose.position,
    )


class TimeSyncError(GeometryError):
    """参与几何求值的位姿时间戳不同步。"""


# ----------------------------------------------------------------------
# 批量工具
# ----------------------------------------------------------------------

def pairwise_relations(
    observers: Sequence[Tuple[str, Pose]],
    targets: Sequence[Tuple[str, Pose]],
    time_s: Optional[float] = None,
    time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    skip_self: bool = True,
) -> List[GeometricRelation]:
    """枚举 observer × target 的全部有向关系。

    `skip_self=True` 时跳过 observer_id == target_id 的组合
    （同一实体对自身没有几何意义）。
    """
    out: List[GeometricRelation] = []
    for observer_id, observer_pose in observers:
        for target_id, target_pose in targets:
            if skip_self and observer_id == target_id:
                continue
            out.append(
                relation(
                    observer_id,
                    observer_pose,
                    target_id,
                    target_pose,
                    time_s=time_s,
                    time_tolerance_s=time_tolerance_s,
                )
            )
    return out


def distance_matrix(
    entities: Sequence[Tuple[str, Pose]],
) -> Dict[Tuple[str, str], float]:
    """全部实体对的对称距离矩阵（含自身对自身，值为 0）。

    返回以 (id_a, id_b) 为键、距离为值的字典；`(a,b)` 与 `(b,a)` 都写入，
    且**逐位相等**——这正是"距离对称性"单元测试要钉住的点。
    """
    out: Dict[Tuple[str, str], float] = {}
    for id_a, pose_a in entities:
        for id_b, pose_b in entities:
            out[(id_a, id_b)] = enu_range(pose_a.position, pose_b.position)
    return out
