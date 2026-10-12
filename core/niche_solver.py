#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
niche_solver.py —— 《生境 NICHE》决策求解器的参考实现（Python，仅依赖标准库）

规格来源（均已联网核实，见 D1 手册）：
  1. Infinite Axis Utility System (IAUS) —— Dave Mark, GDC 2013「Architecture Tricks」
     consideration = 传感器输入 → 归一化到 [0,1] → response curve；
     同一 action 的所有 consideration 相乘，再乘该 action 自身 weight。
     关键性质：任一 consideration 为 0 → 该 action 得分为 0（可被"否决"）。
  2. Commitment / Momentum bonus —— 上一动作加成的实践值约为 +25%（Curvature 实现）
  3. 选择关卡：multinomial logit / softmax ⇔ Luce 选择公理 ⇔ Gumbel 随机效用最大化
     （McFadden 1974，2000 年诺贝尔经济学奖）；温度 τ 控制 exploitation/exploration。

本实现相对 IAUS 的两处"我们的扩展"（要在 README 里明写，不要装成原教旨）：
  A. 聚合方式除了 product（原版），还提供 geometric（开 K 次方，做维度补偿）与 additive（对照）
  B. 选择不只是 argmax，而是 softmax(T)，以便把"这只生物有多随机"变成可预注册的旋钮
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# 0. 基础工具
# ----------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def bookend(raw: float, lo: float, hi: float, invert: bool = False) -> float:
    """把原始传感器值按 bookend 夹到 [0,1]。bookend 就是"策划能读懂的量纲"。

    invert=True 用于"距离越远越不成立"这类反向语义（危险/亮度等）。
    """
    if hi == lo:
        x = 0.0
    else:
        x = (raw - lo) / (hi - lo)
    return clamp01(1.0 - x if invert else x)


def curve_linear(x: float, m: float = 1.0, k: float = 1.0, b: float = 0.0, c: float = 0.0) -> float:
    """GDC 讲义原文：y = m * (x - c)^k + b    （m 斜率 / k 指数 / b 纵截距 / c 横截距）"""
    return clamp01(m * ((x - c) ** k) + b)


def curve_logistic(x: float, m: float = 8.0, k: float = 1.0, b: float = 0.0, c: float = 0.5) -> float:
    """标准 logistic：拐点 c、斜率 m、纵向平移 b。（k 在此族中不使用，仅为统一签名）

    注：GDC 讲义流传的转录式 "k * 1/(1+(1000 e m^(-1*x+c)))+b" 疑似 OCR/排版错误，
    这里用等价语义的标准写法（55% 处的斜率由 m 控制），参数含义与讲义一致。
    """
    return clamp01(1.0 / (1.0 + math.exp(-m * (x - c))) + b)


CURVES = {"linear": curve_linear, "logistic": curve_logistic}


# ----------------------------------------------------------------------------
# 1. 世界状态与生物状态（全部可 float 序列化 —— 这是黄金向量可复现的前提）
# ----------------------------------------------------------------------------

@dataclass
class World:
    """环境侧：能改的都在这一层。"""
    water_dist: float = 6.0     # 到水源的距离 (m)
    heat_dist: float = 12.0     # 到热源的距离 (m)
    threat_dist: float = 40.0   # 到威胁物的距离 (m)
    cover_density: float = 0.3  # 遮蔽度 0–1（玩家可以改变）
    frost: float = 0.0          # 寒潮强度 0–1（玩家可以制造）


@dataclass
class Drive:
    """生物内部状态：只能通过环境间接影响。"""
    hydration: float = 0.6      # 1=完全满足
    warmth: float = 0.6
    energy: float = 0.8


# ----------------------------------------------------------------------------
# 2. Consideration / Action 的数据结构
# ----------------------------------------------------------------------------

Sensor = Callable[[World, Drive], float]


@dataclass
class Consideration:
    name: str
    sensor: Sensor
    lo: float
    hi: float
    invert: bool = False
    curve: str = "linear"
    m: float = 1.0
    k: float = 1.0
    b: float = 0.0
    c: float = 0.0

    def evaluate(self, w: World, d: Drive) -> Tuple[float, float, float]:
        """返回 (raw, normalized, curved) —— 三层都返回，方便调参时看清是谁在起作用。"""
        raw = self.sensor(w, d)
        norm = bookend(raw, self.lo, self.hi, self.invert)
        curved = CURVES[self.curve](norm, m=self.m, k=self.k, b=self.b, c=self.c)
        return raw, norm, curved


@dataclass
class Action:
    name: str
    weight: float = 1.0                                   # action 自身权重
    considerations: List[Consideration] = field(default_factory=list)


# ----------------------------------------------------------------------------
# 3. 《生境》的最小行为集
#
#   设计的唯一约束：四个动作都由"环境的凸组合"驱动，没有一个由单一传感器独断，
#   否则该动作只会是某个传感器的镜像，调它等于调传感器 —— 那是伪勤劳。
# ----------------------------------------------------------------------------

def build_actions() -> List[Action]:
    return [
        Action("drink", 1.0, [
            Consideration("口渴驱动",  lambda w, d: d.hydration,        0.3, 0.9, invert=True,
                          curve="logistic", m=9.0, c=0.55),
            Consideration("水源可达",  lambda w, d: w.water_dist,       1.0, 25.0, invert=True),
            Consideration("无威胁压近", lambda w, d: w.threat_dist,      2.0, 20.0, curve="logistic", m=6.0, c=0.35),
        ]),
        Action("warm", 1.0, [
            Consideration("失温驱动",  lambda w, d: d.warmth,           0.3, 0.8, invert=True,
                          curve="logistic", m=9.0, c=0.5),
            Consideration("热源可达",  lambda w, d: w.heat_dist,        1.0, 20.0, invert=True),
            Consideration("遮蔽可用",  lambda w, d: w.cover_density,    0.1, 0.9, curve="logistic", m=5.0, c=0.5),
        ]),
        Action("flee", 1.4, [                                            # 逃生：优先级别高于日常
            Consideration("威胁逼近",  lambda w, d: w.threat_dist,      1.5, 15.0, invert=True,
                          curve="logistic", m=10.0, c=0.55),
            Consideration("体力尚存",  lambda w, d: d.energy,           0.2, 0.6, curve="logistic", m=8.0, c=0.4),
        ]),
        Action("explore", 0.9, [
            Consideration("无紧迫需求", lambda w, d: min(d.hydration, d.warmth), 0.3, 0.75,
                          curve="logistic", m=7.0, c=0.5),
            Consideration("体力充沛",  lambda w, d: d.energy,           0.4, 0.85, curve="logistic", m=7.0, c=0.55),
            Consideration("寒潮压制",  lambda w, d: w.frost,            0.0, 0.8, invert=True),
        ]),
    ]


# ----------------------------------------------------------------------------
# 4. 求解器本体
# ----------------------------------------------------------------------------

@dataclass
class SolverConfig:
    temperature: float = 0.7          # T：随机性旋钮
    aggregation: str = "geometric"    # product | geometric | additive
    commitment: float = 1.0           # 上一动作的 U 乘数（1.0 = 关闭）
    seed: int = 20261011


def aggregate(cvals: Sequence[float], mode: str) -> float:
    if not cvals:
        return 0.0
    if mode == "product":
        out = 1.0
        for v in cvals:
            out *= v
        return out
    if mode == "geometric":
        out = 1.0
        for v in cvals:
            out *= v
        return out ** (1.0 / len(cvals))       # 维度补偿：让 K 的数量变化不自动压低分数
    if mode == "additive":
        return sum(cvals) / len(cvals)
    raise ValueError(f"unknown aggregation: {mode}")


def score_actions(w: World, d: Drive, actions: List[Action], cfg: SolverConfig,
                  prev: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float, float, float]]]]:
    """返回 (U 值字典, 逐 consideration 明细)。注意：只算分，不采样 —— 纯函数。"""
    U: Dict[str, float] = {}
    trace: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for a in actions:
        rows = [c.evaluate(w, d) for c in a.considerations]
        trace[a.name] = [(c.name, r[1], r[2], r[0]) for c, r in zip(a.considerations, rows)]
        u = a.weight * aggregate([r[2] for r in rows], cfg.aggregation)
        if prev is not None and a.name == prev:
            u *= cfg.commitment                 # 乘法加成：保留"U=0 则该动作仍为 0"的性质
        U[a.name] = u
    return U, trace


def softmax(U: Dict[str, float], T: float) -> Dict[str, float]:
    """数值稳定的 softmax。注意恒等性：U 全体加常数不改变结果；U 全体乘 r 等价于 T 除 r。"""
    if T <= 0:                                   # T→0 退化为 argmax（确定性）
        top = max(U.values())
        winners = [k for k, v in U.items() if abs(v - top) < 1e-12]
        return {k: (1.0 / len(winners) if k in winners else 0.0) for k in U}
    top = max(U.values())
    exps = {k: math.exp((v - top) / T) for k, v in U.items()}
    z = sum(exps.values())
    if z <= 0:                                   # 全部 0 → 回退均匀分布（并记为事件）
        n = len(U)
        return {k: 1.0 / n for k in U}
    return {k: v / z for k, v in exps.items()}


def sample(rng: random.Random, dist: Dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for k, v in dist.items():
        acc += v
        if r <= acc:
            return k
    return list(dist.keys())[-1]


# ----------------------------------------------------------------------------
# 5. 世界推进（最小确定性动力学）
# ----------------------------------------------------------------------------

def step_world(w: World, d: Drive, act: str) -> None:
    """执行一步。这里是"环境↔状态"唯一的耦合点，也是 ρ（环境耦合度）的来源。"""
    d.hydration = clamp01(d.hydration - 0.012)
    d.warmth = clamp01(d.warmth - 0.010 - 0.02 * w.frost)
    d.energy = clamp01(d.energy - 0.006)
    if act == "drink":
        d.hydration = clamp01(d.hydration + 0.09)
        d.energy = clamp01(d.energy - 0.004)
    elif act == "warm":
        d.warmth = clamp01(d.warmth + 0.09)
    elif act == "flee":
        d.energy = clamp01(d.energy - 0.03)
    elif act == "explore":
        d.energy = clamp01(d.energy - 0.015)

    # 威胁随机游走（受 seed 控制，可复现）
    return None


def rollout(w: World, d: Drive, actions: List[Action], cfg: SolverConfig,
            steps: int = 600, threat_rng: Optional[random.Random] = None) -> List[str]:
    rng = random.Random(cfg.seed)
    trng = threat_rng or random.Random(cfg.seed + 1)
    hist: List[str] = []
    prev = None
    for _ in range(steps):
        if trng.random() < 0.05:                 # 威胁偶靠近/远离
            w.threat_dist = clamp01(w.threat_dist + trng.uniform(-6.0, 4.0)) * 60.0
            w.threat_dist = max(1.0, min(60.0, w.threat_dist))
        U, _ = score_actions(w, d, actions, cfg, prev=prev)
        dist = softmax(U, cfg.temperature)
        act = sample(rng, dist)
        step_world(w, d, act)
        hist.append(act)
        prev = act
    return hist


# ----------------------------------------------------------------------------
# 6. 指标
# ----------------------------------------------------------------------------

def entropy_bits(counts: Dict[str, int]) -> Tuple[float, float]:
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    H = 0.0
    for c in counts.values():
        if c:
            p = c / total
            H -= p * math.log2(p)
    return H, H / math.log2(len(counts)) if len(counts) > 1 else 0.0


def switch_rate(hist: Sequence[str]) -> float:
    if len(hist) < 2:
        return 0.0
    return sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1]) / (len(hist) - 1)


def tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """全变差距离：用来量"环境变了，行为分布变了多少" —— 这是 ρ 的原始雏形。"""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def dist_of(hist: Sequence[str], keys: Sequence[str]) -> Dict[str, float]:
    n = len(hist) or 1
    return {k: hist.count(k) / n for k in keys}


# ----------------------------------------------------------------------------
# 7. 黄金向量：固定的一组输入用例 → U 值 + softmax 分布（纯确定性，无采样）
# ----------------------------------------------------------------------------

GOLDEN_CASES = [
    ("干渴临界且水近",   World(water_dist=2.0,  heat_dist=20.0, threat_dist=45.0, cover_density=0.2, frost=0.0), Drive(hydration=0.25, warmth=0.6, energy=0.7)),
    ("干渴但水远",       World(water_dist=22.0, heat_dist=20.0, threat_dist=45.0, cover_density=0.2, frost=0.0), Drive(hydration=0.25, warmth=0.6, energy=0.7)),
    ("威胁贴脸",         World(water_dist=8.0,  heat_dist=15.0, threat_dist=3.0,  cover_density=0.4, frost=0.0), Drive(hydration=0.7, warmth=0.5, energy=0.6)),
    ("威胁贴脸但力竭",   World(water_dist=8.0,  heat_dist=15.0, threat_dist=3.0,  cover_density=0.4, frost=0.0), Drive(hydration=0.7, warmth=0.5, energy=0.05)),
    ("寒潮+有遮蔽",      World(water_dist=12.0, heat_dist=6.0,  threat_dist=40.0, cover_density=0.8, frost=0.7), Drive(hydration=0.6, warmth=0.2, energy=0.7)),
    ("寒潮+无遮蔽",      World(water_dist=12.0, heat_dist=18.0, threat_dist=40.0, cover_density=0.05, frost=0.7), Drive(hydration=0.6, warmth=0.2, energy=0.7)),
    ("一切充足",         World(water_dist=10.0, heat_dist=10.0, threat_dist=50.0, cover_density=0.5, frost=0.0), Drive(hydration=0.85, warmth=0.85, energy=0.9)),
    ("全低且水源热源皆远", World(water_dist=25.0, heat_dist=25.0, threat_dist=50.0, cover_density=0.05, frost=0.9), Drive(hydration=0.1, warmth=0.05, energy=0.15)),
]


def write_golden(path: str, cfg: SolverConfig, temps: Sequence[float] = (0.05, 0.2, 0.5, 0.7, 1.0, 2.0)) -> None:
    actions = build_actions()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["case", "case_desc", "T", "action", "U", "P", "aggregation", "commitment"])
        for idx, (desc, world, drive) in enumerate(GOLDEN_CASES):
            U, _ = score_actions(world, drive, actions, cfg)
            for T in temps:
                P = softmax(U, T)
                for aname in U:
                    w_.writerow([f"case_{idx:02d}", desc, f"{T:.3f}", aname,
                                 f"{U[aname]:.10f}", f"{P[aname]:.10f}",
                                 cfg.aggregation, f"{cfg.commitment:.3f}"])
    print(f"[golden] 已写出 {path}  ({len(GOLDEN_CASES)} cases × {len(temps)} T × {len(actions)} actions)")


# ----------------------------------------------------------------------------
# 8. 温度扫描：H(a) 随 T 的变化 —— 第 6 课预注册的对象
# ----------------------------------------------------------------------------

def run_sweep(path: str, base: SolverConfig, steps: int = 600) -> None:
    actions = build_actions()
    rows = []
    for T in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]:
        cfg = SolverConfig(temperature=T, aggregation=base.aggregation,
                           commitment=base.commitment, seed=base.seed)
        world = World(); drive = Drive()
        hist = rollout(world, drive, actions, cfg, steps=steps)
        counts = {a.name: hist.count(a.name) for a in actions}
        H, Hn = entropy_bits(counts)
        rows.append((T, H, Hn, switch_rate(hist), counts))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["T", "H_bits", "H_norm", "switch_rate", "steps"] + [a.name for a in build_actions()])
        for T, H, Hn, sr, counts in rows:
            w_.writerow([T, f"{H:.4f}", f"{Hn:.4f}", f"{sr:.4f}", steps] + [counts[a.name] for a in build_actions()])
    print(f"[sweep] 已写出 {path}")
    print(f"{'T':>6} {'H(bits)':>8} {'H/Hmax':>8} {'切换率':>8}")
    for T, H, Hn, sr, _ in rows:
        print(f"{T:>6} {H:>8.4f} {Hn:>8.4f} {sr:>8.4f}")


# ----------------------------------------------------------------------------
# 9. 自检：不需要人类肉眼比对的基本性质
# ----------------------------------------------------------------------------

def selftest() -> None:
    acts = build_actions()
    cfg = SolverConfig()

    # (1) softmax 归一
    U = {"a": 0.3, "b": 0.9, "c": 0.1}
    for T in (0.05, 0.5, 1.0, 5.0):
        P = softmax(U, T)
        assert abs(sum(P.values()) - 1.0) < 1e-12, f"softmax 未归一 @T={T}"
    print("✓ (1) softmax 对所有 T 归一到 1")

    # (2) T→0 = argmax
    assert softmax(U, 0.0)["b"] == 1.0, "T→0 未退化为 argmax"
    print("✓ (2) T→0 退化为 argmax")

    # (3) 平移不变性：全体 U 加常数，分布不变
    shifted = {k: v + 7.7 for k, v in U.items()}
    p1, p2 = softmax(U, 0.7), softmax(shifted, 0.7)
    assert max(abs(p1[k] - p2[k]) for k in U) < 1e-12
    print("✓ (3) 平移不变：U 全体加常数不改变选择分布 → 绝对分值无意义，只比相对")

    # (4) 尺度/温度不可同时识别：U 乘 r ⇔ T 除 r
    scaled = {k: v * 3.0 for k, v in U.items()}
    assert max(abs(softmax(scaled, 0.7)[k] - softmax(U, 0.7 / 3.0)[k]) for k in U) < 1e-12
    print("✓ (4) 尺度-温度冗余：w 全体×3 与 T÷3 等价 → 必须钉住一个，否则参数不可識別")

    # (5) 乘法聚合的"否决"性质：任一 consideration 为 0 → U = 0
    wZero = World(water_dist=90.0)   # 远超 bookend hi → 水源可达 = 0
    U0, tr = score_actions(wZero, Drive(hydration=0.1), acts,
                           SolverConfig(aggregation="product"))
    assert U0["drink"] == 0.0, "product 聚合未继承否决性质"
    print("✓ (5) product 聚合：任一 consideration 归零 → action 被否决（0 既是无穷小惩罚，也是硬性 Gate）")

    # (6) commitment 以乘法加成时保留否决性质
    Uc, _ = score_actions(wZero, Drive(hydration=0.1), acts,
                          SolverConfig(aggregation="product", commitment=1.25), prev="drink")
    assert Uc["drink"] == 0.0, "乘法 commitment 破坏了否决性质"
    print("✓ (6) commitment 用乘法 → 保留否决性质（改用加法会把被否决的动作重新救活）")

    # (7) 维度补偿：product 随 consideration 数量衰减，geometric 不衰减
    same = [0.5] * 8
    print(f"✓ (7) 维度对比：8 个 0.5 → product={aggregate(same,'product'):.4f} "
          f"geometric={aggregate(same,'geometric'):.4f} additive={aggregate(same,'additive'):.4f}")

    # (8) 黄金用例的可复现性（同 seed 同结果）
    h1 = rollout(World(), Drive(), acts, cfg, steps=200)
    h2 = rollout(World(), Drive(), acts, cfg, steps=200)
    assert h1 == h2, "同 seed 结果不一致 → 存在隐藏随机源"
    print("✓ (8) 同 seed 完全可复现（黄金 trace 成立的前提）")

    print("\nselftest ALL PASS")


# ----------------------------------------------------------------------------
# 10. 环境耦合度 ρ 的雏形：换环境，看行为分布变多少
# ----------------------------------------------------------------------------

def coupling(base: SolverConfig, steps: int = 400) -> None:
    acts = build_actions()
    keys = [a.name for a in acts]
    envA = dict(water_dist=3.0, heat_dist=18.0, threat_dist=45.0, cover_density=0.15, frost=0.0)  # 干热无水旁
    envB = dict(water_dist=18.0, heat_dist=3.0, threat_dist=45.0, cover_density=0.8, frost=0.0)  # 近热源有遮蔽
    dA = rollout(World(**envA), Drive(), acts, base, steps=steps)
    dB = rollout(World(**envB), Drive(), acts, base, steps=steps)
    pA, pB = dist_of(dA, keys), dist_of(dB, keys)
    HA, HAn = entropy_bits({k: dA.count(k) for k in keys})
    HB, HBn = entropy_bits({k: dB.count(k) for k in keys})
    print("[ρ 雏形] TV(A,B) =", round(tv_distance(pA, pB), 4), " (0 = 环境白改，越大 = 环境真在驱动行为)")
    print("  A:", {k: round(v, 3) for k, v in pA.items()}, f"H={HA:.3f}")
    print("  B:", {k: round(v, 3) for k, v in pB.items()}, f"H={HB:.3f}")


# ----------------------------------------------------------------------------
# 11. CLI
# ----------------------------------------------------------------------------

def analyze(base: SolverConfig, steps: int = 600) -> Dict[str, float]:
    """把「行为多样性」拆成两个来源 —— 这是 PROBE 最要紧的一次定义工作。

    恒等式（信息论基础事实）：  H(A) = H(A|S) + I(A;S)

      H(A)    行为熵：整段 rollout 里实际被选出的动作的边际熵。
                     它同时被"策略随机"和"情境多样"贡献，单独看会被误导。
      H(A|S)  策略噪声：在每一个具体状态 s 上，策略分布自身的熵（= T 直接控制的东西）。
      I(A;S)  情境决定度：世界/内驱状态到底决定了多少行为。
                     I→0 ⇒ 这只生物只是在掷骰子，环境是装饰；这就是"生态是假的"的数学定义。

    ⚠️ 这里的 I(A;S) 是对访问到的状态做样本估计，不是精确值；够用于比较不同 T，别当绝对量写进 README。
    """
    acts = build_actions()
    keys = [a.name for a in acts]
    cfg = base
    world, drive = World(), Drive()
    rng = random.Random(cfg.seed)
    trng = random.Random(cfg.seed + 1)
    prev = None
    dists: List[Dict[str, float]] = []
    hist: List[str] = []
    for _ in range(steps):
        if trng.random() < 0.05:
            world.threat_dist = max(1.0, min(60.0, world.threat_dist + trng.uniform(-6.0, 4.0)))
        U, _ = score_actions(world, drive, acts, cfg, prev=prev)
        P = softmax(U, cfg.temperature)
        a = sample(rng, P)
        dists.append(P)
        hist.append(a)
        step_world(world, drive, a)
        prev = a

    counts = {k: hist.count(k) for k in keys}
    H_behavior, _ = entropy_bits(counts)
    entropies = [entropy_bits({k: int(round(P[k] * 1e6)) for k in keys})[0] for P in dists]
    H_policy = sum(entropies) / len(entropies)
    conf = sum(max(P.values()) for P in dists) / len(dists)
    greedy = {max(P, key=P.get) for P in dists}
    return {
        "T": cfg.temperature,
        "H_behavior": H_behavior,
        "H_policy_given_S": H_policy,
        "I_A_S": H_behavior - H_policy,
        "mean_confidence": conf,
        "switch_rate": switch_rate(hist),
        "greedy_action_count": len(greedy),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="《生境 NICHE》决策求解器参考实现")
    ap.add_argument("--selftest", action="store_true", help="跑性质自检")
    ap.add_argument("--golden", metavar="CSV", help="写出黄金向量 CSV")
    ap.add_argument("--sweep", metavar="CSV", help="跑温度扫描并写出 CSV")
    ap.add_argument("--analyze", action="store_true", help="信息论分解：H(A) = H(A|S) + I(A;S)，扫描 T")
    ap.add_argument("--coupling", action="store_true", help="算环境耦合度雏形")
    ap.add_argument("--explain", metavar="CASE", type=int, default=None,
                    help="打印某个黄金用例的逐层归因（0–7）")
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--agg", default="geometric", choices=["product", "geometric", "additive"])
    ap.add_argument("--commitment", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    cfg = SolverConfig(temperature=args.T, aggregation=args.agg,
                       commitment=args.commitment, seed=args.seed)

    if args.selftest:
        selftest()
    if args.golden:
        write_golden(args.golden, cfg)
    if args.sweep:
        run_sweep(args.sweep, cfg, steps=args.steps)
    if args.coupling:
        coupling(cfg, steps=args.steps)
    if args.analyze:
        print(f"{'T':>7} {'H(A)行为':>10} {'H(A|S)策略噪声':>15} {'I(A;S)情境决定':>15} {'置信度':>8} {'切换率':>8} {'贪心动作数':>10}")
        for T in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]:
            c = SolverConfig(temperature=T, aggregation=cfg.aggregation,
                             commitment=cfg.commitment, seed=cfg.seed)
            r = analyze(c, steps=args.steps)
            print(f"{r['T']:>7} {r['H_behavior']:>10.4f} {r['H_policy_given_S']:>15.4f} "
                  f"{r['I_A_S']:>15.4f} {r['mean_confidence']:>8.4f} {r['switch_rate']:>8.4f} "
                  f"{r['greedy_action_count']:>10}")
    if args.explain is not None:
        acts = build_actions()
        desc, world, drive = GOLDEN_CASES[args.explain]
        U, trace = score_actions(world, drive, acts, cfg, prev=None)
        P = softmax(U, cfg.temperature)
        print(f"\n=== {args.explain}: {desc} (T={cfg.temperature}, agg={cfg.aggregation}) ===")
        print(f"world={world}\ndrive={drive}\n")
        for a in acts:
            print(f"[{a.name}] weight={a.weight} U={U[a.name]:.6f} P={P[a.name]:.4f}")
            for cname, norm, curved, raw in trace[a.name]:
                print(f"    - {cname:<8s} raw={raw:7.3f} norm={norm:6.3f} curved={curved:6.3f}")
        print()
    if not any([args.selftest, args.golden, args.sweep, args.coupling, args.analyze,
                args.explain is not None]):
        ap.print_help()


if __name__ == "__main__":
    main()
