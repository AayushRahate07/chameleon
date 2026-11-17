# ============================================================
# The Gauntlet: unified policy registry (15 policies + ablations),
# equal DEV-tuning grid fairness (TUNE_BUDGET=8), stress traces,
# hill-climb adversary, statistical verification & self-tests.
# ============================================================
import copy
import itertools
import json
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.core import (
    RandomPolicy, SlidingWindow, StreamingLLM, HeavyHitter, Hybrid,
    BeladyGreedy, ScheduledPolicy, get_optimum, _lp_optimum,
    simulate, synthetic_attention,
)
from src.chameleon import Chameleon, LastAttention, EXPERTS
from src.hybrids import HitLRU, AdaptiveWindow, HARP
from src.imitation import train_imitation, dagger_train

TUNE_BUDGET = 8   # FAIRNESS: every tunable policy may try at most this many settings on DEV data


@dataclass
class Ctx:
    """Context passed to policy factories. Online/learned policies must ignore A and end."""
    A: np.ndarray = None
    k: int = 0
    end: np.ndarray = None
    students: dict = None
    tuned: dict = None


def _grid(**axes):
    """Cartesian grid; the first value of every axis is the policy's default, so grid[0] = default."""
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


@dataclass
class Entry:
    name: str
    family: str            # "online" | "learned" | "offline" | "ablation"
    factory: Callable      # factory(ctx, **settings) -> Policy
    author: str = ""
    grid: list = None      # settings tried by the 'tune' stage (grid[0] = default)
    legacy: dict = None    # settings reproducing the pre-tuning version

    def params(self, ctx):
        if not self.grid:
            return {}
        if ctx.tuned is None:
            return dict(self.grid[0])
        if self.name not in ctx.tuned:
            raise RuntimeError(f"no tuned settings for '{self.name}' - run the 'tune' stage")
        return dict(ctx.tuned[self.name])

    def make(self, ctx, params=None):
        return self.factory(ctx, **(self.params(ctx) if params is None else params))


def fingerprint(entry, tuned_all):
    """Exact settings hash over all pools so stale result rows are detected and recomputed."""
    if not entry.grid:
        return "-"
    used = {p: v.get(entry.name) for p, v in sorted(tuned_all.items())}
    if entry.legacy is not None and all(u == entry.legacy for u in used.values()):
        return "-"
    return "c%08x" % zlib.crc32(json.dumps(used, sort_keys=True).encode())


def grid_id(entry):
    return "g%08x" % zlib.crc32(json.dumps(entry.grid, sort_keys=True).encode()) if entry.grid else "-"


def _student(c, nm):
    if not c.students or nm not in c.students:
        raise RuntimeError(f"no trained student '{nm}' for this pool - run the 'train' stage")
    s = copy.deepcopy(c.students[nm])
    s.name = nm
    return s


def registry(ablations=False):
    """Single canonical registry of all 15 policies (+ optional ablations) with equal DEV tuning grids."""
    R = [
        Entry("Random", "online", lambda c: RandomPolicy(), "Base"),
        Entry("SlidingWindow", "online", lambda c: SlidingWindow(), "Base"),
        Entry("StreamingLLM", "online", lambda c, **p: StreamingLLM(**p), "Base",
              _grid(sinks=[4, 1, 2, 8]), dict(sinks=4)),
        Entry("HeavyHitter-only", "online", lambda c, **p: HeavyHitter(0.0, name="HeavyHitter-only", **p), "Base",
              _grid(rule=["avg", "sum"]), dict(rule="sum")),
        Entry("H2O", "online", lambda c, **p: HeavyHitter(name="H2O", **p), "Base",
              _grid(rule=["avg", "sum"], recent_frac=[0.5, 0.1, 0.25, 0.75]), dict(rule="sum", recent_frac=0.5)),
        Entry("Hybrid", "online", lambda c, **p: Hybrid(**p), "Base",
              _grid(rule=["avg", "sum"], recent_frac=[0.25, 0.1, 0.5, 0.75]), dict(rule="sum", recent_frac=0.25)),
        Entry("LastAttention", "online", lambda c, **p: LastAttention(**p), "Aayush",
              _grid(sinks=[0, 4]), dict(sinks=0)),
        Entry("Chameleon", "online", lambda c, **p: Chameleon(**p), "Aayush",
              _grid(eta_scale=[1.0, 0.3, 3.0, 10.0], share=[5.0, 1.0]), dict(eta_scale=1.0, share=5.0)),
        Entry("HitLRU", "online", lambda c, **p: HitLRU(**p), "Krishna",
              _grid(theta=[2.0, 0.5, 1.0, 4.0], recent_frac=[0.2, 0.5]), dict(theta=2.0, recent_frac=0.2)),
        Entry("AdaptiveWindow", "online", lambda c, **p: AdaptiveWindow(**p), "Krishna",
              _grid(theta=[2.0, 0.5, 1.0, 4.0], q=[0.9, 0.8]), dict(theta=2.0, q=0.9)),
        Entry("HARP", "online",
              lambda c, protect, decay: HARP(decay=decay, recent_frac=protect, sinks=4 if protect > 0 else 0),
              "Kaushal", _grid(protect=[0.25, 0.1, 0.5, 0.0], decay=[0.9, 0.7]), dict(protect=0.0, decay=0.9)),
        Entry("Imitation", "learned", lambda c: _student(c, "Imitation"), "Aayush"),
        Entry("DAgger", "learned", lambda c: _student(c, "DAgger"), "Aayush"),
        Entry("Belady-Greedy (offline)", "offline", lambda c: BeladyGreedy(c.A), "Base"),
        Entry("Optimal (offline)", "offline", lambda c: ScheduledPolicy(c.end), "Base"),
    ]
    if ablations:
        A = [
            ("Chameleon: equal weights", dict(learn=False)),
            ("Chameleon: no fixed-share", dict(share=0.0)),
            ("Chameleon: no sinks", dict(sinks=0)),
            ("Chameleon: no grace", dict(grace=0.0)),
            ("Chameleon: eta x3", dict(eta_scale=3.0)),
            ("Chameleon: follow-the-leader", dict(sharp=8.0)),
            ("Chameleon: lagged loss k/8", dict(lag_frac=0.125)),
        ]
        A += [(f"Chameleon: without {e}", dict(experts=[j for j in range(len(EXPERTS)) if j != i]))
              for i, e in enumerate(EXPERTS)]
        A += [("HitLRU: lam=1", "hitlru")]
        for nm, kw in A:
            if kw == "hitlru":
                R.append(Entry(nm, "ablation", lambda c, nm=nm: HitLRU(lam=1.0, name=nm), "Krishna"))
            else:
                R.append(Entry(nm, "ablation", lambda c, nm=nm, kw=kw: Chameleon(name=nm, **kw), "Aayush"))
    for e in R:
        assert not e.grid or len(e.grid) <= TUNE_BUDGET, f"{e.name}: tuning grid larger than TUNE_BUDGET"
    return R


def needs_future(e):
    return e.family == "offline"


# ---------------------------------------------------------------- Stress traces & Adversary
def shifting_heavy_hitters(T, seed):
    """Kaushal's diagnostic: one heavy-hitter group (8..15) is replaced by another (T/2..T/2+7) halfway through."""
    rng = np.random.default_rng(seed)
    A = np.zeros((T, T))
    group_a = np.arange(8, min(16, T))
    group_b = np.arange(max(8, T // 2), min(T // 2 + 8, T))
    for t in range(T):
        j = np.arange(t + 1)
        logits = rng.normal(0.0, 0.65, size=t + 1)
        logits[np.isin(j, group_a if t < T // 2 else group_b)] += 3.0
        logits -= 0.025 * (t - j)
        w = np.exp(logits - logits.max())
        A[t, :t + 1] = w / w.sum()
    return A


def stress_traces(T, seed=0):
    """Six hand-built attention patterns including Kaushal's shifting heavy hitters."""
    rng = np.random.default_rng(seed)

    def recency(decay=0.25):
        A = np.zeros((T, T))
        for t in range(T):
            A[t, :t + 1] = np.exp(-decay * (t - np.arange(t + 1)))
        return A / A.sum(1, keepdims=True)

    def blend(base, target, w):
        return (1 - w[:, None]) * base + w[:, None] * target

    def finish(A):
        A = np.tril(A * np.exp(rng.normal(0, 0.15, A.shape)))
        return A / A.sum(1, keepdims=True)

    base = recency()
    tr = {}
    steps = np.arange(T)
    tr["pure recency"] = finish(base)
    sk = np.zeros((T, T))
    sk[:, :4] = 0.25
    tr["sinks + recency"] = finish(blend(base, np.tril(sk), np.full(T, 0.8)))
    heavy = np.arange(20, T - 20, 40)
    tgt = np.zeros((T, T))
    for t in range(T):
        h = heavy[heavy < t]
        if len(h):
            tgt[t, h] = 1 / len(h)
    tr["periodic re-reads"] = finish(blend(base, tgt, np.where(steps % 32 < 4, 0.7, 0.0)))
    h1, h2 = np.arange(10, 26), np.arange(T // 2 - 40, T // 2 - 24)
    tgt = np.zeros((T, T))
    tgt[:T // 2, h1] = 1 / len(h1)
    tgt[T // 2:, h2] = 1 / len(h2)
    tr["sudden topic change"] = finish(blend(base, np.tril(tgt), np.full(T, 0.5)))
    late = np.arange(T // 4, T // 4 + 4)
    tgt = np.zeros((T, T))
    tgt[:, late] = 0.25
    tr["unpredictable late burst"] = finish(blend(base, np.tril(tgt), np.where(steps >= T - 16, 0.6, 0.0)))
    tr["shifting heavy hitters"] = shifting_heavy_hitters(T, [11, 23, 37][seed % 3] + 100 * (seed // 3))
    return tr


def _softmax_rows(L):
    T = L.shape[0]
    A = np.zeros((T, T))
    for t in range(T):
        z = L[t, :t + 1]
        w = np.exp(z - z.max())
        A[t, :t + 1] = w / w.sum()
    return A


def ratio_on(A, k, entry, students, tuned=None):
    end = get_optimum(A, k)
    om = simulate(A, k, ScheduledPolicy(end))[0][k:].mean()
    pol = entry.make(Ctx(A=A, k=k, end=end, students=students, tuned=tuned))
    return simulate(A, k, pol)[0][k:].mean() / om


def adversary(entry, students, T=64, k=16, iters=150, restarts=3, seed=0, tuned=None):
    """Hill-climb over attention logits to FIND an input where the policy is worst vs the optimum."""
    rng = np.random.default_rng(seed)
    best = 2.0
    for _ in range(restarts):
        A0 = synthetic_attention(T, int(rng.integers(1_000_000)), sinks=int(rng.integers(0, 3)),
                                 n_heavy=int(rng.integers(2, 8)))
        L = np.log(A0 + 1e-9)
        cur = ratio_on(_softmax_rows(L), k, entry, students, tuned)
        for _ in range(iters):
            L2 = L.copy()
            kind = rng.integers(3)
            if kind == 0:
                a = int(rng.integers(0, T - 4))
                L2[a:a + 4] += rng.normal(0, 1.5, (4, T))
            elif kind == 1:
                j = int(rng.integers(0, T - 8))
                a = int(rng.integers(j, T - 2))
                b = min(T, a + int(rng.integers(2, 16)))
                L2[a:b, j] += rng.choice([-3.0, 3.0]) * rng.random() * 2
            else:
                t_ = int(rng.integers(1, T))
                L2[t_, :t_ + 1] += rng.normal(0, 2.0, t_ + 1)
            val = ratio_on(_softmax_rows(L2), k, entry, students, tuned)
            if val < cur:
                L, cur = L2, val
        best = min(best, cur)
    return best


# ---------------------------------------------------------------- Statistics
def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    return np.percentile(r.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def pairwise(df):
    piv = df.pivot_table(index=["text", "seq", "budget"], columns="policy", values="ratio")
    P = list(piv.columns)
    W = pd.DataFrame(0.0, index=P, columns=P)
    for a, b in itertools.permutations(P, 2):
        W.loc[a, b] = (piv[a] > piv[b]).mean() + 0.5 * (piv[a] == piv[b]).mean()
    return W


# ---------------------------------------------------------------- Correctness Self-Tests
def check_causal(entry, A, k, end, students, t0=None, seed=0, params=None):
    """Decisions up to t0 must not change when attention rows after t0 are replaced by random ones."""
    T = A.shape[0]
    t0 = T // 2 if t0 is None else t0
    A2 = A.copy()
    r = np.random.default_rng(seed)
    for t in range(t0 + 1, T):
        w = r.random(t + 1) ** 4
        A2[t, :t + 1] = w / w.sum()
    m1 = simulate(A, k, entry.make(Ctx(A=A, k=k, end=end, students=students), params))[0]
    m2 = simulate(A2, k, entry.make(Ctx(A=A2, k=k, end=end, students=students), params))[0]
    return bool(np.allclose(m1[:t0 + 1], m2[:t0 + 1], atol=1e-12))


def _brute(A, k):
    T = A.shape[0]
    col = np.cumsum(A, 0)
    best = [-1.0]

    def rec(j, ends):
        if j == T:
            best[0] = max(best[0], sum(col[ends[i], i] for i in range(T)))
            return
        for e in range(j, T):
            ends.append(e)
            if all(sum(1 for i in range(j + 1) if i <= s <= ends[i]) <= k for s in range(T)):
                rec(j + 1, ends)
            ends.pop()

    rec(0, [])
    return best[0]


def selftest(ablations=True):
    print("self-tests")
    for seed in range(6):
        A = synthetic_attention(7, seed, sinks=1, n_heavy=2)
        k = 2 + seed % 3
        e = get_optimum(A, k)
        tot = sum(np.cumsum(A, 0)[e[j], j] for j in range(7))
        assert abs(tot - _brute(A, k)) < 1e-6, f"optimum != brute force (seed {seed})"
        e2 = _lp_optimum(A, k)
        tot2 = sum(np.cumsum(A, 0)[e2[j], j] for j in range(7))
        assert abs(tot2 - tot) < 1e-6, "LP fallback != min-cost flow"
    print("  [1] optimum == brute force == LP fallback on 6 small instances")
    tr = [(synthetic_attention(96, s), int(b * 96)) for s in (10, 11) for b in (0.2, 0.3)]
    items = [(A, k, get_optimum(A, k)) for A, k in tr]
    st = dict(Imitation=train_imitation(items), DAgger=dagger_train(items, rounds=1)[-1])
    print("  [2] learned policies train on synthetic data")
    A = synthetic_attention(120, 3)
    k = 24
    end = get_optimum(A, k)
    om = simulate(A, k, ScheduledPolicy(end))[0][k:].sum()
    R = registry(ablations=ablations)
    n_set = 0
    for e in R:
        for p in (e.grid or [None]):
            pol = e.make(Ctx(A=A, k=k, end=end, students=st), p)
            n_set += 1
            m = simulate(A, k, pol)[0][k:].sum()
            assert m <= om + 1e-9, f"{e.name} {p} beats the optimum?!"
            if not needs_future(e):
                assert check_causal(e, A, k, end, st, params=p), f"{e.name} {p} uses future information"
    print(f"  [3] {len(R)} policies, {n_set} settings: capacity respected, none beats the optimum, no online policy sees the future")
    p = Chameleon()
    simulate(A, k, p)
    reg, _ = p.regret()
    assert reg <= p.regret_bound() + 1e-9, "Chameleon regret exceeds its bound"
    print(f"  [4] Chameleon regret {reg:.2f} <= bound {p.regret_bound():.2f}")
    print("self-tests passed")
