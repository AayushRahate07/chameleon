# ============================================================
# KV-cache eviction core: base policies, exact offline optimum
# (vectorized min-cost flow + HiGHS LP fallback), and simulator.
# ============================================================
import sys
import time
import numpy as np


class Policy:
    """Base class. Online policies only see attention of steps < t (via observe)."""
    name = "base"
    online = True
    uses_attention = False

    def reset(self, T, k):
        self.T, self.k = T, k

    def select_evictions(self, t, cache):
        """cache: list of token ids (includes t). Return ids to evict (never t)."""
        raise NotImplementedError

    def observe(self, t, cache, attn):
        """attn[i] = attention of step t on cache[i], renormalised over the cache."""
        pass


# ---------------------------------------------------------------- Base: online policies
class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, T, k):
        super().reset(T, k)
        self.rng = np.random.default_rng(self.seed)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        past = cache[:-1]
        return [past[self.rng.integers(len(past))]]


class SlidingWindow(Policy):
    """Keep the k most recent tokens (FIFO). O(1)."""
    name = "SlidingWindow"

    def select_evictions(self, t, cache):
        return [cache[0]] if len(cache) > self.k else []


class StreamingLLM(Policy):
    """Keep the first `sinks` tokens (attention sinks) + the most recent. O(1)."""
    name = "StreamingLLM"

    def __init__(self, sinks=4):
        self.sinks = sinks

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        for j in cache:
            if j >= self.sinks:
                return [j]
        return [cache[0]]


def _hh_key(score, rule, t):
    """Ranking key for heavy-hitter policies.
    'sum' = cumulative attention (original H2O rule).
    'avg' = average attention per step alive = cumulative / age (age-bias fix)."""
    if rule == "sum":
        return lambda j: score[j]
    return lambda j: score[j] / max(t - j, 1)


class HeavyHitter(Policy):
    """H2O-style heavy hitters: protect the recent_frac*k newest tokens, evict the lowest-ranked
    of the rest (rule 'avg' by default, 'sum' = original H2O). recent_frac = 0 -> 'HeavyHitter-only'. O(k)."""
    uses_attention = True

    def __init__(self, recent_frac=0.5, rule="avg", name="H2O"):
        assert rule in ("sum", "avg")
        self.recent_frac, self.rule, self.name = recent_frac, rule, name

    def reset(self, T, k):
        super().reset(T, k)
        self.score = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        r = max(1, int(self.recent_frac * self.k))
        cand = cache[:len(cache) - r] or cache[:-1]
        return [min(cand, key=_hh_key(self.score, self.rule, t))]

    def observe(self, t, cache, attn):
        self.score[np.asarray(cache)] += attn


class Hybrid(Policy):
    """Sinks + recent window + heavy hitters (StreamingLLM U H2O), rule 'avg' by default. O(k)."""
    name = "Hybrid"
    uses_attention = True

    def __init__(self, sinks=4, recent_frac=0.25, rule="avg"):
        assert rule in ("sum", "avg")
        self.sinks, self.recent_frac, self.rule = sinks, recent_frac, rule

    def reset(self, T, k):
        super().reset(T, k)
        self.score = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        r = max(1, int(self.recent_frac * self.k))
        cand = [j for j in cache[:len(cache) - r] if j >= self.sinks]
        if not cand:
            cand = [j for j in cache[:-1] if j >= self.sinks] or cache[:-1]
        return [min(cand, key=_hh_key(self.score, self.rule, t))]

    def observe(self, t, cache, attn):
        self.score[np.asarray(cache)] += attn


# ---------------------------------------------------------------- Base: offline references
class BeladyGreedy(Policy):
    """Evict the token with the least FUTURE attention sum_{s>=t} A[s, j]. Not optimal (permanent eviction)."""
    name = "Belady-Greedy (offline)"
    online = False

    def __init__(self, A):
        self.F = np.flip(np.cumsum(np.flip(A, 0), 0), 0)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        return [min(cache[:-1], key=lambda j: self.F[t, j])]


class ScheduledPolicy(Policy):
    """Replays a schedule: token j is kept during steps j..end[j]."""
    online = False

    def __init__(self, end, name="Optimal (offline)"):
        self.end, self.name = np.asarray(end), name

    def select_evictions(self, t, cache):
        return [j for j in cache if j != t and self.end[j] < t]


class FullCache(Policy):
    name = "Full cache"

    def select_evictions(self, t, cache):
        return []


# ---------------------------------------------------------------- exact offline optimum (min-cost flow)
def optimal_schedule(A, k, scale=1_000_000):
    """EXACT offline optimum by reduction to min-cost flow (OR-Tools).
    Time nodes 0..T + one node per token. Arcs: t->t+1 cap k cost 0 (empty slot);
    tok_j -> e+1 cap 1 cost C - V(j,e) (token j kept during j..e), V(j,e) = sum_{s=j..e} A[s,j].
    Supply +k at 0, -k at T; node j demands 1, tok_j supplies 1 (forces every token in).
    Totally unimodular -> integral. O(T^2) arcs. Returns end[]."""
    from ortools.graph.python import min_cost_flow
    T = A.shape[0]
    colcum = np.cumsum(A, axis=0)
    C = int(scale * (T + 1))
    j_idx, e_idx = np.triu_indices(T)
    v = colcum[e_idx, j_idx]
    tails = np.concatenate([np.arange(T), T + 1 + j_idx])
    heads = np.concatenate([np.arange(1, T + 1), e_idx + 1])
    caps = np.concatenate([np.full(T, k), np.ones(len(j_idx), dtype=np.int64)])
    costs = np.concatenate([np.zeros(T, dtype=np.int64), C - np.round(scale * v).astype(np.int64)])
    smcf = min_cost_flow.SimpleMinCostFlow()
    arcs = smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)
    sup = np.zeros(2 * T + 1, dtype=np.int64)
    sup[0] += k
    sup[T] -= k
    sup[:T] -= 1
    sup[T + 1:] += 1
    smcf.set_nodes_supplies(np.arange(2 * T + 1), sup)
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow failed: status {status}")
    flows = smcf.flows(arcs)[T:]
    end = np.full(T, -1)
    nz = np.nonzero(flows)[0]
    end[j_idx[nz]] = e_idx[nz]
    assert (end >= np.arange(T)).all(), "every token must be scheduled"
    return end


def _lp_optimum(A, k):
    """Same network solved as an LP with HiGHS (fallback when OR-Tools is missing; slow for large T)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    T = A.shape[0]
    col = np.cumsum(A, axis=0)
    C = float(T + 1)
    j_idx, e_idx = np.triu_indices(T)
    tails = np.r_[np.arange(T), T + 1 + j_idx]
    heads = np.r_[np.arange(1, T + 1), e_idx + 1]
    cost = np.r_[np.zeros(T), C - col[e_idx, j_idx]]
    cap = np.r_[np.full(T, k), np.ones(len(j_idx))]
    na = len(tails)
    B = coo_matrix(
        (np.r_[np.ones(na), -np.ones(na)], (np.r_[tails, heads], np.r_[np.arange(na), np.arange(na)])),
        shape=(2 * T + 1, na),
    ).tocsr()
    sup = np.zeros(2 * T + 1)
    sup[0] += k
    sup[T] -= k
    sup[:T] -= 1
    sup[T + 1:] += 1
    r = linprog(cost, A_eq=B, b_eq=sup, bounds=np.c_[np.zeros(na), cap], method="highs-ds")
    assert r.status == 0, r.message
    x = np.round(r.x[T:]).astype(int)
    end = np.full(T, -1)
    nz = np.nonzero(x)[0]
    end[j_idx[nz]] = e_idx[nz]
    assert (end >= np.arange(T)).all()
    return end


_WARNED = [False]


def get_optimum(A, k):
    try:
        return optimal_schedule(A, k)
    except ImportError:
        if not _WARNED[0]:
            print("  [warn] ortools not installed: using the LP fallback (much slower for T > 256)", file=sys.stderr)
            _WARNED[0] = True
        return _lp_optimum(A, k)


# ---------------------------------------------------------------- simulator
def simulate(A, k, policy, record=False):
    """Run `policy` on A with capacity k. Returns (per-step retained attention, mean decision seconds,
    occupancy matrix M[t, j] = 1 if j cached at step t, or None)."""
    T = A.shape[0]
    policy.reset(T, k)
    cache, mass = [], np.zeros(T)
    M = np.zeros((T, T), dtype=bool) if record else None
    tt = 0.0
    for t in range(T):
        cache.append(t)
        t0 = time.perf_counter()
        ev = policy.select_evictions(t, cache)
        tt += time.perf_counter() - t0
        if ev:
            evs = set(ev)
            assert t not in evs, "cannot evict the current token"
            cache = [j for j in cache if j not in evs]
        assert len(cache) <= k, f"{policy.name} exceeded capacity at step {t}"
        if record:
            M[t, cache] = True
        row = A[t, cache]
        mass[t] = row.sum()
        t0 = time.perf_counter()
        policy.observe(t, cache, row / max(row.sum(), 1e-12))
        tt += time.perf_counter() - t0
    return mass, tt / T, M


def teacher_forced(A, k, end, policy):
    """Hand the policy the OPTIMUM's cache every step; how often does it drop a token the optimum drops?
    Returns (agreement, chance level)."""
    T = A.shape[0]
    policy.reset(T, k)
    cache = []
    hit = n = 0
    chance = 0.0
    for t in range(T):
        cache.append(t)
        opt_ev = {j for j in cache[:-1] if end[j] < t}
        if len(cache) > k and opt_ev:
            ev = policy.select_evictions(t, cache)
            hit += bool(ev) and ev[0] in opt_ev
            n += 1
            chance += len(opt_ev) / (len(cache) - 1)
        cache = [j for j in cache if j not in opt_ev]
        row = A[t, cache]
        policy.observe(t, cache, row / max(row.sum(), 1e-12))
    return hit / max(n, 1), chance / max(n, 1)


def synthetic_attention(T, seed=0, sinks=2, n_heavy=8, decay=0.15):
    """Random attention with sinks, recency bias and a few re-read 'heavy' tokens. Testing only."""
    rng = np.random.default_rng(seed)
    heavy = rng.choice(np.arange(sinks, T), size=min(n_heavy, T - sinks), replace=False)
    A = np.zeros((T, T))
    for t in range(T):
        j = np.arange(t + 1)
        logits = -decay * (t - j) + rng.normal(0, 1.0, t + 1)
        logits[j < sinks] += 3.0
        logits[np.isin(j, heavy)] += 2.0
        w = np.exp(logits - logits.max())
        A[t, :t + 1] = w / w.sum()
    return A
