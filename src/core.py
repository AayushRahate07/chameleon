# ============================================================
# KV-cache eviction as an online problem: policies, simulator,
# and the exact offline optimum (min-cost flow reduction).
# Pure numpy + OR-tools; no model needed (testable on synthetic data).
# ============================================================
import time
import numpy as np
from collections import deque

# ----------------------------------------------------------------
# Problem model
#   A[t, j] = attention that query token t puts on key token j (j <= t),
#             averaged over layers and heads; each row sums to 1.
#   Cache capacity k. At step t the new token t is always added
#   (its KV is computed anyway); if the cache exceeds k, the policy
#   evicts past tokens. Eviction is PERMANENT (no re-fetch), which
#   is what makes this different from classic paging.
#   Reward at step t = sum of A[t, j] over j in cache.
# ----------------------------------------------------------------


class Policy:
    """Base class. Online policies only see attention of steps < t."""
    name = "base"
    online = True
    uses_attention = False   # True if observe() needs attention scores

    def reset(self, T, k):
        self.T, self.k = T, k

    def select_evictions(self, t, cache):
        """cache: list of token ids (includes t). Return ids to evict (never t)."""
        raise NotImplementedError

    def observe(self, t, cache, attn):
        """attn[i] = attention of step t on cache[i] (renormalised over the cache)."""
        pass


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
    """Keep the k most recent tokens (FIFO). O(1) per step."""
    name = "SlidingWindow"

    def select_evictions(self, t, cache):
        return [cache[0]] if len(cache) > self.k else []


class StreamingLLM(Policy):
    """Always keep the first `sinks` tokens (attention sinks) + most recent. O(1)."""
    name = "StreamingLLM"

    def __init__(self, sinks=4):
        self.sinks = sinks

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        for j in cache:                 # cache is ordered; first non-sink is oldest
            if j >= self.sinks:
                return [j]
        return [cache[0]]


class HeavyHitter(Policy):
    """H2O-style: score = cumulative attention received so far.
    Protect the `recent_frac*k` most recent tokens; evict the lowest-score
    token among the rest. O(k) per step (all scores change every step,
    so a heap does not help).  recent_frac=0 gives pure 'heavy-hitter only'."""

    uses_attention = True

    def __init__(self, recent_frac=0.5, name="H2O"):
        self.recent_frac, self.name = recent_frac, name

    def reset(self, T, k):
        super().reset(T, k)
        self.score = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        r = max(1, int(self.recent_frac * self.k))      # new token always protected
        candidates = cache[:len(cache) - r] or cache[:-1]
        return [min(candidates, key=lambda j: self.score[j])]

    def observe(self, t, cache, attn):
        self.score[np.asarray(cache)] += attn


class Hybrid(Policy):
    """Sinks + recent window + heavy hitters (StreamingLLM ∪ H2O). O(k) per step."""
    name = "Hybrid"
    uses_attention = True

    def __init__(self, sinks=4, recent_frac=0.25):
        self.sinks, self.recent_frac = sinks, recent_frac

    def reset(self, T, k):
        super().reset(T, k)
        self.score = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        r = max(1, int(self.recent_frac * self.k))
        candidates = [j for j in cache[:len(cache) - r] if j >= self.sinks]
        if not candidates:
            candidates = [j for j in cache[:-1] if j >= self.sinks] or cache[:-1]
        return [min(candidates, key=lambda j: self.score[j])]

    def observe(self, t, cache, attn):
        self.score[np.asarray(cache)] += attn


# ---------------------------- offline (hindsight) policies ----------------------------

class BeladyGreedy(Policy):
    """Offline heuristic: evict the token with the least FUTURE attention
    sum_{s>=t} A[s, j]. Knows the future, but greedy => not always optimal."""
    name = "Belady-Greedy (offline)"
    online = False

    def __init__(self, A):
        # suffix sums: F[t, j] = sum_{s >= t} A[s, j]
        self.F = np.flip(np.cumsum(np.flip(A, 0), 0), 0)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        past = cache[:-1]
        return [min(past, key=lambda j: self.F[t, j])]


class ScheduledPolicy(Policy):
    """Replays a precomputed schedule: token j is kept during steps j..end[j]."""
    online = False

    def __init__(self, end, name="Optimal (offline)"):
        self.end, self.name = np.asarray(end), name

    def select_evictions(self, t, cache):
        return [j for j in cache if j != t and self.end[j] < t]


def optimal_schedule(A, k, scale=1_000_000):
    """EXACT offline optimum via a reduction to min-cost flow.

    Nodes  : time nodes 0..T ("a slot is free at the start of step t")
             + one node per token j ("slot holding token j").
    Arcs   : t -> t+1          cap k  cost 0          (slot stays empty)
             tok_j -> e+1      cap 1  cost C - V(j,e)  (token j kept during j..e)
             where V(j,e) = sum_{s=j}^{e} A[s, j].
    Supply : +k at node 0, -k at node T; node j demands 1 and tok_j supplies 1
             (lower-bound trick forcing every new token to enter the cache).
    Every token uses exactly one exit arc, so the constant C shifts the cost by
    T*C and min-cost  <=>  max retained attention. The cut between times t and t+1
    carries exactly k units, so at most k tokens are cached at every step.
    The constraint matrix of a network flow is totally unimodular, so the
    optimum is integral => it is a valid eviction schedule.
    Arcs: O(T^2).  Returns (end, total_retained_attention, solve_seconds)."""
    from ortools.graph.python import min_cost_flow
    T = A.shape[0]
    # V[j, e] for e >= j : cumulative sum down column j
    colcum = np.cumsum(A, axis=0)                       # colcum[e, j] = sum_{s<=e} A[s, j]
    C = int(scale * (T + 1))

    tails, heads, caps, costs, meta = [], [], [], [], []
    for t in range(T):                                  # idle arcs
        tails.append(t); heads.append(t + 1); caps.append(k); costs.append(0); meta.append(None)
    for j in range(T):
        tok = T + 1 + j
        before = colcum[j - 1, j] if j > 0 else 0.0     # = 0 since A[s,j]=0 for s<j
        for e in range(j, T):
            v = colcum[e, j] - before
            tails.append(tok); heads.append(e + 1); caps.append(1)
            costs.append(C - int(round(scale * v))); meta.append((j, e))

    smcf = min_cost_flow.SimpleMinCostFlow()
    t0 = time.perf_counter()
    arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        np.array(tails), np.array(heads), np.array(caps), np.array(costs))
    supplies = np.zeros(2 * T + 1, dtype=np.int64)
    supplies[0] += k
    supplies[T] -= k
    for j in range(T):
        supplies[j] -= 1
        supplies[T + 1 + j] += 1
    smcf.set_nodes_supplies(np.arange(2 * T + 1), supplies)
    status = smcf.solve()
    secs = time.perf_counter() - t0
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow failed: status {status}")

    flows = smcf.flows(arcs)
    end = np.full(T, -1)
    for a in np.nonzero(flows)[0]:
        if meta[a] is not None:
            j, e = meta[a]
            end[j] = e
    assert (end >= np.arange(T)).all(), "every token must be scheduled"
    total = sum(colcum[end[j], j] for j in range(T))
    return end, float(total), secs


# ---------------------------- simulator ----------------------------

def simulate(A, k, policy):
    """Run a policy on attention matrix A with cache size k.
    Returns per-step retained attention (length T) and mean decision time (s)."""
    T = A.shape[0]
    policy.reset(T, k)
    cache, mass = [], np.zeros(T)
    t_total = 0.0
    for t in range(T):
        cache.append(t)
        t0 = time.perf_counter()
        ev = policy.select_evictions(t, cache)
        t_total += time.perf_counter() - t0
        if ev:
            evs = set(ev)
            assert t not in evs, "cannot evict the current token"
            cache = [j for j in cache if j not in evs]
        assert len(cache) <= k, f"{policy.name} exceeded capacity at step {t}"
        row = A[t, cache]
        mass[t] = row.sum()
        t0 = time.perf_counter()
        policy.observe(t, cache, row / max(row.sum(), 1e-12))
        t_total += time.perf_counter() - t0
    return mass, t_total / T


def default_policies():
    return [RandomPolicy(), SlidingWindow(), StreamingLLM(), HeavyHitter(0.0, "HeavyHitter-only"),
            HeavyHitter(0.5, "H2O"), Hybrid()]


def all_policies(A, k):
    """Online policies + the two offline ones (needs A, computes the optimum)."""
    end, total, secs = optimal_schedule(A, k)
    return default_policies() + [BeladyGreedy(A), ScheduledPolicy(end)], secs


def synthetic_attention(T, seed=0, sinks=2, n_heavy=8):
    """Random attention with realistic structure: sink tokens, recency bias,
    a few 'heavy' tokens that get revisited. For testing only."""
    rng = np.random.default_rng(seed)
    heavy = rng.choice(np.arange(sinks, T), size=min(n_heavy, T - sinks), replace=False)
    A = np.zeros((T, T))
    for t in range(T):
        j = np.arange(t + 1)
        logits = -0.15 * (t - j) + rng.normal(0, 1.0, t + 1)
        logits[j < sinks] += 3.0
        logits[np.isin(j, heavy)] += 2.0
        w = np.exp(logits - logits.max())
        A[t, :t + 1] = w / w.sum()
    return A