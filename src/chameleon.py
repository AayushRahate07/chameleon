# ============================================================
# CHAMELEON: an online-learning hybrid eviction policy.
#
#   Idea: no single signal (recency, cumulative attention, recent attention,
#   re-read frequency) is best on every kind of text. Instead of hand-picking
#   the mix (H2O = 50/50, Hybrid = 25/..), Chameleon keeps a POOL OF EXPERTS,
#   each expert being one eviction rule, and re-weights them every step with
#   Fixed-Share Hedge, using only attention it has already observed.
#   Decision = evict the token with the lowest weighted-average rank.
#
#   No offline training, no teacher, no leakage: it adapts to each sequence
#   (and to a change of text type in the middle of a sequence).
#
#   Guarantee (see regret_bound): for the weights the algorithm plays,
#       sum_t <w_t, loss_t>  <=  min_i sum_t loss_{i,t} + ln(N)/eta + eta*T/8 + T*ln(1/(1-alpha))/eta
#   for ANY loss sequence in [0,1] (also when the losses depend on our past
#   decisions). The loss is the surrogate defined in _step_losses, NOT the
#   competitive ratio; the guarantee is about the weights, the rank-averaging
#   that turns weights into an eviction is a heuristic.
# ============================================================
import numpy as np
from src.core import Policy

EXPERTS = ["recency", "cumulative", "last-step", "slow-EMA", "hit-rate"]


def _pct(x):
    """Percentile rank in [0,1] of each entry (ties broken by position: older = lower)."""
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


class LastAttention(Policy):
    """TOVA-style baseline (Oren et al. 2024): evict the token that got the LEAST attention
    at the most recent step. Memory-less: O(k)."""
    name = "LastAttention"
    uses_attention = True

    def reset(self, T, k):
        super().reset(T, k); self.last = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k: return []
        cand = cache[:-1]
        return [min(cand, key=lambda j: self.last[j])]

    def observe(self, t, cache, attn):
        self.last[np.asarray(cache)] = attn


class Chameleon(Policy):
    """Fixed-Share-Hedge mixture of five eviction experts. O(N*k log k) per step, N = 5.

    eta_scale : eta = eta_scale * sqrt(8 ln N / T)   (1.0 = textbook Hedge tuning)
    share     : fixed-share rate alpha = share / T   (lets the mix switch when the text changes)
    sinks     : first `sinks` tokens are never evicted (attention sinks, as in StreamingLLM)
    grace     : the newest max(2, grace*k) tokens are never evicted (new tokens have no history yet)
    q_frac    : the loss looks at the q = q_frac*k tokens an expert would evict first
    lag_frac  : experts are scored with their ranking from `lag_frac*k` steps ago against the attention gained since
                then (0 = nowcast: ranking from the previous step vs the attention of this step)
    learn     : False -> weights stay uniform (ablation: 'equal-weight ensemble')
    sharp     : decision weights = w**sharp (renormalised): 1 = weighted average of ranks, large = follow the leader
    experts   : indices into EXPERTS to use (default all five); for leave-one-expert-out ablations
    """
    uses_attention = True

    def __init__(self, eta_scale=1.0, share=5.0, sinks=4, grace=1 / 16, q_frac=0.25, lag_frac=0.0,
                 learn=True, experts=None, sharp=1.0, name="Chameleon"):
        self.eta_scale, self.share, self.sinks, self.grace = eta_scale, share, sinks, grace
        self.q_frac, self.lag_frac, self.learn, self.sharp, self.name = q_frac, lag_frac, learn, sharp, name
        self.sel = list(range(len(EXPERTS))) if experts is None else list(experts)
        self.N = len(self.sel)

    # ------------------------------------------------------------------
    def reset(self, T, k):
        super().reset(T, k)
        self.cum = np.zeros(T); self.ef = np.zeros(T); self.es = np.zeros(T); self.hits = np.zeros(T)
        self.eta = self.eta_scale * np.sqrt(8 * np.log(self.N) / T)
        self.alpha = min(0.5, self.share / T)
        self.w = np.full(self.N, 1.0 / self.N)
        self.q = max(1, int(self.q_frac * k))
        self.lag = int(round(self.lag_frac * k))
        self.snap = {}                      # step -> state snapshot (only when lag > 0)
        self.w_hist, self.loss_hist, self.mix_loss = [], [], []

    def _scores(self, state, t, c):
        cum, ef, es, hits = state
        age = np.maximum(t - c, 1)
        allsc = [c.astype(float), cum[c], ef[c], es[c], hits[c] / age]
        return [allsc[i] for i in self.sel]

    def _state(self):
        return (self.cum, self.ef, self.es, self.hits)

    # ------------------------------------------------------------------
    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        g = max(2, int(self.grace * self.k))
        pool = cache[:len(cache) - g] if len(cache) - g > 0 else cache[:-1]
        cand = [j for j in pool if j >= self.sinks] or [j for j in cache[:-1] if j >= self.sinks] or cache[:-1]
        c = np.asarray(cand)
        wd = self.w ** self.sharp; wd = wd / wd.sum()
        agg = sum(wi * _pct(s) for wi, s in zip(wd, self._scores(self._state(), t, c)))
        return [cand[int(np.argmin(agg))]]

    # ------------------------------------------------------------------
    def _step_losses(self, t, cache, attn):
        """Surrogate loss of every expert at step t, in [0,1].
        Take the tokens the expert would evict FIRST (its q lowest-ranked among the eligible
        ones); loss = attention they just received, rescaled so that 0 = the best possible
        eviction set for this step and 1 = the worst possible. Needs only attention on cached tokens."""
        lag = self.lag
        old = self._state() if lag == 0 else self.snap.get(t - lag)
        if old is None:
            return None
        c = np.asarray(cache); a = np.asarray(attn)
        ok = (c >= self.sinks) & (c <= t - 1 - lag)           # eligible: seen at least once by the snapshot
        if ok.sum() < self.q + 2:
            return None
        c, a = c[ok], a[ok]
        if lag > 0:                                           # target = attention each token earned over the last `lag` steps
            a = self.cum[c] - old[0][c]                       # (cum before this step's update minus cum at the snapshot)
        q = self.q; s = np.sort(a)
        lo, hi = s[:q].sum(), s[-q:].sum()
        if hi - lo < 1e-12:
            return None
        loss = np.empty(self.N)
        for i, sc in enumerate(self._scores(old, t - lag, c)):
            bottom = np.argsort(_pct(sc), kind="stable")[:q]
            loss[i] = (a[bottom].sum() - lo) / (hi - lo)
        return np.clip(loss, 0.0, 1.0)

    def observe(self, t, cache, attn):
        c = np.asarray(cache); a = np.asarray(attn)
        if self.lag > 0:
            self.snap[t] = tuple(x.copy() for x in self._state()); self.snap.pop(t - self.lag - 1, None)
        loss = self._step_losses(t, cache, attn) if (self.learn and len(cache) >= self.k) else None
        if loss is not None:
            self.mix_loss.append(float(self.w @ loss)); self.loss_hist.append(loss)
            v = self.w * np.exp(-self.eta * loss); v /= v.sum()
            self.w = (1 - self.alpha) * v + self.alpha / self.N
        self.w_hist.append(self.w.copy())
        self.cum[c] += a
        self.ef[c] = a                                   # last-step attention (TOVA signal)
        self.es[c] = 0.9 * self.es[c] + 0.1 * a
        self.hits[c] += (a * len(c) >= 1.0)

    # ------------------------------------------------------------------
    def regret_bound(self):
        """Fixed-Share-Hedge bound against the best FIXED expert (see header)."""
        n = len(self.loss_hist)
        return (np.log(self.N) / self.eta + self.eta * n / 8 + n * np.log(1 / (1 - self.alpha)) / self.eta)

    def regret(self):
        """Realised regret of the played weights vs the best fixed expert, on the realised losses."""
        L = np.sum(self.loss_hist, axis=0)
        return float(np.sum(self.mix_loss) - L.min()), int(np.argmin(L))
