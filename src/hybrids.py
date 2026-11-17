# ============================================================
# Team Hybrid Policies:
#   - Krishna: HitLRU, AdaptiveWindow (paging-style hybrids)
#   - Kaushal: HARP (Hybrid Adaptive Retention Policy)
# ============================================================
import numpy as np
from src.core import Policy


# ---------------------------------------------------------------- Krishna: paging-style hybrids
class HitLRU(Policy):
    """Sinks + recent window + attention-LRU: token j is 'hit' at step t if attn >= theta/|cache|.
    Evict the token whose last hit is oldest (+ lam*log(1+hits) bonus). O(k)."""
    uses_attention = True

    def __init__(self, sinks=4, recent_frac=0.2, theta=2.0, lam=0.0, name="HitLRU"):
        self.sinks, self.recent_frac, self.theta, self.lam, self.name = sinks, recent_frac, theta, lam, name

    def reset(self, T, k):
        super().reset(T, k)
        self.last = np.arange(T, dtype=float)
        self.hits = np.zeros(T)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        r = max(1, int(self.recent_frac * self.k))
        cand = [j for j in cache[:len(cache) - r] if j >= self.sinks]
        if not cand:
            cand = [j for j in cache[:-1] if j >= self.sinks] or cache[:-1]
        c = np.asarray(cand)
        pr = self.last[c] + self.lam * np.log1p(self.hits[c])
        return [int(c[np.argmin(pr)])]

    def observe(self, t, cache, attn):
        i = np.asarray(cache)
        h = attn >= self.theta / len(i)
        self.last[i[h]] = t
        self.hits[i[h]] += 1


class AdaptiveWindow(Policy):
    """HitLRU whose recent window r adapts: r = q-quantile of an EMA histogram of attention distance
    (clipped to [lo*k, hi*k]). Note: distances are observed only for cached tokens (feedback bias). O(k + T)."""
    uses_attention = True

    def __init__(self, sinks=4, q=0.9, theta=2.0, beta=0.98, lo=0.05, hi=0.6, name="AdaptiveWindow"):
        self.sinks, self.q, self.theta, self.beta, self.lo, self.hi, self.name = sinks, q, theta, beta, lo, hi, name

    def reset(self, T, k):
        super().reset(T, k)
        self.last = np.arange(T, dtype=float)
        self.hist = np.zeros(T)
        self.r = max(1, int(0.25 * k))

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        cand = [j for j in cache[:len(cache) - self.r] if j >= self.sinks]
        if not cand:
            cand = [j for j in cache[:-1] if j >= self.sinks] or cache[:-1]
        c = np.asarray(cand)
        return [int(c[np.argmin(self.last[c])])]

    def observe(self, t, cache, attn):
        i = np.asarray(cache)
        d = t - i
        ns = i >= self.sinks
        self.hist *= self.beta
        np.add.at(self.hist, d[ns], attn[ns])
        tot = self.hist.sum()
        if tot > 0:
            cdf = np.cumsum(self.hist) / tot
            dq = int(np.searchsorted(cdf, self.q)) + 1
            self.r = max(1, int(np.clip(dq, self.lo * self.k, self.hi * self.k)))
        h = attn >= self.theta / len(i)
        self.last[i[h]] = t


# ---------------------------------------------------------------- Kaushal: HARP
class HARP(Policy):
    """HARP: Hybrid Adaptive Retention Policy. Past tokens compete on four causal signals
    (cumulative attention, EWMA recent attention, attention trend = EWMA change, peak attention).
    Each signal becomes a within-candidate percentile rank (ties averaged); evict the lowest weighted rank.
    recent_frac / sinks: protect the newest recent_frac*k tokens and the first `sinks` tokens."""
    name = "HARP"
    uses_attention = True

    def __init__(self, decay=0.90, weights=(0.35, 0.35, 0.20, 0.10), recent_frac=0.25, sinks=4, name="HARP"):
        self.decay, self.name, self.recent_frac, self.sinks = float(decay), name, recent_frac, sinks
        self.weights = np.asarray(weights, dtype=np.float64)
        if not np.isclose(self.weights.sum(), 1.0) or np.any(self.weights < 0):
            raise ValueError("HARP weights must be non-negative and sum to 1")

    def reset(self, T, k):
        super().reset(T, k)
        self.long_score = np.zeros(T)
        self.ewma_score = np.zeros(T)
        self.previous_ewma = np.zeros(T)
        self.peak_score = np.zeros(T)

    @staticmethod
    def _percentile_rank(values):
        from scipy.stats import rankdata
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        if n <= 1:
            return np.zeros(n)
        return (rankdata(values, method="average") - 1) / (n - 1)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        if self.recent_frac > 0 or self.sinks > 0:
            r = max(1, int(self.recent_frac * self.k))
            cand = (
                [j for j in cache[:len(cache) - r] if j >= self.sinks]
                or [j for j in cache[:-1] if j >= self.sinks]
                or cache[:-1]
            )
        else:
            cand = cache[:-1]
        sig = [self.long_score, self.ewma_score, self.ewma_score - self.previous_ewma, self.peak_score]
        combined = self.weights @ np.vstack([self._percentile_rank(x[cand]) for x in sig])
        return [cand[int(np.argmin(combined))]]

    def observe(self, t, cache, attn):
        ids = np.asarray(cache, dtype=int)
        v = np.asarray(attn, dtype=np.float64)
        self.previous_ewma[ids] = self.ewma_score[ids]
        self.ewma_score[ids] = self.decay * self.ewma_score[ids] + (1.0 - self.decay) * v
        self.long_score[ids] += v
        self.peak_score[ids] = np.maximum(self.peak_score[ids], v)
