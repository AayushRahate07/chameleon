# ============================================================
# Imitation scorer: learn to copy the offline optimum.
#   Teacher : exact min-cost-flow schedule end[j] (uses the FUTURE).
#   Student : logistic regression on features that only use the PAST.
#   Online  : at each eviction, drop the cached token with the lowest
#             predicted "the optimum will still hold this token" probability.
# ============================================================
import numpy as np
from src.core import Policy, ScheduledPolicy, HeavyHitter, Hybrid, SlidingWindow

FEATURES = ["age/k", "recent (age<k/4)", "avg attention", "score vs cache mean",
            "last-step attention", "EMA attention", "is sink (j<4)", "budget k/T"]
EMA_ALPHA = 0.3


class Tracker:
    """Past-only statistics kept per token (updated in observe)."""
    def reset(self, T, k):
        self.T, self.k = T, k
        self.score = np.zeros(T); self.last = np.zeros(T); self.ema = np.zeros(T)

    def update(self, cache, attn):
        c = np.asarray(cache)
        self.score[c] += attn
        self.last[c] = attn
        self.ema[c] = (1 - EMA_ALPHA) * self.ema[c] + EMA_ALPHA * attn

    def features(self, t, cand):
        """Feature matrix for candidate token ids `cand` at step t (uses steps < t only). O(|cand|)."""
        c = np.asarray(cand); k = self.k
        age = t - c
        sc = self.score[c]
        return np.column_stack([
            age / k,
            (age < k / 4).astype(float),
            sc / np.maximum(age, 1) * k,          # average attention per step alive (x k to be scale-free)
            sc / (sc.mean() + 1e-12),             # popularity relative to the rest of the cache
            self.last[c] * k,
            self.ema[c] * k,
            (c < 4).astype(float),
            np.full(len(c), k / self.T),
        ])


class ImitationScorer(Policy):
    """Online policy: evict argmin of learned keep-probability. O(k) per step."""
    uses_attention = True

    def __init__(self, w, b, mu, sd, name="Imitation"):
        self.w, self.b, self.mu, self.sd, self.name = w, b, mu, sd, name
        self.tr = Tracker()

    def reset(self, T, k):
        super().reset(T, k); self.tr.reset(T, k)

    def select_evictions(self, t, cache):
        if len(cache) <= self.k:
            return []
        cand = cache[:-1]                                  # the new token is never evicted
        z = ((self.tr.features(t, cand) - self.mu) / self.sd) @ self.w + self.b
        return [cand[int(np.argmin(z))]]

    def observe(self, t, cache, attn):
        self.tr.update(cache, attn * len(cache))           # scale by cache size so values are comparable across budgets


def collect(A, k, end, base, rng, per_step=16, h_frac=0.5):
    """Roll out `base` policy on A; at every step where an eviction is needed, record
    features of (a sample of) cached tokens, labelled by the optimum:
        y = 1  if the optimum still holds token j at step t + H   (H = h_frac * k, default k/2)
        y = 0  otherwise (includes tokens the optimum has already dropped).
    Rolling out several different policies makes the training states look like the states
    the learned policy will see (reduces train/test distribution shift)."""
    T = A.shape[0]; H = max(1, int(h_frac * k))
    tr = Tracker(); tr.reset(T, k); base.reset(T, k)
    cache, X, Y = [], [], []
    for t in range(T):
        cache.append(t)
        if len(cache) > k:
            cand = np.asarray(cache[:-1])
            pick = cand if len(cand) <= per_step else rng.choice(cand, per_step, replace=False)
            X.append(tr.features(t, pick)); Y.append((end[pick] >= t + H).astype(int))
        ev = set(base.select_evictions(t, cache))
        cache = [j for j in cache if j not in ev]
        row = A[t, cache]
        a = row / max(row.sum(), 1e-12)
        base.observe(t, cache, a)
        tr.update(cache, a * len(cache))
    return np.vstack(X), np.concatenate(Y)


MAX_ROWS = 1_500_000


def fit_student(X, Y, name, seed=0):
    """Fit the logistic-regression scorer on (features, optimum-labels)."""
    from sklearn.linear_model import LogisticRegression
    if len(Y) > MAX_ROWS:                                   # keep refits fast when DAgger data grows
        idx = np.random.default_rng(seed).choice(len(Y), MAX_ROWS, replace=False); X, Y = X[idx], Y[idx]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit((X - mu) / sd, Y)
    pol = ImitationScorer(clf.coef_[0], float(clf.intercept_[0]), mu, sd, name)
    pol.n_train, pol.pos_rate = len(Y), float(Y.mean())
    return pol


def _teacher_data(items, rng, per_step=16, h_frac=0.5):
    Xs, Ys = [], []
    for A, k, end in items:
        for base in [ScheduledPolicy(end), HeavyHitter(0.5, "H2O"), Hybrid(), SlidingWindow()]:
            X, Y = collect(A, k, end, base, rng, per_step, h_frac); Xs.append(X); Ys.append(Y)
    return Xs, Ys


def train_imitation(items, seed=0, name="Imitation", h_frac=0.5):
    """items: list of (A, k, end). Plain imitation: train once on teacher + baseline rollouts."""
    rng = np.random.default_rng(seed)
    Xs, Ys = _teacher_data(items, rng, 16, h_frac)
    return fit_student(np.vstack(Xs), np.concatenate(Ys), name, seed)


def dagger_train(items, rounds=3, seed=0, name="DAgger", h_frac=0.5, per_step=8):
    """DAgger (Ross et al. 2011) with the exact min-cost-flow schedule as the expert.
    Round 0 = plain imitation. Each later round: run the CURRENT student, record the states
    it visits, let the optimum relabel them (labels come from end[j], no new solves),
    add them to the data, refit. Returns the list of students [round 0, 1, ..., rounds]."""
    rng = np.random.default_rng(seed)
    Xs, Ys = _teacher_data(items, rng, per_step, h_frac)
    students = [fit_student(np.vstack(Xs), np.concatenate(Ys), f"{name} r0", seed)]
    for r in range(1, rounds + 1):
        for A, k, end in items:
            X, Y = collect(A, k, end, students[-1], rng, per_step, h_frac); Xs.append(X); Ys.append(Y)
        students.append(fit_student(np.vstack(Xs), np.concatenate(Ys), f"{name} r{r}", seed))
    return students


def state_auc(student, items, own, seed=0, h_frac=0.5):
    """How well does the student rank tokens (optimum would keep > would drop) on
    held-out states?  own=False: states from the optimum's trajectory (what the teacher sees).
    own=True: states the STUDENT itself visits (what it faces at test time).  1.0 = perfect, 0.5 = chance."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); Xs, Ys = [], []
    for A, k, end in items:
        base = student if own else ScheduledPolicy(end)
        X, Y = collect(A, k, end, base, rng, 16, h_frac); Xs.append(X); Ys.append(Y)
    X, Y = np.vstack(Xs), np.concatenate(Ys)
    if len(set(Y.tolist())) < 2: return float("nan")
    return float(roc_auc_score(Y, ((X - student.mu) / student.sd) @ student.w + student.b))
