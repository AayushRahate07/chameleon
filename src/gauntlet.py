# ============================================================
# GAUNTLET: one test battery that EVERY eviction policy goes through, identically.
#
#   A policy is wrapped in a Spec. spec.build(train_items) returns a zero-argument factory of
#   fresh policy instances. Hand-designed policies ignore train_items; learned ones (Imitation,
#   DAgger) are trained ONLY on train_items, which the harness builds leakage-free:
#       test text X  ->  training pool = real sequences of the OTHER text types.
#   Nothing in the battery mentions a specific policy, so adding a new hybrid = adding one Spec.
#
#   Pre-registered rule (decided before any run, see verdict()):
#     primary metric = competitive ratio on UNSEEN budgets, mean of per-text means, TEST sequences only
#     a policy "wins" only if the paired bootstrap CI of (its ratio - runner-up's ratio) excludes 0;
#     otherwise it is reported as a TIE. Worst-text ratio and worst-case adversarial ratio are
#     reported next to it so an average cannot hide a collapse.
#   Dev/test split: sequences with even index = DEV (tuning only), odd index = TEST (reporting only).
# ============================================================
import time, tracemalloc, itertools
import numpy as np, pandas as pd
from scipy import stats
from src.core import (simulate, ScheduledPolicy, optimal_schedule, synthetic_attention)


# ---------------------------------------------------------------- optimum (ortools, else LP fallback)
def get_optimum(A, k):
    """Exact offline optimum schedule end[j]. Uses ortools min-cost flow; falls back to the same
    network solved as an LP with HiGHS (verified equal to brute force on small cases)."""
    try:
        return optimal_schedule(A, k)[0]
    except ImportError:
        return _lp_optimum(A, k)


def _lp_optimum(A, k):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    T = A.shape[0]; col = np.cumsum(A, axis=0); C = float(T + 1)
    tails, heads, cost, cap, meta = list(range(T)), list(range(1, T + 1)), [0.0] * T, [k] * T, [None] * T
    for j in range(T):
        for e in range(j, T):
            tails.append(T + 1 + j); heads.append(e + 1); cost.append(C - col[e, j]); cap.append(1); meta.append((j, e))
    na = len(tails)
    B = coo_matrix((np.r_[np.ones(na), -np.ones(na)], (np.r_[tails, heads], np.r_[np.arange(na), np.arange(na)])),
                   shape=(2 * T + 1, na)).tocsr()
    sup = np.zeros(2 * T + 1); sup[0] += k; sup[T] -= k; sup[:T] -= 1; sup[T + 1:] += 1
    r = linprog(cost, A_eq=B, b_eq=sup, bounds=list(zip([0] * na, cap)), method="highs-ds")
    assert r.status == 0, r.message
    end = np.full(T, -1)
    for a in np.nonzero(np.round(r.x).astype(int))[0]:
        if meta[a] is not None: end[meta[a][0]] = meta[a][1]
    assert (end >= np.arange(T)).all(); return end


# ---------------------------------------------------------------- specs
class Spec:
    def __init__(self, name, make=None, train=None, learned=False):
        """make(): fresh policy.   train(items): returns a make() trained on items (learned policies)."""
        self.name, self._make, self._train, self.learned = name, make, train, learned

    def build(self, items):
        if self.learned:
            return self._train(items)
        return self._make


def default_specs(dagger_rounds=3, include_learned=True):
    from src.core import SlidingWindow, StreamingLLM, HeavyHitter, Hybrid
    from src.chameleon import Chameleon, LastAttention
    S = [Spec("SlidingWindow", SlidingWindow), Spec("StreamingLLM", StreamingLLM),
         Spec("H2O", lambda: HeavyHitter(0.5, "H2O")), Spec("Hybrid", Hybrid),
         Spec("LastAttention", LastAttention), Spec("Chameleon", Chameleon)]
    if include_learned:
        from src.imitation import train_imitation, dagger_train
        S.append(Spec("Imitation", learned=True, train=lambda it: (lambda m: (lambda: m))(train_imitation(it, name="Imitation"))))
        S.append(Spec("DAgger", learned=True, train=lambda it: (lambda m: (lambda: m))(dagger_train(it, rounds=dagger_rounds, name="DAgger")[-1])))
    return S


def chameleon_ablations():
    from src.chameleon import Chameleon, EXPERTS
    S = [Spec("Chameleon", Chameleon),
         Spec("Equal weights (no learning)", lambda: Chameleon(learn=False, name="Equal weights (no learning)")),
         Spec("No fixed-share", lambda: Chameleon(share=0.0, name="No fixed-share")),
         Spec("No sink protection", lambda: Chameleon(sinks=0, name="No sink protection")),
         Spec("No grace window", lambda: Chameleon(grace=0.0, name="No grace window")),
         Spec("Fast eta (x3)", lambda: Chameleon(eta_scale=3.0, name="Fast eta (x3)")),
         Spec("Follow-the-leader (sharp=8)", lambda: Chameleon(sharp=8.0, name="Follow-the-leader (sharp=8)")),
         Spec("Lagged loss (k/8)", lambda: Chameleon(lag_frac=0.125, name="Lagged loss (k/8)"))]
    for i, e in enumerate(EXPERTS):
        keep = [j for j in range(len(EXPERTS)) if j != i]
        S.append(Spec(f"Without {e}", (lambda keep=keep, e=e: Chameleon(experts=keep, name=f"Without {e}"))))
    return S


# ---------------------------------------------------------------- data bookkeeping
def split_of(seq):                      # even = dev, odd = test
    return "dev" if seq % 2 == 0 else "test"


def build_optima(DATA, budgets, verbose=True):
    """DATA[(text, seq)] = attention matrix A.  Returns OPT[(text, seq, b)] = schedule and OPTM = its score."""
    OPT, OPTM = {}, {}; t0 = time.time()
    for (n, s), A in DATA.items():
        for b in budgets:
            k = int(b * A.shape[0]); end = get_optimum(A, k); OPT[(n, s, b)] = end
            OPTM[(n, s, b)] = simulate(A, k, ScheduledPolicy(end))[0][k:].mean()
    if verbose: print(f"{len(OPT)} optimal schedules in {time.time() - t0:.0f}s")
    return OPT, OPTM


def pool_names_for(KIND, test_text):
    """Which text types a model tested on `test_text` may train on: real PURE texts other than test_text."""
    return tuple(sorted(n for n in KIND if KIND[n] == "pure" and n != test_text))


def train_pool(DATA, OPT, KIND, test_text, train_budgets):
    """Leakage-free training items (A, k, end) for a model that will be tested on `test_text`.
    Shift / long-context texts are never used for training."""
    names = set(pool_names_for(KIND, test_text))
    return [(A, int(b * A.shape[0]), OPT[(n, s, b)]) for (n, s), A in DATA.items()
            if n in names for b in train_budgets if (n, s, b) in OPT]


# ---------------------------------------------------------------- G1: scoreboard data
def run_ratio(DATA, OPT, OPTM, KIND, specs, budgets, train_budgets, only_split=None, pool=None):
    """pool = (DATA, OPT, KIND) used for TRAINING learned policies (default: the same dicts). After the call,
    run_ratio.cache[pool_names] holds the trained factories so later tests reuse the very same models."""
    pD, pO, pK = pool or (DATA, OPT, KIND)
    rows = []; unseen = lambda b: b not in train_budgets
    cache = {}                                                   # train each learned policy once per distinct training pool
    for text in dict.fromkeys(n for n, _ in DATA):
        pn = pool_names_for(pK, text)
        if pn not in cache:
            items = train_pool(pD, pO, pK, text, train_budgets)
            cache[pn] = {sp.name: sp.build(items) for sp in specs}
        made = cache[pn]
        for (n, s), A in DATA.items():
            if n != text or (only_split and split_of(s) != only_split): continue
            T = A.shape[0]
            for b in budgets:
                k = int(b * T)
                for sp in specs:
                    mass, dt = simulate(A, k, made[sp.name]())
                    rows.append(dict(text=text, kind=KIND[text], seq=s, split=split_of(s), budget=b, unseen=unseen(b),
                                     policy=sp.name, ratio=mass[k:].mean() / OPTM[(n, s, b)], us_per_step=dt * 1e6))
    run_ratio.cache = cache
    return pd.DataFrame(rows)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    return np.percentile(r.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def leaderboard(df, order=None):
    """One row per policy: mean ratio [CI], mean of per-text means, worst text, 5th percentile over settings,
    mean rank (1 = best) over (text, seq, budget) settings, and cost."""
    d = df.copy()
    d["rank"] = d.groupby(["text", "seq", "budget"])["ratio"].rank(ascending=False, method="average")
    per_seq = d.groupby(["policy", "text", "seq"])["ratio"].mean()
    out = {}
    for p, g in d.groupby("policy"):
        text_means = g.groupby("text")["ratio"].mean()
        ci = boot_ci(per_seq.loc[p].values)
        out[p] = {"mean ratio": g.groupby("text")["ratio"].mean().mean(), "CI lo": ci[0], "CI hi": ci[1],
                  "worst text": text_means.min(), "worst-text name": text_means.idxmin(),
                  "p5 over settings": np.percentile(g["ratio"], 5), "mean rank": g["rank"].mean(),
                  "us/step": g["us_per_step"].mean()}
    t = pd.DataFrame(out).T
    for c in ["mean ratio", "CI lo", "CI hi", "worst text", "p5 over settings", "mean rank", "us/step"]:
        t[c] = t[c].astype(float)
    return t.sort_values("mean ratio", ascending=False)


def pairwise(df):
    """win-rate matrix: entry [row, col] = share of settings where row beats col (ties count half)."""
    piv = df.pivot_table(index=["text", "seq", "budget"], columns="policy", values="ratio")
    P = list(piv.columns); W = pd.DataFrame(0.0, index=P, columns=P)
    for a, b in itertools.permutations(P, 2):
        W.loc[a, b] = ((piv[a] > piv[b]).mean() + 0.5 * (piv[a] == piv[b]).mean())
    return W


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def vs_leader(df, leader):
    """Paired comparison of every policy against `leader`, per-sequence means, Wilcoxon + Holm correction."""
    ps = df.groupby(["policy", "text", "seq"])["ratio"].mean().unstack("policy")
    rows = []
    for p in ps.columns:
        if p == leader: continue
        d = (ps[leader] - ps[p]).values
        try: pv = stats.wilcoxon(d).pvalue
        except ValueError: pv = np.nan
        ci = boot_ci(d)
        rows.append(dict(policy=p, leader_minus_policy=d.mean(), ci_lo=ci[0], ci_hi=ci[1], p_wilcoxon=pv, n_seq=len(d)))
    r = pd.DataFrame(rows)
    if len(r): r["p_holm"] = holm(r["p_wilcoxon"].fillna(1.0).values)
    return r


def verdict(df, df_adv=None, min_abs_gain=0.0):
    """Apply the pre-registered rule. Returns a printable string."""
    d = df[(df.split == "test") & (df.kind == "pure") & df.unseen]
    if d.empty: d = df[(df.split == "test")]
    t = leaderboard(d); lead, runner = t.index[0], t.index[1]
    ps = d.groupby(["policy", "text", "seq"])["ratio"].mean().unstack("policy")
    diff = (ps[lead] - ps[runner]).values; ci = boot_ci(diff)
    clear = ci[0] > min_abs_gain
    s = [f"PRIMARY (unseen budgets, pure text, TEST sequences): {lead} {t.loc[lead,'mean ratio']:.3f} vs {runner} {t.loc[runner,'mean ratio']:.3f}",
         f"  paired difference {diff.mean():+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] -> " + ("CLEAR WIN" if clear else "TIE (CI contains 0)")]
    w = t["worst text"].idxmax(); s.append(f"  most robust (best worst-text ratio): {w} ({t.loc[w,'worst text']:.3f})")
    if df_adv is not None and len(df_adv):
        a = df_adv.groupby("policy")["worst_ratio"].min(); s.append(f"  adversarial worst case: best = {a.idxmax()} ({a.max():.3f}); lowest = {a.idxmin()} ({a.min():.3f})")
    return "\n".join(s)


# ---------------------------------------------------------------- G2: adversary (empirical worst-case competitive ratio)
def _softmax_rows(L):
    T = L.shape[0]; A = np.zeros((T, T))
    for t in range(T):
        z = L[t, :t + 1]; w = np.exp(z - z.max()); A[t, :t + 1] = w / w.sum()
    return A


def _ratio(A, k, make):
    end = get_optimum(A, k); om = simulate(A, k, ScheduledPolicy(end))[0][k:].mean()
    return simulate(A, k, make())[0][k:].mean() / om


def adversary(make, T=48, k=6, iters=150, restarts=2, seed=0):
    """Hill-climb over attention matrices to FIND an input where the policy does as badly as possible vs the
    optimum. The best ratio found is an empirical UPPER bound on the policy's true competitive ratio
    (the real worst case can only be lower). Mutations: re-noise some rows, add/remove a 'popular' column
    over a time window, shift a sink."""
    rng = np.random.default_rng(seed); best_all = (2.0, None)
    for r in range(restarts):
        A0 = synthetic_attention(T, int(rng.integers(1_000_000)), sinks=int(rng.integers(0, 3)), n_heavy=int(rng.integers(2, 8)))
        L = np.log(A0 + 1e-9); cur = _ratio(_softmax_rows(L), k, make)
        for it in range(iters):
            L2 = L.copy(); kind = rng.integers(3)
            if kind == 0:                                   # re-noise a block of rows
                a = int(rng.integers(0, T - 4)); L2[a:a + 4] += rng.normal(0, 1.5, (4, T))
            elif kind == 1:                                 # popular token during a time window
                j = int(rng.integers(0, T - 8)); a = int(rng.integers(j, T - 2)); b = min(T, a + int(rng.integers(2, 16)))
                L2[a:b, j] += rng.choice([-3.0, 3.0]) * rng.random() * 2
            else:                                           # random row perturbation
                t_ = int(rng.integers(1, T)); L2[t_, :t_ + 1] += rng.normal(0, 2.0, t_ + 1)
            val = _ratio(_softmax_rows(L2), k, make)
            if val < cur: L, cur = L2, val
        if cur < best_all[0]: best_all = (cur, _softmax_rows(L))
    return best_all


def run_adversary(specs, train_pool_items, T=48, k=6, iters=150, restarts=2, seed=0):
    rows = []
    for sp in specs:
        make = sp.build(train_pool_items); w, A = adversary(make, T, k, iters, restarts, seed)
        rows.append(dict(policy=sp.name, worst_ratio=w)); print(f"  adversary vs {sp.name:28s} worst ratio found {w:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- G3: stress traces (generic version of the notebook test)
def run_stress(specs, stress_fn, T, train_pool_items, budgets=(0.1, 0.25), seeds=(0, 1, 2)):
    made = {sp.name: sp.build(train_pool_items) for sp in specs}; rows = []
    for seed in seeds:
        for tname, A in stress_fn(T, seed).items():
            for b in budgets:
                k = int(b * T); end = get_optimum(A, k); om = simulate(A, k, ScheduledPolicy(end))[0][k:].mean()
                for sp in specs:
                    rows.append(dict(trace=tname, budget=b, seed=seed, policy=sp.name, ratio=simulate(A, k, made[sp.name]())[0][k:].mean() / om))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- G4: does attention-retained rank policies like perplexity does?
def proxy_validity(ratio_by_policy, dppl_by_policy):
    """Kendall tau between the policy ranking by attention retained (higher = better) and by perplexity
    increase (lower = better). +1 = the proxy ranks the policies exactly as real quality does."""
    common = [p for p in ratio_by_policy.index if p in dppl_by_policy.index]
    tau, pv = stats.kendalltau(ratio_by_policy[common].values, -dppl_by_policy[common].values)
    return tau, pv, common


# ---------------------------------------------------------------- G5: overhead
def overhead(specs, T=512, budget=0.2, seed=0):
    A = synthetic_attention(T, seed); k = int(budget * T); rows = []
    for sp in specs:
        make = sp.build([]) if not sp.learned else None
        if make is None: continue                                   # learned policies: measured inside run_ratio
        tracemalloc.start(); mass, dt = simulate(A, k, make()); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append(dict(policy=sp.name, us_per_step=dt * 1e6, peak_KB=peak / 1024))
    return pd.DataFrame(rows).set_index("policy").round(1)


# ---------------------------------------------------------------- Chameleon-only: regret check
def regret_check(DATA, budgets):
    """Verify the Fixed-Share-Hedge guarantee on every run: realised regret of the played weights vs the
    best fixed expert must be <= regret_bound()."""
    from src.chameleon import Chameleon
    rows = []
    for (n, s), A in DATA.items():
        for b in budgets:
            p = Chameleon(); k = int(b * A.shape[0]); simulate(A, k, p)
            if len(p.loss_hist) < 5: continue
            reg, best = p.regret(); rows.append(dict(text=n, seq=s, budget=b, regret=reg, bound=p.regret_bound(), best_fixed_expert=best,
                                                     holds=reg <= p.regret_bound() + 1e-9))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- correctness self-tests (run before trusting any number)
def check_causal(make, A, k, t0=None, seed=0):
    """An ONLINE policy's decisions up to step t0 must not depend on attention rows after t0.
    We replace the future rows by random ones and require identical retained attention up to t0."""
    T = A.shape[0]; t0 = T // 2 if t0 is None else t0
    A2 = A.copy(); r = np.random.default_rng(seed)
    for t in range(t0 + 1, T):
        w = r.random(t + 1) ** 4; A2[t, :t + 1] = w / w.sum()
    m1 = simulate(A, k, make())[0]; m2 = simulate(A2, k, make())[0]
    return bool(np.allclose(m1[:t0 + 1], m2[:t0 + 1], atol=1e-12))


def _brute(A, k):
    T = A.shape[0]; col = np.cumsum(A, 0); best = [-1.0]
    def rec(j, ends):
        if j == T:
            best[0] = max(best[0], sum(col[ends[i], i] for i in range(T))); return
        for e in range(j, T):
            ends.append(e)
            if all(sum(1 for i in range(j + 1) if i <= s <= ends[i]) <= k for s in range(T)): rec(j + 1, ends)
            ends.pop()
    rec(0, []); return best[0]


def selftest(verbose=True):
    """1) optimum == brute force on small cases   2) optimum >= every policy   3) capacity + causality of every
    online policy   4) Chameleon's regret bound holds   Raises AssertionError on failure."""
    from src.chameleon import Chameleon
    for seed in range(4):
        A = synthetic_attention(7, seed, sinks=1, n_heavy=2); k = 2 + seed % 2
        e = get_optimum(A, k); tot = sum(np.cumsum(A, 0)[e[j], j] for j in range(7))
        assert abs(tot - _brute(A, k)) < 1e-6, f"optimum != brute force (seed {seed})"
    if verbose: print("  [1] optimum matches brute force on 4 small instances")
    A = synthetic_attention(120, 3); k = 24; end = get_optimum(A, k); om = simulate(A, k, ScheduledPolicy(end))[0][k:].sum()
    for sp in default_specs(include_learned=False) + chameleon_ablations():
        make = sp.build([]); m = simulate(A, k, make())[0][k:].sum()          # simulate() asserts capacity <= k
        assert m <= om + 1e-9, f"{sp.name} beats the optimum?!"
        assert check_causal(make, A, k), f"{sp.name} uses future information"
    if verbose: print("  [2,3] optimum dominates, capacity respected, no future leakage: " + str(len(default_specs(include_learned=False) + chameleon_ablations())) + " policies")
    p = Chameleon(); simulate(A, k, p); reg, _ = p.regret()
    assert reg <= p.regret_bound() + 1e-9, "Chameleon regret exceeds its bound"
    if verbose: print(f"  [4] Chameleon regret {reg:.2f} <= bound {p.regret_bound():.2f}")
    print("self-tests passed")
