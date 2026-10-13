"""Numerical checks for the paper on Open Problem 4.48 of Aldous and Fill.

Run:  python checks.py > checks_output.txt      (mpmath, numpy, scipy; a few minutes)

1. The three-state chain of Theorem 1 at eps = 1/100 and 1/1000: tau_0 three ways (closed form,
   eigentime identity, mean hitting times), dbar(t) from the matrix exponential and from the
   spectral decomposition, dbar(t) > 1/e on a grid of [0, 2], tau_1 by bisection, and the three
   probabilistic bounds used in the proof, on the same grid.  Controls: the margin f(eps) is
   negative at eps = 1/20, and the value of eps where tau_1 = tau_0 on this chain is located.
2. The chains on {0, ..., k}: t_k with an interval check, the maximum of t_k/k over k <= 60, the
   limit dbar(t) = P(Gamma(k, 1) > t) of the pure death chain, and tau_1/tau_0 at small eps.
3. Theorem 2 on random reversible chains in continuous and in discrete time: tau_2 <= tau_1^(2)
   and the bound state by state, with tau_1^(2) from (4.47) of the book and, independently, from
   the linear program over occupation measures; mean hitting times from linear systems against the
   fundamental matrix.  Controls: the two-state chain (equality throughout), and the absolute
   relaxation time, which can exceed tau_1^(2) in discrete time.
Every check prints PASS or FAIL; the exit status is 1 if any check fails.
"""
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import linprog

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  [" + detail + "]" if detail else ""))
    if not ok:
        FAILS.append(name)


def fmt(x, d=12):
    return mp.nstr(x, d)


# ---------------------------------------------------------------------------------------------
# Chains on {0, ..., k}: rate 1 from i to i - 1, rate eps from i to i + 1 (mpmath).
# ---------------------------------------------------------------------------------------------

def path_generator(k, eps):
    Q = mp.zeros(k + 1, k + 1)
    for i in range(k + 1):
        if i > 0:
            Q[i, i - 1] = mp.mpf(1)
        if i < k:
            Q[i, i + 1] = mp.mpf(eps)
        Q[i, i] = -sum(Q[i, j] for j in range(k + 1) if j != i)
    return Q


def path_pi(k, eps):
    w = [mp.mpf(eps) ** i for i in range(k + 1)]
    s = sum(w)
    return [x / s for x in w]


class Spectral:
    """P_t = D^{-1/2} V exp(mu t) V^T D^{1/2} for a reversible generator Q with stationary pi."""

    def __init__(self, Q, pi):
        n = Q.rows
        self.n = n
        self.sq = [mp.sqrt(p) for p in pi]
        S = mp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                S[i, j] = self.sq[i] * Q[i, j] / self.sq[j]
        S = (S + S.T) / 2
        self.mu, self.V = mp.eigsy(S)

    def P(self, t):
        n = self.n
        E = [mp.exp(self.mu[m] * t) for m in range(n)]
        out = mp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                out[i, j] = sum(self.V[i, m] * E[m] * self.V[j, m] for m in range(n)) * self.sq[j] / self.sq[i]
        return out

    def rates(self):
        """The nonzero eigenvalues of -Q, increasing."""
        lam = sorted(-self.mu[m] for m in range(self.n))
        return lam[1:]


def dbar_of(P):
    n = P.rows
    best, arg = mp.mpf(0), None
    for i in range(n):
        for j in range(i + 1, n):
            tv = sum(abs(P[i, x] - P[j, x]) for x in range(n)) / 2
            if tv > best:
                best, arg = tv, (i, j)
    return best, arg


def tau1_bisect(dbar, lo, hi, tol=mp.mpf("1e-30")):
    """min{t : dbar(t) <= 1/e} for a continuous nonincreasing dbar with dbar(lo) > 1/e >= dbar(hi)."""
    target = mp.exp(-1)
    assert dbar(lo) > target >= dbar(hi)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if dbar(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hitting_times(Q, j):
    """h_i = E_i T_j from (Q h)_i = -1 for i != j, h_j = 0."""
    n = Q.rows
    idx = [i for i in range(n) if i != j]
    A = mp.matrix(len(idx), len(idx))
    b = mp.matrix(len(idx), 1)
    for a, i in enumerate(idx):
        for c, l in enumerate(idx):
            A[a, c] = Q[i, l]
        b[a] = -1
    x = mp.lu_solve(A, b)
    h = [mp.mpf(0)] * n
    for a, i in enumerate(idx):
        h[i] = x[a]
    return h


def section1():
    print("\n== 1. The three-state chain of Theorem 1 ==")
    mp.mp.dps = 40
    e = mp.e
    for eps in (mp.mpf(1) / 100, mp.mpf(1) / 1000):
        tag = "eps=1/%d" % int(round(1 / eps))
        Q, pi = path_generator(2, eps), path_pi(2, eps)
        sp = Spectral(Q, pi)
        # tau_0 three ways
        t0_closed = 2 * (1 + eps) / (1 + eps + eps ** 2)
        t0_eig = sum(1 / lam for lam in sp.rates())
        H = [hitting_times(Q, j) for j in range(3)]
        t0_from = [sum(pi[j] * H[j][i] for j in range(3)) for i in range(3)]
        t0_avg = sum(pi[i] * pi[j] * H[j][i] for i in range(3) for j in range(3))
        agree = max(abs(t0_eig - t0_closed), abs(t0_avg - t0_closed), max(abs(x - t0_closed) for x in t0_from))
        check("%s: tau_0 closed form = eigentime identity = mean hitting times" % tag, agree < mp.mpf("1e-30"),
              "tau_0 = %s, largest difference %s" % (fmt(t0_closed, 15), mp.nstr(agree, 3)))
        check("%s: tau_0 < 2" % tag, t0_closed < 2)
        # dbar from the matrix exponential and from the spectral decomposition
        worst = mp.mpf(0)
        for t in [mp.mpf(s) / 10 for s in range(0, 41)]:
            P1 = mp.expm(Q * t)
            P2 = sp.P(t)
            worst = max(worst, max(abs(P1[i, j] - P2[i, j]) for i in range(3) for j in range(3)))
        check("%s: expm and spectral P_t agree on t = 0, 0.1, ..., 4" % tag, worst < mp.mpf("1e-25"),
              "largest difference %s" % mp.nstr(worst, 3))
        # dbar > 1/e on a grid of [0, 2]; the three bounds of the proof on the same grid
        Qabs = Q.copy()
        for j in range(3):
            Qabs[0, j] = 0
        grid = [2 * mp.mpf(s) / 2000 for s in range(2001)]
        mind, prev, mono = mp.mpf(1), mp.mpf(2), True
        b1 = b2 = b3 = b4 = True
        lowest_bound = mp.mpf(1)
        for t in grid:
            P = sp.P(t)
            d, _ = dbar_of(P)
            mono = mono and d <= prev + mp.mpf("1e-30")
            prev = d
            mind = min(mind, d)
            g = mp.exp(-(1 + eps) * t) * (1 + (1 + eps) * t)
            hit = mp.expm(Qabs * t)[2, 0] if t in grid[::50] else None
            b1 = b1 and P[0, 0] >= mp.exp(-eps * t) - mp.mpf("1e-30")
            if hit is not None:
                b2 = b2 and P[2, 0] <= hit + mp.mpf("1e-30")
                b3 = b3 and hit <= 1 - g + mp.mpf("1e-30")
            lower = mp.exp(-eps * t) - 1 + g
            b4 = b4 and d >= P[0, 0] - P[2, 0] - mp.mpf("1e-30") and P[0, 0] - P[2, 0] >= lower - mp.mpf("1e-30")
            lowest_bound = min(lowest_bound, lower)
        check("%s: dbar nonincreasing on the grid of [0, 2] (step 0.001)" % tag, mono)
        check("%s: dbar(t) > 1/e on the grid" % tag, mind > mp.exp(-1),
              "min dbar = %s, 1/e = %s" % (fmt(mind, 10), fmt(mp.exp(-1), 10)))
        check("%s: P_0(X_t = 0) >= exp(-eps t) on the grid" % tag, b1)
        check("%s: P_2(X_t = 0) <= P_2(T_0 <= t) <= 1 - g((1+eps) t) (every 50th point)" % tag, b2 and b3)
        check("%s: dbar(t) >= P_0(X_t=0) - P_2(X_t=0) >= exp(-eps t) - 1 + g((1+eps) t)" % tag, b4)
        check("%s: the lower bound exceeds 1/e on the grid" % tag, lowest_bound > mp.exp(-1),
              "min of the bound = %s" % fmt(lowest_bound, 10))
        f = 3 - e - eps * (4 + 2 * e ** 2) - 4 * eps ** 2
        check("%s: f(eps) = 3 - e - eps(4 + 2e^2) - 4eps^2 > 0" % tag, f > 0, "f = %s" % fmt(f, 10))
        d2, arg = dbar_of(sp.P(mp.mpf(2)))
        t1 = tau1_bisect(lambda t: dbar_of(sp.P(t))[0], mp.mpf(2), mp.mpf(10))
        check("%s: tau_1 > 2 > tau_0" % tag, t1 > 2 > t0_closed,
              "dbar(2) = %s at the pair %s, tau_1 = %s, tau_0 = %s, tau_1/tau_0 = %s"
              % (fmt(d2, 6), arg, fmt(t1, 12), fmt(t0_closed, 12), fmt(t1 / t0_closed, 8)))
        t1b = tau1_bisect(lambda t: dbar_of(mp.expm(Q * t))[0], mp.mpf(2), mp.mpf(10), tol=mp.mpf("1e-15"))
        check("%s: tau_1 from expm agrees" % tag, abs(t1b - t1) < mp.mpf("1e-12"), "tau_1 = %s" % fmt(t1b, 12))
    # Controls
    eps = mp.mpf(1) / 20
    f = 3 - e - eps * (4 + 2 * e ** 2) - 4 * eps ** 2
    check("control: f(1/20) < 0, so the hand bound needs small eps", f < 0, "f(1/20) = %s" % fmt(f, 6))

    def ratio(eps):
        Q, pi = path_generator(2, eps), path_pi(2, eps)
        sp = Spectral(Q, pi)
        t0 = sum(1 / lam for lam in sp.rates())
        db = lambda t: dbar_of(sp.P(t))[0]
        hi = mp.mpf(1)
        while db(hi) > mp.exp(-1):
            hi *= 2
        t1 = tau1_bisect(db, mp.mpf(0), hi, tol=mp.mpf("1e-20"))
        return t1 / t0

    mp.mp.dps = 30
    r20 = ratio(mp.mpf(1) / 20)
    check("control: at eps = 1/20 the ratio tau_1/tau_0 still exceeds 1", r20 > 1, "ratio %s" % fmt(r20, 8))
    lo, hi = mp.mpf(1) / 20, mp.mpf(1)
    r_hi = ratio(hi)
    if r_hi < 1:
        for _ in range(40):
            mid = (lo + hi) / 2
            if ratio(mid) > 1:
                lo = mid
            else:
                hi = mid
        print("INFO  three-state chain: tau_1 > tau_0 for eps below about %s; at eps = 1 the ratio is %s"
              % (fmt((lo + hi) / 2, 6), fmt(r_hi, 6)))
    else:
        print("INFO  three-state chain: ratio at eps = 1 is %s" % fmt(r_hi, 6))


# ---------------------------------------------------------------------------------------------
# 2. The chains on {0, ..., k}
# ---------------------------------------------------------------------------------------------

def gamma_tail(k, t):
    return mp.exp(-t) * sum(t ** j / mp.factorial(j) for j in range(k))


def t_k(k):
    f = lambda t: gamma_tail(k, t) - mp.exp(-1)
    lo, hi = mp.mpf(k) / 2, mp.mpf(k) + 3 * mp.sqrt(k) + 3
    assert f(lo) > 0 > f(hi)
    for _ in range(300):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def section2():
    print("\n== 2. The chains on {0, ..., k} ==")
    mp.mp.dps = 60
    tk = {}
    for k in range(1, 61):
        tk[k] = t_k(k)
    for k in range(1, 9):
        t = tk[k]
        lo, hi = t - mp.mpf("1e-40"), t + mp.mpf("1e-40")
        ok = gamma_tail(k, lo) > mp.exp(-1) > gamma_tail(k, hi)
        check("t_%d = %s, t_%d/%d = %s (sign change on an interval of width 2e-40)"
              % (k, fmt(t, 15), k, k, fmt(t / k, 9)), ok)
    kbest = max(range(1, 61), key=lambda k: tk[k] / k)
    check("max of t_k/k over k <= 60 is at k = 4", kbest == 4,
          "t_4/4 = %s, t_60/60 = %s" % (fmt(tk[4] / 4, 12), fmt(tk[60] / 60, 8)))
    check("t_1 = 1 exactly", abs(tk[1] - 1) < mp.mpf("1e-50"))
    # the pure death chain: dbar(t) = P(Gamma(k) > t), attained at the pair (0, k)
    mp.mp.dps = 30
    for k in (2, 3, 4, 6):
        Q0 = path_generator(k, 0)
        worst, pair_ok = mp.mpf(0), True
        for t in [mp.mpf(s) / 4 for s in range(1, 41)]:
            P = mp.expm(Q0 * t)
            d, arg = dbar_of(P)
            worst = max(worst, abs(d - gamma_tail(k, t)))
            tv0k = sum(abs(P[0, x] - P[k, x]) for x in range(k + 1)) / 2
            pair_ok = pair_ok and abs(tv0k - d) < mp.mpf("1e-25")
        check("k = %d, eps = 0: dbar(t) = P(Gamma(%d) > t), attained at (0, %d), t = 0.25..10" % (k, k, k),
              worst < mp.mpf("1e-25") and pair_ok, "largest difference %s" % mp.nstr(worst, 3))
    # tau_1/tau_0 at small eps
    mp.mp.dps = 50
    for k in (2, 3, 4, 6):
        prev = None
        for eps in (mp.mpf("1e-3"), mp.mpf("1e-5"), mp.mpf("1e-7")):
            Q, pi = path_generator(k, eps), path_pi(k, eps)
            sp = Spectral(Q, pi)
            t0 = sum(1 / lam for lam in sp.rates())
            db = lambda t: dbar_of(sp.P(t))[0]
            t1 = tau1_bisect(db, mp.mpf(0), mp.mpf(4 * k + 10), tol=mp.mpf("1e-15"))
            gap = abs(t1 / t0 - tk[k] / k)
            detail = "eps = %s: tau_0 = %s, tau_1 = %s, ratio %s" % (mp.nstr(eps, 1), fmt(t0, 10), fmt(t1, 10),
                                                                     fmt(t1 / t0, 9))
            ok = t1 > t0 and (prev is None or gap < prev)
            check("k = %d: tau_1 > tau_0 and |tau_1/tau_0 - t_k/k| decreasing" % k, ok, detail)
            prev = gap
        check("k = %d: |tau_1/tau_0 - t_k/k| < 1e-5 at eps = 1e-7" % k, prev < mp.mpf("1e-5"),
              "gap %s" % mp.nstr(prev, 3))


# ---------------------------------------------------------------------------------------------
# 3. Theorem 2 on random reversible chains (double precision)
# ---------------------------------------------------------------------------------------------

def random_reversible(rng, n, discrete):
    pi = rng.gamma(0.7, size=n) + 1e-3
    pi /= pi.sum()
    C = rng.exponential(size=(n, n)) * (rng.random((n, n)) < 0.6)
    C = np.triu(C, 1)
    for i in range(n - 1):                      # a path keeps the chain irreducible
        C[i, i + 1] += 0.05 * rng.random() + 1e-3
    C = C + C.T
    if not discrete:
        Q = np.array([[C[i, j] / pi[i] if i != j else 0.0 for j in range(n)] for i in range(n)])
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q, pi
    W = C.copy()
    if rng.random() < 0.5:
        np.fill_diagonal(W, rng.exponential(size=n))
    piw = W.sum(axis=1)
    P = W / piw[:, None]
    return P, piw / piw.sum()


def tau12_formula(Z, pi):
    M = -Z / pi[None, :]
    return M.max(axis=1)


def lp_min_mean(G, pi, i):
    """min sum x subject to x >= 0, x^T G = pi^T - e_i^T (G = Q, or P - I in discrete time)."""
    n = len(pi)
    b = pi.copy()
    b[i] -= 1.0
    res = linprog(np.ones(n), A_eq=G.T, b_eq=b, bounds=[(0, None)] * n, method="highs")
    return res.fun if res.status == 0 else np.nan


def section3():
    print("\n== 3. Theorem 2 on random reversible chains ==")
    rng = np.random.default_rng(20261013)
    for discrete in (False, True):
        kind = "discrete" if discrete else "continuous"
        count, bad_main, bad_state, bad_lp, bad_hit = 0, 0, 0, 0, 0
        min_ratio, abs_exceed = np.inf, 0
        for trial in range(3000):
            n = int(rng.integers(2, 9))
            M, pi = random_reversible(rng, n, discrete)
            G = M - np.eye(n) if discrete else M
            D = np.sqrt(pi)
            S = D[:, None] * G / D[None, :]
            S = (S + S.T) / 2
            mu, V = np.linalg.eigh(S)                      # mu[-1] = 0
            gap = -mu[-2]
            tau2 = 1.0 / gap
            u = V[:, -2] / D
            Pi = np.outer(np.ones(n), pi)
            Z = np.linalg.inv(Pi - G) - Pi
            t12_i = tau12_formula(Z, pi)
            t12 = t12_i.max()
            count += 1
            if not tau2 <= t12 * (1 + 1e-9):
                bad_main += 1
            if np.any(t12_i < tau2 * np.abs(u) / np.abs(u).max() * (1 - 1e-9) - 1e-12):
                bad_state += 1
            min_ratio = min(min_ratio, t12 / tau2)
            if trial < 600:
                for i in range(n):
                    v = lp_min_mean(G, pi, i)
                    if not abs(v - t12_i[i]) <= 1e-7 * max(1.0, abs(v)):
                        bad_lp += 1
                j = int(rng.integers(n))
                idx = [x for x in range(n) if x != j]
                h = np.zeros(n)
                h[idx] = np.linalg.solve(G[np.ix_(idx, idx)], -np.ones(n - 1))
                hz = (Z[j, j] - Z[:, j]) / pi[j]
                if np.abs(h - hz).max() > 1e-7 * max(1.0, np.abs(h).max()):
                    bad_hit += 1
            if discrete:
                beta = 1 + mu                               # eigenvalues of P
                den = 1.0 - max(beta[-2], abs(beta[0]))
                t_abs = np.inf if den <= 1e-15 else 1.0 / den
                if t_abs > t12 * (1 + 1e-9):
                    abs_exceed += 1
        check("%s time: tau_2 <= tau_1^(2) on %d random chains (2 to 8 states)" % (kind, count), bad_main == 0,
              "%d failures; smallest tau_1^(2)/tau_2 = %.6f" % (bad_main, min_ratio))
        check("%s time: min_U E_i U >= tau_2 |u_i| / max|u| at every state" % kind, bad_state == 0,
              "%d chains failed" % bad_state)
        check("%s time: (4.47) = linear program over occupation measures, every state of 600 chains" % kind,
              bad_lp == 0, "%d states disagreed" % bad_lp)
        check("%s time: E_i T_j from linear systems = (Z_jj - Z_ij)/pi_j on 600 chains" % kind, bad_hit == 0)
        if discrete:
            check("control: the absolute relaxation time exceeds tau_1^(2) on some chains", abs_exceed > 0,
                  "%d of %d chains" % (abs_exceed, count))
    # two-state chain: equality
    mp.mp.dps = 30
    worst = mp.mpf(0)
    for a, b in ((1, 1), (2, 0.5), (0.01, 3), (7, 0.2)):
        a, b = mp.mpf(a), mp.mpf(b)
        Q = mp.matrix([[-a, a], [b, -b]])
        pi = [b / (a + b), a / (a + b)]
        Pi = mp.matrix([[pi[0], pi[1]], [pi[0], pi[1]]])
        Z = (Pi - Q) ** -1 - Pi
        t12 = max(-Z[i, j] / pi[j] for i in range(2) for j in range(2))
        t0 = pi[0] * pi[1] * (1 / a + 1 / b)
        P = lambda t: mp.expm(Q * t)
        d = lambda t: abs(P(t)[0, 0] - P(t)[1, 0])
        t1 = tau1_bisect(d, mp.mpf(0), 10 / (a + b), tol=mp.mpf("1e-20"))
        for x in (t12, t0, t1):
            worst = max(worst, abs(x * (a + b) - 1))
    check("two-state chains: tau_2 = tau_1^(2) = tau_0 = tau_1 = 1/(a+b)", worst < mp.mpf("1e-15"),
          "largest relative difference %s" % mp.nstr(worst, 3))
    # near-flip discrete chain: absolute relaxation time 1/(2 delta) against tau_1^(2) = tau_2 = 1/(2(1-delta))
    dlt = 0.01
    P = np.array([[dlt, 1 - dlt], [1 - dlt, dlt]])
    pi = np.array([0.5, 0.5])
    Pi = np.outer(np.ones(2), pi)
    Z = np.linalg.inv(np.eye(2) - P + Pi) - Pi
    t12 = tau12_formula(Z, pi).max()
    check("control: P = [[0.01, 0.99], [0.99, 0.01]]: tau_1^(2) = tau_2 = 1/1.98, absolute relaxation time 50",
          abs(t12 - 1 / 1.98) < 1e-12 and abs(1 / (1 - abs(2 * dlt - 1)) - 50) < 1e-9, "tau_1^(2) = %.12f" % t12)


if __name__ == "__main__":
    print("checks.py: mpmath %s, numpy %s" % (mp.__version__, np.__version__))
    section1()
    section2()
    section3()
    print("\n%d failure(s)" % len(FAILS))
    sys.exit(1 if FAILS else 0)
