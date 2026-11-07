"""Example 11: causal discovery. PC, NOTEARS and a sobering baseline.

Author: Abyn Scaria. MIT License.

Ground truth: a 6-node linear-Gaussian SEM with two v-structures
    X0 -> X2 <- X1,  X2 -> X3,  X1 -> X4,  X3 -> X5 <- X4
Edge weights are drawn with |w| in [0.8, 2.0] and random sign, noise N(0, 1).

Methods (all implemented here, numpy/scipy/sklearn only):
  * PC algorithm (Spirtes-Glymour): Fisher-z partial-correlation tests for the
    skeleton, then v-structures, then Meek rules. Outputs a CPDAG (some edges
    undirected, because observational data only identifies the Markov
    equivalence class). Running it with a d-separation oracle gives the true CPDAG.
  * NOTEARS (Zheng et al. 2018): least squares + L1, acyclicity enforced through
    h(W) = tr(exp(W o W)) - d = 0 with an augmented Lagrangian.
  * sortnregress (Reisach et al. 2021): order variables by marginal variance and
    regress each on the earlier ones with a Lasso. It uses no causal reasoning at
    all. It does well on raw simulated data because variance grows along the causal
    order ("varsortability"), and it fails once the data are standardised.
Each continuous method is run on raw and on standardised data.
"""
from common import PALETTE, d_separated, plt, save_json, savefig  # noqa: I001

import itertools

import networkx as nx
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
from scipy import stats
from sklearn.linear_model import LassoLarsIC, LinearRegression

TRUE_EDGES = [(0, 2), (1, 2), (2, 3), (1, 4), (3, 5), (4, 5)]
D = 6


def simulate(n, rng):
    W = np.zeros((D, D))
    for i, j in TRUE_EDGES:
        W[i, j] = rng.uniform(0.8, 2.0) * rng.choice([-1, 1])
    X = np.zeros((n, D))
    for j in nx.topological_sort(nx.DiGraph(TRUE_EDGES)):
        X[:, j] = X @ W[:, j] + rng.normal(size=n)
    return X, W


# ----------------------------------------------------------------------------
# PC algorithm
# ----------------------------------------------------------------------------
def fisher_z_test(C, n, i, j, S):
    """p-value for X_i _||_ X_j | X_S from the correlation matrix C."""
    idx = [i, j] + list(S)
    P = np.linalg.pinv(C[np.ix_(idx, idx)])
    r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - len(S) - 3)
    return 2 * (1 - stats.norm.cdf(abs(z)))


def pc(ci_test, d):
    """ci_test(i, j, S) -> True if independent. Returns adjacency of a CPDAG:
    A[i,j]=1 and A[j,i]=0 means i -> j; both 1 means undirected i - j."""
    adj = {i: set(range(d)) - {i} for i in range(d)}
    sepset = {}
    level = 0
    while any(len(adj[i]) - 1 >= level for i in adj):
        for i, j in itertools.permutations(range(d), 2):
            if j not in adj[i]:
                continue
            for S in itertools.combinations(sorted(adj[i] - {j}), level):
                if ci_test(i, j, S):
                    adj[i].discard(j)
                    adj[j].discard(i)
                    sepset[frozenset((i, j))] = set(S)
                    break
        level += 1
    A = np.zeros((d, d), dtype=int)
    for i in adj:
        for j in adj[i]:
            A[i, j] = 1
    # v-structures: i - k - j, i and j non-adjacent, k not in sepset(i, j)  =>  i -> k <- j
    for k in range(d):
        for i, j in itertools.combinations(sorted(adj[k]), 2):
            if j in adj[i]:
                continue
            if k not in sepset.get(frozenset((i, j)), set()):
                A[k, i] = 0
                A[k, j] = 0
    # Meek rules R1-R3 until nothing changes
    changed = True
    while changed:
        changed = False
        for i, j in itertools.permutations(range(d), 2):
            if not (A[i, j] and A[j, i]):
                continue  # only undirected edges i - j
            # R1: k -> i - j, k not adjacent to j  =>  i -> j
            if any(A[k, i] and not A[i, k] and not A[k, j] and not A[j, k] for k in range(d) if k not in (i, j)):
                A[j, i] = 0
                changed = True
                continue
            # R2: i -> k -> j and i - j  =>  i -> j
            if any(A[i, k] and not A[k, i] and A[k, j] and not A[j, k] for k in range(d) if k not in (i, j)):
                A[j, i] = 0
                changed = True
                continue
            # R3: i - k1 -> j, i - k2 -> j, k1, k2 non-adjacent  =>  i -> j
            ks = [k for k in range(d) if k not in (i, j) and A[i, k] and A[k, i] and A[k, j] and not A[j, k]]
            if any(not A[a, b] and not A[b, a] for a, b in itertools.combinations(ks, 2)):
                A[j, i] = 0
                changed = True
    return A


# ----------------------------------------------------------------------------
# NOTEARS (linear, least squares + L1)
# ----------------------------------------------------------------------------
def notears_linear(X, lambda1=0.05, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    """Augmented-Lagrangian NOTEARS. W = W_pos - W_neg with W_pos, W_neg >= 0 so the
    L1 term is smooth; L-BFGS-B handles the bounds."""
    n, d = X.shape
    X = X - X.mean(0)

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _loss(W):
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):
        E = sla.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _func(w):
        W = _adj(w)
        loss, g_loss = _loss(W)
        h, g_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        g = g_loss + (rho * h + alpha) * g_h
        return obj, np.concatenate([g + lambda1, -g + lambda1], axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W


# ----------------------------------------------------------------------------
# sortnregress baseline
# ----------------------------------------------------------------------------
def sortnregress(X, w_threshold=0.3):
    order = np.argsort(X.var(0))
    W = np.zeros((X.shape[1], X.shape[1]))
    for k in range(1, len(order)):
        tgt, cov = order[k], order[:k]
        lin = LinearRegression().fit(X[:, cov], X[:, tgt])
        w = np.abs(lin.coef_)
        las = LassoLarsIC(criterion="bic").fit(X[:, cov] * w, X[:, tgt])
        W[cov, tgt] = las.coef_ * w
    W[np.abs(W) < w_threshold] = 0
    return W


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def shd_dag(W_est, W_true):
    """Structural Hamming distance between DAGs: missing + extra edges, a reversal counts once."""
    E, T = (W_est != 0).astype(int), (W_true != 0).astype(int)
    diff = np.abs(E - T)
    rev = ((E == 1) & (T.T == 1) & (T == 0)).sum()
    return int(diff.sum() - rev)


def shd_cpdag(A_est, A_true):
    """Count node pairs whose mark (none, ->, <-, -) differs."""
    s = 0
    for i, j in itertools.combinations(range(A_true.shape[0]), 2):
        s += (A_est[i, j], A_est[j, i]) != (A_true[i, j], A_true[j, i])
    return int(s)


def varsortability(X, W):
    """Fraction of directed paths (here: edges) along which marginal variance increases."""
    v = X.var(0)
    pairs = list(zip(*np.nonzero(W)))
    return float(np.mean([v[j] > v[i] for i, j in pairs]))


def main(n=5000, n_seeds=10):
    G_true = nx.DiGraph(TRUE_EDGES)
    oracle = pc(lambda i, j, S: d_separated(G_true, {i}, {j}, set(S)), D)
    rows = {k: [] for k in ["PC (raw)", "PC (standardised)", "NOTEARS (raw)", "NOTEARS (standardised)",
                            "sortnregress (raw)", "sortnregress (standardised)"]}
    vs = []
    keep = None
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        X, W = simulate(n, rng)
        Xs = (X - X.mean(0)) / X.std(0)
        vs.append(varsortability(X, W))
        C, Cs = np.corrcoef(X.T), np.corrcoef(Xs.T)
        A_pc = pc(lambda i, j, S: fisher_z_test(C, n, i, j, S) > 0.01, D)
        A_pcs = pc(lambda i, j, S: fisher_z_test(Cs, n, i, j, S) > 0.01, D)
        W_nt, W_nts = notears_linear(X), notears_linear(Xs)
        W_sr, W_srs = sortnregress(X), sortnregress(Xs)
        rows["PC (raw)"].append(shd_cpdag(A_pc, oracle))
        rows["PC (standardised)"].append(shd_cpdag(A_pcs, oracle))
        rows["NOTEARS (raw)"].append(shd_dag(W_nt, W))
        rows["NOTEARS (standardised)"].append(shd_dag(W_nts, W))
        rows["sortnregress (raw)"].append(shd_dag(W_sr, W))
        rows["sortnregress (standardised)"].append(shd_dag(W_srs, W))
        print(f"seed {seed}: varsort={vs[-1]:.2f}  " + "  ".join(f"{k}={v[-1]}" for k, v in rows.items()))
        if keep is None:
            keep = (W, oracle, A_pc, W_nt, W_nts, W_srs)
    summary = {k: {"mean_shd": float(np.mean(v)), "exact_recovery_rate": float(np.mean(np.array(v) == 0))}
               for k, v in rows.items()}
    print("\nMean SHD (PC vs true CPDAG; others vs true DAG):")
    for k, v in summary.items():
        print(f"  {k:<30} {v['mean_shd']:.2f}   exact {v['exact_recovery_rate']:.1f}")
    print(f"mean varsortability on raw data: {np.mean(vs):.2f}")

    # Figure: true DAG, true CPDAG, PC, NOTEARS raw, NOTEARS standardised, sortnregress standardised
    W, orac, A_pc, W_nt, W_nts, W_srs = keep
    pos = {0: (0, 1), 1: (1, 1.3), 2: (0.4, 0.5), 3: (0.4, -0.3), 4: (1.4, 0.4), 5: (1, -0.8)}

    def draw(ax, A, title, true_dir=None, cpdag=False):
        G = nx.DiGraph()
        G.add_nodes_from(range(D))
        directed, undirected = [], []
        for i, j in itertools.permutations(range(D), 2):
            if A[i, j] != 0 and (not cpdag or A[j, i] == 0):
                directed.append((i, j))
            elif cpdag and A[i, j] and A[j, i] and i < j:
                undirected.append((i, j))
        cols = []
        for e in directed:
            cols.append(PALETTE[2] if (true_dir is None or true_dir[e]) else PALETTE[3])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#DDDDDD", node_size=600, edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax, labels={i: f"X{i}" for i in range(D)}, font_size=9)
        nx.draw_networkx_edges(G, pos, edgelist=directed, ax=ax, arrows=True, arrowsize=15, node_size=600,
                               edge_color=cols, width=2)
        nx.draw_networkx_edges(G, pos, edgelist=undirected, ax=ax, arrows=False, node_size=600,
                               edge_color=PALETTE[0], width=2, style="dashed")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        ax.margins(0.2)

    T = {(i, j): W[i, j] != 0 for i in range(D) for j in range(D)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    draw(axes[0, 0], W, "True DAG")
    draw(axes[0, 1], orac, "True CPDAG (PC with d-sep oracle)\nblue dashed = not orientable", cpdag=True)
    draw(axes[0, 2], A_pc, f"PC, Fisher-z alpha=0.01 (SHD {shd_cpdag(A_pc, orac)})", cpdag=True)
    draw(axes[1, 0], W_nt, f"NOTEARS, raw data (SHD {shd_dag(W_nt, W)})", T)
    draw(axes[1, 1], W_nts, f"NOTEARS, standardised (SHD {shd_dag(W_nts, W)})", T)
    draw(axes[1, 2], W_srs, f"sortnregress, standardised (SHD {shd_dag(W_srs, W)})", T)
    fig.suptitle("Causal discovery on one simulated dataset (n=5000). Red = wrong edge/direction", fontsize=10)
    savefig(fig, "ex11_causal_discovery")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(summary)
    ax.barh(names, [summary[k]["mean_shd"] for k in names],
            color=[PALETTE[0], PALETTE[5], PALETTE[3], PALETTE[1], PALETTE[4], PALETTE[6]])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean structural Hamming distance over {n_seeds} random graphs' weights (lower is better)")
    ax.set_title("Scale matters: variance-based shortcuts vanish after standardising")
    savefig(fig, "ex11_discovery_shd")
    save_json({"n": n, "n_seeds": n_seeds, "summary": summary, "mean_varsortability_raw": float(np.mean(vs)),
               "true_edges": TRUE_EDGES}, "ex11_causal_discovery")


if __name__ == "__main__":
    main()
