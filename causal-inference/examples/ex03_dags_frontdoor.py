"""Example 3: drawing DAGs, checking d-separation, and the front-door formula.

Author: Abyn Scaria. MIT License.

Part A draws the canonical graphs used throughout the docs with networkx.
Part B uses networkx's d-separation routine to check backdoor adjustment sets
mechanically, for one fairly busy graph.
Part C simulates a front-door model where the T-Y confounder U is never
observed, so backdoor adjustment is impossible, yet the effect is still
identified through the mediator M:

    U -> T, U -> Y, T -> M -> Y       (no direct T -> Y edge, M not confounded)

Binary variables, so we can apply the front-door formula exactly:
    P(y | do(t)) = sum_m P(m | t) * sum_t' P(y | m, t') P(t')
"""
from common import PALETTE, d_separated, plt, save_json, savefig  # noqa: I001

import itertools

import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# Part A: draw canonical DAGs
# ----------------------------------------------------------------------------
GRAPHS = {
    "Confounder (fork)": ([("Z", "T"), ("Z", "Y"), ("T", "Y")],
                          {"Z": (0.5, 1), "T": (0, 0), "Y": (1, 0)}, set()),
    "Mediator (chain)": ([("T", "M"), ("M", "Y"), ("T", "Y")],
                         {"M": (0.5, 1), "T": (0, 0), "Y": (1, 0)}, set()),
    "Collider": ([("T", "C"), ("Y", "C"), ("T", "Y")],
                 {"C": (0.5, -1), "T": (0, 0), "Y": (1, 0)}, set()),
    "M-bias": ([("U1", "T"), ("U1", "M"), ("U2", "M"), ("U2", "Y"), ("T", "Y")],
               {"U1": (0, 1), "U2": (1, 1), "M": (0.5, 0.5), "T": (0, 0), "Y": (1, 0)}, {"U1", "U2"}),
    "Front-door": ([("U", "T"), ("U", "Y"), ("T", "M"), ("M", "Y")],
                   {"U": (0.5, 1), "T": (0, 0), "M": (0.5, 0), "Y": (1, 0)}, {"U"}),
    "Instrument": ([("Z", "T"), ("U", "T"), ("U", "Y"), ("T", "Y")],
                   {"Z": (-0.6, 0), "U": (0.5, 1), "T": (0, 0), "Y": (1, 0)}, {"U"}),
}


def draw_dag(ax, edges, pos, latent, title):
    G = nx.DiGraph(edges)
    colors = []
    for v in G.nodes:
        if v in latent:
            colors.append("#DDDDDD")
        elif v == "T":
            colors.append(PALETTE[0])
        elif v == "Y":
            colors.append(PALETTE[1])
        else:
            colors.append(PALETTE[2])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=900, edgecolors="k")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18, node_size=900, width=1.5,
                           style=[("dashed" if (u in latent) else "solid") for u, _ in G.edges])
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.margins(0.25)


def part_a():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (title, (edges, pos, latent)) in zip(axes.ravel(), GRAPHS.items()):
        draw_dag(ax, edges, pos, latent, title)
    fig.suptitle("Canonical causal graphs (grey/dashed = unobserved; blue = treatment, orange = outcome)",
                 fontsize=10)
    savefig(fig, "ex03_canonical_dags")


# ----------------------------------------------------------------------------
# Part B: backdoor criterion via d-separation
# ----------------------------------------------------------------------------
BUSY_EDGES = [
    ("A", "T"), ("A", "B"), ("B", "Y"),     # backdoor T <- A -> B -> Y
    ("C", "T"), ("C", "Y"),                 # direct confounder
    ("T", "M"), ("M", "Y"),                 # mediator (do NOT adjust)
    ("T", "D"), ("Y", "D"),                 # collider (do NOT adjust)
    ("E", "Y"),                             # outcome-only cause (safe, adds precision)
    ("F", "T"),                             # treatment-only cause (safe, costs precision)
]


def satisfies_backdoor(G, t, y, Z):
    """Pearl's backdoor criterion: (i) no node in Z is a descendant of t;
    (ii) Z blocks every path between t and y that starts with an arrow into t.

    (ii) is checked by deleting t's outgoing edges and asking whether t and y are
    d-separated given Z in the remaining graph.
    """
    Z = set(Z)
    if Z & nx.descendants(G, t):
        return False
    G_back = G.copy()
    G_back.remove_edges_from(list(G.out_edges(t)))
    return d_separated(G_back, {t}, {y}, Z)


def part_b():
    G = nx.DiGraph(BUSY_EDGES)
    candidates = [v for v in G.nodes if v not in ("T", "Y")]
    valid = []
    for r in range(len(candidates) + 1):
        for Z in itertools.combinations(sorted(candidates), r):
            if satisfies_backdoor(G, "T", "Y", Z):
                valid.append(list(Z))
    minimal = [z for z in valid if not any(set(o) < set(z) for o in valid)]
    checks = {
        "{}": satisfies_backdoor(G, "T", "Y", []),
        "{C}": satisfies_backdoor(G, "T", "Y", ["C"]),
        "{A, C}": satisfies_backdoor(G, "T", "Y", ["A", "C"]),
        "{B, C}": satisfies_backdoor(G, "T", "Y", ["B", "C"]),
        "{A, C, M}": satisfies_backdoor(G, "T", "Y", ["A", "C", "M"]),
        "{A, C, D}": satisfies_backdoor(G, "T", "Y", ["A", "C", "D"]),
        "{A, C, E}": satisfies_backdoor(G, "T", "Y", ["A", "C", "E"]),
    }
    print("\nPart B: backdoor checks on the busy graph")
    for k, v in checks.items():
        print(f"  Z = {k:<12} valid backdoor set? {v}")
    print(f"  minimal valid sets: {minimal}   (total valid sets: {len(valid)})")

    pos = {"A": (-1, 1.2), "B": (1, 1.2), "C": (0, 1.6), "F": (-1.6, 0), "T": (-0.8, 0), "M": (0, 0.45),
           "Y": (0.8, 0), "E": (1.6, 0.6), "D": (0, -0.8)}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    draw_dag(ax, BUSY_EDGES, pos, set(), "Which sets satisfy the backdoor criterion for T -> Y?")
    savefig(fig, "ex03_backdoor_graph")
    return {"checks": checks, "minimal_valid_sets": minimal, "n_valid_sets": len(valid)}


# ----------------------------------------------------------------------------
# Part C: front-door adjustment with an unobserved confounder
# ----------------------------------------------------------------------------
def simulate_frontdoor(n, rng):
    U = rng.binomial(1, 0.5, n)
    T = rng.binomial(1, 0.2 + 0.6 * U)            # U pushes people into treatment
    M = rng.binomial(1, 0.1 + 0.7 * T)            # T -> M, M not caused by U
    Y = rng.binomial(1, 0.1 + 0.4 * M + 0.4 * U)  # Y depends on M and U only
    return U, T, M, Y


def true_frontdoor_effect():
    # P(Y=1 | do(T=t)) = sum_u P(u) sum_m P(m|t) P(Y=1|m,u)
    def p_y_do(t):
        pm1 = 0.1 + 0.7 * t
        return sum(0.5 * ((1 - pm1) * (0.1 + 0.4 * u) + pm1 * (0.5 + 0.4 * u)) for u in (0, 1))
    return p_y_do(1) - p_y_do(0)


def frontdoor_estimate(T, M, Y):
    p_t = {t: np.mean(T == t) for t in (0, 1)}
    p_m_given_t = {(m, t): np.mean(M[T == t] == m) for m in (0, 1) for t in (0, 1)}
    p_y_given_mt = {(m, t): Y[(M == m) & (T == t)].mean() for m in (0, 1) for t in (0, 1)}

    def p_y_do(t):
        return sum(p_m_given_t[(m, t)] * sum(p_y_given_mt[(m, tp)] * p_t[tp] for tp in (0, 1)) for m in (0, 1))
    return p_y_do(1) - p_y_do(0)


def part_c(n_sims=300, n=5000):
    rng = np.random.default_rng(3)
    truth = true_frontdoor_effect()
    naive, fd, oracle = [], [], []
    for _ in range(n_sims):
        U, T, M, Y = simulate_frontdoor(n, rng)
        naive.append(Y[T == 1].mean() - Y[T == 0].mean())
        fd.append(frontdoor_estimate(T, M, Y))
        # Oracle backdoor adjustment, only possible because we simulated U.
        oracle.append(sum(0.5 * (Y[(T == 1) & (U == u)].mean() - Y[(T == 0) & (U == u)].mean()) for u in (0, 1)))
    res = {"truth": truth, "naive": float(np.mean(naive)), "naive_sd": float(np.std(naive)),
           "frontdoor": float(np.mean(fd)), "frontdoor_sd": float(np.std(fd)),
           "oracle_backdoor_on_U": float(np.mean(oracle)), "oracle_sd": float(np.std(oracle))}
    print("\nPart C: front-door (U unobserved)")
    for k, v in res.items():
        print(f"  {k:<22} {v:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(min(min(fd), min(oracle)) - 0.02, max(naive) + 0.02, 60)
    ax.hist(naive, bins=bins, alpha=0.6, label="naive difference", color=PALETTE[3])
    ax.hist(fd, bins=bins, alpha=0.6, label="front-door formula (U unobserved)", color=PALETTE[0])
    ax.hist(oracle, bins=bins, alpha=0.4, label="oracle backdoor on U", color=PALETTE[2])
    ax.axvline(truth, color="k", ls="--", label=f"truth = {truth:.3f}")
    ax.set_xlabel("estimated P(Y=1|do(T=1)) - P(Y=1|do(T=0))")
    ax.set_title(f"Front-door identification works without measuring the confounder ({n_sims} sims)")
    ax.legend(fontsize=8)
    savefig(fig, "ex03_frontdoor")
    return res


def main():
    part_a()
    b = part_b()
    c = part_c()
    save_json({"backdoor": b, "frontdoor": c}, "ex03_dags_frontdoor")


if __name__ == "__main__":
    main()
