"""Example 2: "bad controls". Adjusting for the wrong variable creates bias.

Author: Abyn Scaria. MIT License.

Three linear-Gaussian scenarios, each with a known total effect of T on Y.
In each one we compare the regression that leaves the extra variable out with
the regression that puts it in.

  (a) Collider:   T -> Y (effect 1.0),  T -> C <- Y
                  C is caused by both T and Y. Conditioning on it opens a
                  spurious path and biases the effect.
  (b) M-bias:     U1 -> T, U1 -> M <- U2, U2 -> Y, T -> Y (effect 1.0)
                  M is measured before treatment and is correlated with both
                  T and Y, so it looks like a confounder, but it is a collider
                  of two unobserved causes. Adjusting for it opens T<-U1->M<-U2->Y.
  (c) Mediator:   T -> M -> Y and T -> Y. Total effect = 0.5 + 0.8*1.0 = 1.3.
                  Adjusting for M gives the direct effect only (0.5), which is
                  the wrong answer if the question is "what does T do to Y?"
  (d) Confounder: Z -> T, Z -> Y, T -> Y (effect 1.0). Here adjusting is right.
"""
from common import PALETTE, ols, plt, save_json, savefig, summarize  # noqa: I001

import numpy as np


def collider(n, rng):
    T = rng.normal(size=n)
    Y = 1.0 * T + rng.normal(size=n)
    C = 1.0 * T + 1.0 * Y + rng.normal(size=n)
    return T, Y, C, 1.0


def m_bias(n, rng):
    U1, U2 = rng.normal(size=n), rng.normal(size=n)
    M = 1.0 * U1 + 1.0 * U2 + rng.normal(size=n)  # pre-treatment collider
    T = 1.0 * U1 + rng.normal(size=n)
    Y = 1.0 * T + 1.0 * U2 + rng.normal(size=n)
    return T, Y, M, 1.0


def mediator(n, rng):
    T = rng.normal(size=n)
    M = 1.0 * T + rng.normal(size=n)
    Y = 0.5 * T + 0.8 * M + rng.normal(size=n)
    return T, Y, M, 0.5 + 0.8 * 1.0


def confounder(n, rng):
    Z = rng.normal(size=n)
    T = 1.0 * Z + rng.normal(size=n)
    Y = 1.0 * T + 1.0 * Z + rng.normal(size=n)
    return T, Y, Z, 1.0


SCENARIOS = {
    "(a) collider C": collider,
    "(b) M-bias pre-treatment M": m_bias,
    "(c) mediator M": mediator,
    "(d) true confounder Z": confounder,
}


def main(n_sims=500, n=1000):
    rng = np.random.default_rng(0)
    out = {}
    for name, sim in SCENARIOS.items():
        without, with_ = [], []
        for _ in range(n_sims):
            T, Y, W, truth = sim(n, rng)
            without.append(ols(Y, T)[0][1])
            with_.append(ols(Y, np.column_stack([T, W]))[0][1])
        out[name] = {"truth": truth,
                     "omit": summarize(without, truth),
                     "adjust": summarize(with_, truth)}
        print(f"{name:<30} truth={truth:.2f}  omit={np.mean(without):.3f}  adjust={np.mean(with_):.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(out)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [out[k]["omit"]["mean"] for k in names], 0.4, label="leave variable out", color=PALETTE[0])
    ax.bar(x + 0.2, [out[k]["adjust"]["mean"] for k in names], 0.4, label="adjust for it", color=PALETTE[3])
    for i, k in enumerate(names):
        ax.hlines(out[k]["truth"], i - 0.42, i + 0.42, colors="k", linestyles="--", lw=1.5,
                  label="true total effect" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("estimated effect of T on Y")
    ax.set_title("Adjusting helps only for confounders")
    ax.legend(loc="upper right")
    savefig(fig, "ex02_bad_controls")
    save_json({"n": n, "n_sims": n_sims, "scenarios": out}, "ex02_bad_controls")


if __name__ == "__main__":
    main()
