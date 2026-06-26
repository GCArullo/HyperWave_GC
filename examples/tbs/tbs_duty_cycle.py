"""Aggregate the TBS per-segment log Bayes factors into the duty-cycle posterior.

Reads the per-segment ``tbs_pocomc_seg*_*.p`` files written by
``tbs_segment.py`` (one per array task), splits them into injection vs
noise-only by the filename tag, and reconstructs the duty-cycle ``xi``
posterior the way the coherent BBH-background search does (Smith & Thrane 2017;
Kou+ 2506.14179). Also makes the PP plot over many random ``xi_true`` draws,
the calibration check for the whole pipeline.

    python examples/tbs/tbs_duty_cycle.py --dir results/tbs_campaign

Ports the xi-posterior / PP-plot maths from the PI's pp_plot.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


# ---- xi (duty cycle) posterior, ported from pp_plot.py --------------------
def logLtot(xi, logZSs, logZNs):
    if xi == 0.0:
        return np.sum(logZNs)
    if xi == 1.0:
        return np.sum(logZSs)
    x = np.log(xi) + logZSs
    y = np.log(1 - xi) + logZNs
    return np.sum(logsumexp(a=[x, y], axis=0))


def get_xi_posterior(logbfs, xi_min=0.0, xi_max=1.0, npoints=1000):
    """Per-segment log Bayes factors -> normalised p(xi)."""
    logbfs = np.asarray(logbfs, dtype=float)
    logZNs = np.zeros(len(logbfs)) - 10.0       # arbitrary common noise reference
    logZSs = logbfs + logZNs                    # lnB = logZS - logZN
    xi = np.linspace(xi_min, xi_max, npoints)
    Ltot = np.array([logLtot(x, logZSs, logZNs) for x in xi])
    prob = np.exp(Ltot - Ltot.max())
    norm = np.sum(prob) * (xi[1] - xi[0])
    return xi, prob / norm


def credible_level(x, prob_x, x_inj):
    cl = np.sum(prob_x[x < x_inj]) / np.sum(prob_x)
    cl = 2 * (1 - cl) if cl > 0.5 else 2 * cl
    return float(cl)


# ---- loading --------------------------------------------------------------
def load_logbfs(directory):
    inj, noise = [], []
    # sampler-agnostic: tbs_pocomc_seg*  or  tbs_dynesty_seg*
    for p in sorted(glob.glob(os.path.join(directory, "tbs_*_seg*_*.p"))):
        d = pickle.load(open(p, "rb"))
        (inj if d.get("inject") else noise).append(float(d["log_bayes_factor"]))
    return np.array(inj), np.array(noise)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/tbs_campaign")
    ap.add_argument("--out", default="results/tbs_campaign")
    ap.add_argument("--nseg", type=int, default=1000,
                    help="segments per single xi analysis")
    ap.add_argument("--npp", type=int, default=300, help="xi_true draws for the PP plot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inj, noise = load_logbfs(args.dir)
    print(f"loaded {inj.size} injection + {noise.size} noise segments")
    if inj.size == 0 or noise.size == 0:
        raise SystemExit("need both injection and noise segments")

    # --- 1) duty-cycle recovery at a few injected xi -----------------------
    plt.figure(figsize=(7, 5))
    rng = np.random.default_rng(0)
    for xi_true in (0.1, 0.3, 0.5):
        n_inj = int(xi_true * args.nseg)
        lnbfs = np.concatenate([rng.choice(inj, n_inj, replace=inj.size < n_inj),
                                rng.choice(noise, args.nseg - n_inj,
                                           replace=noise.size < args.nseg - n_inj)])
        xi, p = get_xi_posterior(lnbfs)
        line, = plt.plot(xi, p, label=f"$\\xi_{{true}}={xi_true}$")
        plt.axvline(xi_true, color=line.get_color(), ls="--", alpha=0.6)
    plt.xlabel("duty cycle $\\xi$"); plt.ylabel("PDF")
    plt.title(f"TBS duty-cycle recovery ({args.nseg} segments / analysis)")
    plt.legend(); plt.xlim(0, 0.7)
    f1 = os.path.join(args.out, "tbs_duty_cycle.png")
    plt.savefig(f1, dpi=130, bbox_inches="tight"); plt.close()

    # --- 2) PP plot over random xi_true ------------------------------------
    cls = []
    for _ in range(args.npp):
        xi_true = rng.uniform(0, 1)
        n_inj = int(xi_true * args.nseg)
        lnbfs = np.concatenate([rng.choice(inj, n_inj, replace=True),
                                rng.choice(noise, args.nseg - n_inj, replace=True)])
        xi, p = get_xi_posterior(lnbfs)
        cls.append(credible_level(xi, p, n_inj / args.nseg))
    cls = np.sort(cls)
    x = np.linspace(0, 1, 1001)
    y = np.quantile(cls, x)
    plt.figure(figsize=(5.5, 5.5))
    for ci, a in [(0.997, 0.3), (0.95, 0.3), (0.68, 0.3)]:
        e = (1 - ci) / 2
        lo = binom.ppf(1 - e, len(cls), x) / len(cls)
        hi = binom.ppf(e, len(cls), x) / len(cls)
        lo[0] = hi[0] = 0
        plt.fill_between(x, lo, hi, color="grey", alpha=a)
    plt.plot(x, y, color="#B32222", label="TBS (pocoMC)")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("C.I."); plt.ylabel("fraction of events in C.I.")
    plt.title(f"TBS duty-cycle PP ({args.npp} draws)"); plt.legend(loc="upper left")
    f2 = os.path.join(args.out, "tbs_pp.png")
    plt.savefig(f2, dpi=130, bbox_inches="tight"); plt.close()

    print(f"injection lnB: median {np.median(inj):.1f}  (frac>0: {np.mean(inj > 0):.2f})")
    print(f"noise     lnB: median {np.median(noise):.1f}  (frac>0: {np.mean(noise > 0):.2f})")
    print(f"plots -> {f1}\n         {f2}")


if __name__ == "__main__":
    main()
