"""BBH parameter estimation with the heterodyne (relative-binning) likelihood.

Demonstrates HyperWave's fastest likelihood: a reference waveform is computed
once at the injection point, the smooth ratio h/h0 is piecewise-linear over
PN-spaced frequency bins, and every likelihood call needs the waveform only at
the ~300 bin *edges* (via LAL's frequency-sequence API) instead of the full
grid. Per-evaluation cost is independent of the signal duration — the longer
the signal, the bigger the win (measured 4.9x at 4 s, 68.6x at 64 s).

The script (1) checks the heterodyne logL against the exact Gaussian logL,
(2) times both, and (3) runs a 4-parameter PE with Eryn, producing a corner
plot against the injection.

    python examples/heterodyne/bbh_heterodyne_pe.py            # full demo
    python examples/heterodyne/bbh_heterodyne_pe.py --quick    # smoke test
    python examples/heterodyne/bbh_heterodyne_pe.py --duration 64   # big-win regime
"""

from __future__ import annotations

import argparse
import os
import time

import bilby
import numpy as np

from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.inference import LVKinference
from hyperwave.likelihoods import GWLikelihoods, HeterodyneLikelihood

NAMES = ["chirp_mass", "mass_ratio", "luminosity_distance", "phase"]
TRIGGER = 1268189526.951953

STATIC = dict(
    psi=1.1, ra=1.375, dec=-0.2108, chi_1=0.0, chi_2=0.0,
    cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0,
    phi_12=0.0, phi_jl=0.0, geocent_time=TRIGGER,
)


def injected():
    m1, m2 = 36.0, 29.0
    mc = (m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6
    return np.array([mc, m2 / m1, 600.0, 0.9])


def build_problem(duration, fs=2048.0, fmin=20.0, fmax=512.0, seed=42):
    theta_ref = injected()
    noise = DetectorNoise(duration, fs, TRIGGER, ["H1", "L1"],
                          minimum_frequency=fmin, maximum_frequency=fmax)
    noise.generate_noise(real_noise=False, seed=seed)
    template = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
                  parameters=NAMES, static_parameters=STATIC)
    template.make_injections_to_ifo(theta_ref)

    f, asd0 = template.detector_asd_masked(0)
    psd = np.array([asd0 ** 2, template.detector_asd_masked(1)[1] ** 2])
    data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])
    return template, data, f, psd, theta_ref


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=4.0, help="segment length [s]")
    p.add_argument("--eps", type=float, default=0.1, help="max per-bin differential phase [rad]")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--burn", type=int, default=2000)
    p.add_argument("--outdir", default="results/heterodyne")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    template, data, f, psd, theta_ref = build_problem(args.duration)

    # --- exact Gaussian reference + heterodyne ---
    exact = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                          template=template, ddims=False, nsegs=1, gpu=False)
    het = HeterodyneLikelihood.from_lvk_template(
        template, data=data, f=f, psd=psd, ifos_list=["H1", "L1"],
        theta_ref=theta_ref, eps=args.eps)
    print(f"> grid {f.size} bins -> {het.f_edges.size} edges "
          f"({het.n_bins} bins, eps={args.eps})")

    # --- 1. accuracy: logL differences track the exact likelihood ---
    rng = np.random.default_rng(7)
    trials = np.tile(theta_ref, (16, 1))
    trials[:, 0] += rng.uniform(-0.02, 0.02, 16)
    trials[:, 2] *= 1 + rng.uniform(-0.05, 0.05, 16)
    g = np.atleast_1d(exact.gaussian(trials)) - float(exact.gaussian(theta_ref))
    h = np.atleast_1d(het.logl(trials)) - float(het.logl(theta_ref))
    print(f"> accuracy: max |Delta logL| error = {np.max(np.abs(h - g)):.4f} "
          f"over spreads of {np.max(np.abs(g)):.1f}")

    # --- 2. speed ---
    batch = np.tile(theta_ref, (64, 1))
    batch[:, 0] += rng.uniform(-0.02, 0.02, 64)
    exact.gaussian(batch[:2])
    het.logl(batch[:2])  # warm-up
    t0 = time.perf_counter()
    exact.gaussian(batch)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    het.logl(batch)
    t_het = time.perf_counter() - t0
    print(f"> speed: full {t_full/64*1e3:.2f} ms/eval | heterodyne {t_het/64*1e3:.2f} ms/eval "
          f"| speedup {t_full/t_het:.1f}x")

    # --- 3. PE with the heterodyne likelihood ---
    pr = bilby.core.prior.PriorDict()
    pr["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(26.0, 30.0, name="chirp_mass")
    pr["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.5, 1.0, name="mass_ratio")
    pr["luminosity_distance"] = bilby.core.prior.Uniform(300.0, 1000.0, name="luminosity_distance")
    pr["phase"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, name="phase")

    kw = dict(nwalkers=16, ntemps=2, burn=50, nsteps=200) if args.quick else \
         dict(nwalkers=32, ntemps=6, burn=args.burn, nsteps=args.steps)
    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    t0 = time.perf_counter()
    inf = LVKinference(het.logl, sampler_name="eryn", priors=pr, noise_priors={},
                       common_params={"save_dir": args.outdir, "TAG": "het", "like": "heterodyne"},
                       sampler_kwargs=kw, periodic=["phase"])
    inf.run()
    wall = time.perf_counter() - t0
    result = inf.get_result(injection=theta_ref, parameter_names=NAMES)
    med = result.median()
    med_str = ", ".join(f"{k}={float(v):.3f}" for k, v in
                        (med.items() if hasattr(med, "items") else zip(NAMES, np.atleast_1d(med))))
    print(f"\n[eryn+heterodyne] wall {wall:.1f}s | medians: {med_str} "
          f"| injected {np.round(theta_ref, 3)}")
    try:
        fig = result.corner()
        fig.savefig(os.path.join(args.outdir, "heterodyne_corner.png"), dpi=120)
        print(f"  corner -> {args.outdir}/heterodyne_corner.png")
    except Exception as exc:  # pragma: no cover
        print(f"  (corner skipped: {exc})")


if __name__ == "__main__":
    main()
