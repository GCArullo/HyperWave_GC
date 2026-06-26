"""TBS per-segment log Bayes factor — dynesty (paper baseline) vs pocoMC.

Faithful reproduction of the per-segment job in the coherent BBH-background
mock-data challenge (Kou, Saleem, Mandic, Talbot & Thrane, arXiv:2506.14179;
method from Smith & Thrane 2017). For one 4-second L1 segment we compute the
signal-vs-noise log Bayes factor

    ln B = ln Z_s - ln Z_n

with the Whittle ``GravitationalWaveTransient`` likelihood. ``ln Z_n`` is the
analytic noise evidence (``likelihood.noise_log_likelihood()``); only the signal
hypothesis needs a sampler. The reference pipeline runs dynesty
(``result.log_bayes_factor``); this script adds a drop-in pocoMC path and times
both so the per-segment acceleration can be measured.

The pocoMC path is a thin, self-contained adapter (no fastjumps model-averaging):
because the noise likelihood is θ-independent, sampling the *likelihood ratio*
makes pocoMC's evidence equal to ln B directly:

    ln integral (L_s/L_n) π dθ = ln (Z_s / Z_n) = ln B .

    # paper-baseline single segment
    python examples/tbs/tbs_segment.py --segment 0 --sampler dynesty
    # the acceleration candidate, same segment + seed
    python examples/tbs/tbs_segment.py --segment 0 --sampler pocomc
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np


# ---------------------------------------------------------------------------
# Segment construction — mirrors tbs.py (reference). Even segment index ->
# noise+injection, odd -> noise only, matching the mock-data-challenge labels.
# ---------------------------------------------------------------------------
def build_segment(args):
    import bilby
    from bilby.core.utils import create_white_noise, logger
    from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
    from bilby.gw.detector.psd import PowerSpectralDensity
    from bilby.gw.source import lal_binary_black_hole

    logger.setLevel("WARNING")
    fs = args.sampling_frequency
    duration = args.duration
    fine_freq = np.fft.rfftfreq(int(fs * duration), 1.0 / fs)

    # --- PSD: GW170814 pickle (reference) if given, else aLIGO design stand-in.
    if args.psd_pickle and os.path.exists(args.psd_pickle):
        with open(args.psd_pickle, "rb") as f:
            sets = pickle.load(f)
        true_psd = np.asarray(sets["psd"])
        psd_freqs = np.asarray(sets["short_frequencies"])
        psd_source = os.path.basename(args.psd_pickle)
    else:
        psd_obj = PowerSpectralDensity(psd_file="aLIGO_ZERO_DET_high_P_psd.txt")
        true_psd = psd_obj.power_spectral_density_interpolated(fine_freq)
        true_psd[~np.isfinite(true_psd)] = np.inf
        psd_freqs = fine_freq
        psd_source = "aLIGO_ZERO_DET_high_P (stand-in for GW170814 pickle)"

    # --- mock coloured noise + mock estimated PSD (n_average realisations).
    # Seed BOTH numpy and bilby's own RNG: recent bilby samples priors via
    # bilby.core.utils.random.rng, so np.random.seed alone does NOT make the
    # injection reproducible -- without this the dynesty and pocoMC invocations
    # of the same segment drew *different* injections (mc=77 vs mc=27), making
    # their logBFs incomparable.
    rng_seed = args.seed + args.segment
    np.random.seed(rng_seed)
    from bilby.core.utils import random as _bilby_random
    _bilby_random.seed(rng_seed)
    noise_collect = []
    for _ in range(args.n_average + 1):
        white, _ = create_white_noise(sampling_frequency=fs, duration=duration)
        # true_psd may be on a coarser grid; interpolate onto fine_freq.
        psd_on_fine = np.interp(fine_freq, psd_freqs, true_psd, left=np.inf, right=np.inf)
        noise_collect.append(white * psd_on_fine ** 0.5)
    mock_fd_noise = noise_collect[0]

    inject = (args.segment % 2 == 0) if args.inject is None else args.inject
    tag = "injection" if inject else "no_injection"
    label = f"tbs_{args.sampler}_seg{args.segment}_{tag}"

    waveform_arguments = dict(waveform_approximant=args.approximant,
                              minimum_frequency=args.fmin,
                              maximum_frequency=args.fmax)
    start_time = 1219101500 + (args.segment % 200) * duration
    trigtime = start_time + duration - args.post_trigger_duration

    wfg = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=fs, start_time=start_time,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)

    priors = (bilby.gw.prior.BBHPriorDict(args.prior_file)
              if args.prior_file and os.path.exists(args.prior_file)
              else bilby.gw.prior.BBHPriorDict())
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigtime - 0.5, trigtime + 0.5, name="geocent_time")

    ifos = bilby.gw.detector.InterferometerList([args.ifo])
    ifos[0].minimum_frequency = args.fmin
    ifos[0].maximum_frequency = args.fmax
    ifos[0].power_spectral_density = (
        PowerSpectralDensity.from_power_spectral_density_array(psd_freqs, true_psd))
    ifos[0].strain_data.set_from_frequency_domain_strain(
        frequency_domain_strain=mock_fd_noise, frequency_array=fine_freq,
        duration=duration, start_time=start_time)

    injection_parameters = None
    if inject:
        injection_parameters = priors.sample(1)
        injection_parameters = {k: float(np.asarray(v).ravel()[0])
                                for k, v in injection_parameters.items()}
        ifos[0].inject_signal(waveform_generator=wfg,
                              parameters=injection_parameters)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg,
        reference_frame="sky", time_reference="geocent",
        time_marginalization=False, distance_marginalization=False,
        phase_marginalization=False, jitter_time=False, priors=priors)

    return dict(likelihood=likelihood, priors=priors, label=label,
                inject=inject, injection_parameters=injection_parameters,
                psd_source=psd_source, seed=rng_seed)


# ---------------------------------------------------------------------------
# Samplers.
# ---------------------------------------------------------------------------
def run_dynesty(seg, args):
    import bilby
    sampler_kwargs = dict(bound="multi", sample="rwalk", nlive=args.nlive,
                          walks=args.walks, slices=5, update_interval=300,
                          maxmcmc=5000, first_update={"min_ncall": 6000})
    t0 = time.perf_counter()
    result = bilby.run_sampler(
        likelihood=seg["likelihood"], priors=seg["priors"], sampler="dynesty",
        injection_parameters=seg["injection_parameters"], outdir=args.outdir,
        label=seg["label"], dlogz=args.dlogz, **sampler_kwargs)
    wall = time.perf_counter() - t0
    return result.log_bayes_factor, getattr(result, "log_evidence_err", float("nan")), wall, len(result.posterior)


def run_pocomc(seg, args):
    """Self-contained pocoMC adapter; evidence of L-ratio == ln Bayes factor."""
    import pocomc as pc
    from scipy.stats import uniform
    likelihood = seg["likelihood"]
    priors = seg["priors"]
    # sampled keys = priors that actually vary (exclude fixed/constraint)
    from bilby.core.prior import Constraint
    keys = [k for k, p in priors.items() if not isinstance(p, Constraint)]

    def make_dist(p):
        # scipy.stats-like object pocomc accepts: logpdf, rvs, support
        d = type("BP", (), {})()
        d.logpdf = lambda x, _p=p: _p.ln_prob(x)
        d.rvs = lambda size=1, random_state=None, _p=p: np.atleast_1d(
            np.asarray(_p.sample(size)).ravel())
        d.support = lambda _p=p: (float(_p.minimum), float(_p.maximum))
        return d

    prior = pc.Prior([make_dist(priors[k]) for k in keys])

    def log_like(x):
        x = np.atleast_2d(x)
        out = np.empty(x.shape[0])
        for i, row in enumerate(x):
            likelihood.parameters = dict(zip(keys, row))
            out[i] = likelihood.log_likelihood_ratio()
        return out

    t0 = time.perf_counter()
    sampler = pc.Sampler(prior=prior, likelihood=log_like,
                         n_effective=args.n_effective, n_active=args.n_active,
                         vectorize=False)
    sampler.run(n_total=args.n_total)
    logbf, logbf_err = sampler.evidence()
    wall = time.perf_counter() - t0
    samples = sampler.posterior(resample=True)
    if isinstance(samples, tuple):
        samples = samples[0]
    return float(logbf), float(logbf_err), wall, int(np.asarray(samples).shape[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", type=int, default=0, help="segment index (even=inject)")
    p.add_argument("--sampler", choices=["dynesty", "pocomc"], default="pocomc")
    p.add_argument("--inject", dest="inject", action="store_true", default=None)
    p.add_argument("--no-inject", dest="inject", action="store_false")
    p.add_argument("--ifo", default="L1")
    p.add_argument("--approximant", default="IMRPhenomPv2")
    p.add_argument("--sampling-frequency", type=float, default=2048.0)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--post-trigger-duration", type=float, default=1.0)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--n-average", type=int, default=32)
    p.add_argument("--psd-pickle", default=None,
                   help="GW170814_data_coarse_*.pkl (reference); else aLIGO design")
    p.add_argument("--prior-file", default=None,
                   help="full_prior.prior (reference); else bilby BBHPriorDict default")
    # dynesty knobs (paper settings)
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--walks", type=int, default=100)
    p.add_argument("--dlogz", type=float, default=0.1)
    # pocomc knobs
    p.add_argument("--n-total", type=int, default=4096)
    p.add_argument("--n-effective", type=int, default=512)
    p.add_argument("--n-active", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/tbs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seg = build_segment(args)
    print(f"[tbs] segment {args.segment} | inject={seg['inject']} | "
          f"psd={seg['psd_source']} | sampler={args.sampler}", flush=True)
    if seg["inject"]:
        ip = seg["injection_parameters"]
        if ip.get("chirp_mass") is None and ip.get("mass_1") is not None:
            m1, m2 = float(ip["mass_1"]), float(ip["mass_2"])
            mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            q = min(m1, m2) / max(m1, m2)
        else:
            mc, q = ip.get("chirp_mass"), ip.get("mass_ratio")
        print(f"  injected: mc={mc:.1f} q={q:.2f} "
              f"dL={float(ip.get('luminosity_distance', 0)):.0f}", flush=True)

    if args.sampler == "dynesty":
        logbf, err, wall, nsamp = run_dynesty(seg, args)
    else:
        logbf, err, wall, nsamp = run_pocomc(seg, args)

    print(f"\n[result] sampler={args.sampler}  ln B = {logbf:.3f} +/- {err:.3f}  "
          f"| wall {wall:.1f} s ({wall/60:.2f} min) | {nsamp} samples", flush=True)
    out = dict(log_bayes_factor=logbf, log_bayes_factor_err=err, wall_s=wall,
               n_samples=nsamp, sampler=args.sampler, segment=args.segment,
               inject=seg["inject"], psd_source=seg["psd_source"], seed=seg["seed"],
               injection_parameters=seg["injection_parameters"])
    fn = os.path.join(args.outdir, f"{seg['label']}.p")
    pickle.dump(out, open(fn, "wb"))
    print(f"saved -> {fn}")


if __name__ == "__main__":
    main()
