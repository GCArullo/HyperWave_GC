"""Batched likelihood path equals the per-walker path (bit-for-bit)."""

from __future__ import annotations

import numpy as np

from conftest import requires_lal


def _build_likelihood(segment, names):
    from hyperwave.detectors.lvk import noise as noise_mod
    from hyperwave.detectors.lvk import waveform as wf_mod
    from hyperwave.likelihoods import gwparallel

    trig = segment["trigger_time"]
    ng = noise_mod.DetectorNoise(
        segment["duration"], segment["sampling_rate"], trig, segment["detectors"],
        maximum_frequency=segment["maximum_frequency"],
    )
    ng.generate_noise(real_noise=False, seed=42)
    injector = wf_mod.GW(ng, "IMRPhenomPv2", reference_frequency=50.0, parameters=names,
                         static_parameters={"geocent_time": trig})
    q = 29.0 / 36.0
    eta = 36.0 * 29.0 / (36.0 + 29.0) ** 2
    chirp = (36.0 + 29.0) * eta**0.6
    injector.make_injections_to_ifo(
        [chirp, q, 1000.0, 1.2, 0.64, 1.375, 0.21, 0.0, 0.0, np.cos(0.4), 1.0, 1.0, 0.0, 0.0]
    )
    f, a0 = injector.detector_asd_masked(0)
    a1 = injector.detector_asd_masked(1)[1]
    noise = np.array([a0**2, a1**2])
    data = np.array([injector.detector_data_fd(0), injector.detector_data_fd(1)])

    tn = noise_mod.DetectorNoise(
        segment["duration"], segment["sampling_rate"], trig, segment["detectors"],
        maximum_frequency=segment["maximum_frequency"],
    )
    tn.generate_noise(real_noise=False, seed=7)
    template = wf_mod.GW(tn, "IMRPhenomPv2", reference_frequency=50.0, parameters=names,
                         static_parameters={"geocent_time": trig})
    lik = gwparallel.GWLikelihoods(
        data=data, f=f, ifos_list=segment["detectors"], noise=noise, template=template,
        ddims=False, nsegs=4, gpu=False,
    )
    return lik


@requires_lal
def test_batched_equals_per_walker(segment, bbh_parameter_names):
    lik = _build_likelihood(segment, bbh_parameter_names)
    q = 29.0 / 36.0
    eta = 36.0 * 29.0 / (36.0 + 29.0) ** 2
    chirp = (36.0 + 29.0) * eta**0.6
    thetas = np.array([
        [chirp, q, 1000.0, 1.2, 0.64, 1.375, 0.21, 0, 0, np.cos(0.4), 1, 1, 0, 0],
        [chirp * 1.01, q, 1100.0, 1.0, 0.7, 1.4, 0.2, 0, 0, np.cos(0.5), 1, 1, 0, 0],
        [chirp * 0.99, 0.8, 900.0, 0.9, 0.6, 1.3, 0.22, 0, 0, np.cos(0.3), 1, 1, 0, 0],
    ])
    assert lik._batched_template is True
    g_batched = lik.gaussian(thetas)

    lik._batched_template = False  # force per-walker reference path
    g_single = lik.gaussian(thetas)

    np.testing.assert_array_equal(g_batched, g_single)
