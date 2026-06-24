"""Tests for data-only frequency-domain likelihood helpers."""

from __future__ import annotations

import numpy as np

from hyperwave.likelihoods import LogLike


def test_loglike_single_detector_stacked_data_shape():
    f = np.linspace(20.0, 100.0, 8)
    data = np.ones((1, f.size), dtype=complex)
    likelihood = LogLike(data=data, f=f, ifos_list=["H1"], nsegs=2, ddims=False)

    assert likelihood.yy_noise.shape == (f.size,)
    value = likelihood.hyperbolic(np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(value)
