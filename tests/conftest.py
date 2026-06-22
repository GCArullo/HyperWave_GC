"""Shared test fixtures and optional-dependency skips."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _has(module):
    return importlib.util.find_spec(module) is not None


requires_lal = pytest.mark.skipif(
    not (_has("lal") and _has("lalsimulation")),
    reason="lalsuite (lal/lalsimulation) not installed",
)
requires_bilby = pytest.mark.skipif(not _has("bilby"), reason="bilby not installed (reference)")
requires_ml4gw = pytest.mark.skipif(
    not (_has("ml4gw") and _has("torch")), reason="ml4gw/torch not installed"
)


@pytest.fixture(scope="session")
def segment():
    """A standard analysis segment used across tests."""
    trigger_time = 1268189526.951953
    duration = 4.0
    sampling_rate = 4096
    post = 2.0
    end_time = trigger_time + post
    start_time = end_time - duration
    frequency_array = np.linspace(0.0, sampling_rate / 2, int(sampling_rate * duration) // 2 + 1)
    return dict(
        trigger_time=trigger_time,
        duration=duration,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        frequency_array=frequency_array,
        minimum_frequency=20.0,
        maximum_frequency=800.0,
        detectors=["H1", "L1"],
    )


@pytest.fixture
def bbh_theta():
    """A representative non-precessing BBH parameter vector (14 sampled params)."""
    q = 29.0 / 36.0
    eta = 36.0 * 29.0 / (36.0 + 29.0) ** 2
    chirp_mass = (36.0 + 29.0) * eta**0.6
    return [chirp_mass, q, 1000.0, 1.228444, 0.641716, 1.375, 0.2108,
            0.0, 0.0, np.cos(0.4), 1.0, 1.0, 0.0, 0.0]


@pytest.fixture
def bbh_parameter_names():
    return ["chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
            "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
            "cos_tilt_2", "phi_12", "phi_jl"]
