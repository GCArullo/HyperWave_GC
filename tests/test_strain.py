"""Strain FFT conventions vs bilby reference."""

from __future__ import annotations

import numpy as np

from conftest import requires_bilby
from hyperwave.detectors.strain import StrainData, infft, nfft


def test_nfft_infft_roundtrip():
    rng = np.random.default_rng(0)
    fs, dur = 4096, 4.0
    td = rng.normal(0, 1e-21, int(fs * dur))
    fd, freq = nfft(td, fs)
    assert freq[0] == 0.0 and np.isclose(freq[-1], fs / 2)
    np.testing.assert_allclose(infft(fd, fs), td, atol=1e-30)


@requires_bilby
def test_windowed_fd_matches_bilby():
    import bilby

    rng = np.random.default_rng(2)
    fs, dur = 4096, 4.0
    td = rng.normal(0, 1e-21, int(fs * dur))

    sd = StrainData(fs, dur, start_time=0.0)
    sd.set_from_time_domain_strain(td)
    fd = sd.frequency_domain_strain

    bsd = bilby.gw.detector.strain_data.InterferometerStrainData(
        minimum_frequency=0, maximum_frequency=fs / 2
    )
    bsd.set_from_time_domain_strain(td, sampling_frequency=fs, duration=dur, start_time=0.0)
    fd_b = bsd.frequency_domain_strain

    np.testing.assert_array_equal(fd, fd_b)
    np.testing.assert_array_equal(sd.frequency_array, bsd.frequency_array)


def test_model_fd_is_not_windowed():
    fs, dur = 4096, 4.0
    n = int(fs * dur) // 2 + 1
    model = np.ones(n, dtype=complex)
    sd = StrainData(fs, dur)
    sd.set_from_frequency_domain_strain(model)
    np.testing.assert_array_equal(sd.frequency_domain_strain, model)
