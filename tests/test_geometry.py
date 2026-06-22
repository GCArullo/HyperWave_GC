"""Detector geometry vs bilby reference."""

from __future__ import annotations

import numpy as np

from conftest import requires_bilby, requires_lal


@requires_lal
def test_antenna_response_is_batched():
    from hyperwave.detectors.geometry import Detector

    det = Detector("H1")
    n = 7
    rng = np.random.default_rng(0)
    fp, fc = det.antenna_response(
        rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n),
        rng.uniform(0, np.pi, n), 1.2e9 + rng.uniform(-1, 1, n),
    )
    assert fp.shape == (n,)
    assert fc.shape == (n,)


@requires_lal
@requires_bilby
def test_antenna_and_delay_match_bilby(segment):
    import bilby

    from hyperwave.detectors.geometry import Detector

    rng = np.random.default_rng(1)
    n = 10
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = rng.uniform(-np.pi / 2, np.pi / 2, n)
    psi = rng.uniform(0, np.pi, n)
    t = segment["trigger_time"] + rng.uniform(-1, 1, n)

    for name in segment["detectors"]:
        det = Detector(name)
        fp, fc = det.antenna_response(ra, dec, psi, t)
        dt = det.time_delay_from_geocenter(ra, dec, t)

        ifo = bilby.gw.detector.get_empty_interferometer(name)
        fp_b = np.array([ifo.antenna_response(ra[i], dec[i], t[i], psi[i], "plus") for i in range(n)])
        fc_b = np.array([ifo.antenna_response(ra[i], dec[i], t[i], psi[i], "cross") for i in range(n)])
        dt_b = np.array([ifo.time_delay_from_geocenter(ra[i], dec[i], t[i]) for i in range(n)])

        # Antenna patterns agree to ~1e-7 (lal stores the response tensor in float32).
        np.testing.assert_allclose(fp, fp_b, atol=1e-7)
        np.testing.assert_allclose(fc, fc_b, atol=1e-7)
        # Time delays agree to ~1e-12 s.
        np.testing.assert_allclose(dt, dt_b, atol=1e-12)


@requires_lal
def test_detector_is_picklable():
    import pickle

    from hyperwave.detectors.geometry import Detector

    det = Detector("L1")
    restored = pickle.loads(pickle.dumps(det))
    assert restored.prefix == "L1"
    np.testing.assert_array_equal(restored.vertex, det.vertex)
