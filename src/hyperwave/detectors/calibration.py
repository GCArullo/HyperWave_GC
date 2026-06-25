"""Detector calibration response curves for marginalized GW likelihoods.

HyperWave handles calibration uncertainty by *marginalizing* over a bank of
precomputed detector response curves rather than sampling spline nodes (which
would add ``2 * n_nodes * n_ifo`` dimensions to the sampler). This mirrors the
LVK calibration-marginalization scheme (e.g. bilby's
``calibration_marginalization``): a fixed set of curves ``C_k(f)`` is drawn once
from the calibration prior, and the ``*_calmarg`` methods of
:class:`~hyperwave.likelihoods.GWLikelihoods` numerically marginalize over them.

The cubic-spline calibration model is implemented natively on the NumPy/CuPy
backend (:class:`SplineCalibration`) so curve generation needs **no** bilby at
runtime. The algorithm reproduces the LVK / bilby cubic spline
(``C(f) = (1 + dA)(2 + i dphi) / (2 - i dphi)``, log-spaced nodes, the
second-derivative system of LIGO-T2300140) bit-for-bit; bilby is only used as a
reference in the test-suite.
"""

from __future__ import annotations

import numpy as np

from ..backends import get_array_backend


def _nodes_to_spline_coefficients(n_points):
    """Matrix mapping node values to natural-cubic-spline second derivatives.

    Port of bilby's ``CubicSpline._setup_spline_coefficients`` (Eq. 9 of
    LIGO-T2300140). Small (``n_points x n_points``), built on the host.
    """
    tmp1 = np.zeros((n_points, n_points))
    tmp1[0, 0] = -1
    tmp1[0, 1] = 2
    tmp1[0, 2] = -1
    tmp1[-1, -3] = -1
    tmp1[-1, -2] = 2
    tmp1[-1, -1] = -1
    for i in range(1, n_points - 1):
        tmp1[i, i - 1] = 1 / 6
        tmp1[i, i] = 2 / 3
        tmp1[i, i + 1] = 1 / 6
    tmp2 = np.zeros((n_points, n_points))
    for i in range(1, n_points - 1):
        tmp2[i, i - 1] = 1
        tmp2[i, i] = -2
        tmp2[i, i + 1] = 1
    return np.linalg.solve(tmp1, tmp2)


def _spline_basis_matrix(log_frequencies, log_nodes):
    """Linear map ``B`` such that ``delta(f) = B @ node_values``.

    Because the cubic spline is linear in its node values (for fixed nodes and
    evaluation grid), the whole evaluation collapses to one matrix ``B`` of
    shape ``(n_freq, n_nodes)``. This is what keeps the calibration factor a
    cheap, fully-batched matmul on the GPU. Reproduces bilby's
    ``CubicSpline.get_calibration_factor`` evaluation.
    """
    n = log_nodes.size
    delta_log = log_nodes[1] - log_nodes[0]
    x = (log_frequencies - log_nodes[0]) / delta_log
    prev = np.clip(np.floor(x).astype(int), 0, n - 2)
    nxt = prev + 1
    b = x - prev
    a = 1 - b
    c = (a ** 3 - a) / 6
    d = (b ** 3 - b) / 6

    M = _nodes_to_spline_coefficients(n)
    B = np.zeros((log_frequencies.size, n))
    rows = np.arange(log_frequencies.size)
    B[rows, prev] += a
    B[rows, nxt] += b
    B += c[:, None] * M[prev, :] + d[:, None] * M[nxt, :]
    return B


class SplineCalibration:
    """Native (NumPy/CuPy) cubic-spline detector calibration model.

    Parameters
    ----------
    frequency_array : array-like
        The (masked) analysis frequency grid, i.e. the same ``f`` the likelihood
        uses. Values must lie within ``[minimum_frequency, maximum_frequency]``.
    n_nodes : int
        Number of spline nodes (>= 4), log-spaced over the band.
    minimum_frequency, maximum_frequency : float, optional
        Node band edges; default to ``frequency_array[0]`` / ``[-1]``.
    gpu : bool
        Build the basis matrix on CuPy when available.
    """

    def __init__(self, frequency_array, n_nodes=10, minimum_frequency=None,
                 maximum_frequency=None, gpu=False):
        if n_nodes < 4:
            raise ValueError("Cubic spline calibration requires at least 4 nodes.")
        self._backend = get_array_backend(gpu=gpu)
        self.xp = self._backend.xp

        f = np.asarray(frequency_array, dtype=float)
        fmin = float(f[0]) if minimum_frequency is None else float(minimum_frequency)
        fmax = float(f[-1]) if maximum_frequency is None else float(maximum_frequency)
        self.n_nodes = int(n_nodes)
        self.minimum_frequency = fmin
        self.maximum_frequency = fmax
        self.log_nodes = np.linspace(np.log10(fmin), np.log10(fmax), n_nodes)

        B = _spline_basis_matrix(np.log10(f), self.log_nodes)  # (n_freq, n_nodes)
        self.basis = self._backend.asarray(B)
        self._n_freq = f.size

    def factor(self, amplitude_nodes, phase_nodes):
        """Calibration factor ``C(f)`` from node values.

        ``amplitude_nodes`` / ``phase_nodes`` have shape ``(..., n_nodes)``;
        returns ``(..., n_freq)`` complex. ``dA`` and ``dphi`` are each linear in
        the nodes (``@ basis.T``); the factor combines them as
        ``(1 + dA)(2 + i dphi)/(2 - i dphi)``.
        """
        a = self._backend.asarray(amplitude_nodes)
        p = self._backend.asarray(phase_nodes)
        delta_amplitude = a @ self.basis.T
        delta_phase = p @ self.basis.T
        return (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)

    def draw_bank(self, n_curves, amplitude_sigma, phase_sigma, seed=None):
        """Draw ``n_curves`` response curves from constant Gaussian node priors.

        Returns ``(n_curves, n_freq)`` complex on the active backend.
        """
        rng = np.random.default_rng(seed)
        amp = rng.normal(0.0, amplitude_sigma, size=(n_curves, self.n_nodes))
        phase = rng.normal(0.0, phase_sigma, size=(n_curves, self.n_nodes))
        return self.factor(amp, phase)

    def draw_bank_from_envelope(self, envelope_file, n_curves, seed=None):
        """Draw curves from an LVK calibration envelope file (no bilby).

        Envelope columns: ``freq median_amp median_phase -1sigma_amp
        -1sigma_phase +1sigma_amp +1sigma_phase``. Per-node Gaussian
        ``(mu, sigma)`` are interpolated (in log-frequency) from the envelope,
        matching bilby's :meth:`CalibrationPriorDict.from_envelope_file`.
        """
        from scipy.interpolate import InterpolatedUnivariateSpline

        data = np.loadtxt(envelope_file).T
        log_f = np.log10(data[0])
        amp_median = data[1] - 1.0
        phase_median = data[2]
        amp_sigma = (data[5] - data[3]) / 2.0
        phase_sigma = (data[6] - data[4]) / 2.0

        def at_nodes(values):
            return InterpolatedUnivariateSpline(log_f, values)(self.log_nodes)

        amp_mu, amp_sd = at_nodes(amp_median), at_nodes(amp_sigma)
        phase_mu, phase_sd = at_nodes(phase_median), at_nodes(phase_sigma)

        rng = np.random.default_rng(seed)
        amp = rng.normal(amp_mu, amp_sd, size=(n_curves, self.n_nodes))
        phase = rng.normal(phase_mu, phase_sd, size=(n_curves, self.n_nodes))
        return self.factor(amp, phase)


def make_calibration_bank(
    ifo_names,
    frequency_array,
    n_curves=1000,
    n_nodes=10,
    amplitude_sigma=0.05,
    phase_sigma=0.05,
    envelope_files=None,
    seed=None,
    gpu=False,
):
    """Draw a bank of calibration response curves, one stack per detector.

    Native NumPy/CuPy implementation (no bilby at runtime).

    Parameters
    ----------
    ifo_names : sequence of str
        Detector labels, e.g. ``["H1", "L1"]``. The order fixes axis 0 of the
        returned array and must match the ``ifos_list`` / ``data`` ordering of
        the likelihood the bank is handed to.
    frequency_array : array-like
        The (masked) analysis frequency grid the likelihood uses.
    n_curves : int
        Number of response curves per detector (the marginalization sum runs
        over these). 1000 is the common LVK choice.
    n_nodes : int
        Number of cubic-spline nodes (>= 4), log-spaced over the band.
    amplitude_sigma, phase_sigma : float or dict
        Gaussian 1-sigma uncertainty in fractional amplitude and in phase
        (radians). Float = same for all detectors; dict = per detector. Ignored
        for a detector that has an entry in ``envelope_files``.
    envelope_files : dict, optional
        ``{detector_name: path}`` to LVK calibration envelope files; when given
        for a detector, its prior is built from the envelope instead of the
        constant ``*_sigma`` values.
    seed : int, optional
        Base seed; detector ``j`` uses ``seed + j`` so detectors are decorrelated
        yet reproducible.
    gpu : bool
        Return the bank on CuPy when available.

    Returns
    -------
    array-like
        Complex array of shape ``(n_ifo, n_curves, n_freq)`` on the active
        backend. Curve index ``k`` is a *joint* calibration sample across
        detectors (same index in every detector), matching the LVK convention.
    """
    spline = SplineCalibration(frequency_array, n_nodes=n_nodes, gpu=gpu)
    envelope_files = envelope_files or {}

    def _per_ifo(value, name):
        return value[name] if isinstance(value, dict) else value

    curves = []
    for j, name in enumerate(ifo_names):
        sub_seed = None if seed is None else int(seed) + j
        if name in envelope_files:
            curves.append(spline.draw_bank_from_envelope(
                envelope_files[name], n_curves, seed=sub_seed))
        else:
            curves.append(spline.draw_bank(
                n_curves,
                _per_ifo(amplitude_sigma, name),
                _per_ifo(phase_sigma, name),
                seed=sub_seed,
            ))
    return spline.xp.stack(curves, axis=0)


__all__ = ["SplineCalibration", "make_calibration_bank"]
