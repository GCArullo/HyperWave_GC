"""Detector geometry and antenna response.

This module replaces the parts of ``bilby.gw.detector`` that HyperWave used for
detector location, antenna patterns and geocentric time delays. It is backed by
``lal`` for the cached detector definitions (location vector and response
tensor) but performs the antenna-response and time-delay algebra in vectorised
NumPy so that a whole batch of sky positions can be evaluated in one call.

Conventions follow ``bilby``/``bilby_cython`` exactly (Nishizawa et al. 2009,
arXiv:0903.0528) so results are directly comparable:

* ``gmst = greenwich_mean_sidereal_time(t)`` (radians, via ``lal``)
* ``phi = ra - gmst``, ``theta = pi/2 - dec``
* wave-frame vectors ``m`` and ``n`` as in ``bilby_cython.geometry``
* ``F_plus  = D_ij (m_i m_j - n_i n_j)``
* ``F_cross = D_ij (m_i n_j + n_i m_j)``
* ``dt_geocenter = -(vertex . n_hat) / c``
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:  # lal is an optional-at-import dependency; only the LVK layer needs it
    import lal
except Exception as exc:  # pragma: no cover - depends on runtime environment
    lal = None
    _LAL_IMPORT_ERROR = exc
else:
    _LAL_IMPORT_ERROR = None

SPEED_OF_LIGHT = 299792458.0


def _require_lal():
    if lal is None:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "Detector geometry requires the optional 'lal' (lalsuite) dependency."
        ) from _LAL_IMPORT_ERROR
    return lal


def greenwich_mean_sidereal_time(gps_time):
    """Greenwich mean sidereal time in radians.

    Accepts a scalar or any array-like of GPS times and returns a value with the
    same shape. The per-sample value is computed with
    ``lal.GreenwichMeanSiderealTime`` which is bit-identical to the value
    ``bilby`` uses.
    """
    _require_lal()
    times = np.asarray(gps_time, dtype=float)
    flat = np.atleast_1d(times).ravel()
    out = np.fromiter(
        (lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(t))) for t in flat),
        dtype=float,
        count=flat.size,
    )
    if times.ndim == 0:
        return float(out[0])
    return out.reshape(times.shape)


def _wave_frame_vectors(phi, theta, psi):
    """Return the ``m`` and ``n`` wave-frame vectors, shape ``(..., 3)``."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    m = np.stack(
        [
            -ct * cphi * spsi + sphi * cpsi,
            -ct * sphi * spsi - cphi * cpsi,
            st * spsi,
        ],
        axis=-1,
    )
    n = np.stack(
        [
            -ct * cphi * cpsi - sphi * spsi,
            -ct * sphi * cpsi + cphi * spsi,
            st * cpsi,
        ],
        axis=-1,
    )
    return m, n


class Detector:
    """A single interferometer's geometry and antenna response.

    Parameters
    ----------
    prefix:
        Two-character detector prefix understood by ``lal`` (``"H1"``, ``"L1"``,
        ``"V1"``, ``"K1"``, ``"G1"`` ...).
    """

    def __init__(self, prefix: str):
        _require_lal()
        self.prefix = str(prefix)
        cached = lal.cached_detector_by_prefix.get(self.prefix)
        if cached is None:
            raise KeyError(
                f"Unknown detector prefix {self.prefix!r}. "
                f"Known prefixes: {sorted(lal.cached_detector_by_prefix)}"
            )
        self._lal = cached
        #: geocentric vertex position in metres, shape ``(3,)``
        self.vertex = np.asarray(cached.location, dtype=float)
        #: detector response tensor, shape ``(3, 3)`` (lal stores float32; cast up)
        self.detector_tensor = np.asarray(cached.response, dtype=float)

    # -- antenna response -------------------------------------------------
    def antenna_response(self, ra, dec, psi, gps_time):
        """Return ``(F_plus, F_cross)`` for the given sky position(s).

        All arguments broadcast against each other; the return arrays have the
        broadcast shape (a scalar input yields shape ``(1,)``).
        """
        ra = np.atleast_1d(np.asarray(ra, dtype=float))
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        psi = np.atleast_1d(np.asarray(psi, dtype=float))
        gmst = np.atleast_1d(greenwich_mean_sidereal_time(gps_time))
        ra, dec, psi, gmst = np.broadcast_arrays(ra, dec, psi, gmst)

        phi = ra - gmst
        theta = np.pi / 2.0 - dec
        m, n = _wave_frame_vectors(phi, theta, psi)

        d = self.detector_tensor
        fp = np.einsum("...i,...j,ij->...", m, m, d) - np.einsum("...i,...j,ij->...", n, n, d)
        fc = np.einsum("...i,...j,ij->...", m, n, d) + np.einsum("...i,...j,ij->...", n, m, d)
        return fp, fc

    # -- time delay -------------------------------------------------------
    def time_delay_from_geocenter(self, ra, dec, gps_time):
        """Geocentric arrival-time delay (seconds), broadcast over inputs."""
        ra = np.atleast_1d(np.asarray(ra, dtype=float))
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        gmst = np.atleast_1d(greenwich_mean_sidereal_time(gps_time))
        ra, dec, gmst = np.broadcast_arrays(ra, dec, gmst)

        phi = ra - gmst
        theta = np.pi / 2.0 - dec
        st = np.sin(theta)
        n_hat = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)
        return -(n_hat @ self.vertex) / SPEED_OF_LIGHT

    # lal.Detector is not picklable, so reconstruct from the prefix (and reuse
    # the module-level cache). This keeps templates picklable for joblib/MPI.
    def __reduce__(self):
        return (get_detector, (self.prefix,))

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"Detector({self.prefix!r})"


@lru_cache(maxsize=None)
def get_detector(prefix: str) -> Detector:
    """Return a cached :class:`Detector` for ``prefix``."""
    return Detector(prefix)


__all__ = [
    "Detector",
    "get_detector",
    "greenwich_mean_sidereal_time",
    "SPEED_OF_LIGHT",
]
