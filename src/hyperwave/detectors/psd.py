"""Power spectral density container.

Replaces ``bilby.gw.detector.PowerSpectralDensity`` for the (small) feature set
HyperWave used: holding a PSD/ASD on a frequency grid, interpolating it onto an
analysis grid, and drawing a coloured Gaussian noise realisation that matches
bilby's normalisation.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .strain import create_white_noise


class PowerSpectralDensity:
    def __init__(self, frequency_array, psd_array):
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.psd_array = np.asarray(psd_array, dtype=float)
        self._interpolant = None

    @classmethod
    def from_asd(cls, frequency_array, asd_array):
        asd = np.asarray(asd_array, dtype=float)
        return cls(frequency_array, asd**2)

    @property
    def asd_array(self):
        return self.psd_array**0.5

    def _interp(self):
        if self._interpolant is None:
            self._interpolant = interp1d(
                self.frequency_array,
                self.psd_array,
                bounds_error=False,
                fill_value=np.inf,
            )
        return self._interpolant

    def power_spectral_density_interpolated(self, frequencies):
        """Interpolate the PSD onto ``frequencies`` (out of band -> ``inf``)."""
        return self._interp()(np.asarray(frequencies, dtype=float))

    def asd_interpolated(self, frequencies):
        return self.power_spectral_density_interpolated(frequencies) ** 0.5

    def get_noise_realisation(self, sampling_frequency, duration, rng=None):
        """Draw a coloured Gaussian FD noise realisation (bilby convention).

        Parameters
        ----------
        rng:
            A ``numpy.random.Generator``. Pass a seeded generator for
            reproducible noise.
        """
        if rng is None:
            rng = np.random.default_rng()
        white, freqs = create_white_noise(sampling_frequency, duration, rng)
        with np.errstate(invalid="ignore"):
            fd = self.power_spectral_density_interpolated(freqs) ** 0.5 * white
        out_of_band = (freqs < self.frequency_array.min()) | (freqs > self.frequency_array.max())
        fd[out_of_band] = 0.0 + 0.0j
        return fd, freqs


__all__ = ["PowerSpectralDensity"]
