"""Strain-data container and FFT helpers.

A lightweight replacement for ``bilby.gw.detector.strain_data`` that keeps the
exact numerical conventions so HyperWave results stay comparable:

* forward FFT (``nfft``): ``fd = rfft(td) / sampling_frequency`` (single-sided,
  units strain/Hz), with ``frequency_array = linspace(0, fs/2, n_freq)``
* inverse FFT (``infft``): ``td = irfft(fd) * sampling_frequency``
* data is Tukey-windowed before the forward FFT with ``roll_off = 0.2 s`` and
  ``alpha = 2 * roll_off / duration``; a model/injection FD strain set directly
  is **not** windowed
* coloured Gaussian noise uses ``norm = 0.5 * sqrt(duration)`` white noise with
  DC and Nyquist bins zeroed, then multiplied by ``sqrt(PSD)``
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey

DEFAULT_ROLL_OFF = 0.2


def nfft(time_domain_strain, sampling_frequency):
    """Single-sided normalised FFT (bilby ``nfft`` convention)."""
    fd = np.fft.rfft(time_domain_strain) / sampling_frequency
    freq = np.linspace(0.0, sampling_frequency / 2.0, len(fd))
    return fd, freq


def infft(frequency_domain_strain, sampling_frequency):
    """Inverse of :func:`nfft`."""
    return np.fft.irfft(frequency_domain_strain) * sampling_frequency


def frequency_array(sampling_frequency, duration):
    """Real-FFT frequency grid for the given segment."""
    n = int(round(duration * sampling_frequency))
    return np.linspace(0.0, sampling_frequency / 2.0, n // 2 + 1)


def create_white_noise(sampling_frequency, duration, rng):
    """Unit white noise matching ``bilby.core.utils.series.create_white_noise``."""
    freqs = frequency_array(sampling_frequency, duration)
    n_samples = int(round(duration * sampling_frequency))
    norm = 0.5 * duration**0.5
    re, im = rng.normal(0.0, norm, (2, len(freqs)))
    white = re + 1j * im
    white[0] = 0.0
    if n_samples % 2 == 0:
        white[-1] = 0.0
    return white, freqs


class StrainData:
    """Time- and frequency-domain strain for a single detector.

    Mirrors the bilby behaviour the HyperWave code relied on: the
    frequency-domain strain is computed lazily (and cached) from the
    Tukey-windowed time series, while an explicitly-assigned frequency-domain
    strain (a model waveform) is returned verbatim without windowing.
    """

    def __init__(self, sampling_frequency, duration, start_time=0.0, roll_off=DEFAULT_ROLL_OFF):
        self.sampling_frequency = float(sampling_frequency)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.roll_off = float(roll_off)
        self._time_domain_strain = None
        self._frequency_domain_strain = None

    # -- grids ------------------------------------------------------------
    @property
    def alpha(self):
        return 2.0 * self.roll_off / self.duration

    @property
    def frequency_array(self):
        return frequency_array(self.sampling_frequency, self.duration)

    @property
    def time_array(self):
        n = int(round(self.duration * self.sampling_frequency))
        return self.start_time + np.arange(n) / self.sampling_frequency

    def time_domain_window(self):
        return tukey(len(self._time_domain_strain), alpha=self.alpha)

    # -- setters ----------------------------------------------------------
    def set_from_time_domain_strain(self, time_domain_strain):
        self._time_domain_strain = np.asarray(time_domain_strain, dtype=float)
        self._frequency_domain_strain = None
        return self

    def set_from_frequency_domain_strain(self, frequency_domain_strain):
        self._frequency_domain_strain = np.asarray(frequency_domain_strain, dtype=complex)
        self._time_domain_strain = None
        return self

    def set_from_gwpy_timeseries(self, timeseries):
        """Store strain from a ``gwpy`` :class:`~gwpy.timeseries.TimeSeries`."""
        self.sampling_frequency = float(timeseries.sample_rate.value)
        self.duration = float(timeseries.duration.value)
        self.start_time = float(timeseries.t0.value)
        return self.set_from_time_domain_strain(np.asarray(timeseries.value, dtype=float))

    # -- views ------------------------------------------------------------
    @property
    def time_domain_strain(self):
        if self._time_domain_strain is not None:
            return self._time_domain_strain
        if self._frequency_domain_strain is not None:
            return infft(self._frequency_domain_strain, self.sampling_frequency)
        raise ValueError("No strain data set.")

    @property
    def frequency_domain_strain(self):
        if self._frequency_domain_strain is None:
            if self._time_domain_strain is None:
                raise ValueError("No strain data set.")
            window = self.time_domain_window()
            self._frequency_domain_strain, _ = nfft(
                self._time_domain_strain * window, self.sampling_frequency
            )
        return self._frequency_domain_strain

    @frequency_domain_strain.setter
    def frequency_domain_strain(self, value):
        self.set_from_frequency_domain_strain(value)

    def add_frequency_domain_signal(self, signal):
        """Add a (windowed-data-consistent) FD signal to the cached strain.

        Reproduces ``bilby``'s ``Interferometer.inject_signal`` side effect: the
        windowed data spectrum is realised (and cached) and the unwindowed model
        signal is added on top.
        """
        current = self.frequency_domain_strain  # realises + caches windowed FFT
        self._frequency_domain_strain = current + np.asarray(signal, dtype=complex)
        return self._frequency_domain_strain


__all__ = [
    "StrainData",
    "nfft",
    "infft",
    "frequency_array",
    "create_white_noise",
    "DEFAULT_ROLL_OFF",
]
