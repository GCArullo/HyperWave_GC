"""Frequency-domain distribution likelihoods with optional GPU support."""

from __future__ import annotations

import numpy as np

from ..backends import gpu_backend_available
from .base import BaseLikelihood


class LogLike(BaseLikelihood):
    """Hyperbolic likelihoods for data-only frequency-domain analyses."""

    def __init__(
        self,
        data,
        f,
        ifos_list,
        ddims=True,
        nsegs=1,
        gpu=False,
        infs=-1e300,
        freq_domain=True,
    ):
        self._init_backend(gpu=gpu)

        self._nchannels = len(ifos_list)
        self.ifos = list(ifos_list)
        self._f = np.asarray(f)
        self.df = self._f[1] - self._f[0]
        self.data = self._backend.asarray(data)
        self.freq_domain = bool(freq_domain)

        if self.freq_domain:
            self._full_channels = 2 * self._nchannels
            self.yy_noise = self._get_yy_noise()
        else:
            self._full_channels = self._nchannels
            self.yy_noise = self.data**2

        self._d = self._full_channels
        self._lam = (self._d + 1) / 2
        self._C0 = ((1 - self._d) / 2) * np.log(2.0 * np.pi)

        self._ddims = bool(ddims)
        self._nsegs = int(nsegs)
        self._inf = infs
        self._logfreq = np.log10(self._f)
        self._ndims = self._nsegs if self._ddims else 1
        self.hyperbolic = self.hyperbolic2D if self._ddims else self.hyperbolic1D

        self._segi, self._Nd, self._fb = self._build_segments(self._f, self._nsegs)

    def _get_yy_noise(self):
        yy = self.data.conj() * self.data
        if self._nchannels == 1:
            syy = 4.0 * self.df * yy
        else:
            syy = 4.0 * self.df * self.xp.sum(yy, axis=0)
        return self.xp.abs(syy)

    def hyperbolic1D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(10.0 ** theta[:, : self._ndims])[:, 0]
        ratio = self._backend.asarray(10.0 ** theta[:, self._ndims :])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            ratio_i = ratio[:, i]
            delta_i = alpha * ratio_i
            alpha_delta_i = alpha * delta_i

            log_kappa = self._backend.xp.log(self._backend.kve(self._lam, alpha_delta_i)) - alpha_delta_i
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic2D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(10.0 ** theta[:, : self._ndims])
        ratio = self._backend.asarray(10.0 ** theta[:, self._ndims :])
        delta = alpha * ratio

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.xp.log(self._backend.kve(self._lam, alpha_delta_i)) - alpha_delta_i
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic_classic1D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(theta[:, : self._ndims])[:, 0]
        delta = self._backend.asarray(theta[:, self._ndims :])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            delta_i = delta[:, i]
            alpha_delta_i = alpha * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic_classic2D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(theta[:, : self._ndims])
        delta = self._backend.asarray(theta[:, self._ndims :])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()


# Backwards-compatible alias (the class was historically lowercase).
loglike = LogLike


def interpolate_fncn(x, y=None, kind="akima"):
    """
    Interpolation helper kept for backward compatibility.

    ``kind`` is kept for backward compatibility. Only Akima interpolation is
    currently supported.
    """
    from scipy.interpolate import Akima1DInterpolator

    if y is None:
        raise ValueError("interpolate_fncn requires both x and y arrays.")

    if kind.lower() != "akima":
        raise ValueError("Only kind='akima' is currently supported.")

    return Akima1DInterpolator(x, y)


__all__ = ["LogLike", "loglike", "gpu_backend_available", "interpolate_fncn"]
