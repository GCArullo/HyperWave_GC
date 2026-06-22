"""Likelihoods for GW inference with optional GPU acceleration."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..backends import gpu_backend_available
from .base import BaseLikelihood


class GWLikelihoods(BaseLikelihood):
    """
    Log-likelihoods for GW parameter estimation.

    The waveform generation itself is still driven by the template object, which
    is typically CPU-side. When ``gpu=True`` HyperWave moves the array-heavy
    residual and likelihood algebra to CuPy while keeping a safe NumPy fallback.
    """

    def __init__(
        self,
        data,
        f,
        ifos_list,
        noise,
        template,
        ddims=True,
        nsegs=4,
        gpu=False,
        infs=-1e300,
        cpu_cores=32,
    ):
        self._init_backend(gpu=gpu)

        self._template = template
        if self._template.parameters is None:
            raise TypeError("Template must define its parameter list.")

        # Prefer the batched template path (one call for all walkers) when the
        # template exposes it; this avoids the per-walker joblib loop entirely
        # and is the fast path for the new lal/ml4gw backends.
        self._batched_template = callable(
            getattr(self._template, "make_injections_to_ifo_batch", None)
        )

        self._wfdims = len(self._template.parameters)
        self._nchannels = len(ifos_list)
        self.ifos = list(ifos_list)
        self._f = np.asarray(f)
        self.df = self._f[1] - self._f[0]
        self.data = self._backend.asarray(data)
        self.psd = self._backend.asarray(noise)

        self._d = 2 * self._nchannels
        self._lam = (self._d + 1) / 2
        self._C0 = ((1 - self._d) / 2) * np.log(2.0 * np.pi)

        self._ddims = bool(ddims)
        self._nsegs = int(nsegs)
        self._inf = infs
        self._logfreq = np.log10(self._f)
        self._ndims = self._wfdims + 2 * self._nsegs if self._ddims else self._wfdims + self._nsegs + 1
        self._hdims = self._wfdims + self._nsegs if self._ddims else self._wfdims + 1
        self.num_jobs = 1 if self._use_gpu else int(cpu_cores)

        self.whiten = None
        self.logwhiten = None
        self.yy = None
        self.yy_noise = -0.5 * self._get_yy_noise()

        self._segi, self._Nd, self._fb = self._build_segments(self._f, self._nsegs)

    def _alpha_columns(self, theta):
        alpha = theta[:, self._wfdims:self._hdims]
        if self._ddims:
            return alpha
        return alpha[:, :1]

    def _get_yy_noise(self):
        return self.inner_product(self.data, self.data, psd=self.psd)

    def inner_residual(self, theta):
        signal = self._template.make_injections_to_ifo(np.asarray(theta))
        residual = self._backend.zeros(self.data.shape, dtype=self.data.dtype)

        for ifo, channel in enumerate(self.ifos):
            residual[ifo, :] = self.data[ifo, :] - self._backend.asarray(signal[channel])

        yy = residual.conj() * residual
        yy = yy / self.psd
        syy = self.df * self.xp.sum(yy, axis=0)
        return 4.0 * self.xp.real(syy)

    def inner_residual_batch(self, p):
        """Vectorised noise-weighted residual for a batch ``p`` of shape ``(N, wfdims)``.

        Returns ``(N, n_freq)`` = ``4 Re[df * sum_ifo conj(r) r / Sn]`` with one
        batched template call (no per-walker loop).
        """
        signal = self._backend.asarray(self._template.make_injections_to_ifo_batch(p))
        residual = self.data[None, :, :] - signal  # (N, n_ifo, n_freq)
        yy = (residual.conj() * residual) / self.psd[None, :, :]
        syy = self.df * self.xp.sum(yy, axis=1)  # sum over detectors -> (N, n_freq)
        return 4.0 * self.xp.real(syy)

    def _get_yy(self, p):
        p = self._ensure_2d(p)
        if self._batched_template:
            return self.inner_residual_batch(p)
        if self._use_gpu or self.num_jobs <= 1:
            residuals = [self.inner_residual(p[walker, :]) for walker in range(p.shape[0])]
        else:
            residuals = Parallel(n_jobs=self.num_jobs)(
                delayed(self.inner_residual)(p[walker, :]) for walker in range(p.shape[0])
            )
        return self._backend.asarray(residuals)

    def gaussian(self, theta):
        theta = self._ensure_2d(theta)
        likelihood = -0.5 * self.xp.sum(self._get_yy(p=theta), axis=-1).real
        likelihood = np.nan_to_num(
            self._prepare_outputs(likelihood),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def whittle_level(self, theta):
        theta = self._ensure_2d(theta)
        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            level = self._backend.asarray(10.0 ** theta[:, self._wfdims + i])[:, None]
            const = 0.0 if self.whiten is None else 2 * self.logwhiten[si]
            likelihood[:, i] = self.xp.sum(
                -0.5 * self.yy[:, si] / level - self._nchannels * self.xp.log(level) + const,
                axis=-1,
            ).real

        likelihood = np.nan_to_num(
            self._prepare_outputs(self.xp.sum(likelihood, axis=-1)),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def hyperbolic(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(self._alpha_columns(theta))
        ratio = self._backend.asarray(theta[:, self._hdims :])
        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
            ratio_i = ratio[:, i]
            delta_i = alpha_i * ratio_i
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy[:, si]).real, axis=-1)
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

    def hyperbolic_classic(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(self._alpha_columns(theta))
        delta = self._backend.asarray(theta[:, self._hdims :])
        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy[:, si]).real, axis=-1)
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


__all__ = ["GWLikelihoods", "gpu_backend_available"]
