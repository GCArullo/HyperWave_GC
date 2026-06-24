"""Likelihoods for GW inference with optional GPU acceleration."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..backends import gpu_backend_available
from ..detectors.calibration import read_calibration_file
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
        calibration_marginalization=False,
        calibration_draws=None,
        calibration_lookup_table=None,
        number_of_response_curves=1000,
        starting_index=0,
        calibration_correction_type=None,
        calibration_chunk_size=64,
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
        self.calibration_marginalization = bool(calibration_marginalization)
        self.calibration_chunk_size = int(calibration_chunk_size)
        self.number_of_response_curves = int(number_of_response_curves)
        self.starting_index = int(starting_index)
        self.calibration_draws = None
        if self.calibration_marginalization:
            self._setup_calibration_marginalization(
                calibration_draws=calibration_draws,
                calibration_lookup_table=calibration_lookup_table,
                correction_type=calibration_correction_type,
            )

    def _alpha_columns(self, theta):
        alpha = theta[:, self._wfdims:self._hdims]
        if self._ddims:
            return alpha
        return alpha[:, :1]

    def _get_yy_noise(self):
        return self.inner_product(self.data, self.data, psd=self.psd)

    def _setup_calibration_marginalization(
        self, calibration_draws=None, calibration_lookup_table=None, correction_type=None
    ):
        if not self._batched_template:
            raise ValueError("Calibration marginalization requires make_injections_to_ifo_batch.")

        low_template = getattr(self._template, "template", self._template)
        if any(str(name).startswith("recalib") for name in self._template.parameters):
            raise ValueError(
                "Calibration marginalization uses response-curve draws, so calibration "
                "parameters should not be part of template.parameters."
            )
        static_parameters = getattr(low_template, "static_parameters", {})
        if any(str(name).startswith("recalib") for name in static_parameters):
            raise ValueError(
                "Calibration marginalization uses response-curve draws, so calibration "
                "parameters should not be fixed in static_parameters."
            )
        for model in getattr(low_template, "calibration_models", []):
            if model is not None and getattr(model, "name", None) != "none":
                raise ValueError(
                    "Calibration marginalization should be used with identity template "
                    "calibration models to avoid double application."
                )
        if calibration_draws is None and calibration_lookup_table is None:
            raise ValueError(
                "calibration_marginalization requires calibration_draws or "
                "calibration_lookup_table."
            )

        if calibration_draws is None:
            calibration_draws = {}
            for name in self.ifos:
                if isinstance(calibration_lookup_table, dict):
                    filename = calibration_lookup_table[name]
                else:
                    filename = calibration_lookup_table
                curves, _ = read_calibration_file(
                    filename=filename,
                    frequency_array=self._f,
                    number_of_response_curves=self.number_of_response_curves,
                    starting_index=self.starting_index,
                    correction_type=correction_type,
                )
                calibration_draws[name] = curves

        arrays = []
        n_curves = None
        for ii, name in enumerate(self.ifos):
            if isinstance(calibration_draws, dict):
                draws = calibration_draws[name]
            else:
                draws = calibration_draws[ii]
            draws = np.asarray(draws, dtype=complex)
            if draws.ndim == 1:
                draws = draws[None, :]
            if draws.ndim != 2 or draws.shape[1] != self._f.size:
                raise ValueError(
                    f"Calibration draws for {name} must have shape (n_curves, {self._f.size}); "
                    f"got {draws.shape}."
                )
            if n_curves is None:
                n_curves = draws.shape[0]
            elif draws.shape[0] != n_curves:
                raise ValueError("All detectors must use the same number of calibration curves.")
            arrays.append(draws)

        self.number_of_response_curves = int(n_curves)
        self.calibration_draws = self._backend.asarray(np.stack(arrays, axis=0))
        self._calibration_log_norm = np.log(self.number_of_response_curves)

    def _signal_batch(self, p):
        return self._backend.asarray(self._template.make_injections_to_ifo_batch(p))

    def _calibrated_yy_chunks(self, signal):
        chunk = max(1, self.calibration_chunk_size)
        n_curves = self.number_of_response_curves
        for start in range(0, n_curves, chunk):
            stop = min(start + chunk, n_curves)
            calibration = self.calibration_draws[:, start:stop, :]
            residual = (
                self.data[None, :, None, :]
                - signal[:, :, None, :] * calibration[None, :, :, :]
            )
            yy = (residual.conj() * residual) / self.psd[None, :, None, :]
            syy = self.df * self.xp.sum(yy, axis=1)
            yield 4.0 * self.xp.real(syy)

    def _logmeanexp_draws(self, logl):
        max_logl = self.xp.max(logl, axis=1, keepdims=True)
        centered = self.xp.exp(logl - max_logl)
        return (max_logl[:, 0] + self.xp.log(self.xp.mean(centered, axis=1))).real

    def _combine_calibration_logl(self, chunks):
        return self._logmeanexp_draws(self.xp.concatenate(chunks, axis=1))

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
        signal = self._signal_batch(p)
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
        if self.calibration_marginalization:
            signal = self._signal_batch(theta[:, : self._wfdims])
            chunks = [
                -0.5 * self.xp.sum(yy, axis=-1).real
                for yy in self._calibrated_yy_chunks(signal)
            ]
            likelihood = self._combine_calibration_logl(chunks)
        else:
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
        if self.calibration_marginalization:
            return self._whittle_level_calibration_marginalized(theta)

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
        if self.calibration_marginalization:
            return self._hyperbolic_calibration_marginalized(theta, classic=False)

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
        if self.calibration_marginalization:
            return self._hyperbolic_calibration_marginalized(theta, classic=True)

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

    def _whittle_level_calibration_marginalized(self, theta):
        signal = self._signal_batch(theta[:, : self._wfdims])
        chunks = []
        for yy in self._calibrated_yy_chunks(signal):
            likelihood = self.xp.zeros((theta.shape[0], yy.shape[1], len(self._segi)))
            for i, si in enumerate(self._segi):
                level = self._backend.asarray(10.0 ** theta[:, self._wfdims + i])
                term = -0.5 * yy[:, :, si] / level[:, None, None]
                term = term - self._nchannels * self.xp.log(level)[:, None, None]
                if self.whiten is not None:
                    term = term + 2 * self.logwhiten[si][None, None, :]
                likelihood[:, :, i] = self.xp.sum(term, axis=-1).real
            chunks.append(self.xp.sum(likelihood, axis=-1))

        likelihood = self._combine_calibration_logl(chunks)
        likelihood = np.nan_to_num(
            self._prepare_outputs(likelihood),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def _hyperbolic_calibration_marginalized(self, theta, classic):
        alpha = self._backend.asarray(self._alpha_columns(theta))
        tail = self._backend.asarray(theta[:, self._hdims :])
        signal = self._signal_batch(theta[:, : self._wfdims])

        chunks = []
        for yy in self._calibrated_yy_chunks(signal):
            likelihood = self.xp.zeros((theta.shape[0], yy.shape[1], len(self._segi)))
            for i, si in enumerate(self._segi):
                alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
                delta_i = tail[:, i] if classic else alpha_i * tail[:, i]
                alpha_delta_i = alpha_i * delta_i

                log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
                term_sqrt = self.xp.sum(
                    self.xp.sqrt(delta_i[:, None, None] ** 2 + yy[:, :, si]).real,
                    axis=-1,
                )
                term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
                term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa
                likelihood[:, :, i] = (
                    self._Nd[i] * (term_lambda + term_rest)[:, None]
                    - alpha_i[:, None] * term_sqrt
                )
            chunks.append(self.xp.sum(likelihood, axis=-1))

        likelihood = self._combine_calibration_logl(chunks)
        likelihood = self.xp.nan_to_num(
            likelihood,
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()


__all__ = ["GWLikelihoods", "gpu_backend_available"]
