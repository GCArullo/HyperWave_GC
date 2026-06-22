"""Shared base class for HyperWave frequency-domain likelihoods.

Holds the machinery common to the hyperbolic / Gaussian likelihoods so each
concrete class only implements its own log-likelihood: device-backend
acquisition, batch-shape coercion, the noise-weighted inner product, and the
frequency-segment partition.
"""

from __future__ import annotations

import numpy as np

from ..backends import get_array_backend


class BaseLikelihood:
    """Common machinery for the frequency-domain likelihoods.

    Subclasses call :meth:`_init_backend` and set ``self.df`` (the frequency
    resolution) before using :meth:`inner_product`.
    """

    def _init_backend(self, gpu=False):
        """Acquire the array backend and expose ``xp`` / device flags."""
        self._backend = get_array_backend(gpu=gpu)
        self.xp = self._backend.xp
        self._use_gpu = self._backend.use_gpu
        self.backend_name = self._backend.name

    @staticmethod
    def _ensure_2d(theta):
        """Promote a 1-D parameter vector to a ``(1, ndim)`` batch."""
        theta = np.asarray(theta)
        if theta.ndim == 1:
            theta = theta[None, :]
        return theta

    def _prepare_outputs(self, out):
        """Return a host (NumPy) array, copying off the device when on GPU."""
        return self._backend.to_numpy(out) if self._use_gpu else np.asarray(out)

    def inner_product(self, x, y, psd=None):
        """Noise-weighted inner product ``4 Re[df * sum(conj(x) y / Sn)]``.

        Sums over the leading (frequency) axis. ``psd`` is the one-sided noise
        spectrum; omit it for an unweighted product.
        """
        yy = x.conj() * y
        if psd is not None:
            yy = yy / psd
        return 4.0 * self.xp.real(self.df * self.xp.sum(yy, axis=0))

    def _build_segments(self, f, nsegs):
        """Partition the band into ``nsegs`` contiguous frequency segments.

        Returns ``(segi, Nd, fb)``: per-segment index arrays, their sizes, and
        the segment edge frequencies.
        """
        f = np.asarray(f)
        if nsegs > 1:
            fb = np.linspace(f[0], f[-1], num=nsegs + 1, retstep=False)
        else:
            fb = [f[0], f[-1]]
        segi, Nd = [], []
        for i in range(nsegs):
            if nsegs == 1 or i == nsegs - 1:
                mask = np.logical_and(f >= fb[i], f <= fb[i + 1])
            else:
                mask = np.logical_and(f >= fb[i], f < fb[i + 1])
            indices = np.where(mask)[0]
            segi.append(indices)
            Nd.append(len(indices))
        return segi, Nd, fb


__all__ = ["BaseLikelihood"]
