"""LISA A/E/T integration helpers built around ``lisatools``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...likelihoods import GWLikelihoods

AET_CHANNELS = ("A", "E", "T")


def lisatools_available() -> bool:
    """Return ``True`` when ``lisatools``'s A/E/T containers import cleanly.

    The import is done lazily (not at module load) on purpose: importing
    ``lisatools.analysiscontainer`` pulls in compiled kernels that ``SIGILL`` on
    CPUs without the instruction set the wheel was built for (e.g. AVX-512 on a
    Zen3 node). HyperWave's A/E/T adapter only needs those classes when a *user*
    passes a real ``AnalysisContainer``/``DataResidualArray`` — the raw-array and
    bbhx/gbgpu paths duck-type instead — so the bridge must stay importable
    without them. ``BaseException`` is caught because a ``SIGILL`` surfaces here
    as a fatal signal, not an ``ImportError``.
    """
    try:  # pragma: no cover - depends on optional dep + CPU features
        from lisatools.analysiscontainer import AnalysisContainer  # noqa: F401
        from lisatools.datacontainer import DataResidualArray  # noqa: F401
    except BaseException:
        return False
    return True


def _get_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _channels_first_2d(array: Any, nchannels: int, label: str) -> np.ndarray:
    arr = np.asarray(array)

    if arr.ndim == 1:
        if nchannels != 1:
            raise ValueError(f"{label} is one-dimensional, but {nchannels} channels were requested.")
        return arr[None, :]

    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D array, got shape {arr.shape}.")

    if arr.shape[0] == nchannels:
        return arr

    if arr.shape[1] == nchannels:
        return arr.T

    raise ValueError(
        f"Could not interpret {label} with shape {arr.shape} as {nchannels} channels by frequency."
    )


def _channels_first_3d(array: Any, nchannels: int, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"{label} must be a 3D array, got shape {arr.shape}.")

    if arr.shape[:2] == (nchannels, nchannels):
        return arr

    if arr.shape[1:] == (nchannels, nchannels):
        return np.moveaxis(arr, 0, -1)

    raise ValueError(
        f"Could not interpret {label} with shape {arr.shape} as a channel covariance cube."
    )


def _extract_diagonal_noise(sensitivity: Any, nchannels: int) -> np.ndarray:
    raw = _get_attr(sensitivity, "sens_mat")
    if raw is None:
        raw = sensitivity

    arr = np.asarray(raw)
    if arr.ndim <= 2:
        return _channels_first_2d(arr, nchannels=nchannels, label="sensitivity")

    matrix = _channels_first_3d(arr, nchannels=nchannels, label="sensitivity")
    offdiag = matrix.copy()
    inds = np.arange(nchannels)
    offdiag[inds, inds, :] = 0.0

    if not np.allclose(offdiag, 0.0):
        raise ValueError(
            "HyperWave's current LISA adapter only supports diagonal channel PSDs. "
            "Use A/E/T inputs or add full covariance support before using XYZ."
        )

    return np.diagonal(matrix, axis1=0, axis2=1).T


def prepare_lisa_aet_inputs(
    data: Any,
    sensitivity: Any | None = None,
    freqs: Sequence[float] | None = None,
    channels: Sequence[str] = AET_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Convert ``lisatools`` containers into the arrays expected by HyperWave.

    Parameters
    ----------
    data
        A ``lisatools.analysiscontainer.AnalysisContainer``,
        ``lisatools.datacontainer.DataResidualArray``, or a raw array-like object.
    sensitivity
        A ``lisatools`` sensitivity matrix or raw PSD array. If ``data`` is an
        ``AnalysisContainer`` and ``sensitivity`` is omitted, the container's
        sensitivity matrix is used automatically.
    freqs
        Frequency array. If omitted, HyperWave tries to read ``f_arr`` or
        ``frequency_arr`` from the input objects.
    channels
        Channel names, typically ``("A", "E", "T")``.
    """
    channel_names = list(channels)
    nchannels = len(channel_names)

    if sensitivity is None:
        sensitivity = _get_attr(data, "sens_mat", "sensitivity_matrix")

    payload = _get_attr(data, "data_res_arr")
    if payload is None:
        payload = data

    data_arr = _channels_first_2d(payload, nchannels=nchannels, label="data")

    if freqs is None:
        freqs = _get_attr(data, "f_arr", "frequency_arr")
    if freqs is None:
        freqs = _get_attr(sensitivity, "frequency_arr", "f_arr")
    if freqs is None:
        raise ValueError("Could not determine the frequency array. Pass `freqs=` explicitly.")

    freq_arr = np.asarray(freqs)
    if freq_arr.ndim != 1:
        raise ValueError(f"Expected a 1D frequency array, got shape {freq_arr.shape}.")

    if sensitivity is None:
        raise ValueError(
            "Could not determine the LISA sensitivity matrix. Pass `sensitivity=` or an AnalysisContainer."
        )

    noise_arr = _extract_diagonal_noise(sensitivity, nchannels=nchannels)

    if data_arr.shape[-1] != freq_arr.shape[0]:
        raise ValueError(
            f"Data shape {data_arr.shape} does not match frequency array length {freq_arr.shape[0]}."
        )
    if noise_arr.shape[-1] != freq_arr.shape[0]:
        raise ValueError(
            f"Sensitivity shape {noise_arr.shape} does not match frequency array length {freq_arr.shape[0]}."
        )

    return data_arr, freq_arr, noise_arr, channel_names


@dataclass
class LISAAETTemplate:
    """
    Adapter that makes a ``lisatools``-style signal model usable by HyperWave.

    The wrapped ``signal_model`` (single source) can return either:

    - a ``DataResidualArray``
    - a dictionary keyed by channel name
    - a raw array with shape ``(nchannels, nfreq)``

    Pass ``batch_signal_model`` to enable the **vectorised** path: HyperWave's
    ``GWLikelihoods`` probes for ``make_injections_to_ifo_batch`` and, when
    present, generates the whole walker population in a single generator call
    (one bbhx/gbgpu call for all walkers) instead of looping per walker — the fast
    path for both Eryn and pocoMC, which evaluate the likelihood vectorised. The
    batch model receives one array per parameter (length ``N``) and must return
    ``(N, nchannels, nfreq)`` (or a channel-keyed dict of ``(N, nfreq)``). When no
    batch model is given the batched entry point is hidden so the likelihood falls
    back to the per-walker path.
    """

    parameters: Sequence[str]
    signal_model: Callable[..., Any]
    channels: Sequence[str] = AET_CHANNELS
    static_parameters: Mapping[str, Any] | None = None
    call_mode: str = "kwargs"
    batch_signal_model: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        # Only expose make_injections_to_ifo_batch when a batch model exists, so
        # GWLikelihoods picks the vectorised path iff it is actually available.
        if self.batch_signal_model is None:
            self.make_injections_to_ifo_batch = None

    def _batch_parameter_arrays(self, thetas):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        n = thetas.shape[0]
        params = {name: thetas[:, i] for i, name in enumerate(self.parameters)}
        if self.static_parameters:
            for key, value in self.static_parameters.items():
                params[key] = np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value)
        return params, n

    def make_injections_to_ifo_batch(self, thetas) -> np.ndarray:
        """Vectorised signal for a batch of walkers -> ``(N, nchannels, nfreq)``."""
        params, n = self._batch_parameter_arrays(thetas)
        if self.call_mode == "dict":
            out = self.batch_signal_model(params)
        elif self.call_mode == "vector":
            out = self.batch_signal_model(np.atleast_2d(np.asarray(thetas, dtype=float)))
        else:  # kwargs
            out = self.batch_signal_model(**params)

        channel_names = list(self.channels)
        nch = len(channel_names)
        if isinstance(out, Mapping):
            return np.stack([np.asarray(out[c]) for c in channel_names], axis=1)
        arr = np.asarray(out)
        if arr.ndim == 3 and arr.shape[1] == nch:
            return arr
        if arr.ndim == 3 and arr.shape[0] == nch:  # (nch, N, nfreq) -> (N, nch, nfreq)
            return np.moveaxis(arr, 0, 1)
        raise ValueError(
            f"batch_signal_model returned shape {arr.shape}; expected (N, {nch}, nfreq)."
        )

    def parameter_dict(self, theta: Sequence[float]) -> dict[str, Any]:
        params = {key: value for key, value in zip(self.parameters, np.asarray(theta))}
        if self.static_parameters:
            params.update(self.static_parameters)
        return params

    def _call_signal_model(self, theta: Sequence[float]) -> Any:
        params = self.parameter_dict(theta)

        if self.call_mode == "kwargs":
            return self.signal_model(**params)
        if self.call_mode == "dict":
            return self.signal_model(params)
        if self.call_mode == "vector":
            return self.signal_model(np.asarray(theta))

        raise ValueError(f"Unknown call_mode '{self.call_mode}'.")

    def make_injections_to_ifo(self, theta: Sequence[float]) -> dict[str, np.ndarray]:
        out = self._call_signal_model(theta)
        channel_names = list(self.channels)

        if isinstance(out, Mapping):
            return {channel: np.asarray(out[channel]) for channel in channel_names}

        payload = _get_attr(out, "data_res_arr")
        if payload is None:
            payload = out

        signal_arr = _channels_first_2d(payload, nchannels=len(channel_names), label="signal")
        return {channel: signal_arr[i] for i, channel in enumerate(channel_names)}


def build_lisa_aet_likelihood(
    data: Any,
    template: LISAAETTemplate,
    sensitivity: Any | None = None,
    freqs: Sequence[float] | None = None,
    channels: Sequence[str] = AET_CHANNELS,
    **kwargs: Any,
) -> GWLikelihoods:
    """Build a HyperWave likelihood directly from ``lisatools`` A/E/T objects."""
    data_arr, freq_arr, noise_arr, channel_names = prepare_lisa_aet_inputs(
        data=data,
        sensitivity=sensitivity,
        freqs=freqs,
        channels=channels,
    )
    return GWLikelihoods(
        data=data_arr,
        f=freq_arr,
        ifos_list=channel_names,
        noise=noise_arr,
        template=template,
        **kwargs,
    )


__all__ = [
    "AET_CHANNELS",
    "LISAAETTemplate",
    "build_lisa_aet_likelihood",
    "lisatools_available",
    "prepare_lisa_aet_inputs",
]
