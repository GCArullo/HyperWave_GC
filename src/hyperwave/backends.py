"""Array backend helpers for optional NumPy/CuPy execution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import kv as scipy_kv
from scipy.special import kve as scipy_kve

try:
    import cupy as cp
except Exception:  # pragma: no cover - depends on optional runtime
    cp = None

try:
    from cupyx.scipy import special as cupyx_special
except Exception:  # pragma: no cover - depends on optional runtime
    cupyx_special = None


def gpu_backend_available() -> bool:
    """Return ``True`` when CuPy is installed and can see a CUDA device."""
    if cp is None:
        return False

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@dataclass(frozen=True)
class ArrayBackend:
    """Small adapter around NumPy or CuPy."""

    xp: Any
    name: str
    use_gpu: bool

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(data, dtype=dtype)

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return self.xp.array(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = float) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def to_numpy(self, data: Any) -> np.ndarray:
        if self.use_gpu and cp is not None and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)

    def kv(self, order: float, values: Any) -> Any:
        if self.use_gpu and cupyx_special is not None and hasattr(cupyx_special, "kv"):
            return cupyx_special.kv(order, values)
        return self.asarray(scipy_kv(order, self.to_numpy(values)))

    def kve(self, order: float, values: Any) -> Any:
        if self.use_gpu and cupyx_special is not None and hasattr(cupyx_special, "kve"):
            return cupyx_special.kve(order, values)
        return self.asarray(scipy_kve(order, self.to_numpy(values)))

    def log_kv(self, order: float, values: Any) -> Any:
        """Stable ``log(K_v(x))`` using the exponentially scaled Bessel function."""
        return self.xp.log(self.kve(order, values)) - values


def get_array_backend(gpu: bool = False) -> ArrayBackend:
    """
    Return the requested array backend.

    When ``gpu=True`` but CuPy/CUDA is not available, the backend falls back to
    NumPy and emits a warning instead of failing at import time.
    """
    if gpu:
        if gpu_backend_available():
            return ArrayBackend(xp=cp, name="cupy", use_gpu=True)

        warnings.warn(
            "GPU backend requested, but CuPy/CUDA is unavailable. Falling back to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ArrayBackend(xp=np, name="numpy", use_gpu=False)
