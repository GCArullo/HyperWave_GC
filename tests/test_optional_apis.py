"""Regression tests added during the repository scan."""

from __future__ import annotations

import numpy as np

from hyperwave.inference.sampling import LVKinference


def test_get_clean_chain_requires_all_parameters_valid():
    coords = np.zeros((2, 1, 1, 1, 2))
    coords[:, 0, 0, 0, 0] = [1.0, 2.0]
    coords[:, 0, 0, 0, 1] = [10.0, np.nan]

    samples = LVKinference.get_clean_chain(None, coords, ndim=2)

    np.testing.assert_array_equal(samples, np.array([[1.0, 10.0]]))
