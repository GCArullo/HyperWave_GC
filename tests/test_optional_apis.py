"""Graceful failures for optional APIs absent from this source snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from hyperwave import WaveletTemplate
from hyperwave.inference import flow_backend_available, make_flow_distribution_move
from hyperwave.inference.sampling import LVKinference


def test_missing_wavelet_template_raises_importerror():
    with pytest.raises(ImportError, match="Wavelet reconstruction support"):
        WaveletTemplate()


def test_missing_flow_proposal_raises_importerror():
    assert flow_backend_available() is False
    with pytest.raises(ImportError, match="flow_proposals"):
        make_flow_distribution_move({})


def test_get_clean_chain_requires_all_parameters_valid():
    coords = np.zeros((2, 1, 1, 1, 2))
    coords[:, 0, 0, 0, 0] = [1.0, 2.0]
    coords[:, 0, 0, 0, 1] = [10.0, np.nan]

    samples = LVKinference.get_clean_chain(None, coords, ndim=2)

    np.testing.assert_array_equal(samples, np.array([[1.0, 10.0]]))
