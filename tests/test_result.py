"""Tests for the :class:`hyperwave.Result` container."""

import numpy as np
import pytest

from hyperwave import Result


def test_basic_access():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(1000, 3))
    r = Result(s, ["a", "b", "c"], injection={"a": 0.0, "b": 0.0, "c": 0.0}, sampler="eryn")
    assert r.n_samples == 1000 and r.n_dim == 3
    np.testing.assert_allclose(r["a"], s[:, 0])
    assert set(r.median()) == {"a", "b", "c"}
    lo, hi = r.credible_interval(0.9)["a"]
    assert lo < hi


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        Result(np.zeros((10, 3)), ["a", "b"])


def test_credible_level_truth_at_median():
    rng = np.random.default_rng(1)
    r = Result(rng.normal(size=(5000, 1)), ["x"], injection={"x": 0.0})
    assert abs(r.credible_level()["x"] - 0.5) < 0.05  # truth at median -> CL ~ 0.5


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    s = rng.normal(size=(128, 2))
    r = Result(s, ["m", "q"], injection={"m": 1.0, "q": 0.5}, sampler="pocomc",
               log_evidence=-3.2, priors={"m": (0.0, 2.0)}, metadata={"seed": 2})
    for ext in (".npz", ".h5"):
        p = str(tmp_path / f"r{ext}")
        r.save(p)
        r2 = Result.load(p)
        np.testing.assert_allclose(r2.samples, s)
        assert r2.parameter_names == ["m", "q"]
        assert r2.injection == {"m": 1.0, "q": 0.5}
        assert r2.sampler == "pocomc" and r2.log_evidence == -3.2
        assert r2.priors == {"m": (0.0, 2.0)}
        assert r2.metadata["seed"] == 2
