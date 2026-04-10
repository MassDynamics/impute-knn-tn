"""Unit tests for truncated normal MLE."""

import numpy as np
import pytest
from impute_knn_tn.truncnorm_mle import (
    mklhood,
    newton_raphson_like,
    estimates_computation,
    NR_FAILED,
)


def test_mklhood_gradient_near_zero_at_optimum():
    """At the MLE, gradient should be near zero."""
    np.random.seed(0)
    data = np.random.normal(5, 2, size=30)
    lhood = mklhood(data, t=(0, 15))

    res = newton_raphson_like(lhood, np.array([5.0, 2.0]))
    assert isinstance(res, dict)
    grad = lhood.grad_tnorm(res["estimate"])
    assert np.all(np.abs(grad) < 1e-5)


def test_mklhood_likelihood_decreases():
    """Newton-Raphson should decrease the negative log-likelihood."""
    np.random.seed(1)
    data = np.random.normal(3, 1.5, size=20)
    lhood = mklhood(data, t=(0, 10))

    p0 = np.array([3.0, 1.5])
    ll_init = lhood.ll_tnorm2(p0)

    res = newton_raphson_like(lhood, p0)
    assert isinstance(res, dict)
    assert res["value"] <= ll_init + 1e-10


PARAM_ESTIMATE_DATASETS = [
    ("targeted", "truncation"),
    ("targeted", "correlation"),
    ("untargeted", "truncation"),
    ("untargeted", "correlation"),
    ("real_data", "truncation"),
    ("real_data", "correlation"),
    ("sim", "truncation"),
    ("sim", "correlation"),
    ("synthetic_1000", "truncation"),
    ("synthetic_1000", "correlation"),
    pytest.param("synthetic_5000", "truncation", marks=pytest.mark.slow),
    pytest.param("synthetic_5000", "correlation", marks=pytest.mark.slow),
    pytest.param("synthetic_20000", "truncation", marks=pytest.mark.slow),
    pytest.param("synthetic_20000", "correlation", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("name,distance", PARAM_ESTIMATE_DATASETS)
def test_estimates_computation_parity(name, distance):
    """ParamEstim matches R reference."""
    import pandas as pd
    import os

    ref_dir = os.path.join(os.path.dirname(__file__), "reference")
    config = f"{name}_{distance}"
    d = os.path.join(ref_dir, config)

    inp = pd.read_csv(os.path.join(d, "input_matrix.csv"), index_col=0)
    r_params = pd.read_csv(os.path.join(d, "param_estimates.csv"), index_col=0).values

    log2_file = os.path.join(d, "log2.txt")
    use_log2 = os.path.exists(log2_file) and open(log2_file).read().strip() == "true"

    mat = inp.values.astype(np.float64)
    if use_log2:
        mat = np.log2(mat)

    py_params = estimates_computation(mat, perc=0.01)

    np.testing.assert_allclose(py_params, r_params, rtol=1e-6, atol=1e-6)


def test_newton_raphson_sigma_negative_returns_error():
    """N-R should return error code when sigma goes negative."""
    data = np.array([1.0, 1.0, 1.0])  # zero variance
    lhood = mklhood(data, t=(0, 5))
    res = newton_raphson_like(lhood, np.array([1.0, 0.001]))
    # Should return the NR_FAILED sentinel or a valid dict with positive sigma
    if isinstance(res, dict):
        assert res["estimate"][1] > 0, "Converged to negative sigma"
    else:
        assert res == NR_FAILED, f"Expected NR_FAILED sentinel, got {res}"
