"""Unit tests for knn_engine.py."""

import warnings

import numpy as np
import pytest
from impute_knn_tn.knn_engine import _pairwise_complete_cor, impute_knn


# ---- _pairwise_complete_cor ----


def test_pairwise_cor_perfect_positive():
    """Perfectly correlated vectors should give r=1."""
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    r = _pairwise_complete_cor(X, y)
    np.testing.assert_allclose(r, [1.0, 1.0], atol=1e-14)


def test_pairwise_cor_perfect_negative():
    """Perfectly anti-correlated vectors should give r=-1."""
    X = np.array([[3.0], [2.0], [1.0]])
    y = np.array([1.0, 2.0, 3.0])
    r = _pairwise_complete_cor(X, y)
    np.testing.assert_allclose(r, [-1.0], atol=1e-14)


def test_pairwise_cor_with_nan():
    """Pairwise complete correlation skips NaN positions."""
    X = np.array([[1.0], [np.nan], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r = _pairwise_complete_cor(X, y)
    # Should compute correlation using rows 0, 2, 3 only
    x_obs = np.array([1.0, 3.0, 4.0])
    y_obs = np.array([1.0, 3.0, 4.0])
    expected = np.corrcoef(x_obs, y_obs)[0, 1]
    np.testing.assert_allclose(r, [expected], atol=1e-14)


def test_pairwise_cor_zero_variance():
    """Constant column should give NaN correlation."""
    X = np.array([[5.0], [5.0], [5.0]])
    y = np.array([1.0, 2.0, 3.0])
    r = _pairwise_complete_cor(X, y)
    assert np.isnan(r[0])


def test_pairwise_cor_fewer_than_two_complete():
    """Fewer than 2 complete pairs should give NaN."""
    X = np.array([[1.0], [np.nan], [np.nan]])
    y = np.array([1.0, 2.0, 3.0])
    r = _pairwise_complete_cor(X, y)
    assert np.isnan(r[0])


# ---- impute_knn ----


def test_impute_knn_basic_correlation():
    """Small matrix imputation produces no NaN with correlation distance."""
    np.random.seed(42)
    data = np.random.rand(10, 5)
    data[0, 1] = np.nan
    data[2, 0] = np.nan
    result = impute_knn(data, k=2, distance="correlation")
    assert np.sum(np.isnan(result)) == 0
    assert result.shape == data.shape


def test_impute_knn_basic_truncation():
    """Small matrix imputation produces no NaN with truncation distance."""
    np.random.seed(42)
    data = np.random.rand(10, 5) + 1.0  # positive values for truncation
    data[0, 1] = np.nan
    data[2, 0] = np.nan
    result = impute_knn(data, k=2, distance="truncation")
    assert np.sum(np.isnan(result)) == 0
    assert result.shape == data.shape


def test_impute_knn_preserves_observed():
    """Non-missing values should be preserved (within standardize/unstandardize precision)."""
    np.random.seed(42)
    data = np.random.rand(10, 5)
    original = data.copy()
    data[0, 1] = np.nan
    observed_mask = ~np.isnan(data)
    result = impute_knn(data, k=2, distance="correlation")
    np.testing.assert_allclose(
        result[observed_mask], original[observed_mask], rtol=1e-10
    )


def test_impute_knn_invalid_distance():
    """Invalid distance parameter should raise ValueError."""
    data = np.random.rand(5, 3)
    data[0, 1] = np.nan
    with pytest.raises(ValueError, match="distance must be"):
        impute_knn(data, k=2, distance="euclidean")


def test_impute_knn_k_too_large():
    """k > number of rows should raise ValueError."""
    data = np.random.rand(3, 3)
    data[0, 1] = np.nan
    with pytest.raises(ValueError, match="k should be between"):
        impute_knn(data, k=5, distance="correlation")


def test_impute_knn_k_zero():
    """k=0 should raise ValueError."""
    data = np.random.rand(5, 3)
    data[0, 1] = np.nan
    with pytest.raises(ValueError, match="k should be between"):
        impute_knn(data, k=0, distance="correlation")


def test_impute_knn_no_missing():
    """Matrix with no missing values should be returned unchanged."""
    np.random.seed(42)
    data = np.random.rand(10, 5)
    result = impute_knn(data, k=2, distance="correlation")
    np.testing.assert_allclose(result, data, rtol=1e-10)


def test_impute_knn_does_not_mutate_input():
    """Input array should not be modified."""
    np.random.seed(42)
    data = np.random.rand(10, 5)
    data[0, 1] = np.nan
    original = data.copy()
    impute_knn(data, k=2, distance="correlation")
    np.testing.assert_array_equal(data[~np.isnan(data)], original[~np.isnan(original)])
    assert np.isnan(data[0, 1])


# ---- adaptive k ----


def test_impute_knn_adaptive_k_warns():
    """When fewer than k finite distances exist, k is reduced with a warning."""
    # 10 features, 5 samples — enough for valid correlations but k=8 can't
    # always be satisfied since some features share limited pairwise-complete obs.
    np.random.seed(42)
    data = np.random.rand(10, 5)
    # Inject a single missing value — with k=8, need 8 finite distances
    # from 9 candidates. With 5 samples, some correlations may be NaN.
    data[0, 0] = np.nan
    # Add missingness to reduce pairwise-complete counts for some candidates
    data[3, 1] = np.nan
    data[4, 2] = np.nan
    data[5, 0] = np.nan
    data[6, 1] = np.nan
    data[7, 2] = np.nan
    data[8, 0] = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = impute_knn(data, k=8, distance="correlation")
        k_warnings = [x for x in w if "k was reduced" in str(x.message)]

    assert np.sum(np.isnan(result)) == 0, "Output should have no NaN"
    assert len(k_warnings) > 0, "Should warn about k reduction"
    assert "minimum k used=" in str(k_warnings[0].message)


def test_impute_knn_adaptive_k_floor_error():
    """When fewer than 2 finite distances exist, raise ValueError."""
    # 3 features, 3 samples. Feature 0 has only 1 observed value and shares
    # no pairwise-complete observations with the candidates (features 1-2),
    # so all correlations are NaN -> 0 finite distances.
    data = np.array(
        [
            [np.nan, np.nan, 1.0],  # target: only observed at sample 2
            [1.0, 2.0, np.nan],  # candidate: observed at 0,1 — no overlap with target
            [3.0, 4.0, np.nan],  # candidate: observed at 0,1 — no overlap with target
        ]
    )
    with pytest.raises(ValueError, match="Fewer than 2"):
        impute_knn(data, k=2, distance="correlation")
