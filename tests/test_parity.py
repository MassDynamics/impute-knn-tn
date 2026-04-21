"""End-to-end parity tests against R/GSimp reference results."""

import numpy as np
import pytest
from conftest import load_matrix_reference, load_bojkova_reference
from impute_knn_tn import impute_knn, impute_knn_tn


# ---- Matrix-level parity (imputeKNN) ----

MATRIX_DATASETS = [
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


@pytest.mark.parametrize("name,distance", MATRIX_DATASETS)
def test_impute_knn_parity(name, distance):
    """Python impute_knn matches R imputeKNN to machine precision (perc=0.01)."""
    inp, ref, use_log2 = load_matrix_reference(name, distance)

    if use_log2:
        work = np.log2(inp)
        result = impute_knn(work, k=5, distance=distance, perc=0.01)
        result = np.power(2.0, result)
    else:
        result = impute_knn(inp, k=5, distance=distance, perc=0.01)

    assert np.sum(np.isnan(result)) == 0, "Output contains NaN"
    assert result.shape == ref.shape, "Shape mismatch"

    # Relative tolerance: 1e-10 (well above machine eps, catches real divergence)
    np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)


# ---- Matrix-level parity (perc=1.0 — GSimp default) ----

MATRIX_DATASETS_PERC1 = [
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
]


@pytest.mark.parametrize("name,distance", MATRIX_DATASETS_PERC1)
def test_impute_knn_parity_perc1(name, distance):
    """Python impute_knn matches R imputeKNN at perc=1.0 (GSimp default)."""
    inp, ref, use_log2 = load_matrix_reference(name, distance, suffix="_perc1")

    if use_log2:
        work = np.log2(inp)
        result = impute_knn(work, k=5, distance=distance, perc=1.0)
        result = np.power(2.0, result)
    else:
        result = impute_knn(inp, k=5, distance=distance, perc=1.0)

    assert np.sum(np.isnan(result)) == 0, "Output contains NaN"
    assert result.shape == ref.shape, "Shape mismatch"
    np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)


# ---- bojkova2020 full pipeline parity ----


@pytest.mark.parametrize("distance", ["truncation", "correlation"])
def test_bojkova2020_pipeline(distance):
    """Full impute_knn_tn pipeline matches R on bojkova2020."""
    inp_int, inp_meta, ref_int, ref_imp = load_bojkova_reference(distance)

    result = impute_knn_tn(
        inp_int,
        inp_meta,
        feature_col="GroupId",
        replicate_col="replicate",
        k=5,
        distance=distance,
        perc=0.01,
    )

    out = result["intensity"]

    # No NAs remaining
    assert out["NormalisedIntensity"].isna().sum() == 0

    # Imputed==0 values must be EXACTLY unchanged
    inp_non = (
        inp_int[inp_int["Imputed"] == 0][
            ["GroupId", "replicate", "NormalisedIntensity"]
        ]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    out_non = (
        out[out["Imputed"] == 0][["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    assert (
        inp_non["NormalisedIntensity"].values == out_non["NormalisedIntensity"].values
    ).all(), "Imputed==0 values changed"

    # Imputed==1 values match R reference
    out_imp = (
        out[out["Imputed"] == 1][["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    ref_imp_sorted = (
        ref_imp[["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )

    np.testing.assert_allclose(
        out_imp["NormalisedIntensity"].values,
        ref_imp_sorted["NormalisedIntensity"].values,
        rtol=1e-10,
    )

    # Column order preserved
    assert list(out.columns) == list(inp_int.columns)

    # Output keys
    assert set(result.keys()) == {"intensity", "metadata", "runtime_metadata"}


# ---- bojkova2020 full pipeline parity (perc=1.0) ----


@pytest.mark.parametrize("distance", ["truncation", "correlation"])
def test_bojkova2020_pipeline_perc1(distance):
    """Full impute_knn_tn pipeline matches R on bojkova2020 at perc=1.0."""
    inp_int, inp_meta, ref_int, ref_imp = load_bojkova_reference(
        distance, suffix="_perc1"
    )

    result = impute_knn_tn(
        inp_int,
        inp_meta,
        feature_col="GroupId",
        replicate_col="replicate",
        k=5,
        distance=distance,
        perc=1.0,
    )

    out = result["intensity"]

    assert out["NormalisedIntensity"].isna().sum() == 0

    inp_non = (
        inp_int[inp_int["Imputed"] == 0][
            ["GroupId", "replicate", "NormalisedIntensity"]
        ]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    out_non = (
        out[out["Imputed"] == 0][["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    assert (
        inp_non["NormalisedIntensity"].values == out_non["NormalisedIntensity"].values
    ).all(), "Imputed==0 values changed"

    out_imp = (
        out[out["Imputed"] == 1][["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )
    ref_imp_sorted = (
        ref_imp[["GroupId", "replicate", "NormalisedIntensity"]]
        .sort_values(["GroupId", "replicate"])
        .reset_index(drop=True)
    )

    np.testing.assert_allclose(
        out_imp["NormalisedIntensity"].values,
        ref_imp_sorted["NormalisedIntensity"].values,
        rtol=1e-10,
    )
