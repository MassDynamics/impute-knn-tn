"""Public API for kNN-TN imputation.

Provides impute_knn_tn() for long-format intensity data (matching the
MDImputeKnnTn R package interface) and knn_tn() for wide-format matrices.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from .knn_engine import impute_knn


def knn_tn(
    data_wide: pd.DataFrame | np.ndarray,
    k: int = 5,
    distance: str = "truncation",
    perc: float = 0.01,
) -> np.ndarray:
    """kNN-TN imputation on a wide-format matrix.

    Equivalent to R's kNN_TN(): log2 -> impute -> 2^x -> restore originals.

    Parameters
    ----------
    data_wide : DataFrame or ndarray, shape (n_features, n_samples)
        Intensity matrix (linear scale). NaN = missing.
    k : int
        Number of nearest neighbours.
    distance : str
        "truncation" or "correlation".
    perc : float
        Percentile threshold for truncated normal estimation.

    Returns
    -------
    ndarray
        Imputed matrix (linear scale).
    """
    if isinstance(data_wide, pd.DataFrame):
        mat = data_wide.values.astype(np.float64)
    else:
        mat = np.asarray(data_wide, dtype=np.float64)

    original_mat = mat.copy()

    # Validate: log2 requires positive values
    observed = mat[~np.isnan(mat)]
    if np.any(observed <= 0):
        raise ValueError(
            "Input contains non-positive values. knn_tn expects intensities > 0."
        )

    # Log2 transform
    log2_mat = np.log2(mat)

    # Impute in log2 space
    imputed_log2 = impute_knn(log2_mat, k=k, distance=distance, perc=perc)

    # Back-transform
    imputed = np.power(2.0, imputed_log2)

    # Restore original non-missing values (the truncated-normal standardization
    # can alter non-NA values, so overwrite them with originals)
    observed_mask = ~np.isnan(original_mat)
    imputed[observed_mask] = original_mat[observed_mask]

    return imputed


def pivot_to_wide(
    intensities: pd.DataFrame, feature_col: str, replicate_col: str
) -> pd.DataFrame:
    """Pivot long-format intensities to wide (features x replicates)."""
    wide = intensities.pivot(
        index=feature_col,
        columns=replicate_col,
        values="NormalisedIntensity",
    )
    return wide


def pivot_to_long(
    wide_data: pd.DataFrame | np.ndarray,
    intensities: pd.DataFrame,
    replicate_col: str,
    feature_col: str,
) -> pd.DataFrame:
    """Pivot wide-format back to long, preserving feature column type."""
    if isinstance(wide_data, np.ndarray):
        wide_df = pd.DataFrame(
            wide_data,
            index=intensities[feature_col].unique(),
            columns=intensities[replicate_col].unique(),
        )
    else:
        wide_df = wide_data

    long = wide_df.reset_index().melt(
        id_vars=feature_col,
        var_name=replicate_col,
        value_name="NormalisedIntensity",
    )

    # Coerce feature column to original type
    orig_dtype = intensities[feature_col].dtype
    long[feature_col] = long[feature_col].astype(orig_dtype)

    return long


def convert_na_to_strings(metadata: pd.DataFrame) -> pd.DataFrame:
    """Convert NA values in metadata string columns to empty strings.

    Required for parquet compatibility.
    """
    metadata = metadata.copy()
    cols = ["GeneNames", "GroupLabel", "GroupLabelType", "ProteinIds", "Description"]
    for col in cols:
        if col in metadata.columns:
            metadata[col] = metadata[col].astype(str)
            metadata[col] = metadata[col].replace({"nan": "", "<NA>": "", "None": ""})
            metadata.loc[metadata[col].isna(), col] = ""
    return metadata


def impute_knn_tn(
    intensities: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_col: str = "GroupId",
    replicate_col: str = "replicate",
    k: int = 5,
    distance: str = "truncation",
    perc: float = 0.01,
) -> dict[str, pd.DataFrame]:
    """kNN-TN imputation for long-format intensity data.

    Equivalent to R's imputekNN_tn(). Rows with Imputed==1 are set to NA
    and re-imputed. Non-imputed values are preserved exactly.

    Parameters
    ----------
    intensities : DataFrame
        Long-format intensity table with columns including feature_col,
        replicate_col, 'NormalisedIntensity', and optionally 'Imputed'.
    metadata : DataFrame
        Metadata table.
    feature_col : str
        Column name for features (default "GroupId").
    replicate_col : str
        Column name for replicates (default "replicate").
    k : int
        Number of nearest neighbours.
    distance : str
        "truncation" or "correlation".
    perc : float
        Percentile threshold for truncated normal estimation.

    Returns
    -------
    dict with keys:
        'intensity' : DataFrame — imputed intensities
        'metadata' : DataFrame — metadata with NAs converted
        'runtime_metadata' : DataFrame — algorithm parameters
    """
    intensities = intensities.copy()
    col_names = list(intensities.columns)

    # Step 1: Mark Imputed==1 rows as NA
    if "Imputed" in intensities.columns:
        mask = intensities["Imputed"] == 1
        intensities.loc[mask, "NormalisedIntensity"] = np.nan

    # Step 2: Pivot to wide
    data_wide = pivot_to_wide(intensities, feature_col, replicate_col)

    # Step 3: Impute
    imputed_mat = knn_tn(data_wide, k=k, distance=distance, perc=perc)

    # Step 4: Pivot back to long
    imputed_wide = pd.DataFrame(
        imputed_mat,
        index=data_wide.index,
        columns=data_wide.columns,
    )
    data_long = pivot_to_long(imputed_wide, intensities, replicate_col, feature_col)

    # Step 5: Merge with original data
    intensities_no_ni = intensities.drop(columns=["NormalisedIntensity"])
    merged = data_long.merge(
        intensities_no_ni, on=[feature_col, replicate_col], validate="one_to_one"
    )

    # Step 6: Preserve column order
    if not all(c in merged.columns for c in col_names):
        raise ValueError("Not all intensities column names are in output")
    merged = merged[col_names]

    # Step 7: Runtime metadata
    runtime_metadata = pd.DataFrame(
        {
            "PythonVersion": [
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ],
            "replicateColname": [replicate_col],
            "featureColname": [feature_col],
            "imputeMethod": ["kNN-TN"],
            "distance": [distance],
            "k": [k],
            "perc": [perc],
        }
    )

    # Step 8: Return
    metadata = convert_na_to_strings(metadata)
    return {
        "intensity": merged,
        "metadata": metadata,
        "runtime_metadata": runtime_metadata,
    }
