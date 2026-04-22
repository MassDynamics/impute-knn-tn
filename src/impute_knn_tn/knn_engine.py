"""Core kNN imputation engine.

Reimplements imputeKNN from GSimp's Trunc_KNN/Imput_funcs.r.
"""

from __future__ import annotations

import warnings

import numpy as np
from .truncnorm_mle import estimates_computation

# Use Cython-accelerated inner loop if available, fall back to pure Python
try:
    from ._correlation import (  # ty: ignore[unresolved-import]
        pairwise_complete_cor as _pairwise_complete_cor_cy,
        impute_knn_inner as _impute_knn_inner_cy,
    )

    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


def _pairwise_complete_cor_py(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X with y, using pairwise complete obs.

    Pure Python/numpy fallback. Matches R's cor(X, y, use="pairwise.complete.obs").

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_candidates)
    y : ndarray, shape (n_samples,)

    Returns
    -------
    ndarray, shape (n_candidates,)
    """
    y_nan = np.isnan(y)

    # If y has no NaN and X has no NaN, use fully vectorized path
    x_nan_any = np.isnan(X).any()
    if not y_nan.any() and not x_nan_any:
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        num = X_c.T @ y_c
        denom = np.sqrt(np.sum(X_c**2, axis=0) * np.sum(y_c**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / denom
        r[denom == 0] = np.nan
        return r

    # General case: pairwise complete observations
    n_samples, n_cand = X.shape
    valid_y = ~y_nan
    valid_x = ~np.isnan(X)
    valid = valid_x & valid_y[:, np.newaxis]

    counts = valid.sum(axis=0)

    X_masked = np.where(valid, X, 0.0)
    y_broad = np.where(valid, y[:, np.newaxis], 0.0)

    x_sums = X_masked.sum(axis=0)
    y_sums = y_broad.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        x_means = x_sums / counts
        y_means = y_sums / counts

    X_c = np.where(valid, X - x_means, 0.0)
    Y_c = np.where(valid, y[:, np.newaxis] - y_means, 0.0)

    num = (X_c * Y_c).sum(axis=0)
    denom = np.sqrt((X_c**2).sum(axis=0) * (Y_c**2).sum(axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom

    r[counts < 2] = np.nan
    r[denom == 0] = np.nan

    return r


def _pairwise_complete_cor(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation with pairwise complete obs — dispatches to Cython or Python."""
    if _USE_CYTHON:
        return _pairwise_complete_cor_cy(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
        )
    return _pairwise_complete_cor_py(X, y)


def _impute_knn_inner_py(
    data, imp_knn, mv_rows, mv_cols, unique_cols, col_starts, k, nr
):
    """Pure Python inner loop for kNN imputation.

    Returns
    -------
    tuple (min_k_used, n_reduced)
        min_k_used: smallest k actually used (equals k if no reduction needed).
        n_reduced: number of missing values where k was reduced.
    """
    t_data = data.T
    ngenes = np.arange(nr)
    min_k_used = k
    n_reduced = 0

    for i in range(len(unique_cols)):
        start = col_starts[i]
        if i + 1 < len(unique_cols):
            end = col_starts[i + 1]
        else:
            end = len(mv_rows)

        set_indices = list(range(start, end))

        missing_rows_in_set = set(mv_rows[set_indices])
        cand_genes = np.array([g for g in ngenes if g not in missing_rows_in_set])

        if len(cand_genes) == 0:
            continue

        cand_vectors = t_data[:, cand_genes]
        exp_num = unique_cols[i]

        for j in set_indices:
            gene_num = mv_rows[j]
            tar_vector = data[gene_num, :]

            r = _pairwise_complete_cor(cand_vectors, tar_vector)
            dist = 1.0 - np.abs(r)

            dist[np.isnan(dist)] = np.inf
            dist[np.abs(r) == 1] = np.inf

            zero_mask = dist == 0
            if np.any(zero_mask):
                pos_dists = dist[dist > 0]
                finite_pos = pos_dists[np.isfinite(pos_dists)]
                if len(finite_pos) > 0:
                    dist[zero_mask] = np.min(finite_pos) / 2
                else:
                    dist[zero_mask] = 1.0

            n_finite = int(np.sum(np.isfinite(dist)))
            if n_finite < 2:
                raise ValueError(
                    f"Fewer than 2 finite distances found for feature {gene_num} "
                    f"in sample {exp_num} ({n_finite} available). "
                    f"Consider lowering k or removing sparse features."
                )

            k_eff = min(k, n_finite)
            if k_eff < k:
                n_reduced += 1
                if k_eff < min_k_used:
                    min_k_used = k_eff

            top_k = np.argpartition(dist, k_eff)[:k_eff]
            k_genes_ind = top_k[np.argsort(dist[top_k])]
            k_genes = cand_genes[k_genes_ind]

            w = 1.0 / dist[k_genes_ind]
            wghts = (w / np.sum(w)) * np.sign(r[k_genes_ind])
            imp_knn[gene_num, exp_num] = np.dot(wghts, data[k_genes, exp_num])

    return min_k_used, n_reduced


def impute_knn(
    data: np.ndarray, k: int, distance: str = "correlation", perc: float = 1.0
) -> np.ndarray:
    """kNN imputation with correlation or truncation distance.

    Exact translation of GSimp's imputeKNN().

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Data matrix with NaN for missing values.
    k : int
        Number of nearest neighbours.
    distance : str
        "correlation" or "truncation".
    perc : float
        Percentile threshold for truncated normal estimation.

    Returns
    -------
    ndarray
        Imputed matrix (no NaNs).
    """
    data = np.asarray(data, dtype=np.float64).copy()

    if distance not in ("correlation", "truncation"):
        raise ValueError(
            f"distance must be 'correlation' or 'truncation', got '{distance}'"
        )

    nr = data.shape[0]
    if k < 1 or k > nr:
        raise ValueError("k should be between 1 and the number of rows")

    # Standardize rows
    if distance == "correlation":
        genemeans = np.nanmean(data, axis=1)
        genesd = np.nanstd(data, axis=1, ddof=1)
        data = ((data.T - genemeans) / genesd).T

    elif distance == "truncation":
        param_mat = estimates_computation(data, perc=perc)
        genemeans = param_mat[:, 0]
        genesd = param_mat[:, 1]
        data = ((data.T - genemeans) / genesd).T

    # Replace non-finite with NaN
    imp_knn = data.copy()
    imp_knn[~np.isfinite(data)] = np.nan

    # Find missing value indices: (row, col) pairs
    # R's which(arr.ind=TRUE) returns sorted by COLUMN first, then row.
    # np.where sorts by row first. We must match R's ordering.
    mv_rows, mv_cols = np.where(np.isnan(imp_knn))
    if len(mv_rows) == 0:
        # No missing values, just inverse-standardize and return
        imp_knn = ((imp_knn.T * genesd) + genemeans).T
        return imp_knn

    # Sort by column first (like R), then by row within column
    sort_order = np.lexsort((mv_rows, mv_cols))
    mv_rows = mv_rows[sort_order]
    mv_cols = mv_cols[sort_order]

    # Group by column (R calls these "arrays")
    # R: arrays <- unique(mv.ind[, 2])
    # R: array.ind <- match(arrays, mv.ind[, 2])
    unique_cols = []
    col_starts = []
    seen = set()
    for idx, c in enumerate(mv_cols):
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
            col_starts.append(idx)

    unique_cols_arr = np.array(unique_cols, dtype=np.int64)
    col_starts_arr = np.array(col_starts, dtype=np.int64)

    # Dispatch to Cython or Python inner loop
    if _USE_CYTHON:
        min_k_used, n_reduced = _impute_knn_inner_cy(
            np.ascontiguousarray(data),
            np.ascontiguousarray(imp_knn),
            mv_rows.astype(np.int64),
            mv_cols.astype(np.int64),
            unique_cols_arr,
            col_starts_arr,
            k,
            nr,
        )
    else:
        min_k_used, n_reduced = _impute_knn_inner_py(
            data,
            imp_knn,
            mv_rows,
            mv_cols,
            unique_cols,
            col_starts,
            k,
            nr,
        )

    if n_reduced > 0:
        warnings.warn(
            f"k was reduced for {n_reduced} missing value(s) due to insufficient "
            f"finite distances (requested k={k}, minimum k used={min_k_used}). "
            f"This is common with few samples or high missingness.",
            stacklevel=2,
        )

    # Inverse standardize
    imp_knn = ((imp_knn.T * genesd) + genemeans).T

    return imp_knn
