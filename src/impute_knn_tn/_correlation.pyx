# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated kNN imputation internals.

Moves the hot path (correlation + distance + neighbour selection + imputation)
into compiled C, eliminating Python loop overhead and temporary array allocations.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, isnan, isinf, NAN, INFINITY

np.import_array()


def pairwise_complete_cor(
    double[:, :] X not None,
    double[:] y not None,
):
    """Pearson correlation of each column of X with y, pairwise complete obs.

    Matches R's cor(X, y, use="pairwise.complete.obs").

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_candidates), dtype float64
    y : ndarray, shape (n_samples,), dtype float64

    Returns
    -------
    ndarray, shape (n_candidates,), dtype float64
    """
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_cand = X.shape[1]
    cdef Py_ssize_t i, j
    cdef int count
    cdef double x_sum, y_sum, x_mean, y_mean
    cdef double xx_sum, yy_sum, xy_sum
    cdef double xv, yv, xc, yc
    cdef double denom

    result = np.empty(n_cand, dtype=np.float64)
    cdef double[:] r = result

    for j in range(n_cand):
        x_sum = 0.0
        y_sum = 0.0
        count = 0

        for i in range(n_samples):
            xv = X[i, j]
            yv = y[i]
            if not isnan(xv) and not isnan(yv):
                x_sum += xv
                y_sum += yv
                count += 1

        if count < 2:
            r[j] = NAN
            continue

        x_mean = x_sum / count
        y_mean = y_sum / count

        xx_sum = 0.0
        yy_sum = 0.0
        xy_sum = 0.0

        for i in range(n_samples):
            xv = X[i, j]
            yv = y[i]
            if not isnan(xv) and not isnan(yv):
                xc = xv - x_mean
                yc = yv - y_mean
                xy_sum += xc * yc
                xx_sum += xc * xc
                yy_sum += yc * yc

        denom = sqrt(xx_sum * yy_sum)
        if denom == 0.0:
            r[j] = NAN
        else:
            r[j] = xy_sum / denom

    return result


def impute_knn_inner(
    double[:, :] data not None,
    double[:, :] imp_knn not None,
    long[:] mv_rows not None,
    long[:] mv_cols not None,
    long[:] unique_cols_arr not None,
    long[:] col_starts_arr not None,
    int k,
    int nr,
):
    """Run the full kNN inner loop in compiled C.

    Modifies imp_knn in-place. Raises ValueError if fewer than k finite
    distances are found for any missing value.

    Parameters
    ----------
    data : ndarray (n_features, n_samples) — standardized, with NaN for non-finite
    imp_knn : ndarray (n_features, n_samples) — working copy, modified in-place
    mv_rows, mv_cols : sorted missing value indices (column-first order)
    unique_cols_arr : unique columns with missing values
    col_starts_arr : start index in mv_rows/mv_cols for each unique column
    k : number of neighbours
    nr : number of rows (features)
    """
    cdef Py_ssize_t n_samples = data.shape[1]
    cdef Py_ssize_t n_unique = unique_cols_arr.shape[0]
    cdef Py_ssize_t n_mv = mv_rows.shape[0]
    cdef Py_ssize_t i, j, s, m, ci, si
    cdef Py_ssize_t start, end, gene_num, exp_num
    cdef Py_ssize_t n_cand, n_finite
    cdef double xv, yv, xc, yc
    cdef double x_sum, y_sum, x_mean, y_mean
    cdef double xx_sum, yy_sum, xy_sum, denom
    cdef double d, r_val, min_pos, w_sum, imp_val
    cdef int count

    # Temporary arrays — allocated once, reused across all missing values
    cdef long[:] cand_genes = np.empty(nr, dtype=np.int64)
    cdef double[:] r = np.empty(nr, dtype=np.float64)
    cdef double[:] dist = np.empty(nr, dtype=np.float64)

    # For top-k selection
    cdef long[:] k_idx = np.empty(k, dtype=np.int64)
    cdef double[:] k_dist = np.empty(k, dtype=np.float64)

    for ci in range(n_unique):
        exp_num = unique_cols_arr[ci]
        start = col_starts_arr[ci]
        if ci + 1 < n_unique:
            end = col_starts_arr[ci + 1]
        else:
            end = n_mv

        # Build candidate genes: rows NOT missing in this column group
        # First, mark which rows are missing
        # Use a boolean mask (reuse r array as scratch, 0.0 = not missing, 1.0 = missing)
        for j in range(nr):
            r[j] = 0.0
        for m in range(start, end):
            r[mv_rows[m]] = 1.0

        n_cand = 0
        for j in range(nr):
            if r[j] == 0.0:
                cand_genes[n_cand] = j
                n_cand += 1

        if n_cand == 0:
            continue

        for m in range(start, end):
            gene_num = mv_rows[m]

            # Compute pairwise-complete correlation between target gene and each candidate
            for ci2 in range(n_cand):
                j = cand_genes[ci2]

                # Two-pass correlation
                x_sum = 0.0
                y_sum = 0.0
                count = 0
                for s in range(n_samples):
                    xv = data[j, s]       # candidate gene values
                    yv = data[gene_num, s] # target gene values
                    if not isnan(xv) and not isnan(yv):
                        x_sum += xv
                        y_sum += yv
                        count += 1

                if count < 2:
                    r[ci2] = NAN
                    dist[ci2] = INFINITY
                    continue

                x_mean = x_sum / count
                y_mean = y_sum / count

                xx_sum = 0.0
                yy_sum = 0.0
                xy_sum = 0.0
                for s in range(n_samples):
                    xv = data[j, s]
                    yv = data[gene_num, s]
                    if not isnan(xv) and not isnan(yv):
                        xc = xv - x_mean
                        yc = yv - y_mean
                        xy_sum += xc * yc
                        xx_sum += xc * xc
                        yy_sum += yc * yc

                denom = sqrt(xx_sum * yy_sum)
                if denom == 0.0:
                    r[ci2] = NAN
                    dist[ci2] = INFINITY
                else:
                    r_val = xy_sum / denom
                    r[ci2] = r_val
                    d = 1.0 - fabs(r_val)

                    # NaN distance
                    if isnan(d):
                        dist[ci2] = INFINITY
                    # Perfect correlation
                    elif fabs(r_val) == 1.0:
                        dist[ci2] = INFINITY
                    else:
                        dist[ci2] = d

            # Zero distances: replace with min(positive) / 2
            min_pos = INFINITY
            for ci2 in range(n_cand):
                d = dist[ci2]
                if d > 0.0 and not isinf(d):
                    if d < min_pos:
                        min_pos = d
            for ci2 in range(n_cand):
                if dist[ci2] == 0.0:
                    if min_pos < INFINITY:
                        dist[ci2] = min_pos / 2.0
                    else:
                        dist[ci2] = 1.0

            # Count finite distances
            n_finite = 0
            for ci2 in range(n_cand):
                if not isinf(dist[ci2]) and not isnan(dist[ci2]):
                    n_finite += 1

            if n_finite < k:
                raise ValueError(
                    f"Fewer than {k} finite distances found for feature {gene_num} "
                    f"in sample {exp_num} ({n_finite} available)"
                )

            # Find k nearest neighbours using partial sort
            # Initialize with first k finite entries
            _find_k_nearest(dist, n_cand, k, k_idx, k_dist)

            # Compute weights and impute
            w_sum = 0.0
            for si in range(k):
                w_sum += 1.0 / k_dist[si]

            imp_val = 0.0
            for si in range(k):
                ci2 = k_idx[si]
                j = cand_genes[ci2]
                imp_val += (1.0 / k_dist[si]) / w_sum * (1.0 if r[ci2] >= 0 else -1.0) * data[j, exp_num]

            imp_knn[gene_num, exp_num] = imp_val


cdef void _find_k_nearest(
    double[:] dist,
    Py_ssize_t n,
    int k,
    long[:] out_idx,
    double[:] out_dist,
) noexcept:
    """Find k indices with smallest dist values. Simple insertion for small k."""
    cdef Py_ssize_t i, j
    cdef double d, tmp_d
    cdef long tmp_i

    # Initialize with large values
    for i in range(k):
        out_dist[i] = INFINITY
        out_idx[i] = -1

    for i in range(n):
        d = dist[i]
        if d < out_dist[k - 1]:
            out_dist[k - 1] = d
            out_idx[k - 1] = i
            # Insertion sort to maintain order
            j = k - 1
            while j > 0 and out_dist[j] < out_dist[j - 1]:
                tmp_d = out_dist[j - 1]
                tmp_i = out_idx[j - 1]
                out_dist[j - 1] = out_dist[j]
                out_idx[j - 1] = out_idx[j]
                out_dist[j] = tmp_d
                out_idx[j] = tmp_i
                j -= 1
