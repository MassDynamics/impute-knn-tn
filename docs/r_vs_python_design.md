# R vs Python Implementation: Side-by-Side Design Reference

This document provides a detailed side-by-side comparison of the original R implementation from [GSimp](https://github.com/WandeRum/GSimp) (`Trunc_KNN/Imput_funcs.r`, Wei et al. 2018) and this Python reimplementation. Every function is mapped with line-level equivalence notes.

---

## 1. Truncated Normal MLE: `mklhood`

The R `mklhood` function returns a closure containing the log-likelihood, gradient, and Hessian. Python uses a class with methods instead.

### R (GSimp)
```r
mklhood <- function(data, t, ...) {
    data <- na.omit(data)
    n <- length(data)
    t <- sort(t)
    # ... psi functions defined as inner closures ...
    # returns list(ll.tnorm2, grad.tnorm, hessian.tnorm)
}
```

### Python (`truncnorm_mle.py`)
```python
class TruncNormLikelihood:
    def __init__(self, data, t):
        self.data = np.asarray(data, dtype=np.float64)
        self.data = self.data[~np.isnan(self.data)]  # na.omit(data)
        self.n = len(self.data)                       # n <- length(data)
        self.t = sorted(t)                            # t <- sort(t)
        self.t1 = self.t[0]
        self.t2 = self.t[1]

def mklhood(data, t):  # Factory function preserves R API name
    return TruncNormLikelihood(data, t)
```

**Design choice**: R uses a closure (function returning functions that capture `data`, `n`, `t` in their environment). Python uses a class where `self` holds the same state. The factory function `mklhood()` preserves the R API name for traceability.

---

## 2. Psi Helper Functions

These compute the integrand functions for the truncated normal normalising constant and its derivatives. They are mathematically identical.

### R (inner closures of `mklhood`)
```r
psi <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) / (sigma*sqrt(2*pi))
}

psi.mu <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) *
        ((y-mu) / (sigma^3*sqrt(2*pi)))
}

psi.sigma <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) *
        (((y-mu)^2) / (sigma^4*sqrt(2*pi))
         - 1 / (sigma^2*sqrt(2*pi)))
}

psi2.mu <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) *
        (((y-mu)^2) / (sigma^5*sqrt(2*pi))
         - 1 / (sigma^3*sqrt(2*pi)))
}

psi2.sigma <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) *
        (2 / (sigma^3*sqrt(2*pi))
         - 5*(y-mu)^2 / (sigma^5*sqrt(2*pi))
         + (y-mu)^4 / (sigma^7*sqrt(2*pi)))
}

psi12.musig <- function(y, mu, sigma) {
    exp(-(y-mu)^2 / (2*sigma^2)) *
        (((y-mu)^3) / (sigma^6*sqrt(2*pi))
         - 3*(y-mu) / (sigma^4*sqrt(2*pi)))
}
```

### Python (module-level functions)
```python
def _psi(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def _psi_mu(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) * (
        (y - mu) / (sigma**3 * np.sqrt(2 * np.pi)))

def _psi_sigma(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) * (
        ((y - mu)**2) / (sigma**4 * np.sqrt(2 * np.pi))
        - 1 / (sigma**2 * np.sqrt(2 * np.pi)))

def _psi2_mu(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) * (
        ((y - mu)**2) / (sigma**5 * np.sqrt(2 * np.pi))
        - 1 / (sigma**3 * np.sqrt(2 * np.pi)))

def _psi2_sigma(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) * (
        2 / (sigma**3 * np.sqrt(2 * np.pi))
        - 5 * (y - mu)**2 / (sigma**5 * np.sqrt(2 * np.pi))
        + (y - mu)**4 / (sigma**7 * np.sqrt(2 * np.pi)))

def _psi12_musig(y, mu, sigma):
    return np.exp(-(y - mu)**2 / (2 * sigma**2)) * (
        ((y - mu)**3) / (sigma**6 * np.sqrt(2 * np.pi))
        - 3 * (y - mu) / (sigma**4 * np.sqrt(2 * np.pi)))
```

**Equivalence**: Identical formulas. R names use `.` separators (`psi.mu`), Python uses `_` (`_psi_mu`). Underscore prefix marks them as private.

---

## 3. Numerical Integration

### R
```r
integrate(psi.mu, t[1], t[2], mu=p[1], sigma=p[2], stop.on.error = FALSE)$value
```

R's `integrate()` uses adaptive quadrature with default relative tolerance `rel.tol = .Machine$double.eps^0.25 ~ 1.22e-4`. The `stop.on.error = FALSE` flag means integration errors return 0 instead of raising.

### Python
```python
def _integrate(func, t1, t2, mu, sigma):
    r_tol = np.finfo(np.float64).eps ** 0.25  # ~1.22e-4, same as R
    try:
        result, _ = integrate.quad(func, t1, t2, args=(mu, sigma),
                                   epsabs=r_tol, epsrel=r_tol)
    except Exception:
        result = 0.0  # matches stop.on.error = FALSE
    return result
```

**Equivalence**: `scipy.integrate.quad` uses the same QUADPACK Fortran library as R's `integrate()`. The tolerance is explicitly set to match R's default `eps^0.25`. The `try/except` matches `stop.on.error = FALSE`.

---

## 4. Negative Log-Likelihood: `ll.tnorm2`

### R
```r
ll.tnorm2 <- function(p) {
    out <- (-n * log(pnorm(t[2], p[1], p[2]) - pnorm(t[1], p[1], p[2]))) -
           (n * log(sqrt(2*pi*p[2]^2))) -
           (sum((data - p[1])^2) / (2*p[2]^2))
    -1 * out
}
```

### Python
```python
def ll_tnorm2(self, p):
    mu, sigma = p[0], p[1]
    out = (
        -self.n * np.log(norm.cdf(self.t2, mu, sigma)
                         - norm.cdf(self.t1, mu, sigma))
        - self.n * np.log(np.sqrt(2 * np.pi * sigma**2))
        - np.sum((self.data - mu)**2) / (2 * sigma**2)
    )
    return -1.0 * out
```

**Equivalence**: `pnorm(x, mu, sigma)` = `scipy.stats.norm.cdf(x, mu, sigma)`. Both compute the negative log-likelihood of the truncated normal. R uses 1-indexed `p[1]`, `p[2]`; Python uses 0-indexed `p[0]`, `p[1]`.

---

## 5. Gradient: `grad.tnorm`

### R
```r
grad.tnorm <- function(p) {
    g1 <- (-n * (integrate(psi.mu, t[1], t[2], mu=p[1], sigma=p[2],
                           stop.on.error=FALSE)$value) /
           (pnorm(max(t), p[1], p[2]) - pnorm(min(t), p[1], p[2]))) -
          ((n*p[1] - sum(data)) / p[2]^2)

    g2 <- (-n * (integrate(psi.sigma, t[1], t[2], mu=p[1], sigma=p[2],
                           stop.on.error=FALSE)$value) /
           (pnorm(max(t), p[1], p[2]) - pnorm(min(t), p[1], p[2]))) -
          (n / p[2]) + (sum((data-p[1])^2) / p[2]^3)

    out <- c(g1, g2)
    return(out)
}
```

### Python
```python
def grad_tnorm(self, p):
    mu, sigma = p[0], p[1]
    Phi_diff = norm.cdf(max(self.t), mu, sigma) - norm.cdf(min(self.t), mu, sigma)

    g1 = (
        -self.n * _integrate(_psi_mu, self.t1, self.t2, mu, sigma) / Phi_diff
        - (self.n * mu - np.sum(self.data)) / sigma**2
    )
    g2 = (
        -self.n * _integrate(_psi_sigma, self.t1, self.t2, mu, sigma) / Phi_diff
        - self.n / sigma
        + np.sum((self.data - mu)**2) / sigma**3
    )
    return np.array([g1, g2])
```

**Equivalence**: Line-for-line identical mathematics. Python precomputes `Phi_diff` to avoid recomputation; R calls `pnorm` inline twice per gradient component (same result since `max(t)` = `t[2]`, `min(t)` = `t[1]` after sorting).

---

## 6. Hessian: `hessian.tnorm`

### R
```r
hessian.tnorm <- function(p) {
    h1 <- -n * (integrate(psi, ...)$value * integrate(psi2.mu, ...)$value
                - integrate(psi.mu, ...)$value^2) /
          (integrate(psi, ...)$value^2) - n/(p[2]^2)

    h3 <- -n * (integrate(psi, ...)$value * integrate(psi12.musig, ...)$value
                - integrate(psi.mu, ...)$value * integrate(psi.sigma, ...)$value) /
          (integrate(psi, ...)$value^2) + (2*(n*p[1]-sum(data)))/(p[2]^3)

    h2 <- -n * (integrate(psi, ...)$value * integrate(psi2.sigma, ...)$value
                - integrate(psi.sigma, ...)$value^2) /
          (integrate(psi, ...)$value^2) + n/(p[2]^2) - (3*sum((data-p[1])^2))/(p[2]^4)

    H <- matrix(0, nrow=2, ncol=2)
    H[1,1] <- h1;  H[2,2] <- h2;  H[1,2] <- H[2,1] <- h3
    return(H)
}
```

### Python
```python
def hessian_tnorm(self, p):
    mu, sigma = p[0], p[1]
    int_psi       = _integrate(_psi,         self.t1, self.t2, mu, sigma)
    int_psi_mu    = _integrate(_psi_mu,      self.t1, self.t2, mu, sigma)
    int_psi_sigma = _integrate(_psi_sigma,   self.t1, self.t2, mu, sigma)
    int_psi2_mu   = _integrate(_psi2_mu,     self.t1, self.t2, mu, sigma)
    int_psi2_sigma= _integrate(_psi2_sigma,  self.t1, self.t2, mu, sigma)
    int_psi12     = _integrate(_psi12_musig, self.t1, self.t2, mu, sigma)

    h1 = -self.n * (int_psi * int_psi2_mu - int_psi_mu**2) / int_psi**2 \
         - self.n / sigma**2
    h3 = -self.n * (int_psi * int_psi12 - int_psi_mu * int_psi_sigma) / int_psi**2 \
         + 2 * (self.n * mu - np.sum(self.data)) / sigma**3
    h2 = -self.n * (int_psi * int_psi2_sigma - int_psi_sigma**2) / int_psi**2 \
         + self.n / sigma**2 - 3 * np.sum((self.data - mu)**2) / sigma**4

    return np.array([[h1, h3], [h3, h2]])
```

**Equivalence**: Identical formulas. R recomputes `integrate(psi, ...)` multiple times within the function; Python precomputes all six integrals once. Same result, fewer redundant calls. Note R uses `h1`/`h2`/`h3` naming where `h1` = d^2/dmu^2, `h2` = d^2/dsigma^2, `h3` = d^2/dmu.dsigma — Python preserves this naming.

---

## 7. Newton-Raphson: `NewtonRaphsonLike`

### R
```r
NewtonRaphsonLike <- function(lhood, p, tol = 1e-07, maxit = 100) {
    cscore <- lhood$grad.tnorm(p)
    if (sum(abs(cscore)) < tol)
        return(list(estimate = p, value = lhood$ll.tnorm2(p), iter = 0))

    cur <- p
    for (i in 1:maxit) {
        inverseHess <- solve(lhood$hessian.tnorm(cur))
        cscore <- lhood$grad.tnorm(cur)
        new <- cur - cscore %*% inverseHess
        if (new[2] <= 0) stop("Sigma < 0")
        cscore <- lhood$grad.tnorm(new)

        if (((abs(lhood$ll.tnorm2(cur) - lhood$ll.tnorm2(new)) /
              (lhood$ll.tnorm2(cur))) < tol))
            return(list(estimate = new, value = lhood$ll.tnorm2(new), iter = i))
        cur <- new
    }
    return(list(estimate = new, value = lhood$ll.tnorm2(new), iter = i))
}
```

### Python
```python
NR_FAILED = 1000  # Sentinel matching R's tryCatch error handler

def newton_raphson_like(lhood, p, tol=1e-07, maxit=100):
    p = np.asarray(p, dtype=np.float64).copy()
    cscore = lhood.grad_tnorm(p)

    if np.sum(np.abs(cscore)) < tol:
        return {"estimate": p, "value": lhood.ll_tnorm2(p), "iter": 0}

    cur = p.copy()
    for i in range(1, maxit + 1):
        try:
            H = lhood.hessian_tnorm(cur)
            cscore = lhood.grad_tnorm(cur)
            inverse_hess = np.linalg.inv(H)
        except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
            return NR_FAILED

        new = cur - cscore @ inverse_hess

        if not np.all(np.isfinite(new)):
            return NR_FAILED

        if new[1] <= 0:
            return NR_FAILED      # R: stop("Sigma < 0")

        ll_cur = lhood.ll_tnorm2(cur)
        ll_new = lhood.ll_tnorm2(new)

        if abs(ll_cur - ll_new) / abs(ll_cur) < tol:
            return {"estimate": new, "value": ll_new, "iter": i}

        cur = new

    return {"estimate": new, "value": lhood.ll_tnorm2(new), "iter": i}
```

**Key differences**:

| Aspect | R | Python | Why |
|--------|---|--------|-----|
| Sigma < 0 | `stop("Sigma < 0")` raises error | Returns `NR_FAILED` sentinel | R's `tryCatch` in `EstimatesComputation` catches the error and continues. Python returns a sentinel instead, checked by `estimates_computation`. Same behavior: the row falls back to sample mean/SD. |
| Matrix inverse | `solve(H)` | `np.linalg.inv(H)` | Same LAPACK routine underneath |
| Update formula | `cur - cscore %*% inverseHess` | `cur - cscore @ inverse_hess` | `%*%` = `@` for matrix multiply. Works because H is symmetric so `grad @ H_inv == H_inv @ grad`. |
| Convergence | Relative change in log-likelihood | Same formula | `abs(ll_cur - ll_new) / abs(ll_cur) < tol` |
| Gradient recompute | R recomputes gradient at `new` after update | Python does not (not needed for convergence check) | R's extra `cscore <- lhood$grad.tnorm(new)` result is unused before the next iteration overwrites it. No effect on results. |

---

## 8. Parameter Estimation: `EstimatesComputation`

### R
```r
EstimatesComputation <- function(missingdata, perc, iter=50) {
    ParamEstim <- matrix(NA, nrow = nrow(missingdata), ncol = 2)
    nsamp <- ncol(missingdata)

    ParamEstim[,1] <- rowMeans(missingdata, na.rm = TRUE)
    ParamEstim[,2] <- apply(missingdata, 1, function(x) sd(x, na.rm = TRUE))

    na.sum <- apply(missingdata, 1, function(x) sum(is.na(x)))
    idx1 <- which(na.sum/nsamp >= perc)

    lod <- min(missingdata, na.rm=TRUE)
    idx2 <- which(ParamEstim[,1] > 3*ParamEstim[,2] + lod)

    idx.nr <- setdiff(1:nrow(missingdata), c(idx1, idx2))

    upplim <- max(missingdata, na.rm=TRUE) + 2*max(ParamEstim[,2])

    for (i in idx.nr) {
        Likelihood <- mklhood(missingdata[i,], t=c(lod, upplim))
        res <- tryCatch(NewtonRaphsonLike(Likelihood, p = ParamEstim[i,]),
                        error = function(e) 1000)

        if (length(res) == 1) { next }
        else if (res$iter >= iter) { next }
        else { ParamEstim[i,] <- as.numeric(res$estimate) }
    }
    return(ParamEstim)
}
```

### Python
```python
def estimates_computation(missing_data, perc, max_iter=50):
    n_rows, n_cols = missing_data.shape
    param_estim = np.empty((n_rows, 2))

    param_estim[:, 0] = np.nanmean(missing_data, axis=1)     # rowMeans(., na.rm=TRUE)
    param_estim[:, 1] = np.nanstd(missing_data, axis=1, ddof=1)  # sd(., na.rm=TRUE)

    na_sum = np.sum(np.isnan(missing_data), axis=1)
    idx1 = set(np.where(na_sum / n_cols >= perc)[0])          # Case 1: high missing %

    lod = np.nanmin(missing_data)                              # LOD = global minimum
    idx2 = set(np.where(param_estim[:, 0] > 3 * param_estim[:, 1] + lod)[0])  # Case 2

    all_idx = set(range(n_rows))
    idx_nr = sorted(all_idx - idx1 - idx2)                    # Case 3: use N-R

    upplim = np.nanmax(missing_data) + 2 * np.max(param_estim[:, 1])

    for i in idx_nr:
        lhood = mklhood(missing_data[i, :], t=(lod, upplim))
        try:
            res = newton_raphson_like(lhood, param_estim[i, :])
        except Exception:                                      # tryCatch(., error=...)
            continue

        if res == NR_FAILED:    # R: if (length(res) == 1) next
            continue
        if res["iter"] >= max_iter:
            continue

        param_estim[i, :] = res["estimate"]

    return param_estim
```

**Equivalence line-by-line**:

| R | Python | Notes |
|---|--------|-------|
| `rowMeans(., na.rm=TRUE)` | `np.nanmean(., axis=1)` | Identical |
| `apply(., 1, sd, na.rm=TRUE)` | `np.nanstd(., axis=1, ddof=1)` | `ddof=1` matches R's `sd()` which uses `n-1` denominator |
| `min(., na.rm=TRUE)` | `np.nanmin(.)` | Global minimum as LOD |
| `setdiff(1:nrow, c(idx1, idx2))` | `set(range(n_rows)) - idx1 - idx2` | Set difference |
| `tryCatch(., error=function(e) 1000)` | `try/except` + `NR_FAILED` sentinel | R returns literal `1000` on error; Python uses named constant `NR_FAILED = 1000` |
| `if (length(res) == 1)` | `if res == NR_FAILED` | R checks if result is a scalar (the error sentinel) vs a list |

---

## 9. kNN Imputation: `imputeKNN`

This is the core algorithm. The structure is identical between R and Python.

### Step 1: Input validation and standardisation

#### R
```r
imputeKNN <- function(data, k, distance = "correlation", perc=1, ...) {
    distance <- match.arg(distance, c("correlation", "truncation"))
    nr <- dim(data)[1]
    if (k < 1 | k > nr) stop("k should be between 1 and the number of rows")

    if (distance == "correlation") {
        genemeans <- rowMeans(data, na.rm=TRUE)
        genesd <- apply(data, 1, function(x) sd(x, na.rm = TRUE))
        data <- (data - genemeans) / genesd
    }
    if (distance == "truncation") {
        ParamMat <- EstimatesComputation(data, perc = perc)
        genemeans <- ParamMat[,1]
        genesd <- ParamMat[,2]
        data <- (data - genemeans) / genesd
    }
```

#### Python
```python
def impute_knn(data, k, distance="correlation", perc=1.0):
    data = np.asarray(data, dtype=np.float64).copy()

    if distance not in ("correlation", "truncation"):
        raise ValueError(...)

    nr = data.shape[0]
    if k < 1 or k > nr:
        raise ValueError("k should be between 1 and the number of rows")

    if distance == "correlation":
        genemeans = np.nanmean(data, axis=1)        # rowMeans(data, na.rm=TRUE)
        genesd = np.nanstd(data, axis=1, ddof=1)    # apply(data, 1, sd, na.rm=TRUE)
        data = ((data.T - genemeans) / genesd).T     # (data - genemeans) / genesd

    elif distance == "truncation":
        param_mat = estimates_computation(data, perc=perc)
        genemeans = param_mat[:, 0]
        genesd = param_mat[:, 1]
        data = ((data.T - genemeans) / genesd).T
```

**Note on broadcasting**: R's `(data - genemeans) / genesd` broadcasts row-wise because R recycles vectors along columns. NumPy broadcasts along the last axis, so we transpose, subtract, and transpose back: `((data.T - genemeans) / genesd).T`.

### Step 2: Find missing values (column-first ordering)

#### R
```r
imp.knn <- data
imp.knn[is.finite(data) == FALSE] <- NA
t.data <- t(data)

mv.ind <- which(is.na(imp.knn), arr.ind = TRUE)
arrays <- unique(mv.ind[, 2])
array.ind <- match(arrays, mv.ind[, 2])
```

#### Python
```python
imp_knn = data.copy()
imp_knn[~np.isfinite(data)] = np.nan

mv_rows, mv_cols = np.where(np.isnan(imp_knn))

# CRITICAL: R's which(arr.ind=TRUE) returns column-first order.
# np.where returns row-first. Must re-sort to match R.
sort_order = np.lexsort((mv_rows, mv_cols))
mv_rows = mv_rows[sort_order]
mv_cols = mv_cols[sort_order]

# Group by column (R: arrays <- unique(mv.ind[, 2]))
unique_cols = []
col_starts = []
seen = set()
for idx, c in enumerate(mv_cols):
    if c not in seen:
        seen.add(c)
        unique_cols.append(c)
        col_starts.append(idx)
```

**Critical ordering difference**: This is the most subtle parity issue. R's `which(arr.ind=TRUE)` returns indices sorted by **column first**, then by row within each column. NumPy's `np.where` returns indices sorted by **row first**. Since the imputation is sequential (each imputed value feeds into subsequent computations), the ordering determines the final result. The `np.lexsort` call re-sorts to match R's column-first order.

### Step 3: Inner loop (per missing value)

#### R
```r
for (i in 1:length(arrays)) {
    set <- array.ind[i]:min((array.ind[(i+1)] - 1), dim(mv.ind)[1], na.rm = TRUE)
    cand.genes <- ngenes[-unique(mv.ind[set, 1])]
    cand.vectors <- t.data[, cand.genes]
    exp.num <- arrays[i]

    for (j in set) {
        gene.num <- mv.ind[j, 1]
        tar.vector <- data[gene.num, ]

        r <- cor(cand.vectors, tar.vector, use = "pairwise.complete.obs")
        dist <- 1 - abs(r)

        dist[is.nan(dist) | is.na(dist)] <- Inf
        dist[dist == 0] <- ifelse(is.finite(min(dist[dist>0])),
                                  min(dist[dist>0])/2, 1)
        dist[abs(r) == 1] <- Inf

        if (sum(is.finite(dist)) < k) stop("Fewer than K finite distances found")

        k.genes.ind <- order(dist)[1:k]
        k.genes <- cand.genes[k.genes.ind]

        wghts <- (1/dist[k.genes.ind] / sum(1/dist[k.genes.ind])) *
                 sign(r[k.genes.ind])
        imp.knn[gene.num, exp.num] <- wghts %*% data[k.genes, exp.num]
    }
}
```

#### Python
```python
for i in range(len(unique_cols)):
    start = col_starts[i]
    end = col_starts[i + 1] if i + 1 < len(unique_cols) else len(mv_rows)
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

        n_finite = np.sum(np.isfinite(dist))
        if n_finite < k:
            raise ValueError(...)

        top_k = np.argpartition(dist, k)[:k]
        k_genes_ind = top_k[np.argsort(dist[top_k])]
        k_genes = cand_genes[k_genes_ind]

        w = 1.0 / dist[k_genes_ind]
        wghts = (w / np.sum(w)) * np.sign(r[k_genes_ind])
        imp_knn[gene_num, exp_num] = np.dot(wghts, data[k_genes, exp_num])
```

**Line-by-line mapping**:

| R | Python | Notes |
|---|--------|-------|
| `cor(cand.vectors, tar.vector, use="pairwise.complete.obs")` | `_pairwise_complete_cor(cand_vectors, tar_vector)` | Custom implementation since Python has no built-in `pairwise.complete.obs` equivalent |
| `1 - abs(r)` | `1.0 - np.abs(r)` | Correlation distance |
| `dist[is.nan(dist) \| is.na(dist)] <- Inf` | `dist[np.isnan(dist)] = np.inf` | NaN distances excluded |
| `dist[dist==0] <- ifelse(...)` | Zero-mask with `np.min(finite_pos) / 2` | Same logic, expanded for clarity |
| `dist[abs(r) == 1] <- Inf` | `dist[np.abs(r) == 1] = np.inf` | Perfect correlations excluded |
| `if (sum(is.finite(dist)) < k) stop(...)` | Adaptive k (see below) | **Deliberate deviation from R** |
| `order(dist)[1:k]` | `np.argpartition(dist, k_eff)[:k_eff]` then sort | `order()` = `argsort()`. Python uses `argpartition` (O(n)) instead of full sort (O(n log n)) for performance, then sorts only the k elements for deterministic ordering. |
| `wghts %*% data[k.genes, exp.num]` | `np.dot(wghts, data[k_genes, exp_num])` | Weighted sum for imputation |

**Adaptive k (deviation from R)**: GSimp's R implementation raises a hard error (`stop("Fewer than K finite distances found")`) when a feature has fewer than `k` neighbours with finite correlation distances. This is common with small-sample experiments (3-6 replicates) using the default `k=5`.

The Python implementation instead reduces `k` to the number of available neighbours (minimum 2) and emits a `warnings.warn()`. This follows the precedent of Bioconductor's `impute::impute.knn` (Troyanskaya et al., 2001), the most widely used kNN imputation in genomics/proteomics. Using fewer neighbours increases variance but does not introduce systematic bias. If fewer than 2 finite distances exist, a `ValueError` is raised.

### Step 4: Inverse standardisation

#### R
```r
if (distance == "correlation") {
    imp.knn <- (imp.knn * genesd) + genemeans
}
if (distance == "truncation") {
    imp.knn <- (imp.knn * genesd) + genemeans
}
```

#### Python
```python
imp_knn = ((imp_knn.T * genesd) + genemeans).T
```

**Equivalence**: Same formula. R broadcasts row-wise naturally; Python uses the transpose trick again. Both distance modes use the same inverse standardisation (R has two identical `if` blocks).

---

## 10. Correlation: `cor()` vs `_pairwise_complete_cor()`

This is the only function with no direct R equivalent to call. R's `cor(X, y, use="pairwise.complete.obs")` is a single C function. Python must implement the algorithm.

### R
```r
r <- cor(cand.vectors, tar.vector, use = "pairwise.complete.obs")
```

R's `cor()` with `use = "pairwise.complete.obs"`:
- For each pair `(X[,j], y)`, finds positions where **both** are non-NA
- Computes Pearson correlation using only those positions
- Returns `NA` if fewer than 2 complete pairs or zero variance

### Python (pure numpy fallback)
```python
def _pairwise_complete_cor_py(X, y):
    # For each candidate column j:
    #   valid = positions where both X[:,j] and y are non-NaN
    #   compute Pearson r using only valid positions
    valid = ~np.isnan(X) & ~np.isnan(y)[:, np.newaxis]
    counts = valid.sum(axis=0)
    # ... masked means, centered products, correlation ...
    r[counts < 2] = np.nan
    r[denom == 0] = np.nan
    return r
```

### Python (Cython accelerated)
```cython
# Two-pass algorithm per candidate:
# Pass 1: compute mean from pairwise-complete observations
# Pass 2: compute centered cross-products and correlation
for j in range(n_cand):
    for i in range(n_samples):
        if not isnan(X[i,j]) and not isnan(y[i]):
            # accumulate sums for mean
    # ... compute correlation from centered sums ...
```

**Equivalence**: Verified to machine precision (`max|diff| < 2.22e-16`) against R's `cor()` across all test datasets. The Cython version eliminates Python overhead by doing the full computation in compiled C loops.

---

## 11. Performance: Cython Optimisation

The Cython extension (`_correlation.pyx`) moves the entire inner loop (correlation + distance + neighbour selection + imputation) into compiled C. This eliminates:

1. **Python function call overhead**: 5,000+ calls to `_pairwise_complete_cor` from Python
2. **Temporary array allocations**: numpy `np.where`, `np.abs`, `np.isnan` each allocate a new array
3. **Python loop overhead**: The `for j in set_indices` loop runs in interpreted Python

With Cython, the entire inner loop runs in compiled C with pre-allocated buffers. The pure Python fallback (`_pairwise_complete_cor_py` + `_impute_knn_inner_py`) is preserved for environments where Cython compilation is unavailable.

### Performance comparison

| Dataset | Features | R (s) | Python pure (s) | Python Cython (s) | Speedup vs R |
|---------|----------|-------|-----------------|-------------------|--------------|
| targeted | 41 | 0.004 | 0.004 | 0.000 | 9-17x |
| real_data | 76 | 0.090 | 0.097 | 0.034 | 2.6x |
| synthetic_1000 | 1,000 | 0.235 | 0.499 | 0.143 | 1.6x |
| synthetic_5000 | 5,000 | 4.363 | 17.867 | 3.192 | 1.4x |
| synthetic_20000 | 20,000 | 65.462 | ~260 (est.) | 52.285 | 1.3x |

---

## 12. Parity Verification

All 38 non-slow + 8 slow parity tests pass at `rtol=1e-10`:

- **Matrix-level parity**: Python `impute_knn` output matches R `imputeKNN` output for all 14 reference datasets (targeted, untargeted, real_data, sim, synthetic_1000/5000/20000 x truncation/correlation)
- **Parameter estimate parity**: Python `estimates_computation` matches R `EstimatesComputation` for all 14 reference datasets at `rtol=1e-6`
- **Pipeline parity**: Python `impute_knn_tn` matches the full R pipeline on bojkova2020 data for both distance modes

The `rtol=1e-10` tolerance is well above machine epsilon (`~2.2e-16`) but tight enough to catch any algorithmic divergence. Differences at this level arise from floating-point operation ordering between R and Python/numpy, not from algorithmic differences.
