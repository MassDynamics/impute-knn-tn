"""Truncated normal MLE parameter estimation.

Reimplements mklhood, NewtonRaphsonLike, and EstimatesComputation from the
GSimp package (Wei et al., 2018) — file Trunc_KNN/Imput_funcs.r.

Uses scipy.integrate.quad to match R's integrate() for numerical parity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import integrate
from scipy.stats import norm

# Sentinel value returned by newton_raphson_like on convergence failure.
# Matches R's GSimp convention.
NR_FAILED = 1000


def _psi(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _psi_mu(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) * (
        (y - mu) / (sigma**3 * np.sqrt(2 * np.pi))
    )


def _psi_sigma(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) * (
        ((y - mu) ** 2) / (sigma**4 * np.sqrt(2 * np.pi))
        - 1 / (sigma**2 * np.sqrt(2 * np.pi))
    )


def _psi2_mu(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) * (
        ((y - mu) ** 2) / (sigma**5 * np.sqrt(2 * np.pi))
        - 1 / (sigma**3 * np.sqrt(2 * np.pi))
    )


def _psi2_sigma(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) * (
        2 / (sigma**3 * np.sqrt(2 * np.pi))
        - 5 * (y - mu) ** 2 / (sigma**5 * np.sqrt(2 * np.pi))
        + (y - mu) ** 4 / (sigma**7 * np.sqrt(2 * np.pi))
    )


def _psi12_musig(y: float, mu: float, sigma: float) -> float:
    return np.exp(-((y - mu) ** 2) / (2 * sigma**2)) * (
        ((y - mu) ** 3) / (sigma**6 * np.sqrt(2 * np.pi))
        - 3 * (y - mu) / (sigma**4 * np.sqrt(2 * np.pi))
    )


def _integrate(func: Any, t1: float, t2: float, mu: float, sigma: float) -> float:
    """Numerical integration matching R's integrate(stop.on.error=FALSE).

    R's integrate() defaults: rel.tol = .Machine$double.eps^0.25 ≈ 1.22e-4
    We match this tolerance for parity.
    """
    r_tol = np.finfo(np.float64).eps ** 0.25  # ~1.22e-4, same as R
    try:
        result, _ = integrate.quad(
            func,
            t1,
            t2,
            args=(mu, sigma),
            epsabs=r_tol,
            epsrel=r_tol,
        )
    except Exception:
        result = 0.0
    return result


class TruncNormLikelihood:
    """Truncated normal log-likelihood with gradient and Hessian.

    Equivalent to GSimp's mklhood(data, t) closure.
    """

    def __init__(self, data: np.ndarray, t: tuple[float, float]) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.data = self.data[~np.isnan(self.data)]
        self.n = len(self.data)
        self.t = sorted(t)
        self.t1 = self.t[0]
        self.t2 = self.t[1]

    def ll_tnorm2(self, p: np.ndarray) -> float:
        """Negative log-likelihood. Matches GSimp's ll.tnorm2(p)."""
        mu, sigma = p[0], p[1]
        out = (
            -self.n
            * np.log(norm.cdf(self.t2, mu, sigma) - norm.cdf(self.t1, mu, sigma))
            - self.n * np.log(np.sqrt(2 * np.pi * sigma**2))
            - np.sum((self.data - mu) ** 2) / (2 * sigma**2)
        )
        return -1.0 * out

    def grad_tnorm(self, p: np.ndarray) -> np.ndarray:
        """Gradient. Matches GSimp's grad.tnorm(p)."""
        mu, sigma = p[0], p[1]
        Phi_diff = norm.cdf(max(self.t), mu, sigma) - norm.cdf(min(self.t), mu, sigma)

        g1 = (
            -self.n * _integrate(_psi_mu, self.t1, self.t2, mu, sigma) / Phi_diff
            - (self.n * mu - np.sum(self.data)) / sigma**2
        )

        g2 = (
            -self.n * _integrate(_psi_sigma, self.t1, self.t2, mu, sigma) / Phi_diff
            - self.n / sigma
            + np.sum((self.data - mu) ** 2) / sigma**3
        )

        return np.array([g1, g2])

    def hessian_tnorm(self, p: np.ndarray) -> np.ndarray:
        """Hessian matrix. Matches GSimp's hessian.tnorm(p)."""
        mu, sigma = p[0], p[1]

        int_psi = _integrate(_psi, self.t1, self.t2, mu, sigma)
        int_psi_mu = _integrate(_psi_mu, self.t1, self.t2, mu, sigma)
        int_psi_sigma = _integrate(_psi_sigma, self.t1, self.t2, mu, sigma)
        int_psi2_mu = _integrate(_psi2_mu, self.t1, self.t2, mu, sigma)
        int_psi2_sigma = _integrate(_psi2_sigma, self.t1, self.t2, mu, sigma)
        int_psi12 = _integrate(_psi12_musig, self.t1, self.t2, mu, sigma)

        h1 = (
            -self.n * (int_psi * int_psi2_mu - int_psi_mu**2) / int_psi**2
            - self.n / sigma**2
        )

        h3 = (
            -self.n * (int_psi * int_psi12 - int_psi_mu * int_psi_sigma) / int_psi**2
            + 2 * (self.n * mu - np.sum(self.data)) / sigma**3
        )

        h2 = (
            -self.n * (int_psi * int_psi2_sigma - int_psi_sigma**2) / int_psi**2
            + self.n / sigma**2
            - 3 * np.sum((self.data - mu) ** 2) / sigma**4
        )

        H = np.array([[h1, h3], [h3, h2]])
        return H


def mklhood(data: np.ndarray, t: tuple[float, float]) -> TruncNormLikelihood:
    """Create a truncated normal likelihood object.

    Equivalent to GSimp's mklhood(data, t).
    """
    return TruncNormLikelihood(data, t)


def newton_raphson_like(
    lhood: TruncNormLikelihood, p: np.ndarray, tol: float = 1e-07, maxit: int = 100
) -> dict[str, Any] | int:
    """Newton-Raphson optimization for truncated normal parameters.

    Equivalent to GSimp's NewtonRaphsonLike(lhood, p, tol, maxit).
    """
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
            return NR_FAILED

        ll_cur = lhood.ll_tnorm2(cur)
        ll_new = lhood.ll_tnorm2(new)

        if abs(ll_cur - ll_new) / abs(ll_cur) < tol:
            return {"estimate": new, "value": ll_new, "iter": i}

        cur = new

    return {"estimate": new, "value": lhood.ll_tnorm2(new), "iter": i}


def estimates_computation(
    missing_data: np.ndarray, perc: float, max_iter: int = 50
) -> np.ndarray:
    """Compute truncated normal parameter estimates per row.

    Equivalent to GSimp's EstimatesComputation(missingdata, perc, iter).

    Parameters
    ----------
    missing_data : ndarray, shape (n_features, n_samples)
    perc : float
    max_iter : int

    Returns
    -------
    ndarray, shape (n_features, 2) — column 0 = mean, column 1 = SD.
    """
    n_rows, n_cols = missing_data.shape
    param_estim = np.empty((n_rows, 2))

    param_estim[:, 0] = np.nanmean(missing_data, axis=1)
    param_estim[:, 1] = np.nanstd(missing_data, axis=1, ddof=1)

    na_sum = np.sum(np.isnan(missing_data), axis=1)

    idx1 = set(np.where(na_sum / n_cols >= perc)[0])

    lod = np.nanmin(missing_data)
    idx2 = set(np.where(param_estim[:, 0] > 3 * param_estim[:, 1] + lod)[0])

    all_idx = set(range(n_rows))
    idx_nr = sorted(all_idx - idx1 - idx2)

    if not idx_nr:
        return param_estim

    upplim = np.nanmax(missing_data) + 2 * np.max(param_estim[:, 1])

    for i in idx_nr:
        lhood = mklhood(missing_data[i, :], t=(lod, upplim))
        try:
            res = newton_raphson_like(lhood, param_estim[i, :])
        except Exception:
            continue

        if not isinstance(res, dict):
            continue
        if res["iter"] >= max_iter:
            continue

        param_estim[i, :] = res["estimate"]

    return param_estim
