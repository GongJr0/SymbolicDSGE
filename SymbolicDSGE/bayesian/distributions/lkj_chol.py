from .distribution import Distribution, Size, RandomState, MatF64
from ..support import Support

import numpy as np
from numpy import float64, linalg
from numba import njit

from typing import TypedDict, Callable, cast
from scipy.special import gammaln
from functools import lru_cache, wraps


class LKJParams(TypedDict):
    eta: float
    K: int
    random_state: RandomState


LKJ_DEFAULTS = LKJParams(
    eta=1.0,
    K=-1,  # Will raise if not provided
    random_state=None,
)


def _is_symmetric(x: MatF64, atol: float = 1e-10) -> bool:
    return bool(np.allclose(x, x.T, atol=atol, rtol=0.0))


def _is_positive_definite(x: MatF64) -> bool:
    try:
        linalg.cholesky(x)
        return True
    except linalg.LinAlgError:
        return False


def _is_correlation_matrix(x: MatF64, atol: float = 1e-10) -> bool:
    d = np.diag(x)
    return bool(np.allclose(d, 1.0, atol=atol, rtol=0.0))


def _is_lower_triangular(x: MatF64, atol: float = 1e-12) -> bool:
    return bool(np.allclose(x, np.tril(x), atol=atol, rtol=0.0))


def _has_unit_row_norms(L: MatF64, atol: float = 1e-10) -> bool:
    K = L.shape[0]
    for i in range(K):
        row = L[i, : i + 1]
        if not np.allclose(np.dot(row, row), 1.0, atol=atol, rtol=0.0):
            return False
    return True


def corr_chol(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self: object, L: MatF64) -> float64:
        L = np.asarray(L, dtype=float64)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Input must be a square matrix.")
        if not _is_lower_triangular(L):
            raise ValueError("Input must be lower triangular.")
        d = np.diag(L).astype(float64, copy=False)
        if np.any(d <= 0):
            raise ValueError("Diagonal of Cholesky factor must be strictly positive.")
        if not _has_unit_row_norms(L):
            raise ValueError(
                "Each row of a correlation Cholesky factor must have unit norm."
            )
        return cast(float64, func(self, L))

    return wrapper


def _log_beta(a: float64, b: float64) -> float64:
    return float64(gammaln(a) + gammaln(b) - gammaln(a + b))


@lru_cache(maxsize=None)
def _log_lkj_normalizer_C(K: int, eta: float) -> float64:
    if K < 1:
        raise ValueError("K must be >= 1.")
    eta = float64(eta)
    if eta <= 0:
        raise ValueError("eta must be > 0.")

    if K == 1:
        return float64(0.0)

    s = float64(0.0)
    for k in range(1, K):
        t = float64(2.0 * (eta - 1.0) + (K - k))
        u = float64(K - k)
        s += t * u

    log_p = float64(0.0)
    for k in range(1, K):
        a = float64(eta + (K - k - 1) / 2.0)
        log_p += float64(K - k) * _log_beta(a, a)

    return float64(s * np.log(2.0) + log_p)


class LKJChol(Distribution[MatF64, MatF64]):
    def __init__(self, eta: float, K: int, random_state: RandomState) -> None:
        eta = float64(eta)
        if eta <= 0:
            raise ValueError("eta must be > 0.")
        self._eta = eta
        self._K = K
        self._random_state = random_state

    def __repr__(self) -> str:
        return self.__class__.__name__

    @corr_chol
    def logpdf(self, L: MatF64) -> float64:
        """
        Cholesky-LKJ density on L, where L is a Cholesky factor of a correlation matrix.

        Constraints (checked):
          - square, lower triangular
          - diag(L) > 0
          - each row i has ||L[i, :i+1]||_2 = 1

        Kernel:
          ∏_{k=2}^K L_{kk}^{K-k+2η-2}
        """
        eta = self._eta
        K = self._K

        logC = _log_lkj_normalizer_C(K, float(eta))

        d = np.diag(L).astype(float64, copy=False)

        log_kernel = float64(0.0)
        for i in range(1, K):  # i = 1..K-1 corresponds to k=2..K
            exponent = float64(K - i + 2.0 * eta - 3.0)
            log_kernel += exponent * np.log(d[i])

        return float64(logC + log_kernel)

    def logpdf_from_R(self, R: MatF64) -> float64:
        """
        Convenience wrapper: accept correlation matrix R, compute L=chol(R), evaluate logpdf(L).
        This is NOT a statement that this equals log p(R) under LKJ on R; it's just evaluation
        of the Cholesky-form density at chol(R).
        """
        R = np.asarray(R, dtype=float64)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("Input must be a square matrix.")
        if not _is_symmetric(R):
            raise ValueError("Input matrix must be symmetric.")
        if not _is_positive_definite(R):
            raise ValueError("Input matrix must be positive definite.")
        if not _is_correlation_matrix(R):
            raise ValueError("Input must be a correlation matrix (unit diagonal).")

        L = np.tril(linalg.cholesky(R))
        return float64(self.logpdf(L))

    def grad_logpdf(self, L: MatF64) -> MatF64:
        G = np.zeros_like(L, dtype=float64)
        K = self._K
        eta = self._eta

        for k in range(1, K):
            exp = float64(K - k - 2.0 * (eta - 1.0))
            G[k, k] = exp / L[k, k]
        return G

    def cdf(self, x: MatF64) -> MatF64:
        raise NotImplementedError("CDF is not defined for LKJChol.")

    def ppf(self, q: MatF64) -> MatF64:
        raise NotImplementedError("PPF is not defined for LKJChol.")

    @property
    def mean(self) -> MatF64:
        raise NotImplementedError("Mean is not defined for LKJChol.")

    @property
    def var(self) -> MatF64:
        raise NotImplementedError("Variance is not defined for LKJChol.")

    @property
    def mode(self) -> MatF64:
        raise NotImplementedError("Mode is not defined for LKJChol.")

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> MatF64:
        rng = self._rng(random_state or self._random_state)
        eta = self._eta
        K = self._K

        if isinstance(size, int):
            size = (size,)

        @njit
        def _one(eta: float64, K: int) -> MatF64:
            n_cpc = (K * (K - 1)) // 2
            CPC = np.empty(n_cpc, dtype=float64)

            alpha = float64(eta + 0.5 * (K - 1))
            idx = 0
            for k in range(K - 1):
                alpha = float64(alpha - 0.5)
                for _ in range(k + 1, K):
                    CPC[idx] = float64(2 * rng.beta(alpha, alpha) - 1.0)
                    idx += 1

            L = np.zeros((K, K), dtype=float64)
            L[0, 0] = 1.0

            idx = 0
            for k in range(1, K):
                rem = float64(1.0)
                for j in range(k):
                    z = CPC[idx]
                    v = float64(np.sqrt(rem))
                    L[k, j] = float64(z * v)

                    rem = float64(rem - L[k, j] * L[k, j])
                    idx += 1
                L[k, k] = float64(np.sqrt(rem))
            return L

        out = np.empty((size + (self._K, self._K)), dtype=float64)
        it = np.nditer(np.empty(size, dtype=np.int8), flags=["multi_index"])
        for _ in it:
            out[it.multi_index] = _one(eta, K)
        return out

    @property
    def support(self) -> Support:
        return Support(
            float64(-1.0),
            float64(1.0),
            low_inclusive=True,
            high_inclusive=True,
        )
