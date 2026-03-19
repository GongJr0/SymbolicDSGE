from __future__ import annotations

from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .transform import Transform
from ..support import Support


NDF = NDArray[np.float64]


def _coerce_z(z: float64 | NDArray[float64], K: int) -> tuple[NDF, bool]:
    expected = (K * (K - 1)) // 2
    arr = np.asarray(z, dtype=float64)
    if arr.ndim == 0:
        if expected != 1:
            raise ValueError(
                f"Expected {expected} unconstrained CPC elements, got scalar input."
            )
        return np.asarray([float64(arr)], dtype=float64), True
    if arr.ndim != 1:
        raise ValueError("Unconstrained correlation coordinates must be a 1D array.")
    if arr.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} unconstrained CPC elements, got {arr.shape[0]}."
        )
    return arr.astype(float64, copy=False), False


def _validate_corr_chol(L: NDF, K: int) -> NDF:
    arr = np.asarray(L, dtype=float64)
    if arr.ndim != 2 or arr.shape != (K, K):
        raise ValueError(
            f"Expected a {K}x{K} lower-triangular correlation Cholesky factor."
        )
    if not np.allclose(arr, np.tril(arr), atol=1e-12, rtol=0.0):
        raise ValueError("Input must be lower triangular.")
    diag = np.diag(arr).astype(float64, copy=False)
    if np.any(diag <= 0.0):
        raise ValueError("Diagonal of a correlation Cholesky factor must be positive.")
    for i in range(K):
        row = arr[i, : i + 1]
        if not np.allclose(np.dot(row, row), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                "Each row of a correlation Cholesky factor must have unit norm."
            )
    return arr


class CholeskyCorrTransform(Transform):
    def __init__(self, K: int) -> None:
        if K < 2:
            raise ValueError(
                "K must be at least 2 for a correlation Cholesky transform."
            )
        self._K = int(K)

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def K(self) -> int:
        return self._K

    @overload
    def safe_forward(self, x: float64) -> float64: ...
    @overload
    def safe_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def safe_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # Correlation Cholesky factors live on a constrained manifold with exact
        # unit row norms, so the generic boundary nudging in Transform.safe_forward
        # would invalidate perfectly good inputs by moving diagonal 1s off-manifold.
        return self.forward(x)

    @overload
    def safe_log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def safe_log_det_abs_jacobian_forward(
        self, x: NDArray[float64]
    ) -> NDArray[float64]: ...

    def safe_log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return self.log_det_abs_jacobian_forward(x)

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        L = _validate_corr_chol(np.asarray(x, dtype=float64), self._K)
        z = np.empty(((self._K * (self._K - 1)) // 2,), dtype=float64)
        idx = 0
        for k in range(1, self._K):
            rem = float64(1.0)
            for j in range(k):
                v = float64(np.sqrt(max(rem, 1e-14)))
                cpc = float64(L[k, j] / v) if v > 0.0 else float64(0.0)
                cpc = float64(np.clip(cpc, -1.0 + 1e-14, 1.0 - 1e-14))
                z[idx] = float64(np.arctanh(cpc))
                rem = float64(rem - L[k, j] * L[k, j])
                idx += 1
        return z

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        z, _ = _coerce_z(y, self._K)
        cpc = np.tanh(z).astype(float64, copy=False)
        L = np.zeros((self._K, self._K), dtype=float64)
        L[0, 0] = 1.0
        idx = 0
        for k in range(1, self._K):
            rem = float64(1.0)
            for j in range(k):
                v = float64(np.sqrt(max(rem, 1e-14)))
                L[k, j] = float64(cpc[idx] * v)
                rem = float64(rem - L[k, j] * L[k, j])
                idx += 1
            L[k, k] = float64(np.sqrt(max(rem, 1e-14)))
        return L

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        raise NotImplementedError(
            "CholeskyCorrTransform.grad_forward would require the full Jacobian of a matrix-valued transform."
        )

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        raise NotImplementedError(
            "CholeskyCorrTransform.grad_inverse would require the full Jacobian of a matrix-valued transform."
        )

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        z = np.asarray(self.forward(x), dtype=float64)
        return float64(-self.log_det_abs_jacobian_inverse(z))

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        z, _ = _coerce_z(y, self._K)
        cpc = np.tanh(z).astype(float64, copy=False)
        total = float64(0.0)
        idx = 0
        for k in range(1, self._K):
            rem = float64(1.0)
            for _ in range(k):
                cpc_i = float64(cpc[idx])
                total += float64(0.5 * np.log(max(rem, 1e-300)))
                total += float64(np.log1p(-(cpc_i * cpc_i)))
                rem = float64(rem * (1.0 - cpc_i * cpc_i))
                idx += 1
        return total

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        z, scalar_input = _coerce_z(y, self._K)
        cpc = np.tanh(z).astype(float64, copy=False)
        grad = np.empty_like(cpc, dtype=float64)
        idx = 0
        for row_len in range(1, self._K):
            for j in range(row_len):
                coeff = float64(1.0 + 0.5 * (row_len - j - 1))
                grad[idx] = float64(-2.0 * coeff * cpc[idx])
                idx += 1
        if scalar_input:
            return float64(grad[0])
        return grad

    @property
    def support(self) -> Support:
        return Support(
            float64(-1.0),
            float64(1.0),
            low_inclusive=True,
            high_inclusive=True,
        )

    @property
    def maps_to(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )
