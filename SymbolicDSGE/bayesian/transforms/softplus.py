from .transform import Transform
from ..support import Support, OutOfSupportError
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from scipy.special import expit


class SoftplusTransform(Transform):
    """
    Maps x in (0, +inf) <-> y in (-inf, +inf)

    forward:  y = inv_softplus(x) = log(exp(x) - 1)
    inverse:  x = softplus(y)     = log(1 + exp(y))
    """

    def __repr__(self) -> str:
        return self.__class__.__name__

    # ---- private numerics helpers ----
    def _softplus(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # softplus(y) = log(1 + exp(y))
        return float64(np.logaddexp(float64(0.0), y))

    def _inv_softplus(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # inv_softplus(x) = log(exp(x) - 1)
        return float64(np.log(np.expm1(x)))

    def _sigmoid(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # sigmoid(y) = 1 / (1 + exp(-y)) = expit(y)
        return expit(y)

    # ---- API ----
    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return self._inv_softplus(x)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return self._softplus(y)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        exm1 = np.expm1(x)  # exp(x)-1, stable near 0
        return float64(1.0 + 1.0 / exm1)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return self._sigmoid(y)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return float64(x - np.log(np.expm1(x)))

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return -np.logaddexp(float64(0.0), -y)

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return float64(1 - expit(y))

    @property
    def support(self) -> Support:
        return Support(
            float64(0.0),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )

    @property
    def maps_to(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )
