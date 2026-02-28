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
        if self.support.contains(x):
            return self._inv_softplus(x)
        elif self.support.at_boundary(x, "low"):
            return self._inv_softplus(self.eps)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return self._softplus(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            exm1 = np.expm1(x)  # exp(x)-1, stable near 0
            return float64(1.0 + 1.0 / exm1)
        elif self.support.at_boundary(x, "low"):
            exm1 = np.expm1(self.eps)
            return float64(1.0 + 1.0 / exm1)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return self._sigmoid(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(x - np.log(np.expm1(x)))
        elif self.support.at_boundary(x, "low"):
            return float64(self.eps - np.log(np.expm1(self.eps)))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return -np.logaddexp(float64(0.0), -y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return float64(1 - expit(y))
        elif self.maps_to.at_boundary(y, "low"):
            return float64(1 - expit(self.eps))
        else:
            raise OutOfSupportError(y, self.support)

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
