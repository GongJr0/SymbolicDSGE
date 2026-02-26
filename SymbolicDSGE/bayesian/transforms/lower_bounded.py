from .transform import Transform
from ..support import Support, OutOfSupportError
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class LowerBoundedTransform(Transform):

    def __init__(self, lower: float64):
        self.lower = float64(lower)

    def _x_minus_a(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return x - self.lower

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return np.log(self._x_minus_a(x))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return self.lower + np.exp(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dy/dx = 1 / (x - a)
        if self.support.contains(x):
            return float64(1.0) / self._x_minus_a(x)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dx/dy = exp(y)
        if self.maps_to.contains(y):
            return np.exp(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dy/dx| = -log(x - a)
        if self.support.contains(x):
            return -np.log(self._x_minus_a(x))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dx/dy| = log(exp(y)) = y
        if self.maps_to.contains(y):
            return y
        else:
            raise OutOfSupportError(y, self.maps_to)

    @property
    def support(self) -> Support:
        return Support(
            self.lower,
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
