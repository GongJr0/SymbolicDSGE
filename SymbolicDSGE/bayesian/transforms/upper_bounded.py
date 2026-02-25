from .transform import Transform, OutOfSupportError
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class UpperBoundedTransform(Transform):

    def __init__(self, upper: float64):
        self.upper = float64(upper)

    # ---- private helper ----
    def _b_minus_x(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return self.upper - x

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return np.log(self._b_minus_x(x))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return self.upper - np.exp(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dy/dx = -1 / (b - x)
        if self.support.contains(x):
            return -float64(1.0) / self._b_minus_x(x)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dx/dy = -exp(y)
        if self.maps_to.contains(y):
            return -np.exp(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dy/dx| = -log(b - x)
        if self.support.contains(x):
            return -np.log(self._b_minus_x(x))
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
            float64(-np.inf),
            self.upper,
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
