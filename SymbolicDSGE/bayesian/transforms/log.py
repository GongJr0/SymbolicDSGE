from .transform import Transform, OutOfSupportError
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class LogTransform(Transform):

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(np.log(x))
        elif self.support.at_boundary(
            x, "low"
        ):  # Bound must be non-inclusive if the contains check falied but we're at the boundary
            return float64(np.log(self.eps))  # == log(x+eps)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return np.exp(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(1 / x)
        elif self.support.at_boundary(x, "low"):
            return float64(1 / (self.eps))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
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
        if self.support.contains(x):
            return float64(-np.log(x))
        elif self.support.at_boundary(x, "low"):
            return float64(-np.log(self.eps))
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
            return float64(y)  # dx/dy = exp(y) => log|dx/dy| = log(exp(y)) = y
        else:
            raise OutOfSupportError(y, self.maps_to)

    @property
    def support(self) -> Support:
        return Support(
            float64(0),
            float64(np.inf),
            low_inclusive=True,
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
