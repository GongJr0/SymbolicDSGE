from .transform import Transform, OutOfSupportError
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class LogitTransform(Transform):

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(np.log(x / (1 - x)))
        elif self.support.at_boundary(x, "low"):
            return float64(np.log(self.eps / (1 - self.eps)))
        elif self.support.at_boundary(x, "high"):
            return float64(np.log((1 - self.eps) / self.eps))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return float64(1 / (1 + np.exp(-y)))

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(1 / (x * (1 - x)))
        elif self.support.at_boundary(x, "low"):
            return float64(1 / (self.eps * (1 - self.eps)))
        elif self.support.at_boundary(x, "high"):
            return float64(1 / ((1 - self.eps) * self.eps))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        p = self.inverse(y)
        return float64(p * (1 - p))

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return float64(-np.log(x) - np.log(1 - x))
        elif self.support.at_boundary(x, "low"):
            return float64(-np.log(self.eps) - np.log(1 - self.eps))
        elif self.support.at_boundary(x, "high"):
            return float64(-np.log(1 - self.eps) - np.log(self.eps))
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
            return float64(-y - 2 * np.log(1 + np.exp(-y)))
        else:
            raise OutOfSupportError(y, self.maps_to)

    @property
    def support(self) -> Support:
        return Support(
            float64(0),
            float64(1),
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
