from scipy.special import expit
from .transform import Transform
from ._affine_helpers import affine_to_unit, unit_to_affine
from ..support import Support, OutOfSupportError
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class AffineLogitTransform(Transform):
    def __init__(self, low: float64, high: float64):
        low = float64(low)
        high = float64(high)
        if not np.isfinite(low) or not np.isfinite(high) or not (low < high):
            raise ValueError("AffineLogitTransform requires finite low < high.")
        self.low = low
        self.high = high

    def __repr__(self) -> str:
        return self.__class__.__name__

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            z = affine_to_unit(x, self.low, self.high)
            return np.log(z / (1 - z))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            z = float64(1.0) / (float64(1.0) + np.exp(-y))
            return unit_to_affine(z, self.low, self.high)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dy/dx = 1 / ((b-a) * z * (1 - z)), with z = (x-a)/(b-a)
        if self.support.contains(x):
            z = affine_to_unit(x, self.low, self.high)
            return float64(1.0) / (self._span * z * (1 - z))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dx/dy = (b-a) * sigmoid(y) * (1 - sigmoid(y))
        if self.maps_to.contains(y):
            z = float64(1.0) / (float64(1.0) + np.exp(-y))
            return self._span * z * (1 - z)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dy/dx| = -log(b-a) - log(z) - log(1-z)
        if self.support.contains(x):
            z = affine_to_unit(x, self.low, self.high)
            return float64(-np.log(self._span) - np.log(z) - np.log(1 - z))
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dx/dy| = log(b-a) + log(sigmoid(y)(1-sigmoid(y)))
        #            = log(b-a) - y - 2*log(1 + exp(-y))
        if self.maps_to.contains(y):
            return float64(np.log(self._span) - y - 2.0 * np.log1p(np.exp(-y)))
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
            return float64(1 - 2 * expit(y))
        else:
            raise OutOfSupportError(y, self.maps_to)

    @property
    def _span(self) -> float64:
        return float64(self.high - self.low)

    @property
    def support(self) -> Support:
        return Support(
            self.low,
            self.high,
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
