from .transform import Transform
from ._affine_helpers import affine_to_unit, unit_to_affine
from ..support import Support, OutOfSupportError
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from scipy.stats import norm


class AffineProbitTransform(Transform):

    def __init__(self, low: float64, high: float64):
        low = float64(low)
        high = float64(high)
        if not np.isfinite(low) or not np.isfinite(high) or not (low < high):
            raise ValueError("AffineProbitTransform requires finite low < high.")
        self.low = low
        self.high = high

    def __repr__(self) -> str:
        return self.__class__.__name__

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        z = affine_to_unit(x, self.low, self.high)
        return float64(norm.ppf(z))

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        z = float64(norm.cdf(y))
        return unit_to_affine(z, self.low, self.high)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dy/dx = 1 / ((high-low) * phi(y)), where y = Phi^{-1}(z) and z = (x-low)/(high-low)
        z = affine_to_unit(x, self.low, self.high)
        y = float64(norm.ppf(z))
        return float64(1.0 / (self._span * float64(norm.pdf(y))))

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        # dx/dy = (high-low) * phi(y)
        return float64(self._span * norm.pdf(y))

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dy/dx| = -log(high-low) - log(phi(y))
        z = affine_to_unit(x, self.low, self.high)
        y = norm.ppf(z)
        return float64(-np.log(self._span) - float64(norm.logpdf(y)))

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # log|dx/dy| = log(high-low) + log(phi(y))
        return float64(np.log(self._span) + float64(norm.logpdf(y)))

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        # d/dy log|dx/dy| = d/dy log(phi(y)) = -y
        return float64(-y)

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
