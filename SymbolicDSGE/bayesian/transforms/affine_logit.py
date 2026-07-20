from .transform import Transform, TransformMethod
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..._ckernels.transforms import (
    aff_logit_fwd,
    aff_logit_inv,
    aff_logit_grad_fwd,
    aff_logit_grad_inv,
    aff_logit_ldet_abs_jac_fwd,
    aff_logit_ldet_abs_jac_inv,
    aff_logit_grad_ldet_abs_jac_inv,
)


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

    def to_spec(self) -> tuple[str, dict[str, float]]:
        return TransformMethod.AFFINE_LOGIT.value, {
            "low": float(self.low),
            "high": float(self.high),
        }

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return aff_logit_fwd(x, self.low, self.high)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return aff_logit_inv(y, self.low, self.high)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return aff_logit_grad_fwd(x, self.low, self.high)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return aff_logit_grad_inv(y, self.low, self.high)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return aff_logit_ldet_abs_jac_fwd(x, self.low, self.high)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return aff_logit_ldet_abs_jac_inv(y, self.low, self.high)

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return aff_logit_grad_ldet_abs_jac_inv(y)

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
