from .transform import Transform, TransformMethod
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..._ckernels.transforms import (
    upper_fwd,
    upper_inv,
    upper_grad_fwd,
    upper_grad_inv,
    upper_ldet_abs_jac_fwd,
    upper_ldet_abs_jac_inv,
    upper_grad_ldet_abs_jac_inv,
)


class UpperBoundedTransform(Transform):
    def __init__(self, high: float64):
        self.high = float64(high)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def to_spec(self) -> tuple[str, dict[str, float]]:
        return TransformMethod.UPPER_BOUNDED.value, {"high": float(self.high)}

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return upper_fwd(x, self.high)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return upper_inv(y, self.high)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return upper_grad_fwd(x, self.high)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return upper_grad_inv(y)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return upper_ldet_abs_jac_fwd(x, self.high)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return upper_ldet_abs_jac_inv(y)

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return upper_grad_ldet_abs_jac_inv(y)

    @property
    def support(self) -> Support:
        return Support(
            float64(-np.inf),
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
