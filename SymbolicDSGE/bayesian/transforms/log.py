from .transform import Transform, TransformMethod
from ..support import Support, OutOfSupportError
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..._ckernels.transforms import (
    log_fwd,
    log_inv,
    log_grad_fwd,
    log_grad_inv,
    log_ldet_abs_jac_fwd,
    log_ldet_abs_jac_inv,
    log_grad_ldet_abs_jac_inv,
)


class LogTransform(Transform):
    def __repr__(self) -> str:
        return self.__class__.__name__

    def to_spec(self) -> tuple[str, dict[str, float]]:
        return TransformMethod.LOG.value, {}

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.support.contains(x):
            return log_fwd(x)
        elif self.support.at_boundary(
            x, "low"
        ):  # Bound must be non-inclusive if the contains check falied but we're at the boundary
            return log_fwd(self.eps)  # == log(x+eps)
        else:
            raise OutOfSupportError(x, self.support)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        if self.maps_to.contains(y):
            return log_inv(y)
        else:
            raise OutOfSupportError(y, self.maps_to)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return log_grad_fwd(x)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return log_grad_inv(y)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return log_ldet_abs_jac_fwd(x)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return log_ldet_abs_jac_inv(y)  # dx/dy = exp(y) => log|dx/dy| = log(exp(y)) = y

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return log_grad_ldet_abs_jac_inv(y)  # d/dy log|dx/dy| = d/dy y = 1

    @property
    def support(self) -> Support:
        return Support(
            float64(0),
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
