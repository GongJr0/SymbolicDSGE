from .transform import Transform, TransformMethod
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..._ckernels.transforms import (
    tanh_fwd,
    tanh_inv,
    tanh_grad_fwd,
    tanh_grad_inv,
    tanh_ldet_abs_jac_fwd,
    tanh_ldet_abs_jac_inv,
    tanh_grad_ldet_abs_jac_inv,
)


class TanhTransform(Transform):
    """
    Maps x in (-1, 1) <-> y in (-inf, +inf)

    forward:  y = atanh(x)
    inverse:  x = tanh(y)
    """

    def __repr__(self) -> str:
        return self.__class__.__name__

    def to_spec(self) -> tuple[str, dict[str, float]]:
        return TransformMethod.TANH.value, {}

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return tanh_fwd(x)

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return tanh_inv(y)

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def grad_forward(self, x: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return tanh_grad_fwd(x)

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def grad_inverse(self, y: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        return tanh_grad_inv(y)

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_forward(
        self, x: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return tanh_ldet_abs_jac_fwd(x)

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return tanh_ldet_abs_jac_inv(y)

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def grad_log_det_abs_jacobian_inverse(
        self, y: float64 | NDArray[float64]
    ) -> float64 | NDArray[float64]:
        return tanh_grad_ldet_abs_jac_inv(y)

    @property
    def support(self) -> Support:
        return Support(
            float64(-1.0),
            float64(1.0),
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
