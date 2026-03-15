from enum import StrEnum
from abc import ABC, abstractmethod
from numpy import float64
from numpy.typing import NDArray
from typing import TypeVar, overload

import numpy as np

from ..support import Support, OutOfSupportError

T = TypeVar("T", float64, NDArray[float64])


class TransformMethod(StrEnum):
    IDENTITY = "identity"  # (-inf, inf)
    LOG = "log"  # (0, inf)  via y=log(x)
    SOFTPLUS = "softplus"  # (0, inf)  via x=softplus(y)  (sampler-friendly alternative)
    LOGIT = "logit"  # (0, 1)    via y=log(x/(1-x))
    PROBIT = "probit"  # (0, 1)    via y=Phi^{-1}(x)

    AFFINE_LOGIT = "affine_logit"  # (low, high) via x=low+(high-low)*sigmoid(y)
    AFFINE_PROBIT = "affine_probit"  # (low, high) via x=low+(high-low)*Phi(y)

    LOWER_BOUNDED = "lower_bounded"  # (low, inf) via x=low+exp(y) (or low+softplus(y))
    UPPER_BOUNDED = (
        "upper_bounded"  # (-inf, high) via x=high-exp(y) (or high-softplus(y))
    )

    SIMPLEX = (
        "simplex"  # weights on simplex (sum=1, each>0) via softmax / stick-breaking
    )

    CHOLESKY_COV = "cholesky_cov"  # SPD covariance via unconstrained -> L -> Sigma=LL'
    CHOLESKY_CORR = "cholesky_corr"  # correlation matrix (LKJ) via unconstrained -> corr Cholesky factor


class Transform(ABC):
    @abstractmethod
    def __repr__(self) -> str: ...

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def forward(self, x: T) -> T:
        pass

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def inverse(self, y: T) -> T:
        pass

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def grad_forward(self, x: T) -> T:
        pass

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def grad_inverse(self, y: T) -> T:
        pass

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def log_det_abs_jacobian_forward(self, x: T) -> T:
        pass

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def log_det_abs_jacobian_inverse(self, y: T) -> T:
        pass

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    @abstractmethod
    def grad_log_det_abs_jacobian_inverse(self, y: T) -> T:
        pass

    @overload
    def _get_adjusted_forward(self, x: float64) -> float64: ...
    @overload
    def _get_adjusted_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def _get_adjusted_forward(self, x: T) -> T:
        sup = self.support
        if isinstance(x, (float64, float)):
            x = float64(x)
            if np.isfinite(sup.low) and sup.at_boundary(x, "low"):
                return x + self.eps
            if np.isfinite(sup.high) and sup.at_boundary(x, "high"):
                return x - self.eps
            if not sup.contains(x):
                raise OutOfSupportError(x, sup)
            return x

        arr = x.astype(float64, copy=False)
        low_mask = (
            np.isclose(arr, sup.low, atol=float64(1e-6))
            if np.isfinite(sup.low)
            else False
        )
        high_mask = (
            np.isclose(arr, sup.high, atol=float64(1e-6))
            if np.isfinite(sup.high)
            else False
        )
        adjusted = np.where(
            low_mask, arr + self.eps, np.where(high_mask, arr - self.eps, arr)
        )
        if not sup.contains(adjusted):
            raise OutOfSupportError(arr, sup)
        return adjusted

    @overload
    def _get_adjusted_inverse(self, z: float64) -> float64: ...
    @overload
    def _get_adjusted_inverse(self, z: NDArray[float64]) -> NDArray[float64]: ...

    def _get_adjusted_inverse(self, z: T) -> T:
        maps_to = self.maps_to
        if isinstance(z, (float64, float)):
            z = float64(z)
            if np.isfinite(maps_to.low) and maps_to.at_boundary(z, "low"):
                return z + self.eps
            if np.isfinite(maps_to.high) and maps_to.at_boundary(z, "high"):
                return z - self.eps
            if not maps_to.contains(z):
                raise OutOfSupportError(z, maps_to)
            return z

        arr = z.astype(float64, copy=False)
        low_mask = (
            np.isclose(arr, maps_to.low, atol=float64(1e-6))
            if np.isfinite(maps_to.low)
            else False
        )
        high_mask = (
            np.isclose(arr, maps_to.high, atol=float64(1e-6))
            if np.isfinite(maps_to.high)
            else False
        )
        adjusted = np.where(
            low_mask, arr + self.eps, np.where(high_mask, arr - self.eps, arr)
        )
        if not maps_to.contains(adjusted):
            raise OutOfSupportError(arr, maps_to)
        return adjusted

    @overload
    def safe_forward(self, x: float64) -> float64: ...
    @overload
    def safe_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def safe_forward(self, x: T) -> T:
        x = self._get_adjusted_forward(x)
        return self.forward(x)

    @overload
    def safe_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def safe_inverse(self, y: T) -> T:
        y = self._get_adjusted_inverse(y)
        return self.inverse(y)

    @overload
    def safe_grad_forward(self, x: float64) -> float64: ...
    @overload
    def safe_grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def safe_grad_forward(self, x: T) -> T:
        x = self._get_adjusted_forward(x)
        return self.grad_forward(x)

    @overload
    def safe_grad_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def safe_grad_inverse(self, y: T) -> T:
        y = self._get_adjusted_inverse(y)
        return self.grad_inverse(y)

    @overload
    def safe_log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def safe_log_det_abs_jacobian_forward(
        self, x: NDArray[float64]
    ) -> NDArray[float64]: ...

    def safe_log_det_abs_jacobian_forward(self, x: T) -> T:
        x = self._get_adjusted_forward(x)
        return self.log_det_abs_jacobian_forward(x)

    @overload
    def safe_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def safe_log_det_abs_jacobian_inverse(self, y: T) -> T:
        y = self._get_adjusted_inverse(y)
        return self.log_det_abs_jacobian_inverse(y)

    @property
    @abstractmethod
    def support(self) -> Support:
        pass

    @property
    @abstractmethod
    def maps_to(self) -> Support:
        pass

    @property
    def eps(self) -> float64:
        return float64(1e-8)
