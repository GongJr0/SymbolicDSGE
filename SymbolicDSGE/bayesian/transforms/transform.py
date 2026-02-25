from enum import StrEnum
from abc import ABC, abstractmethod
from numpy import float64
from numpy.typing import NDArray
from typing import TypeVar, overload

from ..support import Support

T = TypeVar("T", float64, NDArray[float64])


class TransformMethod(StrEnum):
    IDENTITY = "identity"  # (-inf, inf)
    LOG = "log"  # (0, inf)  via y=log(x)
    SOFTPLUS = "softplus"  # (0, inf)  via x=softplus(y)  (sampler-friendly alternative)
    LOGIT = "logit"  # (0, 1)    via y=log(x/(1-x))
    PROBIT = "probit"  # (0, 1)    via y=Phi^{-1}(x)

    AFFINE_LOGIT = "affine_logit"  # (a, b)    via x=a+(b-a)*sigmoid(y)
    AFFINE_PROBIT = "affine_probit"  # (a, b)    via x=a+(b-a)*Phi(y)

    LOWER_BOUNDED = "lower_bounded"  # (a, inf)  via x=a+exp(y)   (or a+softplus(y))
    UPPER_BOUNDED = "upper_bounded"  # (-inf, b) via x=b-exp(y)   (or b-softplus(y))

    ORDERED = "ordered"  # x1 < x2 < ... (thresholds / regime cutoffs)

    SIMPLEX = (
        "simplex"  # weights on simplex (sum=1, each>0) via softmax / stick-breaking
    )

    CHOLESKY_COV = "cholesky_cov"  # SPD covariance via unconstrained -> L -> Sigma=LL'
    CHOLESKY_CORR = "cholesky_corr"  # correlation matrix (LKJ) via unconstrained -> corr Cholesky factor


class Transform(ABC):

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


class OutOfSupportError(ValueError):
    def __init__(self, value: float64 | NDArray[float64], support: Support):
        message = f"Value(s) {value} out of support {support} for this transform."
        super().__init__(message)
