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
    TANH = "tanh"  # (-1, 1)   via y=tanh^{-1}(x)

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
    """Base class for transformations applied to prior distributions in any a priori routine."""

    @abstractmethod
    def __repr__(self) -> str: ...

    def to_spec(self) -> tuple[str, dict[str, float]]:
        """Return ``(method, kwargs)`` for serialization into a ``PriorSpec``.

        ``kwargs`` is keyed as the transform's constructor expects, so
        ``get_transform(method)(**kwargs)`` rebuilds it. Subclasses declare
        their own mapping; the default raises for transforms without a
        round-trippable spec (e.g. ``cholesky_corr``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support spec serialization."
        )

    @overload
    def forward(self, x: float64) -> float64: ...
    @overload
    def forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def forward(self, x: T) -> T:
        """Forward transformation from parameter space to unconstrained space.

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space to be transformed.

        Returns
        -------
        T
            Parameter value(s) in the unconstrained space after applying the transformation.

        """
        pass

    @overload
    def inverse(self, y: float64) -> float64: ...
    @overload
    def inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def inverse(self, y: T) -> T:
        """Inverse transformation from unconstrained space back to parameter space.

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space to be transformed back.

        Returns
        -------
        T
            Parameter value(s) in the constrained space after applying the inverse transformation.

        """
        pass

    @overload
    def grad_forward(self, x: float64) -> float64: ...
    @overload
    def grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def grad_forward(self, x: T) -> T:
        """Gradient of the forward transformation with respect to the input parameter(s).

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space at which to evaluate the gradient.

        Returns
        -------
        T
            Gradient of the forward transformation evaluated at the given parameter value(s).

        """
        pass

    @overload
    def grad_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def grad_inverse(self, y: T) -> T:
        """Gradient of the inverse transformation with respect to the input parameter(s).

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space at which to evaluate the gradient.

        Returns
        -------
        T
            Gradient of the inverse transformation evaluated at the given parameter value(s).

        """
        pass

    @overload
    def log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def log_det_abs_jacobian_forward(self, x: T) -> T:
        """Log absolute determinant of the Jacobian of the forward transformation, evaluated at the given parameter value(s).

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space at which to evaluate the log determinant of the Jacobian.

        Returns
        -------
        T
            Log absolute determinant of the Jacobian of the forward transformation evaluated at the given parameter value(s).

        """
        pass

    @overload
    def log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def log_det_abs_jacobian_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    @abstractmethod
    def log_det_abs_jacobian_inverse(self, y: T) -> T:
        """Log absolute determinant of the Jacobian of the inverse transformation, evaluated at the given parameter value(s).

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space at which to evaluate the log determinant of the Jacobian.

        Returns
        -------
        T
            Log absolute determinant of the Jacobian of the inverse transformation evaluated at the given parameter value(s).

        """
        pass

    @overload
    def grad_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def grad_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    @abstractmethod
    def grad_log_det_abs_jacobian_inverse(self, y: T) -> T:
        """Gradient of `log(det(abs(Jacobian)))` of the inverse transformation, evaluated at the given parameter value(s).

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space at which to evaluate the gradient of the log determinant of the Jacobian.

        Returns
        -------
        T
            Gradient of the log absolute determinant of the Jacobian of the inverse transformation evaluated at the given parameter value(s).

        """
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
        """Forward transformation from parameter space to unconstrained space, with adjustments for boundary values.

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space to be transformed.

        Returns
        -------
        T
            Parameter value(s) in the unconstrained space after applying the transformation, with adjustments for boundary values.

        """
        x = self._get_adjusted_forward(x)
        return self.forward(x)

    @overload
    def safe_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def safe_inverse(self, y: T) -> T:
        """Inverse transformation from unconstrained space back to parameter space, with adjustments for boundary values.

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space to be transformed back.

        Returns
        -------
        T
            Parameter value(s) in the constrained space after applying the inverse transformation, with adjustments for boundary values.

        """
        y = self._get_adjusted_inverse(y)
        return self.inverse(y)

    @overload
    def safe_grad_forward(self, x: float64) -> float64: ...
    @overload
    def safe_grad_forward(self, x: NDArray[float64]) -> NDArray[float64]: ...

    def safe_grad_forward(self, x: T) -> T:
        """Gradient of the forward transformation with respect to the input parameter(s), with adjustments for boundary values.

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space at which to evaluate the gradient, with adjustments for boundary values.

        Returns
        -------
        T
            Gradient of the forward transformation evaluated at the given parameter value(s), with adjustments for boundary values.

        """
        x = self._get_adjusted_forward(x)
        return self.grad_forward(x)

    @overload
    def safe_grad_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_grad_inverse(self, y: NDArray[float64]) -> NDArray[float64]: ...

    def safe_grad_inverse(self, y: T) -> T:
        """Gradient of the inverse transformation with respect to the input parameter(s), with adjustments for boundary values.

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space at which to evaluate the gradient, with adjustments for boundary values.

        Returns
        -------
        T
            Gradient of the inverse transformation evaluated at the given parameter value(s), with adjustments for boundary values.

        """
        y = self._get_adjusted_inverse(y)
        return self.grad_inverse(y)

    @overload
    def safe_log_det_abs_jacobian_forward(self, x: float64) -> float64: ...
    @overload
    def safe_log_det_abs_jacobian_forward(
        self, x: NDArray[float64]
    ) -> NDArray[float64]: ...

    def safe_log_det_abs_jacobian_forward(self, x: T) -> T:
        """Log absolute determinant of the Jacobian of the forward transformation, evaluated at the given parameter value(s), with adjustments for boundary values.

        Parameters
        ----------
        x : T
            Parameter value(s) in the constrained space at which to evaluate the log determinant of the Jacobian, with adjustments for boundary values.

        Returns
        -------
        T
            Log absolute determinant of the Jacobian of the forward transformation evaluated at the given parameter value(s), with adjustments for boundary values.

        """
        x = self._get_adjusted_forward(x)
        return self.log_det_abs_jacobian_forward(x)

    @overload
    def safe_log_det_abs_jacobian_inverse(self, y: float64) -> float64: ...
    @overload
    def safe_log_det_abs_jacobian_inverse(
        self, y: NDArray[float64]
    ) -> NDArray[float64]: ...

    def safe_log_det_abs_jacobian_inverse(self, y: T) -> T:
        """Log absolute determinant of the Jacobian of the inverse transformation, evaluated at the given parameter value(s), with adjustments for boundary values.

        Parameters
        ----------
        y : T
            Parameter value(s) in the unconstrained space at which to evaluate the log determinant of the Jacobian, with adjustments for boundary values.

        Returns
        -------
        T
            Log absolute determinant of the Jacobian of the inverse transformation evaluated at the given parameter value(s), with adjustments for boundary values.

        """
        y = self._get_adjusted_inverse(y)
        return self.log_det_abs_jacobian_inverse(y)

    @property
    @abstractmethod
    def support(self) -> Support:
        """Support of the parameter space for this transformation."""
        pass

    @property
    @abstractmethod
    def maps_to(self) -> Support:
        """Support of the unconstrained space that this transformation maps to."""
        pass

    @property
    def eps(self) -> float64:
        """Additive epsilon used to adjust parameter values at the boundaries of the support to avoid numerical issues."""
        return float64(1e-8)
