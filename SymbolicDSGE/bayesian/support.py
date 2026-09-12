"""Support specification for distributions and transforms."""

from dataclasses import dataclass
from typing import Union, Literal, Callable, cast
from numpy import float64
from numpy.typing import NDArray
import numpy as np
from functools import wraps
from numba import njit

FLOAT_VEC_SCA = Union[float64, NDArray[float64]]


@njit(cache=True)
def _contains_scalar(
    low: float64, high: float64, low_inclusive: bool, high_inclusive: bool, x: float64
) -> bool:
    if low_inclusive:
        if x < low:
            return False
    else:
        if x <= low:
            return False

    if high_inclusive:
        if x > high:
            return False
    else:
        if x >= high:
            return False

    return True


@njit(cache=True)
def _contains_vectorized(
    low: float64,
    high: float64,
    low_inclusive: bool,
    high_inclusive: bool,
    x: NDArray[float64],
) -> bool:
    flat = x.ravel()
    for i in range(flat.size):
        xi = flat[i]

        if low_inclusive:
            if xi < low:
                return False
        else:
            if xi <= low:
                return False

        if high_inclusive:
            if xi > high:
                return False
        else:
            if xi >= high:
                return False

    return True


@njit(cache=True)
def _at_boundary_scalar(
    x: float64, bound: str, lim: float64, atol: float64 = float64(1e-6)
) -> bool:
    if bound == "low":
        return bool(np.isclose(x, lim, atol=atol))
    return bool(np.isclose(x, lim, atol=atol))


@njit(cache=True)
def _at_boundary_vectorized(
    x: NDArray[float64], bound: str, lim: float64, atol: float64 = float64(1e-6)
) -> bool:
    if bound == "low":
        return bool(np.any(np.isclose(x, lim, atol=atol)))
    return bool(np.any(np.isclose(x, lim, atol=atol)))


@dataclass(frozen=True)
class Support:
    """Support of distributions and transforms.

    Attributes
    ----------
    low : float64
        Lower bound of the support.
    high : float64
        Upper bound of the support.
    low_inclusive : bool
        Whether the lower bound is inclusive.
    high_inclusive : bool
        Whether the upper bound is inclusive.

    """

    low: float64
    high: float64
    low_inclusive: bool = True
    high_inclusive: bool = True

    def contains(self, x: FLOAT_VEC_SCA) -> bool:
        """Check if a value or array is within the support.

        Parameters
        ----------
        x : FLOAT_VEC_SCA
            Value(s) to check for inclusion in the support.

        Returns
        -------
        bool
            Whether the value(s) are within the support, considering inclusivity of bounds.

        """
        if isinstance(x, (float64, float)):
            x = float64(x)
            return _contains_scalar(
                self.low, self.high, self.low_inclusive, self.high_inclusive, x
            )
        x = x.astype(float64)
        return _contains_vectorized(
            self.low, self.high, self.low_inclusive, self.high_inclusive, x
        )

    def at_boundary(self, x: FLOAT_VEC_SCA, bound: Literal["high", "low"]) -> bool:
        """Check if a value or array is at the specified boundary of the support.

        Parameters
        ----------
        x : FLOAT_VEC_SCA
            Value(s) to check for being at the boundary.
        bound : Literal["high", "low"]
            Which boundary to check against. "high" for the upper bound, "low" for the lower bound.

        Returns
        -------
        bool
            Whether the value(s) are at the specified boundary, considering inclusivity of bounds.

        """
        lim = self.low if bound == "low" else self.high
        if isinstance(x, (float64, float)):
            x = float64(x)
            return _at_boundary_scalar(x, bound, lim)
        x = x.astype(float64)
        return _at_boundary_vectorized(x, bound, lim)

    def contains_support(self, other: "Support") -> bool:
        """Check if this support fully contains another support.

        Parameters
        ----------
        other : "Support"
            Support to check for containment within this support.

        Returns
        -------
        bool
            Whether this support fully contains the other support, considering inclusivity of bounds.

        """
        # Ignore inclusivity, eps injection should handle boundary cases
        high_check = self.high >= other.high
        low_check = self.low <= other.low
        return bool(low_check and high_check)

    @property
    def is_finite(self) -> bool:
        """Whether both bounds of the support are finite."""
        return bool(np.isfinite(self.low) and np.isfinite(self.high))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Support):
            raise NotImplementedError(
                "Equality is only implemented between Support instances."
            )
        return (
            self.low == other.low
            and self.high == other.high
            and self.low_inclusive == other.low_inclusive
            and self.high_inclusive == other.high_inclusive
        )

    def __lshift__(self, other: "Support") -> bool:
        return self.contains_support(other)

    def __rlshift__(self, other: "Support") -> bool:
        return other.contains_support(self)


class OutOfSupportError(ValueError):
    """Raised when a value or array is outside the defined support of a distribution or transform."""

    def __init__(self, value: float64 | NDArray[float64], support: Support) -> None:
        message = f"Value(s) {value} out of support {support} for this transform."
        super().__init__(message)


class UnsetSupportError(ValueError):
    """Raised when a bounded operation is attempted on an object without a defined support."""

    def __init__(self) -> None:
        msg = "A bounded operation was defined on a object without a support function. Please make a bug report if you encounter this error."
        super().__init__(msg)


def bounded(
    func: Callable | None = None,
    *,
    domain: Literal["support", "maps_to"] = "support",
) -> Callable:
    """Bound-check the decorated function against the specified domain of the object.

    Parameters
    ----------
    func : Callable | None
        Function returning numeric scalar or array.
    domain : Literal["support", "maps_to"]
        Which domain of the object to check against.

    Returns
    -------
    Callable
        Decorated function that checks if the input is within the specified domain before executing.

    """

    def _decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self: object, x: FLOAT_VEC_SCA) -> FLOAT_VEC_SCA:
            target = getattr(self, domain, None)
            if target is None:
                raise UnsetSupportError()
            if not target.contains(x):
                raise OutOfSupportError(x, target)
            return cast(FLOAT_VEC_SCA, fn(self, x))

        return wrapper

    if func is None:
        return _decorate
    return _decorate(func)
