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

    low: float64
    high: float64
    low_inclusive: bool = True
    high_inclusive: bool = True

    def contains(self, x: FLOAT_VEC_SCA) -> bool:
        if isinstance(x, (float64, float)):
            x = float64(x)
            return cast(
                bool,
                _contains_scalar(
                    self.low, self.high, self.low_inclusive, self.high_inclusive, x
                ),
            )
        x = x.astype(float64)
        return cast(
            bool,
            _contains_vectorized(
                self.low, self.high, self.low_inclusive, self.high_inclusive, x
            ),
        )

    def at_boundary(self, x: FLOAT_VEC_SCA, bound: Literal["high", "low"]) -> bool:
        lim = self.low if bound == "low" else self.high
        if isinstance(x, (float64, float)):
            x = float64(x)
            return cast(bool, _at_boundary_scalar(x, bound, lim))
        x = x.astype(float64)
        return cast(bool, _at_boundary_vectorized(x, bound, lim))

    def contains_support(self, other: "Support") -> bool:
        # Ignore inclusivity, eps injection should handle boundary cases
        high_check = self.high >= other.high
        low_check = self.low <= other.low
        return bool(low_check and high_check)

    @property
    def is_finite(self) -> bool:
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
    def __init__(self, value: float64 | NDArray[float64], support: Support) -> None:
        message = f"Value(s) {value} out of support {support} for this transform."
        super().__init__(message)


class UnsetSupportError(ValueError):
    def __init__(self) -> None:
        msg = "A bounded operation was defined on a object without a support function. Please make a bug report if you encounter this error."
        super().__init__(msg)


def bounded(
    func: Callable | None = None,
    *,
    domain: Literal["support", "maps_to"] = "support",
) -> Callable:
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
