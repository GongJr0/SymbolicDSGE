from dataclasses import dataclass
from typing import Union, Literal, Callable, cast
from numpy import float64
from numpy.typing import NDArray
import numpy as np
from functools import wraps

FLOAT_VEC_SCA = Union[float64, NDArray[float64]]


@dataclass(frozen=True)
class Support:

    low: float64
    high: float64
    low_inclusive: bool = True
    high_inclusive: bool = True

    def contains(self, x: FLOAT_VEC_SCA) -> bool:
        x_arr = np.asarray(x)
        if self.low_inclusive:
            low_check = x_arr >= self.low
        else:
            low_check = x_arr > self.low

        if self.high_inclusive:
            high_check = x_arr <= self.high
        else:
            high_check = x_arr < self.high

        return bool(np.all(low_check) and np.all(high_check))

    def at_boundary(self, x: FLOAT_VEC_SCA, bound: Literal["high", "low"]) -> bool:
        x_arr = np.asarray(x)

        if bound == "low":
            return bool(np.any(np.isclose(x_arr, self.low, atol=1e-6)))

        else:
            return bool(np.any(np.isclose(x_arr, self.high, atol=1e-6)))

    def contains_support(self, other: "Support") -> bool:
        low_check = self.low < other.low or (
            self.low == other.low and (self.low_inclusive or not other.low_inclusive)
        )
        high_check = self.high > other.high or (
            self.high == other.high
            and (self.high_inclusive or not other.high_inclusive)
        )
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


class OutOfSupportError(ValueError):
    def __init__(self, value: float64 | NDArray[float64], support: Support) -> None:
        message = f"Value(s) {value} out of support {support} for this transform."
        super().__init__(message)


class UnsetSupportError(ValueError):
    def __init__(self) -> None:
        msg = "A bounded operation was defined on a object without a support function. Please make a bug report if you encounter this error."
        super().__init__(msg)


def bounded(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self: object, x: FLOAT_VEC_SCA) -> FLOAT_VEC_SCA:
        support = getattr(self, "support", None)
        if support is None:
            raise UnsetSupportError()
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        return cast(FLOAT_VEC_SCA, func(self, x))

    return wrapper
