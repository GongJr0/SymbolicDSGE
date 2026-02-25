from dataclasses import dataclass
from typing import Union, Literal
from numpy import float64
from numpy.typing import NDArray
import numpy as np

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
            return bool(np.any(x_arr == self.low))

        else:
            return bool(np.any(x_arr == self.high))

    def contains_support(self, other: "Support") -> bool:
        low_check = self.low < other.low or (
            self.low == other.low and (self.low_inclusive or not other.low_inclusive)
        )
        high_check = self.high > other.high or (
            self.high == other.high
            and (self.high_inclusive or not other.high_inclusive)
        )
        return bool(low_check and high_check)
