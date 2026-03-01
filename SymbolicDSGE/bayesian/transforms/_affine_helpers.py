from numpy import float64
from numpy.typing import NDArray


def affine_to_unit(
    x: float64 | NDArray[float64],
    low: float64,
    high: float64,
) -> float64 | NDArray[float64]:
    return (x - low) / (high - low)


def unit_to_affine(
    z: float64 | NDArray[float64],
    low: float64,
    high: float64,
) -> float64 | NDArray[float64]:
    return low + (high - low) * z
