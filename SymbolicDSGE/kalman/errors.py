from numpy import float64
from enum import IntEnum
import numpy as np
from typing import Any


class ShapeMismatchError(Exception):
    def __init__(self, *args: Any) -> None:
        message = f"Matrix '{args[0]}' has incompatible shape. Expected: {args[1]}, got: {args[2]}."
        super().__init__(message)


class MatrixConditionError(Exception):
    def __init__(self) -> None:
        message = f"Matrix(s) is ill-conditioned."
        super().__init__(message)


class MemoryAllocationError(Exception):
    def __init__(self) -> None:
        message = "Memory allocation failed. (Possibly due to insufficient memory.)"
        super().__init__(message)


class ErrorCode(IntEnum):
    """Mirrors the KF_* codes in _ckernels/kalman/kalman.h."""

    SUCCESS = 0
    SHAPE_MISMATCH = -1101
    MATRIX_CONDITION = -1102
    LINALG_ERROR = -1103
    ALLOC_ERROR = -1104


def get_error_constructor(code: ErrorCode) -> type[Exception]:
    if code == ErrorCode.SHAPE_MISMATCH:
        return ShapeMismatchError
    elif code == ErrorCode.MATRIX_CONDITION:
        return MatrixConditionError
    elif code == ErrorCode.LINALG_ERROR:
        return np.linalg.LinAlgError
    elif code == ErrorCode.ALLOC_ERROR:
        return MemoryAllocationError
    else:
        raise ValueError(f"Unknown error code: {code}")
