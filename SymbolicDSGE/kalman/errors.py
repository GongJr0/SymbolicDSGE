from numpy import float64
from enum import IntEnum
import numpy as np
from typing import Any


class ComplexMatrixError(Exception):
    def __init__(self, *args: Any) -> None:
        message = f"Matrix '{args[0]}' has significant imaginary parts (max abs imag: {args[1]})."
        super().__init__(message)


class ShapeMismatchError(Exception):
    def __init__(self, *args: Any) -> None:
        message = f"Matrix '{args[0]}' has incompatible shape. Expected: {args[1]}, got: {args[2]}."
        super().__init__(message)


class MatrixConditionError(Exception):
    def __init__(self, *args: Any) -> None:
        message = f"Matrix(s) is ill-conditioned. Condition number(s): {args}."
        super().__init__(message)


class ErrorCode(IntEnum):
    SUCCESS = 0
    COMPLEX_MATRIX = -1
    SHAPE_MISMATCH = -2
    MATRIX_CONDITION = -3
    LINALG_ERROR = -4


def get_error_constructor(code: ErrorCode) -> type[Exception]:
    if code == ErrorCode.COMPLEX_MATRIX:
        return ComplexMatrixError
    elif code == ErrorCode.SHAPE_MISMATCH:
        return ShapeMismatchError
    elif code == ErrorCode.MATRIX_CONDITION:
        return MatrixConditionError
    elif code == ErrorCode.LINALG_ERROR:
        return np.linalg.LinAlgError
    else:
        raise ValueError(f"Unknown error code: {code}")
