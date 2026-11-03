"""Error types raised by the Kalman filter implementation."""

from enum import IntEnum, unique
from typing import Any

from .._ckernels.kalman._kalman import (
    KF_OK,
    KF_ERR_SHAPE_MISMATCH,
    KF_ERR_ALLOC,
    KF_ERR_LYAPUNOV_CONVERGENCE,
    KF_ERR_P_FACTORIZATION,
    KF_ERR_Q_FACTORIZATION,
    KF_ERR_S_FACTORIZATION,
    KF_ERR_BAD_HYPERPARAMS,
)


@unique
class ErrorCode(IntEnum):
    """Mirrors the KF_* codes in _ckernels/kalman/kalman.h."""

    SUCCESS = KF_OK
    SHAPE_MISMATCH = KF_ERR_SHAPE_MISMATCH
    ALLOC_ERROR = KF_ERR_ALLOC
    LYAPUNOV_CONVERGENCE = KF_ERR_LYAPUNOV_CONVERGENCE
    P_FACTORIZATION = KF_ERR_P_FACTORIZATION
    Q_FACTORIZATION = KF_ERR_Q_FACTORIZATION
    S_FACTORIZATION = KF_ERR_S_FACTORIZATION
    BAD_HYPERPARAMS = KF_ERR_BAD_HYPERPARAMS


class ShapeMismatchError(Exception):
    """Raised on unexpected matrix shapes in the Kalman filter implementation."""

    def __init__(self, *args: Any) -> None:
        message = f"Matrix '{args[0]}' has incompatible shape. Expected: {args[1]}, got: {args[2]}."
        super().__init__(message)


class MatrixFactorizationError(Exception):
    """Raised when a matrix is ill-conditioned in the Kalman filter implementation."""

    def __init__(self, code: ErrorCode) -> None:
        default_msg = (
            "{name!r}Matrix factorization failed. The matrix may be ill-conditioned, "
            "singular, or not positive definite."
        )
        match code:
            case ErrorCode.P_FACTORIZATION:
                msg = (
                    default_msg.format(name="P ")
                    + " Models without shocks can lead to "
                    "this error on the default P0 handling."
                )
            case ErrorCode.Q_FACTORIZATION:
                msg = default_msg.format(name="Q ")
            case ErrorCode.S_FACTORIZATION:
                msg = default_msg.format(name="S ")
            case _:
                msg = default_msg.format(name="")
        super().__init__(msg)


class MemoryAllocationError(Exception):
    """Raised when a native memory allocation fails."""

    def __init__(self, *args: Any) -> None:
        message = "Memory allocation failed. (Possibly due to insufficient memory.)"
        super().__init__(message)


def get_error_constructor(code: ErrorCode) -> type[Exception]:
    """Map an error code to the corresponding exception class.

    Parameters
    ----------
    code : ErrorCode
        Native error code returned by the Kalman filter implementation.

    Returns
    -------
    type[Exception]
        Python error wrapper class corresponding to the given error code.

    """
    if code == ErrorCode.SHAPE_MISMATCH:
        return ShapeMismatchError
    elif (
        code == ErrorCode.P_FACTORIZATION
        or code == ErrorCode.Q_FACTORIZATION
        or code == ErrorCode.S_FACTORIZATION
    ):
        return MatrixFactorizationError
    elif code == ErrorCode.ALLOC_ERROR:
        return MemoryAllocationError
    else:
        raise ValueError(f"Unknown error code: {code}")
