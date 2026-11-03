"""Error types raised by the Kalman filter implementation."""

from enum import IntEnum, unique
from typing import Any

from .._ckernels.kalman._kalman import (
    KF_OK,
    KF_ERR_P_FACTORIZATION,
    KF_ERR_Q_FACTORIZATION,
    KF_ERR_S_FACTORIZATION,
    KF_ERR_BAD_HYPERPARAMS,
)


@unique
class ErrorCode(IntEnum):
    """Codes from _ckernels/kalman/kalman.h that are returned as status codes from the Kalman filter implementations."""

    SUCCESS = KF_OK
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
            "{name}Matrix factorization failed. The matrix may be ill-conditioned, "
            "singular, or not positive definite."
        )
        match code:
            case ErrorCode.P_FACTORIZATION:
                msg = (
                    default_msg.format(name="P ")
                    + " A bad P0 initialization or a model without shocks "
                    "can lead to this error in UKF."
                )
            case ErrorCode.Q_FACTORIZATION:
                msg = default_msg.format(name="Q ")
            case ErrorCode.S_FACTORIZATION:
                msg = default_msg.format(name="S ")
            case _:
                msg = default_msg.format(name="")
        super().__init__(msg)


class HyperparameterError(Exception):
    """Raised when hyperparameters are invalid in the Kalman filter implementation."""

    def __init__(self, *args: Any) -> None:
        message = (
            "The alpha and kappa combination does not produce positive weights. "
            "Please check the hyperparameter values."
        )
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
    if (
        code == ErrorCode.P_FACTORIZATION
        or code == ErrorCode.Q_FACTORIZATION
        or code == ErrorCode.S_FACTORIZATION
    ):
        return MatrixFactorizationError
    elif code == ErrorCode.BAD_HYPERPARAMS:
        return HyperparameterError
    else:
        raise ValueError(f"Unknown error code: {code}")
