# cython: language_level=3
"""Public Monte Carlo step statuses and their descriptions.

Native values remain defined by their owning headers. Custom transforms must
return a code represented here. Runner control outcomes are not step statuses.
"""

from enum import IntEnum, unique


cdef extern from "sdsge_common.h":
    int c_SDSGE_OK "SDSGE_OK"

cdef extern from "../diag/diag.h":
    int c_DIAG_BAD_SHAPE "DIAG_BAD_SHAPE"
    int c_DIAG_LINALG "DIAG_LINALG"
    int c_DIAG_UDEF_VARIANCE "DIAG_UDEF_VARIANCE"
    int c_DIAG_BAD_LAG "DIAG_BAD_LAG"
    int c_DIAG_INSUFFICIENT_SAMPLES "DIAG_INSUFFICIENT_SAMPLES"
    int c_DIAG_ITERATIVE_NONCONVERGENCE "DIAG_ITERATIVE_NONCONVERGENCE"
    int c_DIAG_BAD_PARAMETER "DIAG_BAD_PARAMETER"
    int c_DIAG_FALLBACK "DIAG_FALLBACK"

cdef extern from "../kalman/kalman.h":
    int c_KF_ERR_SHAPE_MISMATCH "KF_ERR_SHAPE_MISMATCH"
    int c_KF_ERR_MATRIX_CONDITION "KF_ERR_MATRIX_CONDITION"

cdef extern from "runner.h":
    int c_SDSGE_MC_RUN_BAD_ARG "SDSGE_MC_RUN_BAD_ARG"
    int c_SDSGE_MC_NOT_RUN "SDSGE_MC_NOT_RUN"

cdef extern from "transforms.h":
    int c_SDSGE_TRANSFORM_BAD_ARG "SDSGE_TRANSFORM_BAD_ARG"
    int c_SDSGE_TRANSFORM_OUT_OF_DOMAIN "SDSGE_TRANSFORM_OUT_OF_DOMAIN"
    int c_SDSGE_TRANSFORM_CUSTOM_FAILED "SDSGE_TRANSFORM_CUSTOM_FAILED"
cdef extern from "../regression/regression.h":
    int c_REGRESSION_RANK_DEFICIENT "REGRESSION_RANK_DEFICIENT"
    int c_REGRESSION_NON_CONVERGENT "REGRESSION_NON_CONVERGENT"


@unique
class MCStatus(IntEnum):
    """Codes produced by Monte Carlo steps, including post-loop failures."""

    SUCCESS = c_SDSGE_OK
    POSTPROC_FAILED = -1

    DIAG_BAD_SHAPE = c_DIAG_BAD_SHAPE
    DIAG_LINALG = c_DIAG_LINALG
    DIAG_UDEF_VARIANCE = c_DIAG_UDEF_VARIANCE
    DIAG_BAD_LAG = c_DIAG_BAD_LAG
    DIAG_INSUFFICIENT_SAMPLES = c_DIAG_INSUFFICIENT_SAMPLES
    DIAG_ITERATIVE_NONCONVERGENCE = c_DIAG_ITERATIVE_NONCONVERGENCE
    DIAG_BAD_PARAMETER = c_DIAG_BAD_PARAMETER
    DIAG_FALLBACK = c_DIAG_FALLBACK

    FILTER_SHAPE_MISMATCH = c_KF_ERR_SHAPE_MISMATCH
    FILTER_MATRIX_CONDITION = c_KF_ERR_MATRIX_CONDITION

    BAD_ARG = c_SDSGE_MC_RUN_BAD_ARG
    NOT_RUN = c_SDSGE_MC_NOT_RUN

    TRANSFORM_BAD_ARG = c_SDSGE_TRANSFORM_BAD_ARG
    TRANSFORM_OUT_OF_DOMAIN = c_SDSGE_TRANSFORM_OUT_OF_DOMAIN
    TRANSFORM_CUSTOM_FAILED = c_SDSGE_TRANSFORM_CUSTOM_FAILED

    REGRESSION_RANK_DEFICIENT = c_REGRESSION_RANK_DEFICIENT
    REGRESSION_NON_CONVERGENT = c_REGRESSION_NON_CONVERGENT

    @property
    def message(self) -> str:
        """Return the explanation of this status without step or run context."""
        return _MESSAGES[self]


_MESSAGES = {
    MCStatus.SUCCESS: "The step completed successfully.",
    MCStatus.POSTPROC_FAILED: (
        "The post-processing callable or artifact normalization raised an exception."
    ),
    MCStatus.DIAG_BAD_SHAPE: "The diagnostic inputs have invalid dimensions.",
    MCStatus.DIAG_LINALG: "A linear algebra operation in the diagnostic failed.",
    MCStatus.DIAG_UDEF_VARIANCE: "The diagnostic variance is undefined.",
    MCStatus.DIAG_BAD_LAG: "The diagnostic lag specification is invalid.",
    MCStatus.DIAG_INSUFFICIENT_SAMPLES: (
        "There are too few samples to evaluate the diagnostic."
    ),
    MCStatus.DIAG_ITERATIVE_NONCONVERGENCE: (
        "An iterative algorithm in the diagnostic did not converge."
    ),
    MCStatus.DIAG_BAD_PARAMETER: "A diagnostic parameter is invalid.",
    MCStatus.DIAG_FALLBACK: (
        "The diagnostic design is rank-deficient and requires a fallback solver."
    ),
    MCStatus.FILTER_SHAPE_MISMATCH: "The filter inputs have incompatible dimensions.",
    MCStatus.FILTER_MATRIX_CONDITION: (
        "A filter matrix could not be factorized with the required conditioning."
    ),
    MCStatus.BAD_ARG: "The runner received an invalid argument or input binding.",
    MCStatus.NOT_RUN: "The step was skipped due to its source step(s) failing.",
    MCStatus.TRANSFORM_BAD_ARG: "A transform argument or input dimension is invalid.",
    MCStatus.TRANSFORM_OUT_OF_DOMAIN: (
        "A transform input is outside the operation's domain."
    ),
    MCStatus.TRANSFORM_CUSTOM_FAILED: (
        "A custom transform callable returned a non-zero status code. "

    ),
    MCStatus.REGRESSION_RANK_DEFICIENT: "The regression system is rank-deficient.",
    MCStatus.REGRESSION_NON_CONVERGENT: "The regression solver did not converge.",
}
