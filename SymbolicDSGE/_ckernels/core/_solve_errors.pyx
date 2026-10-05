# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""The status codes a native solve returns, enum-ified for the Python surface."""

from enum import IntEnum, unique


cdef extern from "klein_classify.h":
    int SDSGE_KLEIN_RANK_FAIL
    int SDSGE_KLEIN_INFINITE_ROOT


cdef extern from "klein_solve.h":
    int SDSGE_KLEIN_SOLVE_QZ
    int SDSGE_KLEIN_SOLVE_SHOCK_SINGULAR
    int SDSGE_KLEIN_SOLVE_QR
    int SDSGE_KLEIN_SOLVE_STATIC_SINGULAR
    int SDSGE_KLEIN_NO_STABLE_SOLUTION


cdef extern from "second_order.h":
    int SDSGE_SECOND_ORDER_SINGULAR
    int SDSGE_SECOND_ORDER_RISK


cdef extern from "steady_state.h":
    int SDSGE_NEWTON_SINGULAR
    int SDSGE_NEWTON_NO_CONVERGE


@unique
class SolveStatus(IntEnum):
    """The value a solve prepends to its result tuple.

    Grouped by the stage that produces them, which is also what decides how much
    of the diagnostic a failure can carry: the steady state resolves before the
    partition exists, and the eigenvalues land at the top of the post-proc.
    """

    NEWTON_SINGULAR = SDSGE_NEWTON_SINGULAR
    NEWTON_NO_CONVERGE = SDSGE_NEWTON_NO_CONVERGE

    QR = SDSGE_KLEIN_SOLVE_QR
    QZ = SDSGE_KLEIN_SOLVE_QZ

    RANK_FAIL = SDSGE_KLEIN_RANK_FAIL
    INFINITE_ROOT = SDSGE_KLEIN_INFINITE_ROOT
    NO_STABLE_SOLUTION = SDSGE_KLEIN_NO_STABLE_SOLUTION
    STATIC_SINGULAR = SDSGE_KLEIN_SOLVE_STATIC_SINGULAR
    SHOCK_SINGULAR = SDSGE_KLEIN_SOLVE_SHOCK_SINGULAR

    SECOND_ORDER_SINGULAR = SDSGE_SECOND_ORDER_SINGULAR
    SECOND_ORDER_RISK = SDSGE_SECOND_ORDER_RISK
