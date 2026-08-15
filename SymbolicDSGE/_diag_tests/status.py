from enum import IntEnum


class TestStatus(IntEnum):
    """Mirrors the DIAG_* codes in _ckernels/diag/diag.h."""

    __test__ = False

    OK = 0
    BAD_SHAPE = -1001
    LINALG = -1002
    UDEF_VARIANCE = -1003
    BAD_LAG = -1004
    INSUFFICIENT_SAMPLES = -1005
    ITERATIVE_ALG_NONCONVERGENCE = -1006
    BAD_PARAMETER = -1007
