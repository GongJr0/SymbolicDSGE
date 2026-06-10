from enum import IntEnum


class TestStatus(IntEnum):
    __test__ = False

    OK = 0
    BAD_SHAPE = -1
    LINALG = -2
    UDEF_VARIANCE = -3
    BAD_LAG = -4
    INSUFFICIENT_SAMPLES = -5
    ITERATIVE_ALG_NONCONVERGENCE = -6
    BAD_PARAMETER = -7
