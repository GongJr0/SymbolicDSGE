from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

class StepArenas:
    float_in_work: NDArray[np.float64]
    int_in_work: NDArray[np.int64]
    float_live_out: NDArray[np.float64]
    int_live_out: NDArray[np.int64]
    float_retained: NDArray[np.float64]
    int_retained: NDArray[np.int64]
    retained_reps: NDArray[np.int64]
    retained_row_by_rep: NDArray[np.int64]

class ArenaAllocation:
    n_rep: int
    n_workers: int
    plan: dict[str, Any]
    steps: dict[str, StepArenas]
    failure_step_by_rep: NDArray[np.int64]
    failure_status_by_rep: NDArray[np.int64]

def resolve_retention(
    n_retain: int, n_rep: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def resolve_n_workers(n_jobs: int | None = None) -> int: ...
def allocate_arenas(
    plan: Mapping[str, Any], n_rep: int, n_jobs: int | None = None
) -> ArenaAllocation: ...
