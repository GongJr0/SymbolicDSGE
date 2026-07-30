from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from SymbolicDSGE._ckernels.monte_carlo._arenas import (
    allocate_arenas,
    resolve_n_workers,
    resolve_retention,
)
from SymbolicDSGE.monte_carlo.allocation import ArenaSize, FieldLayout, StepBufferPlan


@pytest.mark.parametrize(
    ("n_retain", "expected_reps"),
    [
        (-1, [0, 1, 2, 3, 4]),
        (0, []),
        (1, [0]),
        (3, [0, 2, 4]),
        (4, [0, 1, 2, 4]),
    ],
)
def test_resolve_retention_selects_exact_deterministic_rows(
    n_retain: int,
    expected_reps: list[int],
) -> None:
    retained_reps, row_by_rep = resolve_retention(n_retain, 5)

    assert retained_reps.tolist() == expected_reps
    assert row_by_rep.tolist() == [
        expected_reps.index(rep) if rep in expected_reps else -1 for rep in range(5)
    ]


@pytest.mark.parametrize("n_retain", (-2, 6))
def test_resolve_retention_rejects_invalid_counts(n_retain: int) -> None:
    with pytest.raises(ValueError):
        resolve_retention(n_retain, 5)


@pytest.mark.parametrize(
    ("n_jobs", "expected"),
    [(None, 1), (1, 1), (3, 3), (-1000, 1)],
)
def test_resolve_n_workers_supports_joblib_style_counts(
    n_jobs: int | None,
    expected: int,
) -> None:
    assert resolve_n_workers(n_jobs) == expected


@pytest.mark.parametrize("n_jobs", (0, True, 1.5, "2"))
def test_resolve_n_workers_rejects_invalid_values(n_jobs: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_n_workers(n_jobs)


def test_allocate_arenas_uses_plan_sizes_and_compact_retained_rows() -> None:
    plan = {
        "transform": StepBufferPlan(
            name="transform",
            input_size=ArenaSize(11, 2),
            output_size=ArenaSize(6, 1),
            out_fields={
                "payload": FieldLayout((2, 3), 6, np.dtype(np.float64), 0),
                "status": FieldLayout((), 1, np.dtype(np.int64), 0),
            },
            n_retain=3,
        ),
        "test": StepBufferPlan(
            name="test",
            input_size=ArenaSize(),
            output_size=ArenaSize(1, 1),
            out_fields={},
            n_retain=0,
        ),
    }

    allocation = allocate_arenas(plan, 8, n_jobs=2)
    transform = allocation.steps["transform"]
    test = allocation.steps["test"]

    assert allocation.n_rep == 8
    assert allocation.n_workers == 2
    assert allocation.plan == plan
    not_run = np.iinfo(np.int64).min
    assert allocation.failure_step_by_rep.tolist() == [not_run] * 8
    assert allocation.failure_status_by_rep.tolist() == [not_run] * 8
    assert transform.float_in_work.shape == (2, 11)
    assert transform.int_in_work.shape == (2, 2)
    assert transform.float_live_out.shape == (2, 6)
    assert transform.int_live_out.shape == (2, 1)
    assert transform.float_retained.shape == (3, 6)
    assert transform.int_retained.shape == (3, 1)
    assert transform.float_retained.dtype == np.dtype(np.float64)
    assert transform.int_retained.dtype == np.dtype(np.int64)
    assert transform.retained_reps.tolist() == [0, 3, 7]
    assert transform.retained_row_by_rep.tolist() == [0, -1, -1, 1, -1, -1, -1, 2]
    assert test.float_in_work.shape == (2, 0)
    assert test.int_in_work.shape == (2, 0)
    assert test.float_retained.shape == (0, 1)
    assert test.int_retained.shape == (0, 1)
