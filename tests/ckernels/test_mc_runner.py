from __future__ import annotations

import numpy as np

from SymbolicDSGE._ckernels.monte_carlo._arenas import allocate_arenas
from SymbolicDSGE._ckernels.monte_carlo._runner import (
    payload_step,
    run,
    transform_step,
)
from SymbolicDSGE.monte_carlo.allocation import ArenaSize, StepBufferPlan
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc


def _plan(name: str, n_float_in: int, n_float_out: int) -> dict[str, StepBufferPlan]:
    return {
        name: StepBufferPlan(
            name=name,
            input_size=ArenaSize(n_float_in, 0),
            output_size=ArenaSize(n_float_out, 0),
            out_fields={},
            n_retain=-1,
        )
    }


def _numba_first_difference(sample, output):
    output[:] = sample[1:] - sample[:-1]
    return 0


def test_runner_lowers_arenas_and_retains_batched_payload_rows() -> None:
    payload = np.arange(10.0).reshape(5, 2, 1)
    allocation = allocate_arenas(_plan("payload", 0, 2), 5, n_jobs=2)

    result = run(allocation, [payload_step("payload", payload)])

    assert result.status == 0
    assert result.halt_rep_idx == -1
    np.testing.assert_array_equal(
        allocation.steps["payload"].float_retained,
        payload.reshape(5, 2),
    )
    assert allocation.failure_step_by_rep.tolist() == [-1] * 5
    assert allocation.failure_status_by_rep.tolist() == [0] * 5


def test_runner_profiles_step_work_per_worker_and_wall_time() -> None:
    payload = np.arange(10.0).reshape(5, 2, 1)
    allocation = allocate_arenas(_plan("payload", 0, 2), 5, n_jobs=2)

    result = run(
        allocation,
        [payload_step("payload", payload)],
        profile_steps=True,
    )

    assert result.status == 0
    assert result.wall_elapsed_s > 0.0
    assert result.step_elapsed_s_by_worker.shape == (2, 1)
    assert result.step_counts_by_worker.shape == (2, 1)
    assert result.step_failures_by_worker.shape == (2, 1)
    assert result.step_elapsed_s_by_worker.min() >= 0.0
    assert result.step_counts_by_worker.sum() == 5
    assert result.step_failures_by_worker.sum() == 0


def test_runner_passes_null_for_zero_width_optional_integer_output() -> None:
    allocation = allocate_arenas(_plan("log", 4, 4), 3, n_jobs=2)
    allocation.steps["log"].float_in_work[:] = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    )

    result = run(allocation, [transform_step("log", "log", 2, 2)])

    assert result.status == 0
    np.testing.assert_allclose(
        allocation.steps["log"].float_retained,
        np.log(np.array([[1.0, 2.0, 3.0, 4.0]] * 3)),
    )


def test_runner_calls_numba_user_transform() -> None:
    allocation = allocate_arenas(_plan("custom", 6, 4), 3, n_jobs=2)
    allocation.steps["custom"].float_in_work[:] = np.arange(6.0)
    callback = NumbaCustomFunc(_numba_first_difference)

    result = run(
        allocation,
        [
            transform_step(
                "custom",
                "custom",
                3,
                2,
                function_address=callback.address,
                backing=callback,
                output_n=2,
                output_p=2,
            )
        ],
    )

    assert result.status == 0
    np.testing.assert_allclose(
        allocation.steps["custom"].float_retained,
        [[2.0, 2.0, 2.0, 2.0]] * 3,
    )


def test_runner_fail_fast_sanitizes_failed_and_skipped_retained_rows() -> None:
    allocation = allocate_arenas(_plan("diff", 2, 2), 5, n_jobs=2)

    result = run(
        allocation,
        [transform_step("diff", "diff", 2, 1, order=0)],
        fail_fast=True,
    )

    not_run = np.iinfo(np.int64).min
    assert result.status == 1
    assert 0 <= result.halt_rep_idx < 5
    assert result.halt_step_idx == 0
    # SDSGE_TRANSFORM_BAD_ARG, _ckernels/monte_carlo/transforms.h.
    bad_arg = -1301
    assert result.halt_status == bad_arg
    assert set(allocation.failure_step_by_rep.tolist()) <= {0, not_run}
    assert set(allocation.failure_status_by_rep.tolist()) <= {bad_arg, not_run}
    assert np.isnan(allocation.steps["diff"].float_retained).all()
