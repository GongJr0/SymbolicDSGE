from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.allocation import ArenaSize, StepBufferPlan
from SymbolicDSGE.monte_carlo.builder import run_pipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCStep, OpType
from SymbolicDSGE.monte_carlo.memory import (
    RESERVE_FLOOR_BYTES,
    RESERVE_FRACTION,
    MCMemoryProfiler,
    _format_bytes,
)
from SymbolicDSGE.monte_carlo.step_factories import simulation_step

ARENA_NAMES = (
    "float_in_work",
    "int_in_work",
    "float_live_out",
    "int_live_out",
    "float_retained",
    "int_retained",
    "retained_reps",
    "retained_row_by_rep",
)


#: More memory than any plan here sizes to.
ABUNDANT_BYTES = 1 << 50


@pytest.fixture(scope="module")
def solved() -> SolvedModel:
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _pin_memory(
    monkeypatch: pytest.MonkeyPatch,
    available: int,
    swap_free: int = 0,
) -> None:
    """Fix what the profiler reads for physical memory and for swap."""
    import SymbolicDSGE.monte_carlo.memory as memory

    class _Reading:
        def __init__(self, **fields: int) -> None:
            self.__dict__.update(fields)

    monkeypatch.setattr(
        memory.psutil, "virtual_memory", lambda: _Reading(available=available)
    )
    monkeypatch.setattr(memory.psutil, "swap_memory", lambda: _Reading(free=swap_free))


@pytest.fixture(autouse=True)
def abundant_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the machine's own free memory out of every test in this file.

    Tests that care about scarcity call :func:`_pin_memory` again.
    """
    _pin_memory(monkeypatch, available=ABUNDANT_BYTES, swap_free=ABUNDANT_BYTES)


def _allocated_bytes(allocation: Any) -> int:
    """Every dynamic array one allocation owns, measured after the fact."""
    total = int(allocation.failure_step_by_rep.nbytes)
    total += int(allocation.failure_status_by_rep.nbytes)
    for arenas in allocation.steps.values():
        total += sum(int(getattr(arenas, name).nbytes) for name in ARENA_NAMES)
    return total


def _datagen_stub() -> MCStep:
    """A datagen step that binds no shock slab, for plan-arithmetic tests."""
    return MCStep(
        name="datagen",
        op_type=OpType.DATAGEN,
        step_type="raw_model_data",
        kwargs={"states": (), "observables": ()},
    )


def test_planned_bytes_match_the_allocation_exactly(solved: SolvedModel) -> None:
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim",
                target="reference",
                T=6,
                shocks={
                    name: Shock(dist="norm", seed=index)
                    for index, name in enumerate(solved.compiled.layout.exo_state_names)
                },
                observables=True,
            )
        ]
    )
    report = pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=3)
    lowered = pipeline.lower_native(reference=solved, n_rep=8, n_jobs=3)

    assert report.shock_bytes == 0
    assert report.planned_bytes == _allocated_bytes(lowered.allocation)
    assert report.total_bytes_w_margin == report.planned_bytes + report.reserve_bytes


@pytest.mark.parametrize(
    ("n_retain", "expected_reps"),
    [(-1, 10), (0, 0), (4, 4)],
)
def test_retention_sentinel_resolves_against_n_rep(
    solved: SolvedModel,
    n_retain: int,
    expected_reps: int,
) -> None:
    plan = {
        "datagen": StepBufferPlan(
            name="datagen",
            input_size=ArenaSize(5, 1),
            output_size=ArenaSize(3, 2),
            out_fields={},
            n_retain=n_retain,
        )
    }
    report = MCMemoryProfiler(
        plan, [_datagen_stub()], reference=solved, n_rep=10, n_jobs=2
    ).report()

    (step,) = report.steps
    assert step.per_rep_bytes == (3 + 2) * 8
    assert step.retained_bytes == expected_reps * (3 + 2) * 8
    assert step.worker_bytes == 2 * (5 + 1 + 3 + 2) * 8
    # Two run-level failure lanes, plus this step's retention indices.
    assert report.bookkeeping_bytes == (2 * 10 + expected_reps + 10) * 8


def test_native_eligible_shocks_prematerialize_nothing(solved: SolvedModel) -> None:
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim",
                target="reference",
                T=6,
                shocks={
                    name: Shock(dist="norm", seed=index)
                    for index, name in enumerate(solved.compiled.layout.exo_state_names)
                },
            )
        ]
    )

    report = pipeline.validate_memory_requirements(reference=solved, n_rep=32, n_jobs=1)

    assert report.shock_bytes == 0


def test_fallback_shocks_are_counted_outside_the_arenas(solved: SolvedModel) -> None:
    T = 6
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim",
                target="reference",
                T=T,
                shocks={
                    name: Shock(dist="t", seed=index, dist_kwargs={"df": 8})
                    for index, name in enumerate(solved.compiled.layout.exo_state_names)
                },
            )
        ]
    )

    report = pipeline.validate_memory_requirements(reference=solved, n_rep=32, n_jobs=1)

    assert report.shock_bytes == 32 * T * solved.compiled.n_exog * 8
    assert report.planned_bytes > _allocated_bytes(
        pipeline.lower_native(reference=solved, n_rep=32, n_jobs=1).allocation
    )


def test_the_reserve_is_a_floor_plus_a_fraction_not_a_multiple(
    solved: SolvedModel,
) -> None:
    """A reserve proportional to the machine withholds far too much on a large host."""
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    report = pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)

    assert report.reserve_bytes == int(
        RESERVE_FLOOR_BYTES + RESERVE_FRACTION * report.planned_bytes
    )
    assert report.total_bytes_w_margin == report.planned_bytes + report.reserve_bytes
    # The floor dominates a small run, so the reserve is nearly flat there.
    assert report.reserve_bytes >= RESERVE_FLOOR_BYTES


def test_paging_warns_but_is_allowed_because_it_only_costs_throughput(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run past physical memory completes, slowly, so it must not be refused."""
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    total = sized.total_bytes_w_margin
    _pin_memory(monkeypatch, available=total // 2, swap_free=total)

    with pytest.warns(UserWarning, match="slowdown from paging"):
        report = pipeline.validate_memory_requirements(
            reference=solved, n_rep=64, n_jobs=1
        )

    assert report.degrades
    assert not report.exceeds_limit


def test_validate_raises_only_once_swap_cannot_absorb_it_either(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    total = sized.total_bytes_w_margin
    _pin_memory(monkeypatch, available=total // 4, swap_free=total // 4)

    with pytest.raises(MemoryError, match="free swap"):
        pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)


def test_the_ceiling_counts_swap_on_top_of_physical(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    # Distinct readings, so summing them cannot be confused with doubling one.
    _pin_memory(monkeypatch, available=ABUNDANT_BYTES, swap_free=ABUNDANT_BYTES // 4)

    report = pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)

    assert report.available_bytes == ABUNDANT_BYTES
    assert report.swap_free_bytes == ABUNDANT_BYTES // 4
    assert report.ceiling_bytes == ABUNDANT_BYTES + ABUNDANT_BYTES // 4


def test_the_raised_message_is_one_line_and_the_table_is_printed(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The table must not ride along in the exception, or it renders after the traceback."""
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    _pin_memory(
        monkeypatch,
        available=sized.total_bytes_w_margin // 4,
        swap_free=sized.total_bytes_w_margin // 4,
    )

    with pytest.raises(MemoryError) as raised:
        pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)

    message = str(raised.value)
    assert "\n" not in message
    assert "check_memory_availability=False" in message
    printed = capsys.readouterr().out
    assert printed.startswith("Memory Availability Error:\n")
    assert "sim" in printed
    assert "available" in printed


def test_a_rule_separates_the_step_rows_from_the_whole_run_totals(
    solved: SolvedModel,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )

    lines = str(
        pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)
    ).splitlines()
    rule_index = next(index for index, line in enumerate(lines) if set(line) == {"-"})

    assert any("sim" in line for line in lines[:rule_index])
    assert any("available" in line for line in lines[rule_index + 1 :])
    assert len(lines[rule_index]) == max(len(line) for line in lines)


def test_the_warned_message_is_one_line_and_the_table_is_printed(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A table at the tail of a warning is trailed by the source line warnings echoes."""
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    _pin_memory(
        monkeypatch,
        available=sized.total_bytes_w_margin // 2,
        swap_free=sized.total_bytes_w_margin,
    )

    with pytest.warns(UserWarning) as caught:
        pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)

    message = str(caught[0].message)
    assert "\n" not in message
    printed = capsys.readouterr().out
    # Not an error on this path, so it must not borrow the error path's title.
    assert "Memory Availability Error" not in message
    assert "Memory Availability Error" not in printed
    assert printed.startswith("Memory Profile:\n")
    assert "sim" in printed


@pytest.mark.parametrize(
    "entry_point", ["validate_memory_requirements", "run_pipeline"]
)
def test_the_warning_blames_the_caller_not_the_library(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    """The shallowest and deepest routes to validate sit four frames apart."""
    steps = [simulation_step("sim", target="reference", T=6, observables=True)]
    pipeline = MCPipeline(steps)
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    _pin_memory(
        monkeypatch,
        available=sized.total_bytes_w_margin // 2,
        swap_free=sized.total_bytes_w_margin,
    )

    with pytest.warns(UserWarning) as caught:
        if entry_point == "validate_memory_requirements":
            pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
        else:
            run_pipeline(
                pipeline.to_spec(),
                reference=solved,
                dgp=None,
                n_rep=64,
                fail_fast=True,
                n_jobs=1,
                verbosity=0,
            )

    assert caught[0].filename == __file__


def test_nothing_is_printed_when_the_plan_fits(
    solved: SolvedModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )

    pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)

    assert capsys.readouterr().out == ""


def test_run_refuses_an_oversized_plan_unless_the_check_is_disabled(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    sized = pipeline.validate_memory_requirements(reference=solved, n_rep=64, n_jobs=1)
    _pin_memory(
        monkeypatch,
        available=sized.total_bytes_w_margin // 4,
        swap_free=sized.total_bytes_w_margin // 4,
    )

    with pytest.raises(MemoryError):
        pipeline.run(reference=solved, n_rep=64, n_jobs=1, verbosity=0)

    result = pipeline.run(
        reference=solved,
        n_rep=64,
        n_jobs=1,
        verbosity=0,
        check_memory_availability=False,
    )

    assert result.n_successful == 64


def test_report_renders_a_table_naming_no_reduction_target(
    solved: SolvedModel,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )

    rendered = str(
        pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)
    )

    assert "sim" in rendered
    assert "available" in rendered
    assert "worker lanes (x1)" in rendered
    assert "prematerialized shocks" not in rendered


def test_report_renders_the_total_both_before_and_after_the_margin(
    solved: SolvedModel,
) -> None:
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=True)]
    )
    report = pipeline.validate_memory_requirements(reference=solved, n_rep=8, n_jobs=1)

    lines = {
        line.split("  ")[0].strip(): line.rsplit("  ", 1)[-1].strip()
        for line in str(report).splitlines()
        if line.strip()
    }

    assert lines["allocated"] == _format_bytes(report.planned_bytes)
    assert lines["total"] == _format_bytes(report.total_bytes_w_margin)
    assert lines["ceiling (+ swap free)"] == _format_bytes(report.ceiling_bytes)
    assert report.planned_bytes < report.total_bytes_w_margin


def test_profiler_rejects_a_non_positive_n_rep(solved: SolvedModel) -> None:
    with pytest.raises(ValueError, match="n_rep must be positive"):
        MCMemoryProfiler({}, [_datagen_stub()], reference=solved, n_rep=0)


def test_profiler_is_not_part_of_the_public_namespace() -> None:
    import SymbolicDSGE.monte_carlo as monte_carlo

    assert "MCMemoryProfiler" not in monte_carlo.__all__
    assert not hasattr(monte_carlo, "MCMemoryProfiler")


def test_unavailable_memory_reads_as_an_infinite_load(
    solved: SolvedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_memory(monkeypatch, available=0)
    profiler = MCMemoryProfiler(
        {
            "datagen": StepBufferPlan(
                name="datagen",
                input_size=ArenaSize(),
                output_size=ArenaSize(1, 0),
                out_fields={},
                n_retain=-1,
            )
        },
        [_datagen_stub()],
        reference=solved,
        n_rep=4,
    )

    report = profiler.report()

    assert report.load == np.inf
    assert report.exceeds_limit
