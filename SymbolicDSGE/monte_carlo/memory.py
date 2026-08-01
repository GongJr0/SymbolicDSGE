"""Preflight memory profiling for a resolved Monte Carlo buffer plan.

Sizes every retained arena a run commits before its first replication, adds the
terms outside the plan, and compares the total against the machine.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

import psutil

from .._ckernels.monte_carlo._arenas import resolve_n_workers
from .allocation import BufferPlan
from .mc_constructs import MCStep, OpType
from .shock_native import native_shock_families

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

#: Every native arena lane is float64 or int64, so a count converts directly.
BYTES_PER_ELEMENT = 8

#: Reserve held for the host process, independent of the size of the run.
RESERVE_FLOOR_BYTES = 1024**3

#: Reserve proportional to the arenas, for allocator overhead.
RESERVE_FRACTION = 0.025


_PACKAGE_ROOT = os.path.normcase(os.path.dirname(os.path.dirname(__file__)))


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that blames the first frame outside this package.

    Callers reach :meth:`MCMemoryProfiler.validate` at four different depths.
    """
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        if not os.path.normcase(frame.f_code.co_filename).startswith(_PACKAGE_ROOT):
            break
        frame = frame.f_back
        level += 1
    return level


def _print_flushed(message: str) -> None:
    """Write to stdout and flush, so the report lands before any traceback."""
    print(message, flush=True)


class StepMemory(NamedTuple):
    """One step's share of a run's dynamic allocation, in bytes."""

    name: str

    #: The step's ``n_retain`` as authored, ``-1`` meaning every replication.
    n_retain: int

    #: Retained bytes one more replication would cost, the whole output row.
    per_rep_bytes: int

    #: The step's retained arena: its resolved retention times the output row.
    retained_bytes: int

    #: Per-worker input, workspace, and live output lanes.
    worker_bytes: int


@dataclass(frozen=True)
class MCMemoryReport:
    """What a run will allocate, against what the machine has available.

    Reports the cost per step and names none of them as the one to shrink, which
    the step graph cannot tell.
    """

    steps: tuple[StepMemory, ...]

    #: The shock slab the Python fallback route materializes at lowering time.
    shock_bytes: int

    #: Retention indices and per-replication failure records.
    bookkeeping_bytes: int

    #: Physical memory available at the moment the profile ran.
    available_bytes: int

    #: Unused swap or pagefile at the moment the profile ran.
    swap_free_bytes: int

    n_rep: int
    n_workers: int

    @property
    def retained_bytes(self) -> int:
        """Every step's retained arena."""
        return sum(step.retained_bytes for step in self.steps)

    @property
    def worker_bytes(self) -> int:
        """Every step's per-worker lanes."""
        return sum(step.worker_bytes for step in self.steps)

    @property
    def planned_bytes(self) -> int:
        """Everything the run allocates, exactly, before the reserve."""
        return (
            self.retained_bytes
            + self.worker_bytes
            + self.bookkeeping_bytes
            + self.shock_bytes
        )

    @property
    def reserve_bytes(self) -> int:
        """Headroom held back for the process that hosts the run."""
        return int(RESERVE_FLOOR_BYTES + RESERVE_FRACTION * self.planned_bytes)

    @property
    def total_bytes_w_margin(self) -> int:
        """The planned total plus the reserve, which is what is checked."""
        return self.planned_bytes + self.reserve_bytes

    @property
    def ceiling_bytes(self) -> int:
        """The most the machine could hold, physical plus unused swap.

        Past this a run fails rather than slows, at allocation or partway through.
        """
        return self.available_bytes + self.swap_free_bytes

    @property
    def load(self) -> float:
        """The checked total as a fraction of physical memory available."""
        if self.available_bytes <= 0:
            return float("inf")
        return self.total_bytes_w_margin / self.available_bytes

    @property
    def degrades(self) -> bool:
        """Whether the run spills past physical memory and starts paging."""
        return self.total_bytes_w_margin > self.available_bytes

    @property
    def exceeds_limit(self) -> bool:
        """Whether the run does not fit even once swap is counted."""
        return self.total_bytes_w_margin > self.ceiling_bytes

    def __str__(self) -> str:
        rows = [
            (
                step.name,
                _format_bytes(step.per_rep_bytes),
                _format_bytes(step.retained_bytes),
                str(step.n_retain),
            )
            for step in self.steps
        ]
        header = ("step", "per rep", "retained", "n_retain")
        widths = [
            (
                max(len(header[column]), *(len(row[column]) for row in rows))
                if rows
                else len(header[column])
            )
            for column in range(4)
        ]
        lines = [
            f"{header[0]:<{widths[0]}}  {header[1]:>{widths[1]}}  "
            f"{header[2]:>{widths[2]}}  {header[3]:>{widths[3]}}"
        ]
        for row in rows:
            lines.append(
                f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  "
                f"{row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}"
            )

        trailer = [
            (f"worker lanes (x{self.n_workers})", self.worker_bytes),
            ("run metadata", self.bookkeeping_bytes),
        ]
        if self.shock_bytes:
            trailer.append(("prematerialized shocks", self.shock_bytes))
        trailer.append(("allocated", self.planned_bytes))
        trailer.append(
            (
                f"reserve ({_format_bytes(RESERVE_FLOOR_BYTES)} + "
                f"{RESERVE_FRACTION:.1%})",
                self.reserve_bytes,
            )
        )
        trailer.append(("total", self.total_bytes_w_margin))
        trailer.append(("available", self.available_bytes))
        trailer.append(("ceiling (+ swap free)", self.ceiling_bytes))

        label_width = max(
            widths[0] + widths[1] + 2, *(len(label) for label, _ in trailer)
        )
        totals = [
            f"{label:<{label_width}}  {_format_bytes(value):>{widths[2]}}"
            for label, value in trailer
        ]
        # Whole-run figures, not more step rows: keep them visually separate.
        rule = "-" * max(len(line) for line in (*lines, *totals))
        return "\n".join((*lines, rule, *totals))


class MCMemoryProfiler:
    """Sizes one native Monte Carlo run before its arenas are allocated.

    Internal: reach a profile through
    :meth:`~.core.MCPipeline.validate_memory_requirements`.
    """

    def __init__(
        self,
        plan: BufferPlan,
        steps: Sequence[MCStep],
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        n_rep: int,
        n_jobs: int | None = None,
    ) -> None:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")
        self._plan = plan
        self._steps = tuple(steps)
        self._reference = reference
        self._dgp = dgp
        self._n_rep = int(n_rep)
        self._n_workers = int(resolve_n_workers(n_jobs))

    def report(self) -> MCMemoryReport:
        """Profile the plan without acting on the result."""
        steps: list[StepMemory] = []
        bookkeeping = 2 * self._n_rep * BYTES_PER_ELEMENT
        for name, step_plan in self._plan.items():
            output_elements = (
                step_plan.output_size.n_float + step_plan.output_size.n_int
            )
            input_elements = step_plan.input_size.n_float + step_plan.input_size.n_int
            retained_reps = (
                self._n_rep if step_plan.n_retain == -1 else step_plan.n_retain
            )
            per_rep_bytes = output_elements * BYTES_PER_ELEMENT
            steps.append(
                StepMemory(
                    name=name,
                    n_retain=step_plan.n_retain,
                    per_rep_bytes=per_rep_bytes,
                    retained_bytes=retained_reps * per_rep_bytes,
                    worker_bytes=(
                        self._n_workers
                        * (input_elements + output_elements)
                        * BYTES_PER_ELEMENT
                    ),
                )
            )
            # One index per retained replication, one reverse lookup per rep.
            bookkeeping += (retained_reps + self._n_rep) * BYTES_PER_ELEMENT

        return MCMemoryReport(
            steps=tuple(steps),
            shock_bytes=self._shock_bytes(),
            bookkeeping_bytes=bookkeeping,
            available_bytes=int(psutil.virtual_memory().available),
            swap_free_bytes=int(psutil.swap_memory().free),
            n_rep=self._n_rep,
            n_workers=self._n_workers,
        )

    def validate(
        self,
        *,
        print_func: Callable[[str], None] = _print_flushed,
    ) -> MCMemoryReport:
        """Profile the plan, warning when it is large and raising when it is too large.

        Both paths print the breakdown and keep their message to one line, so the
        table is not buried under a traceback or a warning's source-line echo.
        """
        report = self.report()
        if report.exceeds_limit:
            print_func(f"Memory Availability Error:\n{report}")
            raise MemoryError(
                f"This run requires {_format_bytes(report.total_bytes_w_margin)}, "
                f"which does not fit in {_format_bytes(report.available_bytes)} free RAM "
                f"+ {_format_bytes(report.swap_free_bytes)} free swap. "
                "Lower n_retain or n_rep, or pass "
                "check_memory_availability=False to run anyway."
            )
        if report.degrades:
            print_func(f"Memory Profile:\n{report}")
            warnings.warn(
                f"Run requires {_format_bytes(report.total_bytes_w_margin)} of RAM "
                f"with {_format_bytes(report.available_bytes)} available. "
                "Expect a noticeable slowdown from paging.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        return report

    def _shock_bytes(self) -> int:
        """Bytes the fallback shock route materializes outside the arenas.

        A spec the native draw cannot reproduce is built in Python instead, as one
        ``(n_rep, T, n_exog)`` slab. Eligibility is read off the raw spec, the same
        way the arena planner and lowering read it.
        """
        step = self._steps[0]
        if step.op_type is not OpType.DATAGEN or step.step_type != "simulation":
            return 0
        shocks = step.kwargs["shocks"]
        if shocks is None:
            return 0  # A single (T, n_exog) matrix, shared by every replication.
        if native_shock_families(shocks) is not None:
            return 0
        model = self._reference if step.kwargs["target"] == "reference" else self._dgp
        if model is None:
            return 0
        T = int(step.kwargs["T"])
        return self._n_rep * T * model.compiled.n_exog * BYTES_PER_ELEMENT


def _format_bytes(n_bytes: int) -> str:
    """Render a byte count in the largest binary unit that keeps it above one."""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")
