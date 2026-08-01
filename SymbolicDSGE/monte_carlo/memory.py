"""Preflight memory profiling for a resolved Monte Carlo buffer plan.

A run commits every retained arena before the first replication executes, and
the default retains all replications of every step. Nothing reports that
commitment, so an oversized plan allocates successfully and then degrades under
paging rather than failing. This module walks a resolved plan, adds the terms
that live outside it, and compares the total against what the machine has
available right now.
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

#: Flat part of the reserve held back for the process around the run: the
#: interpreter growing as results are read back, a notebook kernel holding its
#: own copy, and the allocator's transient peaks. None of that scales with the
#: size of the run, so neither does this term.
RESERVE_FLOOR_BYTES = 1024**3

#: Part of the reserve that does scale, covering allocator overhead proportional
#: to the arenas themselves. Deliberately small: a reserve expressed as a
#: fraction of the machine rather than of the run withholds hundreds of
#: gigabytes on a large host to protect against a fixed-size risk.
RESERVE_FRACTION = 0.025


#: The package directory, against which a frame is judged library or caller.
_PACKAGE_ROOT = os.path.normcase(os.path.dirname(os.path.dirname(__file__)))


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that blames the first frame outside this package.

    Four call chains reach :meth:`MCMemoryProfiler.validate`, at four depths:
    ``validate_memory_requirements`` is one hop from its caller, while a run
    entered through :func:`~.builder.run_pipeline` is four. A constant is wrong
    for three of them, and threading the depth through every signature between
    here and the caller prices a warning header at four public arguments.
    Counting the frames instead gets all four right and adds none.

    This is what ``warnings.warn(skip_file_prefixes=...)`` does, which is 3.12
    and later while this package supports 3.11.
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
    """Write to stdout and flush it.

    Ordering against the traceback is the whole point of printing the report, and
    a buffered stdout is drained after the interpreter has already written the
    traceback to stderr.
    """
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

    The report names no steps as targets for reduction. Retention intent is not
    recoverable from the step graph: a value having downstream consumers does
    not make it disposable, and a step feeding nothing may be the one its author
    wants back. It reports the cost and leaves the choice.
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

        Past this an allocation does not get slower, it fails. On Windows the
        commit limit is physical plus pagefile and a commit beyond it is refused
        outright. Elsewhere the arenas allocate lazily and the process is killed
        partway through the loop instead, which is the same loss arriving later.
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
        # The totals are whole-run figures, not a continuation of the per-step
        # rows they align under. A rule keeps the two from reading as one table.
        rule = "-" * max(len(line) for line in (*lines, *totals))
        return "\n".join((*lines, rule, *totals))


class MCMemoryProfiler:
    """Sizes one native Monte Carlo run before its arenas are allocated.

    Constructed from a resolved :data:`~.allocation.BufferPlan` and the run
    arguments that plan was resolved under. It is not part of the public API:
    a plan cannot be built without the run arguments in the first place, so the
    profile is reached through :meth:`~.core.MCPipeline.validate_memory_requirements`
    or raised automatically by a run.
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

        Both paths print the breakdown rather than carrying it in the message
        they raise or warn with, for the same reason on each. An exception
        message long enough to hold the table is rendered after the traceback,
        which puts the numbers below several screens of frames. A warning
        message ending in the table is followed by the source line
        :mod:`warnings` echoes for the frame it blames, which reads like a
        stray frame under the totals. Printing leaves both with one line to
        read.
        """
        report = self.report()
        if report.exceeds_limit:
            # Titled here rather than in the report itself, which the warning
            # path renders too and which is not an error there.
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

        The native draw runs in C off a per-replication counter and allocates
        nothing. One entry the kernel cannot reproduce sends the whole spec back
        to the Python route, which materializes an ``(n_rep, T, n_exog)`` slab
        while the step is lowered. Eligibility is read off the raw spec, the same
        way the arena planner sizes its draw scratch and lowering picks its
        branch, so the three cannot disagree.
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
