from typing import NamedTuple, Sequence

from ._arenas import ArenaAllocation

class NativeRunResult(NamedTuple):
    status: int
    halt_rep_idx: int
    halt_step_idx: int
    halt_status: int
    wall_elapsed_s: float
    step_elapsed_s_by_worker: object
    step_counts_by_worker: object
    step_failures_by_worker: object

class NativeStep:
    @property
    def name(self) -> str: ...

def payload_step(name: str, value: object) -> NativeStep: ...
def transform_step(
    name: str,
    kind: str,
    n: int,
    p: int,
    ddof: int = 0,
    offset: float = 0.0,
    order: int = 1,
    window: int = 1,
) -> NativeStep: ...
def run(
    allocation: ArenaAllocation,
    steps: Sequence[NativeStep],
    fail_fast: bool = False,
    profile_steps: bool = False,
) -> NativeRunResult: ...
