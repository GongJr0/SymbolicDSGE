from enum import IntEnum
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from ...core.shock.plan import ShockEntry, ArrayEntry

_F64 = NDArray[np.float64]

class ShockCode(IntEnum):
    _value_: int

    PATH = ...
    NORMAL = ...
    UNIFORM = ...
    STUDENT_T = ...

    @classmethod
    def for_dist(cls, dist: object) -> "ShockCode | None": ...

class NativeShockPlan:
    @property
    def scratch_size(self) -> int: ...
    @property
    def n_entries(self) -> int: ...
    def fill(self, out: _F64, rep_idx: int) -> None: ...
    def draw(self, rep_idx: int) -> _F64: ...

def native_shock_plan(
    entries: Sequence[ShockEntry | ArrayEntry],
    T: int,
    n_exog: int,
    shock_scale: float,
) -> NativeShockPlan: ...
