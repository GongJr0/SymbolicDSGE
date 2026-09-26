import numpy as np
from numpy.typing import NDArray

from ...core.shock.plan import ShockPlan

_F64 = NDArray[np.float64]

SHOCK_NORMAL: int = ...
SHOCK_UNIFORM: int = ...

class NativeShockPlan:
    @property
    def scratch_size(self) -> int: ...
    @property
    def n_entries(self) -> int: ...
    def draw(self, rep_idx: int) -> _F64: ...

def native_shock_plan(
    pyplan: ShockPlan,
    T: int,
    n_exog: int,
    shock_scale: float,
) -> NativeShockPlan: ...
