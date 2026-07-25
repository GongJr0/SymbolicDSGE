import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]

# Native draws over a numpy Generator's borrowed PCG64 state; bit-identical to
# the generator's own standard_normal / random on the same state.

def standard_normal(rng: np.random.Generator, n: int) -> _F64: ...
def standard_uniform(rng: np.random.Generator, n: int) -> _F64: ...
