import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]
_U64 = NDArray[np.uint64]

# Native draws over a numpy Generator's borrowed PCG64 state; bit-identical to
# the generator's own standard_normal / random on the same state.

def standard_normal(rng: np.random.Generator, n: int) -> _F64: ...
def standard_uniform(rng: np.random.Generator, n: int) -> _F64: ...

# Draws from the counter-based Philox engine, selected by (key, stream) rather
# than by a seeding pass. Not numpy-parity; the engine is ours.

def philox_standard_normal(
    key0: int, key1: int, stream0: int, stream1: int, n: int
) -> _F64: ...
def philox_standard_uniform(
    key0: int, key1: int, stream0: int, stream1: int, n: int
) -> _F64: ...
def philox_raw(key0: int, key1: int, stream0: int, stream1: int, n: int) -> _U64: ...
