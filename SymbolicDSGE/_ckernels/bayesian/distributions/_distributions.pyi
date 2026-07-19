"""Type stubs for the compiled ``_distributions`` extension.

A compiled module carries no inspectable types, so these hand-written signatures
keep the package's ``py.typed`` coverage consistent. Each entry mirrors the
numba reference (now living in the test oracles); the parity tests guard the
runtime behavior, not this stub.
"""

import numpy as np
from numpy.typing import NDArray

def ndtri_as241(p: float) -> float: ...
def erfinv_from_as241(y: float) -> float: ...
def ndtri_as241_into(p: NDArray[np.float64]) -> NDArray[np.float64]: ...
