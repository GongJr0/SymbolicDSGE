"""Native RNG: two engines, both under numpy's compiled transforms.

``standard_normal`` / ``standard_uniform`` draw over a numpy ``Generator``'s
BORROWED PCG64 state and are bit-identical to that generator's own methods.

``philox_*`` draw from an OWNED counter-based engine, selected by a
``(key, stream)`` pair instead of by a seeding pass. This is what the Monte
Carlo shock draw runs on: the runner parallelizes over replications and gives a
step only its ``rep_idx``, and a borrowed ``bitgen_t`` can be neither reseeded
nor jumped from C. These draws are not numpy-parity.

Re-exports the compiled ``_rng`` extension. Native code (the MCMC mainloop, the
MC simulation step) consumes the C kernels directly via ``_EXTRA_DEPS``; the
Python surface here exists for parity and reproducibility testing. If the
extension is not built, importing this module raises ``ImportError``.
"""

from ._rng import (
    philox_raw,
    philox_standard_normal,
    philox_standard_uniform,
    standard_normal,
    standard_uniform,
)

__all__ = [
    "philox_raw",
    "philox_standard_normal",
    "philox_standard_uniform",
    "standard_normal",
    "standard_uniform",
]
