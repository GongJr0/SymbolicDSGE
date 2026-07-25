"""Native RNG bridge: draws over a numpy ``Generator``'s borrowed PCG64 state.

Re-exports the compiled ``_rng`` extension. Native code (the MCMC mainloop)
consumes the C kernels directly via ``_EXTRA_DEPS``; the Python surface here
exists for parity testing and any future direct use. If the extension is not
built, importing this module raises ``ImportError``.
"""

from ._rng import standard_normal, standard_uniform

__all__ = [
    "standard_normal",
    "standard_uniform",
]
