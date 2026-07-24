"""Build the native Cython/C extensions for SymbolicDSGE.

Project metadata lives in ``pyproject.toml`` (PEP 621); this file only declares
the compiled extensions. One extension per ``_ckernels`` subsystem that ships a
``_<name>.pyx`` shim; each links its sibling ``*.c`` plus the shared
``_common/*.c`` sources.

If Cython is unavailable, the extension list is empty and the library falls back
to its numba kernels at runtime (so a metadata-only operation still works).
"""

from __future__ import annotations

import glob
import os
from typing import cast

from setuptools import Extension, setup

_CKERNELS = os.path.join("SymbolicDSGE", "_ckernels")
_COMMON = os.path.join(_CKERNELS, "_common")

# Extra intra-_ckernels subsystem deps: an extension that calls another
# subsystem's leaf C must link its hand-written sources (Windows .pyds cannot
# share symbols). Keyed by extension subdir name. `_common` is linked into every
# extension already; this is for the higher-level subsystems (core, kalman, ...).
_EXTRA_DEPS = {
    "estimation": ["core", "kalman", "optim", "rng"],
}

# Subsystems whose hand-written C draws randoms through numpy's low-level RNG
# C-API (numpy/random headers + the `npyrandom` static lib). An extension needs
# the numpy include path and the `npyrandom` link iff it compiles `rng` sources:
# either it IS `rng`, or it lists `rng` in _EXTRA_DEPS. Keeping this scoped means
# the six RNG-free subsystems never pull the numpy build dependency.


def _hand_c(subdir: str) -> list[str]:
    """Hand-written C in ``subdir``. A leading underscore marks the
    cythonize-generated ``_<name>.c`` (added by cythonize itself; globbing it
    duplicates the object -> LNK4042), so those are excluded."""
    return [
        c
        for c in sorted(glob.glob(os.path.join(subdir, "*.c")))
        if not os.path.basename(c).startswith("_")
    ]


def _compile_args() -> list[str]:
    # -O3 everywhere we can; never -ffast-math (it breaks IEEE parity with the
    # numba/numpy reference, which the parity tests rely on). MSVC (CI Windows)
    # is IEEE-safe at /O2 by default; /fp:fast is never added.
    if os.name == "nt":
        return ["/O2"]
    return ["-O3", "-fno-fast-math"]


def _extensions() -> list[Extension]:
    try:
        from Cython.Build import cythonize
    except ImportError:
        return []

    common_sources = sorted(glob.glob(os.path.join(_COMMON, "*.c")))
    extra_args = _compile_args()

    extensions: list[Extension] = []
    for pyx in sorted(glob.glob(os.path.join(_CKERNELS, "*", "_*.pyx"))):
        subdir = os.path.dirname(pyx)
        module = os.path.relpath(pyx, ".").replace(os.sep, ".")[: -len(".pyx")]
        hand_c = _hand_c(subdir)

        dep_dirs = [
            os.path.join(_CKERNELS, dep)
            for dep in _EXTRA_DEPS.get(os.path.basename(subdir), [])
        ]
        dep_c = [c for d in dep_dirs for c in _hand_c(d)]

        sources = [pyx] + hand_c + dep_c + common_sources

        subname = os.path.basename(subdir)
        include_dirs = [subdir, *dep_dirs, _COMMON]
        ext_kwargs: dict[str, object] = {}
        if subname == "rng" or "rng" in _EXTRA_DEPS.get(subname, []):
            import numpy as np

            include_dirs.append(np.get_include())
            ext_kwargs["library_dirs"] = [
                os.path.join(os.path.dirname(np.__file__), "random", "lib")
            ]
            ext_kwargs["libraries"] = ["npyrandom"]

        extensions.append(
            Extension(
                module,
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_args,
                **ext_kwargs,
            )
        )

    # cythonize is untyped (Cython ships no stubs); cast to satisfy
    # warn_return_any without loosening the public return type.
    cythonized = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
    return cast("list[Extension]", cythonized)


setup(ext_modules=_extensions())
