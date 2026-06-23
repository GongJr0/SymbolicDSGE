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
        # Hand-written C only; a leading underscore marks the Cython-generated
        # _<name>.c (cythonize adds that itself, so globbing it duplicates the
        # object -> LNK4042).
        hand_c = [
            c
            for c in sorted(glob.glob(os.path.join(subdir, "*.c")))
            if not os.path.basename(c).startswith("_")
        ]
        sources = [pyx] + hand_c + common_sources
        extensions.append(
            Extension(
                module,
                sources=sources,
                include_dirs=[subdir, _COMMON],
                extra_compile_args=extra_args,
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
