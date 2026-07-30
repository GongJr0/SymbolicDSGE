"""Build the native Cython/C extensions for SymbolicDSGE.

Project metadata lives in ``pyproject.toml`` (PEP 621); this file only declares
the compiled extensions. One extension per ``_ckernels`` subsystem that ships a
``_<name>.pyx`` shim; each links its sibling ``*.c`` plus the shared
``_common/*.c`` sources.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import cast

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_CKERNELS = os.path.join("SymbolicDSGE", "_ckernels")
_COMMON = os.path.join(_CKERNELS, "_common")

# Extra intra-_ckernels subsystem deps: an extension that calls another
# subsystem's leaf C must link its hand-written sources (Windows .pyds cannot
# share symbols). Keyed by extension subdir name. `_common` is linked into every
# extension already; this is for the higher-level subsystems (core, kalman, ...).
_EXTRA_DEPS = {
    "estimation": ["core", "kalman", "optim", "rng"],
    "monte_carlo": ["core", "kalman", "rng", "regression", "diag"],
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
    # These flags preserve the expected floating-point behavior without using
    # the unsafe whole-program relaxation implied by ``-ffast-math``.
    flags = [
        "-O3",
        "-fno-math-errno",
        "-ffp-contract=fast",
        "-fassociative-math",
        "-fopenmp",
        "-Wno-visibility",
        "-Wno-unused-function",
        "-Wno-unused-but-set-variable",
    ]

    if os.name == "nt":
        if os.environ.get("SDSGE_TOOLCHAIN", "").lower() == "clang-cl":
            # clang-cl emits LLVM bitcode; lld-link consumes it and runs the
            # ThinLTO phase directly, without MSVC's /GL or /LTCG flags.
            flags.append("-flto=thin")
        return ["/clang:" + flag for flag in flags]
    return flags


def _link_args() -> list[str]:
    """Link the OpenMP runtime on compilers that do not do so implicitly."""
    return [] if os.name == "nt" else ["-fopenmp"]


class ClangCLBuildExt(build_ext):
    """Opt into clang-cl while retaining setuptools' MSVC ABI backend."""

    def build_extensions(self) -> None:
        toolchain = os.environ.get("SDSGE_TOOLCHAIN", "").lower()

        if not toolchain:
            super().build_extensions()
            return

        if toolchain != "clang-cl":
            raise RuntimeError(
                "SDSGE_TOOLCHAIN must be 'clang-cl' when set, " f"got {toolchain!r}."
            )

        if self.compiler.compiler_type != "msvc":
            raise RuntimeError(
                "clang-cl build expected the MSVC backend, "
                f"got {self.compiler.compiler_type!r}."
            )

        clang_cl = shutil.which("clang-cl.exe")
        lld_link = shutil.which("lld-link.exe")
        if clang_cl is None or lld_link is None:
            missing = [
                name
                for name, path in (
                    ("clang-cl.exe", clang_cl),
                    ("lld-link.exe", lld_link),
                )
                if path is None
            ]
            raise RuntimeError(
                "SDSGE_TOOLCHAIN=clang-cl requires " + ", ".join(missing) + " on PATH."
            )

        # Force the MSVC backend to discover its include/library environment
        # before replacing cl.exe and link.exe.
        if not getattr(self.compiler, "initialized", True):
            self.compiler.initialize()

        self.compiler.cc = clang_cl
        self.compiler.linker = lld_link

        compile_options = getattr(self.compiler, "compile_options", None)
        if compile_options is not None:
            self.compiler.compile_options = [
                flag for flag in compile_options if flag != "/GL"
            ]

        ldflags = getattr(self.compiler, "_ldflags", None)
        if ldflags is not None:
            for flags in ldflags.values():
                flags[:] = [flag for flag in flags if flag != "/LTCG"]

        super().build_extensions()


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
                extra_link_args=_link_args(),
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


setup(ext_modules=_extensions(), cmdclass={"build_ext": ClangCLBuildExt})
