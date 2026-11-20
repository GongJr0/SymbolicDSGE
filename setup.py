"""Build the native Cython/C extensions for SymbolicDSGE.

Project metadata lives in ``pyproject.toml`` (PEP 621); this file only declares
the compiled extensions. One extension per ``_<name>.pyx`` shim in a
``_ckernels`` subsystem. Each subsystem's hand-written C, ``_common`` included,
compiles once into a static library; an extension links the libraries of its
own subsystem and its dependency closure.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.modified import newer_group

_CKERNELS = os.path.join("SymbolicDSGE", "_ckernels")
_COMMON = os.path.join(_CKERNELS, "_common")
_COMMON_NAME = "_common"
_LIB_DIR = "sdsge_libs"

# Extra intra-_ckernels subsystem deps: an extension that calls another
# subsystem's leaf C must link its static library (Windows .pyds cannot share
# symbols). Keyed by extension subdir name. `_common` is linked into every
# extension already; this is for the higher-level subsystems (core, kalman, ...).
_EXTRA_DEPS = {
    "core": ["rng"],
    "estimation": ["core", "kalman", "optim", "rng"],
    "monte_carlo": ["core", "kalman", "regression", "diag"],
    "occbin": ["core"],
}

# Subsystems whose hand-written C carries `#pragma omp`. An extension links the
# OpenMP runtime iff it compiles one of these: either it IS one, or it depends
# on one directly or transitively.
_OPENMP_OWNERS = {"monte_carlo", "estimation"}

# Subsystems whose hand-written C draws randoms through numpy's low-level RNG
# C-API (numpy/random headers + the `npyrandom` static lib). An extension needs
# the numpy include path and the `npyrandom` link iff it compiles `rng` sources:
# either it IS `rng`, or it depends on `rng` directly or transitively.


def _dependency_closure(subname: str) -> list[str]:
    """Collect dependencies once in discovery order, rejecting cycles."""
    dependencies: list[str] = []
    visited: set[str] = set()
    active: list[str] = []

    def visit(name: str) -> None:
        if name in active:
            cycle = " -> ".join([*active, name])
            raise ValueError(f"Cyclic kernel dependency: {cycle}")
        if name in visited:
            return
        visited.add(name)
        active.append(name)
        if name != subname:
            dependencies.append(name)
        for dependency in _EXTRA_DEPS.get(name, []):
            visit(dependency)
        active.pop()

    visit(subname)
    return dependencies


def _link_order(subname: str) -> list[str]:
    """``subname`` and its closure, each placed before every subsystem it calls.

    Single-pass linkers resolve archives left to right.
    """
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dependency in _EXTRA_DEPS.get(name, []):
            visit(dependency)
        order.append(name)

    visit(subname)
    return order[::-1]


def _needs_openmp(subname: str) -> bool:
    """Whether ``subname`` compiles OpenMP pragmas, its own or a dependency's."""
    return subname in _OPENMP_OWNERS or any(
        dep in _OPENMP_OWNERS for dep in _dependency_closure(subname)
    )


def _needs_numpy_random(subname: str) -> bool:
    return subname == "rng" or "rng" in _dependency_closure(subname)


def _hand_c(subdir: str) -> list[str]:
    """Hand-written C in ``subdir``. A leading underscore marks the
    cythonize-generated ``_<name>.c`` (added by cythonize itself; globbing it
    duplicates the object -> LNK4042), so those are excluded."""
    return [
        c
        for c in sorted(glob.glob(os.path.join(subdir, "*.c")))
        if not os.path.basename(c).startswith("_")
    ]


def _lib_name(subname: str) -> str:
    return "sdsge_" + subname.lstrip("_")


def _include_dirs(subname: str) -> list[str]:
    subsystems = [subname, *_dependency_closure(subname)]
    dirs = [os.path.join(_CKERNELS, s) for s in subsystems]
    if subname != _COMMON_NAME:
        dirs.append(_COMMON)
    if _needs_numpy_random(subname):
        import numpy as np

        dirs.append(np.get_include())
    return dirs


def _static_libraries() -> list[tuple[str, list[str], list[str]]]:
    """``(name, sources, include_dirs)`` for every subsystem carrying hand-written C."""
    libraries: list[tuple[str, list[str], list[str]]] = []
    for subdir in sorted(glob.glob(os.path.join(_CKERNELS, "*"))):
        sources = _hand_c(subdir)
        if sources:
            subname = os.path.basename(subdir)
            libraries.append((_lib_name(subname), sources, _include_dirs(subname)))
    return libraries


def _compile_args() -> list[str]:
    # Reassociation requires non-stop IEEE arithmetic on GCC. Wheel tests
    # validate the resulting numerical behavior for each compiler target.
    flags = [
        "-O3",
        "-fno-math-errno",
        "-ffp-contract=fast",
        "-fassociative-math",
        "-fno-signed-zeros",
        "-fno-trapping-math",
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


def _clang_cl_lib_dir() -> str | None:
    """Return the LLVM library directory used by the clang-cl OpenMP runtime."""
    if os.name != "nt" or os.environ.get("SDSGE_TOOLCHAIN", "").lower() != "clang-cl":
        return None

    clang_cl = shutil.which("clang-cl.exe")
    if clang_cl is None:
        return None
    llvm_lib = os.path.join(os.path.dirname(os.path.dirname(clang_cl)), "lib")
    if not os.path.isfile(os.path.join(llvm_lib, "libomp.lib")):
        raise RuntimeError(f"clang-cl OpenMP runtime not found in {llvm_lib!r}.")
    return llvm_lib


def _clang_cl_runtime_dll() -> str | None:
    """Return the OpenMP DLL used by an in-place clang-cl build."""
    if os.name != "nt" or os.environ.get("SDSGE_TOOLCHAIN", "").lower() != "clang-cl":
        return None

    clang_cl = shutil.which("clang-cl.exe")
    if clang_cl is None:
        return None
    libomp_dll = os.path.join(os.path.dirname(clang_cl), "libomp.dll")
    if not os.path.isfile(libomp_dll):
        raise RuntimeError(f"clang-cl OpenMP runtime not found at {libomp_dll!r}.")
    return libomp_dll


def _unix_lto_toolchain(cc: str) -> tuple[list[str], str]:
    """LTO flags and the LTO-aware archiver matching the unix compiler ``cc``.

    The archiver and linker must read the compiler's own LTO object format, so
    each is taken from the compiler's installation rather than from ``PATH``.
    """
    path = shutil.which(cc)
    if path is None:
        raise RuntimeError(f"C compiler {cc!r} not found on PATH.")
    version = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=True
    ).stdout

    if "clang" in version:
        bindir = os.path.dirname(os.path.realpath(path))
        archiver = os.path.join(bindir, "llvm-ar")
        flags = ["-flto=thin"]
        if sys.platform != "darwin":
            # GNU ld reads bitcode only through the LLVMgold plugin.
            if not os.path.isfile(os.path.join(bindir, "ld.lld")):
                raise RuntimeError(f"ld.lld not found beside {path!r}.")
            flags.append("-fuse-ld=lld")
    elif "Free Software Foundation" in version:
        base = os.path.basename(path)
        candidates = [shutil.which("gcc-ar")]
        if "gcc" in base:
            sibling = base.replace("gcc", "gcc-ar")
            candidates.insert(0, os.path.join(os.path.dirname(path), sibling))
        archiver = next((c for c in candidates if c and os.path.isfile(c)), "")
        flags = ["-flto=auto"]
    else:
        raise RuntimeError(f"Unrecognized C compiler {path!r}; expected GCC or clang.")

    if not os.path.isfile(archiver):
        raise RuntimeError(f"LTO archiver for {path!r} not found.")
    return flags + ["-fvisibility=hidden"], archiver


class SDSGEBuildExt(build_ext):
    """Build the static subsystem libraries, then the extensions linking them.

    The libraries compile with this command's own compiler, so toolchain swaps
    and LTO flags reach them exactly as they reach the extensions.
    """

    # True is "one worker per CPU" to the base class's parallel builder.
    parallel: int | bool | None

    def finalize_options(self) -> None:
        super().finalize_options()
        # None is "not given"; an explicit -j, 0 included, is kept as given.
        if self.parallel is None:
            self.parallel = True

    def _workers(self) -> int:
        if self.parallel is True:
            return os.cpu_count() or 1
        return int(self.parallel or 0)

    def run(self) -> None:
        super().run()

        if not self.inplace:
            return
        libomp_dll = _clang_cl_runtime_dll()
        if libomp_dll is None:
            return

        # cibuildwheel repairs wheels with delvewheel. Local in-place builds
        # instead need the runtime beside each importing extension: a .pyd's own
        # directory is searched for its dependencies, its siblings are not.
        for ext in self.extensions or []:
            parts = ext.name.split(".")
            if _needs_openmp(parts[-2]):
                shutil.copy2(libomp_dll, os.path.join(*parts[:-1]))

    def build_extensions(self) -> None:
        toolchain = os.environ.get("SDSGE_TOOLCHAIN", "").lower()
        lib_args = _compile_args()

        if self.compiler.compiler_type == "msvc" and not getattr(
            self.compiler, "initialized", True
        ):
            # Once, up front: initialize() would undo the clang-cl swap, and
            # compile() would otherwise run it lazily from racing threads.
            self.compiler.initialize()

        if toolchain:
            self._use_clang_cl(toolchain)
        elif self.compiler.compiler_type == "unix":
            lto_flags, archiver = _unix_lto_toolchain(self.compiler.compiler_so[0])
            self.compiler.set_executables(archiver=[archiver, "-crs"], ranlib=None)
            lib_args = lib_args + lto_flags
            for ext in self.extensions:
                ext.extra_compile_args = list(ext.extra_compile_args) + lto_flags
                # GCC regenerates code at link time; it must see every codegen
                # flag, except -fopenmp, which at link only pulls in the runtime.
                uses_openmp = _needs_openmp(ext.name.split(".")[-2])
                ext.extra_link_args = [
                    flag for flag in lib_args if uses_openmp or flag != "-fopenmp"
                ]

        lib_dir = os.path.join(self.build_temp, _LIB_DIR)
        self._build_static_libraries(lib_dir, lib_args)
        for ext in self.extensions:
            ext.library_dirs = [lib_dir, *ext.library_dirs]

        # Workers take extensions in list order; the largest links start first
        # so none is left running alone at the end.
        self.extensions.sort(
            key=lambda ext: sum(os.path.getsize(s) for s in ext.sources), reverse=True
        )
        super().build_extensions()

    def _build_static_libraries(self, lib_dir: str, args: list[str]) -> None:
        def build(name: str, sources: list[str], include_dirs: list[str]) -> None:
            target = self.compiler.library_filename(name, output_dir=lib_dir)
            if not (self.force or newer_group(sources, target, "newer")):
                return
            objects = self.compiler.compile(
                sources,
                output_dir=lib_dir,
                include_dirs=include_dirs,
                extra_postargs=args,
                debug=self.debug,
            )
            # `ar -r` keeps members it is not handed, including deleted sources'.
            if os.path.exists(target):
                os.remove(target)
            self.compiler.create_static_lib(
                objects, name, output_dir=lib_dir, debug=self.debug
            )

        libraries = _static_libraries()
        workers = self._workers()
        if workers <= 1:
            for library in libraries:
                build(*library)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for future in [executor.submit(build, *lib) for lib in libraries]:
                future.result()

    def _use_clang_cl(self, toolchain: str) -> None:
        """Opt into clang-cl while retaining setuptools' MSVC ABI backend."""
        if toolchain != "clang-cl":
            raise RuntimeError(
                "SDSGE_TOOLCHAIN must be 'clang-cl' when set, " f"got {toolchain!r}."
            )

        if self.compiler.compiler_type != "msvc":
            raise RuntimeError(
                "clang-cl build expected the MSVC backend, "
                f"got {self.compiler.compiler_type!r}."
            )

        tools = {
            name: shutil.which(name)
            for name in ("clang-cl.exe", "lld-link.exe", "llvm-lib.exe")
        }
        missing = [name for name, path in tools.items() if path is None]
        if missing:
            raise RuntimeError(
                "SDSGE_TOOLCHAIN=clang-cl requires " + ", ".join(missing) + " on PATH."
            )

        self.compiler.cc = tools["clang-cl.exe"]
        self.compiler.linker = tools["lld-link.exe"]
        # lib.exe cannot index LLVM bitcode members.
        self.compiler.lib = tools["llvm-lib.exe"]

        compile_options = getattr(self.compiler, "compile_options", None)
        if compile_options is not None:
            self.compiler.compile_options = [
                flag for flag in compile_options if flag != "/GL"
            ]

        ldflags = getattr(self.compiler, "_ldflags", None)
        if ldflags is not None:
            for flags in ldflags.values():
                flags[:] = [flag for flag in flags if flag != "/LTCG"]


def _extensions() -> list[Extension]:
    try:
        from Cython.Build import cythonize
    except ImportError:
        return []

    common_sources = _hand_c(_COMMON)
    extra_args = _compile_args()
    clang_cl_lib_dir = _clang_cl_lib_dir()

    extensions: list[Extension] = []
    for pyx in sorted(glob.glob(os.path.join(_CKERNELS, "*", "_*.pyx"))):
        subdir = os.path.dirname(pyx)
        module = os.path.relpath(pyx, ".").replace(os.sep, ".")[: -len(".pyx")]
        subname = os.path.basename(subdir)
        linked = [
            s for s in _link_order(subname) if _hand_c(os.path.join(_CKERNELS, s))
        ]

        library_dirs: list[str] = []
        libraries = [_lib_name(s) for s in linked] + [_lib_name(_COMMON_NAME)]
        if _needs_numpy_random(subname):
            import numpy as np

            library_dirs.append(
                os.path.join(os.path.dirname(np.__file__), "random", "lib")
            )
            libraries.append("npyrandom")
        if clang_cl_lib_dir is not None and _needs_openmp(subname):
            library_dirs.append(clang_cl_lib_dir)
            libraries.append("libomp")

        extensions.append(
            Extension(
                module,
                sources=[pyx],
                # The libraries' sources: relinks the extension when they change.
                depends=[c for s in linked for c in _hand_c(os.path.join(_CKERNELS, s))]
                + common_sources,
                include_dirs=_include_dirs(subname),
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=extra_args,
            )
        )

    # cythonize is untyped (Cython ships no stubs); cast to satisfy
    # warn_return_any without loosening the public return type.
    cythonized = cythonize(
        extensions,
        nthreads=os.cpu_count() or 0,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
    return cast("list[Extension]", cythonized)


# cythonize's process workers re-import the main module under spawn
# (Windows, macOS); the guard keeps them from re-entering setup().
if __name__ == "__main__":
    setup(ext_modules=_extensions(), cmdclass={"build_ext": SDSGEBuildExt})
