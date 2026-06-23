# `_ckernels` — native numeric kernels

Compiled replacements for the hot numba `@njit` kernels. The goal is to remove
JIT warm-up (and the on-disk `.nbc` cache fragility) for the *static*,
monomorphic `float64` kernels by shipping precompiled C.

## Layout

```
_ckernels/
  __init__.py            # leaf package; no SymbolicDSGE imports
  _common/               # shared pure-C primitives (no Python, no NumPy C-API)
    sdsge_common.h       # portable macros (restrict, ...)
    sdsge_linalg.{c,h}   # (future) matmul/cholesky/triangular solves, etc.
  core/                  # one subsystem (also: kalman regression distributions diag)
    __init__.py          # re-exports the compiled _<name>
    _core.pyx            # thin Cython shim: NumPy buffer -> double*
    core.{c,h}           # pure-C numeric kernels
```

Every subsystem currently holds empty placeholders (`<name>.{c,h}`,
`_<name>.pyx`, `__init__.py`) — the scaffold, no kernels yet.

One compiled extension **per subsystem** (`core/_core.pyx` → `_core`). Small
incremental rebuilds, parallel compilation, import only what you need.

## Division of labor

- **Pure C (`*.c` / `*.h`)** — all numeric work. Takes `double*` + lengths,
  knows nothing about Python. Hand-written; this is where the algorithms live.
- **Cython shim (`_<name>.pyx`)** — *only* maps NumPy memoryviews to `double*`
  and calls the C. No logic. `double[:, ::1]` validates dtype + C-contiguity and
  hands `&A[0, 0]` to C; `with nogil:` drops the GIL around the kernel.

No `cimport numpy` / `import_array()` anywhere: memoryviews ride the buffer
protocol, so NumPy's C-API never enters the build.

## Conventions

- The Cython shim is named `_<subsystem>.pyx`; the compiled module is private
  (`_core`) and re-exported from the subsystem `__init__.py`.
- **Hand-written C must not be named `_*.c`** — that pattern is reserved for the
  Cython-generated `_<name>.c` (git-ignored).
- Shared primitives go in `_common/` and are listed in each `Extension.sources`
  that needs them (the same `.c` compiled into multiple extensions is fine).
- **No `-ffast-math`.** Keep IEEE semantics so the C kernels stay bit-comparable
  to the numba/numpy reference; every subsystem ships a parity test against that
  reference.

## Adding a subsystem (recipe)

1. Write `foo.{c,h}` (pure C) under `_ckernels/foo/`.
2. Write `_ckernels/foo/_foo.pyx`: `cdef extern from "foo.h"` the prototypes, then
   a `def` wrapper per entrypoint with memoryview args + `with nogil:`.
3. Re-export in `_ckernels/foo/__init__.py`: `from ._foo import bar as bar`.
4. In the consumer (`SymbolicDSGE/foo/...`), `try` the native import and fall
   back to the numba kernel on `ImportError`.
5. Add a parity test under `tests/_ckernels/`.

`setup.py` auto-discovers any `_ckernels/*/_*.pyx` and links its sibling `*.c`
plus all `_common/*.c` — no per-subsystem build edits needed.
