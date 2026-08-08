# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C OccBin kernels.

Buffer to pointer marshalling and the GIL release only; the algorithms live in
occbin.c and regime_pencil.c. The kernels' f64/i8/i64 are exactly
double/int8_t/int64_t, so the externs are declared with those.
"""

from libc.stdint cimport int8_t, int64_t

import numpy as np


cdef extern from "../_common/sdsge_common.h" nogil:
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int


cdef extern from "occbin.h" nogil:
    ctypedef void (*sdsge_constraint_fn)(
        double *cur, double *par, int8_t *flags) noexcept
    int64_t sdsge_constraint_path(
        sdsge_constraint_fn cond, double *path, double *par,
        const int8_t *regime_in, int8_t *regime_out,
        int64_t T, int64_t n_var, int64_t n_constraint)


cdef extern from "regime_pencil.h" nogil:
    ctypedef void (*sdsge_regime_pencil_fn)(
        const double *cur, const double *par, double *out) noexcept
    ctypedef struct regime_ctx:
        sdsge_regime_pencil_fn pencil
        int64_t n_row
        const int64_t *rows
        double *a
        double *b
        double *c
    arena_size sdsge_regime_pencil_arena_size(int64_t n_var, int64_t n_row)
    void sdsge_regime_pencil(
        regime_ctx *regime, const double *ss, const double *par,
        const double *a_ref, const double *b_ref,
        int64_t n_var, double *arena)


#: Constraints the native flag buffer is sized for (``i8 flags[4]`` in occbin.c,
#: two slots per constraint). Mirrors the OccBin cap in model_parser.
MAX_CONSTRAINTS = 2


def constraint_path(size_t cond_addr, path, par, regime_in,
                    int64_t n_constraint, out=None):
    """Latched regime mask ``(T,)`` from a constraint @cfunc over a path.

    ``path`` is ``(T, n_var)`` in cur-variable order and in levels, not
    deviations from the reference steady state. ``regime_in`` is the incoming
    ``(T,)`` mask over declaration-ordered constraints, which the latch carries
    across guess-and-verify iterations at a fixed period; periods never interact.

    ``out`` is an optional C-contiguous int8 buffer written in place, and may be
    ``regime_in`` itself to latch in place. Other inputs are coerced. Returns
    ``(out, changed)``, where ``changed`` counts the periods whose mask moved and
    so is 0 exactly at a fixed point.
    """
    if not 0 < n_constraint <= MAX_CONSTRAINTS:
        raise ValueError(
            f"n_constraint must be in 1..{MAX_CONSTRAINTS}, got {n_constraint}."
        )

    cdef double[:, ::1] pv = np.ascontiguousarray(path, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(par, dtype=np.float64)
    cdef const int8_t[::1] inv = np.ascontiguousarray(regime_in, dtype=np.int8)
    cdef int64_t T = pv.shape[0]
    cdef int64_t n_var = pv.shape[1]
    cdef int8_t[::1] ov
    cdef int64_t changed

    if inv.shape[0] != T:
        raise ValueError(
            f"regime_in has length {inv.shape[0]}, expected {T} to match path."
        )
    if out is None:
        out = np.empty((T,), dtype=np.int8)
    ov = out
    if ov.shape[0] != T:
        raise ValueError(
            f"out has length {ov.shape[0]}, expected {T} to match path."
        )

    cdef double *path_ptr = &pv[0, 0] if (T * n_var) > 0 else NULL
    cdef double *par_ptr = &parv[0] if parv.shape[0] > 0 else NULL
    cdef const int8_t *in_ptr = &inv[0] if T > 0 else NULL
    cdef int8_t *out_ptr = &ov[0] if T > 0 else NULL
    cdef sdsge_constraint_fn fn = <sdsge_constraint_fn><void*>cond_addr
    with nogil:
        changed = sdsge_constraint_path(
            fn, path_ptr, par_ptr, in_ptr, out_ptr, T, n_var, n_constraint
        )
    return out, changed


def regime_pencil(size_t pencil_addr, rows, ss, par, a_ref, b_ref):
    """One regime's pencil ``(a, b, c)``, from the reference pencil and a patch.

    ``pencil_addr`` is the ``.address`` of a regime pencil @cfunc
    (``CompiledModel.construct_regime_pencil_func()``), and ``rows`` the
    reference rows that regime replaces. The reference pencil is copied, then
    those rows are overwritten; ``c`` is the regime's residual at ``ss`` and is
    zero off ``rows``, so ``a E[y+] = b y - c``.

    ``pencil_addr`` of 0 is the reference regime itself: the copy alone, with
    ``c`` zero and ``rows`` ignored. ``ss`` is the *reference* steady state in
    levels, which every regime linearizes around.

    Returns freshly allocated ``(a, b, c)``, shaped ``(n_var, n_var)``,
    ``(n_var, n_var)`` and ``(n_var,)``; nothing aliases the inputs.
    """
    cdef double[:, ::1] arefv = np.ascontiguousarray(a_ref, dtype=np.float64)
    cdef double[:, ::1] brefv = np.ascontiguousarray(b_ref, dtype=np.float64)
    cdef int64_t n_var = arefv.shape[0]

    if arefv.shape[1] != n_var:
        raise ValueError(
            f"a_ref is {arefv.shape[0]}x{arefv.shape[1]}, expected square."
        )
    if brefv.shape[0] != n_var or brefv.shape[1] != n_var:
        raise ValueError(
            f"b_ref is {brefv.shape[0]}x{brefv.shape[1]}, expected "
            f"{n_var}x{n_var} to match a_ref."
        )

    cdef double[::1] ssv = np.ascontiguousarray(ss, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(par, dtype=np.float64)
    cdef const int64_t[::1] rowv = np.ascontiguousarray(rows, dtype=np.int64)
    cdef int64_t n_row = rowv.shape[0]
    cdef int64_t i

    if ssv.shape[0] != n_var:
        raise ValueError(
            f"ss has length {ssv.shape[0]}, expected {n_var} to match a_ref."
        )
    # The kernel scatters straight into a[row], so an out-of-range row would
    # write past the output rather than raise.
    for i in range(n_row):
        if not 0 <= rowv[i] < n_var:
            raise ValueError(
                f"rows[{i}] is {rowv[i]}, outside 0..{n_var - 1}."
            )

    a = np.empty((n_var, n_var), dtype=np.float64)
    b = np.empty((n_var, n_var), dtype=np.float64)
    c = np.empty((n_var,), dtype=np.float64)
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b
    cdef double[::1] cv = c

    arena = np.empty(
        sdsge_regime_pencil_arena_size(n_var, n_row).n_float, dtype=np.float64
    )
    cdef double[::1] arv = arena

    cdef regime_ctx ctx
    ctx.pencil = <sdsge_regime_pencil_fn><void*>pencil_addr
    ctx.n_row = n_row
    ctx.rows = &rowv[0] if n_row > 0 else NULL
    ctx.a = &av[0, 0] if n_var > 0 else NULL
    ctx.b = &bv[0, 0] if n_var > 0 else NULL
    ctx.c = &cv[0] if n_var > 0 else NULL

    cdef const double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &parv[0] if parv.shape[0] > 0 else NULL
    cdef const double *a_ptr = &arefv[0, 0] if n_var > 0 else NULL
    cdef const double *b_ptr = &brefv[0, 0] if n_var > 0 else NULL
    cdef double *arena_ptr = &arv[0] if arv.shape[0] > 0 else NULL
    with nogil:
        sdsge_regime_pencil(
            &ctx, ss_ptr, par_ptr, a_ptr, b_ptr, n_var, arena_ptr
        )
    return a, b, c
