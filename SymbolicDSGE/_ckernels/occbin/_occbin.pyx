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


# regime_pencil.h first: occbin.h includes it and occbin_ctx holds a regime_ctx.
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


cdef extern from "occbin.h" nogil:
    ctypedef void (*sdsge_constraint_fn)(
        double *cur, double *par, double *err) noexcept
    int64_t sdsge_constraint_path(
        sdsge_constraint_fn cond, double *path, double *par,
        const int8_t *regime_in, int8_t *regime_out,
        int64_t inclusive, double *max_err, int64_t T,
        int64_t n_var, int64_t n_constraint)
    ctypedef struct occbin_ctx:
        const regime_ctx *table
        const double *f_ref
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
    arena_size sdsge_occbin_recursion_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl)
    int64_t sdsge_occbin_recursion(
        const occbin_ctx *ctx, const int8_t *mask, int64_t T, double *out,
        int64_t *singular_date, double *arena, int64_t *iarena)
    int64_t SDSGE_OCCBIN_RECURSION_OK


#: Constraints the native flag buffer is sized for (``i8 flags[4]`` in occbin.c,
#: two slots per constraint). Mirrors the OccBin cap in model_parser.
MAX_CONSTRAINTS = 2
MAX_REGIME = 4


def constraint_path(size_t cond_addr, path, par, regime_in,
                    int64_t n_constraint, int64_t inclusive, out=None):
    """Latched regime mask ``(T,)`` from a constraint @cfunc over a path.

    ``path`` is ``(T, n_var)`` in cur-variable order and in levels, not
    deviations from the reference steady state. ``regime_in`` is the incoming
    ``(T,)`` mask over declaration-ordered constraints, which the latch carries
    across guess-and-verify iterations at a fixed period; periods never interact.

    The cfunc writes signed distances, so ``inclusive``
    (``ConstraintFunc.inclusive``) is what decides a distance of exactly zero,
    which is where a condition written against the steady state starts.

    ``out`` is an optional C-contiguous int8 buffer written in place, and may be
    ``regime_in`` itself to latch in place. Other inputs are coerced. Returns
    ``(out, changed, max_err)``: ``changed`` counts the periods whose mask moved
    and so is 0 exactly at a fixed point, and ``max_err`` is the largest
    distance that moved one, which ranks iterations that cycle.
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
    cdef double max_err = 0.0

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
            fn, path_ptr, par_ptr, in_ptr, out_ptr, inclusive, &max_err,
            T, n_var, n_constraint,
        )
    return out, changed, max_err


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


def occbin_recursion_arena_size(int64_t n_var, int64_t n_state,
                                int64_t n_ctrl):
    """``(n_float, n_int)`` scratch ``occbin_recursion`` needs for a shape."""
    cdef arena_size sz = sdsge_occbin_recursion_arena_size(
        n_var, n_state, n_ctrl
    )
    return sz.n_float, sz.n_int


def occbin_recursion(a, b, c, mask, f_ref, out=None, arena=None, iarena=None):
    """Piecewise-linear decision rules ``(T, n_var, n_state + 1)`` for a guess.

    ``a``, ``b`` and ``c`` are the regime pencils stacked by bitmask, shaped
    ``(n_regime, n_var, n_var)``, ``(n_regime, n_var, n_var)`` and
    ``(n_regime, n_var)``. Slot ``m`` is what ``regime_pencil`` returns for mask
    ``m`` and slot 0 is the reference, so ``mask`` indexes the stack directly.

    ``f_ref`` is the reference control rule ``(n_ctrl, n_state)``, which closes
    the recursion past the last date, and fixes ``n_state`` and ``n_ctrl``.

    ``out`` is an optional C-contiguous ``(T, n_var, n_state + 1)`` float64
    buffer written in place. At date ``t`` the block is the affine map from
    ``x_t`` to ``[x_{t+1}; u_t]``: ``out[t, :, :n_state] @ x_t + out[t, :,
    n_state]``, state rows first. Raises ``RuntimeError`` naming the date if a
    date's pencil is singular.

    ``arena`` and ``iarena`` are optional scratch buffers, at least as long as
    ``occbin_recursion_arena_size`` reports; both are allocated if omitted.
    """
    cdef double[:, :, ::1] av = np.ascontiguousarray(a, dtype=np.float64)
    cdef double[:, :, ::1] bv = np.ascontiguousarray(b, dtype=np.float64)
    cdef double[:, ::1] cv = np.ascontiguousarray(c, dtype=np.float64)
    cdef double[:, ::1] fv = np.ascontiguousarray(f_ref, dtype=np.float64)
    cdef const int8_t[::1] mv = np.ascontiguousarray(mask, dtype=np.int8)

    cdef int64_t n_regime = av.shape[0]
    cdef int64_t n_var = av.shape[1]
    cdef int64_t n_ctrl = fv.shape[0]
    cdef int64_t n_state = fv.shape[1]
    cdef int64_t T = mv.shape[0]
    cdef int64_t n_rhs = n_state + 1
    cdef int64_t i

    if n_regime < 1:
        raise ValueError("a must hold at least the reference regime.")
    if n_regime > MAX_REGIME:
        raise ValueError(f"a holds {n_regime} regimes, at most {MAX_REGIME}.")
    if av.shape[2] != n_var:
        raise ValueError(
            f"a[0] is {av.shape[1]}x{av.shape[2]}, expected square."
        )
    if bv.shape[0] != n_regime or bv.shape[1] != n_var or bv.shape[2] != n_var:
        raise ValueError(
            f"b is {bv.shape[0]}x{bv.shape[1]}x{bv.shape[2]}, expected "
            f"{n_regime}x{n_var}x{n_var} to match a."
        )
    if cv.shape[0] != n_regime or cv.shape[1] != n_var:
        raise ValueError(
            f"c is {cv.shape[0]}x{cv.shape[1]}, expected {n_regime}x{n_var} "
            f"to match a."
        )
    if n_state + n_ctrl != n_var:
        raise ValueError(
            f"f_ref is {n_ctrl}x{n_state}, so n_state + n_ctrl is "
            f"{n_state + n_ctrl}, expected {n_var} to match a."
        )
    # The kernel indexes table[mask[t]] without a bound check.
    for i in range(T):
        if not 0 <= mv[i] < n_regime:
            raise ValueError(
                f"mask[{i}] is {mv[i]}, outside 0..{n_regime - 1}."
            )

    if out is None:
        out = np.empty((T, n_var, n_rhs), dtype=np.float64)
    cdef double[:, :, ::1] ov = out
    if ov.shape[0] != T or ov.shape[1] != n_var or ov.shape[2] != n_rhs:
        raise ValueError(
            f"out is {ov.shape[0]}x{ov.shape[1]}x{ov.shape[2]}, expected "
            f"{T}x{n_var}x{n_rhs}."
        )

    cdef arena_size sz = sdsge_occbin_recursion_arena_size(
        n_var, n_state, n_ctrl
    )
    if arena is None:
        arena = np.empty(sz.n_float, dtype=np.float64)
    if iarena is None:
        iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iav = iarena
    if arv.shape[0] < sz.n_float or iav.shape[0] < sz.n_int:
        raise ValueError(
            f"arena is {arv.shape[0]}/{iav.shape[0]} floats/ints, needs "
            f"{sz.n_float}/{sz.n_int}."
        )

    cdef const int8_t *mask_ptr = &mv[0] if T > 0 else NULL
    cdef double *out_ptr = &ov[0, 0, 0] if (T * n_var * n_rhs) > 0 else NULL
    cdef double *arena_ptr = &arv[0] if arv.shape[0] > 0 else NULL
    cdef int64_t *iarena_ptr = &iav[0] if iav.shape[0] > 0 else NULL

    cdef occbin_ctx ctx
    cdef regime_ctx table[4]  # MAX_REGIME
    cdef int64_t singular_date = -1
    cdef int64_t status

    for i in range(n_regime):
        table[i].pencil = NULL
        table[i].n_row = 0
        table[i].rows = NULL
        table[i].a = &av[i, 0, 0]
        table[i].b = &bv[i, 0, 0]
        table[i].c = &cv[i, 0]

    ctx.table = table
    ctx.f_ref = &fv[0, 0] if (n_ctrl * n_state) > 0 else NULL
    ctx.n_var = n_var
    ctx.n_state = n_state
    ctx.n_ctrl = n_ctrl

    with nogil:
        status = sdsge_occbin_recursion(
            &ctx, mask_ptr, T, out_ptr, &singular_date, arena_ptr, iarena_ptr
        )

    if status != SDSGE_OCCBIN_RECURSION_OK:
        raise RuntimeError(
            f"occbin_recursion: singular pencil at date {singular_date}."
        )
    return out
