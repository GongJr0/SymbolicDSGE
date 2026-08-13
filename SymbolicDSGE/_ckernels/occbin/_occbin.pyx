# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C OccBin kernels.

Buffer to pointer marshalling and the GIL release only; the algorithms live in
occbin.c and regime_pencil.c. The kernels' f64/i8/i64 are exactly
double/int8_t/int64_t, so the externs are declared with those.
"""

from libc.stdint cimport int8_t, int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer

import math
import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack


cdef extern from "../_common/sdsge_common.h" nogil:
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int


cdef extern from "../_common/sdsge_complex.h" nogil:
    ctypedef struct c128:
        pass


# The reference half of the piecewise solve is core's Klein kernel, so occbin
# links core's C (``_EXTRA_DEPS`` in setup.py) and redeclares what it calls.
cdef extern from "../core/klein_preproc.h" nogil:
    ctypedef void (*sdsge_residual_fn)(
        c128 *fwd, c128 *cur, c128 *prev, c128 *eps, c128 *par, c128 *out)


cdef extern from "../core/klein_qz.h" nogil:
    ctypedef void (*klein_zgges_fn)()


cdef extern from "../core/pencil.h" nogil:
    ctypedef void (*sdsge_dgeqrf_fn)()
    ctypedef void (*sdsge_dormqr_fn)()
    int64_t sdsge_pencil_dim(const signed char *incidence, int64_t n_var)


cdef extern from "../core/klein_solve.h" nogil:
    ctypedef struct klein_spec:
        sdsge_residual_fn residual
        klein_zgges_fn zgges
        sdsge_dgeqrf_fn dgeqrf
        sdsge_dormqr_fn dormqr
        const double *ss_seed
        const double *params
        const signed char *incidence
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_par
    ctypedef struct sdsge_solve1:
        double *ss
        double *a_real
        double *b_real
        double *c_real
        double *d_real
        c128 *s
        c128 *t
        c128 *z
        double *f
        double *p
        c128 *eig
        int64_t stab
        double *A
        double *B
        int64_t *order
        int64_t n_static
        int64_t n_pred
        int64_t n_both
        int64_t n_fwd
    int SDSGE_KLEIN_SOLVE_SS_SINGULAR
    int SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE
    int SDSGE_KLEIN_SOLVE_QZ
    int SDSGE_KLEIN_SOLVE_SINGULAR
    int SDSGE_KLEIN_SOLVE_NO_STATES


# LAPACK ``zgges`` reached through its scipy ``cython_lapack`` capsule address,
# the same runtime-address mechanism ``_core.pyx`` uses. Each extension casts
# the pointer once at import; the capsule is the shared thing, not the cast.
cdef object _zgges_capsule = _cython_lapack.__pyx_capi__["zgges"]
cdef klein_zgges_fn _zgges = <klein_zgges_fn>PyCapsule_GetPointer(
    _zgges_capsule, PyCapsule_GetName(_zgges_capsule)
)

# The static rotation's QR, reached the same way. `Q` is never formed: dgeqrf
# leaves the reflectors in place and dormqr applies Q' straight to each block.
cdef object _dgeqrf_capsule = _cython_lapack.__pyx_capi__["dgeqrf"]
cdef sdsge_dgeqrf_fn _dgeqrf = <sdsge_dgeqrf_fn>PyCapsule_GetPointer(
    _dgeqrf_capsule, PyCapsule_GetName(_dgeqrf_capsule)
)
cdef object _dormqr_capsule = _cython_lapack.__pyx_capi__["dormqr"]
cdef sdsge_dormqr_fn _dormqr = <sdsge_dormqr_fn>PyCapsule_GetPointer(
    _dormqr_capsule, PyCapsule_GetName(_dormqr_capsule)
)


# regime_pencil.h first: occbin.h includes it for the solve's pencil cfuncs.
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
        double *d
        double *cst
    arena_size sdsge_regime_pencil_arena_size(
        int64_t n_var, int64_t n_exog, int64_t n_row)
    void sdsge_regime_pencil(
        regime_ctx *regime, const double *ss, const double *par,
        const double *a_ref, const double *b_ref, const double *c_ref,
        const double *d_ref, int64_t n_var, int64_t n_exog, double *arena)


cdef extern from "occbin.h" nogil:
    ctypedef void (*sdsge_constraint_fn)(
        double *cur, double *par, double *err) noexcept
    int64_t sdsge_constraint_path(
        sdsge_constraint_fn cond, double *path, double *par,
        const int8_t *regime_in, int8_t *regime_out,
        int64_t inclusive, double *max_err, int64_t T,
        int64_t n_var, int64_t n_constraint)
    ctypedef struct occbin_ctx:
        const double *a
        const double *b
        const double *c
        const double *f_ref
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
    ctypedef struct occbin_opts:
        int64_t periodic_solution
        int64_t periodic_threshold
        int64_t periodic_strict
        int64_t curb_retrench
        int64_t reset_regime
        int64_t reset_check_ahead
        int64_t algo_truncation
    ctypedef struct occbin_run_ctx:
        const occbin_ctx *model
        sdsge_constraint_fn cond
        double *par
        const double *ss
        int64_t inclusive
        int64_t n_constraint
        int64_t max_iter
        int64_t T0
        int64_t T_cap
        occbin_opts opts
    ctypedef struct occbin_diag:
        int64_t *iters
        double *max_err
        int8_t *periodic
        int64_t fail_period
        int64_t singular_date
    arena_size sdsge_occbin_recursion_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl)
    int64_t sdsge_occbin_recursion(
        const occbin_ctx *ctx, const int8_t *mask, int64_t T, double *out,
        int64_t *singular_date, double *arena, int64_t *iarena)
    void sdsge_occbin_forward(
        const double *rule, const double *x0, int64_t T, int64_t n_var,
        int64_t n_state, double *path)
    arena_size sdsge_occbin_sim_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t T_cap,
        int64_t max_iter)
    int64_t sdsge_occbin_sim(
        const occbin_run_ctx *run, const double *shocks, int64_t S,
        int64_t n_periods, const double *x_init, const int8_t *init_mask,
        int64_t n_init, double *out, int8_t *regimes, int64_t *T_used,
        occbin_diag *diag, double *rule, double *path, int8_t *mask,
        double *arena, int64_t *iarena)
    int64_t SDSGE_OCCBIN_RECURSION_OK
    int64_t SDSGE_OCCBIN_RECURSION_SINGULAR
    int64_t SDSGE_OCCBIN_PERIOD_OK
    int64_t SDSGE_OCCBIN_PERIODIC
    int64_t SDSGE_OCCBIN_PERIODIC_LOOP
    int64_t SDSGE_OCCBIN_MAXITER
    ctypedef struct occbin_solve1_spec:
        klein_spec first
        const sdsge_regime_pencil_fn *pencil
        const int64_t **rows
        const int64_t *n_row
        int64_t n_constraint
    ctypedef struct sdsge_occbin1:
        sdsge_solve1 ref
        double *a
        double *b
        double *c
        double *d
        double *cst
    arena_size sdsge_occbin_solve1_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd, int64_t max_n_row)
    int64_t sdsge_occbin_solve1(
        const occbin_solve1_spec *spec, sdsge_occbin1 *out, double *arena,
        int64_t *iarena)


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
    if not 0 < n_constraint <= 2:
        raise ValueError(
            "A model may declare one or two constraints under "
            f"equations.constraint, got {n_constraint}."
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


def regime_pencil(size_t pencil_addr, rows, ss, par, a_ref, b_ref, c_ref, d_ref):
    """One regime's pencil ``(a, b, c, d, cst)``, from the reference and a patch.

    ``pencil_addr`` is the ``.address`` of a regime pencil @cfunc
    (``CompiledModel.construct_regime_pencil_func()``), and ``rows`` the
    reference rows that regime replaces. The reference pencil is copied, then
    those rows are overwritten; ``cst`` is the regime's residual at ``ss`` and
    is zero off ``rows``, so ``a E[y+] = b y + c y_prev + d eps - cst``.

    ``pencil_addr`` of 0 is the reference regime itself: the copy alone, with
    ``cst`` zero and ``rows`` ignored. ``ss`` is the *reference* steady state in
    levels, which every regime linearizes around.

    Returns freshly allocated ``(a, b, c, d, cst)``; the three date blocks are
    ``(n_var, n_var)``, ``d`` is ``(n_var, n_exog)`` and ``cst`` is ``(n_var,)``.
    Nothing aliases the inputs.
    """
    cdef double[:, ::1] arefv = np.ascontiguousarray(a_ref, dtype=np.float64)
    cdef double[:, ::1] brefv = np.ascontiguousarray(b_ref, dtype=np.float64)
    cdef double[:, ::1] crefv = np.ascontiguousarray(c_ref, dtype=np.float64)
    cdef double[:, ::1] drefv = np.ascontiguousarray(d_ref, dtype=np.float64)
    cdef int64_t n_var = arefv.shape[0]
    cdef int64_t n_exog = drefv.shape[1]

    if arefv.shape[1] != n_var:
        raise ValueError(
            f"a_ref is {arefv.shape[0]}x{arefv.shape[1]}, expected square."
        )
    if brefv.shape[0] != n_var or brefv.shape[1] != n_var:
        raise ValueError(
            f"b_ref is {brefv.shape[0]}x{brefv.shape[1]}, expected "
            f"{n_var}x{n_var} to match a_ref."
        )
    if crefv.shape[0] != n_var or crefv.shape[1] != n_var:
        raise ValueError(
            f"c_ref is {crefv.shape[0]}x{crefv.shape[1]}, expected "
            f"{n_var}x{n_var} to match a_ref."
        )
    if drefv.shape[0] != n_var:
        raise ValueError(
            f"d_ref has {drefv.shape[0]} rows, expected {n_var} to match a_ref."
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
    c = np.empty((n_var, n_var), dtype=np.float64)
    d = np.empty((n_var, n_exog), dtype=np.float64)
    cst = np.empty((n_var,), dtype=np.float64)
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b
    cdef double[:, ::1] cv = c
    cdef double[:, ::1] dv = d
    cdef double[::1] cstv = cst

    arena = np.empty(
        sdsge_regime_pencil_arena_size(n_var, n_exog, n_row).n_float,
        dtype=np.float64,
    )
    cdef double[::1] arv = arena

    cdef regime_ctx ctx
    ctx.pencil = <sdsge_regime_pencil_fn><void*>pencil_addr
    ctx.n_row = n_row
    ctx.rows = &rowv[0] if n_row > 0 else NULL
    ctx.a = &av[0, 0] if n_var > 0 else NULL
    ctx.b = &bv[0, 0] if n_var > 0 else NULL
    ctx.c = &cv[0, 0] if n_var > 0 else NULL
    ctx.d = &dv[0, 0] if n_var > 0 and n_exog > 0 else NULL
    ctx.cst = &cstv[0] if n_var > 0 else NULL

    cdef const double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &parv[0] if parv.shape[0] > 0 else NULL
    cdef const double *a_ptr = &arefv[0, 0] if n_var > 0 else NULL
    cdef const double *b_ptr = &brefv[0, 0] if n_var > 0 else NULL
    cdef const double *c_ptr = &crefv[0, 0] if n_var > 0 else NULL
    cdef const double *d_ptr = (
        &drefv[0, 0] if n_var > 0 and n_exog > 0 else NULL)
    cdef double *arena_ptr = &arv[0] if arv.shape[0] > 0 else NULL
    with nogil:
        sdsge_regime_pencil(
            &ctx, ss_ptr, par_ptr, a_ptr, b_ptr, c_ptr, d_ptr, n_var, n_exog,
            arena_ptr
        )
    return a, b, c, d, cst


cdef _check_pencils(double[:, :, ::1] av, double[:, :, ::1] bv,
                    double[:, ::1] cv, double[:, ::1] fv):
    """``(n_regime, n_var, n_state, n_ctrl)`` for stacked pencils and ``f_ref``.

    The kernels index the stack and partition the rows without a bound check,
    so every shape relation they assume is settled here.
    """
    cdef int64_t n_regime = av.shape[0]
    cdef int64_t n_var = av.shape[1]
    cdef int64_t n_ctrl = fv.shape[0]
    cdef int64_t n_state = fv.shape[1]

    if n_regime < 1:
        raise ValueError("a must hold at least the reference regime.")
    if n_regime > 4:
        raise ValueError(f"a holds {n_regime} regimes, at most 4.")
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
    return n_regime, n_var, n_state, n_ctrl


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

    cdef int64_t n_regime, n_var, n_state, n_ctrl
    n_regime, n_var, n_state, n_ctrl = _check_pencils(av, bv, cv, fv)

    cdef int64_t T = mv.shape[0]
    cdef int64_t n_rhs = n_state + 1
    cdef int64_t i

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
    cdef int64_t singular_date = -1
    cdef int64_t status

    ctx.a = &av[0, 0, 0]
    ctx.b = &bv[0, 0, 0]
    ctx.c = &cv[0, 0]
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


def occbin_forward(rule, x0, out=None):
    """``(T, n_var)`` path in deviations, from decision rules and a state.

    ``rule`` is a stack of blocks as ``occbin_recursion`` returns them, ``(T,
    n_var, n_state + 1)``, and ``x0`` the ``(n_state,)`` state entering date 0.

    Row ``t`` is ``[x_t; u_t]``: the state half comes from date ``t - 1``'s
    block and the control half from date ``t``'s, which is the pairing a
    constraint on period ``t`` reads. The state the last block implies falls off
    the end. ``out`` is an optional C-contiguous buffer written in place.
    """
    cdef double[:, :, ::1] rv = np.ascontiguousarray(rule, dtype=np.float64)
    cdef double[::1] xv = np.ascontiguousarray(x0, dtype=np.float64)

    cdef int64_t T = rv.shape[0]
    cdef int64_t n_var = rv.shape[1]
    cdef int64_t n_state = rv.shape[2] - 1

    if n_state < 0 or n_state > n_var:
        raise ValueError(
            f"rule is {T}x{n_var}x{rv.shape[2]}, expected a last axis in "
            f"1..{n_var + 1}."
        )
    if xv.shape[0] != n_state:
        raise ValueError(
            f"x0 has length {xv.shape[0]}, expected {n_state} to match rule."
        )

    if out is None:
        out = np.empty((T, n_var), dtype=np.float64)
    cdef double[:, ::1] ov = out
    if ov.shape[0] != T or ov.shape[1] != n_var:
        raise ValueError(
            f"out is {ov.shape[0]}x{ov.shape[1]}, expected {T}x{n_var}."
        )

    cdef const double *rule_ptr = &rv[0, 0, 0] if T * n_var > 0 else NULL
    cdef const double *x_ptr = &xv[0] if n_state > 0 else NULL
    cdef double *out_ptr = &ov[0, 0] if T * n_var > 0 else NULL
    with nogil:
        sdsge_occbin_forward(rule_ptr, x_ptr, T, n_var, n_state, out_ptr)
    return out


def occbin_sim_arena_size(int64_t n_var, int64_t n_state, int64_t n_ctrl,
                          int64_t T_cap, int64_t max_iter):
    """``(n_float, n_int)`` scratch ``occbin_sim`` needs for a shape."""
    cdef arena_size sz = sdsge_occbin_sim_arena_size(
        n_var, n_state, n_ctrl, T_cap, max_iter
    )
    return sz.n_float, sz.n_int


def occbin_sim(a, b, c, f_ref, ss, par, size_t cond_addr,
               int64_t n_constraint, int64_t inclusive, shocks, x_init,
               *,
               int64_t check_ahead_periods=200, int64_t max_check_ahead_periods=-1,
               n_periods=None, int64_t max_iter=30, init_regime=None,
               bint periodic_solution=False, int64_t periodic_threshold=1,
               bint periodic_strict=True, bint curb_retrench=False,
               bint reset_regime=False, bint reset_check_ahead=False,
               int64_t algo_truncation=1):
    """Piecewise path ``(n_periods, n_var)`` for a sequence of unforeseen shocks.

    ``a``, ``b``, ``c`` and ``f_ref`` are the pencils and reference rule
    ``occbin_recursion`` takes, except that the stack must be complete: the latch
    produces every mask in ``0..2 ** n_constraint - 1``, so all of them must be
    slots. ``ss`` is the ``(n_var,)`` reference steady state, which turns the
    path into the levels the constraint @cfunc at ``cond_addr`` reads.

    ``shocks`` is ``(S, n_state)`` in state slots, added to the carried state at
    the head of each period. A period whose row is all zero is not re-solved:
    the previous guess and the path it implies, both shifted forward one period,
    already describe it. Period 0 always solves.

    ``check_ahead_periods`` is how far ahead a guess looks and
    ``max_check_ahead_periods`` how far it may grow, one date at a time, whenever
    a guess still binds at its last date. At ``-1`` the growth budget is
    ``ceil(sqrt(check_ahead_periods))`` past the horizon; equal to
    ``check_ahead_periods`` it forbids growth and forces the last date relaxed
    instead. Dynare's ``inf`` has no counterpart: growth is reserved up front,
    so it is always bounded here.

    ``n_periods`` defaults to ``S`` and any excess is filled from the tail of the
    last path solved. ``init_regime`` is one guess row per shock period, in the
    layout ``regimes`` comes back in, so a run's guesses feed straight back in;
    it may cover the leading periods only, and the rest carry the previous
    period's guess shifted. All relaxed by default. Rows shorter than the buffer
    relax into the tail and rows longer than ``check_ahead_periods`` raise the
    horizon to their own length, both as Dynare does, so a guess round-trips
    between runs whose horizons differ. Only a row past the growth cap is an
    error.

    The remaining keywords are Dynare's ``occbin.simul`` options at their own
    defaults. ``periodic_solution`` accepts the best guess of a cycle instead of
    failing; ``periodic_threshold`` is how far a guess may move and still count
    as a two-cycle rather than a loop; ``periodic_strict`` refuses to weigh a
    loop that is not one; ``curb_retrench`` relaxes one date per pass;
    ``reset_regime`` starts every period from the relaxed guess, dropping a
    grown horizon too when ``reset_check_ahead`` is set; and a ``max_iter`` at
    or below ``algo_truncation`` accepts whatever the last pass produced.

    Returns ``(out, regimes, diag)``. ``out`` is in deviations, ``regimes`` is
    the accepted guess per period, padded to the widest horizon any period grew
    to, and ``diag`` carries ``T_used``, ``iters``, ``max_err`` and
    ``periodic``, one entry per period. ``T_used`` is how much of that period's
    row is the guess and how much is padding.
    ``periodic`` is nonzero where a period was accepted under
    ``periodic_solution``, carrying the code it would otherwise have raised.
    Raises ``RuntimeError`` naming the period that failed to converge.
    """
    cdef double[:, :, ::1] av = np.ascontiguousarray(a, dtype=np.float64)
    cdef double[:, :, ::1] bv = np.ascontiguousarray(b, dtype=np.float64)
    cdef double[:, ::1] cv = np.ascontiguousarray(c, dtype=np.float64)
    cdef double[:, ::1] fv = np.ascontiguousarray(f_ref, dtype=np.float64)

    cdef int64_t n_regime, n_var, n_state, n_ctrl
    n_regime, n_var, n_state, n_ctrl = _check_pencils(av, bv, cv, fv)

    if n_state < 1:
        raise ValueError("f_ref must hold at least one state; the solve "
                         "carries the path forward through the states.")
    if not 0 < n_constraint <= 2:
        raise ValueError(
            "A model may declare one or two constraints under "
            f"equations.constraint, got {n_constraint}."
        )
    # The latch emits every mask over n_constraint bits and the kernel indexes
    # table[mask[t]] with it, so a partial stack would read past the table.
    if n_regime != 1 << n_constraint:
        raise ValueError(
            f"a holds {n_regime} regimes, expected {1 << n_constraint} to "
            f"cover every mask over {n_constraint} constraints."
        )

    if check_ahead_periods < 1:
        raise ValueError(
            "check_ahead_periods is how many periods ahead a guess looks, it "
            f"must be at least 1, got {check_ahead_periods}."
        )

    cdef int64_t T0 = check_ahead_periods + 1
    cdef int64_t T_cap
    if max_check_ahead_periods == -1:
        root = math.isqrt(check_ahead_periods)
        T_cap = T0 + root + int(root * root < check_ahead_periods)
    elif max_check_ahead_periods < check_ahead_periods:
        raise ValueError(
            "max_check_ahead_periods takes the default growth budget at -1. "
            "Any other value is a horizon and must be at least "
            f"check_ahead_periods of {check_ahead_periods}, got "
            f"{max_check_ahead_periods}."
        )
    else:
        # Equal to check_ahead_periods this lands on T0, forbidding growth.
        T_cap = max_check_ahead_periods + 1

    if max_iter < 1:
        raise ValueError(f"max_iter is {max_iter}, expected at least 1.")

    cdef double[::1] ssv = np.ascontiguousarray(ss, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(par, dtype=np.float64)
    cdef double[:, ::1] shv = np.ascontiguousarray(shocks, dtype=np.float64)
    cdef double[::1] xv = np.ascontiguousarray(x_init, dtype=np.float64)
    cdef int64_t S = shv.shape[0]

    if ssv.shape[0] != n_var:
        raise ValueError(
            f"ss has length {ssv.shape[0]}, expected {n_var} to match a."
        )
    if S < 1:
        raise ValueError("shocks must hold at least one period.")
    if shv.shape[1] != n_state:
        raise ValueError(
            f"shocks is {S}x{shv.shape[1]}, expected {S}x{n_state} to match "
            f"f_ref."
        )
    if xv.shape[0] != n_state:
        raise ValueError(
            f"x_init has length {xv.shape[0]}, expected {n_state} to match "
            f"f_ref."
        )

    cdef int8_t[:, ::1] imv
    cdef const int8_t *im = NULL
    cdef int64_t n_init = 0
    if init_regime is not None:
        init_arr = np.ascontiguousarray(init_regime, dtype=np.int8)
        if init_arr.ndim != 2:
            raise ValueError(
                f"init_regime is {init_arr.ndim}-D, expected one guess row per "
                f"shock period."
            )
        n_init = init_arr.shape[0]
        if not 0 < n_init <= S:
            raise ValueError(
                f"init_regime holds {n_init} shock periods, expected 1..{S}."
            )
        width = init_arr.shape[1]
        if width < 1:
            raise ValueError("init_regime rows must cover at least one date.")
        if width > T_cap:
            raise ValueError(
                f"init_regime covers {width} dates, more than the "
                f"{T_cap} a max_check_ahead_periods of {T_cap - 1} reserves. "
                "Raise max_check_ahead_periods, or shorten the guess."
            )
        if np.any(init_arr >= (1 << n_constraint)) or np.any(init_arr < 0):
            raise ValueError(
                f"init_regime holds a mask outside 0..{(1 << n_constraint) - 1}."
            )
        # A guess is authoritative about the horizon it was solved on, as in
        # Dynare, but only up to what the arena reserved. Short guesses relax
        # into the tail, which is where a converged one ends anyway.
        if width > T0:
            T0 = width
        if width < T_cap:
            padded = np.zeros((n_init, T_cap), dtype=np.int8)
            padded[:, :width] = init_arr
            init_arr = padded
        imv = init_arr
        im = &imv[0, 0]

    cdef int64_t P = S if n_periods is None else n_periods
    if P < S:
        raise ValueError(f"n_periods is {P}, expected at least S of {S}.")
    # The tail is read off one path, which is only T0 long to start with.
    if P - S > T0:
        raise ValueError(
            f"n_periods is {P}, at most S + T0 of {S + T0} for a tail the "
            f"last path can fill."
        )

    out = np.empty((P, n_var), dtype=np.float64)
    regimes = np.zeros((S, T_cap), dtype=np.int8)
    T_used = np.zeros(S, dtype=np.int64)
    iters = np.zeros(S, dtype=np.int64)
    max_err = np.zeros(S, dtype=np.float64)
    periodic = np.zeros(S, dtype=np.int8)
    cdef double[:, ::1] ov = out
    cdef int8_t[:, ::1] rgv = regimes
    cdef int64_t[::1] tuv = T_used
    cdef int64_t[::1] itv = iters
    cdef double[::1] mev = max_err
    cdef int8_t[::1] pev = periodic

    rule = np.empty((T_cap, n_var, n_state + 1), dtype=np.float64)
    path = np.empty((T_cap, n_var), dtype=np.float64)
    mask = np.empty(T_cap, dtype=np.int8)
    cdef double[:, :, ::1] rulev = rule
    cdef double[:, ::1] pathv = path
    cdef int8_t[::1] maskv = mask

    cdef arena_size sz = sdsge_occbin_sim_arena_size(
        n_var, n_state, n_ctrl, T_cap, max_iter
    )
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iav = iarena

    cdef occbin_ctx model
    model.a = &av[0, 0, 0]
    model.b = &bv[0, 0, 0]
    model.c = &cv[0, 0]
    model.f_ref = &fv[0, 0] if n_ctrl * n_state > 0 else NULL
    model.n_var = n_var
    model.n_state = n_state
    model.n_ctrl = n_ctrl

    cdef occbin_run_ctx run
    run.model = &model
    run.cond = <sdsge_constraint_fn><void*>cond_addr
    run.par = &parv[0] if parv.shape[0] > 0 else NULL
    run.ss = &ssv[0] if n_var > 0 else NULL
    run.inclusive = inclusive
    run.n_constraint = n_constraint
    run.max_iter = max_iter
    run.T0 = T0
    run.T_cap = T_cap
    run.opts.periodic_solution = periodic_solution
    run.opts.periodic_threshold = periodic_threshold
    run.opts.periodic_strict = periodic_strict
    run.opts.curb_retrench = curb_retrench
    run.opts.reset_regime = reset_regime
    run.opts.reset_check_ahead = reset_check_ahead
    run.opts.algo_truncation = algo_truncation

    cdef occbin_diag diag
    diag.iters = &itv[0]
    diag.max_err = &mev[0]
    diag.periodic = &pev[0]
    diag.fail_period = -1
    diag.singular_date = -1

    cdef int64_t status
    with nogil:
        status = sdsge_occbin_sim(
            &run, &shv[0, 0], S, P, &xv[0], im, n_init, &ov[0, 0], &rgv[0, 0],
            &tuv[0], &diag, &rulev[0, 0, 0], &pathv[0, 0], &maskv[0], &arv[0],
            &iav[0]
        )

    if status != SDSGE_OCCBIN_PERIOD_OK:
        raise RuntimeError(_sim_error(status, &diag, max_iter))
    # int8 is the kernel's wire format; callers get a width that will not wrap.
    return out, regimes.astype(np.int64), {
        "T_used": T_used,
        "iters": iters,
        "max_err": max_err,
        "periodic": periodic.astype(np.int64),
    }


cdef _sim_error(int64_t status, occbin_diag *diag, int64_t max_iter):
    """Dynare's error code for a period that never settled, as a message."""
    cdef int64_t s = diag.fail_period
    if status == SDSGE_OCCBIN_RECURSION_SINGULAR:
        return (
            f"occbin_sim: singular pencil at date {diag.singular_date} of "
            f"shock period {s}."
        )
    if status == SDSGE_OCCBIN_PERIODIC:
        return f"occbin_sim: shock period {s} loops between two regimes."
    if status == SDSGE_OCCBIN_PERIODIC_LOOP:
        return f"occbin_sim: shock period {s} loops over guess regimes."
    if status == SDSGE_OCCBIN_MAXITER:
        return (
            f"occbin_sim: shock period {s} did not converge in {max_iter} "
            f"iterations."
        )
    return f"occbin_sim: shock period {s} failed with status {status}."


cdef _raise_klein_error(int64_t err):
    """The reference solve's failures, worded as ``_core.pyx`` words them.

    ``solver_backend._bk_dating_hint`` matches on the Blanchard-Kahn text, so a
    piecewise model has to fail in the same words a reference one does.
    """
    if err == SDSGE_KLEIN_SOLVE_SS_SINGULAR:
        raise ValueError("steady_state_newton: singular Jacobian (a - b).")
    if err == SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE:
        raise ValueError(
            "steady_state_newton: did not converge within max_iter "
            "(or the residual went non-finite)."
        )
    if err == SDSGE_KLEIN_SOLVE_QZ:
        raise RuntimeError("klein_qz: LAPACK zgges failed.")
    if err == SDSGE_KLEIN_SOLVE_SINGULAR:
        raise ValueError(
            "klein_postprocess: singular z11/s11 (Blanchard-Kahn failure)."
        )
    if err == SDSGE_KLEIN_SOLVE_NO_STATES:
        raise ValueError("klein_postprocess: model has no states.")


def occbin_solve1(size_t residual_addr, seed, params, incidence,
                  int64_t n_state, pencil_addrs, rows, int64_t n_constraint,
                  int64_t n_exog=0):
    """Reference solve and every regime's pencil, in a single GIL release.

    ``pencil_addrs`` and ``rows`` are indexed by binding bitmask and dense over
    ``0..2 ** n_constraint - 1``. Pass address 0 and an empty rows array for
    slot 0: that is the reference regime, and it comes back as the pencil the
    reference solve linearized at.

    Returns ``(ss, f, p, stab, eig, A, B, a, b, c)``. The leading seven are the
    reference solve, identical to ``klein_solve1``'s. ``a``/``b`` are
    ``(n_regime, n_var, n_var)`` and ``c`` is ``(n_regime, n_var)``, the pencils
    ``a^r E[y_{t+1}] = b^r y_t - c^r`` that ``occbin_sim`` steps through.
    ``stab`` is the reference regime's and is reported, never raised on: a
    binding regime alone is routinely indeterminate and that is not an error.
    """
    cdef double[::1] seedv = np.ascontiguousarray(seed, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)
    cdef signed char[::1] incv = np.ascontiguousarray(incidence, dtype=np.int8)
    cdef int64_t n_var = seedv.shape[0]
    cdef int64_t n_par = parv.shape[0]
    cdef int64_t n_ctrl = n_var - n_state
    cdef int64_t nd = sdsge_pencil_dim(&incv[0], n_var)

    if n_state < 1:
        raise ValueError("occbin_solve1 requires n_states >= 1.")
    if n_ctrl < 0:
        raise ValueError("n_states exceeds the matrix dimension.")
    # The kernel's table is MAX_REGIME slots on the stack, so an over-wide
    # model would write past it rather than raise.
    if not 0 < n_constraint <= 2:
        raise ValueError(
            "A model may declare one or two constraints under "
            f"equations.constraint, got {n_constraint}."
        )

    cdef int64_t n_regime = 1 << n_constraint
    if len(pencil_addrs) != n_regime or len(rows) != n_regime:
        raise ValueError(
            f"pencil_addrs and rows must hold {n_regime} entries to cover "
            f"every mask over {n_constraint} constraints, got "
            f"{len(pencil_addrs)} and {len(rows)}."
        )
    if pencil_addrs[0] or len(rows[0]):
        raise ValueError(
            "Slot 0 is the reference regime and swaps no rows; pass address 0 "
            "and an empty rows array."
        )

    cdef sdsge_regime_pencil_fn pencils[4]  # MAX_REGIME
    cdef const int64_t *row_ptrs[4]
    cdef int64_t n_rows[4]
    cdef const int64_t[::1] rowv
    cdef int64_t max_n_row = 0
    cdef int64_t m, i
    keep = []
    for m in range(n_regime):
        pencils[m] = <sdsge_regime_pencil_fn><void*><size_t>pencil_addrs[m]
        row_arr = np.ascontiguousarray(rows[m], dtype=np.int64)
        keep.append(row_arr)
        rowv = row_arr
        n_rows[m] = rowv.shape[0]
        # The kernel scatters straight into a[row], so an out-of-range row
        # would write past the output rather than raise.
        for i in range(n_rows[m]):
            if not 0 <= rowv[i] < n_var:
                raise ValueError(
                    f"rows[{m}][{i}] is {rowv[i]}, outside 0..{n_var - 1}."
                )
        row_ptrs[m] = &rowv[0] if n_rows[m] > 0 else NULL
        if n_rows[m] > max_n_row:
            max_n_row = n_rows[m]

    ss = np.empty(n_var, dtype=np.float64)
    order = np.empty(n_var, dtype=np.int64)
    a_ref = np.empty((n_var, n_var), dtype=np.float64)
    b_ref = np.empty((n_var, n_var), dtype=np.float64)
    c_ref = np.empty((n_var, n_var), dtype=np.float64)
    d_ref = np.empty((n_var, n_exog), dtype=np.float64)
    s = np.empty((nd, nd), dtype=np.complex128)
    t = np.empty((nd, nd), dtype=np.complex128)
    z = np.empty((nd, nd), dtype=np.complex128)
    f = np.empty((n_ctrl, n_state), dtype=np.float64)
    p = np.empty((n_state, n_state), dtype=np.float64)
    eig = np.empty(nd, dtype=np.complex128)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    a = np.empty((n_regime, n_var, n_var), dtype=np.float64)
    b = np.empty((n_regime, n_var, n_var), dtype=np.float64)
    c = np.empty((n_regime, n_var, n_var), dtype=np.float64)
    d = np.empty((n_regime, n_var, n_exog), dtype=np.float64)
    cst = np.empty((n_regime, n_var), dtype=np.float64)

    cdef double[::1] ssv = ss
    cdef int64_t[::1] orderv = order
    cdef double[:, ::1] arefv = a_ref
    cdef double[:, ::1] brefv = b_ref
    cdef double[:, ::1] crefv = c_ref
    cdef double[:, ::1] drefv = d_ref
    cdef double complex[:, ::1] sv = s
    cdef double complex[:, ::1] tv = t
    cdef double complex[:, ::1] zv = z
    cdef double[:, ::1] fv = f
    cdef double[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef double[:, :, ::1] av = a
    cdef double[:, :, ::1] bv = b
    cdef double[:, :, ::1] cv = c
    cdef double[:, :, ::1] dv = d
    cdef double[:, ::1] cstv = cst

    cdef occbin_solve1_spec spec
    spec.first.residual = <sdsge_residual_fn><void*>residual_addr
    spec.first.zgges = _zgges
    spec.first.dgeqrf = _dgeqrf
    spec.first.dormqr = _dormqr
    spec.first.ss_seed = &seedv[0]
    spec.first.params = &parv[0] if n_par > 0 else NULL
    spec.first.incidence = &incv[0]
    spec.first.n_var = n_var
    spec.first.n_state = n_state
    spec.first.n_ctrl = n_ctrl
    spec.first.n_exog = n_exog
    spec.first.n_par = n_par
    spec.pencil = &pencils[0]
    spec.rows = &row_ptrs[0]
    spec.n_row = &n_rows[0]
    spec.n_constraint = n_constraint

    cdef sdsge_occbin1 out
    out.ref.ss = &ssv[0]
    out.ref.a_real = &arefv[0, 0]
    out.ref.b_real = &brefv[0, 0]
    out.ref.c_real = &crefv[0, 0]
    out.ref.d_real = &drefv[0, 0] if n_exog > 0 else NULL
    out.ref.s = <c128 *>&sv[0, 0]
    out.ref.t = <c128 *>&tv[0, 0]
    out.ref.z = <c128 *>&zv[0, 0]
    out.ref.f = &fv[0, 0] if n_ctrl > 0 else NULL
    out.ref.p = &pv[0, 0]
    out.ref.eig = <c128 *>&eigv[0]
    out.ref.stab = 0
    out.ref.A = &Av[0, 0]
    out.ref.B = &Bv[0, 0] if n_exog > 0 else NULL
    out.ref.order = &orderv[0]
    out.a = &av[0, 0, 0]
    out.b = &bv[0, 0, 0]
    out.c = &cv[0, 0, 0]
    out.d = &dv[0, 0, 0] if n_exog > 0 else NULL
    out.cst = &cstv[0, 0]

    cdef int64_t err
    cdef arena_size sz = sdsge_occbin_solve1_arena_size(
        n_var, n_state, n_ctrl, n_par, n_exog, nd, max_n_row
    )
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_occbin_solve1(&spec, &out, &arv[0], &iarv[0])

    _raise_klein_error(err)
    return ss, f, p, int(out.ref.stab), eig, a, b, c, d, cst
