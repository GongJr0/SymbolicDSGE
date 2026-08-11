"""Type stubs for the compiled ``_occbin`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_occbin.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_occbin.pyx`` / ``occbin.c``; the tests guard the
runtime behavior, not this stub.
"""

from numpy import float64, int8, int64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_I8 = NDArray[int8]
_I64 = NDArray[int64]

MAX_CONSTRAINTS: int

def constraint_path(
    cond_addr: int,
    path: _F64,
    par: _F64,
    regime_in: _I8,
    n_constraint: int,
    inclusive: int,
    out: _I8 | None = ...,
) -> tuple[_I8, int, float]:
    """(regime_out, changed, max_err) <- latched mask over a (T, n_var) path.

    ``out`` may alias ``regime_in`` to latch in place; ``changed`` counts the
    periods whose mask moved and ``max_err`` is the largest distance that moved
    one. ``inclusive`` is ``ConstraintFunc.inclusive``, which decides a distance
    of exactly zero.
    """

def regime_pencil(
    pencil_addr: int,
    rows: _I64,
    ss: _F64,
    par: _F64,
    a_ref: _F64,
    b_ref: _F64,
) -> tuple[_F64, _F64, _F64]:
    """(a, b, c) <- the reference pencil with ``rows`` patched by one regime.

    ``pencil_addr`` of 0 is the reference regime: the copy alone, ``c`` zero.
    ``c`` is ``(n_var,)`` and zero off ``rows``.
    """

def occbin_recursion_arena_size(
    n_var: int,
    n_state: int,
    n_ctrl: int,
) -> tuple[int, int]:
    """(n_float, n_int) scratch ``occbin_recursion`` needs for a shape."""

def occbin_recursion(
    a: _F64,
    b: _F64,
    c: _F64,
    mask: _I8,
    f_ref: _F64,
    out: _F64 | None = ...,
    arena: _F64 | None = ...,
    iarena: _I64 | None = ...,
) -> _F64:
    """(T, n_var, n_state + 1) rules <- pencils stacked by bitmask and a guess.

    Block ``t`` is the affine map from ``x_t`` to ``[x_{t+1}; u_t]``, state rows
    first, with the constant in the last column. ``f_ref`` is ``(n_ctrl,
    n_state)`` and closes the recursion past the last date.
    """

def occbin_forward(rule: _F64, x0: _F64, out: _F64 | None = ...) -> _F64:
    """(T, n_var) path in deviations <- a rule stack and an initial state.

    Row ``t`` is ``[x_t; u_t]``, the state half from date ``t - 1``'s block and
    the control half from date ``t``'s.
    """

def occbin_solve_arena_size(
    n_var: int,
    n_state: int,
    n_ctrl: int,
    T_cap: int,
    max_iter: int,
) -> tuple[int, int]:
    """(n_float, n_int) scratch ``occbin_solve`` needs for a shape."""

def occbin_solve(
    a: _F64,
    b: _F64,
    c: _F64,
    f_ref: _F64,
    ss: _F64,
    par: _F64,
    cond_addr: int,
    n_constraint: int,
    inclusive: int,
    shocks: _F64,
    x_init: _F64,
    *,
    T0: int,
    T_cap: int = ...,
    n_periods: int | None = ...,
    max_iter: int = ...,
    init_regime: _I8 | None = ...,
    periodic_solution: bool = ...,
    periodic_threshold: int = ...,
    periodic_strict: bool = ...,
    curb_retrench: bool = ...,
    reset_regime: bool = ...,
    reset_check_ahead: bool = ...,
    algo_truncation: int = ...,
) -> tuple[_F64, _I8, dict[str, _F64 | _I8 | _I64]]:
    """(out, regimes, diag) <- pencils, a constraint and a shock sequence.

    ``out`` is the ``(n_periods, n_var)`` piecewise path in deviations,
    ``regimes`` the ``(S, T_cap)`` accepted guess per shock period, and ``diag``
    holds ``T_used``, ``iters``, ``max_err`` and ``periodic``, one entry per
    period. The pencil stack must cover every mask over ``n_constraint``
    constraints. The keywords past ``max_iter`` are Dynare's ``occbin.simul``
    options at their own defaults.
    """
