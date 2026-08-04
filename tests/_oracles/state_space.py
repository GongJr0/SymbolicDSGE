"""Residuals of a model's own equations, evaluated on a path from its state space.

Innovations are substituted to zero before the solve, and the state space
assembles ``B`` instead of solving it, so no existing check ties ``(A, B)`` back
to the equations the user wrote. This evaluates those equations directly,
reading offsets off the parsed config rather than the compiled residual.

An equation carrying an innovation is an ex-post identity: it defines the shock
target's own path, so every term is a realisation and the innovation lands on
the index of the target's leading appearance. An equation carrying none is a
first-order condition: its offset-one terms are ``E_t``, which the solution puts
at ``A y_t``. Feeding a realisation into those instead leaves ``a B eps`` behind
and reports a violation where there is none.

Within this framework the split is exactly the syntactic one. ``shock_map``
requires every innovation to target a declared variable, and that variable
becomes an exogenous state, so an innovation only ever appears in the equation
defining its own target.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]


def _offset(call: sp.Expr, t: sp.Symbol) -> int:
    return int(sp.simplify(call.args[0] - t))


def simulate(A: NDF, B: NDF, shocks: NDF) -> NDF:
    """Deviation path ``y_k = A y_{k-1} + B eps_k``, starting from ``y_0 = 0``.

    ``shocks[k]`` is the innovation realised at index ``k``; ``shocks[0]`` is
    unused, so the path leaves the steady state only from index 1 onward.
    """
    path = np.zeros((shocks.shape[0], A.shape[0]), dtype=float64)
    for k in range(1, shocks.shape[0]):
        path[k] = A @ path[k - 1] + B @ shocks[k]
    return path


def structural_residuals(
    compiled: Any,
    t: sp.Symbol,
    A: NDF,
    path: NDF,
    shocks: NDF,
    index: int,
    ss: NDF | None = None,
) -> dict[str, float]:
    """Every model equation evaluated at ``index``, keyed by equation name.

    ``path`` holds deviations from steady state and ``ss`` is added back before
    evaluation, so a levels model is evaluated where it was written. Exactly
    zero for a linear model, order ``deviation ** 2`` otherwise.
    """
    conf = compiled.config
    zero = np.zeros(path.shape[1]) if ss is None else ss
    levels = path + zero
    expected = zero + A @ path[index]

    params: Mapping[sp.Symbol, float] = {
        k: float(v) for k, v in conf.calibration.parameters.items()
    }
    rev = {str(v): k for k, v in conf.shock_map.items()}
    shock_col = {
        rev[name]: j for j, name in enumerate(compiled.var_names[: compiled.n_exog])
    }
    targets = {sym: name for name, sym in rev.items()}

    out: dict[str, float] = {}
    for name, eq in conf.equations.model.items():
        calls = eq.atoms(sp.core.function.AppliedUndef)
        present = [s for s in shock_col if s in eq.free_symbols]
        ex_post = bool(present)

        subs: dict[sp.Expr, float] = {}
        for call in calls:
            var, k = call.func.__name__, _offset(call, t)
            col = compiled.idx[var]
            if k == 1 and not ex_post:
                subs[call] = float(expected[col])
            else:
                subs[call] = float(levels[index + k, col])

        for sym in present:
            lead = max(
                (_offset(c, t) for c in calls if c.func.__name__ == targets[sym]),
                default=0,
            )
            subs[sym] = float(shocks[index + lead, shock_col[sym]])

        out[name] = float((eq.lhs - eq.rhs).subs(subs).subs(params))
    return out


def worst_residual(
    compiled: Any,
    t: sp.Symbol,
    A: NDF,
    path: NDF,
    shocks: NDF,
    ss: NDF | None = None,
) -> dict[str, float]:
    """``structural_residuals`` maximised over every index with a full window.

    Skips index 0, which has no lag, and the last index, whose lead is missing.
    """
    worst: dict[str, float] = {}
    for index in range(1, path.shape[0] - 1):
        for name, value in structural_residuals(
            compiled, t, A, path, shocks, index, ss
        ).items():
            worst[name] = max(worst.get(name, 0.0), abs(value))
    return worst
