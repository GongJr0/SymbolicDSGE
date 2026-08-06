"""Fused ``klein_solve1`` against the staged shims it replaces.

The staged path bridged two layout conventions by copying: ``klein_qz`` emits
column-major, ``klein_postprocess`` reads row-major, and the
``ascontiguousarray`` between them is that transpose. The fused driver transposes
in place instead. Same permutation of the same widening, so parity here is exact
rather than approximate; a tolerance would hide a real divergence.

The ``a``/``b`` comparison is also what licenses ``_solve_second_order`` to read
the pencil off the solution instead of rebuilding it with a second
``klein_preprocess``.

Parity runs in levels only. The log-linear wrap expands at ``log(ss)``, which no
shipped fixture survives; ``test_klein_preproc.py`` covers that branch on a
hand-built residual with a positive steady state.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import (
    assemble_state_space,
    klein_postprocess,
    klein_preprocess,
    klein_qz,
    klein_solve1,
    steady_state_newton,
)
from SymbolicDSGE.core import DSGESolver, ModelParser

# Three (n_var, n_state, n_ctrl, n_exog) shapes, including a model with a
# nonempty control block and one whose B block is wider than one column.
MODELS = [
    "tests/fixtures/models/rbc_second_order.yaml",
    "MODELS/test.yaml",
    "MODELS/POST82.yaml",
]

OUTPUTS = ("ss", "a", "b", "f", "p", "stab", "eig", "A", "B")


def _model(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    return compiled, par, seed


def _staged(compiled, par, seed, log_linear):
    """The call sequence ``solver_backend.klein_solve`` ran before it was fused."""
    addr = compiled.construct_objective_cfunc().address
    n_eq = len(compiled.var_names)
    ss, _ = steady_state_newton(addr, seed, par)
    a, b = klein_preprocess(addr, ss, par, n_eq, log_linear)
    s, t, z = klein_qz(a, b)
    f, p, stab, eig = klein_postprocess(s, t, z, compiled.n_state)
    A, B = assemble_state_space(
        p, f, compiled.n_state, n_eq - compiled.n_state, compiled.n_exog
    )
    return ss, a, b, f, p, stab, eig, A, B


def _assert_same(got, want):
    for name, g, w in zip(OUTPUTS, got, want):
        if name == "stab":
            assert g == w
        else:
            np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("path", MODELS)
def test_matches_the_staged_shims_exactly(path):
    compiled, par, seed = _model(path)

    got = klein_solve1(
        compiled.construct_objective_cfunc().address,
        seed,
        par,
        compiled.n_state,
        compiled.n_exog,
        False,
    )

    _assert_same(got, _staged(compiled, par, seed, False))


@pytest.mark.parametrize("path", MODELS)
def test_log_linear_flag_reaches_the_kernel(path):
    """Pinned by failure mode, because no shipped fixture can solve in log space.

    The log-linear wrap expands at ``log(ss)``, and every fixture here has at
    least one zero steady-state entry (the deviation-form models are all zero),
    which collapses those columns of the pencil. Identical failures still pin the
    flag: were it dropped on the way into the spec struct, the fused call would
    solve in levels where the staged path cannot solve in logs.
    """
    compiled, par, seed = _model(path)

    with pytest.raises((ValueError, RuntimeError)) as fused:
        klein_solve1(
            compiled.construct_objective_cfunc().address,
            seed,
            par,
            compiled.n_state,
            compiled.n_exog,
            True,
        )
    with pytest.raises((ValueError, RuntimeError)) as staged:
        _staged(compiled, par, seed, True)

    assert type(fused.value) is type(staged.value)
    assert str(fused.value) == str(staged.value)


@pytest.mark.parametrize("path", MODELS)
def test_python_wrapper_carries_the_native_outputs(path):
    """``klein_solve`` must hand back the pencil and state space it was given."""
    from SymbolicDSGE.core.solver_backend import klein_solve

    compiled, par, seed = _model(path)
    sol = klein_solve(
        compiled.construct_objective_cfunc(),
        par,
        seed,
        compiled.n_state,
        n_exog=compiled.n_exog,
    )

    got = (
        sol.steady_state,
        sol.a,
        sol.b,
        sol.f,
        sol.p,
        sol.stab,
        sol.eig,
        sol.A,
        sol.B,
    )
    _assert_same(got, _staged(compiled, par, seed, False))


def test_reports_stab_instead_of_raising():
    """A stability verdict is data here; only the caller decides it is fatal."""
    compiled, par, seed = _model("MODELS/POST82.yaml")

    stab = klein_solve1(
        compiled.construct_objective_cfunc().address,
        seed,
        par,
        compiled.n_state,
        compiled.n_exog,
    )[5]

    assert stab == 0


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        (lambda c: (0, 0), "n_states >= 1"),
        (lambda c: (len(c.var_names) + 1, 0), "exceeds the matrix dimension"),
        (lambda c: (c.n_state, c.n_state + 1), "cannot exceed n_state"),
    ],
)
def test_rejects_dimensions_the_solve_cannot_hold(dims, match):
    compiled, par, seed = _model("MODELS/POST82.yaml")
    n_state, n_exog = dims(compiled)

    with pytest.raises(ValueError, match=match):
        klein_solve1(
            compiled.construct_objective_cfunc().address, seed, par, n_state, n_exog
        )
