"""Fused ``sgu_klein_solve2`` against the staged shims it replaces.

The fused driver runs the same three kernels on the same buffers in the same
order; the staging it removes was copies, not arithmetic. So parity here is
exact rather than approximate, for the reason it is exact in
``test_klein_solve1``: a tolerance would hide a real divergence.

What the staged path materialized in Python and handed straight back was the
residual Hessian, which no Python caller ever read. Asserting on the second-order
blocks is what licenses keeping it native.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import (
    bicomplex_hessian,
    klein_preprocess,
    klein_solve1,
    second_order,
    sgu_klein_solve2,
)
from SymbolicDSGE.core import DSGESolver, ModelParser

# The three (n_var, n_state, n_ctrl, n_exog) shapes test_klein_solve1 covers.
MODELS = [
    "tests/fixtures/models/rbc_second_order.yaml",
    "MODELS/test.yaml",
    "MODELS/POST82.yaml",
]

OUTPUTS = (
    "ss",
    "f",
    "p",
    "stab",
    "eig",
    "gxx",
    "hxx",
    "gxu",
    "hxu",
    "guu",
    "huu",
    "gss",
    "hss",
    "A",
    "B",
)


def _model(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    Q = DSGESolver._build_Q(compiled)
    return compiled, par, seed, Q


def _staged(compiled, par, seed, Q):
    """The call sequence ``_solve_second_order`` ran before it was fused."""
    addr = compiled.construct_objective_cfunc().address
    bc_addr = compiled.construct_objective_cfunc_bicomplex().address
    n_eq = len(compiled.var_names)
    n_state = compiled.n_state

    ss, f, p, stab, eig, A, B = klein_solve1(
        addr, seed, par, compiled.incidence, n_state, compiled.n_exog
    )
    # The second-order stage takes the pencil, which the fused solve keeps
    # internal, so the staged reference rebuilds it at the same steady state.
    a, b, _, _ = klein_preprocess(addr, ss, par, n_eq, compiled.n_exog)
    f_xx = bicomplex_hessian(bc_addr, ss, par, compiled.n_exog, n_eq)
    blocks = second_order(a, b, f_xx, f, p, B, Q, n_state)
    return (ss, f, p, stab, eig, *blocks, A, B)


def _assert_same(got, want):
    for name, g, w in zip(OUTPUTS, got, want):
        if name == "stab":
            assert g == w
        else:
            np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("path", MODELS)
def test_matches_the_staged_shims_exactly(path):
    compiled, par, seed, Q = _model(path)

    got = sgu_klein_solve2(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        seed,
        par,
        Q,
        compiled.incidence,
        compiled.n_state,
        compiled.n_exog,
    )

    _assert_same(got, _staged(compiled, par, seed, Q))


@pytest.mark.parametrize("path", MODELS)
def test_python_wrapper_carries_the_native_outputs(path):
    """``sgu_solve`` must hand back the tensors and state space it was given."""
    from SymbolicDSGE.core.solver_backend import sgu_solve

    compiled, par, seed, Q = _model(path)
    pert = sgu_solve(
        compiled.construct_objective_cfunc(),
        compiled.construct_objective_cfunc_bicomplex(),
        par,
        seed,
        Q,
        compiled.incidence,
        compiled.n_state,
        n_exog=compiled.n_exog,
    )

    assert pert.order == 2
    got = (
        pert.steady_state,
        pert.f,
        pert.p,
        pert.stab,
        pert.eig,
        pert.gxx,
        pert.hxx,
        pert.gxu,
        pert.hxu,
        pert.guu,
        pert.huu,
        pert.gss,
        pert.hss,
        pert.A,
        pert.B,
    )
    _assert_same(got, _staged(compiled, par, seed, Q))


@pytest.mark.parametrize("path", MODELS)
def test_first_order_block_matches_the_first_order_solve(path):
    """The second-order solve must not perturb the first order it is built on."""
    compiled, par, seed, Q = _model(path)

    ss1, f1, p1, stab1, eig1, A1, B1 = klein_solve1(
        compiled.construct_objective_cfunc().address,
        seed,
        par,
        compiled.incidence,
        compiled.n_state,
        compiled.n_exog,
    )
    second = sgu_klein_solve2(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        seed,
        par,
        Q,
        compiled.incidence,
        compiled.n_state,
        compiled.n_exog,
    )
    ss2, f2, p2, stab2, eig2 = second[:5]
    A2, B2 = second[-2:]

    assert stab2 == stab1
    for name, g, w in (
        ("ss", ss2, ss1),
        ("f", f2, f1),
        ("p", p2, p1),
        ("eig", eig2, eig1),
        ("A", A2, A1),
        ("B", B2, B1),
    ):
        np.testing.assert_array_equal(g, w, err_msg=name)


def test_rejects_a_covariance_of_the_wrong_shape():
    """Q is read at (n_exog, n_exog); a mismatch would walk off its buffer. A
    singular Q is not rejected: the risk correction integrates against the
    covariance rather than factoring it, so a zero Q simply means no risk."""
    compiled, par, seed, Q = _model("MODELS/POST82.yaml")

    with pytest.raises(ValueError, match=r"shape \(n_exog, n_exog\)"):
        sgu_klein_solve2(
            compiled.construct_objective_cfunc().address,
            compiled.construct_objective_cfunc_bicomplex().address,
            seed,
            par,
            np.eye(compiled.n_exog + 1),
            compiled.incidence,
            compiled.n_state,
            compiled.n_exog,
        )
