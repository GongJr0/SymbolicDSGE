"""Fused ``sgu_klein_solve2`` against the staged shims it replaces.

The fused driver runs the same four kernels on the same buffers in the same
order; the staging it removes was copies, not arithmetic. So parity here is
exact rather than approximate, for the reason it is exact in
``test_klein_solve1``: a tolerance would hide a real divergence.

What the staged path materialized in Python and handed straight back was the
residual Hessian, ``32 * n_var**3`` bytes that no Python caller ever read.
Asserting on ``gxx``/``hxx``/``gss``/``hss`` is what licenses keeping it native.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import (
    bicomplex_hessian,
    klein_preprocess,
    klein_solve1,
    second_order,
    second_order_risk,
    sgu_klein_solve2,
)
from SymbolicDSGE.core import DSGESolver, ModelParser

# The three (n_var, n_state, n_ctrl, n_exog) shapes test_klein_solve1 covers.
MODELS = [
    "tests/fixtures/models/rbc_second_order.yaml",
    "MODELS/test.yaml",
    "MODELS/POST82.yaml",
]

OUTPUTS = ("ss", "f", "p", "stab", "eig", "gxx", "hxx", "gss", "hss", "A", "B")


def _model(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    eta = DSGESolver._build_eta(compiled)
    return compiled, par, seed, eta


def _staged(compiled, par, seed, eta):
    """The call sequence ``_solve_second_order`` ran before it was fused."""
    addr = compiled.construct_objective_cfunc().address
    bc_addr = compiled.construct_objective_cfunc_bicomplex().address
    n_eq = len(compiled.var_names)
    n_state = compiled.n_state

    ss, f, p, stab, eig, A, B = klein_solve1(addr, seed, par, n_state, compiled.n_exog)
    # The second-order stages take the pencil, which the fused solve keeps
    # internal, so the staged reference rebuilds it at the same steady state.
    a, b = klein_preprocess(addr, ss, par, n_eq)
    gx, hx = f, p  # klein_solve1 already projects
    f_xx = bicomplex_hessian(bc_addr, ss, par, n_eq)
    gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)
    gss, hss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)
    return ss, f, p, stab, eig, gxx, hxx, gss, hss, A, B


def _assert_same(got, want):
    for name, g, w in zip(OUTPUTS, got, want):
        if name == "stab":
            assert g == w
        else:
            np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("path", MODELS)
def test_matches_the_staged_shims_exactly(path):
    compiled, par, seed, eta = _model(path)

    got = sgu_klein_solve2(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        seed,
        par,
        compiled.n_state,
        eta,
        compiled.n_exog,
    )

    _assert_same(got, _staged(compiled, par, seed, eta))


@pytest.mark.parametrize("path", MODELS)
def test_python_wrapper_carries_the_native_outputs(path):
    """``sgu_solve`` must hand back the tensors and state space it was given."""
    from SymbolicDSGE.core.solver_backend import sgu_solve

    compiled, par, seed, eta = _model(path)
    pert = sgu_solve(
        compiled.construct_objective_cfunc(),
        compiled.construct_objective_cfunc_bicomplex(),
        par,
        seed,
        compiled.n_state,
        eta,
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
        pert.gss,
        pert.hss,
        pert.A,
        pert.B,
    )
    _assert_same(got, _staged(compiled, par, seed, eta))


@pytest.mark.parametrize("path", MODELS)
def test_first_order_block_matches_the_first_order_solve(path):
    """The second-order solve must not perturb the first order it is built on."""
    compiled, par, seed, eta = _model(path)

    ss1, f1, p1, stab1, eig1, A1, B1 = klein_solve1(
        compiled.construct_objective_cfunc().address,
        seed,
        par,
        compiled.n_state,
        compiled.n_exog,
    )
    ss2, f2, p2, stab2, eig2, _, _, _, _, A2, B2 = sgu_klein_solve2(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        seed,
        par,
        compiled.n_state,
        eta,
        compiled.n_exog,
    )

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


def test_rejects_an_eta_the_risk_correction_cannot_read():
    compiled, par, seed, eta = _model("MODELS/POST82.yaml")

    with pytest.raises(ValueError, match="eta has shape"):
        sgu_klein_solve2(
            compiled.construct_objective_cfunc().address,
            compiled.construct_objective_cfunc_bicomplex().address,
            seed,
            par,
            compiled.n_state,
            eta[:-1],
            compiled.n_exog,
        )
