# type: ignore
"""Native ``regime_pencil``: the reference pencil with one regime's rows patched.

The fixture is a levels RBC rather than a gap model on purpose. A regime's
constant is its residual at the *reference* steady state, so at ss = 0 it can
read as right while being zero for the wrong reason. Here the expansion point is
nonzero and the constant is the delta * k_ss the shut-off investment rule leaves
behind, which is what the piecewise solve actually consumes.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core import (
    klein_preprocess,
    residual_eval,
    steady_state_newton,
)
from SymbolicDSGE._ckernels.occbin import regime_pencil
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)

LOW = 0b1


@pytest.fixture(scope="module")
def compiled(rbc_second_order_test_model_path):
    """Levels RBC where a bad-TFP regime shuts investment off."""
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    conf = copy.deepcopy(model)
    _, k, z = conf.variables.variables
    delta = sp.Symbol("delta")

    conf.equations.constraint = {"low": Constraint(bind=z(t) < 0, relax=z(t) >= 0)}
    conf.equations.regime = {
        frozenset({"low"}): {"euler": sp.Eq(k(t + 1), (1 - delta) * k(t))}
    }
    return DSGESolver(conf, kalman).compile()


@pytest.fixture(scope="module")
def par(compiled):
    calib = compiled.config.calibration.parameters
    return np.array([float(calib[p]) for p in compiled.calib_params])


@pytest.fixture(scope="module")
def ss_ref(compiled, par):
    """Reference steady state, which every regime linearizes around.

    A lag aux starts where its origin does, so the suffix is stripped before the
    `<name>_ss` lookup; an unseeded variable starts at zero.
    """
    calib = compiled.config.calibration.parameters
    seed = []
    for name in compiled.var_names:
        origin = re.sub(r"_lag\d+$", "", name)
        sym = sp.Symbol(f"{origin}_ss")
        seed.append(float(calib[sym]) if sym in calib else 0.0)

    ss, _ = steady_state_newton(
        compiled.construct_objective_cfunc().address,
        np.array(seed),
        par,
        compiled.n_exog,
    )
    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(ss).max() > 1.0
    return ss


@pytest.fixture(scope="module")
def reference(compiled, par, ss_ref):
    n_var = len(compiled.var_names)
    return klein_preprocess(
        compiled.construct_objective_cfunc().address,
        ss_ref,
        par,
        n_var,
        compiled.n_exog,
    )


@pytest.fixture(scope="module")
def patched(compiled, par, ss_ref, reference):
    func = compiled.construct_regime_pencil_func()
    return regime_pencil(func.address(LOW), func.rows[LOW], ss_ref, par, *reference)


def _rows(compiled):
    return compiled.construct_regime_pencil_func().rows[LOW]


def test_patched_rows_match_a_full_sweep_of_the_regime(compiled, par, ss_ref, patched):
    # The patch is only worth doing if it lands where sweeping the whole regime
    # residual would have.
    n_var = len(compiled.var_names)
    rows = _rows(compiled)
    a, b, c, d, _ = patched

    regime_cfunc = compiled.construct_regime_cfuncs()[LOW]
    a_r, b_r, c_r, d_r = klein_preprocess(
        regime_cfunc.address, ss_ref, par, n_var, compiled.n_exog
    )

    np.testing.assert_allclose(a[rows], a_r[rows])
    np.testing.assert_allclose(b[rows], b_r[rows])
    np.testing.assert_allclose(c[rows], c_r[rows])
    np.testing.assert_allclose(d[rows], d_r[rows])


def test_unreplaced_rows_are_copied_verbatim(compiled, reference, patched):
    # memcpy, not arithmetic: anything but bit equality means a row moved.
    n_var = len(compiled.var_names)
    other = np.setdiff1d(np.arange(n_var), _rows(compiled))
    a_ref, b_ref, c_ref, d_ref = reference
    a, b, c, d, _ = patched

    assert other.size > 0
    np.testing.assert_array_equal(a[other], a_ref[other])
    np.testing.assert_array_equal(b[other], b_ref[other])
    np.testing.assert_array_equal(c[other], c_ref[other])
    np.testing.assert_array_equal(d[other], d_ref[other])


def test_constants_are_the_regime_residual_at_the_reference(
    compiled, par, ss_ref, patched
):
    n_var = len(compiled.var_names)
    rows = _rows(compiled)
    *_, cst = patched

    regime_cfunc = compiled.construct_regime_cfuncs()[LOW]
    c_r = residual_eval(
        regime_cfunc.address,
        ss_ref,
        ss_ref,
        ss_ref,
        np.zeros(compiled.n_exog),
        par,
        n_var,
    ).real

    want = np.zeros(n_var)
    want[rows] = c_r[rows]
    np.testing.assert_allclose(cst, want, atol=1e-10)

    # In levels the constant is delta * k_ss, and it is the whole mechanism: a
    # zero here would pass the comparison above while solving the wrong model.
    assert np.abs(cst[rows]).min() > 0.5
    assert np.count_nonzero(cst) == rows.size


def test_reference_regime_is_a_verbatim_copy(compiled, par, ss_ref, reference):
    # A null pencil is how the mask-0 slot of the table gets filled, so the
    # backward recursion can index it without a branch.
    n_var = len(compiled.var_names)
    a_ref, b_ref, c_ref, d_ref = reference
    a, b, c, d, cst = regime_pencil(
        0, np.empty(0, dtype=np.int64), ss_ref, par, *reference
    )

    np.testing.assert_array_equal(a, a_ref)
    np.testing.assert_array_equal(b, b_ref)
    np.testing.assert_array_equal(c, c_ref)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(cst, np.zeros(n_var))


def test_rows_outside_the_pencil_are_rejected(compiled, par, ss_ref, reference):
    # The kernel scatters straight into a[row]; an unchecked index writes past
    # the output instead of raising.
    n_var = len(compiled.var_names)
    func = compiled.construct_regime_pencil_func()

    with pytest.raises(ValueError, match=rf"outside 0\.\.{n_var - 1}"):
        regime_pencil(
            func.address(LOW),
            np.array([n_var], dtype=np.int64),
            ss_ref,
            par,
            *reference,
        )
