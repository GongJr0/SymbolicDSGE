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
    `<name>_ss` lookup; a shock state and an unseeded variable start at zero.
    """
    calib = compiled.config.calibration.parameters
    seed = []
    for name in compiled.var_names:
        origin = re.sub(r"_lag\d+$", "", name)
        sym = sp.Symbol(f"{origin}_ss")
        seed.append(float(calib[sym]) if sym in calib else 0.0)

    ss, _ = steady_state_newton(
        compiled.construct_objective_cfunc().address, np.array(seed), par
    )
    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(ss).max() > 1.0
    return ss


@pytest.fixture(scope="module")
def reference(compiled, par, ss_ref):
    n_var = len(compiled.var_names)
    return klein_preprocess(
        compiled.construct_objective_cfunc().address, ss_ref, par, n_var
    )


@pytest.fixture(scope="module")
def patched(compiled, par, ss_ref, reference):
    func = compiled.construct_regime_pencil_func()
    a_ref, b_ref = reference
    return regime_pencil(func.address(LOW), func.rows[LOW], ss_ref, par, a_ref, b_ref)


def _rows(compiled):
    return compiled.construct_regime_pencil_func().rows[LOW]


def test_patched_rows_match_a_full_sweep_of_the_regime(compiled, par, ss_ref, patched):
    # The patch is only worth doing if it lands where sweeping the whole regime
    # residual would have.
    n_var = len(compiled.var_names)
    rows = _rows(compiled)
    a, b, _ = patched

    regime_cfunc = compiled.construct_regime_cfuncs()[LOW]
    a_r, b_r = klein_preprocess(regime_cfunc.address, ss_ref, par, n_var)

    np.testing.assert_allclose(a[rows], a_r[rows])
    np.testing.assert_allclose(b[rows], b_r[rows])


def test_unreplaced_rows_are_copied_verbatim(compiled, reference, patched):
    # memcpy, not arithmetic: anything but bit equality means a row moved.
    n_var = len(compiled.var_names)
    other = np.setdiff1d(np.arange(n_var), _rows(compiled))
    a_ref, b_ref = reference
    a, b, _ = patched

    assert other.size > 0
    np.testing.assert_array_equal(a[other], a_ref[other])
    np.testing.assert_array_equal(b[other], b_ref[other])


def test_constants_are_the_regime_residual_at_the_reference(
    compiled, par, ss_ref, patched
):
    n_var = len(compiled.var_names)
    rows = _rows(compiled)
    _, _, c = patched

    regime_cfunc = compiled.construct_regime_cfuncs()[LOW]
    c_r = residual_eval(regime_cfunc.address, ss_ref, ss_ref, par, n_var).real

    want = np.zeros(n_var)
    want[rows] = c_r[rows]
    np.testing.assert_allclose(c, want, atol=1e-10)

    # In levels the constant is delta * k_ss, and it is the whole mechanism: a
    # zero here would pass the comparison above while solving the wrong model.
    assert np.abs(c[rows]).min() > 0.5
    assert np.count_nonzero(c) == rows.size


def test_reference_regime_is_a_verbatim_copy(compiled, par, ss_ref, reference):
    # A null pencil is how the mask-0 slot of the table gets filled, so the
    # backward recursion can index it without a branch.
    n_var = len(compiled.var_names)
    a_ref, b_ref = reference
    a, b, c = regime_pencil(0, np.empty(0, dtype=np.int64), ss_ref, par, a_ref, b_ref)

    np.testing.assert_array_equal(a, a_ref)
    np.testing.assert_array_equal(b, b_ref)
    np.testing.assert_array_equal(c, np.zeros(n_var))


def test_rows_outside_the_pencil_are_rejected(compiled, par, ss_ref, reference):
    # The kernel scatters straight into a[row]; an unchecked index writes past
    # the output instead of raising.
    n_var = len(compiled.var_names)
    func = compiled.construct_regime_pencil_func()
    a_ref, b_ref = reference

    with pytest.raises(ValueError, match=rf"outside 0\.\.{n_var - 1}"):
        regime_pencil(
            func.address(LOW),
            np.array([n_var], dtype=np.int64),
            ss_ref,
            par,
            a_ref,
            b_ref,
        )
