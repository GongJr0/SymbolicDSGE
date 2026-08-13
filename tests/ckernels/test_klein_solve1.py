"""What the fused ``klein_solve1`` guarantees about its own outputs.

The staged comparison this module used to make is gone with the two-date pencil.
A first-order solve is no longer a sequence of exposed shims a test can replay:
the static rotation, the pencil assembly, the ``nspred`` split and the shock
solve all live inside ``sdsge_klein_from_pencil``, and none of them is reachable
from Python on its own. What the solve produces is checked against Dynare in
``tests/core/test_dynare_post82_parity``; what is checked here is the contract
the driver keeps regardless of the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import klein_solve1
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import klein_solve

# One model with an empty control block, one with a nonempty one, and one whose
# shock block is wider than a single column.
MODELS = [
    "tests/fixtures/models/rbc_second_order.yaml",
    "MODELS/test.yaml",
    "MODELS/POST82.yaml",
]


def _model(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    return compiled, par, seed


@pytest.mark.parametrize("path", MODELS)
def test_policy_is_real(path):
    """``f``/``p`` leave the solve projected, so no caller collapses them again."""
    compiled, par, seed = _model(path)
    cfunc = compiled.construct_objective_cfunc()

    _, f, p, _, _, _, _ = klein_solve1(
        cfunc.address, seed, par, compiled.incidence, compiled.n_state, compiled.n_exog
    )
    assert f.dtype == np.float64
    assert p.dtype == np.float64

    sol = klein_solve(
        cfunc, par, seed, compiled.incidence, compiled.n_state, n_exog=compiled.n_exog
    )
    assert sol.f.dtype == np.float64
    assert sol.p.dtype == np.float64
    assert sol.eig.dtype == np.complex128


@pytest.mark.parametrize("path", MODELS)
def test_the_transition_reads_only_the_state_columns(path):
    """A control at ``t`` is pinned by the state at ``t-1``, so its own column
    contributes nothing and ``A`` is the rule scattered rather than a product."""
    compiled, par, seed = _model(path)
    sol = klein_solve(
        compiled.construct_objective_cfunc(),
        par,
        seed,
        compiled.incidence,
        compiled.n_state,
        n_exog=compiled.n_exog,
    )
    n_state = compiled.n_state

    assert np.abs(sol.A[:, n_state:]).max() == 0.0
    np.testing.assert_array_equal(sol.A[:n_state, :n_state], sol.p)
    np.testing.assert_array_equal(sol.A[n_state:, :n_state], sol.f)


def test_reports_stab_instead_of_raising():
    """A stability verdict is data here; only the caller decides it is fatal."""
    compiled, par, seed = _model("MODELS/POST82.yaml")

    stab = klein_solve1(
        compiled.construct_objective_cfunc().address,
        seed,
        par,
        compiled.incidence,
        compiled.n_state,
        compiled.n_exog,
    )[3]

    assert stab == 0


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        (lambda c: (0, 0), "n_states >= 1"),
        (lambda c: (len(c.var_names) + 1, 0), "exceeds the matrix dimension"),
    ],
)
def test_rejects_dimensions_the_solve_cannot_hold(dims, match):
    compiled, par, seed = _model("MODELS/POST82.yaml")
    n_state, n_exog = dims(compiled)

    with pytest.raises(ValueError, match=match):
        klein_solve1(
            compiled.construct_objective_cfunc().address,
            seed,
            par,
            compiled.incidence,
            n_state,
            n_exog,
        )
