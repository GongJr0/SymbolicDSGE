"""``Linearizer`` against compiled models: methods, the point, and ``to_config``.

Everything here needs a compiled model, either for its layout or for the
residual cfunc Newton solves the expansion point with. The expansion math itself
lives in ``test_expansion.py``, and ``test_log_expansion_agrees_with_the_helper``
ties the two files together.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.core.compiled_model import CompiledModel
from SymbolicDSGE.linearization import Linearizer, linearize_model
from SymbolicDSGE.linearization.linearizer import (
    LinearizationMethod,
    _linearize_equation,
)
from SymbolicDSGE.linearization.view import PointLinearView

#: No test fixture mints an aux variable, so the one model that needs a minted
#: lag is written here. ``u(t-2)`` mints ``u_lag1``, the steady state is ``ubar``
#: and positive so LOG is available on it, and ``log_u_obs`` is the non-affine
#: observable that linearization is supposed to flatten.
AUX_MODEL = """
name: AUX_LEVELS
variables: [u]
ss_seed:
  u: ubar
shocks: [e]
equations:
  model:
    u_eq: "u(t) = (1 - rho)*ubar + rho*u(t-2) + e"
  observables:
    u_obs: "u(t)"
    log_u_obs: "log(u(t))"
calibration:
  parameters:
    rho: 0.5
    ubar: 2.0
    sig: 0.1
  shocks:
    std:
      e: sig
    corr: {}
"""

#: The deterministic steady state recorded in rbc_second_order.yaml. Newton
#: refines the rounded figures in the file, hence the loose tolerance below.
K_SS = 28.348418455829357
C_SS = 2.3066172286382751

T = sp.Symbol("t", integer=True)


@pytest.fixture(scope="module")
def rbc(rbc_second_order_test_model_path) -> CompiledModel:
    """Levels RBC: c and k have positive steady states, z sits at zero."""
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    return DSGESolver(model, kalman).compile()


@pytest.fixture(scope="module")
def gap(test_model_path) -> CompiledModel:
    """A model already authored in gaps, so Newton returns the origin."""
    model, kalman = ModelParser(test_model_path).get_all()
    return DSGESolver(model, kalman).compile()


@pytest.fixture(scope="module")
def aux() -> CompiledModel:
    model, kalman = ModelParser.from_string(AUX_MODEL).get_all()
    return DSGESolver(model, kalman).compile()


@pytest.fixture(scope="module")
def rbc_log(rbc: CompiledModel) -> PointLinearView:
    return Linearizer(rbc, {"c": "log", "k": "log"}).at()


@pytest.fixture(scope="module")
def gap_linear(gap: CompiledModel) -> PointLinearView:
    return Linearizer(gap).at()


# ---------------- methods ----------------


def test_rejects_an_undeclared_variable(rbc):
    with pytest.raises(ValueError, match="does not have"):
        Linearizer(rbc, {"nope": "log"})


def test_rejects_a_minted_variable(aux):
    assert aux.layout.generated_names == ("u_lag1",)

    with pytest.raises(ValueError, match="compiler-minted"):
        Linearizer(aux, {"u_lag1": "log"})


def test_an_invalid_method_names_the_valid_ones(rbc):
    with pytest.raises(ValueError, match=r"Valid options are: \['log', 'level'\]"):
        Linearizer(rbc, {"c": "taylor"})


# ---------------- the expansion point ----------------


def test_the_solved_point_is_the_models_steady_state(rbc_log):
    assert float(rbc_log.steady_state["k"]) == pytest.approx(K_SS, rel=1e-6)
    assert float(rbc_log.steady_state["c"]) == pytest.approx(C_SS, rel=1e-6)
    assert float(rbc_log.steady_state["z"]) == pytest.approx(0.0, abs=1e-12)


def test_a_gap_model_solves_to_the_origin(gap_linear):
    assert all(
        float(value) == pytest.approx(0.0, abs=1e-12)
        for value in gap_linear.steady_state.values()
    )


def test_log_rejects_a_nonpositive_point(rbc):
    # z is an AR(1) in gaps, so its steady state is exactly zero and no
    # log deviation exists to expand in.
    with pytest.raises(ValueError, match="positive steady state"):
        Linearizer(rbc, {"z": "log"}).at()


# ---------------- parameters ----------------


def test_params_override_the_calibration(rbc):
    view = Linearizer(rbc, {"c": "log", "k": "log"}).at({"beta": 0.985})

    assert float(view.parameters["beta"]) == 0.985
    assert float(view.parameters["alpha"]) == 0.33
    # Discounting the future harder leaves less capital.
    assert float(view.steady_state["k"]) < K_SS


def test_params_rejects_an_undeclared_parameter(rbc):
    with pytest.raises(ValueError, match="does not have"):
        Linearizer(rbc).at({"nope": 1.0})


# ---------------- equations and observables ----------------


def test_a_gap_model_keeps_every_coefficient(gap, gap_linear):
    # Every variable sits at the origin and the model is already linear, so the
    # expansion cannot touch a slope. Only the constant can move, which is what
    # the next test pins down.
    authored = gap.config.equations.model
    assert list(gap_linear.model) == list(authored)

    for name, equation in gap_linear.model.items():
        difference = sp.expand(
            (equation.lhs - equation.rhs) - (authored[name].lhs - authored[name].rhs)
        )
        assert not difference.atoms(AppliedUndef), name


def test_the_value_at_the_expansion_point_is_subtracted_from_each_side(gap, gap_linear):
    # taylor reads r_star(t) = rbar + ..., so rbar is part of the right side's
    # value at the point and comes off with it. Parameters are never substituted
    # numerically, so it comes off as a symbol.
    authored = gap.config.equations.model["taylor"]
    linear = gap_linear.model["taylor"]

    difference = (linear.lhs - linear.rhs) - (authored.lhs - authored.rhs)
    assert sp.simplify(difference) == sp.Symbol("rbar")

    # The removal is not conditional on rbar being zero. At rbar = 0.5 the policy
    # rule's own feedback puts the point at -0.325 rather than the origin, and the
    # emitted equation is the same one. So estimating rbar on a to_config() model
    # cannot move this equation, while on the original it relocates the point.
    shifted = Linearizer(gap).at({"rbar": 0.5})

    assert float(shifted.steady_state["r_star"]) == pytest.approx(-0.32520325, rel=1e-6)
    assert shifted.model["taylor"] == linear


def test_observables_keep_their_intercept(gap_linear, rbc_log):
    # pi_mean is what maps the deviation back to an observed inflation level.
    assert gap_linear.observables["Infl"] == sp.Symbol("pi_mean") + sp.Function("Pi")(T)

    c_obs = rbc_log.observables["c_obs"]
    assert float(c_obs.coeff(sp.Function("c")(T))) == pytest.approx(C_SS, rel=1e-6)
    assert float(c_obs.xreplace({sp.Function("c")(T): 0})) == pytest.approx(
        C_SS, rel=1e-6
    )


def test_a_nonlinear_observable_comes_out_affine(aux):
    # log(ubar*exp(u_hat)) is log(ubar) + u_hat, which is the whole point of the
    # pass: the measurement equation stops being nonlinear.
    view = Linearizer(aux, {"u": "log"}).at()
    log_u_obs = view.observables["log_u_obs"]

    assert float(log_u_obs.coeff(sp.Function("u")(T))) == pytest.approx(1.0)
    assert float(log_u_obs.xreplace({sp.Function("u")(T): 0})) == pytest.approx(
        sp.log(2.0)
    )


def test_log_expansion_agrees_with_the_helper(rbc, rbc_log):
    # The guard between the two files. If at() ever stops routing through the
    # helpers, test_expansion.py keeps passing and this fails.
    methods = {
        "c": LinearizationMethod.LOG,
        "k": LinearizationMethod.LOG,
        "z": LinearizationMethod.LEVEL,
    }
    point = {str(name): float(value) for name, value in rbc_log.steady_state.items()}
    shocks = frozenset(rbc.config.shocks)

    expected = _linearize_equation(
        rbc.config.equations.model["euler"], methods, point, shocks
    )
    emitted = rbc_log.model["euler"]
    difference = (emitted.lhs - emitted.rhs) - (expected.lhs - expected.rhs)

    assert sp.simplify(difference) == 0


# ---------------- to_config ----------------


def test_to_config_keeps_the_equation_key_order(rbc, rbc_log):
    assert list(rbc_log.to_config().equations.model) == list(rbc.config.equations.model)


def test_to_config_zeroes_every_seed(rbc_log):
    # The deviation system clears at the origin. Left alone, the authored k_ss
    # would seed Newton at exp(k_ss) times the level it was written for.
    seeds = rbc_log.to_config().variables.ss_seed

    assert set(seeds) == set(rbc_log.steady_state)
    assert all(float(value) == 0.0 for value in seeds.values())


def test_to_config_carries_the_override_parameters(rbc):
    config = Linearizer(rbc, {"c": "log", "k": "log"}).at({"beta": 0.985}).to_config()

    assert float(config.calibration.parameters["beta"]) == 0.985


def test_to_config_marks_every_observable_affine(aux):
    # Both methods emit an affine observable, so the non-affine measurement path
    # in solved_model has nothing left to evaluate.
    config = Linearizer(aux, {"u": "log"}).at().to_config()

    assert all(config.equations.obs_is_affine.values())


def test_to_config_does_not_alias_the_view(rbc_log):
    config = rbc_log.to_config()
    config.equations.model.clear()
    config.equations.observable.clear()
    config.calibration.parameters.clear()

    assert rbc_log.model
    assert rbc_log.observables
    assert rbc_log.parameters
    assert rbc_log.to_config().equations.model


# ---------------- the convenience wrapper ----------------


def test_linearize_model_matches_the_class_at_base_calibration(rbc, rbc_log):
    view = linearize_model(rbc, {"c": "log", "k": "log"})

    assert view.model == rbc_log.model
    assert dict(view.steady_state) == dict(rbc_log.steady_state)


# ---------------- solver rejects a view ----------------


def test_solver_rejects_a_view(gap_linear):
    with pytest.raises(TypeError, match="point-linearized model is not accepted"):
        DSGESolver(gap_linear)
