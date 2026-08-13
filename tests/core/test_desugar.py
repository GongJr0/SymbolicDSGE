# type: ignore
"""Desugaring, which now only reaches lags deeper than one.

The residual carries three dates and the innovations directly, so a lag of one
and a shock both survive as the author wrote them. What is left to lift is depth
two and beyond, which is genuinely outside the dates the pencil holds: ``v(t-n)``
for ``n >= 2`` becomes ``v_lag{n-1}(t-1)``, defined by the chain ``v_lag1(t) =
v(t-1)``, ``v_lag2(t) = v_lag1(t-1)`` and onward.

A model written with ordinary lags therefore compiles to exactly the variables it
declares, which is what these check first.
"""

from __future__ import annotations

import copy

import pytest
import sympy as sp
import yaml
from sympy.core.function import AppliedUndef

from SymbolicDSGE.core import ModelParser, desugar_model
from SymbolicDSGE.core.linearization import LinearizationMethod

t = sp.Symbol("t", integer=True)

# MODELS/test.yaml written the way the author would write it, with the lags left
# in place instead of shifted forward. Every lag here is depth one.
NATURAL_TIME_MODEL = {
    "name": "NATURAL",
    "variables": ["u", "v", "r", "Pi", "x", "r_star"],
    "shock_map": {"e_u": "u", "e_v": "v"},
    "observables": ["Infl", "Rate"],
    "equations": {
        "model": {
            "nkpc": "Pi(t) = beta*Pi(t+1) + kappa*x(t)",
            "euler": "x(t) = x(t+1) - sigma*(r(t) - Pi(t+1)) + u(t)",
            "u_process": "u(t) = rho_u*u(t-1) + e_u",
            "taylor": "r_star(t) = rbar + phi_pi*Pi(t) + phi_x*x(t) + v(t)",
            "policy": "r(t) = rho_r*r(t-1) + (1 - rho_r)*r_star(t)",
            "v_process": "v(t) = rho_v*v(t-1) + e_v",
        },
        "constraint": {},
        "observables": {"Infl": "Pi(t) + pi_mean", "Rate": "r(t)"},
    },
    "calibration": {
        "parameters": {
            "beta": 0.99,
            "sigma": 1.0,
            "kappa": 0.024,
            "phi_pi": 2.15,
            "phi_x": 0.93,
            "rho_r": 0.79,
            "rho_u": 0.70,
            "sig_u": 0.50,
            "rho_v": 0.50,
            "sig_v": 0.25,
            "rbar": 0.0,
            "pi_mean": 3.25,
        },
        "shocks": {"std": {"e_u": "sig_u", "e_v": "sig_v"}, "corr": {}},
    },
}


def _parse(tmp_path, data):
    path = tmp_path / f"model_{len(list(tmp_path.iterdir()))}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return ModelParser(path).get_all().model


def _deep(tmp_path, **equations: str):
    """The same model with something lagged past the dates the residual holds."""
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-2) + e_u"
    data["equations"]["model"].update(equations)
    return _parse(tmp_path, data)


@pytest.fixture
def natural(tmp_path):
    return _parse(tmp_path, NATURAL_TIME_MODEL)


@pytest.fixture
def deep(tmp_path):
    return _deep(tmp_path)


def _offsets(conf):
    """Every (variable, offset) pair the config's expressions reference."""
    declared = {f.__name__ for f in conf.variables.variables}
    exprs = []
    for eq in conf.equations.model.values():
        exprs += [eq.lhs, eq.rhs]
    for replacements in (conf.equations.regime or {}).values():
        for eq in replacements.values():
            exprs += [eq.lhs, eq.rhs]
    exprs += list(conf.equations.observable.values())

    seen = set()
    for expr in exprs:
        for call in expr.atoms(AppliedUndef):
            name = call.func.__name__
            if name not in declared or not call.args:
                continue
            offset = sp.simplify(call.args[0] - t)
            if offset.is_Integer:
                seen.add((name, int(offset)))
    return seen


def test_a_model_of_ordinary_lags_mints_nothing(natural):
    # The point of the whole exercise: what the author declares is what compiles.
    result = desugar_model(natural)

    assert result.names == ()
    assert [f.__name__ for f in result.config.variables.variables] == [
        f.__name__ for f in natural.variables.variables
    ]


def test_a_shock_is_left_where_it_was_written(natural):
    # No state carries an innovation any more, so the raw symbol stays in every
    # equation that uses it and reaches the pencil through the residual.
    result = desugar_model(natural)
    e_u = sp.Symbol("e_u")

    carriers = [
        name
        for name, eq in result.config.equations.model.items()
        if e_u in (eq.lhs - eq.rhs).free_symbols
    ]
    assert carriers == ["u_process"]


def test_a_lag_of_one_survives_as_written(natural):
    result = desugar_model(natural)
    u = sp.Function("u")
    rho_u, e_u = sp.Symbol("rho_u"), sp.Symbol("e_u")

    rhs = result.config.equations.model["u_process"].rhs
    assert sp.simplify(rhs - (rho_u * u(t - 1) + e_u)) == 0


def test_offsets_stay_within_the_three_dates_the_residual_holds(deep):
    result = desugar_model(deep)

    assert all(-1 <= offset <= 1 for _, offset in _offsets(result.config))
    assert any(offset < -1 for _, offset in _offsets(deep))


def test_a_deeper_lag_reads_its_aux_one_date_back(deep):
    result = desugar_model(deep)
    conf = result.config
    u, u_lag1 = sp.Function("u"), sp.Function("u_lag1")

    # Depth two needs one aux, not two: the residual's own lagged date carries
    # the last step.
    assert result.names == ("u_lag1",)
    assert conf.equations.model["u_lag1"] == sp.Eq(u_lag1(t), u(t - 1))
    assert u_lag1(t - 1) in conf.equations.model["u_process"].rhs.atoms(AppliedUndef)


def test_a_deeper_chain_numbers_its_auxes_rather_than_stacking_suffixes(tmp_path):
    conf = _deep(tmp_path, u_process="u(t) = rho_u*u(t-3) + e_u")
    result = desugar_model(conf)
    u, u_lag1, u_lag2 = (sp.Function(n) for n in ("u", "u_lag1", "u_lag2"))

    assert result.names == ("u_lag1", "u_lag2")
    assert not any("_lag_lag" in name for name in result.names)
    assert result.config.equations.model["u_lag1"] == sp.Eq(u_lag1(t), u(t - 1))
    assert result.config.equations.model["u_lag2"] == sp.Eq(u_lag2(t), u_lag1(t - 1))


def test_lag_scan_covers_regimes_the_reference_never_lags(natural):
    # A regime lagging something the reference does not would otherwise widen the
    # regime pencil past the reference it has to stay row-aligned with.
    conf = copy.deepcopy(natural)
    r, Pi = (sp.Function(n) for n in ("r", "Pi"))
    conf.equations.regime = {frozenset({"elb"}): {"policy": sp.Eq(r(t), Pi(t - 2))}}
    result = desugar_model(conf)

    assert "Pi_lag1" in result.names
    assert "Pi_lag1" in result.config.equations.model
    replacement = result.config.equations.regime[frozenset({"elb"})]["policy"]
    assert replacement == sp.Eq(r(t), sp.Function("Pi_lag1")(t - 1))


def test_observables_are_rewritten(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["observables"]["Rate"] = "r(t-2)"
    result = desugar_model(_parse(tmp_path, data))

    assert result.config.equations.observable[sp.Symbol("Rate")] == sp.Function(
        "r_lag1"
    )(t - 1)


def test_lag_aux_inherits_its_origin_expansion_point(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-2) + e_u"
    data["variables"] = {
        "u": {"ss_seed": "rbar", "linearization": "log"},
        "v": None,
        "r": None,
        "Pi": None,
        "x": None,
        "r_star": None,
    }
    result = desugar_model(_parse(tmp_path, data))
    variables = result.config.variables

    assert variables.ss_seed["u_lag1"] == variables.ss_seed["u"]
    assert variables.linearization["u_lag1"] is LinearizationMethod.LOG


def test_generated_variables_are_appended_to_the_declared_set(deep):
    result = desugar_model(deep)
    names = [f.__name__ for f in result.config.variables.variables]

    assert names[: len(deep.variables.variables)] == [
        f.__name__ for f in deep.variables.variables
    ]
    assert names[len(deep.variables.variables) :] == list(result.names)


def test_positions_maps_generated_names_to_their_slot(deep):
    result = desugar_model(deep)
    order = [f.__name__ for f in result.config.variables.variables]

    assert result.positions(order) == {name: order.index(name) for name in result.names}
    assert result.positions(["u", "Pi"]) == {}


def test_source_config_is_untouched(deep):
    # Compared by content rather than against a deepcopy: the parser leaves time
    # arguments unevaluated (`t - 1*1`) and deepcopy normalizes them, so a copied
    # baseline differs from its source on expressions nothing ever rewrote.
    names = [f.__name__ for f in deep.variables.variables]
    equations = set(deep.equations.model)
    desugar_model(deep)

    assert [f.__name__ for f in deep.variables.variables] == names
    assert set(deep.equations.model) == equations
    assert any(offset < -1 for _, offset in _offsets(deep))


def test_a_second_pass_is_a_no_op(deep):
    # Desugaring leaves every lag at depth one, so running it again finds nothing
    # to lift. It used to raise here, back when a pass re-lifted its own auxes.
    once = desugar_model(deep)
    twice = desugar_model(once.config)

    assert twice.names == ()
    assert [f.__name__ for f in twice.config.variables.variables] == [
        f.__name__ for f in once.config.variables.variables
    ]
    assert set(twice.config.equations.model) == set(once.config.equations.model)


def test_generated_name_colliding_with_a_declared_variable_is_rejected(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-2) + e_u"
    data["variables"] = [*data["variables"], "u_lag1"]
    data["equations"]["model"]["spare"] = "u_lag1(t) = 0"

    with pytest.raises(ValueError, match="already declares"):
        desugar_model(_parse(tmp_path, data))
