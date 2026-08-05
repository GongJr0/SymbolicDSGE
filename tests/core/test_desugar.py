# type: ignore
from __future__ import annotations

import copy

import pytest
import sympy as sp
import yaml
from sympy.core.function import AppliedUndef

from SymbolicDSGE.core import ModelParser, desugar_model
from SymbolicDSGE.core.desugar import GeneratedKind
from SymbolicDSGE.core.linearization import LinearizationMethod

t = sp.Symbol("t", integer=True)

# MODELS/test.yaml written the way the author would write it, with the lags left
# in place instead of shifted forward.
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
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return ModelParser(path).get_all().model


@pytest.fixture
def natural(tmp_path):
    return _parse(tmp_path, NATURAL_TIME_MODEL)


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


def test_lifts_every_lag_and_every_shock(natural):
    result = desugar_model(natural)

    assert result.names == (
        "u_lag1",
        "v_lag1",
        "r_lag1",
        "e_u_st",
        "e_v_st",
    )
    kinds = {g.name: g.kind for g in result.generated}
    assert kinds["u_lag1"] is GeneratedKind.LAG
    assert kinds["e_u_st"] is GeneratedKind.SHOCK
    origins = {g.name: g.origin for g in result.generated}
    assert origins == {
        "u_lag1": "u",
        "v_lag1": "v",
        "r_lag1": "r",
        "e_u_st": "e_u",
        "e_v_st": "e_v",
    }


def test_rewritten_model_carries_no_negative_offsets(natural):
    result = desugar_model(natural)

    assert all(offset >= 0 for _, offset in _offsets(result.config))
    assert any(offset < 0 for _, offset in _offsets(natural))


def test_lagged_reference_becomes_the_aux_read_contemporaneously(natural):
    result = desugar_model(natural)
    conf = result.config
    u, u_lag1, e_u_st = (sp.Function(n) for n in ("u", "u_lag1", "e_u_st"))
    rho_u = sp.Symbol("rho_u")

    assert (
        sp.simplify(
            conf.equations.model["u_process"].rhs - (rho_u * u_lag1(t) + e_u_st(t))
        )
        == 0
    )
    assert conf.equations.model["u_lag1"] == sp.Eq(u_lag1(t + 1), u(t))


def test_shock_state_is_the_only_equation_the_raw_shock_survives_in(natural):
    # The whole shock impact block reduces to a gather because of this: the raw
    # symbol reaches exactly one row, and that row is its own state's.
    result = desugar_model(natural)
    e_u = sp.Symbol("e_u")

    carriers = [
        name
        for name, eq in result.config.equations.model.items()
        if e_u in (eq.lhs - eq.rhs).free_symbols
    ]
    assert carriers == ["e_u_st"]
    assert result.config.equations.model["e_u_st"] == sp.Eq(
        sp.Function("e_u_st")(t + 1), e_u
    )


def test_shocks_lift_even_when_the_model_already_wraps_them(tmp_path):
    # MODELS/test.yaml's hand-shifted form has no lags at all. The shock lift is
    # uniform regardless, which is what keeps the gather claim unconditional.
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["model"]["u_process"] = "u(t+1) = rho_u*u(t) + e_u"
    data["equations"]["model"]["v_process"] = "v(t+1) = rho_v*v(t) + e_v"
    data["equations"]["model"][
        "policy"
    ] = "r(t+1) = rho_r*r(t) + (1 - rho_r)*r_star(t+1)"
    result = desugar_model(_parse(tmp_path, data))

    assert result.names == ("e_u_st", "e_v_st")


def test_deeper_lag_numbers_its_aux_rather_than_stacking_suffixes(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-2) + e_u"
    result = desugar_model(_parse(tmp_path, data))
    conf = result.config
    u, u_lag1, u_lag2 = (sp.Function(n) for n in ("u", "u_lag1", "u_lag2"))

    assert "u_lag1" in result.names and "u_lag2" in result.names
    assert not any("_lag_lag" in name for name in result.names)
    assert conf.equations.model["u_lag1"] == sp.Eq(u_lag1(t + 1), u(t))
    assert conf.equations.model["u_lag2"] == sp.Eq(u_lag2(t + 1), u_lag1(t))
    assert u_lag2(t) in conf.equations.model["u_process"].rhs.atoms(AppliedUndef)


def test_lag_scan_covers_regimes_the_reference_never_lags(natural):
    # A regime lagging something the reference does not would otherwise widen the
    # regime pencil past the reference it has to stay row-aligned with.
    conf = copy.deepcopy(natural)
    r, Pi = (sp.Function(n) for n in ("r", "Pi"))
    conf.equations.regime = {frozenset({"elb"}): {"policy": sp.Eq(r(t), Pi(t - 1))}}
    result = desugar_model(conf)

    assert "Pi_lag1" in result.names
    assert "Pi_lag1" in result.config.equations.model
    replacement = result.config.equations.regime[frozenset({"elb"})]["policy"]
    assert replacement == sp.Eq(r(t), sp.Function("Pi_lag1")(t))


def test_observables_are_rewritten(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["equations"]["observables"]["Rate"] = "r(t-1)"
    result = desugar_model(_parse(tmp_path, data))

    assert result.config.equations.observable[sp.Symbol("Rate")] == sp.Function(
        "r_lag1"
    )(t)


def test_lag_aux_inherits_its_origin_expansion_point(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["variables"] = {
        "u": {"ss_seed": "rbar", "linearization": "log"},
        "v": None,
        "r": None,
        "Pi": None,
        "x": None,
        "r_star": None,
    }
    conf = _parse(tmp_path, data)
    result = desugar_model(conf)
    variables = result.config.variables

    assert variables.ss_seed["u_lag1"] == variables.ss_seed["u"]
    assert variables.linearization["u_lag1"] is LinearizationMethod.LOG


def test_shock_state_expands_at_zero_and_is_never_linearized(natural):
    variables = desugar_model(natural).config.variables

    assert variables.ss_seed["e_u_st"] == 0
    assert variables.linearization["e_u_st"] is LinearizationMethod.NONE


def test_generated_variables_are_appended_to_the_declared_set(natural):
    result = desugar_model(natural)
    names = [f.__name__ for f in result.config.variables.variables]

    assert names[: len(natural.variables.variables)] == [
        f.__name__ for f in natural.variables.variables
    ]
    assert names[len(natural.variables.variables) :] == list(result.names)


def test_positions_maps_generated_names_to_their_slot(natural):
    result = desugar_model(natural)
    order = [f.__name__ for f in result.config.variables.variables]

    assert result.positions(order) == {name: order.index(name) for name in result.names}
    assert result.positions(["u", "Pi"]) == {}


def test_source_config_is_untouched(natural):
    # Compared by content rather than against a deepcopy: the parser leaves time
    # arguments unevaluated (`t - 1*1`) and deepcopy normalizes them, so a copied
    # baseline differs from its source on expressions nothing ever rewrote.
    names = [f.__name__ for f in natural.variables.variables]
    equations = set(natural.equations.model)
    desugar_model(natural)

    assert [f.__name__ for f in natural.variables.variables] == names
    assert set(natural.equations.model) == equations
    assert any(offset < 0 for _, offset in _offsets(natural))


def test_second_pass_is_rejected(natural):
    once = desugar_model(natural).config

    with pytest.raises(ValueError, match="already desugared"):
        desugar_model(once)


def test_generated_name_colliding_with_a_declared_variable_is_rejected(tmp_path):
    data = copy.deepcopy(NATURAL_TIME_MODEL)
    data["variables"] = [*data["variables"], "u_lag1"]
    data["equations"]["model"]["spare"] = "u_lag1(t) = 0"

    with pytest.raises(ValueError, match="already declares"):
        desugar_model(_parse(tmp_path, data))
