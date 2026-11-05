"""``ModelConfig.token`` and the config/compiled pairing it guards.

Every field of ``ModelConfig``, ``Equations`` and ``Calib`` is classified below
as covered by the token, excluded from it, or traversed for the fields under
it, and a field that is none of the three fails
``test_every_field_is_classified``. The
per-field mutations then prove the classification instead of restating it: a
covered field has to move the token and an excluded one has to leave it alone.
"""

from copy import deepcopy
from dataclasses import fields

import pytest
import sympy as sp

from SymbolicDSGE import parse_config, solve_model
from SymbolicDSGE.core.config import (
    Calib,
    Constraint,
    CovarianceSpec,
    Equations,
    ModelConfig,
)
from SymbolicDSGE.core.types import PairGetterDict, SymbolGetterDict
from SymbolicDSGE.estimation import Estimator

#: Paths the token answers for, because a compile reads them.
COVERED = {
    "variables",
    "shocks",
    "observables",
    "equations.model",
    "equations.observables",
    "equations.constraints",
    "equations.regimes",
    "calibration.parameters",
}

#: Paths a solve is free to vary while a compiled model stays valid.
EXCLUDED = {
    "name",
    "ss_seed",
    "equations.obs_is_affine",
    "calibration.shocks",
    "calibration.measurement_noise",
}

#: Walked for the fields under them, never hashed as a unit.
TRAVERSED = {"equations", "calibration"}


def _reorder(conf, attr):
    setattr(conf, attr, list(reversed(getattr(conf, attr))))


def _edit_equation(conf):
    name, eq = next(iter(conf.equations.model.items()))
    conf.equations.model[name] = eq.func(eq.lhs + 1, eq.rhs)


def _edit_observable(conf):
    name, expr = next(iter(conf.equations.observables.items()))
    conf.equations.observables[name] = expr + 1


def _edit_constraint(conf):
    constraint = next(iter(conf.equations.constraints.values()))
    bind = constraint.bind
    constraint.bind = bind.func(bind.lhs, bind.rhs + 1)


def _edit_regime(conf):
    regime = next(iter(conf.equations.regimes.values()))
    name, eq = next(iter(regime.items()))
    regime[name] = eq.func(eq.lhs + 1, eq.rhs)


def _edit_ss_seed(conf):
    name = next(iter(conf.ss_seed))
    conf.ss_seed[name] = sp.Float(9.0)


def _add_parameter(conf):
    conf.calibration.parameters[sp.Symbol("extra_parameter")] = 0.5


def _retarget_std(conf):
    shock = next(iter(conf.calibration.shocks.std))
    conf.calibration.shocks.std[shock] = str(next(iter(conf.calibration.parameters)))


def _set_measurement_noise(conf):
    conf.calibration.measurement_noise = CovarianceSpec(
        std=SymbolGetterDict({str(obs): "0.1" for obs in conf.observables}),
        corr=PairGetterDict({}),
    )


def _flip_obs_is_affine(conf):
    name, affine = next(iter(conf.equations.obs_is_affine.items()))
    conf.equations.obs_is_affine[name] = not affine


MOVES = {
    "variables": lambda conf: _reorder(conf, "variables"),
    "shocks": lambda conf: _reorder(conf, "shocks"),
    "observables": lambda conf: _reorder(conf, "observables"),
    "equations.model": _edit_equation,
    "equations.observables": _edit_observable,
    "equations.constraints": _edit_constraint,
    "equations.regimes": _edit_regime,
    "calibration.parameters": _add_parameter,
}

STAYS = {
    "name": lambda conf: setattr(conf, "name", "renamed"),
    "ss_seed": _edit_ss_seed,
    "equations.obs_is_affine": _flip_obs_is_affine,
    "calibration.shocks": _retarget_std,
    "calibration.measurement_noise": _set_measurement_noise,
}


@pytest.fixture
def constrained(parsed_test):
    """``test.yaml`` carrying two constraints and two regimes, assigned as plain dicts.

    Two of each is what makes a reordering observable. Nothing here is compiled,
    so the regimes reuse the model's own equations as their replacements.
    """
    conf = deepcopy(parsed_test)
    t = sp.Symbol("t", integer=True)
    pi = next(var for var in conf.variables if str(var) == "Pi")
    first, second = list(conf.equations.model)[:2]

    conf.equations.constraints = {
        "floor": Constraint(bind=pi(t) < 0, relax=pi(t) >= 0),
        "ceiling": Constraint(bind=pi(t) > 1, relax=pi(t) <= 1),
    }
    conf.equations.regimes = {
        frozenset({"floor"}): {first: conf.equations.model[first]},
        frozenset({"ceiling"}): {second: conf.equations.model[second]},
    }
    return conf


@pytest.fixture(params=["reference", "constrained"])
def config(request, parsed_test, constrained):
    return deepcopy(parsed_test) if request.param == "reference" else constrained


def test_every_field_is_classified():
    paths = (
        {f.name for f in fields(ModelConfig)}
        | {f"equations.{f.name}" for f in fields(Equations)}
        | {f"calibration.{f.name}" for f in fields(Calib)}
    )

    assert COVERED | EXCLUDED | TRAVERSED == paths


@pytest.mark.parametrize("path", sorted(COVERED))
def test_an_edit_to_a_covered_field_moves_the_token(constrained, path):
    before = constrained.token

    MOVES[path](constrained)

    assert constrained.token != before


@pytest.mark.parametrize("path", sorted(EXCLUDED))
def test_an_edit_to_an_excluded_field_leaves_the_token_alone(constrained, path):
    before = constrained.token

    STAYS[path](constrained)

    assert constrained.token == before


def test_the_token_survives_a_lock_and_a_round_trip(config):
    before = config.token

    assert deepcopy(config).lock().token == before
    assert deepcopy(config).lock().unlock().token == before


def test_the_token_survives_a_deepcopy(config):
    assert deepcopy(config).token == config.token


def test_two_parses_of_one_file_agree(test_model_path):
    assert parse_config(test_model_path).token == parse_config(test_model_path).token


def test_the_token_follows_the_order_of_the_compiled_rows(constrained):
    before = constrained.token
    model = constrained.equations.model

    constrained.equations.model = type(model)(dict(reversed(list(model.items()))))

    assert constrained.token != before


def test_the_token_follows_the_constraint_declaration_order(constrained):
    before = constrained.token

    constrained.equations.constraints = dict(
        reversed(list(constrained.equations.constraints.items()))
    )

    assert constrained.token != before


def test_the_token_follows_the_parameter_declaration_order(constrained):
    """An array ``theta0`` is documented as being in calibration order."""
    before = constrained.token
    parameters = constrained.calibration.parameters

    constrained.calibration.parameters = type(parameters)(
        dict(reversed(list(parameters.items())))
    )

    assert constrained.token != before


def test_dropping_a_parameter_moves_the_token(constrained):
    before = constrained.token

    del constrained.calibration.parameters[
        next(iter(constrained.calibration.parameters))
    ]

    assert constrained.token != before


def test_a_parameter_value_write_leaves_the_token_alone(constrained):
    before = constrained.token
    param = next(iter(constrained.calibration.parameters))

    constrained.calibration.parameters[param] = 0.123

    assert constrained.token == before


def test_the_token_ignores_the_order_regimes_were_assigned_in(constrained):
    before = constrained.token

    constrained.equations.regimes = dict(
        reversed(list(constrained.equations.regimes.items()))
    )

    assert constrained.token == before


def test_compilation_records_the_token_of_the_config_it_read(
    parsed_test, compiled_test
):
    assert compiled_test.config_token == parsed_test.token


def test_a_matched_pair_solves(parsed_test, compiled_test):
    solved = solve_model(deepcopy(parsed_test), compiled=compiled_test)

    assert solved.compiled is compiled_test


def test_a_solved_models_own_pair_solves_again(solved_test):
    again = solve_model(solved_test.config, compiled=solved_test.compiled)

    assert again.compiled is solved_test.compiled


def test_a_parameter_write_keeps_the_pair_valid(parsed_test, compiled_test):
    conf = deepcopy(parsed_test)
    name = next(iter(conf.calibration.parameters))

    conf.update_parameters({str(name): 0.123})

    assert conf.token == compiled_test.config_token


def test_a_parameter_copy_keeps_the_pair_valid(parsed_test, compiled_test):
    """``with_parameters`` is how ``solve_model`` reaches the config it solves,
    so a copy that lost the token would reject every compiled model passed with
    it."""
    name = next(iter(parsed_test.calibration.parameters))

    conf = parsed_test.with_parameters({str(name): 0.123})

    assert conf.token == compiled_test.config_token
    assert solve_model(conf, compiled=compiled_test).compiled is compiled_test


def test_solving_rejects_a_compiled_model_from_another_config(
    parsed_post82, compiled_test
):
    with pytest.raises(ValueError, match="not compatible"):
        solve_model(parsed_post82, compiled=compiled_test)


def test_solving_rejects_a_compiled_model_the_config_outgrew(
    parsed_test, compiled_test
):
    conf = deepcopy(parsed_test)
    _edit_equation(conf)

    with pytest.raises(ValueError, match="not compatible"):
        solve_model(conf, compiled=compiled_test)


def test_the_estimator_rejects_a_compiled_model_from_another_config(
    solved_post82, y_post82, compiled_test
):
    with pytest.raises(ValueError, match="not compatible"):
        Estimator(config=solved_post82.config, y=y_post82, compiled=compiled_test)
