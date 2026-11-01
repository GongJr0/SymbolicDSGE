"""``lock``/``unlock`` over a whole config, checked by reflection.

The walks below find members through dataclass fields and mapping values rather
than through what each class names in its own ``_freeze``, so they fail on a
field nobody declared. A member frozen on the way in and forgotten on the way
out is the one defect the base's ordering cannot rule out.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass

import pytest
import sympy as sp

from SymbolicDSGE.core.config import Constraint
from SymbolicDSGE.core.types import _Lockable

#: Containers that carry no lock of their own, so holding one is the defect.
BARE = (dict, list, set)


def _lockables(node):
    """Every ``_Lockable`` reachable through dataclass fields and mapping values."""
    if isinstance(node, _Lockable):
        yield node
    if is_dataclass(node) and not isinstance(node, type):
        for field in fields(node):
            yield from _lockables(getattr(node, field.name))
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _lockables(value)


def _members(node, path):
    if is_dataclass(node) and not isinstance(node, type):
        return [(f"{path}.{f.name}", getattr(node, f.name)) for f in fields(node)]
    if isinstance(node, Mapping):
        return [(f"{path}[{k!r}]", v) for k, v in node.items()]
    return []


def _bare_containers(node, path="config"):
    """Paths whose value is one of ``BARE``, reported as ``path is a dict``."""
    found = []
    for where, value in _members(node, path):
        if type(value) in BARE:
            found.append(f"{where} is a {type(value).__name__}")
        found += _bare_containers(value, where)
    return found


def _still_locked(config):
    return [type(obj).__name__ for obj in _lockables(config) if obj.is_locked]


def _unlocked(config):
    return [type(obj).__name__ for obj in _lockables(config) if not obj.is_locked]


@pytest.fixture(scope="module")
def constrained_source(parsed_test):
    """The reference model carrying a constraint and a regime, assigned as plain dicts.

    Nothing here is compiled or solved, so the regime reuses one of the model's
    own equations as its replacement; the walk cares that it is a mapping of
    equations, not what it says.
    """
    conf = deepcopy(parsed_test)
    t = sp.Symbol("t", integer=True)
    pi = next(var for var in conf.variables if str(var) == "Pi")

    conf.equations.constraints = {"floor": Constraint(bind=pi(t) < 0, relax=pi(t) >= 0)}
    conf.equations.regimes = {
        frozenset({"floor"}): {"taylor": conf.equations.model["taylor"]}
    }
    return conf


@pytest.fixture(params=["reference", "constrained"])
def config(request, parsed_test, constrained_source):
    source = parsed_test if request.param == "reference" else constrained_source
    return deepcopy(source)


def test_locking_reaches_every_lockable_in_the_config(config):
    config.lock()

    assert not _unlocked(config)


def test_a_locked_config_holds_no_bare_mutable_container(config):
    config.lock()

    assert not _bare_containers(config)


def test_unlocking_releases_every_lockable_locking_reached(config):
    config.lock()
    config.unlock()

    assert not _still_locked(config)


def test_locking_and_unlocking_are_idempotent(config):
    assert config.lock() is config
    config.lock()
    assert not _unlocked(config)

    assert config.unlock() is config
    config.unlock()
    assert not _still_locked(config)

    config.lock()
    assert not _unlocked(config)


def test_the_frozen_sequences_follow_the_lock_state(config):
    names = ("variables", "shocks", "observables")

    config.lock()
    assert [type(getattr(config, n)) for n in names] == [tuple] * 3

    config.unlock()
    assert [type(getattr(config, n)) for n in names] == [list] * 3


def test_a_locked_config_deepcopies_and_the_copy_arrives_locked(config):
    config.lock()

    copy = deepcopy(config)

    assert not _unlocked(copy)
    assert copy.equations.model is not config.equations.model


def test_locking_walks_every_kind_of_member(constrained_source):
    config = deepcopy(constrained_source)

    config.lock()

    assert {
        "ModelConfig",
        "Equations",
        "Calib",
        "CovarianceSpec",
        "Constraint",
        "LockableDict",
        "RegimeGetterDict",
        "SymbolGetterDict",
        "FunctionGetterDict",
    } <= {type(obj).__name__ for obj in _lockables(config)}


def test_locking_normalizes_a_mapping_field_assigned_as_a_plain_dict(
    constrained_source,
):
    config = deepcopy(constrained_source)
    assert type(config.equations.constraints) is dict
    assert type(config.equations.regimes) is dict

    config.lock()

    equations = config.equations
    assert type(equations.constraints).__name__ == "LockableDict"
    assert type(equations.regimes).__name__ == "RegimeGetterDict"
    assert [type(v).__name__ for v in equations.regimes.values()] == ["LockableDict"]


def test_an_unlocked_config_is_editable_again_after_a_round_trip(config):
    config.lock()
    config.unlock()
    beta = next(iter(config.calibration.parameters))

    config.calibration.parameters[beta] = 0.5
    config.equations.model["taylor"] = config.equations.model["euler"]
    config.variables.append(sp.Function("extra"))

    assert config.calibration.parameters[beta] == 0.5
