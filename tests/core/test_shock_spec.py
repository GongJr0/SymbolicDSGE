"""Normalizing a shock spec: what the two authored shapes accept and reject.

``_normalized_spec`` is the one seam between what a user writes and what every
consumer reads, so what it lets past is what every later stage has to cope with.
It answers one question, whether an entry is a spec at all; whether the names it
carries are model shocks is ``validate_shock_targets``, which runs where a model
is in hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.shock.generators import Shock, ShockPath
from SymbolicDSGE.core.shock.spec import _normalized_spec


# --- the two authored shapes ------------------------------------------------


def test_none_is_the_empty_spec() -> None:
    assert _normalized_spec(None) == []


def test_a_mapping_binds_its_values_to_its_keys() -> None:
    (shock,) = _normalized_spec({"e_u": Shock("norm", seed=1)})
    assert isinstance(shock, Shock)
    assert shock.target == ("e_u",)

    # A grouped key binds one joint entry over the names it lists.
    (joint,) = _normalized_spec({("e_u", "e_v"): Shock("norm", seed=1)})
    assert joint.target == ("e_u", "e_v")


def test_a_mapping_value_may_be_the_path_itself() -> None:
    path = np.zeros((4, 1))
    (entry,) = _normalized_spec({"e_u": path})
    assert isinstance(entry, ShockPath)
    assert entry.target == ("e_u",)
    assert entry.path is path


def test_a_sequence_takes_entries_that_name_themselves() -> None:
    spec = [ShockPath(np.zeros((4, 1)), "e_u"), Shock("norm", seed=2).joint("e_v")]
    assert [type(s) for s in _normalized_spec(spec)] == [ShockPath, Shock]


def test_normalizing_an_already_normal_spec_changes_nothing() -> None:
    # Consumers normalize independently, so the second pass has to be a no-op.
    once = _normalized_spec({"e_u": Shock("norm", seed=1)})
    twice = _normalized_spec(once)
    assert [s.target for s in twice] == [s.target for s in once]


# --- what is not a spec -----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        lambda scale: np.zeros(4),  # a callable was a spec two formats ago
        "norm",
        object(),
    ],
)
def test_a_mapping_value_must_be_a_shock_or_a_path(value) -> None:
    with pytest.raises(TypeError, match="must be a Shock to draw from"):
        _normalized_spec({"e_u": value})


def test_a_mapping_rejects_a_shockpath_that_carries_its_own_target() -> None:
    # The key could only restate the path's target or contradict it, and the
    # sequence shape is where an entry names itself.
    with pytest.raises(TypeError, match="must be a Shock to draw from"):
        _normalized_spec({"e_v": ShockPath(np.zeros((4, 1)), "e_u")})


@pytest.mark.parametrize("value", [np.zeros((4, 1)), object(), "e_u"])
def test_a_sequence_entry_must_name_itself(value) -> None:
    # A bare array in a sequence names no shock; neither does anything else
    # that is not one of the two entry types.
    with pytest.raises(TypeError, match="must be a bound Shock or a ShockPath"):
        _normalized_spec([value])


def test_a_bare_string_is_not_a_spec() -> None:
    # A str is a Sequence, so it reaches the entry check one character at a time
    # rather than being read as a shock name.
    with pytest.raises(TypeError, match="must be a bound Shock or a ShockPath"):
        _normalized_spec("e_u")


def test_an_unbound_shock_in_a_sequence_names_the_binders() -> None:
    with pytest.raises(ValueError, match=r"\.joint\(\*keys\).*\.independent"):
        _normalized_spec([Shock("norm", seed=1)])


def test_a_spec_must_be_a_mapping_or_a_sequence() -> None:
    with pytest.raises(TypeError, match="must be a mapping or sequence"):
        _normalized_spec(7)
