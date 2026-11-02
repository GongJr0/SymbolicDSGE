# type: ignore
"""The module level kernel cache keyed by ``(config_token, kind, variant)``.

The cache is append-only and shared by every compiled model in the process, so
these read the entries a construction is expected to leave behind rather than
the size or emptiness of the whole dict.
"""

from __future__ import annotations

import pytest

from SymbolicDSGE.core.compiled_model import _CFUNC_CACHE


def _key(compiled, kind, variant=None):
    return (compiled.config_token, kind, variant)


@pytest.mark.parametrize(
    "kind, call",
    [
        ("residual_c128", lambda c: c.construct_objective_cfunc()),
        ("residual_bicomplex", lambda c: c.construct_objective_cfunc_bicomplex()),
    ],
)
def test_a_residual_constructor_stores_what_it_returns(compiled_post82, kind, call):
    built = call(compiled_post82)

    assert _CFUNC_CACHE[_key(compiled_post82, kind)] is built


@pytest.mark.parametrize(
    "kind, call",
    [
        ("measurement", lambda c, obs: c.construct_measurement_cfunc(obs)),
        ("measurement_jac", lambda c, obs: c.construct_measurement_jacobian_cfunc(obs)),
    ],
)
def test_a_measurement_constructor_keys_on_the_observables(compiled_post82, kind, call):
    # The variant is the normalized tuple, not the argument: a request in a
    # different order resolves to the entry already built.
    built = call(compiled_post82, None)
    names = compiled_post82.observable_names

    assert _CFUNC_CACHE[_key(compiled_post82, kind, names)] is built
    assert call(compiled_post82, tuple(reversed(names))) is built


def test_an_observable_subset_gets_its_own_entry(compiled_post82):
    subset = compiled_post82.observable_names[:2]

    whole = compiled_post82.construct_measurement_cfunc(None)
    part = compiled_post82.construct_measurement_cfunc(subset)

    assert part is not whole
    assert _CFUNC_CACHE[_key(compiled_post82, "measurement", subset)] is part


def test_the_constraint_and_regime_kernels_are_stored(compiled_rbc_obc):
    constraint = compiled_rbc_obc.construct_constraint_func()
    pencil = compiled_rbc_obc.construct_regime_pencil_func()

    assert _CFUNC_CACHE[_key(compiled_rbc_obc, "constraint")] is constraint
    assert _CFUNC_CACHE[_key(compiled_rbc_obc, "regime")] is pencil


def test_a_repeated_construction_hands_back_the_same_object(compiled_post82):
    # Nothing outside the cache keeps these, so identity across calls is the
    # whole observable effect of a hit.
    assert (
        compiled_post82.construct_objective_cfunc()
        is compiled_post82.construct_objective_cfunc()
    )


def test_two_models_do_not_share_an_entry(compiled_post82, compiled_test):
    assert compiled_post82.config_token != compiled_test.config_token

    one = compiled_post82.construct_objective_cfunc()
    other = compiled_test.construct_objective_cfunc()

    assert one is not other
    assert _CFUNC_CACHE[_key(compiled_post82, "residual_c128")] is one
    assert _CFUNC_CACHE[_key(compiled_test, "residual_c128")] is other
