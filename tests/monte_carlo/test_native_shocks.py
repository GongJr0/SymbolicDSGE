# type: ignore
"""The shock block the Monte Carlo loop draws for itself (#374).

The native draw replaces a prematerialized ``(n_rep, T, n_exog)`` slab, so the
two things worth pinning are that the deterministic half is exact and that the
stochastic half is addressable. Every draw here is recomputed independently
from :mod:`SymbolicDSGE._ckernels.rng`, so a covariance factorization, a
location shift, a uniform rescaling, or a column scatter that drifts shows up as
an exact mismatch rather than as a distributional one.

Specifications the kernel cannot reproduce stay on the Python route, and the
last tests here check that route is still chosen and still bit-identical to a
resolved :class:`ShockPlan`.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import Shock
from SymbolicDSGE._ckernels.monte_carlo._runner import run as run_native
from SymbolicDSGE._ckernels.rng import (
    philox_standard_normal,
    philox_standard_uniform,
)
from SymbolicDSGE.monte_carlo import MCPipeline, replication_shocks
from SymbolicDSGE.monte_carlo.native_lowering import lower_native_run
from SymbolicDSGE.monte_carlo.shock_native import (
    ShockCode,
    build_native_plan,
    native_shock_entries,
    native_shock_families,
    native_shock_scratch,
)
from SymbolicDSGE.monte_carlo.step_factories import simulation_step
from SymbolicDSGE.core.shock.spec import _normalized_spec, resolve_shock_plan

T = 16


def _plan(solved_test_model, shocks, shock_scale=1.0):
    step = simulation_step(T=T, shocks=shocks, shock_scale=shock_scale)
    return build_native_plan(solved_test_model, step, T)


def _entries(solved_test_model, shocks):
    families = native_shock_families(_normalized_spec(shocks))
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    return native_shock_entries(resolved, families)


# --- eligibility ------------------------------------------------------------


def test_native_families_accepts_normal_and_univariate_uniform() -> None:
    def families(spec):
        return native_shock_families(_normalized_spec(spec))

    assert families({("e_u",): Shock("norm", seed=0)}) == {("e_u",): ShockCode.NORMAL}
    assert families({("e_u", "e_v"): Shock("norm", seed=0)}) == {
        ("e_u", "e_v"): ShockCode.NORMAL
    }
    assert families({("e_u",): Shock("uni", seed=0)}) == {("e_u",): ShockCode.UNIFORM}


@pytest.mark.parametrize(
    "shocks",
    [
        {},
        {("e_u",): Shock("t", seed=0, dist_kwargs={"df": 5})},
        {("e_u", "e_v"): Shock("uni", seed=0)},
        {("e_u",): np.zeros(T)},
        # One ineligible entry sends the whole specification back.
        {
            ("e_u",): Shock("norm", seed=0),
            ("e_v",): Shock("t", seed=1, dist_kwargs={"df": 5}),
        },
    ],
)
def test_native_families_rejects_unported_specs(shocks) -> None:
    spec = _normalized_spec(shocks)
    assert native_shock_families(spec) == {}
    assert native_shock_scratch(spec, T) == 0


def test_native_scratch_sizes_on_the_widest_entry() -> None:
    wide = _normalized_spec({("e_u", "e_v"): Shock("norm", seed=0)})
    assert native_shock_scratch(wide, T) == T * 2
    narrow = _normalized_spec({("e_u",): Shock("norm", seed=0)})
    assert native_shock_scratch(narrow, T) == T


# --- the draw itself --------------------------------------------------------


def test_univariate_normal_draw_is_the_scaled_engine_stream(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(3)

    z = philox_standard_normal(entry.key, 0, 3, 0, T)
    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.columns[0]] = z * entry.factor[0, 0]

    np.testing.assert_array_equal(block, expected)


def test_multivariate_normal_draw_applies_the_covariance_factor(
    solved_test_model,
) -> None:
    shocks = {("e_u", "e_v"): Shock("norm", seed=11)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(2)

    width = len(entry.columns)
    z = philox_standard_normal(entry.key, 0, 2, 0, T * width).reshape(T, width)
    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.columns] = z @ entry.factor.T

    np.testing.assert_array_equal(block, expected)
    # The factor is the thing under test, so it must not be the identity.
    assert not np.allclose(entry.factor, np.eye(width))


def test_normal_draw_applies_the_location_shift(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7, dist_kwargs={"loc": 2.5})}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    z = philox_standard_normal(entry.key, 0, 0, 0, T)
    np.testing.assert_array_equal(
        block[:, entry.columns[0]], z * entry.factor[0, 0] + 2.5
    )


def test_shock_scale_multiplies_the_whole_block(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    plain = _plan(solved_test_model, shocks, shock_scale=1.0).draw(1)
    scaled = _plan(solved_test_model, shocks, shock_scale=2.5).draw(1)

    np.testing.assert_array_equal(scaled, 2.5 * plain)


def test_untargeted_columns_stay_zero(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    untargeted = [
        i
        for i in range(solved_test_model.compiled.n_exog)
        if i not in set(entry.columns)
    ]
    assert untargeted
    np.testing.assert_array_equal(block[:, untargeted], 0.0)


# --- addressing -------------------------------------------------------------


def test_a_seeded_spec_replays_across_plans(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=1), ("e_v",): Shock("uni", seed=2)}
    first = _plan(solved_test_model, shocks)
    second = _plan(solved_test_model, shocks)

    for rep_idx in (0, 1, 97):
        np.testing.assert_array_equal(first.draw(rep_idx), second.draw(rep_idx))


def test_replications_do_not_share_a_stream(solved_test_model) -> None:
    plan = _plan(solved_test_model, {("e_u", "e_v"): Shock("norm", seed=1)})
    blocks = [plan.draw(rep_idx) for rep_idx in range(4)]

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert not np.array_equal(blocks[i], blocks[j])


def test_entries_sharing_a_seed_stay_independent(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=5), ("e_v",): Shock("norm", seed=5)}
    entries = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    assert entries[0].key == entries[1].key
    left = block[:, entries[0].columns[0]]
    right = block[:, entries[1].columns[0]]
    assert not np.array_equal(left, right)


def test_an_unseeded_spec_redraws_each_run(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=None)}
    assert not np.array_equal(
        _plan(solved_test_model, shocks).draw(0),
        _plan(solved_test_model, shocks).draw(0),
    )


def test_negative_replication_index_is_rejected(solved_test_model) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _plan(solved_test_model, {("e_u",): Shock("norm", seed=0)}).draw(-1)


# --- the run reads the same blocks -----------------------------------------


def _run_states(solved_test_model, shocks, n_rep, n_jobs):
    """Simulate ``n_rep`` replications and return their retained state paths."""
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim", target="reference", T=T, shocks=shocks, observables=False
            )
        ]
    )
    lowered = lower_native_run(
        pipeline, models={"reference": solved_test_model}, n_rep=n_rep, n_jobs=n_jobs
    )
    assert (
        run_native(lowered.allocation, lowered.steps, lowered.input_bindings).status
        == 0
    )

    layout = lowered.plan["sim"].out_fields["states"]
    retained = lowered.allocation.steps["sim"].float_retained
    return retained[:, layout.offset : layout.offset + layout.flat_count].reshape(
        n_rep, *layout.shape
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("n_rep", [3, 8])
def test_run_states_match_the_addressed_blocks(
    solved_test_model, n_rep, n_jobs
) -> None:
    shocks = {("e_u", "e_v"): Shock("norm", seed=1)}
    states = _run_states(solved_test_model, shocks, n_rep, n_jobs)

    plan = _plan(solved_test_model, shocks)
    (entry,) = _entries(solved_test_model, shocks)
    for rep_idx in range(n_rep):
        block = plan.draw(rep_idx)[:, entry.columns]
        expected = solved_test_model.sim(T, shocks={("e_u", "e_v"): block}).X
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


# --- reproducing one replication --------------------------------------------


@pytest.mark.parametrize(
    "shocks",
    [
        {("e_u", "e_v"): Shock("norm", seed=1)},
        {("e_u",): Shock("norm", seed=1), ("e_v",): Shock("uni", seed=2)},
        # The Python fallback route.
        {("e_u",): Shock("t", seed=3, dist_kwargs={"df": 5})},
    ],
)
def test_replication_shocks_reproduce_a_single_replication(
    solved_test_model, shocks
) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=shocks)
    states = _run_states(solved_test_model, shocks, 5, 1)

    for rep_idx in (0, 2, 4):
        drawn = replication_shocks(solved_test_model, step, rep_idx)
        expected = solved_test_model.sim(T, shocks=drawn, shock_scale=1.0).X
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


def test_replication_shocks_rejects_a_deterministic_step(solved_test_model) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=None)
    with pytest.raises(ValueError, match="draws no shocks"):
        replication_shocks(solved_test_model, step, 0)


# --- the fallback route ------------------------------------------------------


def test_unported_spec_still_runs_off_the_python_slab(solved_test_model) -> None:
    shocks = {("e_u",): Shock("t", seed=3, dist_kwargs={"df": 5})}
    step = simulation_step(T=T, target="reference", shocks=shocks, observables=False)
    assert build_native_plan(solved_test_model, step, T) is None

    states = _run_states(solved_test_model, shocks, 3, 1)
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    for rep_idx in range(3):
        drawn = resolved.matrix(T, 1.0, rep_idx)
        expected = solved_test_model.sim(T, shocks={("e_u",): drawn[:, 0]}).X
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


# --- entry invariance, on both routes (#507) --------------------------------
#
# An entry's stream must be a function of the entry alone. The two routes key
# differently -- Philox on ``(seed, columns[0], rep_idx)`` in C, a mixed
# ``SeedSequence`` on the same triple in Python -- so they never produce equal
# numbers, but they owe the same invariances. Parameterizing on the family is
# what selects the route: ``norm`` lowers to the kernel, ``t`` does not, and
# ``_native_draw`` asserts the lowering actually happened rather than trusting it.


def _python_draw(solved_test_model, spec, rep_idx):
    return resolve_shock_plan(solved_test_model.compiled, spec, T).matrix(
        T, 1.0, rep_idx
    )


def _native_draw(solved_test_model, spec, rep_idx):
    plan = _plan(solved_test_model, spec)
    assert plan is not None, "spec was expected to lower to the native draw"
    return plan.draw(rep_idx)


ROUTES = [
    pytest.param("t", _python_draw, id="python"),
    pytest.param("norm", _native_draw, id="native"),
]


def _entry(family, seed, target):
    kwargs = {"df": 5} if family == "t" else {}
    return Shock(family, seed=seed, dist_kwargs=kwargs).independent(target)[0]


@pytest.mark.parametrize("family, draw", ROUTES)
def test_entry_draw_does_not_depend_on_its_position_in_the_spec(
    solved_test_model, family, draw
) -> None:
    """Reordering a spec must move nothing.

    This is the native half of #507: the Philox key carried the entry's position
    in the spec list, so every entry that moved drew differently even though the
    columns it writes are canonical. The Python route always held this.
    """
    spec = [_entry(family, 11, "e_u"), _entry(family, 12, "e_v")]

    np.testing.assert_array_equal(
        draw(solved_test_model, spec, 3),
        draw(solved_test_model, list(reversed(spec)), 3),
    )


@pytest.mark.parametrize("family, draw", ROUTES)
def test_entry_draw_does_not_depend_on_the_rest_of_the_spec(
    solved_test_model, family, draw
) -> None:
    """Adding a second seeded entry leaves the first untouched.

    The Python half of #507: the per-replication shift was the count of seeded
    entries, so adding one moved every other entry at every replication past
    zero. Native keyed per entry and already held this.
    """
    alone = [_entry(family, 11, "e_u")]
    joined = alone + [_entry(family, 12, "e_v")]
    col = solved_test_model.compiled.shock_idx["e_u"]

    np.testing.assert_array_equal(
        draw(solved_test_model, alone, 3)[:, col],
        draw(solved_test_model, joined, 3)[:, col],
    )


@pytest.mark.parametrize("family, draw", ROUTES)
def test_identically_seeded_copies_are_not_degenerate(
    solved_test_model, family, draw
) -> None:
    """Copies sharing one seed still draw independently.

    ``offset_seeds=False`` hands every copy the template's seed. On the Python
    route that used to give one standardized variate scaled per column, a
    rank-deficient block whatever ``shock_corr`` declared. Rank is scale-free,
    so this reads the same on either route.
    """
    kwargs = {"df": 5} if family == "t" else {}
    spec = Shock(family, seed=5, dist_kwargs=kwargs).independent(
        "e_u", "e_v", offset_seeds=False
    )
    assert [shock.seed for shock in spec] == [5, 5]

    assert np.linalg.matrix_rank(draw(solved_test_model, spec, 0)) == len(spec)
