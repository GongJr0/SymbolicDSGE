"""The built-in TRANSFORM steps: numerics, rejections, factories, persistence.

The transform arithmetic lives in C and the library exports no per-step runner,
so the pipeline is the interface. One pipeline reaches all the numerics: a
producer per kind of input, then one transform per case reading a producer
directly. Nothing chains, which keeps every expected value a function of the
declared input rather than of an upstream step's parameters, and lets a
degenerate input be injected for the branches a well-behaved sample never
enters. See ``conftest.py`` for that fixture.

The failure cases build their own pipelines, since a rejection is the result.
The last test carries the whole thing through a bundle.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.monte_carlo import MCPipeline, OpType
from SymbolicDSGE.monte_carlo.step_factories import (
    add_payload_step,
    diff_step,
    log_diff_step,
    log_step,
    raw_model_data_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    standardize_step,
)

PERIODS = 20
COLUMNS = 2


def _trailing(sample, window, reduce):
    """The reduction over every trailing ``window``, as the steps align it."""
    return np.stack(
        [
            reduce(sample[i : i + window], axis=0)
            for i in range(sample.shape[0] - window + 1)
        ]
    )


# ---- numerics --------------------------------------------------------------


def test_every_declared_transform_retains_a_payload(transform_run) -> None:
    assert set(transform_run.out) == {
        "pos",
        "zed",
        "std",
        "std_zero_var",
        "log",
        "log_diff",
        "diff",
        "rmean",
        "rstd",
        "rvar",
    }


def test_injected_payload_round_trips(transform_run) -> None:
    np.testing.assert_allclose(transform_run.out["pos"], transform_run.positive)
    np.testing.assert_allclose(transform_run.out["zed"], transform_run.zeros)


def test_standardize_centers_and_scales_per_column(transform_run) -> None:
    sample = transform_run.observables
    np.testing.assert_allclose(
        transform_run.out["std"],
        (sample - sample.mean(axis=0)) / sample.std(axis=0, ddof=0),
    )


def test_standardize_zero_variance_column_returns_zeros(transform_run) -> None:
    # A constant column has no scale to divide by. The step returns zeros rather
    # than dividing and propagating NaN into everything downstream.
    got = transform_run.out["std_zero_var"]
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, np.zeros_like(got))


def test_log_applies_elementwise(transform_run) -> None:
    np.testing.assert_allclose(transform_run.out["log"], np.log(transform_run.positive))


def test_log_diff_is_the_difference_of_logs(transform_run) -> None:
    got = transform_run.out["log_diff"]
    assert got.shape == (transform_run.periods - 1, transform_run.columns)
    np.testing.assert_allclose(got, np.diff(np.log(transform_run.positive), axis=0))


def test_diff_applies_the_requested_order(transform_run) -> None:
    order = transform_run.order
    got = transform_run.out["diff"]
    assert got.shape == (transform_run.periods - order, transform_run.columns)
    np.testing.assert_allclose(got, np.diff(transform_run.observables, n=order, axis=0))


def test_rolling_mean_is_a_trailing_window(transform_run) -> None:
    window = transform_run.window
    got = transform_run.out["rmean"]
    assert got.shape == (transform_run.periods - window + 1, transform_run.columns)
    np.testing.assert_allclose(
        got, _trailing(transform_run.observables, window, np.mean)
    )


@pytest.mark.parametrize(
    "name, reduce",
    [
        ("rstd", lambda z, axis: z.std(axis=axis, ddof=0)),
        ("rvar", lambda z, axis: z.var(axis=axis, ddof=0)),
    ],
)
def test_rolling_dispersion_is_a_trailing_window(transform_run, name, reduce) -> None:
    window = transform_run.window
    got = transform_run.out[name]
    assert got.shape == (transform_run.periods - window + 1, transform_run.columns)
    np.testing.assert_allclose(
        got, _trailing(transform_run.observables, window, reduce)
    )


# ---- failure states --------------------------------------------------------
#
# A bad transform is rejected at one of three places, and which one is part of
# the contract: the factory validates what it can see on its own, ``MCPipeline``
# validates the graph, and the kernel reports the rest per replication. The
# cases below pin each tier, and the last two pin inputs that are *not* rejected
# so a change in that behaviour is visible rather than silent.


def _sample(periods: int = PERIODS, columns: int = COLUMNS) -> np.ndarray:
    return np.zeros((periods, columns), dtype=np.float64)


def _datagen(observables: np.ndarray | None = None):
    obs = _sample() if observables is None else observables
    return raw_model_data_step("dat", observables=obs, observable_names=("y", "x"))


@pytest.mark.parametrize(
    "steps, match",
    [
        (
            lambda: [
                _datagen(),
                standardize_step("s", source="absent", field="observables"),
            ],
            "unknown producer",
        ),
        (
            lambda: [
                _datagen(),
                standardize_step("dat", source="dat", field="observables"),
            ],
            "step names must be unique",
        ),
        (
            lambda: [standardize_step("s", source="dat", field="observables")],
            "unknown producer",
        ),
    ],
    ids=["unknown-producer", "duplicate-name", "no-datagen"],
)
def test_graph_faults_are_rejected_at_construction(steps, match) -> None:
    # No model and no run: these never reach the kernel.
    with pytest.raises(ValueError, match=match):
        MCPipeline(steps())


@pytest.mark.parametrize(
    "step",
    [
        lambda: rolling_mean_step(
            "bad", source="dat", field="observables", window=PERIODS + 5
        ),
        lambda: rolling_mean_step("bad", source="dat", field="observables", window=0),
        lambda: diff_step("bad", source="dat", field="observables", order=0),
        lambda: diff_step("bad", source="dat", field="observables", order=-1),
    ],
    ids=["window-over-length", "window-zero", "order-zero", "order-negative"],
)
def test_invalid_step_parameters_fail_in_the_run(
    mc_run, solved_test_model, step
) -> None:
    # These build and lower without complaint; the kernel rejects them per
    # replication. The status code is carried in the message and deliberately not
    # asserted via catching the `fail_fast` raise.
    pipeline = MCPipeline([_datagen(), step()])

    with pytest.raises(RuntimeError, match="'bad'"):
        mc_run(pipeline, solved_test_model, n_rep=1, fail_fast=True)


def test_run_failures_are_collected_per_replication_when_not_failing_fast(
    mc_run,
    solved_test_model,
) -> None:
    pipeline = MCPipeline(
        [
            _datagen(),
            rolling_mean_step(
                "bad", source="dat", field="observables", window=PERIODS + 5
            ),
        ]
    )

    result = mc_run(pipeline, solved_test_model, n_rep=3, fail_fast=False)

    assert result.n_successful == 0
    assert len(result.failures) == 3
    assert all("bad" in f.failures for f in result.failures)
    assert [f.rep_idx for f in result.failures] == [0, 1, 2]


def test_log_of_a_non_positive_sample_is_rejected(
    mc_run,
    solved_test_model,
) -> None:
    result = MCPipeline(
        [
            _datagen(),
            add_payload_step("nonpositive", -np.ones((PERIODS, COLUMNS))),
            log_step("log_bad", source="nonpositive", field="payload"),
        ]
    ).run(
        solved_test_model,
        n_rep=1,
        fail_fast=False,
        check_memory_availability=False,
    )

    assert not result.succeeded
    assert all("log_bad" in f.failures for f in result.failures)


def test_diff_order_at_the_sample_length_fails(mc_run, solved_test_model) -> None:
    result = MCPipeline(
        [
            _datagen(),
            diff_step("drained", source="dat", field="observables", order=PERIODS),
        ]
    ).run(
        solved_test_model,
        n_rep=1,
        fail_fast=False,
        check_memory_availability=False,
    )

    assert not result.succeeded
    assert all("drained" in f.failures for f in result.failures)


# ---- factories -------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, kwargs, runner_kwargs",
    [
        (standardize_step, {}, {"ddof": 0}),
        (log_step, {"offset": 1.0}, {"offset": 1.0}),
        (log_diff_step, {}, {"offset": 0.0}),
        (diff_step, {"order": 1}, {"order": 1}),
        (rolling_mean_step, {"window": 5}, {"window": 5}),
        (rolling_std_step, {"window": 5}, {"window": 5, "ddof": 0}),
        (rolling_var_step, {"window": 5}, {"window": 5, "ddof": 0}),
    ],
    ids=["standardize", "log", "log_diff", "diff", "rmean", "rstd", "rvar"],
)
def test_transform_factories_produce_a_bound_transform_step(
    factory, kwargs, runner_kwargs
) -> None:
    """Each factory yields a TRANSFORM step carrying only its runner kwargs.

    The source binding is separate from the runner configuration, so the kwargs
    the kernel reads must not pick up ``source`` or ``field``.
    """
    step = factory("step_a", source="datagen", field="observables", **kwargs)

    assert step.op_type is OpType.TRANSFORM
    assert step.name == "step_a"
    assert dict(step.kwargs) == runner_kwargs
    assert len(step.source_args) == 1


# ---- persistence -----------------------------------------------------------


def test_transform_pipeline_round_trips_through_bundle(tmp_path) -> None:
    """Authoring a transform-containing pipeline and re-opening it from a
    bundle preserves every node and its bound params."""
    import pathlib

    from SymbolicDSGE import BundleBuilder, load_bundle
    from SymbolicDSGE.monte_carlo import MCPipeline
    from SymbolicDSGE.ui.mc import build_pipeline as build_live_pipeline
    from tests._spec_helpers import as_posted

    yaml_text = pathlib.Path("MODELS/test.yaml").read_text(encoding="utf-8")
    # Authored the way the GUI posts it, then lowered through the UI boundary,
    # which is what resolves op kinds, source legs and the wald target field.
    pipeline = as_posted(
        {
            "nodes": [
                {
                    "id": "sim",
                    "step_type": "simulation",
                    "name": "datagen",
                    "params": {"T": 50},
                },
                {
                    "id": "std",
                    "step_type": "standardize",
                    "name": "standardize",
                    "params": {"source": "datagen", "field": "observables"},
                },
                {
                    "id": "rm",
                    "step_type": "rolling_mean",
                    "name": "rmean",
                    "params": {
                        "source": "standardize",
                        "field": "payload",
                        "window": 3,
                    },
                },
                {
                    "id": "wm",
                    "step_type": "wald",
                    "name": "wald_mean",
                    "params": {
                        "kind": "mean",
                        "source": "rmean",
                        "field": "payload",
                        "target_vector": [0.0],
                    },
                },
            ],
            "edges": [],
            "postprocs": [],
        }
    )

    target = (
        BundleBuilder(created_by="tx-test")
        .add_model("reference", yaml_text, compile_kwargs={})
        .add_mc(build_live_pipeline(pipeline))
        .write(tmp_path / "tx.sdsge")
    )

    loaded = load_bundle(target)
    assert loaded.mc is not None
    restored = loaded.mc.pipeline
    assert [step.step_type for step in restored.replication_steps] == [
        "simulation",
        "standardize",
        "rolling_mean",
        "wald",
    ]
    # The restored spec still rebuilds (no drift between the spec's Literal and
    # the catalog at load time).
    rebuilt = MCPipeline.from_spec(restored.to_spec())
    assert [step.name for step in rebuilt.replication_steps] == [
        "datagen",
        "standardize",
        "rmean",
        "wald_mean",
    ]
