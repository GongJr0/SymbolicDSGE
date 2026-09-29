from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE._diag_tests.breusch_godfrey import breusch_godfrey
from SymbolicDSGE._diag_tests.breusch_pagan import (
    breusch_pagan,
    robust_breusch_pagan,
)
from SymbolicDSGE._diag_tests.chow import chow
from SymbolicDSGE._diag_tests.cusum import cusum
from SymbolicDSGE._diag_tests.cusumsq import cusumsq_test
from SymbolicDSGE._diag_tests.jarque_bera import jarque_bera
from SymbolicDSGE._diag_tests.ljung_box import ljung_box
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE._diag_tests.wald_test import wald_mean_hac
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.monte_carlo import MCPipeline, MCStep, OpType
from SymbolicDSGE.monte_carlo.allocation import (
    ArenaSize,
    FieldLayout,
    StepBufferPlan,
    resolve_output_specs,
)
from SymbolicDSGE.monte_carlo.mc_constructs import (
    report_mc_performance,
    report_mc_step_performance,
)
from SymbolicDSGE.monte_carlo.step_factories import (
    add_payload_step,
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    log_diff_step,
    raw_model_data_step,
    filter_step,
    regression_step,
    wald_test_step,
)

from SymbolicDSGE.regression.enums import RegressionStatus


def _assert_output_plan(
    plan: StepBufferPlan,
    name: str,
    *fields: tuple[str, tuple[int, ...], object],
) -> None:
    """Assert a step's output layout from the fields that belong to it.

    The int lane every native output carries is closed over here, the way
    ``_with_int_flags`` closes it over in the resolvers, so a call site states
    only its own fields and the two flags stay stated in one place.
    """
    named = [field_name for field_name, _, _ in fields]
    assert not {"has_failed_sources", "status"}.intersection(
        named
    ), "The runner's int lane is supplied here, not named by the call site."

    float_offset = 0
    int_offset = 0
    layouts: dict[str, FieldLayout] = {}
    for field_name, shape, dtype in (
        *fields,
        ("has_failed_sources", (), np.int64),
        ("status", (), np.int64),
    ):
        flat_count = int(np.prod(shape, dtype=np.intp)) if shape else 1
        # The layout carries the bare type the lane is keyed on, not a dtype
        # instance. They compare equal but hash apart, so spelling it the way
        # the planner spells it keeps the comparison an exact one.
        if dtype is np.float64:
            offset = float_offset
            float_offset += flat_count
        else:
            assert dtype is np.int64
            offset = int_offset
            int_offset += flat_count
        layouts[field_name] = FieldLayout(shape, flat_count, dtype, offset)
    assert plan.name == name
    assert plan.output_size == ArenaSize(float_offset, int_offset)
    assert plan.out_fields == layouts
    assert plan.n_retain == -1


def _quadratic_sample() -> np.ndarray:
    return np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
                [1.5, -0.5],
                [-2.0, 1.0],
                [0.5, 3.0],
            ],
            dtype=np.float64,
        )
    )


def _batched_states() -> np.ndarray:
    return _quadratic_sample()


def test_raw_model_data_pipeline_runs_without_dgp_and_aggregates_wald_results(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    states = _batched_states()
    target = np.zeros(2, dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(states=states),
            wald_test_step(
                "state_mean",
                source="datagen",
                field="states",
                target=target,
                bandwidth=0,
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=2)

    statistic = wald_mean_hac(states, target, bandwidth=0).statistic
    expected = np.full(2, statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["state_mean"], expected)
    assert out.test_summaries["state_mean"].n_retained == 2
    np.testing.assert_allclose(
        out.pval_traces["state_mean"],
        out.test_summaries["state_mean"].pval_trace,
    )


def test_raw_model_data_pipeline_accepts_observables_without_states(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    observables = _batched_states()[:, :1]
    target = np.zeros(1, dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("obs",)),
            wald_test_step(
                "obs_mean",
                source="datagen",
                field="observables",
                target=target,
                bandwidth=0,
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=2)

    statistic = wald_mean_hac(observables, target, bandwidth=0).statistic
    expected = np.full(2, statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["obs_mean"], expected)


def test_ljung_box_pipeline_selects_column_and_aggregates_results(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    first = np.column_stack(
        [
            np.array([1.0, 2.0, 0.0, 4.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.5, -1.0, 2.0], dtype=np.float64),
        ]
    )
    observables = first
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("a", "b")),
            ljung_box_test_step(
                "lb_b",
                source="datagen",
                field="observables",
                column=[1],
                lags=2,
                alpha=0.1,
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=2)

    statistic = ljung_box(observables[:, 1], L=2, alpha=0.1).statistic
    expected = np.full(2, statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["lb_b"], expected)
    assert out.test_summaries["lb_b"].n_retained == 2
    assert out.test_summaries["lb_b"].df == np.float64(2.0)
    assert out.test_summaries["lb_b"].status_trace == (TestStatus.OK,) * 2


def test_ljung_box_pipeline_rejects_multi_column_inputs(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    observables = np.array([[1.0, 2.0]], dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("a", "b")),
            ljung_box_test_step(
                "lb",
                source="datagen",
                field="observables",
                lags=1,
            ),
        ]
    )

    with pytest.raises(ValueError, match="single-column source"):
        mc_run(pipeline, reference, n_rep=1)


def test_jarque_bera_pipeline_selects_column_and_aggregates_results(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    base = np.column_stack(
        [
            np.linspace(-2.0, 3.0, 12, dtype=np.float64) ** 2,
            np.linspace(1.0, 4.0, 12, dtype=np.float64),
        ]
    )
    observables = base
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("a", "b")),
            jarque_bera_test_step(
                "jb_a",
                source="datagen",
                field="observables",
                column=0,
                alpha=0.1,
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=2, verbosity=0)

    statistic = jarque_bera(observables[:, 0], alpha=0.1).statistic
    expected = np.full(2, statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["jb_a"], expected)
    assert out.succeeded
    assert out.test_summaries["jb_a"].df == 12
    assert out.test_summaries["jb_a"].status_trace == (TestStatus.OK,) * 2


def test_jarque_bera_pipeline_rejects_multi_column_inputs(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    observables = np.arange(24.0, dtype=np.float64).reshape(12, 2)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            jarque_bera_test_step("jb", source="datagen", field="observables"),
        ]
    )

    with pytest.raises(ValueError, match="single-column source"):
        mc_run(pipeline, reference, n_rep=1, verbosity=0)


def test_breusch_pagan_pipeline_selects_columns_and_aggregates_results(
    mc_run, solved_test_model
) -> None:
    rng = np.random.default_rng(512)
    X = rng.normal(size=(40, 2))
    eps = rng.normal(scale=np.exp(0.4 * X[:, 0]))
    observables = np.column_stack((eps, X))
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            breusch_pagan_test_step(
                "bp",
                residuals_source="datagen",
                residuals_field="observables",
                X_source="datagen",
                X_field="observables",
                residual_col=0,
                X_columns=[1, 2],
            ),
            breusch_pagan_test_step(
                "robust_bp",
                residuals_source="datagen",
                residuals_field="observables",
                X_source="datagen",
                X_field="observables",
                residual_col=0,
                X_columns=[1, 2],
                robust=True,
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, breusch_pagan(eps, X).statistic, dtype=np.float64)
    robust_expected = np.full(
        2, robust_breusch_pagan(eps, X).statistic, dtype=np.float64
    )
    np.testing.assert_allclose(out.statistic_traces["bp"], expected)
    np.testing.assert_allclose(out.statistic_traces["robust_bp"], robust_expected)
    assert out.test_summaries["bp"].df == 2
    assert out.test_summaries["robust_bp"].df == 2
    assert out.test_status_traces["bp"] == (TestStatus.OK, TestStatus.OK)
    assert out.test_status_traces["robust_bp"] == (TestStatus.OK, TestStatus.OK)


def test_breusch_pagan_pipeline_supports_separate_residual_and_regressor_sources(
    mc_run, solved_test_model
) -> None:
    rng = np.random.default_rng(128)
    states = rng.normal(size=(30, 2))
    residuals = rng.normal(scale=np.exp(0.4 * states[:, 0]))
    observables = residuals[:, None]
    pipeline = MCPipeline(
        [
            raw_model_data_step(states=states, observables=observables),
            breusch_pagan_test_step(
                "bp",
                residuals_source="datagen",
                residuals_field="observables",
                X_source="datagen",
                X_field="states",
                residual_col=0,
                X_columns=[0, 1],
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, breusch_pagan(residuals, states).statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["bp"], expected)


def test_breusch_pagan_pipeline_supports_separate_payload_sources(
    mc_run, solved_test_model
) -> None:
    rng = np.random.default_rng(256)
    X = rng.normal(size=(30, 2))
    residuals = rng.normal(scale=np.exp(0.4 * X[:, 0]), size=30)

    # Constant inputs are injected, not computed: ``transform_step`` is for a
    # function of a source, and ``add_payload_step`` is for data the author
    # supplies. The point here is that the two legs can name different producers.
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=residuals[:, None]),
            add_payload_step("residual_payload", residuals),
            add_payload_step("regressor_payload", X),
            breusch_pagan_test_step(
                "bp",
                residuals_source="residual_payload",
                residuals_field="payload",
                X_source="regressor_payload",
                X_field="payload",
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    np.testing.assert_allclose(
        out.statistic_traces["bp"],
        np.full(2, breusch_pagan(residuals, X).statistic, dtype=np.float64),
    )


def test_add_payload_step_registers_1d_payload_for_downstream_steps(
    mc_run, solved_test_model
) -> None:
    payload = np.asarray([1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    target = np.asarray([0.0], dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=np.zeros((payload.size, 1))),
            add_payload_step("external", payload),
            wald_test_step(
                "payload_mean",
                source="external",
                field="payload",
                target=target,
                bandwidth=0,
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2)

    expected = wald_mean_hac(payload.reshape(-1, 1), target, bandwidth=0).statistic
    np.testing.assert_allclose(
        out.statistic_traces["payload_mean"],
        np.full(2, expected, dtype=np.float64),
    )
    got = np.asarray(out.transform_outputs["external"])
    np.testing.assert_allclose(got[0].reshape(-1), payload)


def test_add_payload_step_registers_2d_payload_with_column_selection(
    mc_run, solved_test_model
) -> None:
    payload = np.column_stack(
        [
            np.random.default_rng(0).normal(size=30),
            np.random.default_rng(1).normal(size=30),
        ]
    )
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=np.zeros((payload.shape[0], 1))),
            add_payload_step("external", payload),
            jarque_bera_test_step(
                "payload_jb",
                source="external",
                field="payload",
                column=1,
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2)

    expected = jarque_bera(payload[:, 1]).statistic
    np.testing.assert_allclose(
        out.statistic_traces["payload_jb"],
        np.full(2, expected, dtype=np.float64),
    )


def test_add_payload_step_selects_batched_payload_by_replication(
    mc_run, solved_test_model
) -> None:
    payload = np.arange(12.0, dtype=np.float64).reshape(2, 3, 2)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=np.zeros((3, 1))),
            add_payload_step("cube", payload),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2)

    np.testing.assert_allclose(out.transform_outputs["cube"], payload)


def test_breusch_pagan_pipeline_validates_residual_and_regressor_inputs(
    mc_run, solved_test_model
) -> None:
    observables = np.arange(150.0, dtype=np.float64).reshape(50, 3)
    reference = solved_test_model

    with pytest.raises(ValueError, match="single-column source"):
        MCPipeline(
            [
                raw_model_data_step(observables=observables),
                breusch_pagan_test_step(
                    "bp",
                    residuals_source="datagen",
                    residuals_field="observables",
                    X_source="datagen",
                    X_field="observables",
                    X_columns=[1, 2],
                ),
            ]
        ).run(
            {"reference": reference},
            n_rep=1,
            n_jobs=1,
            verbosity=0,
            check_memory_availability=False,
        )

    degenerate = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            breusch_pagan_test_step(
                "bp",
                residuals_source="datagen",
                residuals_field="observables",
                X_source="datagen",
                X_field="observables",
                residual_col=0,
                X_columns=[],
            ),
        ]
    ).run(
        {"reference": reference},
        n_rep=1,
        check_memory_availability=False,
        fail_fast=False,
    )
    np.isnan(degenerate.statistic_traces["bp"]).all()
    assert degenerate.test_summaries["bp"].status_trace == (
        TestStatus.INSUFFICIENT_SAMPLES,
    )

    with pytest.raises(ValueError, match="matching row counts"):
        MCPipeline(
            [
                raw_model_data_step(
                    states=np.arange(33.0, dtype=np.float64).reshape(11, 3),
                    observables=observables,
                ),
                breusch_pagan_test_step(
                    "bp",
                    residuals_source="datagen",
                    residuals_field="observables",
                    X_source="datagen",
                    X_field="states",
                    residual_col=0,
                    X_columns=[0, 1],
                ),
            ]
        ).run(
            {"reference": reference},
            n_rep=1,
            n_jobs=1,
            verbosity=0,
            check_memory_availability=False,
        )


def test_breusch_godfrey_pipeline_selects_columns_and_aggregates_results(
    mc_run, solved_test_model
) -> None:
    rng = np.random.default_rng(512)
    X = rng.normal(size=(40, 2))
    eps = rng.normal(size=40)
    observables = np.column_stack((eps, X))
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            breusch_godfrey_test_step(
                "bg",
                residuals_source="datagen",
                residuals_field="observables",
                X_source="datagen",
                X_field="observables",
                residual_col=0,
                X_columns=[1, 2],
                lags=2,
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, breusch_godfrey(eps, X, lags=2).statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["bg"], expected)
    assert out.test_summaries["bg"].df == 2
    assert out.test_status_traces["bg"] == (TestStatus.OK, TestStatus.OK)


def test_breusch_godfrey_pipeline_validates_residual_and_regressor_inputs(
    solved_test_model,
) -> None:
    observables = np.arange(30.0, dtype=np.float64).reshape(10, 3)
    reference = solved_test_model

    with pytest.raises(ValueError, match="single-column source"):
        MCPipeline(
            [
                raw_model_data_step(observables=observables),
                breusch_godfrey_test_step(
                    "bg",
                    residuals_source="datagen",
                    residuals_field="observables",
                    X_source="datagen",
                    X_field="observables",
                    X_columns=[1, 2],
                ),
            ]
        ).run(
            {"reference": reference},
            n_rep=1,
            n_jobs=1,
            verbosity=0,
            check_memory_availability=False,
        )

    with pytest.raises(ValueError, match="matching row counts"):
        MCPipeline(
            [
                raw_model_data_step(
                    states=np.arange(33.0, dtype=np.float64).reshape(11, 3),
                    observables=observables,
                ),
                breusch_godfrey_test_step(
                    "bg",
                    residuals_source="datagen",
                    residuals_field="observables",
                    X_source="datagen",
                    X_field="states",
                    residual_col=0,
                    X_columns=[0, 1],
                ),
            ]
        ).run(
            {"reference": reference},
            n_rep=1,
            n_jobs=1,
            verbosity=0,
            check_memory_availability=False,
        )


def test_cusum_pipeline_aggregates_results_with_nan_df(
    mc_run, solved_test_model
) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    X[:, 0] = 1.0  # constant column for a well-posed recursion
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=60)
    observables = np.column_stack((y, X))
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            cusum_test_step(
                "cs",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2],
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, cusum(y, X).statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["cs"], expected)
    assert out.test_status_traces["cs"] == (TestStatus.OK, TestStatus.OK)
    # CUSUM is parameter-free: the NaN df placeholder must survive aggregation
    # across replications (the metadata-equality check is NaN-aware).
    assert np.isnan(out.test_summaries["cs"].df)


def test_cusumsq_pipeline_aggregates_results(mc_run, solved_test_model) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    X[:, 0] = 1.0  # constant column for a well-posed recursion
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=60)
    observables = np.column_stack((y, X))
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            cusumsq_test_step(
                "csq",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2],
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, cusumsq_test(y, X).statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["csq"], expected)
    assert out.test_status_traces["csq"] == (TestStatus.OK, TestStatus.OK)
    # CUSUMSQ is parameterized by the recursive-residual count n = T - p, which
    # is identical across equal-shape replications and survives aggregation as
    # the shared df.
    assert out.test_summaries["csq"].df == 60 - 2


def test_chow_pipeline_aggregates_results(mc_run, solved_test_model) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    X[:, 0] = 1.0  # constant column for a well-posed partition
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=60)
    observables = np.column_stack((y, X))
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            chow_test_step(
                "ch",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2],
                t_break=30,
            ),
        ]
    )

    out = mc_run(pipeline, solved_test_model, n_rep=2, verbosity=0)

    expected = np.full(2, chow(y, X, t_break=30).statistic, dtype=np.float64)
    np.testing.assert_allclose(out.statistic_traces["ch"], expected)
    assert out.test_status_traces["ch"] == (TestStatus.OK, TestStatus.OK)
    # Chow uses an F reference with df = (p, T - 2p), identical across
    # equal-shape replications and preserved through aggregation.
    assert out.test_summaries["ch"].df == (2, 60 - 2 * 2)


def test_raw_model_data_pipeline_rejects_empty_raw_model_data(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    pipeline = MCPipeline(
        [
            raw_model_data_step(),
            wald_test_step(
                "obs_mean",
                source="datagen",
                field="observables",
                target=np.zeros(1, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="does not produce source field"):
        mc_run(pipeline, reference, n_rep=1)


def test_pipeline_result_reports_overall_and_step_performance(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_model_data_step(states=states),
            wald_test_step(
                "state_mean",
                source="datagen",
                field="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    # Per-step timings are collected only at the highest verbosity, which is
    # what the step-level assertions below read.
    out = mc_run(pipeline, reference, n_rep=2, verbosity=2)

    assert out.meta.elapsed_s >= 0.0
    assert out.meta.it_s > 0.0
    assert set(out.meta.step_elapsed_s) == {"datagen", "state_mean"}
    assert out.meta.step_counts == {"datagen": 2, "state_mean": 2}
    assert out.meta.step_failures == {"datagen": 0, "state_mean": 0}
    assert set(out.meta.step_it_s) == {"datagen", "state_mean"}
    assert set(out.meta.step_worker_it_s) == {"datagen", "state_mean"}
    assert set(out.meta.step_wall_it_s) == {"datagen", "state_mean"}

    lines: list[str] = []
    report_mc_performance(out.meta, print_func=lines.append)
    assert lines[0].startswith("MC run concluded successfully in ")
    assert lines[0].endswith("it/s.")

    lines.clear()
    out.report_step_performance(print_func=lines.append)
    # Overall header, then an indented worker and wall it/s line per step. The
    # post-processing section is suppressed because this pipeline has no
    # postproc steps.
    assert lines[0].startswith("MC run concluded successfully in ")
    assert any("datagen" in line and "wall it/s." in line for line in lines)
    assert any("state_mean" in line and "wall it/s." in line for line in lines)
    assert not any("Post-processing Report" in line for line in lines)

    lines.clear()
    report_mc_step_performance(out.meta, print_func=lines.append)
    assert lines[0].startswith("MC run concluded successfully in ")


def test_pipeline_run_verbosity_controls_performance_output(
    mc_run,
    solved_test_model,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = solved_test_model
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_model_data_step(states=states),
            wald_test_step(
                "state_mean",
                source="datagen",
                field="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    # Every level is stated here rather than relying on a default: the shared
    # ``mc_run`` fixture runs quiet, while the library's own default is 1.
    mc_run(pipeline, reference, n_rep=2, verbosity=1)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("MC run concluded successfully in ")

    mc_run(pipeline, reference, n_rep=2, verbosity=0)
    assert capsys.readouterr().out == ""

    mc_run(pipeline, reference, n_rep=2, verbosity=2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("MC run concluded successfully in ")
    assert any("datagen" in line and "wall it/s." in line for line in lines)
    assert any("state_mean" in line and "wall it/s." in line for line in lines)
    assert not any("Post-processing Report" in line for line in lines)

    with pytest.raises(ValueError, match="verbosity"):
        mc_run(pipeline, reference, n_rep=2, verbosity=3)


def test_output_shape_resolution_tracks_selected_transform_payloads(
    solved_test_model,
) -> None:
    observables = np.zeros((2, 8, 3), dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            log_diff_step(
                "growth",
                source="datagen",
                field="observables",
                columns=[1, 2],
                burn_in=1,
            ),
            regression_step(
                "ols",
                y_source="growth",
                y_field="payload",
                y_column=0,
                X_source="growth",
                X_field="payload",
                X_columns=1,
            ),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    # A datagen step allocates all three of its fields; the ones its author did
    # not supply are zero-width rather than absent.
    _assert_output_plan(
        specs["datagen"],
        "datagen",
        ("states", (0, 0), np.float64),
        ("shocks", (0, 0), np.float64),
        ("observables", (8, 3), np.float64),
    )
    _assert_output_plan(specs["growth"], "growth", ("payload", (6, 2), np.float64))
    _assert_output_plan(
        specs["ols"],
        "ols",
        ("coef", (2,), np.float64),
        ("ssr", (), np.float64),
        ("sst", (), np.float64),
        ("se", (2,), np.float64),
    )
    assert specs["datagen"].input_size == ArenaSize()
    assert specs["growth"].input_size == ArenaSize(16, 0)
    assert specs["ols"].input_size == ArenaSize(30, 0)


@pytest.mark.parametrize(
    "kind",
    ("ridge", "lasso", "elastic_net", "ridge_gs", "lasso_gs", "elastic_net_gs"),
)
def test_output_specs_for_non_ols_regressions(solved_test_model, kind: str) -> None:
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=np.zeros((8, 2), dtype=np.float64)),
            regression_step(
                kind,
                y_source="datagen",
                y_field="observables",
                y_column=0,
                X_source="datagen",
                X_field="observables",
                X_columns=1,
                kind=kind,
            ),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    _assert_output_plan(
        specs[kind],
        kind,
        ("coef", (2,), np.float64),
        ("ssr", (), np.float64),
        ("sst", (), np.float64),
        # Non-OLS kinds report no standard errors, and the slot is allocated
        # zero-width rather than omitted.
        ("se", (0,), np.float64),
    )


@pytest.mark.parametrize("filter_mode", ("linear", "extended"))
def test_output_shape_resolution_includes_linear_filter_fields(
    solved_test_model, filter_mode: str
) -> None:
    """A filter step's fields are sized off the model, not off its input.

    Every shape below is derived from the solved model rather than written as a
    literal: the state width, the number of observables the step selects, and the
    shock count all come from the reference, which is the thing being resolved
    against.
    """
    compiled = solved_test_model.compiled
    names = tuple(compiled.observable_names)
    periods = 8
    observables = np.zeros((2, periods, len(names)), dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=names),
            filter_step(
                target="reference",
                obs_source="datagen",
                obs_field="observables",
                filter_mode=filter_mode,
                observables=list(names),
                return_shocks=True,
            ),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    # The filter carries the full variable vector, not just the predetermined
    # states, so its state width is ``n_var``.
    n_state = compiled.n_var
    n_obs = len(names)
    _assert_output_plan(
        specs["filter"],
        "filter",
        ("x_pred", (periods, n_state), np.float64),
        ("x_filt", (periods, n_state), np.float64),
        ("P_pred", (periods, n_state, n_state), np.float64),
        ("P_filt", (periods, n_state, n_state), np.float64),
        ("y_pred", (periods, n_obs), np.float64),
        ("y_filt", (periods, n_obs), np.float64),
        ("innov", (periods, n_obs), np.float64),
        ("std_innov", (periods, n_obs), np.float64),
        ("S", (periods, n_obs, n_obs), np.float64),
        ("eps_hat", (periods, compiled.n_exog), np.float64),
        ("loglik", (), np.float64),
    )


def test_output_shape_resolution_includes_scalar_test_channels(
    solved_test_model,
) -> None:
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                observables=np.zeros((8, 1), dtype=np.float64),
                observable_names=("obs",),
            ),
            jarque_bera_test_step("jb", source="datagen", field="observables"),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    _assert_output_plan(
        specs["jb"],
        "jb",
        ("statistic", (), np.float64),
    )


def test_output_shape_resolution_normalizes_payload_values_to_source_shapes(
    solved_test_model,
) -> None:
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=np.zeros((4, 1), dtype=np.float64)),
            add_payload_step("vector", np.arange(4.0, dtype=np.float64)),
            add_payload_step("matrix", np.zeros((4, 2), dtype=np.float64)),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    _assert_output_plan(specs["vector"], "vector", ("payload", (4, 1), np.float64))
    _assert_output_plan(specs["matrix"], "matrix", ("payload", (4, 2), np.float64))


def test_output_plan_carries_step_retention_count(solved_test_model) -> None:
    datagen = replace(
        raw_model_data_step(observables=np.zeros((4, 1), dtype=np.float64)),
        n_retain=2,
    )
    pipeline = MCPipeline([datagen])

    specs = resolve_output_specs(
        pipeline.replication_steps,
        pipeline._source_indices,
        {"reference": solved_test_model},
    )

    assert specs["datagen"].n_retain == 2


def test_output_shape_resolution_includes_unscented_filter_fields(
    solved_rbc_second_order,
) -> None:
    reference = solved_rbc_second_order
    compiled = reference.compiled

    # The state paths span the compiled layout, generated variables included.
    n_var = len(compiled.var_names)
    T = 6
    n_obs = len(reference.compiled.observable_names)
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                observables=np.zeros((T, n_obs), dtype=np.float64),
                observable_names=reference.compiled.observable_names,
            ),
            filter_step(
                target="reference",
                obs_source="datagen",
                obs_field="observables",
                filter_mode="unscented",
            ),
        ]
    )

    specs = resolve_output_specs(
        pipeline.replication_steps, pipeline._source_indices, {"reference": reference}
    )
    n_state = reference.compiled.n_state
    n_z = 2 * n_state

    _assert_output_plan(
        specs["filter"],
        "filter",
        ("x_pred", (T, n_var), np.float64),
        ("x_filt", (T, n_var), np.float64),
        ("P_pred", (T, n_z, n_z), np.float64),
        ("P_filt", (T, n_z, n_z), np.float64),
        ("y_pred", (T, n_obs), np.float64),
        ("y_filt", (T, n_obs), np.float64),
        ("innov", (T, n_obs), np.float64),
        ("std_innov", (T, n_obs), np.float64),
        ("S", (T, n_obs, n_obs), np.float64),
        ("loglik", (), np.float64),
        ("x1_pred", (T, n_state), np.float64),
        ("x2_pred", (T, n_state), np.float64),
        ("x1_filt", (T, n_state), np.float64),
        ("x2_filt", (T, n_state), np.float64),
    )


def test_regression_step_runs_ols_and_stores_result_payload(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    x = np.arange(1.0, 7.0, dtype=np.float64)
    y = 2.5 * x
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("y", "x")),
            regression_step(
                "ols",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=1)

    summary = out.regression_summaries["ols"]
    assert out.test_summaries == {}
    assert summary.kind == "ols"
    assert summary.variables == ("Intercept", "x")
    assert summary.n == x.size and summary.k == 2
    np.testing.assert_allclose(
        out.coefficient_traces["ols"],
        np.array([[0.0, 2.5]], dtype=np.float64),
        atol=1e-12,
    )
    assert summary.status_trace == (RegressionStatus.OK,)


def test_regression_step_runs_ridge_kind_and_aggregates_summary(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    x = np.arange(1.0, 7.0, dtype=np.float64)
    y = 1.0 + 2.0 * x
    alpha = np.float64(0.5)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("y", "x")),
            regression_step(
                "ridge",
                kind="ridge",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
                alpha=alpha,
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=1)

    X = np.column_stack([np.ones_like(x), x])
    G = (X.T @ X) / X.shape[0]
    g = (X.T @ y) / X.shape[0]
    expected_coef = np.linalg.solve(
        G + np.diag([0.0, alpha]),
        g,
    )
    summary = out.regression_summaries["ridge"]
    assert summary.variables == ("Intercept", "x")
    np.testing.assert_allclose(out.coefficient_traces["ridge"], expected_coef[None, :])
    # ``r2`` is reported off the same run's residual and total sums.
    np.testing.assert_allclose(
        summary.r2_trace, 1.0 - summary.ssr_trace / summary.sst_trace
    )
    # Standard errors are an OLS-only channel. A non-OLS kind does not refuse
    # the request, it warns and hands back NaN of the right shape.
    with pytest.warns(UserWarning, match="unbiased standard errors"):
        se = summary.se_trace
    assert se.shape == (summary.n_retained, summary.k)
    assert np.isnan(se).all()


def test_regression_step_runs_lasso_kind_and_aggregates_summary(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            regression_step(
                "lasso",
                kind="lasso",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2, 3],
                intercept=False,
                alpha=np.float64(0.5),
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=1)

    summary = out.regression_summaries["lasso"]
    # The soft-threshold solution for this design is sparse in two of three.
    np.testing.assert_allclose(
        out.coefficient_traces["lasso"],
        np.array([[1.5, 0.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        summary.r2_trace, 1.0 - summary.ssr_trace / summary.sst_trace
    )


def test_regression_step_runs_elastic_net_kind_and_aggregates_summary(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            regression_step(
                "elastic_net",
                kind="elastic_net",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2, 3],
                intercept=False,
                alpha=np.float64(0.5),
                l1_ratio=np.float64(0.5),
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=1)

    expected_coef = np.array([9.0 / 7.0, -1.0 / 7.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(
        out.coefficient_traces["elastic_net"], expected_coef[None, :]
    )
    assert out.regression_summaries["elastic_net"].status_trace == (
        RegressionStatus.OK,
    )


def test_regression_step_runs_elastic_net_grid_search_kind(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    x = np.eye(2, dtype=np.float64)
    y = np.array([3.0, -1.0], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            regression_step(
                "elastic_net_gs",
                kind="elastic_net_gs",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=0,
                X_columns=[1, 2],
                intercept=False,
                start=np.float64(0.5),
                stop=np.float64(2.0),
                num=3,
                l1_ratio=np.float64(0.5),
            ),
        ]
    )

    out = mc_run(pipeline, reference, n_rep=1)

    assert out.coefficient_traces["elastic_net_gs"].shape == (1, 2)
    assert out.regression_summaries["elastic_net_gs"].status_trace == (
        RegressionStatus.OK,
    )


def test_regression_step_requires_single_response_column(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    observables = np.column_stack(
        [
            np.arange(1.0, 6.0, dtype=np.float64),
            np.arange(2.0, 7.0, dtype=np.float64),
            np.arange(3.0, 8.0, dtype=np.float64),
        ]
    )
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables),
            regression_step(
                "ols",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="observables",
                y_column=[0, 1],
                X_columns=[2],
            ),
        ]
    )

    with pytest.raises(ValueError, match="response must resolve to one column"):
        mc_run(pipeline, reference, n_rep=1)


def test_regression_step_requires_matching_row_counts(
    mc_run, solved_test_model
) -> None:
    reference = solved_test_model
    states = np.arange(10.0, dtype=np.float64).reshape(5, 2)
    observables = np.arange(4.0, dtype=np.float64).reshape(4, 1)
    pipeline = MCPipeline(
        [
            raw_model_data_step(states=states, observables=observables),
            regression_step(
                "ols",
                y_source="datagen",
                y_field="observables",
                X_source="datagen",
                X_field="states",
                y_column=0,
                X_columns=[0],
            ),
        ]
    )

    with pytest.raises(ValueError, match="same number of rows"):
        mc_run(pipeline, reference, n_rep=1)


def test_pipeline_validates_step_order_and_unique_names() -> None:
    with pytest.raises(ValueError, match="unknown producer"):
        MCPipeline(
            [
                wald_test_step(
                    "state_mean",
                    source="datagen",
                    field="states",
                    target=np.zeros(2, dtype=np.float64),
                )
            ]
        )

    with pytest.raises(ValueError, match="unique"):
        MCPipeline(
            [
                raw_model_data_step(name="dup", states=_batched_states()),
                MCStep(
                    name="dup",
                    op_type=OpType.TRANSFORM,
                    func=lambda **_: np.zeros((1, 1), dtype=np.float64),
                ),
            ]
        )
