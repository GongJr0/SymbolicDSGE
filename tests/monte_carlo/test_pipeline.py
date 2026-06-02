from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE import Shock
from SymbolicDSGE._diag_tests.ljung_box import ljung_box
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE._diag_tests.wald_test import wald_mean_hac
from SymbolicDSGE.kalman.filter import FilterResult
from SymbolicDSGE.monte_carlo import (
    MCPipeline,
    MCData,
    MCStep,
    OpType,
    ljung_box_test_step,
    raw_data_step,
    reference_filter_step,
    regression_step,
    report_mc_performance,
    report_mc_step_performance,
    simulation_step,
    transform_step,
    wald_test_step,
)
from SymbolicDSGE.regression.ols import OLSResult
from SymbolicDSGE.regression.lasso import LassoResult
from SymbolicDSGE.regression.ridge import RidgeResult


class _FakeSolvedModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.compiled = SimpleNamespace(n_exog=1, observable_names=["obs"])
        self.kalman_calls: list[dict] = []
        self.sim_shocks: list[dict[str, np.ndarray]] = []

    def sim(
        self,
        T,
        shocks=None,
        shock_scale=1.0,
        x0=None,
        observables=False,
    ):
        del shock_scale, x0
        shock_draws = {}
        if shocks is not None:
            for name, shock in shocks.items():
                if callable(shock):
                    scale = (
                        np.eye(len(name.split(",")), dtype=np.float64)
                        if "," in name
                        else np.float64(1.0)
                    )
                    shock_draws[name] = np.asarray(shock(scale), dtype=np.float64)
                else:
                    shock_draws[name] = np.asarray(shock, dtype=np.float64)
        self.sim_shocks.append(shock_draws)
        t = np.arange(T + 1, dtype=np.float64)
        states = np.column_stack(
            [
                t + self.offset,
                ((t % 3.0) - 1.0) + 0.5 * self.offset,
            ]
        )
        out = {
            "_X": states,
            "x": states[:, 0],
            "z": states[:, 1],
        }
        if observables:
            out["obs"] = 0.5 * states[:, 0] + states[:, 1]
        return out

    def kalman(self, y, **kwargs):
        y = np.ascontiguousarray(y, dtype=np.float64)
        self.kalman_calls.append({"y": y.copy(), "kwargs": kwargs})
        n_obs, n_meas = y.shape
        cov = np.zeros((n_obs, n_meas, n_meas), dtype=np.float64)
        return FilterResult(
            x_pred=y.copy(),
            x_filt=y.copy(),
            P_pred=cov.copy(),
            P_filt=cov.copy(),
            y_pred=0.5 * y,
            y_filt=0.5 * y,
            innov=y - y.mean(axis=0),
            std_innov=y + 0.25,
            S=cov.copy(),
            loglik=np.float64(0.0),
        )


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
    base = _quadratic_sample()
    return np.stack([base, base + np.array([0.25, -0.5], dtype=np.float64)])


def test_raw_data_pipeline_runs_without_dgp_and_aggregates_wald_results() -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    target = np.zeros(2, dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            wald_test_step(
                "state_mean",
                source="states",
                target=target,
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, retain_contexts=True)

    expected = np.asarray(
        [wald_mean_hac(states[i], target, bandwidth=0).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["state_mean"], expected)
    assert out.test_summaries["state_mean"].n == 2
    assert out.test_results is not None
    assert len(out.test_results["state_mean"]) == 2
    assert all(result._pval is None for result in out.test_results["state_mean"])
    assert all(result._frozen_dist is None for result in out.test_results["state_mean"])
    np.testing.assert_allclose(
        out.pval_traces["state_mean"],
        out.test_summaries["state_mean"].pval_trace,
    )
    assert out.payloads is not None
    assert "datagen" in out.payloads[0]
    assert out.contexts is not None
    assert out.contexts[0].dgp is None


def test_raw_data_pipeline_accepts_observables_without_states() -> None:
    reference = _FakeSolvedModel()
    observables = _batched_states()[:, :, :1]
    target = np.zeros(1, dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("obs",)),
            wald_test_step(
                "obs_mean",
                source="observables",
                target=target,
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, retain_contexts=True)

    expected = np.asarray(
        [
            wald_mean_hac(observables[i], target, bandwidth=0).statistic
            for i in range(2)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["obs_mean"], expected)
    assert out.contexts is not None
    assert out.contexts[0].data is not None
    assert out.contexts[0].data.states is None
    np.testing.assert_allclose(out.contexts[0].data.observables, observables[0])


def test_ljung_box_pipeline_selects_column_and_aggregates_results() -> None:
    reference = _FakeSolvedModel()
    first = np.column_stack(
        [
            np.array([1.0, 2.0, 0.0, 4.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.5, -1.0, 2.0], dtype=np.float64),
        ]
    )
    second = np.column_stack(
        [
            np.array([2.0, 1.0, 3.0, 0.0, 4.0], dtype=np.float64),
            np.array([1.5, -0.5, 0.0, 2.5, 1.0], dtype=np.float64),
        ]
    )
    observables = np.stack([first, second])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("a", "b")),
            ljung_box_test_step(
                "lb_b",
                source="observables",
                column=[1],
                lags=2,
                alpha=0.1,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2)

    expected = np.asarray(
        [ljung_box(observables[i, :, 1], L=2, alpha=0.1).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["lb_b"], expected)
    assert out.test_summaries["lb_b"].n == 2
    assert out.test_summaries["lb_b"].df == np.float64(2.0)
    assert out.test_results is not None
    assert all(result.status is TestStatus.OK for result in out.test_results["lb_b"])


def test_ljung_box_pipeline_rejects_multi_column_inputs() -> None:
    reference = _FakeSolvedModel()
    observables = np.array([[1.0, 2.0]], dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("a", "b")),
            ljung_box_test_step(
                "lb",
                source="observables",
                lags=1,
            ),
        ]
    )

    with pytest.raises(ValueError, match="single column"):
        pipeline.run(reference=reference, n_rep=1)


def test_raw_data_pipeline_rejects_empty_raw_data() -> None:
    reference = _FakeSolvedModel()
    pipeline = MCPipeline(
        [
            raw_data_step(),
            wald_test_step(
                "obs_mean",
                source="observables",
                target=np.zeros(1, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="requires states, observables, or both"):
        pipeline.run(reference=reference, n_rep=1)


def test_pipeline_retention_controls_drop_payload_and_result_traces() -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            wald_test_step(
                "state_mean",
                source="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(
        reference=reference,
        n_rep=2,
        retain_payloads=False,
        retain_test_results=False,
    )

    assert out.payloads is None
    assert out.test_results is None
    assert out.contexts is None
    assert "state_mean" in out.test_summaries
    assert out.statistic_traces["state_mean"].shape == (2,)


def test_pipeline_result_reports_overall_and_step_performance() -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            wald_test_step(
                "state_mean",
                source="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2)

    assert out.elapsed_s >= 0.0
    assert out.it_s > 0.0
    assert set(out.step_elapsed_s) == {"datagen", "state_mean"}
    assert out.step_counts == {"datagen": 2, "state_mean": 2}
    assert out.step_failures == {"datagen": 0, "state_mean": 0}
    assert set(out.step_it_s) == {"datagen", "state_mean"}

    lines: list[str] = []
    report_mc_performance(out, print_func=lines.append)
    assert lines[0].startswith("MC run concluded successfully with ")
    assert lines[0].endswith(" it/s.")

    lines.clear()
    out.report_step_performance(print_func=lines.append)
    assert len(lines) == 2
    assert lines[0].startswith("datagen concluded successfully with ")
    assert lines[0].endswith(" it/s.")
    assert lines[1].startswith("state_mean concluded successfully with ")
    assert lines[1].endswith(" it/s.")

    lines.clear()
    report_mc_step_performance(out, print_func=lines.append)
    assert len(lines) == 2


def test_pipeline_run_verbosity_controls_performance_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            wald_test_step(
                "state_mean",
                source="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    pipeline.run(reference=reference, n_rep=2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("MC run concluded successfully with ")

    pipeline.run(reference=reference, n_rep=2, verbosity=0)
    assert capsys.readouterr().out == ""

    pipeline.run(reference=reference, n_rep=2, verbosity=2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("datagen concluded successfully with ")
    assert lines[1].startswith("state_mean concluded successfully with ")

    with pytest.raises(ValueError, match="verbosity"):
        pipeline.run(reference=reference, n_rep=2, verbosity=3)


def test_sim_filter_wald_pipeline_uses_reference_filter_payload() -> None:
    reference = _FakeSolvedModel()
    dgp = _FakeSolvedModel(offset=1.0)
    pipeline = MCPipeline(
        [
            simulation_step(T=8, observables=True),
            reference_filter_step(),
            wald_test_step(
                "std_innov_mean",
                source="std_innov",
                target=np.zeros(1, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(reference=reference, dgp=dgp, n_rep=2)

    assert len(reference.kalman_calls) == 2
    assert reference.kalman_calls[0]["kwargs"]["observables"] == ["obs"]
    assert out.payloads is not None
    expected = np.asarray(
        [
            wald_mean_hac(
                out.payloads[i]["filter"].std_innov,
                np.zeros(1, dtype=np.float64),
                bandwidth=0,
            ).statistic
            for i in range(2)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["std_innov_mean"], expected)
    assert out.test_summaries["std_innov_mean"].n == 2


def test_simulation_step_can_advance_seeded_shock_spec_as_stream() -> None:
    T = 6
    reference = _FakeSolvedModel()
    dgp = _FakeSolvedModel(offset=1.0)
    shocks = {
        "g,z": Shock(T, "norm", multivar=True, seed=0),
        "r": Shock(T, "norm", multivar=False, seed=1),
    }
    pipeline = MCPipeline(
        [
            simulation_step(
                T=T,
                shocks=shocks,
                seed_increment="auto",
                observables=False,
            )
        ]
    )

    pipeline.run(reference=reference, dgp=dgp, n_rep=3)

    expected_seeds = [(0, 1), (2, 3), (4, 5)]
    for rep_idx, (gz_seed, r_seed) in enumerate(expected_seeds):
        expected_gz = Shock(T, "norm", multivar=True, seed=gz_seed).shock_generator()(
            np.eye(2, dtype=np.float64)
        )
        expected_r = Shock(T, "norm", multivar=False, seed=r_seed).shock_generator()(
            np.float64(1.0)
        )
        np.testing.assert_allclose(dgp.sim_shocks[rep_idx]["g,z"], expected_gz)
        np.testing.assert_allclose(dgp.sim_shocks[rep_idx]["r"], expected_r)


def test_transform_step_returning_mcdata_updates_downstream_data() -> None:
    reference = _FakeSolvedModel()
    observables = _batched_states()[:, :, :1]

    def add_observation_noise(
        *,
        context,
        reference,
        dgp,
        rep_idx,
    ) -> MCData:
        del reference, dgp, rep_idx
        data = context.require_data()
        assert data.observables is not None
        return MCData(
            states=data.states,
            observables=data.observables + 1.0,
            n_exog=data.n_exog,
            raw=data.raw,
            observable_names=data.observable_names,
        )

    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("obs",)),
            transform_step("add_noise", add_observation_noise),
            reference_filter_step(),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, retain_contexts=True)

    np.testing.assert_allclose(reference.kalman_calls[0]["y"], observables[0] + 1.0)
    assert out.contexts is not None
    assert out.contexts[0].data is not None
    np.testing.assert_allclose(out.contexts[0].data.observables, observables[0] + 1.0)
    assert out.payloads is not None
    assert isinstance(out.payloads[0]["add_noise"], MCData)


def test_regression_step_runs_ols_and_stores_result_payload() -> None:
    reference = _FakeSolvedModel()
    x = np.arange(1.0, 7.0, dtype=np.float64)
    y = 2.5 * x
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("y", "x")),
            regression_step(
                "ols",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=1)

    assert out.payloads is not None
    result = out.payloads[0]["ols"]
    assert isinstance(result, OLSResult)
    assert result.variables == ["Intercept", "x"]
    np.testing.assert_allclose(result.coefficients, np.array([0.0, 2.5]), atol=1e-12)
    np.testing.assert_allclose(result.y, y)
    np.testing.assert_allclose(
        result.X,
        np.column_stack([np.ones_like(x), x]),
    )
    assert out.test_summaries == {}
    assert "ols" in out.regression_summaries
    np.testing.assert_allclose(
        out.coefficient_traces["ols"],
        np.array([[0.0, 2.5]], dtype=np.float64),
        atol=1e-12,
    )
    assert out.regression_summaries["ols"].status_trace == (result.status,)


def test_regression_summary_does_not_depend_on_payload_retention() -> None:
    reference = _FakeSolvedModel()
    x = np.arange(1.0, 7.0, dtype=np.float64)
    y = 3.0 * x
    observables = np.stack(
        [
            np.column_stack([y, x]),
            np.column_stack([y + x, x]),
        ]
    )
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("y", "x")),
            regression_step(
                "ols",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, retain_payloads=False)

    assert out.payloads is None
    assert out.test_summaries == {}
    np.testing.assert_allclose(
        out.coefficient_traces["ols"],
        np.array([[0.0, 3.0], [0.0, 4.0]], dtype=np.float64),
        atol=1e-12,
    )
    assert out.regression_summaries["ols"].n_rep == 2


def test_regression_step_runs_ridge_kind_and_aggregates_summary() -> None:
    reference = _FakeSolvedModel()
    x = np.arange(1.0, 7.0, dtype=np.float64)
    y = 1.0 + 2.0 * x
    alpha = np.float64(0.5)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("y", "x")),
            regression_step(
                "ridge",
                kind="ridge",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
                alpha=alpha,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=1)

    assert out.payloads is not None
    result = out.payloads[0]["ridge"]
    assert isinstance(result, RidgeResult)
    X = np.column_stack([np.ones_like(x), x])
    G = (X.T @ X) / X.shape[0]
    g = (X.T @ y) / X.shape[0]
    expected_coef = np.linalg.solve(
        G + np.diag([0.0, alpha]),
        g,
    )
    assert result.variables == ["Intercept", "x"]
    np.testing.assert_allclose(result.coefficients, expected_coef)
    np.testing.assert_allclose(out.coefficient_traces["ridge"], expected_coef[None, :])
    np.testing.assert_allclose(
        out.regression_summaries["ridge"].r2_trace,
        np.asarray([result.r2], dtype=np.float64),
    )
    with pytest.raises(TypeError, match="OLS-specific"):
        _ = out.regression_summaries["ridge"].se_trace


def test_regression_step_runs_lasso_kind_and_aggregates_summary() -> None:
    reference = _FakeSolvedModel()
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            regression_step(
                "lasso",
                kind="lasso",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1, 2, 3],
                intercept=False,
                alpha=np.float64(0.5),
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=1)

    assert out.payloads is not None
    result = out.payloads[0]["lasso"]
    assert isinstance(result, LassoResult)
    np.testing.assert_allclose(result.coefficients, np.array([1.5, 0.0, 0.0]))
    np.testing.assert_allclose(
        out.coefficient_traces["lasso"],
        np.array([[1.5, 0.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.regression_summaries["lasso"].r2_trace,
        np.asarray([result.r2], dtype=np.float64),
    )


def test_regression_step_requires_single_response_column() -> None:
    reference = _FakeSolvedModel()
    observables = np.column_stack(
        [
            np.arange(1.0, 6.0, dtype=np.float64),
            np.arange(2.0, 7.0, dtype=np.float64),
            np.arange(3.0, 8.0, dtype=np.float64),
        ]
    )
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            regression_step(
                "ols",
                y_source="observables",
                X_source="observables",
                y_column=[0, 1],
                X_columns=[2],
            ),
        ]
    )

    with pytest.raises(ValueError, match="exactly one column"):
        pipeline.run(reference=reference, n_rep=1)


def test_regression_step_requires_matching_row_counts() -> None:
    reference = _FakeSolvedModel()
    states = np.arange(10.0, dtype=np.float64).reshape(5, 2)
    observables = np.arange(4.0, dtype=np.float64).reshape(4, 1)
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, observables=observables),
            regression_step(
                "ols",
                y_source="observables",
                X_source="states",
                y_column=0,
                X_columns=[0],
            ),
        ]
    )

    with pytest.raises(ValueError, match="same number of rows"):
        pipeline.run(reference=reference, n_rep=1)


def test_regression_step_validates_result_type() -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            MCStep(
                name="bad_regression",
                op_type=OpType.REGRESSION,
                func=lambda **_: np.zeros(1, dtype=np.float64),
            ),
        ]
    )

    with pytest.raises(
        TypeError, match="REGRESSION steps must return RegressionResult"
    ):
        pipeline.run(reference=reference, n_rep=1)


def test_pipeline_validates_step_order_and_unique_names() -> None:
    with pytest.raises(ValueError, match="first step"):
        MCPipeline(
            [
                wald_test_step(
                    "state_mean",
                    source="states",
                    target=np.zeros(2, dtype=np.float64),
                )
            ]
        )

    with pytest.raises(ValueError, match="unique"):
        MCPipeline(
            [
                raw_data_step(name="dup", states=_batched_states()),
                MCStep(
                    name="dup",
                    op_type=OpType.TRANSFORM,
                    func=lambda **_: np.zeros((1, 1), dtype=np.float64),
                ),
            ]
        )


def test_pipeline_collects_failures_when_fail_fast_is_false() -> None:
    reference = _FakeSolvedModel()
    states = _batched_states()
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, n_exog=0),
            wald_test_step(
                "state_mean",
                source="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth=0,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=3, fail_fast=False)

    assert out.n_successful == 2
    assert len(out.failures) == 1
    assert out.failures[0].rep_idx == 2
    assert out.failures[0].step_name == "datagen"
    assert out.statistic_traces["state_mean"].shape == (2,)
    assert out.step_counts == {"datagen": 3, "state_mean": 2}
    assert out.step_failures == {"datagen": 1, "state_mean": 0}

    lines: list[str] = []
    out.report_performance(print_func=lines.append)
    assert lines[0].startswith("MC run concluded unsuccessfully with ")

    lines.clear()
    report_mc_step_performance(out, print_func=lines.append)
    assert lines[0].startswith("datagen concluded unsuccessfully with ")
    assert lines[1].startswith("state_mean concluded successfully with ")
