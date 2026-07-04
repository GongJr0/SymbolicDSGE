from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE import Shock
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
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.filter import FilterResult
from SymbolicDSGE.monte_carlo import (
    MCPipeline,
    MCContext,
    MCData,
    MCStep,
    OpType,
)
from SymbolicDSGE.monte_carlo.mc_constructs import (
    report_mc_performance,
    report_mc_step_performance,
)
from SymbolicDSGE.monte_carlo.reference_constructs import MCReferenceConstruct
from SymbolicDSGE.monte_carlo.operations.core import (
    raw_data_step,
    reference_filter_step,
    simulation_step,
)
from SymbolicDSGE.monte_carlo.operations.core.ops import simulate_dgp
from SymbolicDSGE.monte_carlo.operations.regressions import regression_step
from SymbolicDSGE.monte_carlo.operations.tests import (
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    wald_test_step,
)
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step
from SymbolicDSGE.monte_carlo.operations.utils import (
    _clone_or_pass_shocks,
    _resolve_context_array,
    _resolve_seed_increment,
    _select_raw_rep_array,
)
from SymbolicDSGE.regression.elastic_net import ElasticNetResult
from SymbolicDSGE.regression.ols import OLSResult
from SymbolicDSGE.regression.lasso import LassoResult
from SymbolicDSGE.regression.ridge import RidgeResult


class _FakeSolvedModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.compiled = SimpleNamespace(
            idx={"x": 0, "z": 1},
            var_names=["x", "z"],
            n_exog=1,
            observable_names=["obs"],
        )
        self.kalman_calls: list[dict] = []
        self.sim_shocks: list[dict[str, np.ndarray]] = []

    def _draw_shocks(self, shocks=None) -> None:
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

    def _simulate_state_matrix(
        self,
        T,
        shocks=None,
        shock_scale=1.0,
        x0=None,
    ):
        del shock_scale, x0
        self._draw_shocks(shocks)
        t = np.arange(T + 1, dtype=np.float64)
        return np.column_stack(
            [
                t + self.offset,
                ((t % 3.0) - 1.0) + 0.5 * self.offset,
            ]
        )

    def _simulate_observable_matrix(self, states, *, drop_initial=False):
        start = 1 if drop_initial else 0
        obs = 0.5 * states[:, 0] + states[:, 1]
        return np.ascontiguousarray(obs[start:].reshape(-1, 1), dtype=np.float64)

    def sim(
        self,
        T,
        shocks=None,
        shock_scale=1.0,
        x0=None,
        observables=False,
    ):
        states = self._simulate_state_matrix(
            T=T,
            shocks=shocks,
            shock_scale=shock_scale,
            x0=x0,
        )
        out = {
            "_X": states,
            "x": states[:, 0],
            "z": states[:, 1],
        }
        if observables:
            out["obs"] = self._simulate_observable_matrix(
                states,
                drop_initial=False,
            )[:, 0]
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


def test_jarque_bera_pipeline_selects_column_and_aggregates_results() -> None:
    reference = _FakeSolvedModel()
    base = np.column_stack(
        [
            np.linspace(-2.0, 3.0, 12, dtype=np.float64) ** 2,
            np.linspace(1.0, 4.0, 12, dtype=np.float64),
        ]
    )
    observables = np.stack([base, base + np.array([0.5, -0.25])])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("a", "b")),
            jarque_bera_test_step(
                "jb_a",
                source="observables",
                column=0,
                alpha=0.1,
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, verbosity=0)

    expected = np.asarray(
        [jarque_bera(observables[i, :, 0], alpha=0.1).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["jb_a"], expected)
    assert out.succeeded
    assert out.test_summaries["jb_a"].df == 12
    assert out.test_results is not None
    assert all(result.status is TestStatus.OK for result in out.test_results["jb_a"])
    assert all(result._pval is None for result in out.test_results["jb_a"])


def test_jarque_bera_pipeline_rejects_multi_column_inputs() -> None:
    reference = _FakeSolvedModel()
    observables = np.arange(24.0, dtype=np.float64).reshape(12, 2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            jarque_bera_test_step("jb", source="observables"),
        ]
    )

    with pytest.raises(ValueError, match="single column"):
        pipeline.run(reference=reference, n_rep=1, verbosity=0)


def test_jarque_bera_pipeline_handles_burn_in_that_removes_all_samples() -> None:
    reference = _FakeSolvedModel()
    observables = np.stack(
        [
            np.arange(6.0, dtype=np.float64).reshape(-1, 1),
            np.arange(6.0, 12.0, dtype=np.float64).reshape(-1, 1),
        ]
    )
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            jarque_bera_test_step(
                "jb",
                source="observables",
                burn_in=observables.shape[1],
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=2, verbosity=0)

    assert out.succeeded
    assert out.failures == ()
    assert out.test_results is not None
    assert all(
        result.status is TestStatus.INSUFFICIENT_SAMPLES
        for result in out.test_results["jb"]
    )
    assert out.test_status_traces["jb"] == (TestStatus.INSUFFICIENT_SAMPLES,) * 2
    assert (
        out.test_summaries["jb"].status_trace == (TestStatus.INSUFFICIENT_SAMPLES,) * 2
    )
    assert np.isnan(out.statistic_traces["jb"]).all()
    assert np.isnan(out.pval_traces["jb"]).all()


def test_breusch_pagan_pipeline_selects_columns_and_aggregates_results() -> None:
    rng = np.random.default_rng(512)
    X = rng.normal(size=(2, 40, 2))
    eps = rng.normal(scale=np.exp(0.4 * X[:, :, 0]))
    observables = np.concatenate((eps[:, :, None], X), axis=2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            breusch_pagan_test_step(
                "bp",
                residual_source="observables",
                X_source="observables",
                residual_col=0,
                X_columns=[1, 2],
            ),
            breusch_pagan_test_step(
                "robust_bp",
                residual_source="observables",
                X_source="observables",
                residual_col=0,
                X_columns=[1, 2],
                robust=True,
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [breusch_pagan(eps[i], X[i]).statistic for i in range(2)],
        dtype=np.float64,
    )
    robust_expected = np.asarray(
        [robust_breusch_pagan(eps[i], X[i]).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["bp"], expected)
    np.testing.assert_allclose(out.statistic_traces["robust_bp"], robust_expected)
    assert out.test_summaries["bp"].df == 2
    assert out.test_summaries["robust_bp"].df == 2
    assert out.test_status_traces["bp"] == (TestStatus.OK, TestStatus.OK)
    assert out.test_status_traces["robust_bp"] == (TestStatus.OK, TestStatus.OK)


def test_breusch_pagan_pipeline_supports_separate_residual_and_regressor_sources() -> (
    None
):
    rng = np.random.default_rng(128)
    states = rng.normal(size=(2, 30, 2))
    residuals = rng.normal(scale=np.exp(0.4 * states[:, :, 0]))
    observables = residuals[:, :, None]
    pipeline = MCPipeline(
        [
            raw_data_step(states=states, observables=observables),
            breusch_pagan_test_step(
                "bp",
                residual_source="observables",
                X_source="states",
                residual_col=0,
                X_columns=[0, 1],
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [breusch_pagan(residuals[i], states[i]).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["bp"], expected)


def test_breusch_pagan_pipeline_supports_separate_payload_sources() -> None:
    rng = np.random.default_rng(256)
    X = rng.normal(size=(30, 2))
    residuals = rng.normal(scale=np.exp(0.4 * X[:, 0]), size=30)

    def residual_payload(**_: object) -> np.ndarray:
        return residuals

    def regressor_payload(**_: object) -> np.ndarray:
        return X

    pipeline = MCPipeline(
        [
            raw_data_step(observables=residuals[:, None]),
            transform_step("residual_payload", residual_payload),
            transform_step("regressor_payload", regressor_payload),
            breusch_pagan_test_step(
                "bp",
                residual_source="payload",
                X_source="payload",
                residual_payload_key="residual_payload",
                x_payload_key="regressor_payload",
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    np.testing.assert_allclose(
        out.statistic_traces["bp"],
        np.full(2, breusch_pagan(residuals, X).statistic, dtype=np.float64),
    )


def test_breusch_pagan_pipeline_validates_residual_and_regressor_inputs() -> None:
    observables = np.arange(30.0, dtype=np.float64).reshape(10, 3)
    reference = _FakeSolvedModel()

    with pytest.raises(ValueError, match="exactly one column"):
        MCPipeline(
            [
                raw_data_step(observables=observables),
                breusch_pagan_test_step(
                    "bp",
                    residual_source="observables",
                    X_source="observables",
                    X_columns=[1, 2],
                ),
            ]
        ).run(reference=reference, n_rep=1, verbosity=0)

    with pytest.raises(ValueError, match="at least one variance regressor"):
        MCPipeline(
            [
                raw_data_step(observables=observables),
                breusch_pagan_test_step(
                    "bp",
                    residual_source="observables",
                    X_source="observables",
                    residual_col=0,
                    X_columns=[],
                ),
            ]
        ).run(reference=reference, n_rep=1, verbosity=0)

    with pytest.raises(ValueError, match="same number of rows"):
        MCPipeline(
            [
                raw_data_step(
                    states=np.arange(33.0, dtype=np.float64).reshape(11, 3),
                    observables=observables,
                ),
                breusch_pagan_test_step(
                    "bp",
                    residual_source="observables",
                    X_source="states",
                    residual_col=0,
                    X_columns=[0, 1],
                ),
            ]
        ).run(reference=reference, n_rep=1, verbosity=0)


def test_breusch_pagan_pipeline_handles_burn_in_that_removes_all_samples() -> None:
    base = np.arange(30.0, dtype=np.float64).reshape(10, 3)
    observables = np.stack((base, base + 0.5))
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            breusch_pagan_test_step(
                "bp",
                residual_source="observables",
                X_source="observables",
                residual_col=0,
                X_columns=[1, 2],
                burn_in=observables.shape[1],
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    assert out.succeeded
    assert out.test_status_traces["bp"] == (TestStatus.INSUFFICIENT_SAMPLES,) * 2
    assert np.isnan(out.statistic_traces["bp"]).all()
    assert np.isnan(out.pval_traces["bp"]).all()


def test_breusch_godfrey_pipeline_selects_columns_and_aggregates_results() -> None:
    rng = np.random.default_rng(512)
    X = rng.normal(size=(2, 40, 2))
    eps = rng.normal(size=(2, 40))
    observables = np.concatenate((eps[:, :, None], X), axis=2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            breusch_godfrey_test_step(
                "bg",
                residual_source="observables",
                X_source="observables",
                residual_col=0,
                X_columns=[1, 2],
                lags=2,
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [breusch_godfrey(eps[i], X[i], lags=2).statistic for i in range(2)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out.statistic_traces["bg"], expected)
    assert out.test_summaries["bg"].df == 2
    assert out.test_status_traces["bg"] == (TestStatus.OK, TestStatus.OK)


def test_breusch_godfrey_pipeline_validates_residual_and_regressor_inputs() -> None:
    observables = np.arange(30.0, dtype=np.float64).reshape(10, 3)
    reference = _FakeSolvedModel()

    with pytest.raises(ValueError, match="exactly one column"):
        MCPipeline(
            [
                raw_data_step(observables=observables),
                breusch_godfrey_test_step(
                    "bg",
                    residual_source="observables",
                    X_source="observables",
                    X_columns=[1, 2],
                ),
            ]
        ).run(reference=reference, n_rep=1, verbosity=0)

    with pytest.raises(ValueError, match="same number of rows"):
        MCPipeline(
            [
                raw_data_step(
                    states=np.arange(33.0, dtype=np.float64).reshape(11, 3),
                    observables=observables,
                ),
                breusch_godfrey_test_step(
                    "bg",
                    residual_source="observables",
                    X_source="states",
                    residual_col=0,
                    X_columns=[0, 1],
                ),
            ]
        ).run(reference=reference, n_rep=1, verbosity=0)


def test_breusch_godfrey_pipeline_handles_burn_in_that_removes_all_samples() -> None:
    base = np.arange(30.0, dtype=np.float64).reshape(10, 3)
    observables = np.stack((base, base + 0.5))
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            breusch_godfrey_test_step(
                "bg",
                residual_source="observables",
                X_source="observables",
                residual_col=0,
                X_columns=[1, 2],
                burn_in=observables.shape[1],
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    assert out.succeeded
    assert out.test_status_traces["bg"] == (TestStatus.INSUFFICIENT_SAMPLES,) * 2
    assert np.isnan(out.statistic_traces["bg"]).all()
    assert np.isnan(out.pval_traces["bg"]).all()


def test_cusum_pipeline_aggregates_results_with_nan_df() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2, 60, 2))
    X[:, :, 0] = 1.0  # constant column for a well-posed recursion
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=(2, 60))
    observables = np.concatenate((y[:, :, None], X), axis=2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            cusum_test_step(
                "cs",
                y_source="observables",
                x_source="observables",
                y_column=0,
                X_columns=[1, 2],
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [cusum(y[i], X[i]).statistic for i in range(2)], dtype=np.float64
    )
    np.testing.assert_allclose(out.statistic_traces["cs"], expected)
    assert out.test_status_traces["cs"] == (TestStatus.OK, TestStatus.OK)
    # CUSUM is parameter-free: the NaN df placeholder must survive aggregation
    # across replications (the metadata-equality check is NaN-aware).
    assert np.isnan(out.test_summaries["cs"].df)


def test_cusumsq_pipeline_aggregates_results() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2, 60, 2))
    X[:, :, 0] = 1.0  # constant column for a well-posed recursion
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=(2, 60))
    observables = np.concatenate((y[:, :, None], X), axis=2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            cusumsq_test_step(
                "csq",
                y_source="observables",
                x_source="observables",
                y_column=0,
                X_columns=[1, 2],
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [cusumsq_test(y[i], X[i]).statistic for i in range(2)], dtype=np.float64
    )
    np.testing.assert_allclose(out.statistic_traces["csq"], expected)
    assert out.test_status_traces["csq"] == (TestStatus.OK, TestStatus.OK)
    # CUSUMSQ is parameterized by the recursive-residual count n = T - p, which
    # is identical across equal-shape replications and survives aggregation as
    # the shared df.
    assert out.test_summaries["csq"].df == 60 - 2


def test_chow_pipeline_aggregates_results() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2, 60, 2))
    X[:, :, 0] = 1.0  # constant column for a well-posed partition
    y = X @ np.array([0.5, -0.3]) + rng.normal(size=(2, 60))
    observables = np.concatenate((y[:, :, None], X), axis=2)
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            chow_test_step(
                "ch",
                y_source="observables",
                x_source="observables",
                y_column=0,
                X_columns=[1, 2],
                t_break=30,
            ),
        ]
    )

    out = pipeline.run(reference=_FakeSolvedModel(), n_rep=2, verbosity=0)

    expected = np.asarray(
        [chow(y[i], X[i], t_break=30).statistic for i in range(2)], dtype=np.float64
    )
    np.testing.assert_allclose(out.statistic_traces["ch"], expected)
    assert out.test_status_traces["ch"] == (TestStatus.OK, TestStatus.OK)
    # Chow uses an F reference with df = (p, T - 2p), identical across
    # equal-shape replications and preserved through aggregation.
    assert out.test_summaries["ch"].df == (2, 60 - 2 * 2)


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


def test_simulate_dgp_fast_path_for_real_solved_model() -> None:
    T = 3
    shock = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    A = np.array([[0.5, 0.0], [0.0, 0.25]], dtype=np.float64)
    B = np.array([[1.0], [0.5]], dtype=np.float64)
    C = np.array([[2.0, 0.5]], dtype=np.float64)
    d = np.array([1.0], dtype=np.float64)

    config = SimpleNamespace(
        shock_map={},
        calibration=SimpleNamespace(parameters={}, shock_std={}),
        equations=SimpleNamespace(obs_is_affine={"obs": True}),
    )

    def build_affine_measurement_matrices(params, y_names):
        assert params == {}
        assert y_names == ["obs"]
        return C, d

    compiled = SimpleNamespace(
        idx={"u": 0, "x": 1},
        var_names=["u", "x"],
        n_exog=1,
        n_state=1,
        observable_names=["obs"],
        config=config,
        build_affine_measurement_matrices=build_affine_measurement_matrices,
    )
    dgp = SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(f=np.array([[0.0]], dtype=np.float64)),
        A=A,
        B=B,
    )

    data = simulate_dgp(
        reference=dgp,
        dgp=dgp,
        rep_idx=0,
        T=T,
        shocks={"u": shock},
        observables=True,
    )

    expected_states = np.empty((T + 1, 2), dtype=np.float64)
    expected_states[0] = 0.0
    for t in range(T):
        expected_states[t + 1] = A @ expected_states[t] + B[:, 0] * shock[t]
    expected_obs = expected_states @ C.T + d

    np.testing.assert_allclose(data.states, expected_states)
    np.testing.assert_allclose(data.observables, expected_obs[1:])
    np.testing.assert_allclose(data.raw["_X"], expected_states)
    np.testing.assert_allclose(data.raw["u"], expected_states[:, 0])
    np.testing.assert_allclose(data.raw["obs"], expected_obs[:, 0])
    assert data.observable_names == ("obs",)
    assert data.n_exog == 1


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


def test_regression_step_runs_elastic_net_kind_and_aggregates_summary() -> None:
    reference = _FakeSolvedModel()
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            regression_step(
                "elastic_net",
                kind="elastic_net",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1, 2, 3],
                intercept=False,
                alpha=np.float64(0.5),
                l1_ratio=np.float64(0.5),
            ),
        ]
    )

    out = pipeline.run(reference=reference, n_rep=1)

    assert out.payloads is not None
    result = out.payloads[0]["elastic_net"]
    assert isinstance(result, ElasticNetResult)
    np.testing.assert_allclose(
        result.coefficients,
        np.array([9.0 / 7.0, -1.0 / 7.0, 0.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.coefficient_traces["elastic_net"],
        result.coefficients[None, :],
    )
    assert out.regression_summaries["elastic_net"].status_trace == (result.status,)


def test_regression_step_runs_elastic_net_grid_search_kind() -> None:
    reference = _FakeSolvedModel()
    x = np.eye(2, dtype=np.float64)
    y = np.array([3.0, -1.0], dtype=np.float64)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables),
            regression_step(
                "elastic_net_gs",
                kind="elastic_net_gs",
                y_source="observables",
                X_source="observables",
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

    out = pipeline.run(reference=reference, n_rep=1)

    assert out.payloads is not None
    result = out.payloads[0]["elastic_net_gs"]
    assert isinstance(result, ElasticNetResult)
    assert result.alpha_grid is not None
    assert result.coefficient_path is not None
    assert out.coefficient_traces["elastic_net_gs"].shape == (1, 2)


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
    with pytest.raises(ValueError, match="first per-rep step"):
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


def test_mc_operation_utils_validate_seeded_shock_specs() -> None:
    arr = np.ones((2, 1), dtype=np.float64)
    callable_shock = lambda scale: np.ones((2,), dtype=np.float64) * scale

    assert _clone_or_pass_shocks(None, T=2, rep_idx=0, seed_increment="auto") is None
    out = _clone_or_pass_shocks(
        {"arr": arr, "callable": callable_shock},
        T=2,
        rep_idx=3,
        seed_increment=1,
    )
    assert out is not None
    assert out["arr"] is arr
    assert out["callable"] is callable_shock
    assert _resolve_seed_increment({"arr": arr}, "auto") == 0
    assert _resolve_seed_increment({"eps": Shock(2, "norm", seed=5)}, "auto") == 1
    assert _resolve_seed_increment({"eps": Shock(2, "norm", seed=None)}, "auto") == 0
    assert _resolve_seed_increment({"eps": Shock(2, "norm", seed=5)}, 4) == 4

    with pytest.raises(ValueError, match="non-negative"):
        _resolve_seed_increment({"eps": Shock(2, "norm", seed=5)}, -1)
    with pytest.raises(ValueError, match="expected 2"):
        _clone_or_pass_shocks(
            {"eps": Shock(3, "norm", seed=5)},
            T=2,
            rep_idx=0,
            seed_increment="auto",
        )
    with pytest.raises(ValueError, match="generator-style"):
        _clone_or_pass_shocks(
            {"eps": Shock(2, "norm", seed=5, shock_arr=np.zeros(2))},
            T=2,
            rep_idx=0,
            seed_increment="auto",
        )
    with pytest.raises(ValueError, match="multivar=True"):
        _clone_or_pass_shocks(
            {"eps,z": Shock(2, "norm", multivar=False, seed=5)},
            T=2,
            rep_idx=0,
            seed_increment="auto",
        )
    with pytest.raises(ValueError, match="multivar=False"):
        _clone_or_pass_shocks(
            {"eps": Shock(2, "norm", multivar=True, seed=5)},
            T=2,
            rep_idx=0,
            seed_increment="auto",
        )


def test_mc_operation_utils_resolve_context_and_raw_arrays() -> None:
    reference = _FakeSolvedModel()
    states = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    observables = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    context = MCContext(
        rep_idx=0,
        reference=reference,
        dgp=None,
        data=MCData(states=states, observables=observables),
        payloads={"vector": np.arange(5.0, dtype=np.float64)},
    )

    selected = _resolve_context_array(
        context,
        source="states",
        filter_key="filter",
        payload_key=None,
        columns=[1],
        burn_in=1,
        drop_initial=True,
    )
    np.testing.assert_allclose(selected, states[2:, [1]])

    payload = _resolve_context_array(
        context,
        source="payload",
        filter_key="filter",
        payload_key="vector",
        columns=None,
        burn_in=2,
        drop_initial=False,
    )
    np.testing.assert_allclose(payload, np.arange(2.0, 5.0).reshape(3, 1))

    filt = reference.kalman(observables)
    context.payloads["filter"] = filt
    np.testing.assert_allclose(
        _resolve_context_array(
            context,
            source="std_innov",
            filter_key="filter",
            payload_key=None,
            columns=slice(0, 1),
            burn_in=0,
            drop_initial=False,
        ),
        filt.std_innov[:, :1],
    )

    with pytest.raises(ValueError, match="burn_in"):
        _resolve_context_array(
            context,
            source="states",
            filter_key="filter",
            payload_key=None,
            columns=None,
            burn_in=-1,
            drop_initial=False,
        )
    with pytest.raises(ValueError, match="payload_key"):
        _resolve_context_array(
            context,
            source="payload",
            filter_key="filter",
            payload_key=None,
            columns=None,
            burn_in=0,
            drop_initial=False,
        )
    context.payloads["not_filter"] = np.zeros(1, dtype=np.float64)
    with pytest.raises(TypeError, match="FilterResult"):
        _resolve_context_array(
            context,
            source="std_innov",
            filter_key="not_filter",
            payload_key=None,
            columns=None,
            burn_in=0,
            drop_initial=False,
        )
    context.payloads["cube"] = np.zeros((1, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="1D or 2D"):
        _resolve_context_array(
            context,
            source="payload",
            filter_key="filter",
            payload_key="cube",
            columns=None,
            burn_in=0,
            drop_initial=False,
        )

    raw = np.arange(24.0, dtype=np.float64).reshape(2, 4, 3)
    np.testing.assert_allclose(
        _select_raw_rep_array(raw, rep_idx=1, name="raw", allow_vector=False),
        raw[1],
    )
    np.testing.assert_allclose(
        _select_raw_rep_array(
            np.arange(4.0, dtype=np.float64),
            rep_idx=0,
            name="vector",
            allow_vector=True,
        ),
        np.arange(4.0, dtype=np.float64).reshape(4, 1),
    )
    with pytest.raises(IndexError, match="rep_idx=2"):
        _select_raw_rep_array(raw, rep_idx=2, name="raw", allow_vector=False)
    with pytest.raises(ValueError, match="2D or 3D"):
        _select_raw_rep_array(
            np.arange(4.0, dtype=np.float64),
            rep_idx=0,
            name="vector",
            allow_vector=False,
        )


def test_reference_construct_wraps_model_and_external_generators() -> None:
    model = _FakeSolvedModel(offset=1.0)
    construct = MCReferenceConstruct(model, T=3, N=5)

    assert construct.model is model
    assert construct.T == 3
    assert construct.N == 5
    data = construct._data_from_sim(observables=True)
    assert data.state_mat is not None
    assert data.obs_mat is not None
    assert data.n_exog == 1
    assert data.state_mat.shape == (4, 2)
    assert data.obs_mat.shape == (3, 1)

    def generator(T: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.ones((T + 1, 2), dtype=np.float64),
            np.ones((T, 1), dtype=np.float64) * 2.0,
        )

    callable_construct = MCReferenceConstruct.DataGeneratingCallable(
        generator,
        T=4,
        N=7,
    )
    out = callable_construct()
    assert callable_construct.func is generator
    assert callable_construct.T == 4
    assert callable_construct.N == 7
    np.testing.assert_allclose(out.state_mat, np.ones((5, 2), dtype=np.float64))
    np.testing.assert_allclose(out.obs_mat, np.ones((4, 1), dtype=np.float64) * 2.0)
    assert out.n_exog == -1
