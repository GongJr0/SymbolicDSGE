from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE._ckernels.monte_carlo._runner import run as run_native
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.core.solver_backend import PerturbationSolution
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.step_factories import (
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    raw_model_data_step,
    reference_filter_step,
    regression_step,
    simulation_step,
    log_diff_step,
    transform_step,
    wald_test_step,
)


def _custom_first_difference(sample: np.ndarray, output: np.ndarray) -> int:
    output[:] = sample[1:] - sample[:-1]
    return 0


def test_native_lowering_runs_custom_transform() -> None:
    n_rep, T = 3, 12
    observables = np.arange(n_rep * T * 2, dtype=np.float64).reshape(n_rep, T, 2)
    pipeline = MCPipeline(
        [
            raw_model_data_step("data", observables=observables),
            transform_step(
                "difference",
                _custom_first_difference,
                source="data",
                field="observables",
                output_shape=(T - 1, 2),
            ),
        ]
    )

    lowered = pipeline.lower_native(
        reference=cast(SolvedModel, object()), n_rep=n_rep, n_jobs=1
    )
    result = run_native(lowered.allocation, lowered.steps, lowered.input_bindings)

    assert result.status == 0
    layout = lowered.plan["difference"].out_fields["payload"]
    actual = (
        lowered.allocation.steps["difference"]
        .float_retained[:, layout.offset : layout.offset + layout.flat_count]
        .reshape(n_rep, *layout.shape)
    )
    np.testing.assert_allclose(actual, np.diff(observables, axis=1))


def test_native_lowering_runs_raw_transform_ols_and_diagnostic_pipeline() -> None:
    n_rep, T = 4, 20
    rng = np.random.default_rng(20260730)
    observables = np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(n_rep, T, 2)), axis=1))
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "datagen", observables=observables, observable_names=("y", "x")
            ),
            log_diff_step(
                "growth",
                source="datagen",
                field="observables",
                columns=(0, 1),
            ),
            regression_step(
                "ols",
                y_source="growth",
                y_field="payload",
                y_column=0,
                X_source="growth",
                X_field="payload",
                X_columns=(1,),
                variables=["x"],
            ),
            jarque_bera_test_step(
                "jb_growth",
                source="growth",
                field="payload",
                column=0,
            ),
        ]
    )
    reference = cast(SolvedModel, object())

    python_result = pipeline.run(reference=reference, n_rep=n_rep, verbosity=0)
    lowered = pipeline.lower_native(reference=reference, n_rep=n_rep, n_jobs=1)
    native_result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
        profile_steps=True,
    )

    assert native_result.status == 0
    assert lowered.allocation.failure_step_by_rep.tolist() == [-1] * n_rep
    assert native_result.step_counts_by_worker.sum(axis=0).tolist() == [
        n_rep,
        n_rep,
        n_rep,
        n_rep,
    ]

    growth_layout = lowered.plan["growth"].out_fields["payload"]
    growth = (
        lowered.allocation.steps["growth"]
        .float_retained[
            :, growth_layout.offset : growth_layout.offset + growth_layout.flat_count
        ]
        .reshape(n_rep, *growth_layout.shape)
    )
    expected_growth = np.diff(np.log(observables), axis=1)
    np.testing.assert_allclose(growth, expected_growth)

    ols_layout = lowered.plan["ols"].out_fields["coef"]
    coefficients = (
        lowered.allocation.steps["ols"]
        .float_retained[
            :, ols_layout.offset : ols_layout.offset + ols_layout.flat_count
        ]
        .reshape(n_rep, *ols_layout.shape)
    )
    np.testing.assert_allclose(
        coefficients,
        python_result.coefficient_traces["ols"],
        rtol=1e-10,
        atol=1e-12,
    )

    jb_layout = lowered.plan["jb_growth"].out_fields["statistic"]
    statistics = lowered.allocation.steps["jb_growth"].float_retained[
        :, jb_layout.offset
    ]
    np.testing.assert_allclose(
        statistics,
        python_result.statistic_traces["jb_growth"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_native_lowering_runs_all_regression_kinds() -> None:
    n_rep, n = 3, 48
    rng = np.random.default_rng(20260801)
    X = rng.normal(size=(n_rep, n, 2))
    y = (
        0.5
        + 1.25 * X[:, :, :1]
        - 0.75 * X[:, :, 1:]
        + rng.normal(scale=0.1, size=(n_rep, n, 1))
    )
    kinds = {
        "ols": {},
        "ridge": {"alpha": 0.3},
        "ridge_gs": {"start": 0.01, "stop": 1.0, "num": 5, "criterion": "aic"},
        "lasso": {"alpha": 0.03, "max_iter": 2000},
        "lasso_gs": {"start": 0.01, "stop": 1.0, "num": 5, "max_iter": 2000},
        "elastic_net": {"alpha": 0.03, "l1_ratio": 0.4, "max_iter": 2000},
        "elastic_net_gs": {
            "start": 0.01,
            "stop": 1.0,
            "num": 5,
            "l1_ratio": 0.4,
            "criterion": "aic",
            "max_iter": 2000,
        },
    }
    pipeline = MCPipeline(
        [
            raw_model_data_step("data", states=X, observables=y),
            *[
                regression_step(
                    name,
                    y_source="data",
                    y_field="observables",
                    X_source="data",
                    X_field="states",
                    kind=name,
                    **kwargs,
                )
                for name, kwargs in kinds.items()
            ],
        ]
    )
    reference = cast(SolvedModel, object())

    expected = pipeline.run(reference=reference, n_rep=n_rep, verbosity=0)
    lowered = pipeline.lower_native(reference=reference, n_rep=n_rep, n_jobs=1)
    result = run_native(lowered.allocation, lowered.steps, lowered.input_bindings)

    assert result.status == 0
    for name in kinds:
        layout = lowered.plan[name].out_fields["coef"]
        actual = (
            lowered.allocation.steps[name]
            .float_retained[:, layout.offset : layout.offset + layout.flat_count]
            .reshape(n_rep, *layout.shape)
        )
        np.testing.assert_allclose(
            actual,
            expected.coefficient_traces[name],
            rtol=1e-9,
            atol=1e-11,
        )


def test_native_lowering_runs_all_diagnostic_kinds() -> None:
    n_rep, n = 3, 64
    rng = np.random.default_rng(20260802)
    X = rng.normal(size=(n_rep, n, 2))
    y = (
        0.5
        + 1.25 * X[:, :, :1]
        - 0.75 * X[:, :, 1:]
        + rng.normal(scale=0.1, size=(n_rep, n, 1))
    )
    pipeline = MCPipeline(
        [
            raw_model_data_step("data", states=X, observables=y),
            wald_test_step(
                "wald",
                source="data",
                field="states",
                target=np.zeros(2, dtype=np.float64),
                bandwidth="auto",
            ),
            wald_test_step(
                "wald_covariance",
                source="data",
                field="states",
                target=np.eye(2, dtype=np.float64),
                kind="covariance",
                kernel="parzen",
                bandwidth=3,
            ),
            wald_test_step(
                "wald_second_moment",
                source="data",
                field="states",
                target=np.eye(2, dtype=np.float64),
                kind="second_moment",
                kernel="qs",
                bandwidth="andrews",
            ),
            ljung_box_test_step(
                "lb", source="data", field="observables", column=0, lags=4
            ),
            jarque_bera_test_step("jb", source="data", field="observables", column=0),
            breusch_pagan_test_step(
                "bp",
                residuals_source="data",
                residuals_field="observables",
                X_source="data",
                X_field="states",
                robust=True,
            ),
            breusch_godfrey_test_step(
                "bg",
                residuals_source="data",
                residuals_field="observables",
                X_source="data",
                X_field="states",
                lags=2,
            ),
            cusum_test_step(
                "cusum",
                y_source="data",
                y_field="observables",
                X_source="data",
                X_field="states",
            ),
            cusumsq_test_step(
                "cusumsq",
                y_source="data",
                y_field="observables",
                X_source="data",
                X_field="states",
            ),
            chow_test_step(
                "chow",
                y_source="data",
                y_field="observables",
                X_source="data",
                X_field="states",
                t_break=32,
            ),
        ]
    )
    reference = cast(SolvedModel, object())

    expected = pipeline.run(reference=reference, n_rep=n_rep, verbosity=0)
    lowered = pipeline.lower_native(reference=reference, n_rep=n_rep, n_jobs=1)
    result = run_native(lowered.allocation, lowered.steps, lowered.input_bindings)

    assert result.status == 0
    specs = lowered.test_result_specs
    assert set(specs) == {
        "wald",
        "wald_covariance",
        "wald_second_moment",
        "lb",
        "jb",
        "bp",
        "bg",
        "cusum",
        "cusumsq",
        "chow",
    }
    assert specs["wald"].dist is ReferenceDistribution.CHI2
    assert specs["wald"].df == 2
    assert specs["wald_covariance"].df == 3
    assert specs["wald_second_moment"].df == 3
    assert specs["lb"].df == 4
    assert specs["jb"].dist is ReferenceDistribution.JB_LOOKUP
    assert specs["jb"].df == n
    assert specs["bp"].df == 2
    assert specs["bg"].df == 2
    assert specs["cusum"].dist is ReferenceDistribution.CUSUM
    assert np.isnan(specs["cusum"].df)
    assert specs["cusumsq"].df == n - 2
    assert specs["chow"].dist is ReferenceDistribution.F
    assert specs["chow"].df == (2, n - 4)
    assert all(spec.pval_method is PvalMethod.SF for spec in specs.values())
    for name in (
        "wald",
        "wald_covariance",
        "wald_second_moment",
        "lb",
        "jb",
        "bp",
        "bg",
        "cusum",
        "cusumsq",
        "chow",
    ):
        layout = lowered.plan[name].out_fields["statistic"]
        actual = lowered.allocation.steps[name].float_retained[:, layout.offset]
        np.testing.assert_allclose(
            actual,
            expected.statistic_traces[name],
            rtol=1e-9,
            atol=1e-11,
        )


def test_native_diagnostic_status_is_retained_not_a_runner_failure() -> None:
    n_rep, n = 2, 4
    observables = np.arange(n_rep * n, dtype=np.float64).reshape(n_rep, n, 1)
    pipeline = MCPipeline(
        [
            raw_model_data_step("data", observables=observables),
            jarque_bera_test_step("jb", source="data", field="observables", column=0),
        ]
    )
    reference = cast(SolvedModel, object())

    lowered = pipeline.lower_native(reference=reference, n_rep=n_rep, n_jobs=1)
    result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
        fail_fast=True,
    )

    assert result.status == 0
    assert lowered.allocation.failure_step_by_rep.tolist() == [-1] * n_rep
    status_layout = lowered.plan["jb"].out_fields["status"]
    actual_status = lowered.allocation.steps["jb"].int_retained[:, status_layout.offset]
    np.testing.assert_array_equal(actual_status, int(TestStatus.INSUFFICIENT_SAMPLES))


def test_native_lowering_runs_first_order_simulation_with_observables() -> None:
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    solved = solver.solve(solver.compile())
    T = 7
    shocks = {name: np.linspace(0.01, 0.03, T) for name in solved.compiled.shock_names}
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim",
                target="reference",
                T=T,
                shocks=shocks,
                observables=True,
            )
        ]
    )

    lowered = pipeline.lower_native(reference=solved, n_rep=2, n_jobs=1)
    native_result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
    )

    assert native_result.status == 0
    expected = solved.sim(T, shocks=shocks, observables=True)
    for field, expected_values in (
        ("states", expected["_X"]),
        (
            "observables",
            np.column_stack(
                [expected[name] for name in solved.compiled.observable_names]
            ),
        ),
    ):
        layout = lowered.plan["sim"].out_fields[field]
        actual = (
            lowered.allocation.steps["sim"]
            .float_retained[0, layout.offset : layout.offset + layout.flat_count]
            .reshape(layout.shape)
        )
        np.testing.assert_allclose(actual, expected_values, rtol=1e-12, atol=1e-12)


def test_native_lowering_runs_second_order_simulation() -> None:
    hx = np.array([[0.5, 0.1], [0.0, 0.8]], dtype=np.float64)
    gx = np.array([[2.0, -1.0]], dtype=np.float64)
    bx = np.array([[1.0], [0.25]], dtype=np.float64)
    hxx = np.array(
        [
            [[0.2, 0.1], [0.1, -0.2]],
            [[0.0, 0.3], [0.3, 0.1]],
        ],
        dtype=np.float64,
    )
    gxx = np.array([[[0.4, -0.1], [-0.1, 0.2]]], dtype=np.float64)
    hss = np.array([0.01, -0.02], dtype=np.float64)
    gss = np.array([0.03], dtype=np.float64)
    compiled = SimpleNamespace(
        var_names=["e", "k", "c"],
        n_exog=1,
        n_state=2,
        observable_names=[],
        calib_params=[],
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={})),
    )
    policy = PerturbationSolution(
        p=hx,
        f=gx,
        stab=0,
        eig=np.empty(0, dtype=np.complex128),
        order=2,
        hxx=hxx,
        gxx=gxx,
        hss=hss,
        gss=gss,
        steady_state=np.zeros(3, dtype=np.float64),
    )
    solved = SolvedModel(
        compiled=compiled,
        policy=policy,
        A=np.eye(3, dtype=np.float64),
        B=np.vstack([bx, np.zeros((1, 1), dtype=np.float64)]),
    )
    pipeline = MCPipeline(
        [simulation_step("sim", target="reference", T=6, observables=False)]
    )

    lowered = pipeline.lower_native(reference=solved, n_rep=2, n_jobs=1)
    native_result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
    )

    assert native_result.status == 0
    layout = lowered.plan["sim"].out_fields["states"]
    actual = (
        lowered.allocation.steps["sim"]
        .float_retained[0, layout.offset : layout.offset + layout.flat_count]
        .reshape(layout.shape)
    )
    np.testing.assert_allclose(
        actual,
        solved._simulate_state_matrix(6),
        rtol=1e-12,
        atol=1e-12,
    )


def test_native_lowering_runs_linear_and_extended_filters() -> None:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    solved = solver.solve(solver.compile())
    T = 8
    expected_simulation = solved.sim(T, observables=True)
    expected_y = np.column_stack(
        [expected_simulation[name] for name in solved.compiled.observable_names]
    )

    for mode in ("linear", "extended"):
        pipeline = MCPipeline(
            [
                simulation_step("sim", target="reference", T=T, observables=True),
                reference_filter_step("filter", filter_mode=mode),
            ]
        )
        lowered = pipeline.lower_native(reference=solved, n_rep=1, n_jobs=1)
        native_result = run_native(
            lowered.allocation,
            lowered.steps,
            lowered.input_bindings,
        )

        assert native_result.status == 0
        expected_filter = solved._kalman_raw(y=expected_y, filter_mode=mode)
        for field in ("x_pred", "x_filt", "P_pred", "innov", "loglik"):
            layout = lowered.plan["filter"].out_fields[field]
            actual = (
                lowered.allocation.steps["filter"]
                .float_retained[0, layout.offset : layout.offset + layout.flat_count]
                .reshape(layout.shape)
            )
            np.testing.assert_allclose(
                actual,
                getattr(expected_filter, field),
                rtol=1e-10,
                atol=1e-12,
            )


def test_native_lowering_reorders_linear_filter_inputs_and_overrides() -> None:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    solved = solver.solve(solver.compile())
    T = 8
    requested = ["Rate", "OutGap"]
    n_var = len(solved.compiled.var_names)
    shocks = {
        name: np.linspace(0.01, 0.03, T, dtype=np.float64)
        for name in solved.compiled.shock_names
    }
    x0 = np.full(n_var, 0.05, dtype=np.float64)
    P0 = 0.2 * np.eye(n_var, dtype=np.float64)
    R = np.array([[0.1, 0.02], [0.02, 0.3]], dtype=np.float64)
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim",
                target="reference",
                T=T,
                shocks=shocks,
                observables=True,
            ),
            reference_filter_step(
                "filter",
                filter_mode="linear",
                observables=requested,
                x0=x0,
                P0=P0,
                R=R,
                return_shocks=True,
            ),
        ]
    )

    lowered = pipeline.lower_native(reference=solved, n_rep=1, n_jobs=1)
    native_result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
    )

    assert native_result.status == 0
    simulated = solved.sim(T, shocks=shocks, observables=True)
    expected_y = np.column_stack([simulated[name] for name in requested])
    expected_filter = solved._kalman_raw(
        y=expected_y,
        filter_mode="linear",
        observables=requested,
        x0=x0,
        P0=P0,
        R=R,
        return_shocks=True,
    )
    for field in ("x_pred", "P_filt", "innov", "eps_hat", "loglik"):
        layout = lowered.plan["filter"].out_fields[field]
        actual = (
            lowered.allocation.steps["filter"]
            .float_retained[0, layout.offset : layout.offset + layout.flat_count]
            .reshape(layout.shape)
        )
        np.testing.assert_allclose(
            actual,
            getattr(expected_filter, field),
            rtol=1e-10,
            atol=1e-12,
        )


def test_native_lowering_runs_unscented_filter_with_rbc_fixture() -> None:
    model, _ = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    n_var = len(model.variables.variables)
    solver = DSGESolver(
        model,
        KalmanConfig(
            R=np.array([[0.01]], dtype=np.float64),
            P0=0.1 * np.eye(n_var, dtype=np.float64),
        ),
    )
    solved = solver.solve(solver.compile(), order=2)
    T = 5
    y = np.zeros((T, len(solved.compiled.observable_names)), dtype=np.float64)
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "data",
                observables=y,
                observable_names=solved.compiled.observable_names,
            ),
            reference_filter_step("filter", filter_mode="unscented"),
        ]
    )

    lowered = pipeline.lower_native(reference=solved, n_rep=1, n_jobs=1)
    native_result = run_native(
        lowered.allocation,
        lowered.steps,
        lowered.input_bindings,
    )

    assert native_result.status == 0
    expected_filter = solved._kalman_raw(y=y, filter_mode="unscented")
    for field in ("x_pred", "x_filt", "P_pred", "innov", "loglik", "x2_filt"):
        layout = lowered.plan["filter"].out_fields[field]
        actual = (
            lowered.allocation.steps["filter"]
            .float_retained[0, layout.offset : layout.offset + layout.flat_count]
            .reshape(layout.shape)
        )
        np.testing.assert_allclose(
            actual,
            getattr(expected_filter, field),
            rtol=1e-10,
            atol=1e-12,
        )
