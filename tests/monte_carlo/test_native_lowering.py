from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE._ckernels.monte_carlo._runner import run as run_native
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.core.solver_backend import PerturbationSolution
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.operations.core import (
    raw_model_data_step,
    simulation_step,
)
from SymbolicDSGE.monte_carlo.operations.regressions import regression_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.operations.transforms import log_diff_step


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
    expected_growth = np.stack(
        [payload["growth"] for payload in python_result.payloads]
    )
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


def test_native_lowering_runs_first_order_simulation_with_observables() -> None:
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    solved = solver.solve(solver.compile())
    T = 7
    shocks = {
        name: np.linspace(0.01, 0.03, T)
        for name in solved.compiled.layout.exo_state_names
    }
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
