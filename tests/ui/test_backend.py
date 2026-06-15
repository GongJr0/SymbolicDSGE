from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.optimize import OptimizeResult

from SymbolicDSGE.estimation.results import MCMCResult, OptimizationResult
from SymbolicDSGE.ui.app import create_app
from SymbolicDSGE.ui.estimation import (
    build_estimation_inputs,
    serialize_estimation_result,
)
from SymbolicDSGE.ui.mc import (
    serialize_pipeline_result,
    validate_pipeline_spec,
)
from SymbolicDSGE.ui.mc_schemas import MCPipelineSpec
from SymbolicDSGE.ui.schemas import ArrayEnvelope, EstimationParameterSpec
from SymbolicDSGE.ui.serializers import decode_array, encode_array

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.result import MCResult
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE.monte_carlo import MCContext, MCData, MCPipelineResult
from SymbolicDSGE.regression.ols import MCRegressionResult, ols


def test_array_envelope_round_trips_float64_payload() -> None:
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)

    envelope = ArrayEnvelope.model_validate(encode_array(arr))
    out = decode_array(envelope)

    assert out.dtype == np.float64
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, arr)


def test_ui_backend_loads_solves_and_simulates_model() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    loaded = client.post(
        "/api/model/load-yaml",
        json={"role": "reference", "path": "MODELS/test.yaml"},
    )
    assert loaded.status_code == 200
    loaded_body = loaded.json()
    assert loaded_body["loaded"] is True
    assert loaded_body["solved"] is False
    assert loaded_body["name"] == "TEST"
    assert 'name: "TEST"' in loaded_body["raw_yaml"]
    assert loaded_body["shock_specs"] == [
        {"shock": "e_u", "target": "u", "std_param": "sig_u", "std_value": 0.5},
        {"shock": "e_v", "target": "v", "std_param": "sig_v", "std_value": 0.25},
    ]
    assert loaded_body["shock_corr_specs"] == []

    solved = client.post(
        "/api/model/solve",
        json={
            "role": "reference",
            "compile_kwargs": {"n_state": 3, "n_exog": 2},
        },
    )
    assert solved.status_code == 200
    solved_body = solved.json()
    assert solved_body["solved"] is True
    assert solved_body["n_state"] == 3
    assert solved_body["n_exog"] == 2

    simulated = client.post(
        "/api/run/sim",
        json={"role": "reference", "T": 5, "observables": True},
    )
    assert simulated.status_code == 200
    sim_body = simulated.json()
    assert sim_body["kind"] == "sim"
    assert sim_body["role"] == "reference"
    names = {series["name"] for series in sim_body["series"]}
    assert "_X" in names

    x_series = next(series for series in sim_body["series"] if series["name"] == "_X")
    x_arr = decode_array(ArrayEnvelope.model_validate(x_series["array"]))
    assert x_arr.shape == (6, solved_body["A_shape"][0])

    fetched = client.get(f"/api/run/{sim_body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == sim_body["run_id"]

    shocked = client.post(
        "/api/run/sim",
        json={
            "role": "reference",
            "T": 5,
            "observables": False,
            "shocks": {"u": encode_array(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))},
        },
    )
    assert shocked.status_code == 200
    shock_body = shocked.json()
    u_series = next(series for series in shock_body["series"] if series["name"] == "u")
    u_arr = decode_array(ArrayEnvelope.model_validate(u_series["array"]))
    assert np.max(np.abs(u_arr)) > 0.0

    generated = client.post(
        "/api/run/sim",
        json={
            "role": "reference",
            "T": 5,
            "observables": False,
            "shock_generation": {"dist": "norm", "seed": 10, "loc": 0.0},
            "shock_params": {"std": {"e_u": 2.0, "e_v": 1.0}, "corr": {}},
        },
    )
    assert generated.status_code == 200
    generated_body = generated.json()
    generated_u = next(
        series for series in generated_body["series"] if series["name"] == "u"
    )
    generated_u_arr = decode_array(ArrayEnvelope.model_validate(generated_u["array"]))
    assert np.max(np.abs(generated_u_arr)) > 0.0

    generated_t = client.post(
        "/api/run/sim",
        json={
            "role": "reference",
            "T": 5,
            "observables": False,
            "shock_generation": {"dist": "t", "seed": 10, "loc": 0.0, "df": 5.0},
        },
    )
    assert generated_t.status_code == 200


def test_ui_backend_loads_yaml_content_and_reports_user_errors() -> None:
    client = TestClient(create_app())
    content = Path("MODELS/test.yaml").read_text(encoding="utf-8")

    loaded = client.post(
        "/api/model/load-yaml",
        json={"role": "dgp", "content": content},
    )
    assert loaded.status_code == 200
    assert loaded.json()["source"] == "<content>"

    unsolved_sim = client.post(
        "/api/run/sim",
        json={"role": "dgp", "T": 3, "observables": False},
    )
    assert unsolved_sim.status_code == 400
    detail = unsolved_sim.json()["detail"]
    assert detail["error_type"] == "ValueError"
    assert "does not have a solved model" in detail["message"]

    bad_load = client.post(
        "/api/model/load-yaml",
        json={"role": "reference", "path": "MODELS/test.yaml", "content": content},
    )
    assert bad_load.status_code == 400
    assert bad_load.json()["detail"]["error_type"] == "ValueError"

    missing_run = client.get("/api/run/not-a-run")
    assert missing_run.status_code == 404


def test_ui_backend_reports_configured_shock_correlations() -> None:
    client = TestClient(create_app())

    loaded = client.post(
        "/api/model/load-yaml",
        json={"role": "reference", "path": "MODELS/POST82.yaml"},
    )

    assert loaded.status_code == 200
    assert loaded.json()["shock_corr_specs"] == [
        {
            "pair": ["e_g", "e_z"],
            "key": "e_g,e_z",
            "corr_param": "rho_gz",
            "corr_value": 0.36,
        }
    ]


def test_ui_estimation_inputs_build_scalar_priors_and_validate_selection() -> None:
    parameters = [
        EstimationParameterSpec.model_validate(
            {
                "name": "beta",
                "estimate": True,
                "initial": 0.99,
                "lower": 0.9,
                "upper": 1.0,
                "prior": {
                    "distribution": "normal",
                    "parameters": {"mean": 0.99, "std": 0.01},
                    "transform": "identity",
                },
            }
        ),
        EstimationParameterSpec(name="sigma", estimate=False, initial=1.0),
    ]

    names, theta0, priors, bounds = build_estimation_inputs(parameters, method="map")

    assert names == ["beta"]
    assert theta0 == {"beta": 0.99}
    assert priors is not None
    assert set(priors) == {"beta"}
    assert bounds == [(0.9, 1.0)]

    with np.testing.assert_raises_regex(ValueError, "Select at least one parameter"):
        build_estimation_inputs(
            [EstimationParameterSpec(name="beta", initial=0.99)],
            method="mle",
        )


def test_ui_estimation_serializes_mcmc_traces_for_charts() -> None:
    result = MCMCResult(
        param_names=["beta", "sigma"],
        samples=np.array([[0.98, 1.1], [0.99, 1.0]], dtype=np.float64),
        logpost_trace=np.array([-3.0, -2.5], dtype=np.float64),
        accept_rate=np.float64(0.5),
        n_draws=2,
        burn_in=1,
        thin=1,
    )

    payload = serialize_estimation_result(result)

    assert payload["samples"] == {"beta": [0.98, 0.99], "sigma": [1.1, 1.0]}
    assert payload["logpost_trace"] == [-3.0, -2.5]


def test_ui_backend_dispatches_estimation_and_estimate_and_solve(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)
    loaded = client.post(
        "/api/model/load-yaml",
        json={"role": "reference", "path": "MODELS/test.yaml"},
    )
    assert loaded.status_code == 200
    assert loaded.json()["parameter_values"]["beta"] == 0.99

    catalog = client.get("/api/estimation/catalog")
    assert catalog.status_code == 200
    catalog_body = catalog.json()
    assert catalog_body["distributions"]["normal"]["std"] == 1.0
    assert "lkj_chol" not in catalog_body["distributions"]
    assert "cholesky_corr" not in catalog_body["transforms"]

    solved = client.post(
        "/api/model/solve",
        json={
            "role": "reference",
            "compile_kwargs": {"n_state": 3, "n_exog": 2},
        },
    )
    assert solved.status_code == 200

    slot = app.state.ui_session._slot("reference")
    assert slot.solver is not None
    assert slot.solved is not None
    solved_model = slot.solved
    result = OptimizationResult(
        kind="mle",
        x=np.array([0.98], dtype=np.float64),
        theta={"beta": np.float64(0.98)},
        success=True,
        message="converged",
        fun=np.float64(1.25),
        loglik=np.float64(-1.25),
        logprior=np.float64(0.0),
        logpost=np.float64(-1.25),
        nfev=4,
        nit=2,
        raw=OptimizeResult(),
    )
    captured: dict[str, object] = {}

    def fake_estimate(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(slot.solver, "estimate", fake_estimate)
    request = {
        "role": "reference",
        "method": "mle",
        "y": [[3.2, 0.1], [3.3, 0.2]],
        "observables": ["Infl", "Rate"],
        "parameters": [
            {
                "name": "beta",
                "estimate": True,
                "initial": 0.99,
                "lower": 0.9,
                "upper": 1.0,
            }
        ],
        "method_kwargs": {"options": {"maxiter": 25}},
    }
    response = client.post("/api/run/estimation", json=request)

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "estimation"
    assert body["solved"] is False
    assert body["result"]["theta"] == {"beta": 0.98}
    assert captured["method"] == "mle"
    assert captured["theta0"] == {"beta": 0.99}
    assert captured["estimated_params"] == ["beta"]
    assert captured["bounds"] == [(0.9, 1.0)]
    assert captured["options"] == {"maxiter": 25}

    invalid = client.post(
        "/api/run/estimation",
        json={**request, "method_kwargs": {"method": "Powell"}},
    )
    assert invalid.status_code == 400
    assert "reserved arguments" in invalid.json()["detail"]["message"]

    def fake_estimate_and_solve(**kwargs):
        captured.update(kwargs)
        return result, solved_model

    monkeypatch.setattr(slot.solver, "estimate_and_solve", fake_estimate_and_solve)
    estimate_and_solve = client.post(
        "/api/run/estimation",
        json={**request, "estimate_and_solve": True},
    )

    assert estimate_and_solve.status_code == 200
    assert estimate_and_solve.json()["solved"] is True
    assert app.state.ui_session.solved_model("reference") is solved_model


def test_ui_backend_validates_and_runs_monte_carlo_pipeline() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    catalog = client.get("/api/mc/catalog")
    assert catalog.status_code == 200
    assert {step["step_type"] for step in catalog.json()["steps"]} == {
        "simulation",
        "filter",
        "wald",
        "ljung_box",
        "jarque_bera",
        "breusch_pagan",
        "breusch_godfrey",
        "cusum",
        "cusumsq",
        "chow",
        "regression",
        "standardize",
        "log",
        "log_diff",
        "diff",
        "rolling_mean",
        "rolling_std",
        "rolling_var",
    }

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {
                    "T": 8,
                    "observables": True,
                    "distribution": "norm",
                    "seed": 10,
                },
            }
        ],
        "edges": [],
    }
    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json() == {"valid": True, "order": ["sim"]}

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 3, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["kind"] == "mc"
    assert body["n_rep"] == 3
    assert body["n_successful"] == 3
    assert body["succeeded"] is True
    assert body["step_it_s"]["datagen"] > 0
    assert body["step_counts"]["datagen"] == 3
    assert body["data_summaries"]["states"]["n_rep"] == 3
    assert body["data_summaries"]["states"]["n_finite"] > 0

    fetched = client.get(f"/api/run/{body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == body["run_id"]


def test_ui_backend_accepts_fanout_and_rejects_terminal_forward_links() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 8},
            },
            {
                "id": "a",
                "step_type": "ljung_box",
                "name": "a",
                "params": {
                    "source": "states",
                    "column": [0],
                    "lags": 1,
                },
            },
            {
                "id": "b",
                "step_type": "ljung_box",
                "name": "b",
                "params": {
                    "source": "states",
                    "column": [1],
                    "lags": 1,
                },
            },
        ],
        "edges": [
            {"source": "sim", "target": "a"},
            {"source": "sim", "target": "b"},
        ],
    }

    response = client.post("/api/mc/validate", json=pipeline)
    terminal_forward = client.post(
        "/api/mc/validate",
        json={
            **pipeline,
            "edges": [
                {"source": "sim", "target": "a"},
                {"source": "a", "target": "b"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["order"] == ["sim", "a", "b"]
    assert terminal_forward.status_code == 400
    assert "Terminal step" in terminal_forward.json()["detail"]["message"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    assert set(run.json()["test_summaries"]) == {"a", "b"}


def test_ui_backend_runs_jarque_bera_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 12},
            },
            {
                "id": "jb",
                "step_type": "jarque_bera",
                "name": "normality",
                "params": {
                    "source": "states",
                    "column": [0],
                    "burn_in": 1,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "jb"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "jb"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"normality"}
    summary = body["test_summaries"]["normality"]
    assert summary["distribution"] == "jb_lookup"
    assert summary["df"] == 12
    assert summary["n"] == 2


def test_ui_backend_runs_breusch_pagan_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 20},
            },
            {
                "id": "bp",
                "step_type": "breusch_pagan",
                "name": "heteroskedasticity",
                "params": {
                    "residual_source": "states",
                    "X_source": "states",
                    "residual_col": [0],
                    "X_columns": [1],
                    "burn_in": 1,
                    "robust": True,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "bp"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "bp"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"heteroskedasticity"}
    summary = body["test_summaries"]["heteroskedasticity"]
    assert summary["test_name"] == "heteroskedasticity"
    assert summary["distribution"] == "chi2"
    assert summary["df"] == 1
    assert summary["n"] == 2


def test_ui_backend_runs_breusch_godfrey_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 20},
            },
            {
                "id": "bg",
                "step_type": "breusch_godfrey",
                "name": "serial_correlation",
                "params": {
                    "residual_source": "states",
                    "X_source": "states",
                    "residual_col": [0],
                    "X_columns": [1],
                    "burn_in": 1,
                    "lags": 2,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "bg"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "bg"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"serial_correlation"}
    summary = body["test_summaries"]["serial_correlation"]
    assert summary["test_name"] == "serial_correlation"
    assert summary["distribution"] == "chi2"
    assert summary["df"] == 2
    assert summary["n"] == 2


def test_ui_backend_runs_cusum_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 30},
            },
            {
                "id": "cs",
                "step_type": "cusum",
                "name": "stability",
                "params": {
                    "y_source": "states",
                    "x_source": "states",
                    "y_column": [0],
                    "X_columns": [1],
                    "burn_in": 1,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "cs"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "cs"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"stability"}
    summary = body["test_summaries"]["stability"]
    assert summary["test_name"] == "stability"
    assert summary["distribution"] == "cusum"
    assert summary["n"] == 2


def test_ui_backend_runs_cusumsq_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 30},
            },
            {
                "id": "csq",
                "step_type": "cusumsq",
                "name": "variance_stability",
                "params": {
                    "y_source": "states",
                    "x_source": "states",
                    "y_column": [0],
                    "X_columns": [1],
                    "burn_in": 1,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "csq"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "csq"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"variance_stability"}
    summary = body["test_summaries"]["variance_stability"]
    assert summary["test_name"] == "variance_stability"
    assert summary["distribution"] == "cusumsq"
    assert summary["n"] == 2


def test_ui_backend_runs_chow_monte_carlo_step() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    pipeline = {
        "nodes": [
            {
                "id": "sim",
                "step_type": "simulation",
                "name": "datagen",
                "params": {"T": 30},
            },
            {
                "id": "ch",
                "step_type": "chow",
                "name": "structural_break",
                "params": {
                    "y_source": "states",
                    "x_source": "states",
                    "y_column": [0],
                    "X_columns": [1],
                    "t_break": 10,
                    "burn_in": 1,
                    "alpha": 0.1,
                },
            },
        ],
        "edges": [{"source": "sim", "target": "ch"}],
    }

    validated = client.post("/api/mc/validate", json=pipeline)
    assert validated.status_code == 200
    assert validated.json()["order"] == ["sim", "ch"]

    run = client.post(
        "/api/run/mc",
        json={"pipeline": pipeline, "n_rep": 2, "fail_fast": True},
    )
    assert run.status_code == 200
    body = run.json()
    assert body["succeeded"] is True
    assert set(body["test_summaries"]) == {"structural_break"}
    summary = body["test_summaries"]["structural_break"]
    assert summary["test_name"] == "structural_break"
    assert summary["distribution"] == "f"
    assert summary["n"] == 2


def test_ui_backend_normalizes_integer_or_keyword_mc_fields() -> None:
    client = TestClient(create_app())
    for role in ("reference", "dgp"):
        loaded = client.post(
            "/api/model/load-yaml",
            json={"role": role, "path": "MODELS/test.yaml"},
        )
        assert loaded.status_code == 200
        solved = client.post(
            "/api/model/solve",
            json={"role": role, "compile_kwargs": {"n_state": 3, "n_exog": 2}},
        )
        assert solved.status_code == 200

    valid = client.post(
        "/api/mc/validate",
        json={
            "nodes": [
                {
                    "id": "sim",
                    "step_type": "simulation",
                    "name": "datagen",
                    "params": {"T": 8, "seed_increment": "2"},
                }
            ],
            "edges": [],
        },
    )
    invalid = client.post(
        "/api/mc/validate",
        json={
            "nodes": [
                {
                    "id": "sim",
                    "step_type": "simulation",
                    "name": "datagen",
                    "params": {"T": 8, "seed_increment": "invalid"},
                }
            ],
            "edges": [],
        },
    )

    assert valid.status_code == 200
    assert invalid.status_code == 400
    assert "seed_increment" in invalid.json()["detail"]["message"]


def test_ui_backend_serializes_detailed_mc_summaries() -> None:
    X = np.arange(5, dtype=np.float64).reshape(-1, 1)
    y = np.array([0.0, 1.1, 1.9, 3.2, 3.8], dtype=np.float64)
    regressions = MCRegressionResult.from_results([ols(X, y), ols(X, y + 0.25)])
    tests = MCResult(
        test_name="diagnostic",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(1.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic_trace=np.array([0.5, 1.5], dtype=np.float64),
        status_trace=(TestStatus.OK, TestStatus.BAD_SHAPE),
    )
    context = MCContext(
        rep_idx=0,
        reference=None,  # type: ignore[arg-type]
        dgp=None,
        data=MCData(
            states=np.arange(12, dtype=np.float64).reshape(4, 3),
            raw={"x": np.arange(4, dtype=np.float64)},
        ),
    )
    result = MCPipelineResult(
        n_rep=2,
        n_successful=2,
        test_summaries={"diagnostic": tests},
        test_results=None,
        payloads=None,
        contexts=(context,),
        regression_summaries={"ols": regressions},
    )

    payload = serialize_pipeline_result(result, run_id="run")

    assert payload["test_summaries"]["diagnostic"]["statistic_ci"][0] is not None
    assert payload["test_summaries"]["diagnostic"]["rejection_ci"][0] is not None
    assert payload["test_summaries"]["diagnostic"]["status_trace"] == [0, -1]
    assert payload["test_summaries"]["diagnostic"]["status_counts"] == {
        "OK": 1,
        "BAD_SHAPE": 1,
    }
    assert payload["regression_summaries"]["ols"]["ols"] is not None
    assert (
        payload["regression_summaries"]["ols"]["coefficient_summaries"][0]["variable"]
        == "Intercept"
    )
    assert payload["data_summaries"]["states"]["shape"] == [4, 3]
    assert payload["data_summaries"]["raw:x"]["mean"] == 1.5


def test_ui_backend_binds_filter_dependencies_from_graph_edges() -> None:
    spec = MCPipelineSpec.model_validate(
        {
            "nodes": [
                {
                    "id": "sim",
                    "step_type": "simulation",
                    "name": "datagen",
                    "params": {"T": 8, "observables": True},
                },
                {"id": "filter", "step_type": "filter", "name": "renamed_filter"},
                {
                    "id": "test",
                    "step_type": "breusch_pagan",
                    "name": "diagnostic",
                    "params": {
                        "residual_source": "std_innov",
                        "X_source": "observables",
                        "residual_col": [0],
                        "X_columns": [0],
                    },
                },
            ],
            "edges": [
                {"source": "sim", "target": "filter"},
                {"source": "filter", "target": "test"},
            ],
        }
    )

    ordered = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)

    assert [node.id for node in ordered] == ["sim", "filter", "test"]
    assert ordered[-1].params["filter_key"] == "renamed_filter"

    direct = spec.model_copy(
        update={
            "edges": [
                spec.edges[0],
                spec.edges[1].model_copy(update={"source": "sim"}),
            ]
        }
    )
    with np.testing.assert_raises_regex(ValueError, "must link from a filter"):
        validate_pipeline_spec(direct, has_reference=True, has_dgp=True)
