from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BACKEND_PATH = Path(__file__).resolve().parents[2] / "sdsge-ui" / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from sdsge_ui_backend.app import create_app  # noqa: E402
from sdsge_ui_backend.mc import (  # noqa: E402
    serialize_pipeline_result,
    validate_pipeline_spec,
)
from sdsge_ui_backend.mc_schemas import MCPipelineSpec  # noqa: E402
from sdsge_ui_backend.schemas import ArrayEnvelope  # noqa: E402
from sdsge_ui_backend.serializers import decode_array, encode_array  # noqa: E402

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.result import MCResult
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
        "regression",
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
                    "step_type": "wald",
                    "name": "diagnostic",
                    "params": {"source": "std_innov", "target_vector": [0.0]},
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
