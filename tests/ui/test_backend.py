from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BACKEND_PATH = Path(__file__).resolve().parents[2] / "sdsge-ui" / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from sdsge_ui_backend.app import create_app  # noqa: E402
from sdsge_ui_backend.schemas import ArrayEnvelope  # noqa: E402
from sdsge_ui_backend.serializers import decode_array, encode_array  # noqa: E402


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
