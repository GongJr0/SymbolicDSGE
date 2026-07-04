# type: ignore
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.interface import KalmanInterface

EXPECTED_CANONICAL_ORDER = ["u", "v", "r", "Pi", "x", "r_star"]
EXPECTED_IDX = {name: i for i, name in enumerate(EXPECTED_CANONICAL_ORDER)}


def _write_misordered_test_model(tmp_path):
    with open("MODELS/test.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Deliberately put controls and the unshocked state before shocked states.
    # The canonical solver layout should still be [shocked states, unshocked
    # states, controls].
    data["variables"] = ["Pi", "x", "r_star", "u", "v", "r"]
    data["kalman"] = {
        "P0": {
            "mode": "diag",
            "scale": 1.0,
            "diag": {
                "u": 1.0,
                "v": 2.0,
                "r": 3.0,
                "Pi": 4.0,
                "x": 5.0,
                "r_star": 6.0,
            },
        },
    }

    path = tmp_path / "misordered_test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _compile_misordered_test_model(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)
    return solver.compile()


def test_compile_default_layout_canonicalizes_misordered_yaml_variables(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)

    assert compiled.var_names == EXPECTED_CANONICAL_ORDER
    assert compiled.idx == EXPECTED_IDX
    assert compiled.layout.declared_names == ("Pi", "x", "r_star", "u", "v", "r")
    assert compiled.layout.canonical_names == tuple(EXPECTED_CANONICAL_ORDER)
    assert compiled.layout.exo_state_names == ("u", "v")
    assert compiled.layout.endo_state_names == ("r",)
    assert compiled.layout.control_names == ("Pi", "x", "r_star")
    assert compiled.n_exog == 2
    assert compiled.n_state == 3


def test_measurement_dispatchers_accept_canonical_state_order_after_yaml_reorder(
    tmp_path,
):
    compiled = _compile_misordered_test_model(tmp_path)
    params = np.array(
        [compiled.config.calibration.parameters[p] for p in compiled.calib_params],
        dtype=np.float64,
    )
    state = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)

    measurement = compiled.construct_measurement_array_func(["Infl", "Rate"])(
        state,
        params,
    )
    jacobian = compiled.construct_observable_jacobian_array_func(["Infl", "Rate"])(
        np.zeros_like(state),
        params,
    )

    expected_jacobian = np.zeros((2, len(EXPECTED_CANONICAL_ORDER)), dtype=np.float64)
    expected_jacobian[0, EXPECTED_IDX["Pi"]] = 1.0
    expected_jacobian[1, EXPECTED_IDX["r"]] = 1.0

    np.testing.assert_allclose(measurement, np.array([43.25, 30.0]))
    np.testing.assert_allclose(jacobian, expected_jacobian)


def test_kalman_order_sensitive_matrices_use_canonical_compiled_layout(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)
    solved = SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(f=np.zeros((3, 3), dtype=np.float64)),
        A=np.eye(len(EXPECTED_CANONICAL_ORDER), dtype=np.float64),
        B=np.vstack(
            [
                np.eye(2, dtype=np.float64),
                np.zeros((len(EXPECTED_CANONICAL_ORDER) - 2, 2), dtype=np.float64),
            ]
        ),
    )
    ki = KalmanInterface.__new__(KalmanInterface)
    ki.model = solved

    np.testing.assert_allclose(
        ki._build_Q(),
        np.diag([0.50**2, 0.25**2]).astype(np.float64),
    )
    np.testing.assert_allclose(
        ki._build_P0(),
        np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64),
    )


def test_simulation_shock_unpack_accepts_shocks_by_canonical_exogenous_names(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)
    solved = SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(f=np.zeros((3, 3), dtype=np.float64)),
        A=np.eye(len(EXPECTED_CANONICAL_ORDER), dtype=np.float64),
        B=np.vstack(
            [
                np.eye(2, dtype=np.float64),
                np.zeros((len(EXPECTED_CANONICAL_ORDER) - 2, 2), dtype=np.float64),
            ]
        ),
    )

    unpacked = solved._shock_unpack(
        {
            "u": np.array([1.0, 2.0], dtype=np.float64),
            "v": np.array([3.0, 4.0], dtype=np.float64),
        }
    )

    assert [idx for idx, _ in unpacked] == [EXPECTED_IDX["u"], EXPECTED_IDX["v"]]
    np.testing.assert_allclose(unpacked[0][1], np.array([1.0, 2.0]))
    np.testing.assert_allclose(unpacked[1][1], np.array([3.0, 4.0]))


def test_compile_rejects_layout_count_overrides_that_disagree(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="n_exog=.*inferred"):
        solver.compile(n_state=3, n_exog=1)

    with pytest.raises(ValueError, match="n_state=.*inferred"):
        solver.compile(n_state=2, n_exog=2)


def test_compile_rejects_explicit_order_with_wrong_state_groups(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="first n_exog variables"):
        solver.compile(
            variable_order=["Pi", "x", "r_star", "u", "v", "r"],
            n_state=3,
            n_exog=2,
        )
