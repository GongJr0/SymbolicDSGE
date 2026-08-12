# type: ignore
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import FirstOrderSolvedModel
from SymbolicDSGE.kalman.interface import KalmanInterface
from SymbolicDSGE.core.solved_model.shocks import shock_unpack

# Shock states, then lag states, then the declared variables. The states are all
# compiler-minted: test.yaml lags u, v and r, and carries two shocks.
EXPECTED_CANONICAL_ORDER = [
    "e_u_st",
    "e_v_st",
    "u_lag1",
    "v_lag1",
    "r_lag1",
    "Pi",
    "x",
    "r_star",
    "u",
    "v",
    "r",
]
EXPECTED_IDX = {name: i for i, name in enumerate(EXPECTED_CANONICAL_ORDER)}
N_VAR = len(EXPECTED_CANONICAL_ORDER)


def _write_misordered_test_model(tmp_path):
    with open("MODELS/test.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Deliberately put controls and the unshocked variable first. The canonical
    # layout should still lead with the generated states.
    data["variables"] = ["Pi", "x", "r_star", "u", "v", "r"]
    data["kalman"] = {
        "P0": {
            "mode": "diag",
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
    assert compiled.layout.declared_names == (
        "Pi",
        "x",
        "r_star",
        "u",
        "v",
        "r",
        "u_lag1",
        "v_lag1",
        "r_lag1",
        "e_u_st",
        "e_v_st",
    )
    assert compiled.layout.canonical_names == tuple(EXPECTED_CANONICAL_ORDER)
    assert compiled.layout.exo_state_names == ("e_u_st", "e_v_st")
    assert compiled.layout.endo_state_names == ("u_lag1", "v_lag1", "r_lag1")
    assert compiled.layout.control_names == ("Pi", "x", "r_star", "u", "v", "r")
    assert compiled.n_exog == 2
    assert compiled.n_state == 5


def test_generated_variables_are_recorded_with_their_canonical_positions(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)

    assert compiled.layout.generated == {
        name: EXPECTED_IDX[name]
        for name in ("e_u_st", "e_v_st", "u_lag1", "v_lag1", "r_lag1")
    }
    # Every state is generated and no control is, which is what makes hiding them
    # a single filtration.
    assert set(compiled.layout.generated) == set(
        EXPECTED_CANONICAL_ORDER[: compiled.n_state]
    )


def test_shock_columns_are_named_by_shock_in_declaration_order(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)

    assert compiled.shock_names == ("e_u", "e_v")
    assert compiled.shock_idx == {"e_u": 0, "e_v": 1}


def test_measurement_dispatchers_accept_canonical_state_order_after_yaml_reorder(
    tmp_path,
):
    compiled = _compile_misordered_test_model(tmp_path)
    params = np.array(
        [compiled.config.calibration.parameters[p] for p in compiled.calib_params],
        dtype=np.float64,
    )
    state = np.zeros(N_VAR, dtype=np.float64)
    state[EXPECTED_IDX["Pi"]] = 40.0
    state[EXPECTED_IDX["r"]] = 30.0

    measurement = compiled.construct_measurement_array_func(["Infl", "Rate"])(
        state,
        params,
    )
    jacobian = compiled.construct_observable_jacobian_array_func(["Infl", "Rate"])(
        np.zeros_like(state),
        params,
    )

    expected_jacobian = np.zeros((2, N_VAR), dtype=np.float64)
    expected_jacobian[0, EXPECTED_IDX["Pi"]] = 1.0
    expected_jacobian[1, EXPECTED_IDX["r"]] = 1.0

    np.testing.assert_allclose(measurement, np.array([43.25, 30.0]))
    np.testing.assert_allclose(jacobian, expected_jacobian)


def _stub_solved(compiled):
    n_ctrl = N_VAR - compiled.n_state
    return FirstOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            f=np.zeros((n_ctrl, compiled.n_state), dtype=np.float64),
            order=1,
            A=np.eye(N_VAR, dtype=np.float64),
            B=np.vstack(
                [
                    np.eye(compiled.n_exog, dtype=np.float64),
                    np.zeros(
                        (N_VAR - compiled.n_exog, compiled.n_exog), dtype=np.float64
                    ),
                ]
            ),
        ),
    )


def test_kalman_order_sensitive_matrices_use_canonical_compiled_layout(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)
    ki = KalmanInterface.__new__(KalmanInterface)
    ki.model = _stub_solved(compiled)

    np.testing.assert_allclose(
        ki._build_Q(),
        np.diag([0.50**2, 0.25**2]).astype(np.float64),
    )
    # P0 is widened over the generated variables then permuted into canonical
    # order. A lag aux takes its origin's variance (u:1, v:2, r:3) and a shock
    # state takes the neutral 1.0.
    np.testing.assert_allclose(
        compiled.kalman.P0,
        np.diag([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0]),
    )


def test_simulation_shock_unpack_accepts_shocks_by_name(tmp_path):
    compiled = _compile_misordered_test_model(tmp_path)
    solved = _stub_solved(compiled)

    unpacked = shock_unpack(
        solved.compiled,
        {
            "e_u": np.array([1.0, 2.0], dtype=np.float64),
            "e_v": np.array([3.0, 4.0], dtype=np.float64),
        },
    )

    # Indices are columns of the (T, n_exog) shock matrix, not state positions.
    assert [idx for idx, _ in unpacked] == [0, 1]
    np.testing.assert_allclose(unpacked[0][1], np.array([1.0, 2.0]))
    np.testing.assert_allclose(unpacked[1][1], np.array([3.0, 4.0]))


def test_explicit_order_permutes_controls_and_leaves_the_states_alone(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile(
        variable_order=["r", "v", "u", "r_star", "x", "Pi"],
    )

    assert compiled.layout.control_names == ("r", "v", "u", "r_star", "x", "Pi")
    assert compiled.var_names[: compiled.n_state] == [
        "e_u_st",
        "e_v_st",
        "u_lag1",
        "v_lag1",
        "r_lag1",
    ]


def test_compile_rejects_an_order_naming_a_generated_state(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="must not appear"):
        solver.compile(
            variable_order=["Pi", "x", "r_star", "u", "v", "r", "u_lag1"],
        )
