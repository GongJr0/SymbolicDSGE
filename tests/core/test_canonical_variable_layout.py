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

# States then controls, each in declaration order. test.yaml lags u, v and r,
# so those are the states; its two shocks are innovations, not variables.
EXPECTED_CANONICAL_ORDER = ["u", "v", "r", "Pi", "x", "r_star"]
EXPECTED_IDX = {name: i for i, name in enumerate(EXPECTED_CANONICAL_ORDER)}
N_VAR = len(EXPECTED_CANONICAL_ORDER)


def _write_misordered_test_model(tmp_path):
    with open("MODELS/test.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Deliberately put controls and the unshocked variable first. The canonical
    # layout should still lead with the states.
    data["variables"] = ["Pi", "x", "r_star", "u", "v", "r"]
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
    assert compiled.layout.state_names == ("u", "v", "r")
    assert compiled.layout.control_names == ("Pi", "x", "r_star")
    assert compiled.n_exog == 2
    assert compiled.n_state == 3


def test_a_single_depth_model_generates_no_variables(tmp_path):
    # Nothing here is displaced past one date, so the compiled set is the
    # declared set and the whole generated block is empty.
    compiled = _compile_misordered_test_model(tmp_path)

    assert compiled.layout.generated_names == ()
    assert compiled.layout.aux_origin == {}
    assert set(compiled.var_names) == set(compiled.layout.declared_names)


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
            steady_state=np.zeros(N_VAR, dtype=np.float64),
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


def test_explicit_order_permutes_within_each_block(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile(
        variable_order=["r", "v", "u", "r_star", "x", "Pi"],
    )

    assert compiled.layout.state_names == ("r", "v", "u")
    assert compiled.layout.control_names == ("r_star", "x", "Pi")
    assert compiled.var_names == ["r", "v", "u", "r_star", "x", "Pi"]


def test_explicit_order_places_a_minted_lag_after_the_named_states(tmp_path):
    # u(t-2) mints u_lag1, which is a state the model never declared. The order
    # names what the model declares, and the minted lag closes the state block.
    with open("MODELS/test.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-2) + e_u"
    path = tmp_path / "deep_test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile(
        variable_order=["r", "v", "u", "r_star", "x", "Pi"],
    )

    assert compiled.layout.state_names == ("r", "v", "u", "u_lag1")
    assert compiled.layout.control_names == ("r_star", "x", "Pi")
    assert compiled.layout.generated_names == ("u_lag1",)
    assert compiled.idx["u_lag1"] == 3

    # declared_names is the user's own list and nothing else. A dense x0 reaches
    # the aux too, so it spans the concatenation, minted last.
    layout = compiled.layout
    assert set(layout.declared_names) == {"Pi", "x", "r_star", "u", "v", "r"}
    assert {*layout.declared_names, *layout.generated_names} == set(compiled.var_names)


def test_explicit_order_rejects_a_control_in_the_state_block(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="must lead with the model's states"):
        solver.compile(variable_order=["u", "v", "Pi", "r", "x", "r_star"])


def test_compile_rejects_an_order_naming_a_generated_state(tmp_path):
    path = _write_misordered_test_model(tmp_path)
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="must not appear"):
        solver.compile(
            variable_order=["Pi", "x", "r_star", "u", "v", "r", "u_lag1"],
        )
