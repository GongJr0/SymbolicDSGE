# type: ignore
from __future__ import annotations

import numpy as np
import pytest
import yaml

from SymbolicDSGE.core import compile_model
from SymbolicDSGE.core import parse_config
from SymbolicDSGE.core.resolvers import shock_covariance
from SymbolicDSGE._ckernels.core import jacobian_eval, measurement_eval

# States then controls, each in declaration order. test.yaml lags u, v and r,
# so those are the states; its two shocks are innovations, not variables.
EXPECTED_CANONICAL_ORDER = ["u", "v", "r", "Pi", "x", "r_star"]
EXPECTED_IDX = {name: i for i, name in enumerate(EXPECTED_CANONICAL_ORDER)}
N_VAR = len(EXPECTED_CANONICAL_ORDER)


def _write_misordered_test_model(tmp_path, test_model_path):
    with open(test_model_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Deliberately put controls and the unshocked variable first. The canonical
    # layout should still lead with the states.
    data["variables"] = ["Pi", "x", "r_star", "u", "v", "r"]
    path = tmp_path / "misordered_test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _compile_misordered_test_model(tmp_path, test_model_path):
    path = _write_misordered_test_model(tmp_path, test_model_path)
    model = parse_config(path)
    return model, compile_model(model)


def test_compile_default_layout_canonicalizes_misordered_yaml_variables(
    tmp_path, test_model_path
):
    _, compiled = _compile_misordered_test_model(tmp_path, test_model_path)

    assert compiled.var_names == EXPECTED_CANONICAL_ORDER
    assert compiled.idx == EXPECTED_IDX
    assert compiled.layout.declared_names == ("Pi", "x", "r_star", "u", "v", "r")
    assert compiled.layout.canonical_names == tuple(EXPECTED_CANONICAL_ORDER)
    assert compiled.layout.state_names == ("u", "v", "r")
    assert compiled.layout.control_names == ("Pi", "x", "r_star")
    assert compiled.n_exog == 2
    assert compiled.n_state == 3


def test_a_single_depth_model_generates_no_variables(tmp_path, test_model_path):
    # Nothing here is displaced past one date, so the compiled set is the
    # declared set and the whole generated block is empty.
    _, compiled = _compile_misordered_test_model(tmp_path, test_model_path)

    assert compiled.layout.generated_names == ()
    assert compiled.layout.aux_origin == {}
    assert set(compiled.var_names) == set(compiled.layout.declared_names)


def test_shock_columns_are_named_by_shock_in_declaration_order(
    tmp_path, test_model_path
):
    _, compiled = _compile_misordered_test_model(tmp_path, test_model_path)

    assert compiled.shock_names == ("e_u", "e_v")
    assert compiled.shock_idx == {"e_u": 0, "e_v": 1}


def test_measurement_dispatchers_accept_canonical_state_order_after_yaml_reorder(
    tmp_path, test_model_path
):
    config, compiled = _compile_misordered_test_model(tmp_path, test_model_path)
    params = np.array(
        [config.calibration.parameters[p] for p in compiled.calib_params],
        dtype=np.float64,
    )
    state = np.zeros(N_VAR, dtype=np.float64)
    state[EXPECTED_IDX["Pi"]] = 40.0
    state[EXPECTED_IDX["r"]] = 30.0

    obs = ["Infl", "Rate"]
    measurement = measurement_eval(
        compiled.construct_measurement_cfunc(obs).address, state, params, len(obs)
    )
    jacobian = jacobian_eval(
        compiled.construct_measurement_jacobian_cfunc(obs).address,
        np.zeros_like(state),
        params,
        len(obs),
        compiled.n_var,
    )

    expected_jacobian = np.zeros((2, N_VAR), dtype=np.float64)
    expected_jacobian[0, EXPECTED_IDX["Pi"]] = 1.0
    expected_jacobian[1, EXPECTED_IDX["r"]] = 1.0

    np.testing.assert_allclose(measurement, np.array([43.25, 30.0]))
    np.testing.assert_allclose(jacobian, expected_jacobian)


def test_kalman_order_sensitive_matrices_use_canonical_compiled_layout(
    tmp_path, test_model_path
):
    config, compiled = _compile_misordered_test_model(tmp_path, test_model_path)

    np.testing.assert_allclose(
        shock_covariance(config),
        np.diag([0.50**2, 0.25**2]).astype(np.float64),
    )
