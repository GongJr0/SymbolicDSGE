# type: ignore
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_solved_model_sim_shapes_and_keys(solved_test):
    T = 12
    out = solved_test.sim(T)

    assert "_X" in out
    assert out["_X"].shape == (T + 1, solved_test.A.shape[0])
    for name in solved_test.compiled.var_names:
        assert name in out
        assert out[name].shape == (T + 1,)


def test_solved_model_sim_rejects_wrong_shock_length(solved_test):
    with pytest.raises(ValueError, match="must have length"):
        solved_test.sim(8, shocks={"u": np.ones(7)})


def test_solved_model_sim_with_observables_includes_measurements(solved_test):
    out = solved_test.sim(10, observables=True)
    for obs in solved_test.compiled.observable_names:
        assert obs in out
        assert out[obs].shape == (11,)


def test_solved_model_irf_validation_errors(solved_test):
    with pytest.raises(ValueError, match="At least one shock"):
        solved_test.irf(shocks=[], T=10)
    with pytest.raises(ValueError, match="not found in exogenous"):
        solved_test.irf(shocks=["Pi"], T=10)


def test_solved_model_irf_runs_for_exogenous_shock(solved_test):
    out = solved_test.irf(shocks=["u"], T=8, observables=True)
    assert out["u"].shape == (9,)
    assert "_X" in out
    assert "Infl" in out and "Rate" in out


def test_solved_model_get_param_and_get_rho_helpers(solved_test):
    assert (
        solved_test._get_param("beta")
        == solved_test.config.calibration.parameters["beta"]
    )
    assert solved_test._get_rho("e_u", "e_u") == 1.0
    assert solved_test._get_rho("e_u", "e_v", default=0.0) == 0.0

    with pytest.raises(KeyError):
        solved_test._get_param("not_a_param")


def test_solved_model_build_measurement_matrices(solved_test):
    spec = {
        "Obs1": {"lin": {"Pi": 2.0, "x": -1.0}, "const": [1.5, "pi_mean"]},
        "Obs2": {"lin": {"r": 1.0}, "const": [0.0]},
    }
    C, d, names = solved_test._build_measurement(spec)

    assert C.shape == (2, solved_test.A.shape[0])
    assert d.shape == (2,)
    assert names == ["Obs1", "Obs2"]

    idx = solved_test.compiled.idx
    assert C[0, idx["Pi"]] == 2.0
    assert C[0, idx["x"]] == -1.0
    assert C[1, idx["r"]] == 1.0


def test_solved_model_build_C_d_from_observables(solved_test):
    C, d = solved_test._build_C_d_from_obs(solved_test.compiled.observable_names)
    m = len(solved_test.compiled.observable_names)
    n = solved_test.A.shape[0]

    assert C.shape == (m, n)
    assert d.shape == (m,)


def test_solved_model_shock_unpack_multivar_key_order_is_canonical(solved_test):
    T = 6

    def mv_shock(cov):
        # deterministic mapping from covariance -> shock matrix
        base = np.array([cov[0, 0], cov[1, 1]], dtype=float)
        return np.tile(base, (T, 1))

    unpack_1 = solved_test._shock_unpack({"u,v": mv_shock})
    unpack_2 = solved_test._shock_unpack({"v,u": mv_shock})

    idx_to_vec_1 = {idx: vec for idx, vec in unpack_1}
    idx_to_vec_2 = {idx: vec for idx, vec in unpack_2}

    assert idx_to_vec_1.keys() == idx_to_vec_2.keys()
    for k in idx_to_vec_1:
        assert np.array_equal(idx_to_vec_1[k], idx_to_vec_2[k])


def test_solved_model_kalman_smoke(solved_post82):
    sim = solved_post82.sim(20, observables=True)
    y = pd.DataFrame({"Infl": sim["Infl"][1:], "Rate": sim["Rate"][1:]})

    out = solved_post82.kalman(y)
    assert out is not None


def test_solved_model_to_dict_contains_main_fields(solved_test):
    d = solved_test.to_dict()
    assert "compiled" in d
    assert "policy" in d
    assert "A" in d
    assert "B" in d
