import numpy as np
import pytest

from SymbolicDSGE import LinearKalman

TOL = 1e-12


def _smoother_run(sol, kf, P0=None):
    return LinearKalman(symmetrize=True).smooth(
        sol,
        kf.obs,
        kf.obs_columns,
        P0=P0,
    )


@pytest.fixture(scope="module")
def sdsge_kf_lyap(solved_post82, post82_kf_goldens):
    """SDSGE kalman filter with the default Lyapunov solver for P0"""
    return _smoother_run(solved_post82, post82_kf_goldens)


@pytest.fixture(scope="module")
def sdsge_kf_eye(solved_post82, post82_kf_goldens):
    """SDSGE kalman filter with P0 = I"""
    n_var = solved_post82.compiled.n_var
    return _smoother_run(
        solved_post82,
        post82_kf_goldens,
        P0=np.eye(n_var, n_var, dtype=np.float64),
    )


def test_kf_loglik_parity(sdsge_kf_lyap, sdsge_kf_eye, post82_kf_goldens):
    """Check that the loglikelihoods match the Dynare golden"""
    dyn = post82_kf_goldens
    assert np.abs(sdsge_kf_lyap.filter_result.loglik - dyn.loglik_uncond) < TOL
    assert np.abs(sdsge_kf_eye.filter_result.loglik - dyn.loglik_fixed) < TOL


def test_kf_predicted_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the predicted states match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert np.abs(sdsge.x_pred[1:] - dyn.filtered[:-1]).max() < TOL


def test_kf_filtered_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the filtered states match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert np.abs(sdsge.x_filt - dyn.updated).max() < TOL


def test_kf_smoothed_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed states match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert np.abs(sdsge.x_smooth - dyn.smoothed).max() < TOL


def test_kf_predicted_cov_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the predicted covariances match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert np.abs(sdsge.P_pred - dyn.variance[:-1]).max() < TOL


def test_kf_smoothed_cov_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed covariances match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert np.abs(sdsge.P_smooth - dyn.state_uncertainty).max() < TOL


def test_kf_smoothed_eps_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed shocks match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert np.abs(sdsge.eps_smooth - dyn.smoothed_shocks).max() < TOL
