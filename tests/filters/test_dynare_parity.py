import numpy as np

TOL = 1e-12


def _close(sdsge, ref):
    return np.abs(sdsge - ref).max() < TOL


def test_kf_loglik_parity(sdsge_kf_lyap, sdsge_kf_eye, post82_kf_goldens):
    """Check that the loglikelihoods match the Dynare golden"""
    dyn = post82_kf_goldens
    assert np.abs(sdsge_kf_lyap.filter_result.loglik - dyn.loglik_uncond) < TOL
    assert np.abs(sdsge_kf_eye.filter_result.loglik - dyn.loglik_fixed) < TOL


def test_kf_predicted_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the predicted states match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert _close(sdsge.x_pred[1:], dyn.filtered[:-1])


def test_kf_filtered_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the filtered states match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert _close(sdsge.x_filt, dyn.updated)


def test_kf_smoothed_state_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed states match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert _close(sdsge.x_smooth, dyn.smoothed)


def test_kf_predicted_cov_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the predicted covariances match the Dynare golden"""
    sdsge = sdsge_kf_lyap.filter_result
    dyn = post82_kf_goldens
    assert _close(sdsge.P_pred, dyn.variance[:-1])


def test_kf_smoothed_cov_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed covariances match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert _close(sdsge.P_smooth, dyn.state_uncertainty)


def test_kf_smoothed_eps_parity(sdsge_kf_lyap, post82_kf_goldens):
    """Check that the smoothed shocks match the Dynare golden"""
    sdsge = sdsge_kf_lyap
    dyn = post82_kf_goldens
    assert _close(sdsge.eps_smooth, dyn.smoothed_shocks)


def test_kf_flag_parity(sdsge_kf_flagged, post82_kf_goldens):
    """Check that every flag combination still matches the Dynare golden"""
    sdsge = sdsge_kf_flagged
    fwd = sdsge.filter_result
    dyn = post82_kf_goldens
    assert np.abs(fwd.loglik - dyn.loglik_uncond) < TOL
    assert _close(fwd.x_pred[1:], dyn.filtered[:-1])
    assert _close(fwd.x_filt, dyn.updated)
    assert _close(fwd.P_pred, dyn.variance[:-1])
    assert _close(sdsge.x_smooth, dyn.smoothed)
    assert _close(sdsge.P_smooth, dyn.state_uncertainty)
    assert _close(sdsge.eps_smooth, dyn.smoothed_shocks)
