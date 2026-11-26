import numpy as np

# Relative to the reference array's own scale: the multishock covariances sit
# near 1e-4 while its states sit near the capital stock.
TOL = 1e-10


def _close(sdsge, ref):
    return np.abs(sdsge - ref).max() <= TOL * np.abs(ref).max()


def test_ekf_loglik_parity(sdsge_ekf, filterpy_ekf):
    """Check that the loglikelihood matches filterpy"""
    assert _close(sdsge_ekf.filter_result.loglik, filterpy_ekf.loglik)


def test_ekf_predicted_state_parity(sdsge_ekf, filterpy_ekf):
    """Check that the predicted states match filterpy"""
    assert _close(sdsge_ekf.filter_result.x_pred, filterpy_ekf.x_pred)


def test_ekf_filtered_state_parity(sdsge_ekf, filterpy_ekf):
    """Check that the filtered states match filterpy"""
    assert _close(sdsge_ekf.filter_result.x_filt, filterpy_ekf.x_filt)


def test_ekf_smoothed_state_parity(sdsge_ekf, filterpy_ekf):
    """Check that the smoothed states match filterpy's RTS smoother"""
    assert _close(sdsge_ekf.x_smooth, filterpy_ekf.x_smooth)


def test_ekf_predicted_cov_parity(sdsge_ekf, filterpy_ekf):
    """Check that the predicted covariances match filterpy"""
    assert _close(sdsge_ekf.filter_result.P_pred, filterpy_ekf.P_pred)


def test_ekf_filtered_cov_parity(sdsge_ekf, filterpy_ekf):
    """Check that the filtered covariances match filterpy"""
    assert _close(sdsge_ekf.filter_result.P_filt, filterpy_ekf.P_filt)


def test_ekf_smoothed_cov_parity(sdsge_ekf, filterpy_ekf):
    """Check that the smoothed covariances match filterpy's RTS smoother"""
    assert _close(sdsge_ekf.P_smooth, filterpy_ekf.P_smooth)


def test_ekf_innovation_parity(sdsge_ekf, filterpy_ekf):
    """Check that the predicted observables and innovations match filterpy"""
    sdsge = sdsge_ekf.filter_result
    ref = filterpy_ekf
    assert _close(sdsge.y_pred, ref.y_pred)
    assert _close(sdsge.innov, ref.innov)
    assert _close(sdsge.S, ref.S)


def test_ekf_flag_parity(sdsge_ekf_flagged, filterpy_ekf):
    """Check that every flag combination still matches filterpy"""
    sdsge = sdsge_ekf_flagged
    fwd = sdsge.filter_result
    ref = filterpy_ekf
    assert _close(fwd.loglik, ref.loglik)
    assert _close(fwd.x_pred, ref.x_pred)
    assert _close(fwd.x_filt, ref.x_filt)
    assert _close(fwd.P_pred, ref.P_pred)
    assert _close(fwd.P_filt, ref.P_filt)
    assert _close(fwd.innov, ref.innov)
    assert _close(fwd.S, ref.S)
    assert _close(sdsge.x_smooth, ref.x_smooth)
    assert _close(sdsge.P_smooth, ref.P_smooth)
