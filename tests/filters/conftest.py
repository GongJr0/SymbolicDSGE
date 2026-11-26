from types import SimpleNamespace

import numpy as np
import pytest
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter

from SymbolicDSGE import ExtendedKalman, LinearKalman, ShockPath
from SymbolicDSGE.core.resolvers import measurement_covariance, shock_covariance


def _smoother_run(kalman, sol, kf, P0=None):
    return kalman.smooth(
        sol,
        kf.obs,
        kf.obs_columns,
        P0=P0,
    )


@pytest.fixture(scope="module")
def sdsge_kf_lyap(solved_post82, post82_kf_goldens):
    """SDSGE kalman filter with the default Lyapunov solver for P0"""
    return _smoother_run(
        LinearKalman(symmetrize=True), solved_post82, post82_kf_goldens
    )


@pytest.fixture(scope="module")
def sdsge_kf_eye(solved_post82, post82_kf_goldens):
    """SDSGE kalman filter with P0 = I"""
    n_var = solved_post82.compiled.n_var
    return _smoother_run(
        LinearKalman(symmetrize=True),
        solved_post82,
        post82_kf_goldens,
        P0=np.eye(n_var, n_var, dtype=np.float64),
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(LinearKalman(), id="unsymmetrized"),
        pytest.param(LinearKalman(joseph_cov=True), id="joseph"),
        pytest.param(
            LinearKalman(symmetrize=True, joseph_cov=True), id="symmetrized-joseph"
        ),
    ],
)
def sdsge_kf_flagged(request, solved_post82, post82_kf_goldens):
    """SDSGE kalman smoother under each remaining flag combination"""
    return _smoother_run(request.param, solved_post82, post82_kf_goldens)


def _simulate(solved, T, seed):
    Q = shock_covariance(solved.config)
    sn = solved.compiled.shock_names
    eps = np.random.default_rng(seed).multivariate_normal(
        np.zeros(len(sn)), Q, size=T, method="cholesky"
    )
    return solved.sim(T=T, shocks=[ShockPath(eps, *sn)], observables=True)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(("solved_post82", 1.0), id="post82"),
        # A unit prior on states this small leaves the first smoothed covariance
        # conditioned past the tolerance; a prior on the model's scale does not.
        pytest.param(("solved_multishock", 1e-4), id="multishock"),
    ],
)
def ekf_case(request):
    """A model, a panel simulated from it, and the prior both filters start from.

    POST82 measures affinely and the multishock RBC measures ``g`` through
    ``exp``, away from a nonzero steady state.
    """
    name, prior_scale = request.param
    solved = request.getfixturevalue(name)
    sim = _simulate(solved, T=40, seed=7)
    return SimpleNamespace(
        solved=solved,
        y=sim.y,
        observables=sim.observable_names,
        P0=prior_scale * np.eye(solved.compiled.n_var),
    )


@pytest.fixture(scope="session")
def sdsge_ekf(ekf_case):
    """SDSGE extended kalman smoother on the case's model and data"""
    return ExtendedKalman(symmetrize=True).smooth(
        ekf_case.solved, ekf_case.y, ekf_case.observables, P0=ekf_case.P0
    )


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(ExtendedKalman(), id="unsymmetrized"),
        pytest.param(ExtendedKalman(joseph_cov=True), id="joseph"),
        pytest.param(
            ExtendedKalman(symmetrize=True, joseph_cov=True), id="symmetrized-joseph"
        ),
    ],
)
def sdsge_ekf_flagged(request, ekf_case):
    """SDSGE extended kalman smoother under each remaining flag combination"""
    return request.param.smooth(
        ekf_case.solved, ekf_case.y, ekf_case.observables, P0=ekf_case.P0
    )


@pytest.fixture(scope="session")
def filterpy_ekf(ekf_case):
    """filterpy's EKF forward pass and RTS backward pass, reported in levels.

    ``predict_update`` predicts first, so the loop calls ``update`` then
    ``predict`` to open each period on the update as ours does. The predicted
    covariance has rank ``n_state``, which ``inv`` cannot take.
    """
    solved, y, obs = ekf_case.solved, ekf_case.y, list(ekf_case.observables)
    compiled, ss = solved.compiled, solved.policy.steady_state
    params = np.fromiter(
        solved.config.calibration.parameters.values(),
        dtype=np.float64,
        count=compiled.n_par,
    )
    A, B = solved.policy.A, solved.policy.B
    BQB = B @ shock_covariance(solved.config) @ B.T
    n, m = A.shape[0], y.shape[1]

    def h(x):
        return compiled.build_affine_measurement_matrices(params, obs, ss + x)[1]

    def H(x):
        return compiled.build_affine_measurement_matrices(params, obs, ss + x)[0]

    ekf = ExtendedKalmanFilter(dim_x=n, dim_z=m)
    ekf.F, ekf.Q = A, BQB
    ekf.R = measurement_covariance(solved.config, observables=obs)
    ekf.x, ekf.P = np.zeros(n), ekf_case.P0.copy()

    hist = {name: [] for name in ("x_pred", "P_pred", "x_filt", "P_filt", "innov", "S")}
    loglik = 0.0
    for z in y:
        hist["x_pred"].append(ekf.x.copy())
        hist["P_pred"].append(ekf.P.copy())
        ekf.update(z, H, h)
        hist["x_filt"].append(ekf.x.copy())
        hist["P_filt"].append(ekf.P.copy())
        hist["innov"].append(ekf.y.copy())
        hist["S"].append(ekf.S.copy())
        loglik += ekf.log_likelihood
        ekf.predict()
    hist = {name: np.array(series) for name, series in hist.items()}

    x_smooth, P_smooth, _, _ = KalmanFilter(dim_x=n, dim_z=m).rts_smoother(
        hist["x_filt"],
        hist["P_filt"],
        Fs=[A] * len(y),
        Qs=[BQB] * len(y),
        inv=np.linalg.pinv,
    )
    return SimpleNamespace(
        x_pred=hist["x_pred"] + ss,
        x_filt=hist["x_filt"] + ss,
        x_smooth=x_smooth + ss,
        P_pred=hist["P_pred"],
        P_filt=hist["P_filt"],
        P_smooth=P_smooth,
        innov=hist["innov"],
        S=hist["S"],
        y_pred=y - hist["innov"],
        loglik=loglik,
    )
