# type: ignore
import warnings

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.estimation import Estimator


@pytest.fixture(scope="module")
def dense_lkj_bundle(dense_lkj_test_model_path):
    model, kalman = ModelParser(dense_lkj_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=3)

    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, steady_state=steady)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig_g = float(params[std_map[Symbol("e_g")]])
    sig_z = float(params[std_map[Symbol("e_z")]])
    sig_r = float(params[std_map[Symbol("e_r")]])

    T = 16
    rng = np.random.default_rng(20260317)
    sim = solved.sim(
        T=T,
        shocks={
            "g": rng.normal(0.0, sig_g, size=T),
            "z": rng.normal(0.0, sig_z, size=T),
            "r": rng.normal(0.0, sig_r, size=T),
        },
        x0=np.zeros((len(compiled.var_names),), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "OutGap": sim["OutGap"][1:],
            "Infl": sim["Infl"][1:],
            "Rate": sim["Rate"][1:],
        }
    )
    return {
        "solver": solver,
        "compiled": compiled,
        "steady": steady,
        "y": y,
    }


def _assert_valid_corr_draws(samples: np.ndarray) -> None:
    for draw in samples:
        corr = np.eye(3, dtype=np.float64)
        corr[1, 0] = corr[0, 1] = draw[0]
        corr[2, 0] = corr[0, 2] = draw[1]
        corr[2, 1] = corr[1, 2] = draw[2]
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals > 1e-10)


def _notebook_like_prior_spec() -> dict[str, object]:
    return {
        "beta": make_prior(
            "beta",
            parameters={"a": 200 * 0.99, "b": 200 * 0.001},
            transform="logit",
        ),
        "rho_r": make_prior(
            "beta",
            parameters={"a": 200 * 0.84, "b": 200 * 0.16},
            transform="logit",
        ),
        "rho_g": make_prior(
            "beta",
            parameters={"a": 200 * 0.83, "b": 200 * 0.17},
            transform="logit",
        ),
        "rho_z": make_prior(
            "beta",
            parameters={"a": 200 * 0.85, "b": 200 * 0.15},
            transform="logit",
        ),
        "psi_pi": make_prior(
            "gamma",
            parameters={"mean": 2.19, "std": 0.5},
            transform="log",
        ),
        "psi_x": make_prior(
            "gamma",
            parameters={"mean": 0.30, "std": 0.1},
            transform="log",
        ),
        "kappa": make_prior(
            "gamma",
            parameters={"mean": 0.58, "std": 0.1},
            transform="log",
        ),
        "tau_inv": make_prior(
            "gamma",
            parameters={"mean": 1.86, "std": 0.5},
            transform="log",
        ),
        "rho_gz": make_prior(
            "trunc_normal",
            parameters={"mean": 0.0, "std": 0.20, "low": -1.0, "high": 1.0},
            transform="affine_logit",
            transform_kwargs={"low": -1.0, "high": 1.0},
        ),
        "sig_r": make_prior(
            "gamma",
            parameters={"mean": 0.18, "std": 0.1},
            transform="log",
        ),
        "sig_g": make_prior(
            "gamma",
            parameters={"mean": 0.18, "std": 0.1},
            transform="log",
        ),
        "sig_z": make_prior(
            "gamma",
            parameters={"mean": 0.64, "std": 0.1},
            transform="log",
        ),
        "R_corr": make_prior(
            "lkj_chol",
            parameters={"eta": 1.0, "K": 3},
            transform="cholesky_corr",
        ),
    }


def _run_dynamic_r_adaptive_chain(est: Estimator, *, steps: int, seed: int):
    current = est.theta0()
    d = current.shape[0]
    cov = (0.1**2) * np.eye(d, dtype=np.float64)
    scale = (2.38**2) / d
    history = np.empty((steps, d), dtype=np.float64)
    rng = np.random.default_rng(seed)
    dynamic_obs = est._effective_observables()

    def _safe_logpost_chain(theta: np.ndarray) -> np.float64:
        try:
            params = est.theta_to_params(theta)
            R_iter = est_backend.build_R_from_config_params(
                compiled=est.compiled,
                kalman=est.compiled.kalman,
                observables=dynamic_obs,
                params=params,
            )
            lp, n_signals = est._eval_with_warning_capture(
                lambda th: est._logpost_with_overrides(
                    th,
                    params_override=params,
                    R_override=R_iter,
                ),
                theta,
            )
            est._warning_signal_count += n_signals
            if n_signals > 0 or not np.isfinite(lp):
                return np.float64(-np.inf)
            return np.float64(lp)
        except BaseException:
            return np.float64(-np.inf)

    cur_lp = _safe_logpost_chain(current)
    for t in range(steps):
        prop = rng.multivariate_normal(current, cov, check_valid="warn")
        prop_lp = _safe_logpost_chain(prop)
        if np.isfinite(prop_lp) and np.log(rng.random()) < (prop_lp - cur_lp):
            current = prop
            cur_lp = prop_lp
        history[t] = current
        if t >= 100 and (t + 1) % 25 == 0:
            emp = np.cov(history[: t + 1].T, ddof=1)
            cov = scale * (
                np.asarray(emp, dtype=np.float64) + 1e-8 * np.eye(d, dtype=np.float64)
            )
    return current, history, cov


def test_matrix_prior_on_R_runs_full_mcmc_with_real_likelihood(dense_lkj_bundle):
    prior_spec = {"R_corr": LKJChol(eta=2.0, K=3, random_state=None)}
    est = Estimator(
        solver=dense_lkj_bundle["solver"],
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        steady_state=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )

    theta0 = est.theta0()
    theta_alt = est.params_to_theta(
        {
            "meas_rho_gi": -0.08,
            "meas_rho_gr": 0.10,
            "meas_rho_ir": 0.12,
        }
    )
    ll0 = est.loglik(theta0)
    ll1 = est.loglik(theta_alt)

    assert np.isfinite(ll0)
    assert np.isfinite(ll1)
    assert ll1 == pytest.approx(ll0)

    out = est.mcmc(
        n_draws=8,
        burn_in=6,
        thin=1,
        random_state=123,
        adapt=False,
        proposal_scale=0.08,
        update_R_in_iterations=True,
    )

    assert out.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"]
    assert out.samples.shape == (8, 3)
    _assert_valid_corr_draws(out.samples)


def test_matrix_prior_on_Q_runs_full_mcmc_with_real_likelihood(dense_lkj_bundle):
    prior_spec = {"Q_corr": LKJChol(eta=2.0, K=3, random_state=None)}
    est = Estimator(
        solver=dense_lkj_bundle["solver"],
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        steady_state=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )

    theta0 = est.theta0()
    theta_alt = est.params_to_theta(
        {
            "rho_gz": 0.08,
            "rho_gr_shock": -0.12,
            "rho_zr_shock": 0.10,
        }
    )
    ll0 = est.loglik(theta0)
    ll1 = est.loglik(theta_alt)

    assert np.isfinite(ll0)
    assert np.isfinite(ll1)
    assert ll1 != pytest.approx(ll0)

    out = est.mcmc(
        n_draws=8,
        burn_in=6,
        thin=1,
        random_state=321,
        adapt=False,
        proposal_scale=0.08,
    )

    assert out.param_names == ["rho_gz", "rho_gr_shock", "rho_zr_shock"]
    assert out.samples.shape == (8, 3)
    _assert_valid_corr_draws(out.samples)


def test_adaptive_r_block_stays_well_conditioned_under_dynamic_updates(
    dense_lkj_bundle,
):
    prior_spec = _notebook_like_prior_spec()
    est = Estimator(
        solver=dense_lkj_bundle["solver"],
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        steady_state=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        current, history, cov = _run_dynamic_r_adaptive_chain(
            est,
            steps=8000,
            seed=123,
        )

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings

    r_idx = est._matrix_blocks["R_corr"].theta_indices
    max_abs_r = np.max(np.abs(history[:, r_idx]), axis=0)
    min_eig = np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T)))

    assert np.all(max_abs_r < 10.0)
    assert min_eig > 1e-8
    _assert_valid_corr_draws(
        np.asarray(
            [
                [
                    est.theta_to_params(current)["meas_rho_gi"],
                    est.theta_to_params(current)["meas_rho_gr"],
                    est.theta_to_params(current)["meas_rho_ir"],
                ]
            ],
            dtype=np.float64,
        )
    )
