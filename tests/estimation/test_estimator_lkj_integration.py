# type: ignore
import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator


@pytest.fixture(scope="module")
def dense_lkj_bundle(dense_lkj_test_model_path):
    model, kalman = ModelParser(dense_lkj_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

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


def test_packed_logprior_matches_python_path_with_notebook_like_estimator_golden(
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
    theta0 = est.theta0()
    theta = theta0 + np.random.default_rng(20260513).normal(
        loc=0.0,
        scale=0.15,
        size=theta0.shape[0],
    )

    expected_logprior = -3.677756133346315
    expected_loglik = -89.32084071241567
    expected_logpost = -92.99859684576199

    assert est._packed_logprior is not None
    assert float(est._logprior_python(theta)) == pytest.approx(
        expected_logprior, rel=1e-13, abs=1e-13
    )
    assert float(est._packed_logprior.logpdf(theta)) == pytest.approx(
        expected_logprior, rel=1e-13, abs=1e-13
    )
    assert float(est.logprior(theta)) == pytest.approx(
        expected_logprior, rel=1e-13, abs=1e-13
    )
    assert float(est.loglik(theta)) == pytest.approx(
        expected_loglik, rel=1e-13, abs=1e-13
    )
    assert float(est.logpost(theta)) == pytest.approx(
        expected_logpost, rel=1e-13, abs=1e-13
    )


def test_mle_interacting_scalar_corrs_without_prior_hit_spd_gate(dense_lkj_bundle):
    # Two of the three dense Q correlations, estimated prior-free as standalone
    # scalars, share shock e_g. That is not a full dense set (no promotion) and a
    # per-parameter tanh cannot guarantee joint SPD, so it fails fast toward Q_corr.
    with pytest.raises(ValueError, match="joint positive-definiteness"):
        Estimator(
            solver=dense_lkj_bundle["solver"],
            compiled=dense_lkj_bundle["compiled"],
            y=dense_lkj_bundle["y"],
            steady_state=dense_lkj_bundle["steady"],
            estimated_params=["rho_gz", "rho_gr_shock"],
        )


def test_mle_full_dense_q_corr_set_promotes_and_estimates(dense_lkj_bundle):
    # All three dense Q correlations estimated prior-free fold into a Q_corr CPC
    # block (SPD by construction) instead of tripping the gate.
    est = Estimator(
        solver=dense_lkj_bundle["solver"],
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        steady_state=dense_lkj_bundle["steady"],
        estimated_params=["rho_gz", "rho_gr_shock", "rho_zr_shock"],
    )
    assert "Q_corr" in est._matrix_blocks
    assert np.isfinite(est.loglik(est.theta0()))


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
    # R now travels the likelihood (build_R rebuilds it from params every eval),
    # so perturbing the R correlations changes the loglik.
    assert ll1 != pytest.approx(ll0)

    out = est.mcmc(
        n_draws=8,
        burn_in=6,
        thin=1,
        random_state=123,
        adapt=False,
        proposal_scale=0.08,
    )

    assert out.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"]
    assert out.samples.shape == (8, 3)
    _assert_valid_corr_draws(out.samples)


def test_to_spec_round_trips_matrix_prior(dense_lkj_bundle):
    from SymbolicDSGE.estimation.spec import EstimationSpec

    est = Estimator(
        solver=dense_lkj_bundle["solver"],
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        steady_state=dense_lkj_bundle["steady"],
        estimated_params=["R_corr"],
        priors={"R_corr": LKJChol(eta=2.0, K=3, random_state=None)},
    )

    # The block prior is emitted under its reserved target, not as scalar params.
    spec = est.to_spec(method="mcmc")
    assert spec.parameters == []
    assert set(spec.matrix_priors) == {"R_corr"}
    mp = spec.matrix_priors["R_corr"]
    assert mp.distribution == "lkj_chol"
    assert mp.parameters == {"eta": 2.0, "K": 3}
    assert mp.transform == "cholesky_corr"

    # Lowers back to runnable inputs (theta0 deferred to the estimator).
    inputs = spec.to_estimator_inputs()
    assert inputs.estimated_params == ["R_corr"]
    assert inputs.theta0 is None
    assert inputs.priors is not None and "R_corr" in inputs.priors

    assert EstimationSpec.from_json(spec.to_json()).to_dict() == spec.to_dict()


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
