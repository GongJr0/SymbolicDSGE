# type: ignore
import json

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.prior_program import TransformCode


@pytest.fixture(scope="module")
def dense_lkj_bundle(dense_lkj_test_model_path):
    model, kalman = ModelParser(dense_lkj_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    steady = np.zeros((len(compiled.layout.declared_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

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
            "e_g": rng.normal(0.0, sig_g, size=T),
            "e_z": rng.normal(0.0, sig_z, size=T),
            "e_r": rng.normal(0.0, sig_r, size=T),
        },
        x0=np.zeros((compiled.n_var,), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "OutGap": sim.observables["OutGap"],
            "Infl": sim.observables["Infl"],
            "Rate": sim.observables["Rate"],
        }
    )
    y += rng.multivariate_normal(
        mean=np.zeros((3,), dtype=np.float64),
        cov=kalman.R,
        size=T,
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
        # CPC decoding is SPD by construction. Allow a roundoff-scale negative
        # eigenvalue when reconstructing the dense correlation matrix here.
        assert np.all(eigvals >= -1e-10)


def _full_size_prior_spec() -> dict[str, object]:
    """Every estimable role this config can carry at once: scalars, the stds, and
    an LKJ block. A standalone scalar correlation is not one of them, since every
    Q correlation here is named and nonzero for the block's sake."""
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


def test_packed_logprior_matches_python_path_with_full_size_estimator_golden(
    dense_lkj_bundle,
):
    prior_spec = _full_size_prior_spec()
    est = Estimator(
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        ss_seed=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )
    # Off the calibration on every role at once, and inside every support: the
    # stds stay positive and the three block members stay a PD correlation.
    at = {
        "beta": 0.965,
        "rho_r": 0.80,
        "rho_g": 0.86,
        "rho_z": 0.81,
        "psi_pi": 2.05,
        "psi_x": 0.34,
        "kappa": 0.62,
        "tau_inv": 1.74,
        "sig_r": 0.21,
        "sig_g": 0.16,
        "sig_z": 0.58,
        "meas_rho_ir": 0.08,
        "meas_rho_gi": -0.06,
        "meas_rho_gr": 0.02,
    }

    # expected_logprior does not see the filter, so it does not move when the
    # filter does. All three are regression anchors of our own output, not
    # oracles; their correctness rests upstream on the POST82 Dynare parity, and
    # they are recomputed whenever the filter contract changes.
    expected_logprior = -5.607932470661046
    expected_loglik = -80.05811738272585
    expected_logpost = -85.6660498533869

    assert float(est.logprior(at, True)) == pytest.approx(
        expected_logprior, rel=1e-13, abs=1e-13
    )
    assert float(est.loglik(at)) == pytest.approx(expected_loglik, rel=1e-13, abs=1e-13)
    assert float(est.logpost(at, True)) == pytest.approx(
        expected_logpost, rel=1e-13, abs=1e-13
    )


def test_mle_interacting_scalar_corrs_without_prior_hit_spd_gate(dense_lkj_bundle):
    # Two of the three dense Q correlations, estimated prior-free as standalone
    # scalars, share shock e_g. That is not a full dense set (no promotion) and a
    # per-parameter tanh cannot guarantee joint SPD, so it fails fast toward Q_corr.
    with pytest.raises(ValueError, match="joint positive-definiteness"):
        Estimator(
            compiled=dense_lkj_bundle["compiled"],
            y=dense_lkj_bundle["y"],
            ss_seed=dense_lkj_bundle["steady"],
            estimated_params=["rho_gz", "rho_gr_shock"],
        )


def test_mle_full_dense_q_corr_set_promotes_and_estimates(dense_lkj_bundle):
    # All three dense Q correlations estimated prior-free fold into a Q_corr CPC
    # block (SPD by construction) instead of tripping the gate.
    est = Estimator(
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        ss_seed=dense_lkj_bundle["steady"],
        estimated_params=["rho_gz", "rho_gr_shock", "rho_zr_shock"],
    )
    assert "Q_corr" in est._matrix_blocks
    assert np.isfinite(est.loglik())

    # The block owns the whole run's reparameterization, so every slot carries
    # the block code and the role box has nothing left to restrict.
    ctx, _, lo, hi = est._build_call_context(priored=False)
    assert all(
        TransformCode(int(code)) is TransformCode.CHOLESKY_CORR
        for code in ctx.prior.transform_codes
    )
    assert np.all(lo == -np.inf) and np.all(hi == np.inf)


def test_matrix_prior_on_R_runs_full_mcmc_with_real_likelihood(dense_lkj_bundle):
    prior_spec = {
        "R_corr": Prior(
            dist=LKJChol(eta=2.0, K=3, random_state=None),
            transform=CholeskyCorrTransform(K=3),
        )
    }
    est = Estimator(
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        ss_seed=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )

    ll0 = est.loglik()
    ll1 = est.loglik(
        {
            "meas_rho_gi": -0.08,
            "meas_rho_gr": 0.10,
            "meas_rho_ir": 0.12,
        }
    )

    assert np.isfinite(ll0)
    assert np.isfinite(ll1)
    # R now travels the likelihood (_build_R rebuilds it from params every eval),
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
    est = Estimator(
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        ss_seed=dense_lkj_bundle["steady"],
        estimated_params=["R_corr"],
        priors={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=3, random_state=None),
                transform=CholeskyCorrTransform(K=3),
            )
        },
    )

    # The block prior is emitted under its reserved target, alongside the members
    # the block expands to.
    spec = est.to_spec()
    mp = spec.params["priors"]["R_corr"]
    assert mp["distribution"] == "lkj_chol"
    assert mp["parameters"] == {"eta": 2.0, "K": 3}
    assert mp["transform"] == "cholesky_corr"

    # The text half is JSON as it stands, with no emitter in between.
    assert json.loads(json.dumps(spec.params)) == spec.params

    # And it rebuilds into an equivalent estimator.
    rebuilt = Estimator.from_spec(spec, dense_lkj_bundle["compiled"])
    assert rebuilt.param_names == est.param_names
    assert set(rebuilt._matrix_blocks) == {"R_corr"}
    assert float(rebuilt.loglik()) == float(est.loglik())


def test_matrix_prior_on_Q_runs_full_mcmc_with_real_likelihood(dense_lkj_bundle):
    prior_spec = {
        "Q_corr": Prior(
            dist=LKJChol(eta=2.0, K=3, random_state=None),
            transform=CholeskyCorrTransform(K=3),
        )
    }
    est = Estimator(
        compiled=dense_lkj_bundle["compiled"],
        y=dense_lkj_bundle["y"],
        ss_seed=dense_lkj_bundle["steady"],
        estimated_params=list(prior_spec.keys()),
        priors=prior_spec,
    )

    ll0 = est.loglik()
    ll1 = est.loglik(
        {
            "rho_gz": 0.08,
            "rho_gr_shock": -0.12,
            "rho_zr_shock": 0.10,
        }
    )

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
