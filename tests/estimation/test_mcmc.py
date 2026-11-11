# type: ignore
import numpy as np
import pytest

from SymbolicDSGE.estimation import make_prior

_POST82_START = {"psi_pi": 2.0, "rho_r": 0.8}

_MCMC_KW = dict(n_draws=25, burn_in=5, random_state=7)
#: A start the MAP actually walks away from. Most points near the POST82 mode
#: are stationary for L-BFGS-B, which would leave the comparisons below unable
#: to tell a skipped MAP from a MAP that ran and moved nothing.
_OFF_MODE = {"psi_pi": 2.4, "rho_r": 0.85}


def test_mcmc_records_sampler_config(mcmc_estimator):
    out = mcmc_estimator.mcmc(
        n_draws=5, burn_in=2, thin=1, random_state=7, proposal_scale=0.2
    )

    cfg = out.sampler_config
    assert cfg["random_state"] == 7
    assert cfg["proposal_scale"] == 0.2
    assert set(cfg) == {
        "n_draws",
        "burn_in",
        "thin",
        "theta0",
        "adapt",
        "adapt_start",
        "adapt_end",
        "adapt_interval",
        "proposal_scale",
        "adapt_epsilon",
        "compute_map",
        "map_options",
        "proposal_cov",
        "cov_fd_step_scale",
        "cov_fd_absolute_floor",
        "random_state",
    }
    assert cfg["compute_map"] is True
    # the sampler built its own, so there is no user matrix to record
    assert cfg["proposal_cov"] is None
    assert out.to_spec().meta["sampler_config"] == cfg

    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    supplied = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
    )

    cfg = supplied.sampler_config
    assert cfg["compute_map"] is False
    assert np.array_equal(np.asarray(cfg["proposal_cov"], dtype=np.float64), mode.vcov)


def test_mcmc_rejects_invalid_arguments(mcmc_estimator):
    with pytest.raises(ValueError, match="n_draws must be positive"):
        mcmc_estimator.mcmc(n_draws=0)
    with pytest.raises(ValueError, match="burn_in must be non-negative"):
        mcmc_estimator.mcmc(n_draws=1, burn_in=-1)
    with pytest.raises(ValueError, match="thin must be positive"):
        mcmc_estimator.mcmc(n_draws=1, thin=0)
    with pytest.raises(ValueError, match="proposal_scale must be positive"):
        mcmc_estimator.mcmc(n_draws=1, proposal_scale=0.0)
    with pytest.raises(ValueError, match="finite-difference settings"):
        mcmc_estimator.mcmc(n_draws=1, cov_fd_step_scale=0.0)
    with pytest.raises(ValueError, match="finite-difference settings"):
        mcmc_estimator.mcmc(n_draws=1, cov_fd_absolute_floor=0.0)
    with pytest.raises(ValueError, match="adapt_start must be positive"):
        mcmc_estimator.mcmc(n_draws=1, adapt_start=0)
    with pytest.raises(ValueError, match="must not exceed adapt_end"):
        mcmc_estimator.mcmc(n_draws=10, burn_in=4, adapt_start=9)
    with pytest.raises(ValueError, match="adapt_interval must be positive"):
        mcmc_estimator.mcmc(n_draws=1, adapt_interval=0)
    with pytest.raises(ValueError, match="adapt_epsilon must be non-negative"):
        mcmc_estimator.mcmc(n_draws=1, adapt_epsilon=-1e-8)
    with pytest.raises(ValueError, match="unsupported MAP option"):
        mcmc_estimator.mcmc(n_draws=1, map_options={"nope": 1})


def test_mcmc_precomputed_mode_reproduces_the_internal_map_chain(mcmc_estimator):
    """Skipping the MAP changes who finds the mode, not the chain that follows."""
    found = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=True, **_MCMC_KW)
    at_theta0 = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=False, **_MCMC_KW)
    mode = mcmc_estimator.map(theta0=_OFF_MODE)
    assert set(mode.theta) == set(mcmc_estimator.param_names)
    at_mode = mcmc_estimator.mcmc(theta0=mode.theta, compute_map=False, **_MCMC_KW)

    assert not np.array_equal(found.samples, at_theta0.samples)
    assert np.array_equal(found.samples, at_mode.samples)
    assert found.sampler_config["compute_map"] is True
    assert at_theta0.sampler_config["compute_map"] is False


def test_mcmc_supplied_covariance_reproduces_the_internal_hessian_chain(
    mcmc_estimator,
):
    """The documented workflow: pay for the MAP once, hand the sampler its
    covariance, and get the chain the sampler would have built for itself.
    """
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    found = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=True, **_MCMC_KW)
    supplied = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
    )

    assert np.allclose(found.samples, supplied.samples, rtol=1e-12, atol=0.0)
    assert np.allclose(
        found.logpost_trace, supplied.logpost_trace, rtol=1e-12, atol=0.0
    )
    assert found.accept_rate == pytest.approx(supplied.accept_rate)


def test_mcmc_supplied_covariance_seeds_the_adaptation_recursion(mcmc_estimator):
    """The supplied matrix is the Haario seed too, not just the first factor."""
    kw = dict(n_draws=20, burn_in=10, random_state=7, adapt=True, adapt_start=5)
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)

    found = mcmc_estimator.mcmc(theta0=mode.theta, compute_map=False, **kw)
    supplied = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=mode.vcov, **kw
    )

    assert np.allclose(found.samples, supplied.samples, rtol=1e-12, atol=0.0)
    assert found.accept_rate == pytest.approx(supplied.accept_rate)
    # the recursion really ran, so the seed had somewhere to carry into
    assert np.unique(found.samples, axis=0).shape[0] > 1


def test_mcmc_proposal_covariance_sets_the_proposal_width(mcmc_estimator):
    """Scaling the matrix has to move the acceptance rate, or it is being ignored.
    Adaptation is off so the supplied factor governs every step of the run.
    """
    kw = dict(n_draws=25, burn_in=5, random_state=7, adapt=False)
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    cov = np.asarray(mode.vcov)

    narrow = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=1e-4 * cov, **kw
    )
    base = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=cov, **kw
    )

    assert narrow.accept_rate > base.accept_rate
    assert narrow.samples.std(axis=0).max() < base.samples.std(axis=0).max()


@pytest.mark.parametrize(
    "cov",
    [
        pytest.param(-np.ones((2, 2), dtype=np.float64), id="negative"),
        pytest.param(np.zeros((2, 2), dtype=np.float64), id="singular"),
        pytest.param(
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64), id="indefinite"
        ),
    ],
)
def test_mcmc_rejects_a_proposal_covariance_that_is_not_positive_definite(
    mcmc_estimator, cov
):
    with pytest.raises(MemoryError, match="not positive definite"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE, compute_map=False, proposal_cov=cov, **_MCMC_KW
        )


def test_mcmc_rejects_a_proposal_covariance_with_the_wrong_shape(mcmc_estimator):
    with pytest.raises(ValueError, match=r"Expected shape \(2, 2\), got \(3, 3\)"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE,
            compute_map=False,
            proposal_cov=np.eye(3, dtype=np.float64),
            **_MCMC_KW,
        )


def test_mcmc_rejects_a_proposal_covariance_alongside_compute_map(mcmc_estimator):
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    with pytest.raises(ValueError, match="compute_map"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE,
            compute_map=True,
            proposal_cov=mode.vcov,
            **_MCMC_KW,
        )


def test_map_with_logjac_is_the_mode_the_sampler_starts_from(transformed_estimator):
    """``jacobian=True`` is what makes a precomputed mode reusable."""
    kw = dict(n_draws=30, burn_in=5, random_state=11, cov_fd_step_scale=0.5)
    found = transformed_estimator.mcmc(**kw)

    over_theta = transformed_estimator.map(cov=False, jacobian=True)
    reused = transformed_estimator.mcmc(
        theta0=dict(over_theta.theta), compute_map=False, **kw
    )
    # A theta0 is parameter space, so reusing a mode re-forwards it through the
    # transform. That round trip is not bit-exact under a logit.
    np.testing.assert_allclose(found.samples, reused.samples, rtol=0, atol=1e-9)

    over_params = transformed_estimator.map(cov=False)
    mismatched = transformed_estimator.mcmc(
        theta0=dict(over_params.theta), compute_map=False, **kw
    )
    assert not np.allclose(found.samples, mismatched.samples)


def test_mcmc_returns_expected_shapes_and_stats(mcmc_estimator):
    out = mcmc_estimator.mcmc(
        n_draws=5,
        burn_in=2,
        thin=1,
        theta0=_POST82_START,
        random_state=123,
        adapt=True,
    )
    assert out.param_names == ("psi_pi", "rho_r")
    assert out.samples.shape == (5, 2)
    assert out.logpost_trace.shape == (5,)
    assert 0.0 <= out.accept_rate <= 1.0


def test_mcmc_reports_samples_in_constrained_space_for_log_transform(post82_estimator):
    # sig_r is a shock std (positive support -> Log transform), so mcmc must report
    # its samples in constrained (positive) space, not the unconstrained draw space.
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = post82_estimator(estimated_params={"sig_r": prior})
    out = est.mcmc(
        n_draws=20,
        burn_in=10,
        thin=1,
        random_state=123,
        adapt=False,
    )
    assert np.all(out.samples[:, 0] > 0.0)


def test_mcmc_validation_branches(post82_estimator):
    with pytest.raises(ValueError, match="requires priors"):
        post82_estimator().mcmc(n_draws=1)

    # An empty target set never reaches a routine: the constructor is where it
    # is refused, so mcmc's own emptiness guard is no longer reachable.
    with pytest.raises(ValueError, match="at least one parameter"):
        post82_estimator(estimated_params=[])
