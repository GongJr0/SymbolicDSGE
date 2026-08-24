# type: ignore
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from sympy import Matrix, Symbol

import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator, make_prior
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    CholeskyCorrTransform,
    Identity,
    LogTransform,
    TanhTransform,
)
from SymbolicDSGE.core.config import PairGetterDict, SymbolGetterDict
from SymbolicDSGE.estimation.backend import MatrixPriorBlock


def _with_filter_prep(compiled):
    """Complete a stub with the surface Estimator's construction-time filter prep
    needs. ``Estimator.__init__`` builds the filter run unconditionally now (the
    old duck-typed guard is gone), so every stub must satisfy
    ``prepare_filter_run``. No test on a stub evaluates an objective, so the cfunc
    addresses and P0 are never read; they only have to exist."""
    if not hasattr(compiled, "observable_names"):
        compiled.observable_names = ["y"]
    if not hasattr(compiled, "var_names"):
        compiled.var_names = [
            Symbol(f"s{i}") for i in range(len(compiled.observable_names))
        ]
    if not hasattr(compiled, "cur_syms"):
        compiled.cur_syms = list(compiled.var_names)
    compiled.construct_measurement_cfunc = lambda obs: SimpleNamespace(address=0)
    compiled.construct_observable_jacobian_cfunc = lambda obs: SimpleNamespace(
        address=0
    )
    if not hasattr(compiled, "n_state"):
        compiled.n_state = len(compiled.var_names)
    if not hasattr(compiled, "n_var"):
        compiled.n_var = len(compiled.var_names)
    if getattr(compiled.kalman, "P0", None) is None:
        compiled.kalman.P0 = np.eye(len(compiled.var_names), dtype=np.float64)
    if not hasattr(compiled.kalman, "R_param_names"):
        compiled.kalman.R_param_names = None
    if not hasattr(compiled.kalman, "R_std_param_map"):
        compiled.kalman.R_std_param_map = None
    if getattr(compiled.kalman, "R", None) is None:
        compiled.kalman.R = np.eye(len(compiled.observable_names), dtype=np.float64)
    return compiled


def _stub_compiled():
    a = Symbol("a")
    calibration = SimpleNamespace(parameters={a: float64(0.0)})
    config = SimpleNamespace(calibration=calibration)
    kalman = SimpleNamespace(y_names=["y"])
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[a],
            kalman=kalman,
            observable_names=["y"],
        )
    )


def _stub_compiled_with_r():
    a = Symbol("a")
    meas = Symbol("meas")
    calibration = SimpleNamespace(parameters={a: float64(0.0), meas: float64(1.0)})
    config = SimpleNamespace(calibration=calibration)
    kalman = SimpleNamespace(
        R_param_names=["a"],
        R_builder=lambda *vals: np.array([[vals[0]]], dtype=np.float64),
        y_names=["y"],
    )
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[a, meas],
            kalman=kalman,
            observable_names=["y"],
        )
    )


def _stub_compiled_with_dense_r_block():
    meas_a = Symbol("meas_a")
    meas_b = Symbol("meas_b")
    meas_rho_ab = Symbol("meas_rho_ab")
    calibration = SimpleNamespace(
        parameters={
            meas_a: float64(1.0),
            meas_b: float64(1.0),
            meas_rho_ab: float64(0.0),
        }
    )
    config = SimpleNamespace(calibration=calibration)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    def _R_builder(meas_a_val, meas_b_val, meas_rho_ab_val):
        return np.array(
            [
                [meas_a_val**2, meas_a_val * meas_b_val * meas_rho_ab_val],
                [meas_a_val * meas_b_val * meas_rho_ab_val, meas_b_val**2],
            ],
            dtype=np.float64,
        )

    kalman = SimpleNamespace(
        R=R,
        R_symbolic=Matrix(
            [
                [meas_a**2, meas_a * meas_b * meas_rho_ab],
                [meas_a * meas_b * meas_rho_ab, meas_b**2],
            ]
        ),
        R_param_names=["meas_a", "meas_b", "meas_rho_ab"],
        R_builder=_R_builder,
        R_std_param_map={"A": "meas_a", "B": "meas_b"},
        R_corr_param_map={frozenset(("A", "B")): "meas_rho_ab"},
        y_names=["A", "B"],
    )
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[meas_a, meas_b, meas_rho_ab],
            kalman=kalman,
            observable_names=["A", "B"],
        )
    )


def _stub_compiled_with_sparse_q_block():
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    e3 = Symbol("e3")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    x3 = Symbol("x3")
    sig1 = Symbol("sig1")
    sig2 = Symbol("sig2")
    sig3 = Symbol("sig3")
    rho12 = Symbol("rho12")
    calibration = SimpleNamespace(
        parameters={
            sig1: float64(1.0),
            sig2: float64(1.0),
            sig3: float64(1.0),
            rho12: float64(0.0),
        },
        shock_std=SymbolGetterDict({e1: sig1, e2: sig2, e3: sig3}),
        shock_corr=PairGetterDict(
            {
                frozenset((e1, e2)): rho12,
                frozenset((e1, e3)): None,
                frozenset((e2, e3)): None,
            }
        ),
    )
    config = SimpleNamespace(
        calibration=calibration,
        shocks=[e1, e2, e3],
    )
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[sig1, sig2, sig3, rho12],
            kalman=SimpleNamespace(y_names=["y"]),
            observable_names=["y"],
            var_names=[x1, x2, x3],
            shock_names=("e1", "e2", "e3"),
            n_exog=3,
        )
    )


def test_to_spec_captures_construction_state():
    from SymbolicDSGE.estimation.spec import EstimatorSpec

    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        observables=["y"],
        estimated_params=["a"],
    )

    spec = est.to_spec()

    assert isinstance(spec, EstimatorSpec)
    assert spec.params["estimated_params"] == ["a"]
    assert spec.params["observables"] == ["y"]
    assert spec.params["filter_mode"] == "linear"
    assert spec.params["priors"] is None
    assert spec.y == np.zeros((3, 1)).tolist()
    # the run is not construction state, so nothing about a method is here
    assert "method" not in spec.params


def test_to_spec_reverses_live_scalar_priors():
    from SymbolicDSGE.bayesian.priors import make_prior

    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={
            "a": make_prior(
                distribution="normal",
                parameters={"mean": 0.0, "std": 1.0},
                transform="identity",
                transform_kwargs={},
            )
        },
    )

    spec = est.to_spec()  # live Prior objects reversed to their specs

    prior = spec.params["priors"]["a"]
    assert prior["distribution"] == "normal"
    assert prior["parameters"] == {"mean": 0.0, "std": 1.0}
    assert prior["transform"] == "identity"


def test_mle_records_optimizer_config(post82_estimator):
    est = post82_estimator()
    out = est.mle(
        theta0=np.array([2.0, 0.8], dtype=np.float64),
        bounds=[(1.0, 5.0), (0.0, 0.99)],
        maxiter=10,
    )

    cfg = out.optimizer_config
    assert cfg["method"] == "L-BFGS-B"
    assert cfg["bounds"] == [[1.0, 5.0], [0.0, 0.99]]
    # optimizer kwargs are recorded under "options"; maxiter reflects the call
    assert cfg["options"]["maxiter"] == 10
    assert set(cfg["options"]) == {
        "m",
        "maxiter",
        "maxfun",
        "maxls",
        "factr",
        "pgtol",
        "fd_step",
        "xatol",
        "fatol",
        "jacobian",
        "cov",
        "cov_fd_step_scale",
        "cov_fd_absolute_floor",
    }
    # the starting point is run context too, so it rides the config
    assert "theta0" in cfg
    # config survives projection to the serializable document
    assert out.to_spec()["optimizer_config"] == cfg


def _normal_prior(mean, std):
    return make_prior(
        distribution="normal",
        parameters={"mean": mean, "std": std},
        transform="identity",
    )


@pytest.fixture
def mcmc_estimator(post82_estimator):
    """POST82 estimator with normal priors on ``psi_pi`` / ``rho_r``, ready for
    mcmc. The native mcmc path drives the compiled kernels, so these tests need a
    real model rather than a SimpleNamespace stub + fake loglik."""
    return post82_estimator(
        priors={"psi_pi": _normal_prior(2.0, 0.5), "rho_r": _normal_prior(0.8, 0.1)}
    )


def test_mcmc_records_sampler_config(mcmc_estimator):
    out = mcmc_estimator.mcmc(
        n_draws=5, burn_in=2, thin=1, random_state=7, proposal_scale=0.2
    )

    cfg = out.sampler_config
    assert cfg["random_state"] == 7
    assert cfg["proposal_scale"] == 0.2
    assert set(cfg) == {
        "theta0",
        "adapt",
        "adapt_start",
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
    # n_draws/burn_in/thin stay on the result itself (not duplicated in config)
    assert "n_draws" not in cfg
    assert out.to_spec().meta["sampler_config"] == cfg


def test_mcmc_rejects_invalid_draw_counts(mcmc_estimator):
    """``run_mcmc`` owns these, so they raise past ``_build_native_context``."""
    with pytest.raises(ValueError, match="n_draws must be positive"):
        mcmc_estimator.mcmc(n_draws=0)
    with pytest.raises(ValueError, match="burn_in must be non-negative"):
        mcmc_estimator.mcmc(n_draws=1, burn_in=-1)
    with pytest.raises(ValueError, match="thin must be positive"):
        mcmc_estimator.mcmc(n_draws=1, thin=0)


_MCMC_KW = dict(n_draws=25, burn_in=5, random_state=7)
#: A start the MAP actually walks away from. Most points near the POST82 mode
#: are stationary for L-BFGS-B, which would leave the comparisons below unable
#: to tell a skipped MAP from a MAP that ran and moved nothing.
_OFF_MODE = {"psi_pi": 2.4, "rho_r": 0.85}


def test_mcmc_skipping_the_map_leaves_the_chain_at_theta0(mcmc_estimator):
    """The flag has to move where the chain starts, or it is doing nothing."""
    found = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=True, **_MCMC_KW)
    supplied = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=False, **_MCMC_KW)

    assert not np.array_equal(found.samples, supplied.samples)
    assert found.sampler_config["compute_map"] is True
    assert supplied.sampler_config["compute_map"] is False


def test_mcmc_precomputed_mode_reproduces_the_internal_map_chain(mcmc_estimator):
    """Skipping the MAP changes who finds the mode, not the chain that follows."""
    found = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=True, **_MCMC_KW)
    mode = mcmc_estimator.map(theta0=_OFF_MODE)
    supplied = mcmc_estimator.mcmc(theta0=mode.x, compute_map=False, **_MCMC_KW)

    assert np.array_equal(found.samples, supplied.samples)


def test_mcmc_accepts_a_map_result_as_its_starting_mode(mcmc_estimator):
    """``MAPResult.theta`` names exactly the estimated set, so it is a theta0."""
    mode = mcmc_estimator.map(theta0=_OFF_MODE)
    assert set(mode.theta) == set(mcmc_estimator.param_names)

    from_array = mcmc_estimator.mcmc(theta0=mode.x, compute_map=False, **_MCMC_KW)
    from_dict = mcmc_estimator.mcmc(theta0=mode.theta, compute_map=False, **_MCMC_KW)

    assert np.array_equal(from_array.samples, from_dict.samples)


def test_mcmc_supplied_covariance_reproduces_the_internal_hessian_chain(
    mcmc_estimator,
):
    """The documented workflow: pay for the MAP once, hand the sampler its
    covariance, and get the chain the sampler would have built for itself.

    ``jacobian=True`` is what makes the mode reusable (the chain walks theta), and
    the covariance rides along with it. The comparison is to roundoff rather than
    exact: the internal path factors the Hessian straight into ``chol(H)^-T``,
    while a supplied covariance is refactored out of ``vcov = F @ F.T``, so the
    two factors agree only up to the round trip.
    """
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    found = mcmc_estimator.mcmc(theta0=_OFF_MODE, compute_map=True, **_MCMC_KW)
    supplied = mcmc_estimator.mcmc(
        theta0=mode.x, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
    )

    assert np.allclose(found.samples, supplied.samples, rtol=1e-12, atol=0.0)
    assert np.allclose(
        found.logpost_trace, supplied.logpost_trace, rtol=1e-12, atol=0.0
    )
    assert found.accept_rate == pytest.approx(supplied.accept_rate)


def test_mcmc_supplied_covariance_seeds_the_adaptation_recursion(mcmc_estimator):
    """The supplied matrix is the Haario seed too, not just the first factor.

    ``_MCMC_KW`` never reaches the default ``adapt_start``, so the covariance it
    pins is only the one the proposal opens with. Dropping the start low enough
    for the recursion to run puts the seed on the same footing.
    """
    kw = dict(n_draws=60, burn_in=10, random_state=7, adapt=True, adapt_start=5)
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)

    found = mcmc_estimator.mcmc(theta0=mode.x, compute_map=False, **kw)
    supplied = mcmc_estimator.mcmc(
        theta0=mode.x, compute_map=False, proposal_cov=mode.vcov, **kw
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
        theta0=mode.x, compute_map=False, proposal_cov=1e-4 * cov, **kw
    )
    base = mcmc_estimator.mcmc(theta0=mode.x, compute_map=False, proposal_cov=cov, **kw)
    wide = mcmc_estimator.mcmc(
        theta0=mode.x, compute_map=False, proposal_cov=100.0 * cov, **kw
    )

    assert narrow.accept_rate > base.accept_rate > wide.accept_rate
    assert narrow.samples.std(axis=0).max() < base.samples.std(axis=0).max()
    assert not np.array_equal(narrow.samples, base.samples)
    assert not np.array_equal(wide.samples, base.samples)


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
    """A covariance with no Cholesky factor has no proposal, so the run stops.

    The exception type is the native status trampoline's, shared by every
    non-zero ``sdsge_mcmc_result.status``; the message is what identifies the
    failure.
    """
    with pytest.raises(MemoryError, match="not positive definite"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE, compute_map=False, proposal_cov=cov, **_MCMC_KW
        )


def test_mcmc_rejects_a_proposal_covariance_with_the_wrong_shape(mcmc_estimator):
    """The matrix is indexed against the estimated set, so it is sized by it."""
    with pytest.raises(ValueError, match=r"Expected shape \(2, 2\), got \(3, 3\)"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE,
            compute_map=False,
            proposal_cov=np.eye(3, dtype=np.float64),
            **_MCMC_KW,
        )


def test_mcmc_records_the_supplied_covariance_on_the_result(mcmc_estimator):
    """A run driven by a user matrix has to be distinguishable from one that
    built its own, or the result cannot be reconstructed from its config."""
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    out = mcmc_estimator.mcmc(
        theta0=mode.x, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
    )

    cfg = out.sampler_config
    assert cfg["compute_map"] is False
    assert np.array_equal(np.asarray(cfg["proposal_cov"], dtype=np.float64), mode.vcov)


def test_mcmc_rejects_a_proposal_covariance_alongside_compute_map(mcmc_estimator):
    """The internal MAP builds its own covariance, which would silently win."""
    mode = mcmc_estimator.map(theta0=_OFF_MODE, jacobian=True)
    with pytest.raises(ValueError, match="compute_map"):
        mcmc_estimator.mcmc(
            theta0=_OFF_MODE,
            compute_map=True,
            proposal_cov=mode.vcov,
            **_MCMC_KW,
        )


def test_mle_reports_the_covariance_at_the_optimum(post82_estimator):
    """Uncertainty is on by default, and se is the root of vcov's diagonal
    wherever the transforms are the identity."""
    est = post82_estimator()
    res = est.mle(bounds=[(1.0, 5.0), (0.0, 0.99)])

    assert res.cov_status == 0
    assert res.vcov.shape == (len(res.theta), len(res.theta))
    assert np.all(np.isfinite(res.vcov))
    assert set(res.se) == set(res.theta)
    assert np.allclose(res.vcov, res.vcov.T)
    for i, name in enumerate(res.theta):
        assert float(res.se[name]) == pytest.approx(float(np.sqrt(res.vcov[i, i])))


def test_mle_covariance_is_opt_out_and_does_not_move_the_estimate(post82_estimator):
    est = post82_estimator()
    kw = dict(bounds=[(1.0, 5.0), (0.0, 0.99)])
    with_cov = est.mle(**kw)
    without = est.mle(cov=False, **kw)

    assert without.vcov is None
    assert without.se is None
    assert without.cov_status == 0
    assert without.theta == with_cov.theta


def test_se_is_in_the_space_theta_reports(post82_estimator):
    """`sig_r` carries a Log transform, so its se is not sqrt(diag(vcov)): the
    covariance is over theta and has to cross the transform to sit beside a
    constrained value."""
    est = post82_estimator(estimated_params=("psi_pi", "sig_r"))
    res = est.mle()
    names = list(res.theta)
    assert [type(est._param_transforms[n]).__name__ for n in names] == [
        "Identity",
        "LogTransform",
    ]

    se_theta = np.sqrt(np.diag(res.vcov))
    # d exp(t)/dt is exp(t), which is the constrained value itself
    assert float(res.se["psi_pi"]) == pytest.approx(float(se_theta[0]))
    assert float(res.se["sig_r"]) == pytest.approx(
        float(res.theta["sig_r"]) * float(se_theta[1]), rel=1e-6
    )
    assert float(res.se["sig_r"]) != pytest.approx(float(se_theta[1]))


@pytest.fixture
def transformed_estimator(post82_estimator):
    """``rho_r`` under a logit, ``psi_pi`` under the identity.

    A non-identity transform is what makes the two prior densities differ: with
    everything on the identity the jacobian is zero and there is nothing for
    these tests to see.
    """
    return post82_estimator(
        estimated_params=("psi_pi", "rho_r"),
        priors={
            "psi_pi": make_prior(
                distribution="normal",
                parameters={"mean": 2.19, "std": 0.5},
                transform="identity",
            ),
            "rho_r": make_prior(
                distribution="beta",
                parameters={"a": 8.0, "b": 2.0},
                transform="logit",
            ),
        },
    )


def test_map_include_logjac_selects_a_different_mode(transformed_estimator):
    """The jacobian moves the mode, which is the whole reason for the flag."""
    over_params = transformed_estimator.map(cov=False)
    over_theta = transformed_estimator.map(cov=False, jacobian=True)

    assert over_params.success and over_theta.success
    assert not np.allclose(over_params.x, over_theta.x)
    # the coupling carries: psi_pi is on the identity and still moves
    assert float(over_params.theta["psi_pi"]) != pytest.approx(
        float(over_theta.theta["psi_pi"])
    )


def test_map_with_logjac_is_the_mode_the_sampler_starts_from(transformed_estimator):
    """``jacobian=True`` is what makes a precomputed mode reusable.

    The chain walks theta, so the MAP it finds for itself carries the jacobian.
    A mode found without it starts the chain somewhere else.
    """
    kw = dict(n_draws=30, burn_in=5, random_state=11, cov_fd_step_scale=0.5)
    found = transformed_estimator.mcmc(**kw)

    over_theta = transformed_estimator.map(cov=False, jacobian=True)
    reused = transformed_estimator.mcmc(theta0=over_theta.x, compute_map=False, **kw)
    assert np.array_equal(found.samples, reused.samples)

    over_params = transformed_estimator.map(cov=False)
    mismatched = transformed_estimator.mcmc(
        theta0=over_params.x, compute_map=False, **kw
    )
    assert not np.array_equal(found.samples, mismatched.samples)


def test_map_without_priors_raises():
    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="requires priors"):
        est.map(theta0=np.array([0.0], dtype=np.float64))


def test_mcmc_returns_expected_shapes_and_stats(mcmc_estimator):
    out = mcmc_estimator.mcmc(
        n_draws=40,
        burn_in=40,
        thin=1,
        theta0=np.array([2.0, 0.8], dtype=np.float64),
        random_state=123,
        adapt=True,
    )
    assert out.param_names == ["psi_pi", "rho_r"]
    assert out.samples.shape == (40, 2)
    assert out.logpost_trace.shape == (40,)
    assert 0.0 <= out.accept_rate <= 1.0


def test_mcmc_seed_zero_is_exactly_reproducible(mcmc_estimator):
    kwargs = dict(
        n_draws=30,
        burn_in=30,
        thin=1,
        theta0=np.array([2.0, 0.8], dtype=np.float64),
        random_state=0,
        adapt=True,
    )
    out1 = mcmc_estimator.mcmc(**kwargs)
    out2 = mcmc_estimator.mcmc(**kwargs)

    assert np.array_equal(out1.samples, out2.samples)
    assert np.array_equal(out1.logpost_trace, out2.logpost_trace)
    assert out1.accept_rate == pytest.approx(out2.accept_rate)


def test_estimator_make_prior_utility():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    assert isinstance(prior, Prior)


def test_estimation_reports_warning_count_once(post82_estimator, capsys):
    est = post82_estimator()
    _ = est.mle(
        theta0=np.array([2.0, 0.8], dtype=np.float64),
        bounds=[(1.0, 5.0), (0.0, 0.99)],
    )
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "BK stability warnings encountered" in ln
    ]
    assert len(lines) == 1


def test_theta_to_params_uses_prior_inverse_transform():
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    params = est.theta_to_params(np.array([-1.0], dtype=np.float64))
    assert params["a"] > 0.0


def test_params_to_theta_applies_forward_transform_for_mapping():
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    theta = est.params_to_theta({"a": np.e})
    assert np.allclose(theta[0], 1.0)


def test_matrix_prior_on_R_reparameterizes_pairwise_correlation_block():
    prior = Prior(
        dist=LKJChol(eta=2.0, K=2, random_state=None),
        transform=CholeskyCorrTransform(K=2),
    )
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={"R_corr": prior},
    )

    theta = est.params_to_theta({"meas_rho_ab": 0.3})
    assert np.allclose(theta[0], np.arctanh(0.3))

    params = est.theta_to_params(theta)
    assert params["meas_rho_ab"] == pytest.approx(0.3)
    assert params["meas_a"] == pytest.approx(1.0)
    assert params["meas_b"] == pytest.approx(1.0)

    # The block owns its member's density; the packed-program parity for it is
    # asserted against a real model in test_estimator_lkj_integration.
    block = est._matrix_blocks["R_corr"]
    assert block.member_names == ["meas_rho_ab"]
    assert block.prior is prior


def test_matrix_prior_created_via_make_prior_uses_cholesky_corr_transform():
    prior = make_prior(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 2},
        transform="cholesky_corr",
    )
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={"R_corr": prior},
    )

    block = est._matrix_blocks["R_corr"]
    assert isinstance(block.prior.transform, CholeskyCorrTransform)
    assert block.prior.transform.K == 2


def test_matrix_key_in_estimated_params_expands_to_member_names():
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=2, random_state=None),
                transform=CholeskyCorrTransform(K=2),
            )
        },
    )

    assert est.param_names == ["meas_rho_ab"]
    assert list(est.priors.keys()) == ["R_corr"]


def test_estimated_params_none_uses_prior_keys_when_priors_supplied():
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=None,
        priors={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=2, random_state=None),
                transform=CholeskyCorrTransform(K=2),
            ),
            "meas_a": make_prior("log_normal", {"mean": 0.0, "std": 1.0}, "log"),
        },
    )

    assert est.param_names == ["meas_rho_ab", "meas_a"]
    assert list(est.priors.keys()) == ["R_corr", "meas_a"]


def test_priors_outside_estimated_params_are_rejected():
    # A prior on something not being estimated is a mistake, not a no-op: it
    # would otherwise be dropped and the run would silently ignore it.
    with pytest.raises(ValueError, match="not in the estimated parameters"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=2),
                ),
                "meas_a": make_prior("log_normal", {"mean": 0.0, "std": 1.0}, "log"),
            },
        )


def test_matrix_prior_overlap_with_scalar_component_prior_raises():
    with pytest.raises(ValueError, match="meas_rho_ab"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=2),
                ),
                "meas_rho_ab": make_prior(
                    "normal", {"mean": 0.0, "std": 1.0}, "identity"
                ),
            },
        )


# Matrix-prior-on-R MCMC on a real model (valid-correlation-support) is covered
# by test_estimator_lkj_integration.test_matrix_prior_on_R_runs_full_mcmc_with_
# real_likelihood; the SimpleNamespace stub version can no longer drive the native
# path and is not re-added here.


def test_sparse_q_block_for_lkj_prior_raises_descriptive_error():
    with pytest.raises(ValueError, match="dense correlation block") as excinfo:
        Estimator(
            compiled=_stub_compiled_with_sparse_q_block(),
            y=np.zeros((4, 1), dtype=np.float64),
            estimated_params=["Q_corr"],
            priors={
                "Q_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                )
            },
        )

    msg = str(excinfo.value)
    assert "sparse" in msg
    assert "fall back to their defaults" in msg
    assert "config DSL" in msg
    assert "placeholder default value" in msg


def test_mcmc_reports_samples_in_constrained_space_for_log_transform(post82_estimator):
    # sig_r is a shock std (positive support -> Log transform), so mcmc must report
    # its samples in constrained (positive) space, not the unconstrained draw space.
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = post82_estimator(estimated_params=["sig_r"], priors={"sig_r": prior})
    out = est.mcmc(
        n_draws=20,
        burn_in=10,
        thin=1,
        random_state=123,
        adapt=False,
    )
    assert np.all(out.samples[:, 0] > 0.0)


def test_loglik_reads_theta_through_the_parameter_transform(post82_estimator):
    """The transform sits on the likelihood path, not just on the prior: a
    log-transformed parameter at ``log(v)`` scores the same as an untransformed
    one at ``v``. Only the transform differs between the two estimators, since
    the likelihood is evaluated with priors off."""
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    transformed = post82_estimator(
        estimated_params=("psi_pi",), priors={"psi_pi": prior}
    )
    identity = post82_estimator(estimated_params=("psi_pi",))

    value = 2.0
    assert transformed.loglik(
        np.array([np.log(value)], dtype=np.float64)
    ) == pytest.approx(identity.loglik(np.array([value], dtype=np.float64)), rel=1e-9)


def test_estimator_constructor_and_lkj_prior_validation_error_branches():
    with pytest.raises(ValueError, match="are not estimable targets"):
        Estimator(
            compiled=_stub_compiled(),
            y=np.zeros((3, 1), dtype=np.float64),
            estimated_params=["ghost"],
        )

    with pytest.raises(ValueError, match="specified more than once"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr", "meas_rho_ab"],
            priors={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=2),
                )
            },
        )

    with pytest.raises(ValueError, match="CholeskyCorrTransform"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                # Support-valid but not a CholeskyCorrTransform: exercises the
                # estimator's LKJ-transform check (AffineLogit(-1,1) matches the
                # LKJChol (-1, 1) support so Prior construction itself succeeds).
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=AffineLogitTransform(low=-1.0, high=1.0),
                )
            },
        )

    with pytest.raises(ValueError, match="matching K between"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                )
            },
        )

    with pytest.raises(ValueError, match="requires a LKJChol distribution"):
        Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")
            },
        )


def test_cov_to_corr_and_matrix_resolution_error_branches():
    with pytest.raises(ValueError, match="square covariance matrix"):
        Estimator._cov_to_corr(np.array([1.0], dtype=np.float64), "R")
    with pytest.raises(ValueError, match="symmetric covariance matrix"):
        Estimator._cov_to_corr(
            np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64), "R"
        )
    with pytest.raises(ValueError, match="strictly positive diagonal variances"):
        Estimator._cov_to_corr(
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64), "R"
        )

    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )

    with pytest.raises(ValueError, match="named variance parameter"):
        est._build_matrix_resolution(
            key="R_corr",
            labels=["a"],
            std_param_map={},
            corr_param_map={},
        )

    with pytest.raises(ValueError, match="unique named variance parameter"):
        est._build_matrix_resolution(
            key="R_corr",
            labels=["a", "b"],
            std_param_map={"a": "sig", "b": "sig"},
            corr_param_map={frozenset(("b", "a")): "rho_ab"},
        )

    with pytest.raises(ValueError, match="unique named parameter per correlation pair"):
        est._build_matrix_resolution(
            key="R_corr",
            labels=["a", "b", "c"],
            std_param_map={"a": "sig_a", "b": "sig_b", "c": "sig_c"},
            corr_param_map={
                frozenset(("b", "a")): "rho_shared",
                frozenset(("c", "a")): "rho_shared",
                frozenset(("c", "b")): "rho_cb",
            },
        )


def test_resolve_r_and_effective_observables_error_paths():
    a = Symbol("a")
    compiled_no_kalman = SimpleNamespace(
        config=SimpleNamespace(
            calibration=SimpleNamespace(parameters={a: float64(0.0)})
        ),
        calib_params=[a],
        observable_names=["y"],
        kalman=None,
    )
    # Without a config there is nothing to build R from, so the estimator fails
    # fast at construction rather than lazily when the R block is resolved. P0
    # has a default and is not part of the demand.
    with pytest.raises(ValueError, match="R must be provided"):
        Estimator(
            compiled=compiled_no_kalman,
            y=np.zeros((3, 1), dtype=np.float64),
            estimated_params=["a"],
        )

    est_missing_meta = Estimator(
        compiled=_with_filter_prep(
            SimpleNamespace(
                config=SimpleNamespace(
                    calibration=SimpleNamespace(parameters={a: float64(0.0)})
                ),
                calib_params=[a],
                kalman=SimpleNamespace(
                    y_names=["y"],
                    R=np.eye(1, dtype=np.float64),
                    R_std_param_map=None,
                    R_corr_param_map=None,
                ),
                observable_names=["y"],
            )
        ),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="parser-generated R std/correlation metadata"):
        est_missing_meta._resolve_R()

    # Unknown observables are now rejected at construction by the filter prep.
    with pytest.raises(ValueError, match="Unknown observables"):
        Estimator(
            compiled=_stub_compiled(),
            y=np.zeros((3, 1), dtype=np.float64),
            observables=["ghost"],
            estimated_params=["a"],
        )


def test_theta_conversion_logprior_and_safe_wrapper_error_branches():
    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")},
    )

    with pytest.raises(ValueError, match="missing estimated parameters"):
        est.params_to_theta({})
    with pytest.raises(ValueError, match="params array must be 1D"):
        est.params_to_theta(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match estimated parameter count"):
        est.params_to_theta(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="theta must be a 1D array"):
        est.theta_to_params(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match estimated parameter count"):
        est.theta_to_params(np.array([1.0, 2.0], dtype=np.float64))


def test_mcmc_validation_branches():
    est = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")},
    )

    est_no_priors = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="requires priors"):
        est_no_priors.mcmc(n_draws=1)

    est_empty = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=[],
    )
    est_empty.priors = {
        "dummy": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")
    }
    with pytest.raises(ValueError, match="No estimated parameters"):
        est_empty.mcmc(n_draws=1)


def test_resolve_q_missing_pair_key_and_block_validation_branches(monkeypatch):
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    sig1 = Symbol("sig1")
    sig2 = Symbol("sig2")
    calibration = SimpleNamespace(
        parameters={sig1: float64(1.0), sig2: float64(1.0)},
        shock_std=SymbolGetterDict({e1: sig1, e2: sig2}),
        shock_corr={},
    )
    compiled = _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(calibration=calibration, shocks=[e1, e2]),
            calib_params=[sig1, sig2],
            kalman=SimpleNamespace(y_names=["y"]),
            observable_names=["y"],
            var_names=[x1, x2],
            shock_names=("e1", "e2"),
            n_exog=2,
        )
    )
    est = Estimator(
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["sig1"],
    )
    block = est._resolve_Q()
    present = {(int(r), int(c)) for r, c in block.positions}
    missing = [
        (block.labels[row], block.labels[col])
        for row in range(1, block.dim)
        for col in range(row)
        if (row, col) not in present
    ]
    assert missing == [("e2", "e1")]

    est_base = Estimator(
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    # These branches drive _build_matrix_prior_blocks directly with monkeypatched
    # resolutions; mark R_corr as a requested block so the loop runs over it.
    est_base._requested_reserved_keys = ("R_corr",)

    res_dim1 = MatrixPriorBlock(
        dim=1,
        labels=["A"],
        member_names=[],
        positions=np.empty((0, 2), dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est_base.priors = {"R_corr": object()}
    monkeypatch.setattr(
        est_base,
        "_is_lkj_prior",
        lambda name, prior_obj: SimpleNamespace(
            dist=SimpleNamespace(_K=1), logpdf=lambda z: float64(0.0)
        ),
    )
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_dim1)
    with pytest.raises(ValueError, match="dimension at least 2"):
        est_base._build_matrix_prior_blocks()

    res_short = MatrixPriorBlock(
        dim=3,
        labels=["A", "B", "C"],
        member_names=["rho_ba", "rho_ca"],
        positions=np.array([[1, 0], [2, 0]], dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est_base.priors = {
        "R_corr": Prior(
            dist=LKJChol(eta=2.0, K=3, random_state=None),
            transform=CholeskyCorrTransform(K=3),
        )
    }
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_short)
    with pytest.raises(ValueError, match="dense correlation block"):
        est_base._build_matrix_prior_blocks()

    res_missing = est_base._build_matrix_resolution(
        key="R_corr",
        labels=["A", "B"],
        std_param_map={"A": "sig_a", "B": "sig_b"},
        corr_param_map={frozenset(("B", "A")): "rho_ba"},
    )
    est_base.priors = {
        "R_corr": Prior(
            dist=LKJChol(eta=2.0, K=2, random_state=None),
            transform=CholeskyCorrTransform(K=2),
        )
    }
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_missing)
    with pytest.raises(ValueError, match="Missing from estimated_params"):
        est_base._build_matrix_prior_blocks()


def test_matrix_block_overlap_k_mismatch_and_invalid_corr_error(monkeypatch):
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_rho_ab"],
    )
    r_resolution = est._build_matrix_resolution(
        key="R_corr",
        labels=["A", "B"],
        std_param_map={"A": "meas_a", "B": "meas_b"},
        corr_param_map={frozenset(("B", "A")): "meas_rho_ab"},
    )
    q_resolution = MatrixPriorBlock(
        dim=2,
        labels=["u", "v"],
        member_names=["meas_rho_ab"],
        positions=np.array([[1, 0]], dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est.priors = {
        "R_corr": Prior(
            dist=LKJChol(eta=2.0, K=2, random_state=None),
            transform=CholeskyCorrTransform(K=2),
        ),
        "Q_corr": Prior(
            dist=LKJChol(eta=2.0, K=2, random_state=None),
            transform=CholeskyCorrTransform(K=2),
        ),
    }
    est._requested_reserved_keys = ("R_corr", "Q_corr")
    monkeypatch.setattr(est, "_resolve_R", lambda params=None: r_resolution)
    monkeypatch.setattr(est, "_resolve_Q", lambda params=None: q_resolution)
    with pytest.raises(ValueError, match="cannot share member parameters"):
        est._build_matrix_prior_blocks()

    est_k = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
    )
    est_k.priors = {
        "R_corr": Prior(
            dist=LKJChol(eta=2.0, K=3, random_state=None),
            transform=CholeskyCorrTransform(K=3),
        )
    }
    monkeypatch.setattr(est_k, "_resolve_R", lambda params=None: r_resolution)
    with pytest.raises(ValueError, match="has K=3"):
        est_k._build_matrix_prior_blocks()

    block = (
        est_k._build_matrix_prior_blocks.__self__._matrix_blocks
        if hasattr(est_k, "_matrix_blocks")
        else {}
    )
    good_est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=2, random_state=None),
                transform=CholeskyCorrTransform(K=2),
            )
        },
    )
    good_block = good_est._matrix_blocks["R_corr"]
    bad_corr = np.array([[1.0, 1.2], [1.2, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="do not form a valid"):
        good_est._block_cpc_from_corr(good_block, bad_corr)


def test_mle_std_member_without_prior_gets_log_transform():
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_a"],
    )
    # A variance estimated prior-free is positivity-constrained by role.
    assert isinstance(est._param_transforms["meas_a"], LogTransform)


def test_mle_isolated_scalar_corr_without_prior_gets_tanh_transform():
    est = Estimator(
        compiled=_stub_compiled_with_sparse_q_block(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["rho12"],
    )
    # rho12 is the sole named Q correlation (e1, e2): sparse and isolated, so it
    # stays a standalone scalar tanh rather than folding into a block.
    assert "Q_corr" not in est._matrix_blocks
    assert isinstance(est._param_transforms["rho12"], TanhTransform)


def test_mle_full_dense_corr_set_promotes_to_cpc_block():
    est = Estimator(
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_rho_ab"],
    )
    # The dense R correlation set folds into an R_corr CPC block instead of a
    # standalone scalar; its member is block-handled (scalar transform unused).
    assert "R_corr" in est._matrix_blocks
    assert "meas_rho_ab" in est._matrix_blocks["R_corr"].member_names
    assert isinstance(est._param_transforms["meas_rho_ab"], Identity)


def test_spd_std_member_warns_on_conflicting_prior_transform():
    # An Identity-transform prior on a variance would map onto R, not (0, inf),
    # so the role default is substituted and the substitution is announced.
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    with pytest.warns(UserWarning, match="requires a constraint to"):
        est = Estimator(
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["meas_a"],
            priors={"meas_a": prior},
        )
    assert isinstance(est._param_transforms["meas_a"], LogTransform)


@pytest.mark.parametrize("include_logjac", [False, True])
def test_logpost_decomposes_into_loglik_and_logprior(post82_estimator, include_logjac):
    """The three objectives answer consistently under either convention: the
    jacobian is the prior's to carry, so it moves both sides together."""
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = post82_estimator(estimated_params=("psi_pi",), priors={"psi_pi": prior})
    theta = np.array([np.log(2.0)], dtype=np.float64)

    assert est.logpost(theta, include_logjac) == pytest.approx(
        est.loglik(theta) + est.logprior(theta, include_logjac), rel=1e-12
    )


@pytest.mark.parametrize("estimated", [["psi_pi"], ["psi_pi", "rho_r"]])
def test_mcmc_adaptation_runs_for_scalar_and_vector(post82_estimator, estimated):
    # The native running covariance has no d==1 special case (a 1x1 covariance
    # subsumes it), so adaptation must run cleanly for both scalar and vector theta.
    priors = {
        "psi_pi": _normal_prior(2.0, 0.5),
        "rho_r": _normal_prior(0.8, 0.1),
    }
    est = post82_estimator(
        estimated_params=estimated, priors={n: priors[n] for n in estimated}
    )
    out = est.mcmc(
        n_draws=10,
        burn_in=10,
        thin=1,
        random_state=123,
        adapt=True,
        adapt_start=0,
    )
    assert out.samples.shape == (10, len(estimated))
