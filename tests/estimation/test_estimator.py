# type: ignore
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from sympy import Matrix, Symbol

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator, make_prior
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    CholeskyCorrTransform,
)
from SymbolicDSGE.core.config import PairGetterDict, SymbolGetterDict
from SymbolicDSGE.estimation.prior_program import TransformCode, build_prior_tables
from SymbolicDSGE.estimation.resolvers import (
    _CORR_LIMIT,
    _STD_FLOOR,
    _resolve_Q,
    _resolve_R,
    active_Q,
    active_R,
    resolve_bounds,
    theta_from_params,
)

#: Bounds for ``post82_estimator``'s default ("psi_pi", "rho_r") pair. Bounds
#: are keyed by name, so a positional pair no longer says which parameter it
#: belongs to.
_POST82_BOUNDS = {"psi_pi": (1.0, 5.0), "rho_r": (0.0, 0.99)}
_POST82_START = {"psi_pi": 2.0, "rho_r": 0.8}

#: A correlation prior whose transform already confines it to (-1, 1), so any
#: box the resolution writes is the caller's and not the role's.
_BOUNDED_CORR_PRIOR = make_prior(
    "trunc_normal",
    parameters={"mean": 0.0, "std": 0.5, "low": -1.0, "high": 1.0},
    transform="affine_logit",
    transform_kwargs={"low": -1.0, "high": 1.0},
)


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
    compiled.construct_measurement_jacobian_cfunc = lambda obs: SimpleNamespace(
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
    if not hasattr(compiled.kalman, "R_corr_param_map"):
        compiled.kalman.R_corr_param_map = None
    # active_Q reads these three directly, so a stub that names no shocks still
    # has to say so rather than omit them.
    if not hasattr(compiled, "shock_names"):
        compiled.shock_names = []
    if not hasattr(compiled.config.calibration, "shock_std"):
        compiled.config.calibration.shock_std = {}
    if not hasattr(compiled.config.calibration, "shock_corr"):
        compiled.config.calibration.shock_corr = {}
    if getattr(compiled.kalman, "R", None) is None:
        compiled.kalman.R = np.eye(len(compiled.observable_names), dtype=np.float64)
    return compiled


@pytest.fixture
def role_context():
    """Packed transform codes and the theta box for one stub spec.

    ``estim_context`` also resolves the steady-state seed, the param map and
    both covariance specs, none of which these cases assert and none of which
    these stubs can reach, so this runs only the two steps that decide a
    slot's transform and its box. The role merge mirrors the one
    ``estim_context`` does inline. ``observables`` narrows the active set, which
    is what sizes a resolved R block.
    """

    def _build(compiled, y_shape, estimated_params, observables=None):
        est = Estimator(
            compiled=compiled,
            y=np.zeros(y_shape, dtype=np.float64),
            observables=observables,
            estimated_params=estimated_params,
        )
        observables = est._prepared_filter.observables
        tables, transforms = build_prior_tables(
            param_index=est._param_index,
            priors=est.priors,
            matrix_blocks=est._matrix_blocks,
            compiled=est.compiled,
            kalman=est.kalman,
            observables=observables,
        )
        roles = None
        if est.priors is None:
            std_q, corr_q = active_Q(est.param_names, est.compiled)
            std_r, corr_r = active_R(est.param_names, est.kalman, observables)
            roles = (std_q | std_r, corr_q | corr_r)
        lo, hi = resolve_bounds(
            None, est._param_index, est._matrix_blocks, transforms, roles
        )
        return SimpleNamespace(
            est=est,
            transforms=transforms,
            codes={
                name: TransformCode(int(tables.transform_codes[i]))
                for name, i in est._param_index.items()
            },
            box={
                name: (float(lo[i]), float(hi[i]))
                for name, i in est._param_index.items()
            },
        )

    return _build


def _stub_compiled_with_dense_r_block():
    meas_a = Symbol("meas_a")
    meas_b = Symbol("meas_b")
    meas_rho_ab = Symbol("meas_rho_ab")
    calibration = SimpleNamespace(
        parameters=SymbolGetterDict(
            {
                "meas_a": float64(1.0),
                "meas_b": float64(1.0),
                "meas_rho_ab": float64(0.0),
            }
        )
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
        R_std_param_map=SymbolGetterDict({"A": "meas_a", "B": "meas_b"}),
        R_corr_param_map=PairGetterDict({frozenset(("A", "B")): "meas_rho_ab"}),
        y_names=["A", "B"],
    )
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=["meas_a", "meas_b", "meas_rho_ab"],
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
    calibration = SimpleNamespace(
        parameters=SymbolGetterDict(
            {
                "sig1": float64(1.0),
                "sig2": float64(1.0),
                "sig3": float64(1.0),
                "rho12": float64(0.0),
            }
        ),
        shock_std=SymbolGetterDict({e1: "sig1", e2: "sig2", e3: "sig3"}),
        shock_corr=PairGetterDict(
            {
                frozenset((e1, e2)): "rho12",
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
            calib_params=["sig1", "sig2", "sig3", "rho12"],
            kalman=SimpleNamespace(y_names=["y"]),
            observable_names=["y"],
            var_names=[x1, x2, x3],
            shock_names=("e1", "e2", "e3"),
            n_exog=3,
        )
    )


def test_to_spec_captures_construction_state(post82):
    from SymbolicDSGE.estimation.spec import EstimatorSpec

    est = Estimator(
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        estimated_params=["psi_pi"],
        ss_seed=post82["steady"],
    )

    spec = est.to_spec()

    assert isinstance(spec, EstimatorSpec)
    assert spec.params["estimated_params"] == ["psi_pi"]
    assert spec.params["observables"] == post82["obs"]
    assert spec.params["filter_mode"] == "linear"
    assert spec.params["priors"] is None
    assert spec.y == post82["y"].to_numpy().tolist()
    # the run is not construction state, so nothing about a method is here
    assert "method" not in spec.params


def test_to_spec_reverses_live_scalar_priors(post82):
    from SymbolicDSGE.bayesian.priors import make_prior

    est = Estimator(
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        ss_seed=post82["steady"],
        estimated_params={
            "psi_pi": make_prior(
                distribution="normal",
                parameters={"mean": 0.0, "std": 1.0},
                transform="identity",
                transform_kwargs={},
            )
        },
    )

    spec = est.to_spec()  # live Prior objects reversed to their specs

    prior = spec.params["priors"]["psi_pi"]
    assert prior["distribution"] == "normal"
    assert prior["parameters"] == {"mean": 0.0, "std": 1.0}
    assert prior["transform"] == "identity"


def test_mle_records_optimizer_config(post82_estimator):
    est = post82_estimator()
    out = est.mle(
        theta0=_POST82_START,
        bounds=_POST82_BOUNDS,
        maxiter=10,
    )

    cfg = out.optimizer_config
    assert cfg["method"] == "L-BFGS-B"
    assert cfg["bounds"] == {"psi_pi": [1.0, 5.0], "rho_r": [0.0, 0.99]}
    assert cfg["maxiter"] == 10
    assert set(cfg) > {
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
        estimated_params={
            "psi_pi": _normal_prior(2.0, 0.5),
            "rho_r": _normal_prior(0.8, 0.1),
        }
    )


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
    assert set(mode.theta) == set(mcmc_estimator.param_names)
    supplied = mcmc_estimator.mcmc(theta0=mode.theta, compute_map=False, **_MCMC_KW)

    assert np.array_equal(found.samples, supplied.samples)


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
        theta0=mode.theta, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
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
    wide = mcmc_estimator.mcmc(
        theta0=mode.theta, compute_map=False, proposal_cov=100.0 * cov, **kw
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
        theta0=mode.theta, compute_map=False, proposal_cov=mode.vcov, **_MCMC_KW
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
    res = est.mle(bounds=_POST82_BOUNDS)

    assert res.cov_status == 0
    assert res.vcov.shape == (len(res.theta), len(res.theta))
    assert np.all(np.isfinite(res.vcov))
    assert set(res.se) == set(res.theta)
    assert np.allclose(res.vcov, res.vcov.T)
    for i, name in enumerate(res.theta):
        assert float(res.se[name]) == pytest.approx(float(np.sqrt(res.vcov[i, i])))


def test_mle_covariance_is_opt_out_and_does_not_move_the_estimate(post82_estimator):
    est = post82_estimator()
    kw = dict(bounds=_POST82_BOUNDS)
    with_cov = est.mle(**kw)
    without = est.mle(cov=False, **kw)

    assert without.vcov is None
    assert without.se is None
    assert without.cov_status == 0
    assert without.theta == with_cov.theta


def test_mle_se_is_the_plain_diagonal(post82_estimator):
    """MLE carries no transform, so theta is the parameter and the covariance
    needs no change of variables to sit beside the value it describes.

    ``sig_r`` is the case worth pinning: a standard deviation, which the role
    box floors rather than a transform reparameterizing, so its se would be the
    one to pick up a jacobian if any slot did.
    """
    est = post82_estimator(estimated_params=("sig_r",))
    _, _, lo, _ = est._build_call_context(priored=False)
    assert lo[0] == _STD_FLOOR

    res = est.mle()
    assert res.cov_status == 0
    se_theta = np.sqrt(np.diag(res.vcov))
    for i, name in enumerate(res.theta):
        assert float(res.se[name]) == pytest.approx(float(se_theta[i]))


@pytest.fixture
def transformed_estimator(post82_estimator):
    """``rho_r`` under a logit, ``psi_pi`` under the identity.

    A non-identity transform is what makes the two prior densities differ: with
    everything on the identity the jacobian is zero and there is nothing for
    these tests to see.
    """
    return post82_estimator(
        estimated_params={
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
    reused = transformed_estimator.mcmc(
        theta0=dict(over_theta.theta), compute_map=False, **kw
    )
    # A theta0 is parameter space, so reusing a mode re-forwards it through the
    # transform. That round trip is not bit-exact under a logit, which puts the
    # two chains within tolerance of each other rather than on top.
    np.testing.assert_allclose(found.samples, reused.samples, rtol=0, atol=1e-9)

    over_params = transformed_estimator.map(cov=False)
    mismatched = transformed_estimator.mcmc(
        theta0=dict(over_params.theta), compute_map=False, **kw
    )
    assert not np.allclose(found.samples, mismatched.samples)


def test_map_without_priors_raises(post82_estimator):
    est = post82_estimator()
    with pytest.raises(ValueError, match="requires priors"):
        est.map()


def test_mcmc_returns_expected_shapes_and_stats(mcmc_estimator):
    out = mcmc_estimator.mcmc(
        n_draws=40,
        burn_in=40,
        thin=1,
        theta0=_POST82_START,
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
        theta0=_POST82_START,
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
        theta0=_POST82_START,
        bounds=_POST82_BOUNDS,
    )
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "BK stability warnings encountered" in ln
    ]
    assert len(lines) == 1


def test_forward_map_applies_the_prior_transform(post82, role_context):
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    ctx = role_context(post82["compiled"], (3, 3), {"psi_pi": prior})

    theta = theta_from_params(
        np.array([np.e], dtype=np.float64),
        ctx.est.param_names,
        ctx.est._matrix_blocks,
        ctx.transforms,
    )
    assert np.allclose(theta[0], 1.0)


def test_matrix_prior_on_R_reparameterizes_pairwise_correlation_block(
    post82, role_context
):
    # Two observables leave R one pair, where the CPC map is exactly tanh and the
    # expected theta has a closed form. The full three-observable block is
    # covered in test_estimator_lkj_integration.
    prior = Prior(
        dist=LKJChol(eta=2.0, K=2, random_state=None),
        transform=CholeskyCorrTransform(K=2),
    )
    ctx = role_context(
        post82["compiled"], (4, 2), {"R_corr": prior}, observables=["Infl", "Rate"]
    )
    est = ctx.est

    theta = theta_from_params(
        np.array([0.3], dtype=np.float64),
        est.param_names,
        est._matrix_blocks,
        ctx.transforms,
    )
    assert np.allclose(theta[0], np.arctanh(0.3))

    # The block owns its member's reparameterization; the packed-program parity
    # for it is asserted against a real model in test_estimator_lkj_integration.
    block = est._matrix_blocks["R_corr"]
    assert block.member_names == ["meas_rho_ir"]
    assert block.K == 2


def test_matrix_prior_created_via_make_prior_uses_cholesky_corr_transform(post82):
    prior = make_prior(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 3},
        transform="cholesky_corr",
    )
    est = Estimator(
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        ss_seed=post82["steady"],
        estimated_params={"R_corr": prior},
    )

    # make_prior reconciles the distribution's K with the transform's, which is
    # what the block resolution then checks against the resolved dimension.
    assert isinstance(prior.transform, CholeskyCorrTransform)
    assert prior.transform.K == 3
    assert est._matrix_blocks["R_corr"].K == 3


def test_matrix_key_in_estimated_params_expands_to_member_names(post82):
    est = Estimator(
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        ss_seed=post82["steady"],
        estimated_params={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=3, random_state=None),
                transform=CholeskyCorrTransform(K=3),
            )
        },
    )

    assert est.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"]
    assert list(est.priors.keys()) == ["R_corr"]


def test_a_mapping_carries_the_priors_of_the_names_it_lists(post82):
    est = Estimator(
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        ss_seed=post82["steady"],
        estimated_params={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=3, random_state=None),
                transform=CholeskyCorrTransform(K=3),
            ),
            "meas_infl": make_prior("log_normal", {"mean": 0.0, "std": 1.0}, "log"),
        },
    )

    assert est.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"] + [
        "meas_infl"
    ]
    assert list(est.priors.keys()) == ["R_corr", "meas_infl"]


def test_none_estimates_every_calibrated_parameter_without_priors(
    post82, post82_estimator
):
    est = post82_estimator(estimated_params=None)

    assert est.estimated_params == list(post82["compiled"].calib_params)
    assert est.priors is None


def test_matrix_prior_overlap_with_scalar_component_prior_raises(post82):
    with pytest.raises(ValueError, match="meas_rho_ir"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                ),
                "meas_rho_ir": make_prior(
                    "normal", {"mean": 0.0, "std": 1.0}, "identity"
                ),
            },
        )


# Matrix-prior-on-R MCMC on a real model (valid-correlation-support) is covered
# by test_estimator_lkj_integration.test_matrix_prior_on_R_runs_full_mcmc_with_
# real_likelihood; the SimpleNamespace stub version can no longer drive the native
# path and is not re-added here.


def test_sparse_q_block_for_lkj_prior_raises_descriptive_error(
    gali_2015_test_model_path,
):
    # Three shocks, none of whose correlations the config names: the pairs fall
    # back to zero, which is fine until a block prior needs a name per pair.
    model, kalman = ModelParser(gali_2015_test_model_path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    with pytest.raises(ValueError, match="dense correlation block") as excinfo:
        Estimator(
            compiled=compiled,
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params={
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
    est = post82_estimator(estimated_params={"sig_r": prior})
    out = est.mcmc(
        n_draws=20,
        burn_in=10,
        thin=1,
        random_state=123,
        adapt=False,
    )
    assert np.all(out.samples[:, 0] > 0.0)


def test_loglik_is_unchanged_by_the_parameter_transform(post82_estimator):
    """The transform sits on the likelihood path, and a round trip through it
    has to be faithful: one parameter value scores the same whether or not a
    transform stands between it and the filter. Only the transform differs
    between the two estimators, since the likelihood is evaluated with priors
    off."""
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    transformed = post82_estimator(estimated_params={"psi_pi": prior})
    identity = post82_estimator(estimated_params=("psi_pi",))

    at = {"psi_pi": 2.0}
    assert transformed.loglik(at) == pytest.approx(identity.loglik(at), rel=1e-9)


def test_estimator_constructor_and_lkj_prior_validation_error_branches(post82):
    with pytest.raises(ValueError, match="is not found in the compiled model"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params=["ghost"],
        )

    with pytest.raises(ValueError, match="Estimate the block or its members"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                ),
                "meas_rho_ir": make_prior(
                    "normal", {"mean": 0.0, "std": 1.0}, "identity"
                ),
            },
        )

    with pytest.raises(ValueError, match="CholeskyCorrTransform"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={
                # Support-valid but not a CholeskyCorrTransform: exercises the
                # estimator's LKJ-transform check (AffineLogit(-1,1) matches the
                # LKJChol (-1, 1) support so Prior construction itself succeeds).
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=AffineLogitTransform(low=-1.0, high=1.0),
                )
            },
        )

    with pytest.raises(ValueError, match="matching K between"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                )
            },
        )

    with pytest.raises(ValueError, match="requires a LKJChol distribution"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={
                "R_corr": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")
            },
        )


def test_resolve_r_and_effective_observables_error_paths(post82, compiled_test):
    # A model with no `kalman:` block parses to no config at all. Without one
    # there is nothing to build R from, so the estimator fails fast at
    # construction rather than lazily when the R block is resolved. P0 has a
    # default and is not part of the demand.
    assert compiled_test.kalman is None
    with pytest.raises(ValueError, match="R must be provided"):
        Estimator(
            compiled=compiled_test,
            y=np.zeros((3, 2), dtype=np.float64),
            estimated_params=["beta"],
        )

    est_missing_meta = Estimator(
        compiled=_with_filter_prep(
            SimpleNamespace(
                config=SimpleNamespace(
                    calibration=SimpleNamespace(
                        parameters=SymbolGetterDict({"a": float64(0.0)})
                    )
                ),
                calib_params=["a"],
                # What `kalman: {R: {}}` parses to: a config that exists and
                # declares no R, so there are no names to resolve a block from.
                # A numeric R with no maps is not a state the parser can reach,
                # since R is only ever built from the named maps.
                kalman=SimpleNamespace(
                    y_names=["y"],
                    R=None,
                    R_std_param_map=None,
                    R_corr_param_map=PairGetterDict({}),
                ),
                observable_names=["y"],
            )
        ),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="parser-generated R std/correlation metadata"):
        _resolve_R(
            est_missing_meta.kalman, est_missing_meta._prepared_filter.observables
        )

    # Unknown observables are now rejected at construction by the filter prep.
    with pytest.raises(ValueError, match="Unknown observables"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=["ghost"],
            estimated_params=["psi_pi"],
            ss_seed=post82["steady"],
        )


def test_forward_map_rejects_a_mismatched_parameter_vector(post82, role_context):
    ctx = role_context(
        post82["compiled"],
        (3, 3),
        {"psi_pi": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")},
    )

    with pytest.raises(ValueError, match="does not match the 1 estimated parameters"):
        theta_from_params(
            np.array([1.0, 2.0], dtype=np.float64),
            ctx.est.param_names,
            ctx.est._matrix_blocks,
            ctx.transforms,
        )


def test_evaluators_reject_a_bad_parameter_mapping(post82_estimator):
    est = post82_estimator(estimated_params=("psi_pi",))

    with pytest.raises(TypeError, match="must be a mapping"):
        est.loglik(np.array([2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="not estimated"):
        est.loglik({"rho_r": 0.8})


def test_mcmc_validation_branches(post82, post82_estimator):
    with pytest.raises(ValueError, match="requires priors"):
        post82_estimator().mcmc(n_draws=1)

    # An empty target set never reaches a routine: the constructor is where it
    # is refused, so mcmc's own emptiness guard is no longer reachable.
    with pytest.raises(ValueError, match="at least one parameter"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params=[],
        )


def test_resolve_q_reports_the_missing_pair_of_a_sparse_block():
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    calibration = SimpleNamespace(
        parameters=SymbolGetterDict({"sig1": float64(1.0), "sig2": float64(1.0)}),
        shock_std=SymbolGetterDict({e1: "sig1", e2: "sig2"}),
        shock_corr=PairGetterDict({frozenset((e1, e2)): None}),
    )
    compiled = _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(calibration=calibration, shocks=[e1, e2]),
            calib_params=["sig1", "sig2"],
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
    block = _resolve_Q(est.compiled)
    present = {(int(r), int(c)) for r, c in block.positions}
    missing = [
        (block.labels[row], block.labels[col])
        for row in range(1, block.K)
        for col in range(row)
        if (row, col) not in present
    ]
    assert missing == [("e2", "e1")]


def _stub_compiled_sharing_a_corr_name():
    """One parameter name serving both an R pair and a Q pair. The parser keeps
    correlation names unique within a matrix but not across the two."""
    compiled = _stub_compiled_with_dense_r_block()
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    compiled.config.calibration.shock_std = SymbolGetterDict(
        {e1: "meas_a", e2: "meas_b"}
    )
    compiled.config.calibration.shock_corr = PairGetterDict(
        {frozenset((e1, e2)): "meas_rho_ab"}
    )
    compiled.config.shocks = [e1, e2]
    compiled.shock_names = ("e1", "e2")
    return compiled


def _lkj(K: int) -> Prior:
    return Prior(
        dist=LKJChol(eta=2.0, K=K, random_state=None),
        transform=CholeskyCorrTransform(K=K),
    )


def test_reserved_block_key_rejects_a_one_dimensional_matrix(nk_zlb_test_model_path):
    # One observable means a named R std and no pair to correlate, so naming the
    # reserved key asks for a block the matrix has no room for.
    model, kalman = ModelParser(nk_zlb_test_model_path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    with pytest.raises(ValueError, match="dimension at least 2"):
        Estimator(
            compiled=compiled,
            y=np.zeros((4, 1), dtype=np.float64),
            estimated_params=["R_corr"],
        )


def _stub_compiled_with_a_corr_name_in_both_matrices():
    """``rho`` names a Q pair and an R pair, and R stays too sparse to promote.
    ``rho2`` is a nonzero Q sibling over the shared shock."""
    e1, e2, e3 = Symbol("e1"), Symbol("e2"), Symbol("e3")
    params = SymbolGetterDict(
        {
            name: float64(value)
            for name, value in (
                ("rho", 0.1),
                ("rho2", 0.3),
                ("s1", 1.0),
                ("s2", 1.0),
                ("s3", 1.0),
                ("ma", 1.0),
                ("mb", 1.0),
                ("mc", 1.0),
            )
        }
    )
    calibration = SimpleNamespace(
        parameters=params,
        shock_std=SymbolGetterDict({e1: "s1", e2: "s2", e3: "s3"}),
        shock_corr=PairGetterDict(
            {
                frozenset((e1, e2)): "rho",
                frozenset((e1, e3)): "rho2",
                frozenset((e2, e3)): None,
            }
        ),
    )
    return _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(calibration=calibration, shocks=[e1, e2, e3]),
            calib_params=[str(name) for name in params],
            kalman=SimpleNamespace(
                y_names=["A", "B", "C"],
                R_std_param_map=SymbolGetterDict({"A": "ma", "B": "mb", "C": "mc"}),
                R_corr_param_map=PairGetterDict(
                    {
                        frozenset(("A", "B")): "rho",
                        frozenset(("A", "C")): None,
                        frozenset(("B", "C")): None,
                    }
                ),
            ),
            observable_names=["A", "B", "C"],
            shock_names=("e1", "e2", "e3"),
        )
    )


def test_a_corr_name_in_both_matrices_is_checked_against_each():
    """A name unique within each matrix can still serve both. Keyed by name, the
    second matrix read would be the only one left to check against."""
    with pytest.raises(ValueError, match="couples the same variable 'e1' in 'Q'"):
        Estimator(
            compiled=_stub_compiled_with_a_corr_name_in_both_matrices(),
            y=np.zeros((4, 3), dtype=np.float64),
            estimated_params=["rho"],
        )


def test_reserved_block_keys_reject_a_shared_member():
    with pytest.raises(ValueError, match="cannot share member parameters"):
        Estimator(
            compiled=_stub_compiled_sharing_a_corr_name(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params={"R_corr": _lkj(2), "Q_corr": _lkj(2)},
        )


def test_lkj_prior_k_must_match_the_resolved_block(post82):
    # POST82's R resolves to K=3, so the mismatch is a K=2 prior.
    with pytest.raises(ValueError, match="has K=2"):
        Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            ss_seed=post82["steady"],
            estimated_params={"R_corr": _lkj(2)},
        )


def test_matrix_resolution_rejects_a_reused_correlation_name():
    """One name on two pairs of the same matrix. The parser rejects this, so the
    resolver is only reached directly."""
    with pytest.raises(ValueError, match="is reused"):
        _resolve_R(
            SimpleNamespace(
                R_std_param_map=SymbolGetterDict(
                    {"A": "sig_a", "B": "sig_b", "C": "sig_c"}
                ),
                R_corr_param_map=PairGetterDict(
                    {
                        frozenset(("A", "B")): "rho",
                        frozenset(("A", "C")): "rho",
                        frozenset(("B", "C")): None,
                    }
                ),
            ),
            ["A", "B", "C"],
        )


def test_mle_std_member_without_prior_gets_a_positivity_box(post82, role_context):
    ctx = role_context(post82["compiled"], (4, 3), ["meas_infl"])
    # A variance estimated prior-free is positivity-constrained by role, and
    # with no transform to carry that, the box is what holds the floor.
    assert ctx.codes["meas_infl"] is TransformCode.IDENTITY
    assert ctx.box["meas_infl"] == (_STD_FLOOR, np.inf)


def test_mle_isolated_scalar_corr_without_prior_gets_an_interval_box(role_context):
    ctx = role_context(_stub_compiled_with_sparse_q_block(), (3, 1), ["rho12"])
    # rho12 is the sole named Q correlation (e1, e2): sparse and isolated, so it
    # stays a standalone scalar rather than folding into a block, and its role
    # interval is held by the box.
    assert "Q_corr" not in ctx.est._matrix_blocks
    assert ctx.codes["rho12"] is TransformCode.IDENTITY
    assert ctx.box["rho12"] == (-_CORR_LIMIT, _CORR_LIMIT)


def test_a_prior_confines_through_its_transform_and_not_through_the_box(
    post82_estimator,
):
    """With a prior the region is the transform's, so the box stays open. POST82
    names three Q correlations but calibrates two at zero, which is what leaves
    ``rho_gz`` estimable as a standalone scalar."""
    est = post82_estimator(estimated_params={"rho_gz": _BOUNDED_CORR_PRIOR})
    _, _, lo, hi = est._build_call_context(priored=True)

    assert (float(lo[0]), float(hi[0])) == (-np.inf, np.inf)


def test_bounds_tighter_than_the_transform_bind_in_parameter_space(post82_estimator):
    """A bound is given in parameter space and has to be mapped to reach the
    optimizer. Read as a theta box instead, the same pair would pin the
    parameter an order of magnitude short of where it was asked to stop.
    """
    est = post82_estimator(estimated_params={"rho_gz": _BOUNDED_CORR_PRIOR})
    low, high = -0.2, 0.1
    transform = AffineLogitTransform(low=-1.0, high=1.0)

    _, _, lo, hi = est._build_call_context(priored=True, bounds={"rho_gz": (low, high)})
    assert float(lo[0]) == pytest.approx(float(transform.forward(float64(low))))
    assert float(hi[0]) == pytest.approx(float(transform.forward(float64(high))))

    free = est.map(cov=False)
    bounded = est.map(cov=False, bounds={"rho_gz": (low, high)})

    # The unbounded mode has to sit outside, or the bound proves nothing.
    assert float(free.theta["rho_gz"]) < low
    assert float(bounded.theta["rho_gz"]) == pytest.approx(low, abs=1e-9)


def test_spd_std_member_warns_on_conflicting_prior_transform(post82, role_context):
    # An Identity-transform prior on a variance would map onto R, not (0, inf),
    # so the role default is substituted, and the table build that resolves the
    # transform is what announces it.
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    with pytest.warns(UserWarning, match="replaced with `LogTransform`"):
        ctx = role_context(post82["compiled"], (4, 3), {"meas_infl": prior})
    assert ctx.codes["meas_infl"] is TransformCode.LOG


def test_spd_member_without_prior_warns_nothing(post82, role_context):
    # No prior means no density for a role transform to disagree with, and the
    # prior-free leg carries no transform to substitute, so nothing is said.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx = role_context(post82["compiled"], (4, 3), ["meas_infl"])
    assert not [w for w in caught if "replaced with" in str(w.message)]
    assert ctx.codes["meas_infl"] is TransformCode.IDENTITY


@pytest.mark.parametrize("include_logjac", [False, True])
def test_logpost_decomposes_into_loglik_and_logprior(post82_estimator, include_logjac):
    """The three objectives answer consistently under either convention: the
    jacobian is the prior's to carry, so it moves both sides together."""
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = post82_estimator(estimated_params={"psi_pi": prior})
    at = {"psi_pi": 2.0}

    assert est.logpost(at, include_logjac) == pytest.approx(
        est.loglik(at) + est.logprior(at, include_logjac), rel=1e-12
    )


@pytest.mark.parametrize("estimated", [["psi_pi"], ["psi_pi", "rho_r"]])
def test_mcmc_adaptation_runs_for_scalar_and_vector(post82_estimator, estimated):
    # The native running covariance has no d==1 special case (a 1x1 covariance
    # subsumes it), so adaptation must run cleanly for both scalar and vector theta.
    priors = {
        "psi_pi": _normal_prior(2.0, 0.5),
        "rho_r": _normal_prior(0.8, 0.1),
    }
    est = post82_estimator(estimated_params={n: priors[n] for n in estimated})
    out = est.mcmc(
        n_draws=10,
        burn_in=10,
        thin=1,
        random_state=123,
        adapt=True,
        adapt_start=0,
    )
    assert out.samples.shape == (10, len(estimated))
