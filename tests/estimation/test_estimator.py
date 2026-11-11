# type: ignore
import warnings
from types import SimpleNamespace

import yaml

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE import parse_config, compile_model
from SymbolicDSGE.core.model_parser import ModelParser
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator, make_prior
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    CholeskyCorrTransform,
)
from SymbolicDSGE.core.config import PairGetterDict
from SymbolicDSGE.estimation.prior_program import TransformCode, build_prior_tables
from SymbolicDSGE.estimation.resolvers import (
    _CORR_LIMIT,
    _STD_FLOOR,
    _build_matrix_resolution,
    _resolve_Q,
    active_Q,
    active_R,
    resolve_bounds,
    theta_from_params,
)
from SymbolicDSGE.ui.estimation import estimate_and_solve

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


@pytest.fixture
def compiled_variant():
    """Compile a fixture model with parts of its calibration replaced.

    These cases need a calibration shape no fixture model carries: one
    correlation name serving both Q and R, or a partially named correlation
    block. The parser accepts each of them; it is the estimator that rejects
    them, which is what leaves the shape unreachable through a stored model.
    """

    def build(base, *, parameters=None, shock_corr=None, measurement_noise=None):
        data = yaml.safe_load(base.read_text(encoding="utf-8"))
        calib = data["calibration"]
        if parameters:
            calib["parameters"].update(parameters)
        if shock_corr is not None:
            calib["shocks"]["corr"] = shock_corr
        if measurement_noise is not None:
            calib["measurement_noise"] = measurement_noise

        model = ModelParser.from_string(
            yaml.safe_dump(data, sort_keys=False), "<variant>"
        ).get()
        return model, compile_model(model)

    return build


@pytest.fixture
def role_context():
    """Packed transform codes and the theta box for one estimator.

    ``estim_context`` also resolves the steady-state seed, the param map and
    both covariance specs, none of which these cases assert, so this runs only
    the two steps that decide a slot's transform and its box. The role merge
    mirrors the one ``estim_context`` does inline. ``observables`` narrows the
    active set, which is what sizes a resolved R block.
    """

    def _build(config, compiled, y_shape, estimated_params, observables=None, R=None):
        est = Estimator(
            config=config,
            compiled=compiled,
            y=np.zeros(y_shape, dtype=np.float64),
            observables=observables,
            estimated_params=estimated_params,
            R=R,
        )
        observables = est._prepared_filter.observables
        tables, transforms = build_prior_tables(
            param_index=est._param_index,
            priors=est.priors,
            matrix_blocks=est._matrix_blocks,
            config=est.config,
            observables=observables,
        )
        roles = None
        if est.priors is None:
            std_q, corr_q = active_Q(est.param_names, est.config)
            std_r, corr_r = active_R(est.param_names, est.config, observables)
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


def test_to_spec_captures_construction_state(
    compiled_post82, y_post82, post82_estimator
):
    from SymbolicDSGE.estimation.spec import EstimatorSpec

    est = post82_estimator(estimated_params=["psi_pi"])

    spec = est.to_spec()

    assert isinstance(spec, EstimatorSpec)
    assert spec.params["estimated_params"] == ["psi_pi"]
    assert spec.params["observables"] == list(compiled_post82.observable_names)
    assert spec.params["filter_mode"] == "linear"
    assert spec.params["priors"] is None
    assert spec.y == y_post82.to_numpy().tolist()
    # the run is not construction state, so nothing about a method is here
    assert "method" not in spec.params


def test_filter_params_survive_the_spec_round_trip(
    solved_post82, compiled_post82, y_post82
):
    from SymbolicDSGE.estimation import Estimator
    from SymbolicDSGE.filters.gaussian import LinearKalman

    est = Estimator(
        config=solved_post82.config,
        compiled=compiled_post82,
        y=y_post82,
        observables=compiled_post82.observable_names,
        filter=LinearKalman(jitter=1e-8, symmetrize=True, joseph_cov=True),
        estimated_params=["psi_pi"],
        ss_seed=solved_post82.policy.steady_state,
    )

    spec = est.to_spec()

    assert spec.params["filter_mode"] == "linear"
    assert spec.params["filter_params"] == {
        "jitter": 1e-8,
        "symmetrize": True,
        "joseph_cov": True,
    }

    rebuilt = Estimator.from_spec(
        spec, config=solved_post82.config, compiled=compiled_post82
    )

    assert spec.params["filter_params"] == rebuilt.to_spec().params["filter_params"]


def test_to_spec_reverses_live_scalar_priors(post82_estimator):
    from SymbolicDSGE.bayesian.priors import make_prior

    est = post82_estimator(
        estimated_params={
            "psi_pi": make_prior(
                distribution="normal",
                parameters={"mean": 0.0, "std": 1.0},
                transform="identity",
                transform_kwargs={},
            )
        }
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


def test_map_without_priors_raises(post82_estimator):
    est = post82_estimator()
    with pytest.raises(ValueError, match="requires priors"):
        est.map()


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


def test_forward_map_applies_the_prior_transform(solved_post82, role_context):
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )

    config = solved_post82.config
    compiled = solved_post82.compiled
    ctx = role_context(config, compiled, (3, 3), {"psi_pi": prior})

    theta = theta_from_params(
        np.array([np.e], dtype=np.float64),
        ctx.est.param_names,
        ctx.est._matrix_blocks,
        ctx.transforms,
    )
    assert np.allclose(theta[0], 1.0)


def test_matrix_prior_on_R_reparameterizes_pairwise_correlation_block(
    solved_post82, role_context
):
    # Two observables leave R one pair, where the CPC map is exactly tanh and the
    # expected theta has a closed form. The full three-observable block is
    # covered in test_estimator_lkj_integration.
    prior = Prior(
        dist=LKJChol(eta=2.0, K=2, random_state=None),
        transform=CholeskyCorrTransform(K=2),
    )
    config = solved_post82.config
    compiled = solved_post82.compiled
    ctx = role_context(
        config, compiled, (4, 2), {"R_corr": prior}, observables=["Infl", "Rate"]
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


def test_matrix_prior_created_via_make_prior_uses_cholesky_corr_transform(
    post82_estimator,
):
    prior = make_prior(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 3},
        transform="cholesky_corr",
    )
    est = post82_estimator(estimated_params={"R_corr": prior})

    # make_prior reconciles the distribution's K with the transform's, which is
    # what the block resolution then checks against the resolved dimension.
    assert isinstance(prior.transform, CholeskyCorrTransform)
    assert prior.transform.K == 3
    assert est._matrix_blocks["R_corr"].K == 3


def test_matrix_key_in_estimated_params_expands_to_member_names(post82_estimator):
    est = post82_estimator(
        estimated_params={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=3, random_state=None),
                transform=CholeskyCorrTransform(K=3),
            )
        }
    )

    assert est.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"]
    assert list(est.priors.keys()) == ["R_corr"]


def test_a_mapping_carries_the_priors_of_the_names_it_lists(post82_estimator):
    est = post82_estimator(
        estimated_params={
            "R_corr": Prior(
                dist=LKJChol(eta=2.0, K=3, random_state=None),
                transform=CholeskyCorrTransform(K=3),
            ),
            "meas_infl": make_prior("log_normal", {"mean": 0.0, "std": 1.0}, "log"),
        }
    )

    assert est.param_names == ["meas_rho_gi", "meas_rho_gr", "meas_rho_ir"] + [
        "meas_infl"
    ]
    assert list(est.priors.keys()) == ["R_corr", "meas_infl"]


def test_none_estimates_every_calibrated_parameter_without_priors(
    compiled_post82, post82_estimator
):
    est = post82_estimator(estimated_params=None)

    assert est.estimated_params == list(compiled_post82.calib_params)
    assert est.priors is None


def test_matrix_prior_overlap_with_scalar_component_prior_raises(post82_estimator):
    with pytest.raises(ValueError, match="meas_rho_ir"):
        post82_estimator(
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                ),
                "meas_rho_ir": make_prior(
                    "normal", {"mean": 0.0, "std": 1.0}, "identity"
                ),
            }
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
    model = parse_config(gali_2015_test_model_path)
    compiled = compile_model(model)
    with pytest.raises(ValueError, match="dense correlation block") as excinfo:
        Estimator(
            config=model,
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


def test_estimator_constructor_and_lkj_prior_validation_error_branches(
    post82_estimator,
):
    with pytest.raises(ValueError, match="is not found in the compiled model"):
        post82_estimator(estimated_params=["ghost"])

    with pytest.raises(ValueError, match="Estimate the block or its members"):
        post82_estimator(
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                ),
                "meas_rho_ir": make_prior(
                    "normal", {"mean": 0.0, "std": 1.0}, "identity"
                ),
            }
        )

    with pytest.raises(ValueError, match="CholeskyCorrTransform"):
        post82_estimator(
            estimated_params={
                # Support-valid but not a CholeskyCorrTransform: exercises the
                # estimator's LKJ-transform check (AffineLogit(-1,1) matches the
                # LKJChol (-1, 1) support so Prior construction itself succeeds).
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=3, random_state=None),
                    transform=AffineLogitTransform(low=-1.0, high=1.0),
                )
            }
        )

    with pytest.raises(ValueError, match="matching K between"):
        post82_estimator(
            estimated_params={
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                )
            }
        )

    with pytest.raises(ValueError, match="requires a LKJChol distribution"):
        post82_estimator(
            estimated_params={
                "R_corr": make_prior("normal", {"mean": 0.0, "std": 1.0}, "identity")
            }
        )


def test_resolve_r_and_effective_observables_error_paths(
    solved_post82, y_post82, solved_test
):
    # A model with no `calibration.measurement_noise` has nothing to build R
    # from, so the estimator fails fast at construction rather than lazily when
    # the R block is resolved. P0 has a default and is not part of the demand.
    assert solved_test.config.calibration.measurement_noise is None
    with pytest.raises(ValueError, match="R must be provided"):
        Estimator(
            config=solved_test.config,
            compiled=solved_test.compiled,
            y=np.zeros((3, 2), dtype=np.float64),
            estimated_params=["beta"],
        )

    # Unknown observables are now rejected at construction by the filter prep.
    with pytest.raises(ValueError, match="Unknown observables"):
        Estimator(
            config=solved_post82.config,
            compiled=solved_post82.compiled,
            y=y_post82,
            observables=["ghost"],
            estimated_params=["psi_pi"],
            ss_seed=solved_post82.policy.steady_state,
        )


def test_forward_map_rejects_a_mismatched_parameter_vector(solved_post82, role_context):
    config = solved_post82.config
    compiled = solved_post82.compiled
    ctx = role_context(
        config,
        compiled,
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


def test_resolve_q_reports_the_missing_pair_of_a_sparse_block(parsed_test):
    # ``test.yaml`` names two shocks and correlates neither, so the one pair the
    # block expects is the one the resolution has to report as absent.
    block = _resolve_Q(parsed_test)

    present = {(int(r), int(c)) for r, c in block.positions}
    missing = [
        (block.labels[row], block.labels[col])
        for row in range(1, block.K)
        for col in range(row)
        if (row, col) not in present
    ]
    assert block.labels == ["e_u", "e_v"]
    assert missing == [("e_v", "e_u")]


def _lkj(K: int) -> Prior:
    return Prior(
        dist=LKJChol(eta=2.0, K=K, random_state=None),
        transform=CholeskyCorrTransform(K=K),
    )


def test_reserved_block_key_rejects_a_one_dimensional_matrix(nk_zlb_test_model_path):
    # One observable means a named R std and no pair to correlate, so naming the
    # reserved key asks for a block the matrix has no room for.
    model = parse_config(nk_zlb_test_model_path)
    compiled = compile_model(model)
    with pytest.raises(ValueError, match="dimension at least 2"):
        Estimator(
            config=model,
            compiled=compiled,
            y=np.zeros((4, 1), dtype=np.float64),
            estimated_params=["R_corr"],
        )


def test_a_corr_name_in_both_matrices_is_checked_against_each(
    compiled_variant, rbc_multishock_test_model_path
):
    """A name unique within each matrix can still serve both. Keyed by name, the
    second matrix read would be the only one left to check against.

    ``rho`` names one Q pair and one R pair, and R keeps only that one pair so
    it stays too sparse to promote into a block. ``rho2`` is the nonzero Q
    sibling over the shared shock.
    """
    config, compiled = compiled_variant(
        rbc_multishock_test_model_path,
        parameters={"rho": 0.1, "rho2": 0.3},
        shock_corr={"e_z, e_d": "rho", "e_z, e_g": "rho2"},
        measurement_noise={
            "std": {"c_obs": "meas_c", "k_obs": "meas_k", "g_obs": "meas_g"},
            "corr": {"c_obs, k_obs": "rho"},
        },
    )

    with pytest.raises(ValueError, match="couples the same variable 'e_z' in 'Q'"):
        Estimator(
            config=config,
            compiled=compiled,
            y=np.zeros((4, 3), dtype=np.float64),
            estimated_params=["rho"],
        )


def test_reserved_block_keys_reject_a_shared_member(compiled_variant, test_model_path):
    # Two shocks and two observables leave each matrix one pair, so one name on
    # both makes the two blocks complete and overlapping at once.
    config, compiled = compiled_variant(
        test_model_path,
        parameters={"rho_shared": 0.2, "meas_infl": 0.3, "meas_rate": 0.4},
        shock_corr={"e_u, e_v": "rho_shared"},
        measurement_noise={
            "std": {"Infl": "meas_infl", "Rate": "meas_rate"},
            "corr": {"Infl, Rate": "rho_shared"},
        },
    )

    with pytest.raises(ValueError, match="cannot share member parameters"):
        Estimator(
            config=config,
            compiled=compiled,
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params={"R_corr": _lkj(2), "Q_corr": _lkj(2)},
        )


def test_lkj_prior_k_must_match_the_resolved_block(post82_estimator):
    # POST82's R resolves to K=3, so the mismatch is a K=2 prior.
    with pytest.raises(ValueError, match="has K=2"):
        post82_estimator(estimated_params={"R_corr": _lkj(2)})


def test_matrix_resolution_rejects_a_reused_correlation_name():
    """One name on two pairs of the same matrix.

    ``_parse_measurement_noise`` rejects a reused correlation name, so no
    config reaches this; the resolution is driven on its own argument instead.
    """
    with pytest.raises(ValueError, match="is reused"):
        _build_matrix_resolution(
            key="R_corr",
            labels=["A", "B", "C"],
            corr_param_map=PairGetterDict(
                {
                    frozenset(("A", "B")): "rho",
                    frozenset(("A", "C")): "rho",
                    frozenset(("B", "C")): None,
                }
            ),
        )


def test_mle_std_member_without_prior_gets_a_positivity_box(
    solved_post82, role_context
):
    config = solved_post82.config
    compiled = solved_post82.compiled
    ctx = role_context(config, compiled, (4, 3), ["meas_infl"])
    # A variance estimated prior-free is positivity-constrained by role, and
    # with no transform to carry that, the box is what holds the floor.
    assert ctx.codes["meas_infl"] is TransformCode.IDENTITY
    assert ctx.box["meas_infl"] == (_STD_FLOOR, np.inf)


def test_mle_isolated_scalar_corr_without_prior_gets_an_interval_box(
    role_context, compiled_variant, rbc_multishock_test_model_path
):
    # corr_zd is left as the sole named Q correlation of three pairs: sparse and
    # isolated, so it stays a standalone scalar rather than folding into a
    # block, and its role interval is held by the box.
    config, compiled = compiled_variant(
        rbc_multishock_test_model_path, shock_corr={"e_z, e_d": "corr_zd"}
    )
    ctx = role_context(config, compiled, (3, 3), ["corr_zd"])

    assert "Q_corr" not in ctx.est._matrix_blocks
    assert ctx.codes["corr_zd"] is TransformCode.IDENTITY
    assert ctx.box["corr_zd"] == (-_CORR_LIMIT, _CORR_LIMIT)


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


def test_spd_std_member_warns_on_conflicting_prior_transform(
    solved_post82, role_context
):
    # An Identity-transform prior on a variance would map onto R, not (0, inf),
    # so the role default is substituted, and the table build that resolves the
    # transform is what announces it.
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    config = solved_post82.config
    compiled = solved_post82.compiled
    with pytest.warns(UserWarning, match="replaced with `LogTransform`"):
        ctx = role_context(config, compiled, (4, 3), {"meas_infl": prior})
    assert ctx.codes["meas_infl"] is TransformCode.LOG


def test_spd_member_without_prior_warns_nothing(solved_post82, role_context):
    # No prior means no density for a role transform to disagree with, and the
    # prior-free leg carries no transform to substitute, so nothing is said.
    config = solved_post82.config
    compiled = solved_post82.compiled
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx = role_context(config, compiled, (4, 3), ["meas_infl"])
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


def test_estimate_and_solve_leaves_the_given_config_alone(
    solved_post82, post82_estimator
):
    """The estimates reach the solve through a copy, not through the caller's
    config."""
    config = solved_post82.config

    def calibration():
        return {str(k): float(v) for k, v in config.calibration.parameters.items()}

    before = calibration()

    result, solved = estimate_and_solve(
        est=post82_estimator(estimated_params=("psi_pi", "rho_r")),
        ss_seed=solved_post82.policy.steady_state,
        theta0=_POST82_START,
        bounds=_POST82_BOUNDS,
        maxiter=10,
    )

    assert calibration() == before
    assert solved.config is not config
    for name, value in result.theta.items():
        assert solved.config.calibration.parameters[name] == value
        assert before[name] != value
