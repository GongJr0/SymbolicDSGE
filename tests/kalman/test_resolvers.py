# type: ignore
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import SymbolicDSGE.kalman.resolvers as resolvers
from SymbolicDSGE.core import DSGESolver
from SymbolicDSGE.core.resolvers import shock_covariance
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.resolvers import (
    FilterMode,
    LinearRunArgs,
    resolve_extended_args,
    resolve_linear_args,
    resolve_unscented_args,
)

FLOAT = np.float64

#: Canonical observable order of ``rbc_multishock_second_order``. Its canonical
#: variable order is ``[k, z, d, g, c]`` against a declaration order of
#: ``[c, k, z, d, g]``, so every reordering the resolvers do is visible here.
OBS = ("c_obs", "k_obs", "g_obs")

#: The fixture's ``calibration.measurement_noise``: standard deviations 1e-3,
#: 2e-3, and 4e-3 over ``OBS``, correlated 0.25 on ``c_obs, k_obs`` alone.
R_FULL = np.array(
    [
        [1e-6, 5e-7, 0.0],
        [5e-7, 4e-6, 0.0],
        [0.0, 0.0, 1.6e-5],
    ],
    dtype=FLOAT,
)


@pytest.fixture(scope="module")
def multishock_order1(solved_multishock) -> SolvedModel:
    """The shared compile solved to first order, for the unscented order gate."""
    return DSGESolver(solved_multishock.config).solve(
        compiled=solved_multishock.compiled, order=1
    )


@pytest.fixture
def y3() -> np.ndarray:
    """Two periods of three observables, every entry distinct.

    Already canonical when the observables arrive in canonical order, permuted
    otherwise, which is what makes the column alignment legible.
    """
    return np.arange(6, dtype=FLOAT).reshape(2, 3)


def test_resolve_linear_args_is_a_complete_run_argument_set(solved_multishock, y3):
    model = solved_multishock

    args = resolve_linear_args(
        model,
        y3,
        # Columns are read in the order the names arrive, then the two are
        # canonicalized together.
        ["g_obs", "c_obs", "k_obs"],
        jitter=0.125,
        symmetrize=True,
        return_shocks=True,
    )

    assert set(args) == set(LinearRunArgs.__annotations__)

    assert np.array_equal(
        args["y"], np.array([[1.0, 2.0, 0.0], [4.0, 5.0, 3.0]], dtype=FLOAT)
    )
    C, d = model._build_C_d_from_obs(OBS)
    assert np.array_equal(args["C"], C)
    assert np.array_equal(args["d"], d)
    assert np.array_equal(args["A"], model.policy.A)
    assert np.array_equal(args["B"], model.policy.B)
    assert np.allclose(args["Q"], shock_covariance(model.config))
    assert np.allclose(args["R"], R_FULL)
    assert np.array_equal(args["steady_state"], model.policy.steady_state)
    # x0 defaults to the steady state, which is a zero deviation.
    assert np.array_equal(args["x0"], np.zeros((model.compiled.n_var,), dtype=FLOAT))
    assert args["jitter"] == pytest.approx(0.125)
    assert args["symmetrize"] is True
    assert args["joseph_cov"] is False
    assert args["return_shocks"] is True


def test_default_p0_is_solved_from_the_policy(solved_multishock, y3):
    """With no P0 supplied the default is the stationary covariance.

    The identity is the fallback for a policy whose stationary covariance does
    not converge, not the default.
    """
    model = solved_multishock
    n_var = model.compiled.n_var

    P0 = resolve_linear_args(model, y3, list(OBS))["P0"]

    assert P0.shape == (n_var, n_var)
    assert np.allclose(P0, P0.T)
    assert np.all(np.diag(P0) > 0.0)
    assert not np.allclose(P0, np.eye(n_var, dtype=FLOAT))


def test_resolve_extended_args_carries_cfunc_addresses_and_skips_c_and_d(
    solved_multishock, y3, monkeypatch
):
    model = solved_multishock

    def bomb(obs_names):
        raise AssertionError("_build_C_d_from_obs should not run in extended mode")

    monkeypatch.setattr(model, "_build_C_d_from_obs", bomb)

    args = resolve_extended_args(model, y3, list(OBS))

    assert "C" not in args
    assert "d" not in args
    assert args["meas_addr"] == int(
        model.compiled.construct_measurement_cfunc(OBS).address
    )
    assert args["jac_addr"] == int(
        model.compiled.construct_measurement_jacobian_cfunc(OBS).address
    )
    assert args["meas_addr"] != args["jac_addr"]

    params = model.config.calibration.parameters
    assert np.array_equal(
        args["calib_params"],
        np.array([params[name] for name in model.compiled.calib_params], dtype=FLOAT),
    )


def test_resolve_unscented_args_embeds_x0_and_defaults_the_sigma_point_weights(
    solved_multishock, y3
):
    model = solved_multishock
    policy = model.policy
    idx = model.compiled.idx
    ss = policy.steady_state

    args = resolve_unscented_args(
        model,
        y3,
        list(OBS),
        # x0 arrives in levels. ``k`` is a state and ``c`` is the control, so
        # the deviation on ``k`` survives the embedding and the one on ``c`` is
        # truncated away with the rest of the control block.
        x0={"k": ss[idx["k"]] + 0.3, "c": ss[idx["c"]] + 0.2},
        jitter=0.25,
        symmetrize=False,
    )

    # Converted to deviations, truncated to the state block and embedded; the
    # second-order block starts at zero.
    expected_z0 = np.zeros((2 * model.compiled.n_state,), dtype=FLOAT)
    expected_z0[idx["k"]] = 0.3
    assert np.allclose(args["z0"], expected_z0)

    assert args["meas_addr"] == int(
        model.compiled.construct_measurement_cfunc(OBS).address
    )
    assert np.array_equal(args["hx"], policy.p)
    assert np.array_equal(args["gx"], policy.f)
    assert np.array_equal(args["bu"], policy.B)
    assert np.array_equal(args["steady_state"], ss)
    assert args["alpha"] == pytest.approx(1.0)
    assert args["beta"] == pytest.approx(2.0)
    assert args["kappa"] == pytest.approx(1.0)
    assert args["jitter"] == pytest.approx(0.25)
    assert args["symmetrize"] is False
    # No joseph_cov or return_shocks: the unscented kernel takes neither.
    assert "joseph_cov" not in args
    assert "return_shocks" not in args


def test_resolve_unscented_args_rejects_a_first_order_policy_and_a_bad_x0(
    solved_multishock, multishock_order1, y3
):
    with pytest.raises(ValueError, match="requires a second order solution"):
        resolve_unscented_args(multishock_order1, y3, list(OBS))

    with pytest.raises(ValueError, match="must be a complete list/array"):
        resolve_unscented_args(
            solved_multishock, y3, list(OBS), x0=np.array([1.0], dtype=FLOAT)
        )


def test_an_explicit_p0_reaches_the_run_arguments(solved_multishock, y3):
    n_var = solved_multishock.compiled.n_var
    P0 = np.diag(np.arange(2.0, 2.0 + n_var, dtype=FLOAT))

    args = resolve_linear_args(solved_multishock, y3, list(OBS), P0=P0)

    assert np.array_equal(args["P0"], P0)


def test_jitter_defaults_to_zero():
    assert resolvers._jitter(0.25) == pytest.approx(0.25)
    assert resolvers._jitter(None) == pytest.approx(0.0)


def test_a_user_r_is_validated_and_passed_through(solved_multishock):
    with pytest.raises(ValueError, match="Provided R matrix has shape"):
        resolvers._validate_user_R(np.eye(2, dtype=FLOAT), ["k_obs"])

    user_R = np.array([[2.5]], dtype=FLOAT)
    assert np.array_equal(resolvers._validate_user_R(user_R, ["k_obs"]), user_R)
    assert np.array_equal(
        resolvers._build_R(solved_multishock, user_R, ["k_obs"]), user_R
    )


def test_build_r_assembles_the_measurement_noise_block_over_the_subset(
    solved_multishock,
):
    """R is assembled from the named std/corr parameters, then subset.

    The std entries are standard deviations, so the diagonal is ``sig**2`` and
    an off-diagonal is ``sig_i * sig_j * corr``. A pair with no named
    correlation stays at zero. Assembly runs over the whole observable set
    before the subset is taken, which is what lets a correlation pair name an
    observable the subset leaves out.
    """
    model = solved_multishock

    assert np.allclose(resolvers._build_R(model, None, OBS), R_FULL)
    assert np.allclose(
        resolvers._build_R(model, None, ["c_obs", "k_obs"]),
        np.array([[1e-6, 5e-7], [5e-7, 4e-6]], dtype=FLOAT),
    )
    assert np.allclose(
        resolvers._build_R(model, None, ["k_obs"]),
        np.array([[4e-6]], dtype=FLOAT),
    )


def test_measurement_noise_is_required_only_for_r(solved_test):
    """R is the one input with no default.

    ``test.yaml`` carries no ``calibration.measurement_noise``, so there is
    nothing to assemble from and R has to be supplied. P0 has a default, so the
    rest of the argument set still resolves.
    """
    model = solved_test
    obs = ["Infl", "Rate"]
    y = np.arange(4.0, dtype=FLOAT).reshape(2, 2)

    assert model.config.calibration.measurement_noise is None

    with pytest.raises(ValueError, match="R must be provided"):
        resolve_linear_args(model, y, obs)

    R = np.diag(np.array([4.0, 9.0], dtype=FLOAT))
    args = resolve_linear_args(model, y, obs, R=R)

    assert np.array_equal(args["R"], R)
    assert args["P0"].shape == (model.compiled.n_var, model.compiled.n_var)


def test_resolve_p0_passes_through_for_linear_and_extended():
    # Explicit runtime P0 values pass through linear and extended paths verbatim.
    P0 = np.diag(np.array([2.0, 6.0, 10.0, 14.0, 18.0], dtype=FLOAT))
    assert np.array_equal(resolvers._resolve_P0(FilterMode.LINEAR, 4, 5, P0), P0)
    assert np.array_equal(resolvers._resolve_P0(FilterMode.EXTENDED, 4, 5, P0), P0)


def test_resolve_p0_embeds_state_block_for_unscented():
    # Unscented embedding carries the supplied first-order state block. The
    # second-order block starts at zero.
    P0 = np.diag(np.array([2.0, 6.0, 10.0, 14.0, 18.0], dtype=FLOAT))
    assert np.array_equal(
        resolvers._resolve_P0(FilterMode.UNSCENTED, 4, 5, P0),
        np.diag(np.array([2.0, 6.0, 10.0, 14.0, 0.0, 0.0, 0.0, 0.0], dtype=FLOAT)),
    )


def test_resolve_observations_aligns_dataframe_and_ndarray_inputs(solved_multishock):
    model = solved_multishock
    canonical = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=FLOAT)

    # A DataFrame is aligned by label, so its own column order is irrelevant.
    df = pd.DataFrame({"g_obs": [2.0, 5.0], "c_obs": [0.0, 3.0], "k_obs": [1.0, 4.0]})
    obs_df, y_df = resolvers._resolve_observations(model, None, df)
    assert obs_df == OBS
    assert np.array_equal(y_df, canonical)

    # An ndarray's columns are read in the order the names arrive.
    obs_arr, y_arr = resolvers._resolve_observations(
        model,
        ["g_obs", "c_obs", "k_obs"],
        np.array([[2.0, 0.0, 1.0], [5.0, 3.0, 4.0]], dtype=FLOAT),
    )
    assert obs_arr == OBS
    assert np.array_equal(y_arr, canonical)

    obs_default, y_default = resolvers._resolve_observations(model, None, canonical)
    assert obs_default == OBS
    assert np.array_equal(y_default, canonical)


@pytest.mark.parametrize(
    ("observables", "y", "match"),
    [
        ([], np.zeros((2, 0), dtype=FLOAT), "Observable list is empty"),
        (
            ["c_obs", "c_obs"],
            np.zeros((2, 2), dtype=FLOAT),
            "Duplicate observables provided",
        ),
        (["Ghost"], np.zeros((2, 1), dtype=FLOAT), "Unknown observables"),
        (
            ["c_obs"],
            pd.DataFrame({"k_obs": [1.0]}),
            "DataFrame is missing observable columns",
        ),
        (["c_obs"], np.zeros((2,), dtype=FLOAT), "Observation data must be 2D"),
        (
            ["c_obs", "k_obs"],
            np.zeros((2, 1), dtype=FLOAT),
            "y has 1 columns but obs list has 2 names",
        ),
        (
            ["c_obs"],
            np.array([[np.nan]], dtype=FLOAT),
            "Observation data contains NaN values",
        ),
    ],
)
def test_resolve_observations_rejects_invalid_inputs(
    solved_multishock, observables, y, match
):
    with pytest.raises(ValueError, match=match):
        resolvers._resolve_observations(solved_multishock, observables, y)


@pytest.fixture(scope="module")
def solved_rbc_order1(solver_rbc_so, compiled_rbc_so) -> SolvedModel:
    """The RBC at first order. The shared bundle solves it to second."""
    return solver_rbc_so.solve(compiled=compiled_rbc_so, order=1)


@pytest.fixture(scope="module")
def levels_rbc_y(solved_rbc_so):
    """Observations around a nonzero consumption steady state.

    POST82 is deviation form, so every other filter test runs where levels and
    gaps coincide and cannot see which one it was handed. Here c_ss is far from
    zero, so the two are distinguishable. Both orders expand about the same
    Newton point, so one panel serves either.
    """
    compiled = solved_rbc_so.compiled
    ss_c = float(solved_rbc_so.policy.steady_state[compiled.idx["c"]])
    rng = np.random.default_rng(20260815)
    return ss_c + rng.normal(0.0, 0.01, size=(12, 1))


def test_solved_model_filter_reports_levels_at_both_orders(
    solved_rbc_order1, solved_rbc_so, levels_rbc_y
):
    """The public filter path is in levels whatever the order.

    A solved model knows its expansion point and hands it to the kernel, so
    both routes report the same units: the linear kernel adds it to the series
    the recursion produces, and the unscented kernel forms levels itself
    because its measurement is evaluated at them.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")

    solved1, y1 = solved_rbc_order1, levels_rbc_y
    lin = solved1.kalman(y=y1, filter_mode="linear", observables=["c_obs"])
    ss1 = np.asarray(solved1.policy.steady_state, dtype=FLOAT)

    assert np.any(ss1 != 0.0)
    # Levels, not gaps: the filtered consumption sits at its steady state, not
    # near zero.
    c = solved1.compiled.idx["c"]
    assert abs(float(np.mean(lin.x_filt[:, c])) - ss1[c]) < 0.5 * abs(ss1[c])

    solved2, y2 = solved_rbc_so, levels_rbc_y
    ukf = solved2.kalman(y=y2, filter_mode="unscented", observables=["c_obs"])

    ss2 = np.asarray(solved2.policy.steady_state, dtype=FLOAT)
    assert abs(float(np.mean(ukf.x_filt[:, c])) - ss2[c]) < 0.5 * abs(ss2[c])


def test_filter_classes_shift_only_the_state_series(solved_rbc_order1, levels_rbc_y):
    """``steady_state`` moves the state series and nothing else.

    The recursion runs in gaps either way, so the expansion point is an offset
    on the reported states. The observation series carry their own constant
    through ``d``, and the likelihood is a function of the innovations, so
    neither moves with it.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    from SymbolicDSGE.kalman.filter import KalmanFilter

    solved, y = solved_rbc_order1, levels_rbc_y
    pol = solved.policy
    C, d = solved._build_C_d_from_obs(["c_obs"])
    Q = np.asarray(shock_covariance(solved.config), dtype=FLOAT)
    n_var = solved.compiled.n_var
    args = dict(
        A=np.real(pol.A),
        B=np.real(pol.B),
        C=C,
        d=d,
        Q=Q,
        R=np.array([[0.01]], dtype=FLOAT),
        y=y,
        x0=np.zeros(n_var, dtype=FLOAT),
        P0=0.1 * np.eye(n_var, dtype=FLOAT),
    )
    ss = np.asarray(pol.steady_state, dtype=FLOAT)
    assert np.any(ss != 0.0)

    gaps = KalmanFilter.run(**args, steady_state=np.zeros(n_var, dtype=FLOAT))
    levels = KalmanFilter.run(**args, steady_state=ss)

    np.testing.assert_allclose(levels.x_pred, gaps.x_pred + ss, rtol=0, atol=0)
    np.testing.assert_allclose(levels.x_filt, gaps.x_filt + ss, rtol=0, atol=0)
    np.testing.assert_allclose(levels.y_pred, gaps.y_pred, rtol=0, atol=0)
    np.testing.assert_allclose(levels.y_filt, gaps.y_filt, rtol=0, atol=0)
    np.testing.assert_allclose(levels.P_filt, gaps.P_filt, rtol=0, atol=0)
    np.testing.assert_allclose(levels.loglik, gaps.loglik, rtol=0, atol=0)
