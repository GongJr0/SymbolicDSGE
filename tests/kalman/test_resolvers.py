# type: ignore
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

import SymbolicDSGE.kalman.resolvers as resolvers
from SymbolicDSGE.kalman.resolvers import (
    FilterMode,
    resolve_extended_args,
    resolve_linear_args,
    resolve_unscented_args,
)

FLOAT = np.float64

E_U = Symbol("e_u")
E_V = Symbol("e_v")
SIG_U = Symbol("sig_u")
SIG_V = Symbol("sig_v")
RHO_UV = Symbol("rho_uv")
MEAS_A = Symbol("meas_a")
MEAS_B = Symbol("meas_b")

STUB_Q = np.array([[0.04, 0.015], [0.015, 0.09]], dtype=FLOAT)
MEAS_ADDR = 111
JAC_ADDR = 222


def _make_stub_model(
    *,
    kalman_config=...,
    params: dict[Symbol | str, float] | None = None,
    order: int = 2,
):
    observable_names = ["ObsA", "ObsB"]
    var_names = ["u", "v", "x"]
    parameters = {
        SIG_U: 0.2,
        SIG_V: 0.3,
        RHO_UV: 0.25,
        MEAS_A: 4.0,
        MEAS_B: 9.0,
    }
    if params is not None:
        parameters.update(params)

    calibration = SimpleNamespace(
        parameters=parameters,
        shock_std={E_U: SIG_U, E_V: SIG_V},
        shock_corr={frozenset({E_U, E_V}): RHO_UV},
        fingerprint=lambda: hash(
            (
                tuple(parameters.keys()),
                tuple(float(v) for v in parameters.values()),
            )
        ),
    )
    config = SimpleNamespace(
        calibration=calibration,
        shocks=[E_U, E_V],
    )
    compiled = SimpleNamespace(
        observable_names=observable_names,
        var_names=var_names,
        n_var=3,
        n_state=2,
        n_ctrl=1,
        n_exog=2,
        calib_params=[SIG_U, SIG_V],
        construct_measurement_cfunc=lambda obs: SimpleNamespace(address=MEAS_ADDR),
        construct_observable_jacobian_cfunc=lambda obs: SimpleNamespace(
            address=JAC_ADDR
        ),
    )

    rows = {
        "ObsA": np.array([1.0, 0.0, 0.0], dtype=FLOAT),
        "ObsB": np.array([0.0, 1.0, 1.0], dtype=FLOAT),
    }
    constants = {"ObsA": 1.0, "ObsB": -1.0}

    def build_measurement(obs_names):
        C = np.vstack([rows[name] for name in obs_names]).astype(FLOAT)
        d = np.array([constants[name] for name in obs_names], dtype=FLOAT)
        return C, d

    if kalman_config is ...:
        kalman_config = SimpleNamespace(
            y_names=["ObsB", "ObsA"],
            R=np.array([[4.0, 0.6], [0.6, 9.0]], dtype=FLOAT),
            jitter=0.125,
            symmetrize=True,
            R_std_param_map=None,
            R_corr_param_map=None,
        )

    model = SimpleNamespace(
        compiled=compiled,
        config=config,
        kalman_config=kalman_config,
        policy=SimpleNamespace(
            order=order,
            A=np.eye(3, dtype=FLOAT),
            B=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=FLOAT),
            p=np.array([[0.8, 0.1], [0.0, 0.7]], dtype=FLOAT),
            f=np.array([[0.2, 0.3]], dtype=FLOAT),
            hxx=np.zeros((2, 2, 2), dtype=FLOAT),
            gxx=np.zeros((1, 2, 2), dtype=FLOAT),
            hxu=np.zeros((2, 2, 2), dtype=FLOAT),
            gxu=np.zeros((1, 2, 2), dtype=FLOAT),
            huu=np.zeros((2, 2, 2), dtype=FLOAT),
            guu=np.zeros((1, 2, 2), dtype=FLOAT),
            hss=np.array([0.01, 0.02], dtype=FLOAT),
            gss=np.array([0.03], dtype=FLOAT),
            steady_state=np.array([1.0, 2.0, 3.0], dtype=FLOAT),
        ),
    )
    model._build_C_d_from_obs = build_measurement
    model._build_Q = lambda: STUB_Q.copy()
    return model


def test_resolve_linear_args_is_a_complete_run_raw_argument_set():
    model = _make_stub_model()
    y = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=FLOAT)

    args = resolve_linear_args(
        model,
        y,
        ["ObsB", "ObsA"],
        jitter=0.125,
        symmetrize=True,
        return_shocks=True,
    )

    # Observables are canonicalized and y's columns follow them.
    assert np.array_equal(args["y"], np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))
    assert np.array_equal(
        args["C"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=FLOAT)
    )
    assert np.array_equal(args["d"], np.array([1.0, -1.0], dtype=FLOAT))
    assert np.array_equal(args["A"], model.policy.A)
    assert np.array_equal(args["B"], model.policy.B)
    assert np.allclose(args["Q"], STUB_Q)
    assert np.array_equal(args["R"], np.array([[4.0, 0.6], [0.6, 9.0]], dtype=FLOAT))
    assert np.array_equal(args["x0"], np.zeros((3,), dtype=FLOAT))
    assert np.array_equal(args["P0"], np.eye(3, dtype=FLOAT))
    assert args["jitter"] == pytest.approx(0.125)
    assert args["symmetrize"] is True
    assert args["joseph_cov"] is True
    assert args["return_shocks"] is True


def test_resolve_extended_args_carries_cfunc_addresses_and_skips_c_and_d():
    model = _make_stub_model()

    def bomb(obs_names):
        raise AssertionError("_build_C_d_from_obs should not run in extended mode")

    model._build_C_d_from_obs = bomb

    args = resolve_extended_args(model, np.array([[1.0], [2.0]], dtype=FLOAT), ["ObsA"])

    assert "C" not in args
    assert "d" not in args
    assert args["meas_addr"] == MEAS_ADDR
    assert args["jac_addr"] == JAC_ADDR
    assert np.array_equal(args["calib_params"], np.array([0.2, 0.3], dtype=FLOAT))
    assert np.array_equal(args["x0"], np.zeros((3,), dtype=FLOAT))


def test_resolve_unscented_args_embeds_x0_and_defaults_the_sigma_point_weights():
    model = _make_stub_model()

    args = resolve_unscented_args(
        model,
        np.array([[1.0], [2.0]], dtype=FLOAT),
        ["ObsA"],
        x0=np.array([0.2, 0.3, 99.0], dtype=FLOAT),
        jitter=0.25,
        symmetrize=False,
    )

    assert args["meas_addr"] == MEAS_ADDR
    # The full-length x0 is truncated to the state block and embedded; the
    # second-order block starts at zero.
    assert np.array_equal(args["z0"], np.array([0.2, 0.3, 0.0, 0.0], dtype=FLOAT))
    assert np.array_equal(args["hx"], model.policy.p)
    assert np.array_equal(args["gx"], model.policy.f)
    assert np.array_equal(args["bu"], model.policy.B)
    assert np.array_equal(args["steady_state"], np.array([1.0, 2.0, 3.0], dtype=FLOAT))
    assert args["alpha"] == pytest.approx(1.0)
    assert args["beta"] == pytest.approx(2.0)
    assert args["kappa"] == pytest.approx(1.0)
    assert args["jitter"] == pytest.approx(0.25)
    assert args["symmetrize"] is False
    # No joseph_cov or return_shocks: the unscented kernel takes neither.
    assert "joseph_cov" not in args
    assert "return_shocks" not in args


def test_resolve_unscented_args_rejects_a_first_order_policy_and_a_bad_x0():
    with pytest.raises(ValueError, match="requires a second order solution"):
        resolve_unscented_args(
            _make_stub_model(order=1),
            np.array([[1.0], [2.0]], dtype=FLOAT),
            ["ObsA"],
        )

    with pytest.raises(ValueError, match="x0 must have length"):
        resolve_unscented_args(
            _make_stub_model(),
            np.array([[1.0], [2.0]], dtype=FLOAT),
            ["ObsA"],
            x0=np.array([1.0], dtype=FLOAT),
        )


def test_an_explicit_p0_reaches_the_run_arguments():
    P0 = np.diag([2.0, 6.0, 10.0]).astype(FLOAT)
    args = resolve_linear_args(
        _make_stub_model(),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FLOAT),
        ["ObsA", "ObsB"],
        P0=P0,
    )

    assert np.array_equal(args["P0"], P0)


def test_jitter_defaults_to_zero():
    assert resolvers._jitter(0.25) == pytest.approx(0.25)
    assert resolvers._jitter(None) == pytest.approx(0.0)


def test_validate_user_r_and_build_constant_r_subset_paths():
    model = _make_stub_model()

    with pytest.raises(ValueError, match="Provided R matrix has shape"):
        resolvers._validate_user_R(np.eye(2, dtype=FLOAT), ["ObsB"])

    user_R = np.array([[2.5]], dtype=FLOAT)
    assert np.array_equal(resolvers._validate_user_R(user_R, ["ObsB"]), user_R)
    assert np.array_equal(resolvers._build_constant_R(model, user_R, ["ObsB"]), user_R)
    assert np.array_equal(
        resolvers._build_constant_R(model, None, ["ObsB"]),
        np.array([[9.0]], dtype=FLOAT),
    )


def test_build_constant_r_assembles_from_param_maps_and_current_calibration():
    # Named R: the constant R is assembled from the CURRENT calibration via the
    # std/corr param maps (make_R), then subset to the included observables.
    # make_R treats the std params as standard deviations, so the diagonal is
    # sig**2 and the off-diagonal is sig_i * sig_j * corr.
    named_conf = SimpleNamespace(
        y_names=["ObsA", "ObsB"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_std_param_map={"ObsA": "meas_a", "ObsB": "meas_b"},
        R_corr_param_map={frozenset({"ObsA", "ObsB"}): "meas_rho"},
    )
    model = _make_stub_model(
        kalman_config=named_conf,
        params={Symbol("meas_rho"): 0.1},
    )
    # std meas_a=4, meas_b=9, corr meas_rho=0.1 ->
    #   [[16, 3.6], [3.6, 81]]; subset to [ObsB] -> [[81]].
    assert np.allclose(
        resolvers._build_constant_R(model, None, ["ObsA", "ObsB"]),
        np.array([[16.0, 3.6], [3.6, 81.0]], dtype=FLOAT),
    )
    assert np.allclose(
        resolvers._build_constant_R(model, None, ["ObsB"]),
        np.array([[81.0]], dtype=FLOAT),
    )

    # A std/corr param that is absent from calibration is a hard error.
    missing_param_conf = SimpleNamespace(
        y_names=["ObsA", "ObsB"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_std_param_map={"ObsA": "not_calibrated", "ObsB": "meas_b"},
        R_corr_param_map={},
    )
    with pytest.raises(KeyError, match="Missing R parameter"):
        resolvers._build_constant_R(
            _make_stub_model(kalman_config=missing_param_conf), None, ["ObsA"]
        )

    # No param maps and no static R -> nothing to build from.
    no_r_conf = SimpleNamespace(
        y_names=["ObsA"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        R_std_param_map=None,
        R_corr_param_map=None,
    )
    with pytest.raises(ValueError, match="Constant R matrix not specified"):
        resolvers._build_constant_R(
            _make_stub_model(kalman_config=no_r_conf), None, ["ObsA"]
        )


def test_kalman_config_is_required_only_for_R():
    # R is the one input with no default. P0 has one, so a model with no Kalman
    # configuration resolves fine as long as R is supplied.
    model = _make_stub_model(kalman_config=None)
    y = np.array([[1.0], [2.0]], dtype=FLOAT)

    with pytest.raises(ValueError, match="R must be provided"):
        resolve_linear_args(model, y, ["ObsA"])

    R = np.array([[4.0]], dtype=FLOAT)
    args = resolve_linear_args(model, y, ["ObsA"], R=R)
    assert np.array_equal(args["R"], R)
    assert np.array_equal(args["P0"], np.eye(3, dtype=FLOAT))


def test_resolve_p0_passes_through_for_linear_and_extended():
    # Explicit runtime P0 values pass through linear and extended paths verbatim.
    P0 = np.diag([2.0, 6.0, 10.0]).astype(FLOAT)
    assert np.array_equal(resolvers._resolve_P0(FilterMode.LINEAR, 2, 3, P0), P0)
    assert np.array_equal(resolvers._resolve_P0(FilterMode.EXTENDED, 2, 3, P0), P0)


def test_resolve_p0_embeds_state_block_for_unscented():
    # Unscented embedding carries the supplied first-order state block. The
    # second-order block starts at zero.
    P0 = np.diag([2.0, 6.0, 10.0]).astype(FLOAT)  # 3 model vars, n_state=2
    assert np.array_equal(
        resolvers._resolve_P0(FilterMode.UNSCENTED, 2, 3, P0),
        np.diag([2.0, 6.0, 0.0, 0.0]).astype(FLOAT),
    )


def test_resolve_observations_aligns_dataframe_and_ndarray_inputs():
    model = _make_stub_model()
    df = pd.DataFrame({"ObsA": [1.0, 2.0], "ObsB": [10.0, 20.0]})

    obs_df, y_df = resolvers._resolve_observations(model, None, df)
    assert obs_df == ("ObsA", "ObsB")
    assert np.array_equal(y_df, np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))

    obs_arr, y_arr = resolvers._resolve_observations(
        model,
        ["ObsB", "ObsA"],
        np.array([[10.0, 1.0], [20.0, 2.0]], dtype=FLOAT),
    )
    assert obs_arr == ("ObsA", "ObsB")
    assert np.array_equal(y_arr, np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))

    obs_default, y_default = resolvers._resolve_observations(
        model,
        None,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FLOAT),
    )
    assert obs_default == ("ObsA", "ObsB")
    assert np.array_equal(y_default, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FLOAT))


@pytest.mark.parametrize(
    ("observables", "y", "match"),
    [
        ([], np.zeros((2, 0), dtype=FLOAT), "Observable list is empty"),
        (
            ["ObsA", "ObsA"],
            np.zeros((2, 2), dtype=FLOAT),
            "Duplicate observables provided",
        ),
        (["Ghost"], np.zeros((2, 1), dtype=FLOAT), "Unknown observables"),
        (
            ["ObsA"],
            pd.DataFrame({"ObsB": [1.0]}),
            "DataFrame is missing observable columns",
        ),
        (["ObsA"], np.zeros((2,), dtype=FLOAT), "Observation data must be 2D"),
        (
            ["ObsA", "ObsB"],
            np.zeros((2, 1), dtype=FLOAT),
            "y has 1 columns but obs list has 2 names",
        ),
        (
            ["ObsA"],
            np.array([[np.nan]], dtype=FLOAT),
            "Observation data contains NaN values",
        ),
    ],
)
def test_resolve_observations_rejects_invalid_inputs(observables, y, match):
    with pytest.raises(ValueError, match=match):
        resolvers._resolve_observations(_make_stub_model(), observables, y)


def _levels_rbc_solved(order: int):
    """The RBC at a nonzero steady state, with a one-observable Kalman config.

    POST82 is deviation form, so every existing filter test runs where levels
    and gaps coincide and cannot see which one it was handed. This fixture is
    the opposite: c_ss is far from zero, so the two are distinguishable.
    """
    from pathlib import Path
    from SymbolicDSGE import ModelParser, DSGESolver
    from SymbolicDSGE.kalman.config import KalmanConfig

    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "models"
        / "rbc_second_order.yaml"
    )
    model, _ = ModelParser(path).get_all()
    kalman = KalmanConfig(R=np.array([[0.01]], dtype=FLOAT))
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=order)
    rng = np.random.default_rng(20260815)
    ss_c = float(solved.policy.steady_state[compiled.idx["c"]])
    y = ss_c + rng.normal(0.0, 0.01, size=(12, 1))
    return solved, y


def test_solved_model_filter_reports_levels_at_both_orders():
    """The public filter path is in levels whatever the order, and says so.

    ``constant`` is what distinguishes the two routes: the linear filter runs in
    gaps and this layer adds the expansion point, so it reports what it added.
    The unscented kernel forms levels itself, because its measurement is
    evaluated at them, so there is nothing left for this layer to add and it
    reports NaN rather than claiming zero.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")

    solved1, y1 = _levels_rbc_solved(1)
    lin = solved1.kalman(y=y1, filter_mode="linear", observables=["c_obs"])
    ss1 = np.asarray(solved1.policy.steady_state, dtype=FLOAT)

    np.testing.assert_allclose(lin.constant, ss1, rtol=0, atol=0)
    assert np.any(ss1 != 0.0)
    # Levels, not gaps: the filtered consumption sits at its steady state, not
    # near zero.
    c = solved1.compiled.idx["c"]
    assert abs(float(np.mean(lin.x_filt[:, c])) - ss1[c]) < 0.5 * abs(ss1[c])

    solved2, y2 = _levels_rbc_solved(2)
    ukf = solved2.kalman(y=y2, filter_mode="unscented", observables=["c_obs"])

    assert np.all(np.isnan(ukf.constant))
    ss2 = np.asarray(solved2.policy.steady_state, dtype=FLOAT)
    assert abs(float(np.mean(ukf.x_filt[:, c])) - ss2[c]) < 0.5 * abs(ss2[c])


def test_filter_classes_leave_the_constant_to_the_caller():
    """Reaching the filter directly keeps the recursion's own units.

    Omitting ``steady_state`` returns gaps with a zero constant; supplying it
    shifts the state series and records the shift. Nothing else moves.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    from SymbolicDSGE import DSGESolver
    from SymbolicDSGE.kalman.filter import KalmanFilter

    solved, y = _levels_rbc_solved(1)
    pol = solved.policy
    C, d = solved._build_C_d_from_obs(["c_obs"])
    Q = np.asarray(DSGESolver._build_Q(solved.compiled), dtype=FLOAT)
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
    gaps = KalmanFilter.run(**args)
    levels = KalmanFilter.run(**args, steady_state=pol.steady_state)

    np.testing.assert_allclose(gaps.constant, np.zeros(n_var), rtol=0, atol=0)
    np.testing.assert_allclose(levels.constant, pol.steady_state, rtol=0, atol=0)
    np.testing.assert_allclose(
        levels.x_filt, gaps.x_filt + pol.steady_state, rtol=0, atol=0
    )
    # The observation series carry their own constant through d, so they do not
    # move with this one.
    np.testing.assert_allclose(levels.y_pred, gaps.y_pred, rtol=0, atol=0)
    np.testing.assert_allclose(levels.loglik, gaps.loglik, rtol=0, atol=0)
