# type: ignore
from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE.utils.dhm import DenHaanMarcet


@pytest.fixture(scope="module")
def solved_test():
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=2)
    return solver.solve(compiled)


@pytest.fixture(scope="module")
def solved_post82(post82_test_model_path):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=3)
    return solver.solve(compiled)


def test_den_haan_marcet_one_sample_matches_sim_state_path(solved_test):
    T = 12
    shocks = {
        "u": lambda sig: np.full((T,), sig, dtype=np.float64),
        "v": np.linspace(-0.25, 0.25, T, dtype=np.float64),
    }
    dhm = DenHaanMarcet(solved_test)

    expected = solved_test.sim(T, shocks=shocks)["_X"]
    out = dhm.one_sample(
        T,
        shocks=shocks,
        equation_idx=[0, 1],
        instrument_idx=["u", "v"],
        burn_in=1,
        use_conditional_expectation=False,
    )
    path_out = dhm.from_state_path(
        expected,
        equation_idx=[0, 1],
        instrument_idx=["u", "v"],
        burn_in=1,
        use_conditional_expectation=False,
    )

    assert np.allclose(out.states, expected)
    assert np.allclose(
        out.shock_matrix[:, solved_test.compiled.idx["u"]],
        np.full((T,), 0.50, dtype=np.float64),
    )
    assert np.allclose(
        out.shock_matrix[:, solved_test.compiled.idx["v"]],
        shocks["v"],
    )
    assert out.moments.shape == (T - 1, 6)
    assert out.residuals.shape == (T - 1, 2)
    assert out.instruments.shape == (T - 1, 3)
    assert out.variables == list(solved_test.compiled.var_names)
    assert np.allclose(out.raw_residuals, out.residuals)
    assert np.isfinite(out.statistic)
    assert 0.0 <= out.p_value <= 1.0
    assert np.allclose(path_out.states, out.states)
    assert path_out.shock_matrix is None
    assert path_out.variables == out.variables
    assert np.array_equal(path_out.equation_idx, out.equation_idx)
    assert np.array_equal(path_out.instrument_idx, out.instrument_idx)
    assert np.allclose(path_out.raw_residuals, out.raw_residuals)
    assert np.allclose(path_out.residuals, out.residuals)
    assert np.allclose(path_out.instruments, out.instruments)
    assert np.allclose(path_out.moments, out.moments)
    assert np.isfinite(path_out.statistic)
    assert 0.0 <= path_out.p_value <= 1.0


def test_den_haan_marcet_one_sample_uses_canonical_multivar_covariance(solved_post82):
    T = 10
    captured = {}
    dhm = DenHaanMarcet(solved_post82)

    def mv_shock(cov):
        captured["cov"] = cov.copy()
        return np.tile(np.array([cov[0, 0], cov[0, 1]], dtype=np.float64), (T, 1))

    shocks = {"z,g": mv_shock}
    expected = solved_post82.sim(T, shocks=shocks)["_X"]
    out = dhm.one_sample(
        T,
        shocks=shocks,
        equation_idx=[0, 1, 2],
        instrument_idx=["g", "z"],
        burn_in=1,
    )

    sig_g = solved_post82._get_param(solved_post82.config.calibration.shock_std["e_g"])
    sig_z = solved_post82._get_param(solved_post82.config.calibration.shock_std["e_z"])
    rho_gz = solved_post82._get_rho("e_g", "e_z")
    expected_cov = np.array(
        [
            [sig_g**2, sig_g * sig_z * rho_gz],
            [sig_g * sig_z * rho_gz, sig_z**2],
        ],
        dtype=np.float64,
    )

    assert np.allclose(captured["cov"], expected_cov)
    assert np.allclose(out.states, expected)
    assert np.allclose(
        out.shock_matrix[:, solved_post82.compiled.idx["g"]],
        expected_cov[0, 0],
    )
    assert np.allclose(
        out.shock_matrix[:, solved_post82.compiled.idx["z"]],
        expected_cov[0, 1],
    )


def test_den_haan_marcet_conditional_expectation_uses_projected_forward_states(
    solved_test,
):
    T = 10
    shocks = {
        "u": lambda sig: np.full((T,), sig, dtype=np.float64),
        "v": np.linspace(-0.25, 0.25, T, dtype=np.float64),
    }
    dhm = DenHaanMarcet(solved_test)
    states = solved_test.sim(T, shocks=shocks)["_X"]
    out = dhm.from_state_path(
        states,
        equation_idx=[1],
        instrument_idx=["u", "v"],
        burn_in=0,
        use_conditional_expectation=True,
    )
    realized = dhm.from_state_path(
        states,
        equation_idx=[1],
        instrument_idx=["u", "v"],
        burn_in=0,
        use_conditional_expectation=False,
    )

    current_states = states[:-1]
    projected_forward = current_states @ np.asarray(solved_test.A, dtype=np.float64).T
    params = np.array(
        [
            solved_test.compiled.config.calibration.parameters[p]
            for p in solved_test.compiled.calib_params
        ],
        dtype=np.complex128,
    )
    objective = solved_test.compiled.construct_objective_vector_func()
    manual = np.empty((T, 1), dtype=np.float64)
    for t in range(T):
        manual[t, 0] = objective(
            projected_forward[t].astype(np.complex128),
            current_states[t].astype(np.complex128),
            params,
        )[1].real

    assert np.allclose(out.residuals, manual)
    assert np.allclose(out.raw_residuals, manual)
    assert not np.allclose(out.residuals, realized.residuals)


def test_den_haan_marcet_monte_carlo_rejects_non_shock_inputs(solved_test):
    dhm = DenHaanMarcet(solved_test)

    with pytest.raises(TypeError, match="requires Shock instances"):
        dhm.monte_carlo(
            8,
            {"u": Shock(T=8, dist="norm", seed=3).shock_generator()},
            n_rep=2,
        )


def test_den_haan_marcet_monte_carlo_is_reproducible_without_mutation(solved_post82):
    T = 12
    dhm = DenHaanMarcet(solved_post82)
    grouped = Shock(T=T, dist="norm", multivar=True, seed=11)
    rate = Shock(T=T, dist="norm", seed=21)
    base_shocks = {"z,g": grouped, "r": rate}
    kwargs = {
        "equation_idx": [0, 1, 2],
        "instrument_idx": ["g", "z", "r"],
        "burn_in": 1,
    }

    out = dhm.monte_carlo(T, base_shocks, n_rep=2, **kwargs)
    out_repeat = dhm.monte_carlo(T, base_shocks, n_rep=2, **kwargs)

    rep0 = dhm.one_sample(
        T,
        shocks={
            "z,g": Shock(T=T, dist="norm", multivar=True, seed=11).shock_generator(),
            "r": Shock(T=T, dist="norm", seed=21).shock_generator(),
        },
        **kwargs,
    )
    rep1 = dhm.one_sample(
        T,
        shocks={
            "z,g": Shock(T=T, dist="norm", multivar=True, seed=12).shock_generator(),
            "r": Shock(T=T, dist="norm", seed=22).shock_generator(),
        },
        **kwargs,
    )

    assert grouped.seed == 11
    assert rate.seed == 21
    assert np.allclose(out.statistics, np.array([rep0.statistic, rep1.statistic]))
    assert np.allclose(out.p_values, np.array([rep0.p_value, rep1.p_value]))
    assert np.array_equal(
        out.rejections,
        np.array([rep0.rejects_null, rep1.rejects_null], dtype=bool),
    )
    assert out.variables == list(solved_post82.compiled.var_names)
    assert np.array_equal(out.equation_idx, rep0.equation_idx)
    assert out.raw_residuals.shape == (2, *rep0.raw_residuals.shape)
    assert np.allclose(out.raw_residuals[0], rep0.raw_residuals)
    assert np.allclose(out.raw_residuals[1], rep1.raw_residuals)
    assert np.allclose(out.statistics, out_repeat.statistics)
    assert np.allclose(out.p_values, out_repeat.p_values)
    assert np.array_equal(out.rejections, out_repeat.rejections)
    assert np.allclose(out.raw_residuals, out_repeat.raw_residuals)

    out_expected = dhm.monte_carlo(
        T,
        base_shocks,
        n_rep=2,
        use_conditional_expectation=True,
        **kwargs,
    )
    rep0_expected = dhm.one_sample(
        T,
        shocks={
            "z,g": Shock(T=T, dist="norm", multivar=True, seed=11).shock_generator(),
            "r": Shock(T=T, dist="norm", seed=21).shock_generator(),
        },
        use_conditional_expectation=True,
        **kwargs,
    )
    rep1_expected = dhm.one_sample(
        T,
        shocks={
            "z,g": Shock(T=T, dist="norm", multivar=True, seed=12).shock_generator(),
            "r": Shock(T=T, dist="norm", seed=22).shock_generator(),
        },
        use_conditional_expectation=True,
        **kwargs,
    )

    assert np.allclose(
        out_expected.statistics,
        np.array([rep0_expected.statistic, rep1_expected.statistic]),
    )
    assert np.allclose(
        out_expected.p_values,
        np.array([rep0_expected.p_value, rep1_expected.p_value]),
    )
    assert np.allclose(out_expected.raw_residuals[0], rep0_expected.raw_residuals)
    assert np.allclose(out_expected.raw_residuals[1], rep1_expected.raw_residuals)


def test_den_haan_marcet_custom_focs_match_compiled_equation_fallback(solved_test):
    T = 10
    focs = ["x(t) = x(t+1) - sigma*(r(t) - Pi(t+1)) + u(t)"]
    dhm_custom = DenHaanMarcet(solved_test, focs=focs)
    dhm_fallback = DenHaanMarcet(solved_test)

    custom = dhm_custom.one_sample(
        T,
        instrument_idx=["u", "v"],
        burn_in=0,
    )
    fallback = dhm_fallback.one_sample(
        T,
        equation_idx=[1],
        instrument_idx=["u", "v"],
        burn_in=0,
    )

    assert custom.foc_expressions is not None
    assert len(custom.foc_expressions) == 1
    assert np.allclose(custom.residuals[:, 0], fallback.residuals[:, 0])
    assert np.allclose(custom.instruments, fallback.instruments)
    assert np.allclose(custom.moments, fallback.moments)
    assert custom.statistic == pytest.approx(fallback.statistic)
    assert custom.p_value == pytest.approx(fallback.p_value)


def test_den_haan_marcet_custom_focs_match_fallback_under_conditional_expectation(
    solved_test,
):
    T = 10
    focs = ["x(t) = x(t+1) - sigma*(r(t) - Pi(t+1)) + u(t)"]
    dhm_custom = DenHaanMarcet(solved_test, focs=focs)
    dhm_fallback = DenHaanMarcet(solved_test)

    custom = dhm_custom.one_sample(
        T,
        instrument_idx=["u", "v"],
        burn_in=0,
        use_conditional_expectation=True,
    )
    fallback = dhm_fallback.one_sample(
        T,
        equation_idx=[1],
        instrument_idx=["u", "v"],
        burn_in=0,
        use_conditional_expectation=True,
    )

    assert np.allclose(custom.residuals[:, 0], fallback.residuals[:, 0])
    assert np.allclose(custom.raw_residuals[:, 0], fallback.raw_residuals[:, 0])
    assert np.allclose(custom.instruments, fallback.instruments)
    assert np.allclose(custom.moments, fallback.moments)
    assert custom.statistic == pytest.approx(fallback.statistic)
    assert custom.p_value == pytest.approx(fallback.p_value)


def test_den_haan_marcet_custom_foc_function_locals_match_direct_expression(
    solved_test,
):
    T = 10
    dhm_alias = DenHaanMarcet(
        solved_test,
        focs=["C(t) = exp(x(t+1))"],
        foc_locals={"C(t)": "exp(x(t))"},
    )
    dhm_direct = DenHaanMarcet(
        solved_test,
        focs=["exp(x(t)) = exp(x(t+1))"],
    )

    alias_out = dhm_alias.one_sample(
        T,
        instrument_idx=["u", "v"],
        burn_in=0,
    )
    direct_out = dhm_direct.one_sample(
        T,
        instrument_idx=["u", "v"],
        burn_in=0,
    )

    assert alias_out.foc_expressions == direct_out.foc_expressions
    assert np.allclose(alias_out.residuals, direct_out.residuals)
    assert np.allclose(alias_out.instruments, direct_out.instruments)
    assert np.allclose(alias_out.moments, direct_out.moments)
    assert alias_out.statistic == pytest.approx(direct_out.statistic)
    assert alias_out.p_value == pytest.approx(direct_out.p_value)


def test_den_haan_marcet_custom_foc_scalar_locals_match_direct_expression(solved_test):
    T = 10
    dhm = DenHaanMarcet(solved_test)

    alias_out = dhm.one_sample(
        T,
        focs=["x(t) = x(t+1) - gap_term + u(t)"],
        foc_locals={"gap_term": "sigma*(r(t) - Pi(t+1))"},
        instrument_idx=["u", "v"],
        burn_in=0,
    )
    direct_out = dhm.one_sample(
        T,
        focs=["x(t) = x(t+1) - sigma*(r(t) - Pi(t+1)) + u(t)"],
        instrument_idx=["u", "v"],
        burn_in=0,
    )

    assert alias_out.foc_expressions == direct_out.foc_expressions
    assert np.allclose(alias_out.residuals, direct_out.residuals)
    assert np.allclose(alias_out.instruments, direct_out.instruments)
    assert np.allclose(alias_out.moments, direct_out.moments)
    assert alias_out.statistic == pytest.approx(direct_out.statistic)
    assert alias_out.p_value == pytest.approx(direct_out.p_value)


def test_den_haan_marcet_custom_foc_locals_reject_symbol_collisions(solved_test):
    dhm = DenHaanMarcet(solved_test)

    with pytest.raises(ValueError, match="collides with an existing model symbol"):
        dhm.one_sample(
            6,
            focs=["x(t) = x(t+1)"],
            foc_locals={"sigma": "1.0"},
            instrument_idx=["u"],
            burn_in=0,
        )


def test_den_haan_marcet_custom_focs_reject_bare_model_variables(solved_test):
    dhm = DenHaanMarcet(solved_test)

    with pytest.raises(ValueError, match="explicit time index"):
        dhm.one_sample(
            6,
            focs=["x + rho_r"],
            instrument_idx=["u"],
            burn_in=0,
        )


def test_den_haan_marcet_custom_focs_reject_large_time_shifts(solved_test):
    dhm = DenHaanMarcet(solved_test)

    with pytest.raises(ValueError, match="within one period"):
        dhm.one_sample(
            6,
            focs=["x(t+2) + rho_r*Pi(t+1)"],
            instrument_idx=["u"],
            burn_in=0,
        )
