# type: ignore
import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.estimation.results import (
    MCMCResult,
    MLEResult,
    MAPResult,
)
from tests._result_helpers import make_mcmc_result


def _result(
    *,
    samples: np.ndarray,
    logpost: np.ndarray,
    logjac: np.ndarray,
) -> MCMCResult:
    return make_mcmc_result(
        samples=samples.reshape(1, samples.shape[0], samples.shape[1]),
        logpost_trace=logpost.reshape(1, logpost.shape[0]),
        logjac_trace=logjac.reshape(1, logjac.shape[0]),
        burn_in=10,
        thin=2,
    )


def test_hpd_intervals_returns_shortest_marginal_windows():
    res = _result(
        samples=np.array(
            [
                [0.0, 5.0],
                [1.0, 5.1],
                [2.0, 5.2],
                [10.0, 50.0],
                [11.0, 51.0],
            ],
            dtype=np.float64,
        ),
        logpost=np.array([0.0, 1.0, 3.0, 2.0, -1.0], dtype=np.float64),
        logjac=np.zeros((5,), dtype=np.float64),
    )

    out = res.hpd_intervals(alpha=0.4)

    assert out["a"] == pytest.approx((0.0, 2.0))
    assert out["b"] == pytest.approx((5.0, 5.2))


@pytest.mark.parametrize("alpha", [-0.1, 1.0])
def test_hpd_methods_validate_alpha(alpha):
    res = _result(
        samples=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        logpost=np.array([0.0, 1.0], dtype=np.float64),
        logjac=np.zeros((2,), dtype=np.float64),
    )

    with pytest.raises(ValueError, match=r"interval \[0, 1\)"):
        res.hpd_intervals(alpha=alpha)


def test_hpd_end_to_end_interval_covers_the_chain():
    res = _result(
        samples=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        logpost=np.array([0.0, 1.0], dtype=np.float64),
        logjac=np.zeros((2,), dtype=np.float64),
    )
    assert res.hpd_intervals(alpha=0.0) == {
        "a": pytest.approx((0.0, 2.0)),
        "b": pytest.approx((1.0, 3.0)),
    }


def test_result_plot_methods_execute_without_gui(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: list[int] = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(1))

    res = make_mcmc_result(
        param_names=["a", "b", "c"],
        samples=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.5, 1.5, 2.5],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float64,
        ).reshape(1, 3, 3),
        logpost_trace=np.array([0.0, 1.0, 2.0], dtype=np.float64).reshape(1, 3),
        logjac_trace=np.array([0.0, 1.0, 2.0], dtype=np.float64).reshape(1, 3),
    )

    res.posterior_kde_plot()
    res.logpost_trace_plot()

    assert sum(calls) == 2


def test_posterior_kde_plot_warns_and_skips_a_constant_parameter(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: list[str] = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    res = make_mcmc_result(
        param_names=["a", "b", "c"],
        samples=np.array(
            [
                [0.0, 4.0, 2.0],
                [0.5, 4.0, 2.5],
                [1.0, 4.0, 3.0],
            ],
            dtype=np.float64,
        ).reshape(1, 3, 3),
    )

    with pytest.warns(UserWarning, match="'b'") as record:
        res.posterior_kde_plot(grid_points=50)

    assert len(record) == 1
    assert calls == ["show"]
    assert [len(ax.lines) for ax in plt.gcf().axes] == [1, 0, 1]


def test_map_result_spec_roundtrips():
    res = MAPResult(
        x=np.array([0.1, 0.2], dtype=np.float64),
        theta={"a": float64(0.1), "b": float64(0.2)},
        success=True,
        message="converged",
        fun=float64(-3.5),
        nfev=42,
        nit=7,
        optimizer_config={"method": "L-BFGS-B"},
        logpost=float64(-3.5),
        logprior=float64(-0.5),
    )

    d = res.to_spec()
    assert d["theta"] == {"a": 0.1, "b": 0.2}
    assert d["nfev"] == 42 and d["nit"] == 7
    assert d["logpost"] == -3.5 and d["logprior"] == -0.5
    # round-trips through the serializable dict form
    assert MAPResult.from_spec(d).to_spec() == d


def test_mle_result_spec_roundtrips():
    res = MLEResult(
        x=np.array([1.0], dtype=np.float64),
        theta={"a": float64(1.0)},
        success=True,
        message="ok",
        fun=float64(-2.0),
        nfev=5,
        nit=None,
        optimizer_config={
            "method": "L-BFGS-B",
            "bounds": {"a": [0.0, 1.0]},
            "options": {"maxiter": 10},
        },
        loglik=float64(-2.0),
    )

    d = res.to_spec()
    assert d["loglik"] == -2.0 and d["nit"] is None
    assert d["optimizer_config"]["method"] == "L-BFGS-B"
    assert MLEResult.from_spec(d).to_spec() == d


def test_mcmc_result_to_spec():
    samples = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    logpost = np.array([-1.0, -2.0], dtype=np.float64)
    logjac = np.array([0.0, 0.0], dtype=np.float64)
    res = _result(samples=samples, logpost=logpost, logjac=logjac)

    spec = res.to_spec()
    meta = spec.meta
    assert meta["param_names"] == ["a", "b"]
    assert (meta["n_draws"], meta["burn_in"], meta["thin"]) == (2, 10, 2)
    # the bulk columns ride the container, not the meta
    np.testing.assert_array_equal(np.asarray(spec.samples), samples)
    # and the container round-trips back to a live result
    rebuilt = MCMCResult.from_spec(spec)
    np.testing.assert_array_equal(rebuilt.samples, samples.reshape(1, 2, 2))
    assert rebuilt.param_names == ("a", "b")
    # the sampler's own bookkeeping survives the projection field by field, which
    # the distinct defaults are what make visible
    assert (rebuilt.accepted_burn_in, rebuilt.accepted_in_sample) == (
        res.accepted_burn_in,
        res.accepted_in_sample,
    )
    assert (rebuilt.burn_in_seconds, rebuilt.in_sample_seconds) == (
        res.burn_in_seconds,
        res.in_sample_seconds,
    )


def test_ess_is_clipped_at_s_log10_s():
    # AR(1) at phi = -0.8 has tau = 1/9, so the raw ESS is about 9x the draws.
    innovations = np.random.default_rng(0).standard_normal(1000)
    chain = np.empty_like(innovations)
    chain[0] = innovations[0]
    for t in range(1, chain.size):
        chain[t] = -0.8 * chain[t - 1] + innovations[t]
    res = make_mcmc_result(param_names=["a"], samples=chain.reshape(1, -1, 1))

    S = res.n_chains * res.n_draws
    cap = S * np.log10(S)

    assert res.ess["a"] == cap
    assert res.ess_bulk["a"] == cap


def test_ess_is_cached_and_stays_out_of_the_spec():
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(0).standard_normal((1, 100, 1)),
    )
    res.ess
    assert "ess" not in res.to_spec().meta


def test_ess_per_second_divides_by_kernel_wall_clock():
    res = make_mcmc_result(
        param_names=["a", "b"],
        samples=np.random.default_rng(1).standard_normal((1, 100, 2)),
        wall_seconds=2.5,
    )

    assert res.ess_per_second == {name: value / 2.5 for name, value in res.ess.items()}


def test_ess_and_mcse_are_nan_for_a_parameter_that_never_moved():
    res = make_mcmc_result(
        param_names=["moving", "stuck"],
        samples=np.column_stack(
            [np.random.default_rng(2).standard_normal(100), np.ones(100)]
        ).reshape(1, -1, 2),
    )

    assert np.isfinite(res.ess["moving"])
    assert np.isfinite(res.mcse["moving"])
    assert np.isnan(res.ess["stuck"])
    assert np.isnan(res.mcse["stuck"])
    assert np.isnan(res.mcse_quantile(0.5)["stuck"])


def _chains(spread: list[float], offset: list[float], seed: int) -> MCMCResult:
    draws = np.random.default_rng(seed).standard_normal((len(spread), 100, 1))
    draws = draws * np.asarray(spread)[:, None, None]
    draws = draws + np.asarray(offset)[:, None, None]
    return make_mcmc_result(param_names=["a"], samples=draws, n_chains=len(spread))


def test_split_rhat_flags_a_chain_stuck_elsewhere():
    mixed = _chains([1.0] * 4, [0.0] * 4, seed=8)
    shifted = _chains([1.0] * 4, [0.0, 0.0, 0.0, 1.0], seed=8)

    assert mixed.split_rhat["a"] == pytest.approx(1.0, abs=0.01)
    assert shifted.split_rhat["a"] > 1.05


def test_rhat_catches_a_scale_mismatch_split_rhat_misses():
    res = _chains([1.0, 1.0, 1.0, 3.0], [0.0] * 4, seed=9)

    assert res.split_rhat["a"] == pytest.approx(1.0, abs=0.01)
    assert res.rhat["a"] > 1.05


def test_rhat_flags_a_location_shift():
    res = _chains([1.0] * 4, [0.0, 0.0, 0.0, 1.0], seed=10)

    assert res.rhat["a"] > 1.05


def test_rhat_and_ess_bulk_are_nan_for_a_parameter_that_never_moved():
    res = make_mcmc_result(
        param_names=["moving", "stuck"],
        samples=np.column_stack(
            [np.random.default_rng(11).standard_normal(100), np.ones(100)]
        ).reshape(1, -1, 2),
    )

    assert np.isfinite(res.rhat["moving"])
    assert np.isfinite(res.ess_bulk["moving"])
    assert np.isnan(res.rhat["stuck"])
    assert np.isnan(res.ess_bulk["stuck"])


def test_ess_tail_is_the_smaller_of_the_two_tail_quantile_ess():
    res = make_mcmc_result(
        param_names=["a", "b"],
        samples=np.random.default_rng(12).standard_normal((2, 2000, 2)),
        n_chains=2,
    )
    lo = res.ess_quantile(0.05)
    hi = res.ess_quantile(0.95)

    assert res.ess_tail == {name: min(lo[name], hi[name]) for name in res.param_names}


def test_ess_tail_and_ess_quantile_are_nan_for_a_parameter_that_never_moved():
    res = make_mcmc_result(
        param_names=["moving", "stuck"],
        samples=np.column_stack(
            [np.random.default_rng(13).standard_normal(4000), np.ones(4000)]
        ).reshape(1, -1, 2),
    )

    assert np.isfinite(res.ess_tail["moving"])
    assert np.isfinite(res.ess_quantile(0.5)["moving"])
    assert np.isnan(res.ess_tail["stuck"])
    assert np.isnan(res.ess_quantile(0.5)["stuck"])


def test_ess_quantile_rounds_to_the_requested_digits():
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(14).standard_normal((1, 4000, 1)),
    )

    assert res.ess_quantile(0.3, digits=1)["a"] == round(res.ess_quantile(0.3)["a"], 1)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_ess_quantile_requires_an_interior_quantile(q):
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(15).standard_normal((1, 2000, 1)),
    )

    with pytest.raises(ValueError, match=r"interval \(0, 1\)"):
        res.ess_quantile(q)


def test_mcse_of_the_mean_tracks_the_standard_error():
    n = 20000
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(7).standard_normal((1, n, 1)),
    )

    assert res.mcse["a"] == pytest.approx(1.0 / np.sqrt(n), rel=0.1)


def test_mcse_scales_with_the_parameter_it_belongs_to():
    """Proportional columns share an ESS, so their MCSEs differ only by scale."""
    chain = np.random.default_rng(3).standard_normal(6000)
    res = make_mcmc_result(
        param_names=["unit", "ten", "hundred"],
        samples=np.column_stack([chain, 10.0 * chain, 100.0 * chain]).reshape(1, -1, 3),
    )

    out = res.mcse
    assert out["ten"] == pytest.approx(10.0 * out["unit"], rel=1e-12)
    assert out["hundred"] == pytest.approx(100.0 * out["unit"], rel=1e-12)


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_mcse_quantile_is_scale_equivariant(q):
    chain = np.random.default_rng(4).standard_normal(6000)
    plain = make_mcmc_result(param_names=["a"], samples=chain.reshape(1, -1, 1))
    scaled = make_mcmc_result(
        param_names=["a"], samples=(4.0 * chain).reshape(1, -1, 1)
    )

    assert scaled.mcse_quantile(q)["a"] == pytest.approx(
        4.0 * plain.mcse_quantile(q)["a"], rel=1e-12
    )


def test_mcse_quantile_is_larger_in_the_tails_than_at_the_median():
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(5).standard_normal((1, 6000, 1)),
    )

    centre = res.mcse_quantile(0.5)["a"]
    assert res.mcse_quantile(0.05)["a"] > centre
    assert res.mcse_quantile(0.95)["a"] > centre


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_mcse_quantile_requires_an_interior_quantile(q):
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(6).standard_normal((1, 100, 1)),
    )

    with pytest.raises(ValueError, match=r"interval \(0, 1\)"):
        res.mcse_quantile(q)


def _point_result() -> MCMCResult:
    """Four draws whose mean, highest-logpost row, and last row all differ.

    The values are dyadic, so the mean is exact and the branches can be
    compared by equality.
    """
    return make_mcmc_result(
        param_names=("phi_pi", "rho_r"),
        samples=np.array(
            [[1.0, 0.25], [2.0, 0.5], [4.0, 1.0], [6.0, 1.5]], dtype=np.float64
        ).reshape(1, -1, 2),
        logpost_trace=np.array([0.0, 5.0, 1.0, 2.0], dtype=np.float64).reshape(1, -1),
    )


def test_posterior_point_selects_mean_and_map():
    res = _point_result()

    assert res.posterior_point("mean") == {"phi_pi": 3.25, "rho_r": 0.8125}
    assert res.posterior_point("map") == {"phi_pi": 2.0, "rho_r": 0.5}


def test_posterior_point_rejects_an_unknown_point():
    with pytest.raises(ValueError, match="must be one of"):
        _point_result().posterior_point("median")


def test_apply_to_writes_the_optimum_over_the_calibration(parsed_test):
    untouched = parsed_test.calibration.parameters["kappa"]
    res = MLEResult(
        x=np.zeros(2, dtype=np.float64),
        theta={"phi_pi": float64(4.0), "rho_r": float64(0.5)},
        success=True,
        message="ok",
        fun=float64(-1.0),
        nfev=1,
        nit=1,
        optimizer_config={"method": "L-BFGS-B"},
        loglik=float64(-1.0),
    )

    conf = res.apply_to(parsed_test)

    assert conf.calibration.parameters["phi_pi"] == 4.0
    assert conf.calibration.parameters["rho_r"] == 0.5
    assert conf.calibration.parameters["kappa"] == untouched
    assert parsed_test.calibration.parameters["phi_pi"] != 4.0


def test_apply_to_defaults_to_the_posterior_mean(parsed_test):
    res = _point_result()

    assert res.apply_to(parsed_test).calibration.parameters["phi_pi"] == 3.25
