# type: ignore
import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.estimation.results import (
    MCMCResult,
    MLEResult,
    MAPResult,
    _hpd_window_size,
    _validate_hpd_alpha,
    _validate_samples,
)

from tests._result_helpers import make_mcmc_result


def _result(
    *,
    samples: np.ndarray,
    logpost: np.ndarray,
    logjac: np.ndarray,
) -> MCMCResult:
    return make_mcmc_result(
        samples=samples,
        logpost_trace=logpost,
        logjac_trace=logjac,
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


def test_joint_hpd_set_keeps_highest_logpost_draws():
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

    samples_hpd, logpost_hpd, threshold, idx = res.joint_hpd_set(alpha=0.4)

    assert threshold == pytest.approx(1.0)
    assert np.array_equal(idx, np.array([1, 2, 3], dtype=np.int64))
    assert np.allclose(
        samples_hpd,
        np.array(
            [
                [1.0, 5.1],
                [2.0, 5.2],
                [10.0, 50.0],
            ],
            dtype=np.float64,
        ),
    )
    assert np.allclose(logpost_hpd, np.array([1.0, 3.0, 2.0], dtype=np.float64))


def test_joint_hpd_set_includes_all_boundary_ties():
    res = _result(
        samples=np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        ),
        logpost=np.array([3.0, 2.0, 2.0, 1.0], dtype=np.float64),
        logjac=np.zeros((4,), dtype=np.float64),
    )

    samples_hpd, logpost_hpd, threshold, idx = res.joint_hpd_set(alpha=0.5)

    assert threshold == pytest.approx(2.0)
    assert np.array_equal(idx, np.array([0, 1, 2], dtype=np.int64))
    assert samples_hpd.shape == (3, 2)
    assert np.allclose(logpost_hpd, np.array([3.0, 2.0, 2.0], dtype=np.float64))


@pytest.mark.parametrize("alpha", [-0.1, 1.0])
def test_hpd_methods_validate_alpha(alpha):
    res = _result(
        samples=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        logpost=np.array([0.0, 1.0], dtype=np.float64),
        logjac=np.zeros((2,), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="0 <= alpha < 1"):
        res.hpd_intervals(alpha=alpha)
    with pytest.raises(ValueError, match="0 <= alpha < 1"):
        res.joint_hpd_set(alpha=alpha)


def test_hpd_validation_and_window_full_draws_branches():
    res = _result(
        samples=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        logpost=np.array([0.0, 1.0], dtype=np.float64),
        logjac=np.zeros((2,), dtype=np.float64),
    )

    assert _validate_hpd_alpha(0.5) == pytest.approx(0.5)
    assert _hpd_window_size(5, np.float64(0.0)) == 5
    assert res.hpd_intervals(alpha=0.0) == {
        "a": pytest.approx((0.0, 2.0)),
        "b": pytest.approx((1.0, 3.0)),
    }


@pytest.mark.parametrize(
    ("samples", "param_names", "match"),
    [
        (np.array([1.0, 2.0], dtype=np.float64), ["a", "b"], "2D array"),
        (np.empty((0, 2), dtype=np.float64), ["a", "b"], "empty"),
        (np.ones((2, 2), dtype=np.float64), ["a"], "does not match"),
    ],
)
def test_validate_samples_error_branches(samples, param_names, match):
    with pytest.raises(ValueError, match=match):
        _validate_samples(samples, tuple(param_names))


def test_joint_hpd_set_validates_logpost_trace_shape():
    res = make_mcmc_result(
        samples=np.ones((2, 2), dtype=np.float64),
        logpost_trace=np.ones((2, 1), dtype=np.float64),
        logjac_trace=np.ones((2, 1), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="1D array"):
        res.joint_hpd_set()


def test_result_plot_methods_execute_without_gui(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.stats

    calls: list[str] = []

    class _FakeKDE:
        def __call__(self, x):
            return np.ones_like(x, dtype=np.float64)

    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))
    monkeypatch.setattr(scipy.stats, "gaussian_kde", lambda col: _FakeKDE())

    res = make_mcmc_result(
        param_names=["a", "b", "c"],
        samples=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.5, 1.5, 2.5],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float64,
        ),
        logpost_trace=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        logjac_trace=np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )

    res.posterior_kde_plot()
    res.posterior_traces()
    res.logpost_trace_plot()

    assert calls == ["show", "show", "show"]


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
    np.testing.assert_array_equal(rebuilt.samples, samples)
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


def test_ess_is_clipped_to_the_draw_count():
    # tau near 0.46 at any length, so the raw ESS is about 2.2x the draws.
    anticorrelated = np.tile([-1.0, -1.0, 1.0, -1.0, 0.0, -1.0, 0.0], 512)
    res = make_mcmc_result(param_names=["a"], samples=anticorrelated.reshape(-1, 1))

    assert res.ess["a"] == float(res.n_draws)


def test_ess_is_cached_and_stays_out_of_the_spec():
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(0).standard_normal((4000, 1)),
    )

    assert res.ess is res.ess
    assert "ess" in res.__dict__
    assert "ess" not in res.to_spec().meta


def test_ess_per_second_divides_by_the_in_sample_clock():
    res = make_mcmc_result(
        param_names=["a", "b"],
        samples=np.random.default_rng(1).standard_normal((4000, 2)),
        in_sample_seconds=2.5,
    )

    assert res.ess_per_second == {name: value / 2.5 for name, value in res.ess.items()}


def test_ess_and_mcse_are_nan_for_a_parameter_that_never_moved():
    res = make_mcmc_result(
        param_names=["moving", "stuck"],
        samples=np.column_stack(
            [np.random.default_rng(2).standard_normal(4000), np.ones(4000)]
        ),
    )

    assert np.isfinite(res.ess["moving"])
    assert np.isfinite(res.mcse["moving"])
    assert np.isnan(res.ess["stuck"])
    assert np.isnan(res.mcse["stuck"])
    assert np.isnan(res.mcse_quantile(0.5)["stuck"])


def test_mcse_of_the_mean_tracks_the_standard_error():
    n = 20000
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(7).standard_normal((n, 1)),
    )

    assert res.mcse["a"] == pytest.approx(1.0 / np.sqrt(n), rel=0.1)


def test_mcse_scales_with_the_parameter_it_belongs_to():
    """Proportional columns share an ESS, so their MCSEs differ only by scale."""
    chain = np.random.default_rng(3).standard_normal(6000)
    res = make_mcmc_result(
        param_names=["unit", "ten", "hundred"],
        samples=np.column_stack([chain, 10.0 * chain, 100.0 * chain]),
    )

    out = res.mcse
    assert out["ten"] == pytest.approx(10.0 * out["unit"], rel=1e-12)
    assert out["hundred"] == pytest.approx(100.0 * out["unit"], rel=1e-12)


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_mcse_quantile_is_scale_equivariant(q):
    chain = np.random.default_rng(4).standard_normal(8000)
    plain = make_mcmc_result(param_names=["a"], samples=chain.reshape(-1, 1))
    scaled = make_mcmc_result(param_names=["a"], samples=(4.0 * chain).reshape(-1, 1))

    assert scaled.mcse_quantile(q)["a"] == pytest.approx(
        4.0 * plain.mcse_quantile(q)["a"], rel=1e-12
    )


def test_mcse_quantile_is_larger_in_the_tails_than_at_the_median():
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(5).standard_normal((8000, 1)),
    )

    centre = res.mcse_quantile(0.5)["a"]
    assert res.mcse_quantile(0.05)["a"] > centre
    assert res.mcse_quantile(0.95)["a"] > centre


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_mcse_quantile_requires_an_interior_quantile(q):
    res = make_mcmc_result(
        param_names=["a"],
        samples=np.random.default_rng(6).standard_normal((2000, 1)),
    )

    with pytest.raises(ValueError, match="0 < q < 1"):
        res.mcse_quantile(q)
