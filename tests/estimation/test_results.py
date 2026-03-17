# type: ignore
import numpy as np
import pytest

from SymbolicDSGE.estimation.results import MCMCResult


def _result(
    *,
    samples: np.ndarray,
    logpost: np.ndarray,
) -> MCMCResult:
    return MCMCResult(
        param_names=["a", "b"],
        samples=np.asarray(samples, dtype=np.float64),
        logpost_trace=np.asarray(logpost, dtype=np.float64),
        accept_rate=np.float64(0.25),
        n_draws=int(samples.shape[0]),
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
    )

    with pytest.raises(ValueError, match="0 <= alpha < 1"):
        res.hpd_intervals(alpha=alpha)
    with pytest.raises(ValueError, match="0 <= alpha < 1"):
        res.joint_hpd_set(alpha=alpha)


def test_hpd_validation_and_window_full_draws_branches():
    res = _result(
        samples=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        logpost=np.array([0.0, 1.0], dtype=np.float64),
    )

    assert MCMCResult._validate_hpd_alpha(0.5) == pytest.approx(0.5)
    assert MCMCResult._hpd_window_size(5, np.float64(0.0)) == 5
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
    res = MCMCResult(
        param_names=param_names,
        samples=samples,
        logpost_trace=np.array([0.0, 1.0], dtype=np.float64),
        accept_rate=np.float64(0.5),
        n_draws=2,
        burn_in=0,
        thin=1,
    )
    with pytest.raises(ValueError, match=match):
        res._validate_samples()


def test_joint_hpd_set_validates_logpost_trace_shape():
    res = MCMCResult(
        param_names=["a", "b"],
        samples=np.ones((2, 2), dtype=np.float64),
        logpost_trace=np.ones((2, 1), dtype=np.float64),
        accept_rate=np.float64(0.5),
        n_draws=2,
        burn_in=0,
        thin=1,
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

    res = MCMCResult(
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
        accept_rate=np.float64(0.5),
        n_draws=3,
        burn_in=0,
        thin=1,
    )

    res.posterior_kde_plot()
    res.posterior_traces()
    res.logpost_trace_plot()

    assert calls == ["show", "show", "show"]
