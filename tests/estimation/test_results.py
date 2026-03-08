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
