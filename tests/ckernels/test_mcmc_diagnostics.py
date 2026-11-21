"""Tests for the MCMC diagnostics kernels in ``_ckernels/estimation``.

An AR(1) has ``tau = (1 + phi) / (1 - phi)`` in closed form, which is the oracle
for the estimator; the tolerance is wide because ``tau`` is itself estimated off
a finite chain. The four status codes need chains a sampler cannot produce, so
each has a deterministic generator rather than a seeded draw. The marginal KDE's
oracle is ``scipy.stats.gaussian_kde`` evaluated on the kernel's own grid.

The rank-normalized R-hat oracle follows Vehtari et al. (2021): pooled average
ranks through ``scipy.stats.rankdata``, ``ndtri((r - 3/8) / (S + 1/4))``, and the
classic split R-hat on the result, with the fold taken as ``|x - median(x)|``.
The split drops each chain's last draw at odd N, as the kernels do.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import gaussian_kde, rankdata

from SymbolicDSGE._ckernels.estimation._mcmc_diagnostics import (
    bulk_ess_rn_rhat,
    ess_rhat,
    marginal_kde,
    SDSGE_ESS_EBADSHAPE,
    SDSGE_ESS_EUDEFVARIANCE,
    SDSGE_ESS_ENODECAY,
    SDSGE_ESS_ENONPOSTAU,
)

EBADSHAPE = SDSGE_ESS_EBADSHAPE
EUDEFVARIANCE = SDSGE_ESS_EUDEFVARIANCE
ENODECAY = SDSGE_ESS_ENODECAY
ENONPOSTAU = SDSGE_ESS_ENONPOSTAU

NONPOSITIVE_TAU = np.tile([1.0, -1.0, 1.0, -1.0, 1.0, 0.0], 170).reshape(1, 1020, 1)
CONSTANT = np.ones((1, 1024, 1))
half = np.tile([1.0, -1.0], 256)
UNDECAYED = np.concatenate([half + 10.0, half - 10.0]).reshape(1, 1024, 1)


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n)
    chain = np.empty(n)
    chain[0] = innovations[0] / np.sqrt(1.0 - phi**2) if phi else innovations[0]
    for t in range(1, n):
        chain[t] = phi * chain[t - 1] + innovations[t]
    return chain.reshape(1, n, -1)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9, 0.98])
def test_ess_recovers_the_ar1_autocorrelation_time(phi):
    n = 20000
    tau = n / ess_rhat(_ar1(n, phi, seed=int(phi * 100)))[0][0]
    assert tau == pytest.approx((1.0 + phi) / (1.0 - phi), rel=0.15)


@pytest.mark.parametrize(
    ("chain", "code"),
    [
        (np.full((1, 1, 1), 1.0), EBADSHAPE),
        (CONSTANT, EUDEFVARIANCE),
        (UNDECAYED, ENODECAY),
        (NONPOSITIVE_TAU, ENONPOSTAU),
    ],
    ids=["single_draw", "constant", "undecayed", "nonpositive_tau"],
)
def test_kernel_pairs_a_code_with_nan_for_degenerate_chains(chain, code):
    val, _, rc = ess_rhat(chain)
    assert rc[0] == code
    assert np.isnan(val[0])


@pytest.mark.parametrize(
    "chain",
    [CONSTANT, UNDECAYED, NONPOSITIVE_TAU],
    ids=["constant", "undecayed", "nonpositive_tau"],
)
def test_wrappers_forward_nan_instead_of_raising(chain):
    assert np.isnan(ess_rhat(chain)[0][0])


@pytest.mark.parametrize(
    ("scale", "shift"), [(2.0, 0.0), (0.25, 100.0), (3.7, -5.5), (-1.0, 0.0)]
)
def test_ess_is_invariant_to_affine_rescaling(scale, shift):
    chain = _ar1(100, 0.8, seed=5)
    scaled = ess_rhat(scale * chain + shift)[0][0]
    raw = ess_rhat(chain)[0][0]
    assert scaled == pytest.approx(raw, rel=1e-12)


def test_ess_is_invariant_to_time_reversal():
    chain = _ar1(100, 0.8, seed=5)
    forward = ess_rhat(chain)[0][0]
    reverse = ess_rhat(chain[::-1])[0][0]
    assert forward == pytest.approx(reverse, rel=1e-12)


def _ar1_chains(n_chains: int, n: int, phi: float, seed: int) -> np.ndarray:
    return np.concatenate(
        [_ar1(n, phi, seed=seed + c).reshape(1, n) for c in range(n_chains)]
    )


def _split(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, n : 2 * n]])


def _classic_rhat(x: np.ndarray) -> float:
    s = _split(x)
    n = s.shape[1]
    w = s.var(axis=1, ddof=1).mean()
    b = n * s.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def _z_scale(x: np.ndarray) -> np.ndarray:
    r = rankdata(x.reshape(-1), method="average").reshape(x.shape)
    return ndtri((r - 0.375) / (x.size + 0.25))


def _rank_normalized_oracle(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    z_bulk = _z_scale(x)
    z_fold = _z_scale(np.abs(x - np.median(x)))
    return _classic_rhat(z_fold), _classic_rhat(z_bulk), z_bulk


rng = np.random.default_rng(13)
SHIFTED = _ar1_chains(4, 100, 0.3, seed=20) + np.array([0.0, 0.0, 0.0, 1.5])[:, None]
RANK_CASES = {
    "ar1": _ar1_chains(4, 100, 0.9, seed=10),
    "shifted_chain": SHIFTED,
    "cauchy": rng.standard_cauchy((4, 100)),
    "rounded_ties": np.round(_ar1_chains(4, 100, 0.5, seed=30), 1),
    "integer_levels": rng.integers(0, 5, (4, 100)).astype(np.float64),
    "ties_across_median": np.tile([-1.0, 0.0, 1.0, 2.0], (4, 25)),
    "odd_draws": _ar1_chains(3, 777, 0.7, seed=40),
    "one_chain": _ar1_chains(1, 600, 0.5, seed=50),
    "four_draws": rng.standard_normal((2, 4)),
}


@pytest.mark.parametrize("x", RANK_CASES.values(), ids=RANK_CASES.keys())
def test_split_rhat_matches_the_classic_estimator(x):
    _, rhat, _ = ess_rhat(x[:, :, None])
    assert rhat[0] == pytest.approx(_classic_rhat(x), rel=1e-12)


@pytest.mark.parametrize("x", RANK_CASES.values(), ids=RANK_CASES.keys())
def test_rank_normalized_rhat_matches_the_reference(x):
    folded, bulk, _ = bulk_ess_rn_rhat(x[:, :, None])
    ref_folded, ref_bulk, _ = _rank_normalized_oracle(x)
    assert folded[0] == pytest.approx(ref_folded, rel=1e-12)
    assert bulk[0] == pytest.approx(ref_bulk, rel=1e-12)


@pytest.mark.parametrize("x", RANK_CASES.values(), ids=RANK_CASES.keys())
def test_bulk_ess_is_the_ess_of_the_rank_normalized_draws(x):
    _, _, bulk_ess = bulk_ess_rn_rhat(x[:, :, None])
    *_, z_bulk = _rank_normalized_oracle(x)
    np.testing.assert_allclose(
        bulk_ess, ess_rhat(z_bulk[:, :, None])[0], rtol=1e-12, equal_nan=True
    )


def test_bulk_rhat_is_invariant_to_monotone_transforms():
    x = _ar1_chains(4, 1000, 0.8, seed=60)[:, :, None]
    _, bulk, bulk_ess = bulk_ess_rn_rhat(x)
    _, bulk_exp, bulk_ess_exp = bulk_ess_rn_rhat(np.exp(x))
    np.testing.assert_array_equal(bulk_exp, bulk)
    np.testing.assert_array_equal(bulk_ess_exp, bulk_ess)


@pytest.mark.parametrize("scale", [2.0, 0.25, -1.0, -4.0])
def test_rank_normalized_rhat_is_invariant_to_exact_rescaling(scale):
    x = _ar1_chains(4, 1000, 0.8, seed=70)[:, :, None]
    raw = bulk_ess_rn_rhat(x)
    scaled = bulk_ess_rn_rhat(scale * x)
    for a, b in zip(scaled, raw):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_non_finite_parameters_are_nan_and_leave_the_others_alone():
    x = _ar1_chains(4, 1000, 0.5, seed=80)[:, :, None].repeat(4, axis=2)
    x[1, 5, 0] = np.nan
    x[2, 9, 1] = np.inf
    x[:, :, 2] = 2.0
    out = bulk_ess_rn_rhat(x)
    alone = bulk_ess_rn_rhat(x[:, :, 3:])
    for diag, single in zip(out, alone):
        assert np.isnan(diag[:3]).all()
        assert diag[3] == single[0]


@pytest.mark.parametrize(
    "shape",
    [(0, 10, 2), (2, 0, 2), (2, 3, 1)],
    ids=["no_chains", "no_draws", "three_draws"],
)
def test_rank_normalized_rhat_is_nan_for_short_chains(shape):
    for diag in bulk_ess_rn_rhat(np.ones(shape)):
        assert diag.shape == (shape[2],)
        assert np.isnan(diag).all()


def test_rank_normalized_rhat_leaves_the_input_unsorted():
    x = _ar1_chains(4, 500, 0.5, seed=90)[:, :, None].repeat(3, axis=2)
    before = x.copy()
    bulk_ess_rn_rhat(x)
    np.testing.assert_array_equal(x, before)


def test_rank_normalized_rhat_per_parameter_matches_single_calls():
    x = np.random.default_rng(100).standard_normal((4, 500, 37))
    out = np.stack(bulk_ess_rn_rhat(x))
    for p in range(x.shape[2]):
        np.testing.assert_array_equal(
            out[:, p], np.concatenate(bulk_ess_rn_rhat(x[:, :, p : p + 1]))
        )


@pytest.mark.parametrize("bandwidth", ["scott", "silverman", 0.4])
def test_marginal_kde_matches_scipy_gaussian_kde(bandwidth):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 10, 1))
    for p, (grid, density) in enumerate(marginal_kde(x, 10, bandwidth)):
        column = x[:, :, p].reshape(-1)
        linspace = np.linspace(column.min(), column.max(), 10)
        grid_atol = 4 * np.finfo(float).eps * np.abs(linspace).max()
        np.testing.assert_allclose(grid, linspace, rtol=0.0, atol=grid_atol)
        reference = gaussian_kde(column, bw_method=bandwidth)(grid)
        np.testing.assert_allclose(
            density, reference, rtol=0.0, atol=1e-12 * reference.max()
        )
