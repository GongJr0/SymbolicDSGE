"""Parity tests for the native packed log-prior kernels in ``_ckernels/estimation``.

The transform kernel is compared against the ``Transform`` the packed row was
built from, so one transform is checked against itself. The remaining kernels
are still compared against the compiled numba oracle in
``tests/_oracles/estimation``, bit-for-bit up to libm; the njit kernels are the
oracle there, not their ``.py_func`` forms, which raise on ``math.log`` domain
errors where numba and C return -inf/nan.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.estimation import (
    cov_from_unconstrained as native_cov_from_unconstrained,
    dist_logpdf as native_dist_logpdf,
    lkj_chol_logjac as native_lkj_logjac,
    lkj_chol_logpdf_from_z as native_lkj_logpdf,
    transform_inverse_and_logjac as native_transform,
    unconstrained_from_corr_chol as native_unconstrained_from_corr_chol,
)
from SymbolicDSGE.bayesian.distributions.distribution_dispatch import get_distribution
from SymbolicDSGE.bayesian.distributions.lkj_chol import _log_lkj_normalizer_C
from SymbolicDSGE.bayesian.distributions.param_builder import get_dist_params
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    AffineProbitTransform,
    Identity,
    LogitTransform,
    LogTransform,
    LowerBoundedTransform,
    ProbitTransform,
    SoftplusTransform,
    TanhTransform,
    UpperBoundedTransform,
)
from SymbolicDSGE.estimation.prior_program import (
    DistCode,
    N_DIST_PARAMS,
    N_TRANSFORM_PARAMS,
    TransformCode,
    _pack_distribution,
    _pack_transform,
)
from _oracles.estimation import (
    _corr_chol_from_unconstrained,
    _dist_logpdf,
    _lkj_chol_logjac,
    _lkj_chol_logpdf_from_z,
    _R_from_unconstrained,
    _unconstrained_from_corr_chol,
)

RTOL = 1e-12
ATOL = 1e-12


def _build_dist(family, params):
    """Distribution instance without the Prior wrapper -- these kernels pack and
    evaluate the distribution/transform directly, below the Prior abstraction, so
    they must be exercisable with (dist, transform) pairs that Prior's support
    check would reject (e.g. a distribution with an identity transform)."""
    return get_distribution(family)(**(get_dist_params(family) | params))


def _agree(native: float, numba: float) -> None:
    native = float(native)
    numba = float(numba)
    if np.isnan(native) or np.isnan(numba):
        assert np.isnan(native) and np.isnan(numba), (native, numba)
        return
    # assert_allclose treats matching infinities as equal.
    np.testing.assert_allclose(native, numba, rtol=RTOL, atol=ATOL)


_DIST_SPECS = {
    DistCode.NORMAL: ("normal", {"mean": 0.0, "std": 1.0, "random_state": 1}),
    DistCode.LOG_NORMAL: ("log_normal", {"mean": 0.0, "std": 0.5, "random_state": 1}),
    DistCode.HALF_NORMAL: ("half_normal", {"std": 1.0, "random_state": 1}),
    DistCode.TRUNC_NORMAL: (
        "trunc_normal",
        {"mean": 0.0, "std": 1.0, "low": -1.0, "high": 1.0, "random_state": 1},
    ),
    DistCode.HALF_CAUCHY: ("half_cauchy", {"gamma": 1.0, "random_state": 1}),
    DistCode.BETA: ("beta", {"a": 2.0, "b": 3.0, "random_state": 1}),
    DistCode.GAMMA: ("gamma", {"mean": 2.0, "std": 1.0, "random_state": 1}),
    DistCode.INV_GAMMA: ("inv_gamma", {"mean": 2.0, "std": 1.0, "random_state": 1}),
    DistCode.UNIFORM: ("uniform", {"low": -1.0, "high": 1.0, "random_state": 1}),
}

_X_SWEEP = (-2.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 3.0)


@pytest.mark.parametrize("code,spec", list(_DIST_SPECS.items()))
def test_dist_logpdf_parity(code, spec):
    family, params = spec
    # Only the distribution is packed/evaluated here (the x-sweep deliberately
    # probes out-of-support values), so build the dist directly.
    packed_code, packed_params = _pack_distribution(_build_dist(family, params))
    assert int(packed_code) == int(code)
    parr = np.asarray(packed_params, dtype=np.float64)

    for x in _X_SWEEP:
        numba = _dist_logpdf(int(code), parr, np.float64(x))
        native = native_dist_logpdf(int(code), parr, float(x))
        _agree(native, numba)


def test_dist_logpdf_unknown_code_is_nan():
    parr = np.zeros(N_DIST_PARAMS, dtype=np.float64)
    _agree(native_dist_logpdf(999, parr, 0.0), _dist_logpdf(999, parr, np.float64(0.0)))


#: One instance per scalar transform code, with bounds where the code reads them.
_TRANSFORMS = (
    Identity(),
    LogTransform(),
    SoftplusTransform(),
    LogitTransform(),
    ProbitTransform(),
    AffineLogitTransform(low=-1.0, high=2.0),
    AffineProbitTransform(low=-1.0, high=2.0),
    LowerBoundedTransform(low=-1.0),
    UpperBoundedTransform(high=2.0),
    TanhTransform(),
)


@pytest.mark.parametrize("transform", _TRANSFORMS, ids=lambda t: type(t).__name__)
def test_transform_parity(transform):
    """The kernel against the transform its row was packed from.

    The code and the params row both come off ``_pack_transform``, so the
    comparison is of one transform against itself rather than of two hand-kept
    dispatch tables that a renumbering can silently rotate apart.
    """
    code, params = _pack_transform(transform)
    params = np.asarray(params, dtype=np.float64)
    for z in (-3.0, -0.7, 0.0, 0.4, 1.3, 4.0):
        cx, cj = native_transform(int(code), params, float(z))
        _agree(cx, float(transform.inverse(np.float64(z))))
        _agree(cj, float(transform.log_det_abs_jacobian_inverse(np.float64(z))))


def test_every_scalar_transform_code_is_exercised():
    """The sweep is a list of instances, so a code added without one would
    otherwise go untested."""
    packed = {int(_pack_transform(t)[0]) for t in _TRANSFORMS}
    assert packed == {int(c) for c in TransformCode} - {
        int(TransformCode.CHOLESKY_CORR)
    }


def test_transform_block_code_is_nan():
    """A block's run is caught and skipped before the scalar dispatcher, so the
    block code has no scalar inverse to give."""
    params = np.zeros(N_TRANSFORM_PARAMS, dtype=np.float64)
    params[0] = 3.0
    cx, cj = native_transform(int(TransformCode.CHOLESKY_CORR), params, 0.4)
    assert np.isnan(float(cx)) and np.isnan(float(cj))


def test_transform_unknown_code_is_nan():
    params = np.zeros(N_TRANSFORM_PARAMS, dtype=np.float64)
    cx, cj = native_transform(999, params, 0.5)
    assert np.isnan(float(cx)) and np.isnan(float(cj))


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_lkj_parity(dim):
    rng = np.random.default_rng(dim)
    length = dim * (dim - 1) // 2
    z = np.ascontiguousarray(rng.normal(size=length), dtype=np.float64)

    _agree(native_lkj_logjac(z, dim, length), _lkj_chol_logjac(z, dim, length))

    for eta in (0.5, 1.0, 2.5):
        log_const = float(_log_lkj_normalizer_C(dim, eta))
        _agree(
            native_lkj_logpdf(z, dim, length, eta, log_const),
            _lkj_chol_logpdf_from_z(
                z, dim, length, np.float64(eta), np.float64(log_const)
            ),
        )


def test_lkj_length_too_short_is_nan(dim=4):
    z = np.array([0.1, 0.2], dtype=np.float64)
    _agree(native_lkj_logjac(z, dim, z.size), _lkj_chol_logjac(z, dim, z.size))
    _agree(
        native_lkj_logpdf(z, dim, z.size, 1.0, 0.0),
        _lkj_chol_logpdf_from_z(z, dim, z.size, np.float64(1.0), np.float64(0.0)),
    )


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_cov_from_unconstrained_parity(K):
    rng = np.random.default_rng(K)
    n_cpc = K * (K - 1) // 2
    z = rng.standard_normal(n_cpc)
    std = np.exp(0.3 * rng.standard_normal(K))

    cov, L = native_cov_from_unconstrained(z, std)

    # Oracle covariance + correlation Cholesky from u = [log std, z].
    u = np.concatenate([np.log(std), z])
    R_ref, _, Lcorr_ref = _R_from_unconstrained(u, K)
    L_ref = _corr_chol_from_unconstrained(np.ascontiguousarray(z), K)

    np.testing.assert_allclose(cov, R_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(L, L_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(L, Lcorr_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_unconstrained_from_corr_chol_parity(K):
    rng = np.random.default_rng(100 + K)
    n_cpc = K * (K - 1) // 2
    z = rng.standard_normal(n_cpc)
    L = _corr_chol_from_unconstrained(np.ascontiguousarray(z), K)

    z_native = native_unconstrained_from_corr_chol(L)
    z_ref = _unconstrained_from_corr_chol(np.ascontiguousarray(L))

    np.testing.assert_allclose(z_native, z_ref, rtol=RTOL, atol=ATOL)
    # Round-trip recovers the original unconstrained values.
    np.testing.assert_allclose(z_native, z, rtol=1e-10, atol=1e-10)
