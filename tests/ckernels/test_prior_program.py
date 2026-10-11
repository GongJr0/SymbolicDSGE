"""Parity tests for the native packed log-prior kernels in ``_ckernels/estimation``.

Each kernel is compared against the library object its packed row was built
from, so the comparison is of one definition against itself rather than of two
implementations kept in step by hand. Where the two differ by construction it
is stated as such: a distribution raises out of its support where the kernel
returns NaN, and the ``_from_z`` LKJ density carries the change of variables
that ``LKJChol.logpdf`` leaves to the transform.
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
from SymbolicDSGE.bayesian.distributions.lkj_chol import (
    LKJChol,
    _log_lkj_normalizer_C,
)
from SymbolicDSGE.bayesian.distributions.param_builder import get_dist_params
from SymbolicDSGE.bayesian.support import OutOfSupportError
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
from SymbolicDSGE.bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from SymbolicDSGE.estimation.backend import (
    _corr_chol_from_unconstrained,
    _unconstrained_from_corr_chol,
)
from SymbolicDSGE.estimation.prior_program import (
    DistCode,
    N_DIST_PARAMS,
    N_TRANSFORM_PARAMS,
    TransformCode,
    _pack_distribution,
    _pack_transform,
)

RTOL = 1e-12
ATOL = 1e-12


def _build_dist(family, params):
    """Distribution instance without the Prior wrapper -- these kernels pack and
    evaluate the distribution/transform directly, below the Prior abstraction, so
    they must be exercisable with (dist, transform) pairs that Prior's support
    check would reject (e.g. a distribution with an identity transform)."""
    return get_distribution(family)(**(get_dist_params(family) | params))


def _agree(native: float, expected: float) -> None:
    native = float(native)
    expected = float(expected)
    if np.isnan(native) or np.isnan(expected):
        assert np.isnan(native) and np.isnan(expected), (native, expected)
        return
    # assert_allclose treats matching infinities as equal.
    np.testing.assert_allclose(native, expected, rtol=RTOL, atol=ATOL)


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
    """The kernel against the distribution its row was packed from.

    The sweep deliberately leaves each support. The kernel reports that as NaN
    rather than raising, which is what lets the objective fold it into its -inf
    sentinel, so out of support the two agree by that correspondence rather
    than by value.
    """
    family, params = spec
    dist = _build_dist(family, params)
    packed_code, packed_params = _pack_distribution(dist)
    assert int(packed_code) == int(code)
    parr = np.asarray(packed_params, dtype=np.float64)

    for x in _X_SWEEP:
        native = native_dist_logpdf(int(code), parr, float(x))
        try:
            expected = dist.logpdf(np.float64(x))
        except OutOfSupportError:
            assert np.isnan(float(native)), (x, native)
        else:
            _agree(native, expected)


def test_dist_logpdf_unknown_code_is_nan():
    parr = np.zeros(N_DIST_PARAMS, dtype=np.float64)
    assert np.isnan(float(native_dist_logpdf(999, parr, 0.0)))


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
    """``LKJChol.logpdf`` is the density of the factor and the kernel's is the
    density of the unconstrained vector, so the two differ by the transform's
    jacobian and by nothing else."""
    rng = np.random.default_rng(dim)
    length = dim * (dim - 1) // 2
    z = np.ascontiguousarray(rng.normal(size=length), dtype=np.float64)

    logjac = float(CholeskyCorrTransform(K=dim).log_det_abs_jacobian_inverse(z))
    _agree(native_lkj_logjac(z, dim, length), logjac)

    L = _corr_chol_from_unconstrained(z, dim)
    for eta in (0.5, 1.0, 2.5):
        log_const = float(_log_lkj_normalizer_C(dim, eta))
        expected = float(LKJChol(eta=eta, K=dim, random_state=None).logpdf(L)) + logjac
        _agree(native_lkj_logpdf(z, dim, length, eta, log_const), expected)


def test_lkj_length_too_short_is_nan(dim=4):
    """A run shorter than the dimension demands is unreachable through the
    packer, so the kernel's own guard is the whole claim."""
    z = np.array([0.1, 0.2], dtype=np.float64)
    assert np.isnan(float(native_lkj_logjac(z, dim, z.size)))
    assert np.isnan(float(native_lkj_logpdf(z, dim, z.size, 1.0, 0.0)))


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_cov_from_unconstrained_parity(K):
    rng = np.random.default_rng(K)
    n_cpc = K * (K - 1) // 2
    z = rng.standard_normal(n_cpc)
    std = np.exp(0.3 * rng.standard_normal(K))

    cov, L = native_cov_from_unconstrained(z, std)

    L_ref = _corr_chol_from_unconstrained(np.ascontiguousarray(z), K)
    cov_ref = np.outer(std, std) * (L_ref @ L_ref.T)

    np.testing.assert_allclose(cov, cov_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(L, L_ref, rtol=RTOL, atol=ATOL)


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
