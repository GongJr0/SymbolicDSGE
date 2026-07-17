"""Parity tests: native ``_ckernels.distributions`` kernels vs the numba oracle.

The native C in ``_ckernels/_common/as241.c`` must match the independent numba
reference in ``tests/_oracles/distributions`` across every branch of Wichura's
AS 241 (central region, intermediate tail, far tail, and the p<=0 / p>=1
boundaries). The compiled njit oracle is the reference, not its ``.py_func``.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.distributions import (
    erfinv_from_as241 as native_erfinv,
    ndtri_as241 as native_ndtri,
    ndtri_as241_into as native_ndtri_into,
)
from _oracles.distributions import (
    erfinv_from_as241 as oracle_erfinv,
    ndtri_as241 as oracle_ndtri,
)

# One representative from each ndtri branch:
#   central     |p - 0.5| small           -> |q| <= 0.425
#   intermediate sqrt(-log(tail)) <= 5     -> C/D rational
#   far tail     sqrt(-log(tail)) > 5      -> E/F rational (needs tail < ~1.4e-11)
# plus the q<0 / q>=0 sign split.
_NDTRI_POINTS = [
    0.5,
    0.30,
    0.70,
    0.425,
    0.575,
    0.05,
    0.95,
    1e-3,
    1.0 - 1e-3,
    1e-6,
    1e-12,
    1.0 - 1e-12,
]


@pytest.mark.parametrize("p", _NDTRI_POINTS)
def test_ndtri_matches_oracle(p):
    got = native_ndtri(p)
    ref = float(oracle_ndtri(np.float64(p)))
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("p", [0.0, -1.0, -1e-300])
def test_ndtri_lower_boundary_is_neg_inf(p):
    assert native_ndtri(p) == float(oracle_ndtri(np.float64(p))) == -np.inf


@pytest.mark.parametrize("p", [1.0, 2.0, 1e300])
def test_ndtri_upper_boundary_is_pos_inf(p):
    assert native_ndtri(p) == float(oracle_ndtri(np.float64(p))) == np.inf


def test_ndtri_into_matches_scalar_and_preserves_shape():
    # The vectorized kernel must agree elementwise with the scalar native path
    # and return an array shaped like the (2-D, non-contiguous) input.
    grid = np.asarray(_NDTRI_POINTS, dtype=np.float64).reshape(2, 6)
    got = native_ndtri_into(grid[::-1])  # non-contiguous view forces the copy
    expected = np.array([[native_ndtri(v) for v in row] for row in grid[::-1]])
    assert got.shape == (2, 6)
    np.testing.assert_array_equal(got, expected)


def test_ndtri_into_empty():
    out = native_ndtri_into(np.empty(0, dtype=np.float64))
    assert out.shape == (0,)


@pytest.mark.parametrize("y", [-0.999, -0.5, 0.0, 0.25, 0.5, 0.9, 0.999])
def test_erfinv_matches_oracle(y):
    got = native_erfinv(y)
    ref = float(oracle_erfinv(np.float64(y)))
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)
