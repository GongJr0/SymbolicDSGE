"""Parity tests: native ``_ckernels.transforms`` kernels vs the independent
numpy/scipy oracle in ``tests/_oracles/transforms``.

Each of the seven maps (fwd, inv, grad_fwd, grad_inv, ldet_abs_jac_fwd/inv,
grad_ldet_abs_jac_inv) is checked for every transform (log, logit, probit) on
both the vectorized path and the scalar path. The scalar path additionally
asserts the kernel returns ``np.float64`` (not a bare Python ``float``): the
distributions dispatch on ``isinstance(x, float64)`` misroutes a Python float
into the vectorized njit kernel, so a bare-float regression here breaks callers
silently. probit rides on AS 241 vs scipy's inverse-normal, so the tolerance is
relative rather than bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest

native = pytest.importorskip("SymbolicDSGE._ckernels.transforms")

from _oracles import transforms as oracle

RTOL = 1e-12
ATOL = 1e-12

# Per-transform input domains. ``fwd``-family maps consume the transform's
# constrained support; ``inv``-family maps consume the unconstrained real line.
_LOG_FWD_DOM = np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
_LOG_INV_DOM = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
_UNIT_DOM = np.array([1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-4])
_REAL_DOM = np.array([-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0])
_POS_DOM = np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 10.0])
_CORR_DOM = np.array([-1.0 + 1e-4, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1.0 - 1e-4])

_FWD_FAMILY = ("{n}_fwd", "{n}_grad_fwd", "{n}_ldet_abs_jac_fwd")
_INV_FAMILY = (
    "{n}_inv",
    "{n}_grad_inv",
    "{n}_ldet_abs_jac_inv",
    "{n}_grad_ldet_abs_jac_inv",
)

_CASES: list[tuple[str, np.ndarray]] = []
for _name, _fwd_dom, _inv_dom in (
    ("log", _LOG_FWD_DOM, _LOG_INV_DOM),
    ("logit", _UNIT_DOM, _REAL_DOM),
    ("probit", _UNIT_DOM, _REAL_DOM),
    ("softplus", _POS_DOM, _REAL_DOM),
    ("tanh", _CORR_DOM, _REAL_DOM),
):
    _CASES += [(t.format(n=_name), _fwd_dom) for t in _FWD_FAMILY]
    _CASES += [(t.format(n=_name), _inv_dom) for t in _INV_FAMILY]


@pytest.mark.parametrize("fn_name, grid", _CASES, ids=[c[0] for c in _CASES])
def test_transform_kernel_parity(fn_name: str, grid: np.ndarray) -> None:
    native_fn = getattr(native, fn_name)
    oracle_fn = getattr(oracle, fn_name)

    # vectorized path
    got = np.asarray(native_fn(grid), dtype=float)
    exp = np.asarray(oracle_fn(grid), dtype=float)
    np.testing.assert_allclose(got, exp, rtol=RTOL, atol=ATOL)

    # scalar path + dtype fidelity (must be np.float64, never a bare float)
    for v in grid:
        s = native_fn(np.float64(v))
        assert isinstance(
            s, np.float64
        ), f"{fn_name}({v!r}) returned {type(s).__name__}, expected numpy.float64"
        np.testing.assert_allclose(
            float(s), float(np.asarray(oracle_fn(np.float64(v)))), rtol=RTOL, atol=ATOL
        )


def test_empty_array_roundtrips_without_calling_kernel() -> None:
    # The n == 0 guard returns the empty allocation without dereferencing &v[0].
    for fn_name, _ in _CASES:
        out = getattr(native, fn_name)(np.array([], dtype=np.float64))
        assert isinstance(out, np.ndarray)
        assert out.shape == (0,)


# The affine maps carry a (low, high) pair, so they need their own case list and
# argument plumbing. grad_ldet_abs_jac_inv drops the pair (the log-span term
# differentiates away), so it is called y-only like the plain transforms.
_AFF_LOW, _AFF_HIGH = -2.0, 3.0
_AFF_FWD_DOM = np.array([-2.0 + 1e-4, -1.5, -0.5, 0.5, 1.5, 2.5, 3.0 - 1e-4])

_AFF_FAMILY = (
    ("{n}_fwd", _AFF_FWD_DOM, True),
    ("{n}_grad_fwd", _AFF_FWD_DOM, True),
    ("{n}_ldet_abs_jac_fwd", _AFF_FWD_DOM, True),
    ("{n}_inv", _REAL_DOM, True),
    ("{n}_grad_inv", _REAL_DOM, True),
    ("{n}_ldet_abs_jac_inv", _REAL_DOM, True),
    ("{n}_grad_ldet_abs_jac_inv", _REAL_DOM, False),
)

_AFF_CASES: list[tuple[str, np.ndarray, bool]] = [
    (t.format(n=_name), _dom, _bounds)
    for _name in ("aff_logit", "aff_probit")
    for (t, _dom, _bounds) in _AFF_FAMILY
]


@pytest.mark.parametrize(
    "fn_name, grid, needs_bounds", _AFF_CASES, ids=[c[0] for c in _AFF_CASES]
)
def test_affine_kernel_parity(
    fn_name: str, grid: np.ndarray, needs_bounds: bool
) -> None:
    native_fn = getattr(native, fn_name)
    oracle_fn = getattr(oracle, fn_name)
    args = (_AFF_LOW, _AFF_HIGH) if needs_bounds else ()

    # vectorized path
    got = np.asarray(native_fn(grid, *args), dtype=float)
    exp = np.asarray(oracle_fn(grid, *args), dtype=float)
    np.testing.assert_allclose(got, exp, rtol=RTOL, atol=ATOL)

    # empty-array guard
    empty = native_fn(np.array([], dtype=np.float64), *args)
    assert isinstance(empty, np.ndarray) and empty.shape == (0,)

    # scalar path + dtype fidelity (must be np.float64, never a bare float)
    for v in grid:
        s = native_fn(np.float64(v), *args)
        assert isinstance(
            s, np.float64
        ), f"{fn_name}({v!r}) returned {type(s).__name__}, expected numpy.float64"
        np.testing.assert_allclose(
            float(s),
            float(np.asarray(oracle_fn(np.float64(v), *args))),
            rtol=RTOL,
            atol=ATOL,
        )


# lower/upper carry a single bound, but only fwd/inv/grad_fwd/ldj_fwd use it;
# the inverse-side grad/ldj are bound-independent (exp(y)/y/1), so they are
# called y-only. bound is None for those, a scalar for the rest.
_LOWER_LOW = -1.0
_UPPER_HIGH = 1.0
_LOWER_FWD_DOM = np.array([-1.0 + 1e-3, -0.5, 0.0, 1.0, 5.0, 20.0])  # in (low, inf)
_UPPER_FWD_DOM = np.array([-20.0, -5.0, -1.0, 0.0, 0.5, 1.0 - 1e-3])  # in (-inf, high)

_BOUND_CASES: list[tuple[str, np.ndarray, object]] = []
for _prefix, _bnd, _fwd_dom in (
    ("lower", _LOWER_LOW, _LOWER_FWD_DOM),
    ("upper", _UPPER_HIGH, _UPPER_FWD_DOM),
):
    _BOUND_CASES += [
        (f"{_prefix}_fwd", _fwd_dom, _bnd),
        (f"{_prefix}_grad_fwd", _fwd_dom, _bnd),
        (f"{_prefix}_ldet_abs_jac_fwd", _fwd_dom, _bnd),
        (f"{_prefix}_inv", _REAL_DOM, _bnd),
        (f"{_prefix}_grad_inv", _REAL_DOM, None),
        (f"{_prefix}_ldet_abs_jac_inv", _REAL_DOM, None),
        (f"{_prefix}_grad_ldet_abs_jac_inv", _REAL_DOM, None),
    ]


@pytest.mark.parametrize(
    "fn_name, grid, bound", _BOUND_CASES, ids=[c[0] for c in _BOUND_CASES]
)
def test_bounded_kernel_parity(fn_name: str, grid: np.ndarray, bound: object) -> None:
    native_fn = getattr(native, fn_name)
    oracle_fn = getattr(oracle, fn_name)
    args = () if bound is None else (bound,)

    # vectorized path
    got = np.asarray(native_fn(grid, *args), dtype=float)
    exp = np.asarray(oracle_fn(grid, *args), dtype=float)
    np.testing.assert_allclose(got, exp, rtol=RTOL, atol=ATOL)

    # empty-array guard
    empty = native_fn(np.array([], dtype=np.float64), *args)
    assert isinstance(empty, np.ndarray) and empty.shape == (0,)

    # scalar path + dtype fidelity (must be np.float64, never a bare float)
    for v in grid:
        s = native_fn(np.float64(v), *args)
        assert isinstance(
            s, np.float64
        ), f"{fn_name}({v!r}) returned {type(s).__name__}, expected numpy.float64"
        np.testing.assert_allclose(
            float(s),
            float(np.asarray(oracle_fn(np.float64(v), *args))),
            rtol=RTOL,
            atol=ATOL,
        )
