"""Parity + derivative tests for the native bicomplex (``bc256``) primitives.

``bc256`` (``_ckernels/_common/sdsge_bicomplex.h``) is the scalar-arithmetic
substrate for the second-order (bicomplex-step) perturbation preproc. The C ops
are ``static inline`` in the header and exposed to Python through the ``_core``
shim as ``bc_*`` on the 4-tuple ``(real, i, j, ij) = (a.re, a.im, b.re, b.im)``.

Two independent checks:

1. **Algebra** -- every op against a NumPy pair-of-complex reference (a different
   implementation than the C: ``exp`` uses the direct ``e^{z1}(cos z2 + j sin z2)``
   formula vs the header's idempotent projection, so agreement is cross-algorithm,
   not tautological). Random inputs carry nonzero i/j/ij so a projection/reconst
   sign error cannot hide.
2. **Derivatives** -- the actual point: perturb along i and j, read the ij
   component / h^2, compare to the analytic second derivative. Representation-
   independent end-to-end validation.

Numerical note: the bicomplex second derivative has O(h^2) *truncation* (unlike
complex-step's exact first derivative -- this is the bicomplex-vs-hyperdual
tradeoff), so derivative tests use a moderate ``h`` and matched tolerances.
Polynomials up to cubic are exact at any ``h`` (no f'''' term).
"""

from __future__ import annotations

import cmath

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core._core import (
    bc_add,
    bc_conj,
    bc_cpow,
    bc_div,
    bc_exp,
    bc_i_conj,
    bc_ipow,
    bc_j_conj,
    bc_log,
    bc_mul,
    bc_neg,
    bc_proj,
    bc_real_scale,
    bc_reconst,
    bc_spow,
    bc_sqrt,
    bc_sub,
    c_sqrt,
)

RTOL = 1e-12
ATOL = 1e-12

BC = tuple  # (real, i, j, ij)


# --- NumPy pair-of-complex reference ---------------------------------------
# A bicomplex value maps to (za, zb) = (real + i*<i>, <j> + i*<ij>), value za + zb*j.


def _to_pair(x: BC) -> tuple[complex, complex]:
    return complex(x[0], x[1]), complex(x[2], x[3])


def _from_pair(za: complex, zb: complex) -> BC:
    return (za.real, za.imag, zb.real, zb.imag)


def _ref_mul(x: BC, y: BC) -> BC:
    xa, xb = _to_pair(x)
    ya, yb = _to_pair(y)
    return _from_pair(xa * ya - xb * yb, xa * yb + xb * ya)


def _ref_div(x: BC, y: BC) -> BC:
    xa, xb = _to_pair(x)
    ya, yb = _to_pair(y)
    denom = ya * ya + yb * yb
    return _from_pair((xa * ya + xb * yb) / denom, (xb * ya - xa * yb) / denom)


def _ref_exp(x: BC) -> BC:
    # Direct formula (independent of the header's idempotent projection):
    # exp(za + zb j) = exp(za) * (cos zb + j sin zb).
    za, zb = _to_pair(x)
    e = cmath.exp(za)
    return _from_pair(e * cmath.cos(zb), e * cmath.sin(zb))


def _rand_bc(rng: np.random.Generator) -> BC:
    # Nonzero in every component so sign errors in i/j/ij surface.
    mag = rng.uniform(0.5, 2.0, size=4)
    sign = rng.choice((-1.0, 1.0), size=4)
    return tuple(float(m * s) for m, s in zip(mag, sign))


# --- Algebra ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_ring_ops_match_reference(seed):
    rng = np.random.default_rng(seed)
    x, y = _rand_bc(rng), _rand_bc(rng)

    np.testing.assert_allclose(bc_add(x, y), np.add(x, y), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(bc_sub(x, y), np.subtract(x, y), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(bc_neg(x), np.negative(x), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(bc_mul(x, y), _ref_mul(x, y), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(bc_div(x, y), _ref_div(x, y), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        bc_real_scale(x, 3.25), np.multiply(x, 3.25), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("seed", range(8))
def test_div_inverts_mul(seed):
    rng = np.random.default_rng(seed)
    q, y = _rand_bc(rng), _rand_bc(rng)
    np.testing.assert_allclose(bc_div(bc_mul(q, y), y), q, rtol=1e-10, atol=1e-12)


def test_conjugations_flip_expected_components():
    x = (1.0, 2.0, 3.0, 4.0)
    assert bc_i_conj(x) == (1.0, -2.0, 3.0, -4.0)  # i -> -i  (flips i, ij)
    assert bc_j_conj(x) == (1.0, 2.0, -3.0, -4.0)  # j -> -j  (flips j, ij)
    assert bc_conj(x) == (1.0, -2.0, -3.0, 4.0)  # both      (flips i, j)


@pytest.mark.parametrize("seed", range(8))
def test_proj_reconst_roundtrip(seed):
    rng = np.random.default_rng(seed)
    x = _rand_bc(rng)
    # reconst must be the exact inverse of proj (guards the reconst sign).
    np.testing.assert_allclose(bc_reconst(bc_proj(x)), x, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", range(8))
def test_proj_matches_reference(seed):
    rng = np.random.default_rng(seed)
    x = _rand_bc(rng)
    za, zb = _to_pair(x)
    p1_ref, p2_ref = za - 1j * zb, za + 1j * zb
    p1re, p1im, p2re, p2im = bc_proj(x)
    np.testing.assert_allclose(
        (p1re, p1im, p2re, p2im),
        (p1_ref.real, p1_ref.imag, p2_ref.real, p2_ref.imag),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(8))
def test_exp_matches_direct_formula(seed):
    rng = np.random.default_rng(seed)
    # Modest magnitude so the idempotent projection stays well conditioned.
    x = tuple(float(v) for v in rng.uniform(-1.0, 1.0, size=4))
    np.testing.assert_allclose(bc_exp(x), _ref_exp(x), rtol=1e-9, atol=1e-12)


def test_exp_of_real_matches_scalar():
    x = (0.7, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(bc_exp(x), (np.exp(0.7), 0.0, 0.0, 0.0), atol=1e-14)


@pytest.mark.parametrize("seed", range(8))
def test_exp_log_inverse(seed):
    rng = np.random.default_rng(seed)
    # Positive-dominant real part + small other parts -> principal branch safe.
    x = (float(rng.uniform(0.5, 2.0)), *rng.uniform(-0.2, 0.2, size=3))
    np.testing.assert_allclose(bc_exp(bc_log(x)), x, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_spow_consistency_with_exp_log(seed):
    rng = np.random.default_rng(seed)
    x = (float(rng.uniform(0.5, 2.0)), *rng.uniform(-0.2, 0.2, size=3))
    p = float(rng.uniform(0.3, 3.0))
    # x^p == exp(p * log x)
    np.testing.assert_allclose(
        bc_spow(x, p), bc_exp(bc_real_scale(bc_log(x), p)), rtol=1e-10, atol=1e-12
    )


def test_spow_integer_matches_repeated_mul():
    x = (1.3, 0.4, -0.2, 0.1)
    x2 = bc_mul(x, x)
    x3 = bc_mul(x2, x)
    np.testing.assert_allclose(bc_spow(x, 2.0), x2, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(bc_spow(x, 3.0), x3, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8])
def test_ipow_matches_repeated_mul(n):
    # Log-free integer power: exact regardless of base sign (here a negative base).
    x = (-0.7, 0.4, -0.2, 0.1)
    expect = (1.0, 0.0, 0.0, 0.0)
    for _ in range(n):
        expect = bc_mul(expect, x)
    np.testing.assert_allclose(bc_ipow(x, n), expect, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_ipow_negative_is_reciprocal(n):
    # x^(-n) == 1 / x^n, handled in the kernel so the printer need not special-case.
    x = (-0.7, 0.4, -0.2, 0.1)
    one = (1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        bc_ipow(x, -n), bc_div(one, bc_ipow(x, n)), rtol=1e-11, atol=1e-13
    )


@pytest.mark.parametrize("seed", range(6))
def test_c_sqrt_matches_cmath(seed):
    rng = np.random.default_rng(seed)
    z = complex(*rng.uniform(-2.0, 2.0, size=2))
    r = c_sqrt((z.real, z.imag))
    ref = cmath.sqrt(z)
    np.testing.assert_allclose(r, (ref.real, ref.imag), rtol=1e-13, atol=1e-14)


def test_c_sqrt_of_negative_real_is_imaginary():
    np.testing.assert_allclose(c_sqrt((-4.0, 0.0)), (0.0, 2.0), atol=1e-14)
    np.testing.assert_allclose(c_sqrt((0.0, 0.0)), (0.0, 0.0), atol=1e-15)


@pytest.mark.parametrize("seed", range(6))
def test_bc_sqrt_squares_back(seed):
    rng = np.random.default_rng(seed)
    # Positive-dominant real part (the domain a real sqrt lives in).
    x = (float(rng.uniform(0.5, 3.0)), *rng.uniform(-0.3, 0.3, size=3))
    np.testing.assert_allclose(
        bc_mul(bc_sqrt(x), bc_sqrt(x)), x, rtol=1e-11, atol=1e-13
    )


def test_bc_sqrt_matches_spow_half():
    # Direct sqrt vs the exp/log half-power on a positive base -- both valid there.
    x = (1.7, 0.2, -0.1, 0.05)
    np.testing.assert_allclose(bc_sqrt(x), bc_spow(x, 0.5), rtol=1e-10, atol=1e-12)


def test_second_derivative_sqrt_is_cancellation_free():
    # f(x) = sqrt(x),  f''(x) = -1/4 * x^(-3/2). The direct formula keeps the ij
    # component clean, so this holds even at very small h (unlike idempotent spow).
    x0, h = 1.4, 1e-100
    f = bc_sqrt((x0, h, h, 0.0))
    analytic = -0.25 * x0 ** (-1.5)
    assert f[3] / h**2 == pytest.approx(analytic, rel=1e-8)


@pytest.mark.parametrize("seed", range(6))
def test_cpow_consistency_with_exp_log(seed):
    rng = np.random.default_rng(seed)
    x = (float(rng.uniform(0.5, 2.0)), *rng.uniform(-0.2, 0.2, size=3))
    y = tuple(float(v) for v in rng.uniform(-0.5, 0.5, size=4))
    # x^y == exp(y * log x)
    np.testing.assert_allclose(
        bc_cpow(x, y), bc_exp(bc_mul(y, bc_log(x))), rtol=1e-10, atol=1e-12
    )


# --- Second derivatives (the actual purpose) --------------------------------
# Diagonal: perturb one variable along both i and j -> ij component / h^2 = f''.
# Mixed: perturb var a along i, var b along j -> ij / h^2 = d^2/da db.


def _ij(x: BC) -> float:
    return x[3]


def _i(x: BC) -> float:
    return x[1]


def test_second_derivative_polynomial_is_exact():
    # f(x) = x^3 + 2 x^2  ->  f''(x) = 6x + 4.
    # The ij component's next Taylor term is f''''*h^4 = 0 for a cubic, so the
    # second derivative is exact at any h. (The i component is NOT exact under a
    # diagonal i+j perturbation -- it carries an f'''*h^2 term; see the dedicated
    # complex-step test below for f'.)
    x0, h = 1.5, 1e-2
    bx = (x0, h, h, 0.0)
    x2 = bc_mul(bx, bx)
    x3 = bc_mul(x2, bx)
    f = bc_add(x3, bc_real_scale(x2, 2.0))
    assert _ij(f) / h**2 == pytest.approx(6 * x0 + 4, rel=1e-9)


def test_first_derivative_complex_step_is_exact():
    # A pure-i perturbation reduces to ordinary complex step in the (1, i) plane;
    # with tiny h the O(h^2) truncation vanishes. f(x) = x^3 + 2x^2, f' = 3x^2 + 4x.
    x0, h = 1.5, 1e-100
    bx = (x0, h, 0.0, 0.0)
    x2 = bc_mul(bx, bx)
    x3 = bc_mul(x2, bx)
    f = bc_add(x3, bc_real_scale(x2, 2.0))
    assert _i(f) / h == pytest.approx(3 * x0**2 + 4 * x0, rel=1e-12)


def test_mixed_partial_is_exact():
    # f(a, b) = a * b  ->  d^2 f / da db = 1  (bilinear, exact).
    a0, b0, h = 1.3, -0.7, 1e-2
    ba = (a0, h, 0.0, 0.0)  # perturb a along i
    bb = (b0, 0.0, h, 0.0)  # perturb b along j
    f = bc_mul(ba, bb)
    assert _ij(f) / h**2 == pytest.approx(1.0, abs=1e-10)


def test_second_derivative_rational():
    # f(x) = 1 / (1 + x^2),  f''(x) = (6x^2 - 2) / (1 + x^2)^3.
    x0, h = 0.8, 1e-4
    bx = (x0, h, h, 0.0)
    one = (1.0, 0.0, 0.0, 0.0)
    f = bc_div(one, bc_add(one, bc_mul(bx, bx)))
    analytic = (6 * x0**2 - 2) / (1 + x0**2) ** 3
    assert _ij(f) / h**2 == pytest.approx(analytic, rel=1e-6)


def test_second_derivative_exp():
    # f(x) = exp(x),  f''(x) = exp(x). Transcendental path (idempotent exp),
    # so moderate h and looser tolerance (truncation + projection conditioning).
    x0, h = 0.5, 1e-4
    f = bc_exp((x0, h, h, 0.0))
    assert _ij(f) / h**2 == pytest.approx(np.exp(x0), rel=1e-5)


def test_second_derivative_power():
    # f(x) = x^2.5,  f''(x) = 2.5 * 1.5 * x^0.5.
    x0, h = 1.4, 1e-4
    f = bc_spow((x0, h, h, 0.0), 2.5)
    analytic = 2.5 * 1.5 * x0**0.5
    assert _ij(f) / h**2 == pytest.approx(analytic, rel=1e-5)
