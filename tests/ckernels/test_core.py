"""Parity tests for the native core kernels vs the numba reference.

Skips when the ``_ckernels.core`` extension is not built (e.g. local dev without
a compiler); runs in CI where wheels are compiled.
"""

import numpy as np
import pytest

from _oracles.core import (
    _affine_observations_into_numba,
    _simulate_linear_states_into_numba,
    _simulate_second_order_pruned_numba,
)

core = pytest.importorskip("SymbolicDSGE._ckernels.core")

# Cross-compiler FP (FMA contraction, instruction selection) can differ in the
# last ULP between the C build and numba/LLVM, so compare tightly, not exactly.
_RTOL = 1e-12
_ATOL = 1e-12


@pytest.mark.parametrize("n, k, T", [(1, 1, 1), (3, 2, 25), (5, 5, 64), (4, 1, 9)])
@pytest.mark.parametrize("levels", [False, True])
def test_simulate_matches_numba(n: int, k: int, T: int, levels: bool) -> None:
    rng = np.random.default_rng(n * 100 + k * 10 + T)
    A = np.ascontiguousarray(rng.standard_normal((n, n)) * (0.5 / n))
    B = np.ascontiguousarray(rng.standard_normal((n, k)))
    x0 = np.ascontiguousarray(rng.standard_normal(n))
    shock = np.ascontiguousarray(rng.standard_normal((T, k)))
    # A nonzero ss is what separates a fused add from one folded into the
    # recursion: pass it back through A and the path diverges.
    ss = np.ascontiguousarray(rng.standard_normal(n)) if levels else np.zeros(n)

    native = np.empty((T, n))
    ref = np.empty((T, n))
    core.simulate_linear_states_into(A, B, x0, shock, native, ss if levels else None)
    _simulate_linear_states_into_numba(A, B, x0, shock, ref, ss)

    np.testing.assert_allclose(native, ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n, k, T", [(3, 2, 12)])
def test_simulate_levels_is_deviations_plus_ss(n: int, k: int, T: int) -> None:
    """The steady state offsets the rows and nothing else: it must not reach the
    recursion, which would only be invisible if ss were a fixed point of A."""
    rng = np.random.default_rng(7)
    A = np.ascontiguousarray(rng.standard_normal((n, n)) * (0.5 / n))
    B = np.ascontiguousarray(rng.standard_normal((n, k)))
    x0 = np.ascontiguousarray(rng.standard_normal(n))
    shock = np.ascontiguousarray(rng.standard_normal((T, k)))
    ss = np.ascontiguousarray(rng.standard_normal(n))

    dev = np.empty((T, n))
    lvl = np.empty((T, n))
    core.simulate_linear_states_into(A, B, x0, shock, dev)
    core.simulate_linear_states_into(A, B, x0, shock, lvl, ss)

    np.testing.assert_allclose(lvl, dev + ss, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("m, n, T", [(1, 2, 1), (3, 4, 20), (2, 5, 12)])
def test_affine_matches_numba(m: int, n: int, T: int) -> None:
    rng = np.random.default_rng(m * 1000 + n * 100 + T * 10)
    states = np.ascontiguousarray(rng.standard_normal((T, n)))
    C = np.ascontiguousarray(rng.standard_normal((m, n)))
    d = np.ascontiguousarray(rng.standard_normal(m))

    native = np.empty((T, m))
    ref = np.empty((T, m))
    core.affine_observations_into(states, C, d, native)
    _affine_observations_into_numba(states, C, d, ref)

    np.testing.assert_allclose(native, ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "nx, ny, n_exog, T",
    [(1, 1, 1, 1), (2, 1, 1, 4), (3, 2, 2, 7), (2, 0, 1, 5), (2, 1, 0, 5)],
)
def test_second_order_pruned_matches_numba(
    nx: int, ny: int, n_exog: int, T: int
) -> None:
    rng = np.random.default_rng(nx * 1000 + ny * 100 + n_exog * 10 + T)
    n = nx + ny
    hx = np.ascontiguousarray(rng.standard_normal((nx, nx)) * (0.25 / nx))
    gx = np.ascontiguousarray(rng.standard_normal((ny, nx)))
    bu = np.ascontiguousarray(rng.standard_normal((n, n_exog)))
    hxx = np.ascontiguousarray(rng.standard_normal((nx, nx, nx)) * 0.05)
    gxx = np.ascontiguousarray(rng.standard_normal((ny, nx, nx)) * 0.05)
    hxu = np.ascontiguousarray(rng.standard_normal((nx, nx, n_exog)) * 0.05)
    gxu = np.ascontiguousarray(rng.standard_normal((ny, nx, n_exog)) * 0.05)
    huu = np.ascontiguousarray(rng.standard_normal((nx, n_exog, n_exog)) * 0.05)
    guu = np.ascontiguousarray(rng.standard_normal((ny, n_exog, n_exog)) * 0.05)
    hss = np.ascontiguousarray(rng.standard_normal(nx) * 0.01)
    gss = np.ascontiguousarray(rng.standard_normal(ny) * 0.01)
    x0 = np.ascontiguousarray(rng.standard_normal(nx) * 0.1)
    shock = np.ascontiguousarray(rng.standard_normal((T, n_exog)) * 0.1)
    ss = np.ascontiguousarray(rng.standard_normal(n))

    args = (hx, gx, bu, hxx, gxx, hxu, gxu, huu, guu, hss, gss, x0, shock)
    native = core.simulate_second_order_pruned(*args, ss)
    numba = _simulate_second_order_pruned_numba(*args, ss)
    np.testing.assert_allclose(native, numba, rtol=_RTOL, atol=_ATOL)

    # Deviations are the same path without the offset.
    dev = core.simulate_second_order_pruned(*args)
    np.testing.assert_allclose(native, dev + ss, rtol=_RTOL, atol=_ATOL)


def test_simulate_known_answer() -> None:
    # n=1, k=1: A=2, B=1, x0=1, shock=[0.5, 0.5] -> out = [2.5, 5.5].
    A = np.array([[2.0]])
    B = np.array([[1.0]])
    x0 = np.array([1.0])
    shock = np.array([[0.5], [0.5]])
    out = np.empty((2, 1))
    core.simulate_linear_states_into(A, B, x0, shock, out)
    np.testing.assert_allclose(out.ravel(), [2.5, 5.5])

    # ss = 10 offsets both rows; it must not compound through A.
    core.simulate_linear_states_into(A, B, x0, shock, out, np.array([10.0]))
    np.testing.assert_allclose(out.ravel(), [12.5, 15.5])


def test_affine_known_answer() -> None:
    # m=1, n=2: C=[[1,2]], d=[3], states=[[1,1],[2,2]] -> out = [6, 9].
    states = np.array([[1.0, 1.0], [2.0, 2.0]])
    C = np.array([[1.0, 2.0]])
    d = np.array([3.0])
    out = np.empty((2, 1))
    core.affine_observations_into(states, C, d, out)
    np.testing.assert_allclose(out.ravel(), [6.0, 9.0])
