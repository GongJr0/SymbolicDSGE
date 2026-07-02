"""Portability guard (#248): can native code call a numba @cfunc on this platform?

The cfunc-leaf + native-driver architecture rests on one assumption: a numba
``@cfunc`` can be invoked from a hand-written C translation unit through its raw
``.address``, with the GIL released and without numba's runtime (NRT) being
reachable inside the call. This proves it end-to-end:

    numba @cfunc  --.address-->  Cython (cast + `nogil`)  -->  spike_call (C)  --> back

The cfunc is allocation-free (``carray`` views over the caller's buffers, scalar
complex arithmetic only), which is the invariant that keeps it NRT-free. It runs
across the wheel platform matrix -- on demand via the ``cfunc ABI`` workflow
(.github/workflows/cfunc-abi.yml) and on each wheel build via cibuildwheel's
tests/ckernels run. A failure on any platform means the preproc leaf cannot ship
there.
"""

from __future__ import annotations

import threading

import numpy as np
from numba import carray, cfunc, types

from SymbolicDSGE._ckernels.core._core import spike_drive

# ABI: void residual(const c128* a, const c128* b, c128* out, int64 n).
_SIG = types.void(
    types.CPointer(types.complex128),
    types.CPointer(types.complex128),
    types.CPointer(types.complex128),
    types.int64,
)


@cfunc(_SIG)
def _residual(a_ptr, b_ptr, out_ptr, n):  # pragma: no cover - runs as native code
    a = carray(a_ptr, n)
    b = carray(b_ptr, n)
    out = carray(out_ptr, n)
    for i in range(n):
        out[i] = a[i] * a[i] + b[i]


def _rand_c128(rng, n):
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)


def test_native_invokes_numba_cfunc():
    rng = np.random.default_rng(0)
    a = _rand_c128(rng, 6)
    b = _rand_c128(rng, 6)
    out = np.zeros(6, dtype=np.complex128)

    spike_drive(_residual.address, a, b, out)

    np.testing.assert_allclose(out, a * a + b, rtol=1e-13, atol=1e-15)


def test_complex_step_derivative_through_native_path():
    # residual = a^2 + b; perturbing a = x + i*h and reading imag(out)/h gives
    # d/dx(x^2) = 2x -- computed inside the cfunc, extracted after the native call.
    # A mini-preview of the first-order preproc the real driver will run.
    x = np.array([1.5, -0.7, 2.3, 0.0], dtype=np.float64)
    h = 1e-100
    a = (x + 1j * h).astype(np.complex128)
    b = np.zeros_like(a)
    out = np.zeros_like(a)

    spike_drive(_residual.address, a, b, out)

    np.testing.assert_allclose(out.imag / h, 2.0 * x, rtol=1e-12, atol=1e-12)


def test_cfunc_is_reentrant_and_gil_free():
    # 8 threads each drive the native->cfunc path concurrently (each releasing the
    # GIL in spike_drive). Correct results with no deadlock demonstrate the cfunc
    # is GIL-free and reentrant -- the property that later lets the Hessian sweep
    # parallelize.
    n = 4000
    ok: dict[int, bool] = {}

    def worker(k: int) -> None:
        rng = np.random.default_rng(k)
        a = _rand_c128(rng, n)
        b = _rand_c128(rng, n)
        out = np.zeros(n, dtype=np.complex128)
        spike_drive(_residual.address, a, b, out)
        ok[k] = bool(np.allclose(out, a * a + b, rtol=1e-13, atol=1e-15))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ok) == 8 and all(ok.values())
