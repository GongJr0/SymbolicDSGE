"""Drive a residual cfunc from Python with arrays.

``build_cfunc`` is the emitter the library ships. It takes raw pointers and
writes into a caller-owned buffer, which is what the native kernels want and
what a test asserting on emitted values cannot use directly. This wraps one
cfunc in the array signature those tests need, and holds a reference to it so
its ``address`` stays valid for as long as the caller is.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

import numpy as np
import sympy as sp

from SymbolicDSGE._symbolic_printers import ResidualLayout, build_cfunc
from SymbolicDSGE._symbolic_printers.base import OpTable
from SymbolicDSGE._symbolic_printers.residual_printer import C128Ops

_PTR = ctypes.POINTER(ctypes.c_double)


def residual_caller(
    exprs: Sequence[sp.Expr],
    layout: ResidualLayout,
    ops: OpTable | None = None,
) -> Any:
    """``(fwd, cur, prev, eps, par) -> out`` backed by the production cfunc.

    The compiled cfunc is reachable as ``.cfunc`` for callers that also need its
    address.
    """
    table: OpTable = C128Ops() if ops is None else ops
    cf = build_cfunc(list(exprs), layout, ops)
    n_out = table.elems_per_var * layout.n_expr

    def call(fwd: Any, cur: Any, prev: Any, eps: Any, par: Any) -> Any:
        # Bind the converted buffers to names before taking any pointer: a
        # temporary from the conversion would be freed while the call holds it.
        bufs = [
            np.ascontiguousarray(arr, dtype=np.complex128)
            for arr in (fwd, cur, prev, eps, par)
        ]
        out = np.empty(n_out, dtype=np.complex128)
        cf.ctypes(
            *(buf.ctypes.data_as(_PTR) for buf in bufs),
            out.ctypes.data_as(_PTR),
        )
        return out

    call.cfunc = cf  # type: ignore[attr-defined]
    return call
