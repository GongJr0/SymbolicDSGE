"""Measurement expression printers for native filter callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp
from numba import cfunc, types
from sympy import Symbol

from .base import ExpressionPrinter, OpTable


class F64Ops(OpTable):
    """Real valued backend for measurement callbacks."""

    prelude_imports = ("import math",)
    elems_per_var = 1

    def const(self, v: float) -> str:
        return repr(float(v))

    def load(self, buf: str, idx: int) -> str:
        return f"{buf}[{idx}]"

    def store(self, buf: str, idx: int, expr: str) -> str:
        return f"{buf}[{idx}] = {expr}"

    def add(self, a: str, b: str) -> str:
        return f"({a} + {b})"

    def sub(self, a: str, b: str) -> str:
        return f"({a} - {b})"

    def mul(self, a: str, b: str) -> str:
        return f"({a} * {b})"

    def div(self, a: str, b: str) -> str:
        return f"({a} / {b})"

    def neg(self, a: str) -> str:
        return f"(-{a})"

    def real_scale(self, a: str, s: float) -> str:
        return f"({float(s)!r} * {a})"

    def exp(self, a: str) -> str:
        return f"math.exp({a})"

    def log(self, a: str) -> str:
        return f"math.log({a})"

    def sqrt(self, a: str) -> str:
        return f"math.sqrt({a})"


@dataclass
class MeasurementLayout:
    """Maps measurement symbols to native buffer slots."""

    slot: dict[Any, tuple[str, int]]
    n_var: int
    n_par: int
    n_obs: int
    observable_indices: tuple[int, ...] = ()

    @property
    def n_expr(self) -> int:
        return self.n_obs

    @classmethod
    def from_compiled(
        cls,
        compiled: Any,
        observables: list[str] | tuple[str, ...] | None = None,
    ) -> MeasurementLayout:
        obs = cls._normalize_observables(compiled, observables)
        obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
        selected = tuple(obs_idx[name] for name in obs)

        slot: dict[Any, tuple[str, int]] = {}
        for i, name in enumerate(compiled.var_names):
            slot[Symbol(f"cur_{name}")] = ("vars", i)
        for j, p in enumerate(compiled.calib_params):
            slot[p] = ("par", j)
        return cls(
            slot=slot,
            n_var=len(compiled.var_names),
            n_par=len(compiled.calib_params),
            n_obs=len(obs),
            observable_indices=selected,
        )

    @staticmethod
    def _normalize_observables(
        compiled: Any,
        observables: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if observables is None:
            return tuple(compiled.observable_names)

        obs = tuple(observables)
        if len(set(obs)) != len(obs):
            raise ValueError("Observable list contains duplicates.")

        obs_idx = {name: i for i, name in enumerate(compiled.observable_names)}
        missing = [name for name in obs if name not in obs_idx]
        if missing:
            raise KeyError(f"Unknown observables not in compiled model: {missing}")
        return tuple(sorted(obs, key=lambda name: obs_idx[name]))


class MeasurementPrinter(ExpressionPrinter):
    allocated_dtype = "np.float64"
    context_name = "measurement"


def build_measurement_cfunc(
    exprs: list[sp.Expr],
    layout: MeasurementLayout,
    ops: OpTable | None = None,
) -> Any:
    table: OpTable = F64Ops() if ops is None else ops
    body = MeasurementPrinter(table).emit(exprs, layout, allocate=False)
    preamble = [
        f"    vars = carray(vars_ptr, ({layout.n_var},))",
        f"    par = carray(par_ptr, ({layout.n_par},))",
        f"    out = carray(out_ptr, ({layout.n_obs},))",
    ]
    src = "\n".join(
        [
            *table.prelude_imports,
            "from numba import carray",
            "",
            "def _measurement_cf(vars_ptr, par_ptr, out_ptr):",
            *preamble,
            *body,
            "",
        ]
    )
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102
    sig = types.void(
        types.CPointer(types.float64),
        types.CPointer(types.float64),
        types.CPointer(types.float64),
    )
    return cfunc(sig)(ns["_measurement_cf"])
