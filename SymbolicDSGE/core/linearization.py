from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping, TypeAlias

import sympy as sp
from sympy import Eq, Expr, Function, Symbol
from sympy.core.function import AppliedUndef, FunctionClass, UndefinedFunction

if TYPE_CHECKING:
    from .config import ModelConfig

_VAR_FUNC: TypeAlias = FunctionClass | UndefinedFunction


class LinearizationMethod(StrEnum):
    LOG = "log"
    TAYLOR = "taylor"
    NONE = "none"


@dataclass(frozen=True)
class VariableTransformSpec:
    original: _VAR_FUNC
    transformed: _VAR_FUNC
    method: LinearizationMethod
    steady_state: Expr | None

    def reconstruct(self, transformed_call: Expr) -> Expr:
        if self.method == LinearizationMethod.NONE:
            return transformed_call

        if self.steady_state is None:
            raise ValueError(
                f"Variable '{self.original}' is missing a steady state required for {self.method.value} linearization."
            )

        if self.method == LinearizationMethod.LOG:
            return self.steady_state * sp.exp(transformed_call)

        if self.method == LinearizationMethod.TAYLOR:
            return self.steady_state + transformed_call

        raise ValueError(f"Unsupported linearization method '{self.method}'.")


@dataclass(frozen=True)
class LinearizationContext:
    specs: tuple[VariableTransformSpec, ...]
    time_symbol: Symbol
    spec_by_original: dict[_VAR_FUNC, VariableTransformSpec]
    spec_by_transformed: dict[_VAR_FUNC, VariableTransformSpec]
    transformed_call_registry: dict[_VAR_FUNC, _VAR_FUNC]

    def collect_original_calls(self, expr: Expr) -> tuple[AppliedUndef, ...]:
        calls = [
            call
            for call in expr.atoms(AppliedUndef)
            if call.func in self.spec_by_original
        ]
        return tuple(sorted(calls, key=_call_sort_key))

    def collect_transformed_calls(self, expr: Expr) -> tuple[AppliedUndef, ...]:
        calls = [
            call
            for call in expr.atoms(AppliedUndef)
            if call.func in self.spec_by_transformed
        ]
        return tuple(sorted(calls, key=_call_sort_key))

    def timed_call_substitutions(self, expr: Expr) -> dict[AppliedUndef, Expr]:
        subs: dict[AppliedUndef, Expr] = {}
        for call in self.collect_original_calls(expr):
            spec = self.spec_by_original[call.func]
            transformed_call = spec.transformed(*call.args)
            subs[call] = spec.reconstruct(transformed_call)
        return subs

    def zero_point_substitutions(self, expr: Expr) -> dict[AppliedUndef, Expr]:
        return {call: sp.Integer(0) for call in self.collect_transformed_calls(expr)}

    def public_name_substitutions(self, expr: Expr) -> dict[AppliedUndef, Expr]:
        subs: dict[AppliedUndef, Expr] = {}
        for call in self.collect_transformed_calls(expr):
            spec = self.spec_by_transformed[call.func]
            subs[call] = spec.original(*call.args)
        return subs


class Linearizer:
    def __init__(
        self,
        conf: ModelConfig | None = None,
        *,
        method_dict: dict[_VAR_FUNC, str | LinearizationMethod] | None = None,
        steady_state: dict[_VAR_FUNC, Expr | float | None] | None = None,
        equations: list[Eq] | None = None,
        time_symbol: Symbol | None = None,
        variable_order: list[_VAR_FUNC] | None = None,
        shock_symbols: list[Symbol] | None = None,
    ) -> None:
        if conf is not None:
            if conf.symbolically_linearized:
                raise ValueError("ModelConfig is already symbolically linearized.")

            variable_order = list(conf.variables.variables)
            method_dict = {
                var: conf.variables.linearization[var] for var in variable_order
            }
            steady_state = {
                var: conf.variables.steady_state[var] for var in variable_order
            }
            equations = list(conf.equations.model)
            time_symbol = Symbol("t", integer=True)
            shock_symbols = list(conf.shock_map.keys())

        if method_dict is None or steady_state is None or equations is None:
            raise TypeError(
                "Linearizer requires either a ModelConfig or explicit method_dict, steady_state, and equations."
            )

        self._method_dict = self._parse_methods(method_dict)
        self._steady_state = {
            varname: None if val is None else sp.sympify(val)
            for varname, val in steady_state.items()
        }
        self._equations = equations
        self._residuals: list[Expr] = [
            eq.lhs - eq.rhs for eq in equations
        ]  # pyright: ignore
        self._time_symbol = (
            time_symbol if time_symbol is not None else Symbol("t", integer=True)
        )
        self._shock_symbols = tuple(shock_symbols or [])

        ordered_variables = (
            variable_order
            if variable_order is not None
            else list(self._method_dict.keys())
        )
        self._context = self._build_context(ordered_variables)

    @classmethod
    def from_config(cls, conf: ModelConfig) -> "Linearizer":
        return cls(conf)

    def _build_context(
        self,
        ordered_variables: list[_VAR_FUNC],
    ) -> LinearizationContext:
        specs: list[VariableTransformSpec] = []
        spec_by_original: dict[_VAR_FUNC, VariableTransformSpec] = {}
        spec_by_transformed: dict[_VAR_FUNC, VariableTransformSpec] = {}
        transformed_call_registry: dict[_VAR_FUNC, _VAR_FUNC] = {}

        for var in ordered_variables:
            method = self.method_dict[var]
            steady_state = self.steady_state[var]
            transformed = Function(f"__lin_{var.__name__}")
            spec = VariableTransformSpec(
                original=var,
                transformed=transformed,
                method=method,
                steady_state=steady_state,
            )
            specs.append(spec)
            spec_by_original[var] = spec
            spec_by_transformed[transformed] = spec
            transformed_call_registry[transformed] = var

        return LinearizationContext(
            specs=tuple(specs),
            time_symbol=self._time_symbol,
            spec_by_original=spec_by_original,
            spec_by_transformed=spec_by_transformed,
            transformed_call_registry=transformed_call_registry,
        )

    def _validate_ready_for_linearization(self) -> None:
        for spec in self.context.specs:
            if spec.method == LinearizationMethod.NONE:
                continue

            if spec.steady_state is None:
                raise ValueError(
                    f"Variable '{spec.original}' is missing a steady state required for {spec.method.value} linearization."
                )

            if spec.method != LinearizationMethod.LOG:
                continue

            simplified = sp.simplify(spec.steady_state)
            if simplified.is_number and float(simplified) <= 0.0:
                raise ValueError(
                    f"Variable '{spec.original}' has nonpositive steady state {simplified} required for log linearization."
                )

    def _transform_expr(self, expr: Expr) -> Expr:
        return expr.xreplace(self.context.timed_call_substitutions(expr))

    def _expr_zero_point(self, expr: Expr) -> Expr:
        return expr.xreplace(self.context.zero_point_substitutions(expr))

    def _rename_public_variables(self, expr: Expr) -> Expr:
        return expr.xreplace(self.context.public_name_substitutions(expr))

    def _first_order_terms(
        self,
        expr: Expr,
        transformed_calls: tuple[AppliedUndef, ...],
        zero_subs: Mapping[AppliedUndef, Expr],
    ) -> list[Expr]:
        terms: list[Expr] = []
        eval_subs = dict(zero_subs)
        eval_subs.update(self._shock_zero_substitutions())
        for call in transformed_calls:
            coeff = sp.diff(expr, call).subs(eval_subs)
            if coeff == 0:
                continue
            terms.append(coeff * call)
        return terms

    def _shock_zero_substitutions(self) -> dict[Symbol, Expr]:
        return {shock: sp.Integer(0) for shock in self._shock_symbols}

    def linearize_expr(
        self,
        expr: Expr,
        *,
        require_zero_at_expansion_point: bool = False,
        equation_index: int | None = None,
    ) -> tuple[Expr, Expr]:
        self._validate_ready_for_linearization()
        transformed_expr = self._transform_expr(expr)
        transformed_calls = self.context.collect_transformed_calls(transformed_expr)
        zero_subs = {call: sp.Integer(0) for call in transformed_calls}

        residual_at_zero = transformed_expr.xreplace(zero_subs)
        residual_at_expansion_point = sp.simplify(
            residual_at_zero.subs(self._shock_zero_substitutions())
        )
        if require_zero_at_expansion_point and residual_at_expansion_point != 0:
            prefix = (
                f"Equation {equation_index} does not vanish at the supplied steady state"
                if equation_index is not None
                else "Expression does not vanish at the supplied steady state"
            )
            raise ValueError(f"{prefix}: {residual_at_expansion_point}")

        linear_terms = self._first_order_terms(
            transformed_expr,
            transformed_calls,
            zero_subs,
        )
        linear_expr = residual_at_zero + sp.Add(*linear_terms)
        return self._rename_public_variables(linear_expr), residual_at_zero

    def linearize_equations(self) -> list[Eq]:
        out: list[Eq] = []
        shock_zero_subs = self._shock_zero_substitutions()
        for idx, eq in enumerate(self.equations):
            linear_lhs, lhs_at_zero = self.linearize_expr(
                eq.lhs
            )  # pyright: ignore[arg-type]
            linear_rhs, rhs_at_zero = self.linearize_expr(
                eq.rhs
            )  # pyright: ignore[arg-type]

            lhs_expansion_point = sp.simplify(lhs_at_zero.subs(shock_zero_subs))
            rhs_expansion_point = sp.simplify(rhs_at_zero.subs(shock_zero_subs))
            residual_at_expansion_point = sp.simplify(
                lhs_expansion_point - rhs_expansion_point
            )
            if residual_at_expansion_point != 0:
                raise ValueError(
                    f"Equation {idx} does not vanish at the supplied steady state: {residual_at_expansion_point}"
                )

            out.append(
                sp.Eq(
                    sp.simplify(linear_lhs - lhs_expansion_point),
                    sp.simplify(linear_rhs - rhs_expansion_point),
                )
            )
        return out

    @staticmethod
    def _parse_methods(
        method_dict: dict[_VAR_FUNC, str | LinearizationMethod],
    ) -> dict[_VAR_FUNC, LinearizationMethod]:
        parsed_methods: dict[_VAR_FUNC, LinearizationMethod] = {}
        for var, method in method_dict.items():
            if isinstance(method, LinearizationMethod):
                parsed_methods[var] = method
                continue

            normalized = method.strip().lower()
            try:
                parsed_methods[var] = LinearizationMethod(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid method '{method}' for variable '{var}'. Valid methods are: {[member.value for member in LinearizationMethod]}"
                ) from exc
        return parsed_methods

    @property
    def context(self) -> LinearizationContext:
        return self._context

    @property
    def method_dict(self) -> dict[_VAR_FUNC, LinearizationMethod]:
        return self._method_dict

    @property
    def steady_state(self) -> dict[_VAR_FUNC, Expr | None]:
        return self._steady_state

    @property
    def equations(self) -> list[Eq]:
        return self._equations

    @property
    def residuals(self) -> list[Expr]:
        return self._residuals

    @property
    def missing_steady_states(self) -> tuple[_VAR_FUNC, ...]:
        return tuple(
            spec.original
            for spec in self.context.specs
            if spec.method != LinearizationMethod.NONE and spec.steady_state is None
        )


def linearize_model(conf: ModelConfig) -> ModelConfig:
    if conf.symbolically_linearized:
        raise ValueError("ModelConfig is already symbolically linearized.")

    linearizer = Linearizer(conf)
    linearized = deepcopy(conf)
    linearized.equations.model = linearizer.linearize_equations()
    linearized.symbolically_linearized = True
    return linearized


def _call_sort_key(call: AppliedUndef) -> tuple[str, str]:
    return (call.func.__name__, sp.srepr(call))
