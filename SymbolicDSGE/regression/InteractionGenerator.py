from .config import TemplateConfig
from itertools import combinations, combinations_with_replacement
import sympy as sp
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Sequence


@dataclass
class InteractionSpec:
    func: Callable[[Iterable[Any], int], Iterable[tuple[Any, ...]]]
    order_upper: int
    order_lower: int


class InteractionGenerator:

    @staticmethod
    def get_interactions(
        config: TemplateConfig,
        varnames: Sequence[str | sp.Symbol | sp.Function],
        expr: sp.Expr | None,
        t: sp.Symbol,
    ) -> list[tuple[str, ...]]:
        spec = InteractionGenerator._resolve_config(config)
        var_space = InteractionGenerator._resolve_variable_space(
            config, varnames, expr, t
        )

        orderu = spec.order_upper
        orderl = spec.order_lower
        func = spec.func
        interactions: list[tuple[str, ...]] = []

        for r in range(orderl, orderu + 1):
            interactions.extend(func(var_space, r))
        return interactions

    @staticmethod
    def _resolve_config(config: TemplateConfig) -> InteractionSpec:
        orderu = config.poly_interaction_order
        orderl = 2 if config.interaction_only else 1
        func: Callable[[Iterable[Any], int], Iterable[tuple[Any, ...]]] = (
            combinations_with_replacement
            if config.powers_in_interactions
            else combinations
        )
        return InteractionSpec(func=func, order_upper=orderu, order_lower=orderl)

    @staticmethod
    def _resolve_variable_space(
        config: TemplateConfig,
        varnames: Sequence[str | sp.Symbol | sp.Function] | None,
        expr: sp.Expr | None,
        t: sp.Symbol | None,
    ) -> list[str]:
        normalized_varnames = InteractionGenerator._normalize_names(varnames)
        if config.variable_space == "expr":
            if expr is None:
                raise ValueError(
                    "Expression must be provided when variable_space is set to 'expr'."
                )

            # Get Function atoms
            if t is None:
                raise ValueError(
                    "Model variables are denoted as functions of time. Please provide the time symbol used in the configuration."
                )

            func_atoms_str = list(
                set(
                    [
                        atom.func.__name__
                        for atom in expr.atoms(sp.Function)
                        if t in atom.free_symbols
                    ]
                )
            )

            if (normalized_varnames is not None) and (
                not all(var in normalized_varnames for var in func_atoms_str)
            ):
                raise ValueError(
                    "If variable names are passed, all discovered variables in the expression must be present in the variable names list."
                )

            return func_atoms_str

        elif normalized_varnames is not None:
            return normalized_varnames

        else:
            raise ValueError(
                "Variable space is unbounded. Please provide a list of variable names or set variable_space to 'expr' to automatically discover variables from the expression."
            )

    @staticmethod
    def _normalize_names(
        varnames: Sequence[str | sp.Symbol | sp.Function] | None,
    ) -> list[str] | None:
        if varnames is None:
            return None
        out = []
        for var in varnames:
            if isinstance(var, str):
                out.append(var)
            elif isinstance(var, sp.Symbol):
                out.append(var.name)
            elif isinstance(var, sp.Function):
                out.append(var.func.__name__)
            else:
                raise ValueError(
                    f"Variable names must be of type str, sympy.Symbol, or sympy.Function. Found type {type(var)}."
                )
        return out
