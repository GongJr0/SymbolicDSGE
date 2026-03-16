import sympy as sp
from .config import TemplateConfig, ConstantFiltering
from typing import Sequence, cast, Callable


class ConstantHandler:
    def __init__(self, config: TemplateConfig):
        self._config = config
        self._strategy = config.constant_filtering

    def get_handled_exprs(self, exprs: Sequence[sp.Expr]) -> Sequence[sp.Expr]:
        handler = self._method_dispatch[self.strategy]
        return [handler(expr) for expr in exprs]

    @staticmethod
    def _keep(expr: sp.Expr) -> sp.Expr:
        return expr

    @staticmethod
    def _disqualify(expr: sp.Expr) -> sp.Expr:
        terms = ConstantHandler._get_terms(expr)
        if any(ConstantHandler._is_free_constant(term) for term in terms):
            return sp.S.NaN
        return expr

    @staticmethod
    def _parametrize_additive(expr: sp.Expr) -> sp.Expr:
        terms = ConstantHandler._get_terms(expr)
        out = []
        c_idx = 1  # 1-indexed symbol subscript
        for term in terms:
            if ConstantHandler._is_free_constant(term):
                out.append(sp.Dummy(f"C_{c_idx}"))
                c_idx += 1
            else:
                out.append(term)
        return sp.Add(*out)

    @staticmethod
    def _parametrize_all(expr: sp.Expr) -> sp.Expr:
        c_idx = 1  # 1-indexed symbol subscript

        def next_dummy() -> sp.Expr:
            nonlocal c_idx
            dummy = sp.Dummy(f"C_{c_idx}")
            c_idx += 1
            return dummy

        return ConstantHandler._parametrize_all_recursive(expr, next_dummy)

    @staticmethod
    def _parametrize_all_recursive(
        expr: sp.Expr,
        next_dummy: Callable[[], sp.Expr],
    ) -> sp.Expr:
        if ConstantHandler._is_free_constant(expr):
            return next_dummy()
        if expr.is_Atom:
            return expr

        if expr.func is sp.Pow and len(expr.args) == 2:
            base, exp = cast(tuple[sp.Expr, sp.Expr], expr.args)
            new_base = ConstantHandler._parametrize_all_recursive(base, next_dummy)

            # Preserve reciprocal structure for division-like expressions.
            if exp == -1 and not base.is_number:
                new_exp = exp
            else:
                new_exp = ConstantHandler._parametrize_all_recursive(exp, next_dummy)

            return ConstantHandler._rebuild_expr(expr, (new_base, new_exp))

        new_args = tuple(
            ConstantHandler._parametrize_all_recursive(cast(sp.Expr, arg), next_dummy)
            for arg in expr.args
        )
        return ConstantHandler._rebuild_expr(expr, new_args)

    @staticmethod
    def _strip(expr: sp.Expr) -> sp.Expr:
        terms = ConstantHandler._get_terms(expr)
        out = [term for term in terms if not ConstantHandler._is_free_constant(term)]
        return sp.Add(*out)

    @staticmethod
    def _is_free_constant(expr: sp.Expr) -> bool:
        if len(expr.free_symbols) == 0 and expr.is_number:
            return True
        return False

    @staticmethod
    def _get_terms(expr: sp.Expr) -> Sequence[sp.Expr]:
        return cast(Sequence[sp.Expr], expr.as_ordered_terms())

    @staticmethod
    def _rebuild_expr(expr: sp.Expr, args: tuple[sp.Expr, ...]) -> sp.Expr:
        if args == expr.args:
            return expr

        try:
            rebuilt = expr.func(*args, evaluate=False)
        except TypeError:
            rebuilt = expr.func(*args)

        return cast(sp.Expr, rebuilt)

    @property
    def _method_dispatch(self) -> dict[ConstantFiltering, Callable[[sp.Expr], sp.Expr]]:
        dispatch_map: dict[ConstantFiltering, Callable[[sp.Expr], sp.Expr]] = {
            ConstantFiltering.KEEP: self._keep,
            ConstantFiltering.DISQUALIFY: self._disqualify,
            ConstantFiltering.PARAMETRIZE_ADDITIVE: self._parametrize_additive,
            ConstantFiltering.PARAMETRIZE_ALL: self._parametrize_all,
            ConstantFiltering.STRIP: self._strip,
        }
        return dispatch_map

    @property
    def config(self) -> TemplateConfig:
        return self._config

    @property
    def strategy(self) -> ConstantFiltering:
        return ConstantFiltering(self._strategy)
