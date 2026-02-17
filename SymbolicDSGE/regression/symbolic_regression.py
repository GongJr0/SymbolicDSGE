import sympy as sp

from .model_defaults import PySRParams
from pysr import ExpressionSpec, PySRRegressor
from typing import Sequence
import warnings
import numpy as np
import pandas as pd
import re  # :(

from .base_model_parametrizer import BaseModelParametrizer, _normalize_variables


class SymbolicRegressor:
    def __init__(self, parametrizer: BaseModelParametrizer) -> None:
        self.parametrizer = parametrizer
        self.params: PySRParams = self.parametrizer.params
        self.model = self._load_params()

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        variable_names: Sequence[str | sp.Symbol | sp.Function] | None = None,
        param_overrides: dict | None = None,
    ) -> pd.Series:
        if param_overrides is None:
            param_overrides = {}

        if variable_names is None:
            if isinstance(X, np.ndarray):
                raise ValueError(
                    "Please provide the variable names if X is a numpy array."
                )
            elif isinstance(X, pd.DataFrame):
                variable_names = list(X.columns)
            else:
                raise ValueError("X must be a numpy array or a pandas DataFrame.")

        elif isinstance(variable_names, Sequence) and all(
            isinstance(
                var,
                (
                    str,
                    sp.Symbol,
                    sp.Function,
                    sp.core.function.UndefinedFunction,
                ),
            )
            for var in variable_names
        ):
            pass
        else:
            warnings.warn(
                "Variable names could not be determined from X. Using inferred variable names from the parametrizer.",
                UserWarning,
            )
            variable_names = self.parametrizer.variable_names

        model = self._load_params()
        model.set_params(**param_overrides)
        variable_names = self._validate_and_normalize_varnames(variable_names)
        model.fit(X, y, variable_names=variable_names)
        model.equations_ = self._convert_equations_to_sympy(
            model.equations_
        )  # pyright: ignore
        self.model = model

        best: pd.Series = model.get_best()  # pyright: ignore

        return best

    def _get_sp_from_template(self, expr_block: pd.Series) -> pd.Series:
        # PySR template output format maps f1(x, y) -> f1(#1, #2) making direct substitution difficult.
        # Moreover, there's no equation output in templates. Instead you get a ";" delimited list of functions.
        # The "combine" expression must be referred to map positional pysr output to actual variables per function.
        clean_expr = self.parametrizer.clean_expr
        template = self.parametrizer.params.expression_spec
        if template is None:
            raise ValueError("No template found in parametrizer params.")

        if isinstance(template, ExpressionSpec):
            return expr_block  # already in sympy format

        combine = template.combine  # pyright: ignore
        if clean_expr is not None:
            comb_split = list(map(str.strip, combine.split("+")))

            # First index before a function is encountered is the end of the base expression we want to replace with the clean version.
            cutoff_idx = len(comb_split)  # default to end if no functions found
            for i, part in enumerate(comb_split):
                if re.search(
                    r"f_\d+\(", part
                ):  # f_\d+ by convention denotes a function in the template
                    cutoff_idx = i
                    break
            functions = comb_split[cutoff_idx:]
            combine_components = [clean_expr, *functions]
            combine = " + ".join(combine_components)

        # Parse calls in `combine` like: f_1(x, y), f_2(y, z), ...
        # This is where we learn each function's *intended* argument names.
        call_pat = re.compile(r"(f_\d+)\(\s*([^()]*)\s*\)")
        split_args_pat = re.compile(r"\s*,\s*")

        arg_map: dict[str, dict[str, str]] = {}  # {"f_1": {"#1": "x", "#2": "y"}, ...}
        f_keys: dict[str, str] = {}  # {"f_1": "f_1(x, y)", ...}
        matches = call_pat.findall(combine)
        for fn, argstr in matches:
            args = [
                a.strip() for a in split_args_pat.split(argstr.strip()) if a.strip()
            ]
            arg_map[fn] = {f"#{i+1}": var for i, var in enumerate(args)}
            f_keys[fn] = f"{fn}({', '.join(args)})"

        # Parse template function definitions in expr_block.equation:
        #   "f_1 = asinh(asinh(#1)); f_2 = ...; ..."
        model_exprs = [
            s.strip() for s in str(expr_block.equation).split(";") if s.strip()
        ]
        def_pat = re.compile(r"^(f_\d+)\s*=\s*(.*)$")

        # Replace "#k" tokens *as tokens* (NOT via str.translate, which is char-based only).
        token_pat = re.compile(r"#\d+")

        def _sub_tokens(s: str, fn: str) -> str:
            amap = arg_map.get(fn, {})
            return token_pat.sub(lambda m: amap.get(m.group(0), m.group(0)), s)

        subbed: dict[str, str] = {}  # {"f_1(x, y)": "<rhs with x,y>", ...}
        for model_expr in model_exprs:
            m = def_pat.match(model_expr)
            if not m:
                raise ValueError(f"Unparseable template function line: {model_expr!r}")
            fn, rhs = m.groups()
            rhs = rhs.strip()

            if fn not in f_keys:
                # NOTE (data structure mismatch): A function was defined in expr_block.equation
                # but does not appear in `combine`, so we don't know how to map #k -> variable name.
                # Depending on your use-case, you might want to skip, warn, or hard-fail.
                raise KeyError(
                    f"Function {fn!r} appears in template output but not in combine expression."
                )

            rhs_subbed = _sub_tokens(rhs, fn)
            subbed[f_keys[fn]] = rhs_subbed

        # Substitute expanded function bodies back into `combine`.
        # Do NOT use str.translate here: translate cannot replace substrings like "f_1(x, y)".
        keys = sorted(
            subbed.keys(), key=len, reverse=True
        )  # longest-first avoids partial overlaps
        if keys:
            key_pat = re.compile("|".join(re.escape(k) for k in keys))
            expr_str = key_pat.sub(lambda m: subbed[m.group(0)], combine)
        else:
            expr_str = combine

        out = expr_block.copy()
        out["equation"] = expr_str

        var_names = self.parametrizer.variable_names
        locals = (
            {var: sp.symbols(var) for var in var_names} if var_names is not None else {}
        )
        out["sympy_format"] = sp.parse_expr(expr_str, local_dict=locals)
        return out

    def _load_params(self) -> PySRRegressor:
        return PySRRegressor(**self.params)  # pyright: ignore

    def _validate_and_normalize_varnames(
        self, varnames: Sequence[str | sp.Symbol | sp.Function]
    ) -> list[str]:
        varnames = _normalize_variables(varnames)
        if self.parametrizer.variable_names is not None:
            if set(varnames) != set(self.parametrizer.variable_names):
                raise ValueError(
                    f"Variable names {varnames} do not match the parametrizer's variable names {self.parametrizer.variable_names}"
                )
        return varnames

    def _convert_equations_to_sympy(self, equations: pd.DataFrame) -> pd.DataFrame:
        equations["sympy_format"] = None  # Create column for .apply
        converted = equations.apply(lambda row: self._get_sp_from_template(row), axis=1)
        return converted
