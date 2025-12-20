from .model_config import ModelConfig, Equations, Calib
from pathlib import Path
import yaml
import sympy as sp
from sympy import Symbol, Function, Eq, Expr
from sympy.core.relational import Relational
from numpy import float64
import pickle


class ModelParser:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.conf: ModelConfig = self.from_yaml()

        self.__post_init__()

    def __post_init__(self) -> None:
        conf = self.conf
        self.validate_constraints(conf)
        self.validate_calib(conf)

    def get(self) -> ModelConfig:
        return self.conf

    def to_pickle(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.conf, f)

    @classmethod
    def validate_constraints(cls, conf: ModelConfig) -> None:
        is_constrained = conf.constrained
        doesnt_exist = []
        no_constraint = []
        for var, constrained in is_constrained.items():
            if var not in conf.variables:
                doesnt_exist.append(var)
            if constrained and var not in conf.equations.constraint:
                no_constraint.append(var)

        if doesnt_exist and no_constraint:
            raise ValueError(
                f"The following variables are marked as constrained but do not exist: {doesnt_exist}, and the following constrained variables have no constraint equations: {no_constraint}"
            )
        elif doesnt_exist:
            raise ValueError(
                f"The following variables are marked as constrained but do not exist: {doesnt_exist}"
            )
        elif no_constraint:
            raise ValueError(
                f"The following constrained variables have no constraint equations: {no_constraint}"
            )

    @classmethod
    def validate_calib(cls, conf: ModelConfig) -> None:
        nf_param = []

        for param in conf.calibration.parameters:
            if param not in conf.parameters:
                nf_param.append(param)

        if nf_param:
            raise ValueError(f"Calibration contains unknown parameters: {nf_param}")

    def from_yaml(self) -> ModelConfig:
        path = self.config_path
        t = sp.symbols("t", integer=True)

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        name = data.get("name", "Unnamed")
        variables: list[Function] = list(map(Function, data["variables"]))
        constrained: dict[Function, bool] = dict(
            zip(variables, [data["constrained"][var] for var in data["variables"]])
        )
        params: list[Symbol] = list(sp.symbols(data["parameters"]))
        shocks: list[Symbol] = list(sp.symbols(data["shocks"]))
        observables: list[Symbol] = list(sp.symbols(data["observables"]))

        param_map = dict(zip(data["parameters"], params))

        _LOCALS = {
            "t": t,
            **{var.name: var for var in variables},
            **{param.name: param for param in params},
            **{shock.name: shock for shock in shocks},
            **{obs.name: obs for obs in observables},
        }

        def _get_expr(expr: str) -> Expr:
            out = sp.parse_expr(expr, local_dict=_LOCALS, evaluate=False)
            assert isinstance(out, Expr), "Expression is not a valid sympy expression."
            return out

        def _get_relational(expr: str) -> Relational:
            out = sp.parse_expr(expr, local_dict=_LOCALS, evaluate=False)
            # assert isinstance(out, (Relational, Eq)), "Expression is not a valid relational."
            return out

        def _get_eq(expr: str) -> Eq:
            sides = map(lambda x: x.strip(), expr.split("=", maxsplit=2))
            out = sp.Eq(
                sp.parse_expr(next(sides), local_dict=_LOCALS, evaluate=False),
                sp.parse_expr(next(sides), local_dict=_LOCALS, evaluate=False),
            )
            assert isinstance(out, Eq), (
                "Please ensure no inequalities are used where equalities are strictly required. "
                f"Given expression: \n {out}"
            )
            return out

        model: list[Eq] = [_get_eq(eq) for eq in data["equations"]["model"]]

        ordered_vars = [var for var in data["variables"]]
        constraint: dict[Symbol, Relational] = {
            _LOCALS[var_name]: _get_relational(
                data["equations"]["constraint"][var_name]
            )
            for var_name in ordered_vars
            if var_name in data["equations"]["constraint"]
        }

        observables_eq: dict[Symbol, Expr] = {
            _LOCALS[obs_name]: _get_expr(data["equations"]["observables"][obs_name])
            for obs_name in data["observables"]
            if obs_name in data["equations"]["observables"]
        }

        equations = Equations(
            model=model, constraint=constraint, observable=observables_eq
        )

        parameters: dict[Symbol, float64] = {
            _LOCALS[param_name]: float64(data["calibration"]["parameters"][param_name])
            for param_name in data["parameters"]
            if param_name in data["calibration"]["parameters"]
        }

        calibration = Calib(parameters=parameters)

        return ModelConfig(
            name=name,
            variables=variables,
            constrained=constrained,
            parameters=params,
            shocks=shocks,
            observables=observables,
            equations=equations,
            calibration=calibration,
        )
