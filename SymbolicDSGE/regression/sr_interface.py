from sympy import Expr
from .symbolic_regression import SymbolicRegressor
from .model_parametrizer import ModelParametrizer
from .fit_result import FitResult

from ..kalman.filter import FilterResult

from numpy.typing import NDArray
from numpy import float64


from pandas import DataFrame

from typing import TYPE_CHECKING, TypeAlias, Literal, cast

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

NDF: TypeAlias = NDArray[float64]


class SRInterface:
    def __init__(
        self,
        model: "SolvedModel",
        obs_name: str,
        parametrizer: ModelParametrizer,
    ):
        self._model = model

        self._variable_names = parametrizer.variable_names
        expr = self._get_equation(obs_name)
        parametrizer.make_and_add_template(expr)
        self._obs_name = obs_name
        self.sr = SymbolicRegressor(parametrizer)

    def _get_equation(self, obs: str) -> Expr:
        eq_raw = self._model.config.equations.observable[obs]
        subs_map = self._model.config.calibration.parameters
        eq_subbed = eq_raw.subs(subs_map)
        return eq_subbed

    def get_kf(self, y: NDF | DataFrame) -> FilterResult:
        affine = all(self._model.config.equations.obs_is_affine.values())
        mode = "linear" if affine else "extended"
        kf = self._model.kalman(
            y=y,
            observables=self._model.compiled.observable_names,
            filter_mode=cast(Literal["linear", "extended"], mode),
            estimate_R_diag=False,
            return_shocks=False,
            _debug=False,
        )
        return kf

    def fit_to_kf(self, y: NDF | DataFrame) -> FitResult:
        kf: FilterResult = self.get_kf(y)

        var_idx = self._model.compiled.idx
        var_idx_ls = sorted([var_idx[var] for var in self.selected_var_names])
        obs_idx = self.obs_idx[self.selected_obs_name]

        state = kf.x_pred[:, var_idx_ls]

        target = self._resolve_target()
        target_data = (
            kf.innov[:, obs_idx] if target == "innov" else kf.y_pred[:, obs_idx]
        )

        best = self.sr.fit(
            X=state,
            y=target_data,
            variable_names=self.selected_var_names,
        )
        exprs: DataFrame = cast(DataFrame, self.sr.model.equations_)
        return FitResult(expressions=exprs, best=best)

    def _resolve_target(self) -> Literal["innov", "meas_pred"]:
        if self.sr.parametrizer.config.include_expression:
            return "meas_pred"
        else:
            return "innov"

    @property
    def obs_idx(self) -> dict[str, int]:
        return {
            obs: idx for idx, obs in enumerate(self._model.compiled.observable_names)
        }

    @property
    def selected_obs_name(self) -> str:
        return self._obs_name

    @property
    def selected_var_names(self) -> list[str]:
        return self._variable_names
