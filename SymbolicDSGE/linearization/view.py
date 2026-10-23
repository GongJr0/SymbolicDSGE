from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from sympy import Eq, Expr

from ..core.config import ModelConfig, SymbolGetterDict, FunctionGetterDict


@dataclass(frozen=True)
class PointLinearView:
    model: dict[str, Eq]
    observables: SymbolGetterDict[Expr]
    parameters: SymbolGetterDict[float]
    steady_state: FunctionGetterDict[float]

    _config: ModelConfig

    def to_config(self) -> ModelConfig:
        if self._config.equations.constraints is not None:
            raise NotImplementedError(
                "Linear views of constrained models are not convertible to ModelConfig. "
                "Any constraint conditions with literals change interpretation after linearization, "
                "solving a point-linear constrained model may not produce results accurate to the "
                "original specification."
            )

        new = deepcopy(self._config)
        new.equations.model = dict(self.model)

        new.equations.observables = SymbolGetterDict(self.observables)
        new.equations.obs_is_affine = SymbolGetterDict(
            {var: True for var in self.observables}
        )

        new.calibration.parameters = SymbolGetterDict(self.parameters)
        new.variables.ss_seed = FunctionGetterDict(
            {var: 0.0 for var in self.steady_state}
        )

        return new
