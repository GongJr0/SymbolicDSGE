"""Piecewise-linear (OccBin) solved model."""

from __future__ import annotations
from typing import Mapping, Callable, Union
from numpy import float64, int64, ndarray, asarray, zeros
from numpy.typing import NDArray

from .base import SolvedModel, NDF
from .shocks import simulation_shock_matrix
from ..solver_backend import PiecewiseSolution
from ..compiled_model import CompiledModel
from ..shock_generators import Shock
from ..sim_result import OccBinDiagnostics, StatePath
from ..._ckernels.occbin import occbin_sim

NDI = NDArray[int64]


class PiecewiseSolvedModel(SolvedModel[PiecewiseSolution]):
    """A model whose policy is path dependent, one rule per date and regime."""

    def __init__(self, compiled: CompiledModel, policy: PiecewiseSolution) -> None:
        """Initialize a piecewise solved model.

        Args:
            model: The solved model.
            solution: The piecewise solution.
        """
        super().__init__(compiled, policy)

    def _simulate_state_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1,
        x0: ndarray | None = None,
        *,
        check_ahead_periods: int = 200,
        max_check_ahead_periods: int = -1,
        max_iter: int = 30,
        init_regime: NDI | None = None,
        curb_retrench: bool = False,
        reset_regime: bool = False,
        reset_check_ahead: bool = False,
    ) -> StatePath:
        pol = self.policy
        comp = self.compiled
        x0_arr = self._simulation_initial_state(pol.f_ref, x0)[: comp.n_state]

        shock_mat = zeros((T, comp.n_state), dtype=float64)
        shock_mat[:, : comp.n_exog] = simulation_shock_matrix(
            comp, T, shocks, shock_scale
        )

        constraint = comp.construct_constraint_func()

        X, regimes, diag = occbin_sim(
            a=pol.a,
            b=pol.b,
            c=pol.c,
            f_ref=pol.f_ref,
            ss=pol.steady_state,
            par=comp._coerce_param_vector(comp.config.calibration.parameters),
            cond_addr=constraint.address,
            inclusive=constraint.inclusive,
            n_constraint=constraint.n_constraint,
            shocks=shock_mat,
            x_init=x0_arr,
            check_ahead_periods=check_ahead_periods,
            max_check_ahead_periods=max_check_ahead_periods,
            n_periods=T,
            max_iter=max_iter,
            init_regime=init_regime,
            curb_retrench=curb_retrench,
            reset_regime=reset_regime,
            reset_check_ahead=reset_check_ahead,
        )
        diagnostics = OccBinDiagnostics(
            diag["T_used"].astype(int64),
            diag["iters"].astype(int64),
            diag["max_err"].astype(float64),
            diag["periodic"].astype(int64),
        )

        return StatePath(X, regimes=regimes, diagnostics=diagnostics)
