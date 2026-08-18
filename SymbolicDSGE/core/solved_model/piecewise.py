"""Piecewise-linear (OccBin) solved model."""

from __future__ import annotations
from typing import Mapping, Callable, Union
from numpy import float64, int64, ndarray
from numpy.typing import NDArray

from .base import SolvedModel, NDF
from .first_order import FirstOrderSolvedModel
from .shocks import simulation_shock_matrix
from ..solver_backend import PiecewiseSolution
from ..compiled_model import CompiledModel
from ..shock_generators import Shock
from ..sim_result import OccBinDiagnostics, SimResult, StatePath
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
        x0: dict[str, float | float64] | list[float | float64] | ndarray | None = None,
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
        x0_arr = self._initial_state(x0)[: comp.n_state]

        shock_mat = simulation_shock_matrix(comp, T, shocks, shock_scale)

        constraint = comp.construct_constraint_func()

        X, regimes, diag = occbin_sim(
            a=pol.a,
            b=pol.b,
            c=pol.c,
            d=pol.d,
            cst=pol.cst,
            ghx_ref=pol.ghx_ref,
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

        # The perturbation kernels denominate their rows as they write them.
        # occbin_sim has no steady-state argument, so this path adds it here.
        X += pol.steady_state

        return StatePath(X, shocks=shock_mat, regimes=regimes, diagnostics=diagnostics)

    def sim(
        self,
        T: int,
        shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: dict[str, float | float64] | list[float | float64] | ndarray | None = None,
        observables: bool = False,
        *,
        check_ahead_periods: int = 200,
        max_check_ahead_periods: int = -1,
        max_iter: int = 30,
        init_regime: NDI | None = None,
        curb_retrench: bool = False,
        reset_regime: bool = False,
        reset_check_ahead: bool = False,
    ) -> SimResult:
        """Simulate the model over T periods, solving the constraints per date.

        Each shock date is solved by guess and verify: a regime sequence is
        proposed over the look-ahead horizon, the path it implies is checked
        against the conditions, and the guess is latched until it stops moving.
        The returned :class:`SimResult` therefore also carries
        :attr:`~SimResult.regimes` and :attr:`~SimResult.diagnostics`.

        The keyword-only arguments control that search. Their defaults are
        Dynare's ``occbin.simul`` defaults except where noted.

        Parameters
        ----------
        T : int
            Number of time periods to simulate.

        shocks : Mapping[str, Shock | Callable[[float], ndarray] | ndarray], optional
            Maps each exogenous variable name to its shock. A ``"a,b"`` key is a
            joint (multivar) shock over those variables. Each value may be a
            :class:`Shock` distribution spec (materialized into a ``T``-horizon
            draw here), a ``callable`` taking the shock scale and returning a
            ``(T,)``/``(T, k)`` array, or a raw ndarray path of that shape. When
            ``None``, all shocks are zero. Each date's innovation is a surprise:
            agents solve as though no further shock arrives.

        shock_scale : float, optional
            A scaling factor applied to all shocks. The response is not
            proportional to it here, since the regime a path realizes depends on
            how far the shock pushes the model.

        x0 : dict[str, float] | list[float] | ndarray, optional
            Initial state at ``t - 1``, in levels. A sequence covers every
            compiled variable in declaration order, minted lags included; a
            mapping names only the variables it sets. What it leaves out, and
            everything when this is None, starts at the reference steady state.
            Whether a constraint binds depends on where the model starts, so
            this is a modelling choice rather than a formality.

        observables : bool, optional
            If True, compute and include observable variables in the output.

        check_ahead_periods : int, optional
            The expectation horizon (how far ahead each guess looks).

        max_check_ahead_periods : int, optional
            How far a guess may grow when it still binds at its last date.
            ``-1`` takes a budget of ``ceil(sqrt(check_ahead_periods))`` past the
            horizon; passing ``check_ahead_periods`` itself forbids growth and
            forces the last date relaxed instead. Dynare's default is unbounded,
            which is not reachable here: growth is reserved up front in a
            caller-owned arena.

        max_iter : int, optional
            Guess-and-verify passes a date may take before it is reported as
            unconverged.

        init_regime : ndarray, optional
            A starting guess, one row per shock date in the layout
            :attr:`~SimResult.regimes` can feed back in. It may cover the
            leading dates only, and the rest carry the previous date's converged
            guess shifted forward. Rows shorter than the horizon relax into the tail;
            longer ones raise the horizon to their own length, up to the growth cap.
            All relaxed by default, and it seeds the search rather than fixing the answer.

        curb_retrench : bool, optional
            Give up one binding date per pass instead of every date the latch
            wants relaxed. Damps a guess that oscillates.

        reset_regime : bool, optional
            Start every date from the all-relaxed guess rather than the previous
            date's, shifted.

        reset_check_ahead : bool, optional
            Let a reset also drop a horizon that grew back to
            ``check_ahead_periods``. Only meaningful with ``reset_regime``.

        Returns
        -------
        SimResult
            The piecewise path in levels, with the realized regimes and the
            per-date convergence record beside it.
        """
        path = self._simulate_state_matrix(
            T,
            shocks,
            shock_scale,
            x0=x0,
            check_ahead_periods=check_ahead_periods,
            max_check_ahead_periods=max_check_ahead_periods,
            max_iter=max_iter,
            init_regime=init_regime,
            curb_retrench=curb_retrench,
            reset_regime=reset_regime,
            reset_check_ahead=reset_check_ahead,
        )
        return self._assemble_simulation(path, observables)

    def sim_reference(
        self,
        T: int,
        shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: dict[str, float | float64] | list[float | float64] | ndarray | None = None,
        observables: bool = False,
    ) -> SimResult:
        """Simulate the reference regime (a first-order solution) over T periods, ignoring constraints.

        Parameters
        ----------
        T : int
            Number of time periods to simulate.

        shocks : Mapping[str, Shock | Callable[[float], ndarray] | ndarray], optional
            Maps each exogenous variable name to its shock. A ``"a,b"`` key is a
            joint (multivar) shock over those variables. Each value may be a
            :class:`Shock` distribution spec (materialized into a ``T``-horizon
            draw here), a ``callable`` taking the shock scale and returning a
            ``(T,)``/``(T, k)`` array, or a raw ndarray path of that shape. When
            ``None``, all shocks are zero.

        shock_scale : float, optional
            A scaling factor applied to all shocks.

        x0 : dict[str, float] | list[float] | ndarray, optional
            Initial state at ``t - 1``, in levels. A sequence covers every
            compiled variable in declaration order, minted lags included; a
            mapping names only the variables it sets. What it leaves out, and
            everything when this is None, starts at the steady state.

        observables : bool, optional
            If True, compute and include observable variables in the output.

        Returns
        -------
        SimResult
            The simulated path in levels, with each variable's series available
            by name.

        """

        return FirstOrderSolvedModel(self.compiled, self.policy.ref).sim(
            T, shocks, shock_scale, x0=x0, observables=observables
        )
