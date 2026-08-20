"""Container returned by ``SolvedModel.sim`` and ``SolvedModel.irf``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, NamedTuple

from numpy import float64, int64
from numpy.typing import NDArray

NDF = NDArray[float64]
NDI = NDArray[int64]


@dataclass(frozen=True)
class OccBinDiagnostics:
    """Per-period convergence record from a piecewise solve, one entry per date.

    ``T_used`` is the check-ahead horizon the period settled on, ``iters`` the
    guess-and-verify passes it took, and ``max_err`` the largest constraint
    violation left in the accepted guess. ``periodic`` is nonzero where a period
    was accepted as a cycle, carrying the code it would otherwise have raised.
    """

    T_used: NDI
    iters: NDI
    max_err: NDF
    periodic: NDI


class StatePath(NamedTuple):
    """What one simulation produced, before it is named and dressed as a result.

    The regime fields are an OccBin solve's, so a perturbation family returns
    ``StatePath(X)`` and lets them default.
    """

    X: NDF
    shocks: NDF
    regimes: NDI | None = None
    diagnostics: OccBinDiagnostics | None = None


@dataclass(frozen=True, eq=False)
class SimResult:
    """A simulated path.

    :attr:`states` and :attr:`observables` are dicts keyed by variable name.
    :attr:`X and :attr:`y` are the matrices they are cut from in canonical column order.
    :attr:`regimes` and :attr:`diagnostics` are OccBin results and raise on a model that
    has no constraints.
    """

    var_names: Sequence[str]
    X: NDF
    shocks: NDF
    observable_names: Sequence[str] = ()
    y: NDF | None = None
    _regimes: NDI | None = None
    _diagnostics: OccBinDiagnostics | None = None

    @cached_property
    def states(self) -> dict[str, NDF]:
        """Each model variable's path, as a column view of ``X``."""
        return {name: self.X[:, i] for i, name in enumerate(self.var_names)}

    @cached_property
    def observables(self) -> dict[str, NDF]:
        """Each observable's path, as a column view of ``y``."""
        if self.y is None:
            raise ValueError(
                "This simulation did not compute observables. Call sim() with "
                "observables=True."
            )
        return {name: self.y[:, i] for i, name in enumerate(self.observable_names)}

    @property
    def is_piecewise(self) -> bool:
        """Whether this path came from an OccBin solve."""
        return self._regimes is not None

    @property
    def regimes(self) -> NDI:
        """``(T, H)`` accepted regime guess per period.

        ``H`` is the longest check-ahead horizon any period used, so a period
        that settled sooner is padded. Column 0 is the regime realized at that
        date.
        """
        if self._regimes is None:
            raise ValueError(
                "Regimes are an OccBin result; this model has no constraints."
            )
        return self._regimes

    @property
    def diagnostics(self) -> OccBinDiagnostics:
        """Per-period convergence record behind :attr:`regimes`."""
        if self._diagnostics is None:
            raise ValueError(
                "Diagnostics are an OccBin result; this model has no constraints."
            )
        return self._diagnostics
