"""Container returned by ``SolvedModel.sim`` and ``SolvedModel.irf``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from numpy import float64, int64
from numpy.typing import NDArray

NDF = NDArray[float64]
NDI = NDArray[int64]


@dataclass(frozen=True)
class OccBinDiagnostics:
    """Per-period convergence record from ``occbin_solve``, one entry per period.

    ``T_used`` is the check-ahead horizon the period settled on, ``iters`` the
    guess-and-verify passes it took, and ``max_err`` the largest constraint
    violation left in the accepted guess. ``periodic`` is nonzero where a period
    was accepted as a cycle, carrying the code it would otherwise have raised.
    """

    T_used: NDI
    iters: NDI
    max_err: NDF
    periodic: NDI


@dataclass(frozen=True, eq=False)
class SimResult:
    """A simulated path, as series by name.

    :attr:`states` and :attr:`observables` are the two series mappings and are
    kept apart, so a name that is both a variable and an observable stays two
    distinct series. ``X`` and ``y`` are the matrices they are cut from, in
    canonical column order, held so a caller wanting the whole path does not
    have to reassemble one from the series.

    ``regimes`` and ``diagnostics`` are OccBin results and raise on a model that
    has no constraints. Both describe the shock realization this result came
    from rather than the model, so they belong here and not on ``SolvedModel``.

    Equality is identity: the fields are arrays, so a field-wise ``==`` would
    yield arrays rather than a bool.
    """

    var_names: Sequence[str]
    X: NDF
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

    def diagnostics(self) -> OccBinDiagnostics:
        """Per-period convergence record behind :meth:`regimes`."""
        if self._diagnostics is None:
            raise ValueError(
                "Diagnostics are an OccBin result; this model has no constraints."
            )
        return self._diagnostics
