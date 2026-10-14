"""The standardized quantile table every non-builtin shock distribution reduces to.

A distribution enters as a quantile function and leaves as :class:`DistributionTable`:
its inverse CDF sampled on a fixed grid, centered and scaled so the table has mean
zero and unit variance under the measure the kernel samples it with. Scale then comes
from the entry's factor alone, as it already does for the families the kernel draws
analytically.

The grid is uniform in ``z = Phi^-1(u)`` rather than in ``u``. Spacing the knots in
probit space resolves the tails a uniform grid truncates, and it makes the draw an
index rather than a search: the kernel draws a standard normal, clamps it, and reads
the knot it lands between.

The table is the whole representation. Nothing here holds the distribution it came
from, and :attr:`DistributionTable.provenance` is a label for a reader, never a recipe
to rebuild from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypedDict, Protocol

import numpy as np
from numpy import asarray, float64
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr
from scipy.stats._distn_infrastructure import rv_discrete

NDF = NDArray[float64]

#: Knots per table, and the probit-space half-width they span. Neither is a knob: a
#: caller has no basis to choose them and both are load-bearing for accuracy. ``zmax``
#: of 6 clamps about one draw in 500 million, and the knot there already sits far into
#: a fat tail (-76.8 standardized for a Student-t with 5 degrees of freedom).
KNOTS = 1024
ZMAX = 6.0

#: A continuous quantile function gives one distinct knot per grid point; a staircase
#: gives as many as its support has points. This only catches a ``discrete`` flag the
#: knots contradict.
_TIE_FRACTION = 8


class QuantileDist(Protocol):
    def ppf(self, q: NDF, /, *args: Any, **kwds: Any) -> ArrayLike: ...


class DistributionTableParameters(TypedDict):
    """Serialized form of a :class:`DistributionTable`.

    Attributes
    ----------
    q : list[float]
        Standardized knot values, in grid order.
    zmax : float
        Probit-space half-width the knots span.
    discrete : bool
        Whether the knots are a staircase read by nearest knot rather than interpolated.
    source_mean : float
        Mean removed when standardizing.
    source_std : float
        Standard deviation divided out when standardizing.
    provenance : str
        What the table was built from, for a reader. Never read back as a recipe.

    """

    q: list[float]
    zmax: float
    discrete: bool
    source_mean: float
    source_std: float
    provenance: str


def _knot_probabilities(zmax: float, knots: int) -> tuple[NDF, NDF]:
    """The grid in probit space and the probabilities it maps to."""
    z = np.linspace(-zmax, zmax, knots, dtype=float64)
    return z, asarray(ndtr(z), dtype=float64)


def _cell_weights(z: NDF) -> NDF:
    """Probability a nearest-knot read assigns each knot.

    Knot ``i`` is returned for ``z`` within half a step of ``z[i]``, so its weight is
    the normal measure of that cell. The end knots absorb the clamped tails, which is
    what makes these sum to one.
    """
    half = (z[1] - z[0]) / 2.0
    edges = asarray(ndtr(z + half), dtype=float64)
    w = np.empty_like(edges)
    w[0] = edges[0]
    w[1:] = np.diff(edges)
    w[-1] += 1.0 - edges[-1]
    return w


def _table_moments(z: NDF, u: NDF, q: NDF, discrete: bool) -> tuple[float, float]:
    """Mean and variance of the table under a clamped standard normal.

    The moments of the object actually sampled, not of the distribution it was drawn
    from. Standardizing by these is what makes the table exactly unit-variance as the
    kernel sees it, rather than within an error that shrinks in the knot count.
    """
    # The rule has to be the one `DistributionTable.evaluate` reads with:
    # trapezoidal between knots where it interpolates, and the nearest knot's own cell
    # where it does not. Averaging over the knots instead would weight them equally,
    # which the probit grid does not: the same knots give a variance of 12.0 for a
    # standard normal.

    if discrete:
        w = _cell_weights(z)
        m1 = float(np.sum(w * q))
        m2 = float(np.sum(w * q * q))
        return m1, m2 - m1 * m1

    du = np.diff(u)
    lo, hi = q[:-1], q[1:]
    # The draw clamps outside the grid, so the end knots carry the tail mass.
    mass_lo, mass_hi = float(u[0]), float(1.0 - u[-1])
    m1 = float(np.sum(du * (lo + hi) / 2.0) + mass_lo * q[0] + mass_hi * q[-1])
    m2 = float(
        np.sum(du * (lo * lo + lo * hi + hi * hi) / 3.0)
        + mass_lo * q[0] ** 2
        + mass_hi * q[-1] ** 2
    )
    return m1, m2 - m1 * m1


def _is_discrete(dist: QuantileDist) -> bool:
    """Whether a scipy distribution, frozen or not, is a discrete family."""
    return isinstance(getattr(dist, "dist", dist), rv_discrete)


def _scipy_provenance(dist: QuantileDist, kwargs: Mapping[str, Any]) -> str:
    """A label naming the family and the parameters baked into the knots.

    A frozen distribution carries its own, so read them rather than reporting the
    bare family a caller would have to guess the shape of.
    """
    name = getattr(getattr(dist, "dist", dist), "name", None) or type(dist).__name__
    shown = [repr(a) for a in getattr(dist, "args", ())]
    merged = {**getattr(dist, "kwds", {}), **kwargs}
    shown += [f"{k}={v!r}" for k, v in sorted(merged.items())]
    if not shown:
        return f"scipy.stats.{name}"
    return f"scipy.stats.{name}({', '.join(shown)})"


@dataclass(frozen=True, eq=False)
class DistributionTable:
    """A standardized inverse CDF, sampled on a probit-space grid.

    Construct one from a quantile function with :meth:`from_callable`, or from a scipy
    distribution with :meth:`from_scipy`.

    Immutable, and shared by reference: one table serves every bind of a ``Shock``
    template.

    Parameters
    ----------
    q : NDF
        Standardized knot values, in grid order. Its length is the knot count.
    zmax : float
        Probit-space half-width the knots span.
    discrete : bool
        Read the nearest knot instead of interpolating between knots.
    source_mean : float
        Mean removed when standardizing.
    source_std : float
        Standard deviation divided out when standardizing.
    provenance : str
        What it was built from. Descriptive only.

    """

    q: NDF
    zmax: float = ZMAX
    discrete: bool = False
    source_mean: float = 0.0
    source_std: float = 1.0
    provenance: str = ""

    def __post_init__(self) -> None:
        q = asarray(self.q, dtype=float64)
        if q.ndim != 1 or q.shape[0] < 2:
            raise ValueError(
                f"A distribution table needs a one-dimensional run of at least two "
                f"knots; got shape {tuple(q.shape)}."
            )
        if not np.all(np.isfinite(q)):
            bad = int(np.flatnonzero(~np.isfinite(q))[0])
            raise ValueError(
                f"Knot {bad} of {q.shape[0]} is not finite. A quantile function must "
                "be finite across the grid; a distribution with unbounded quantiles "
                "cannot be tabulated accurately."
            )
        if not np.all(np.diff(q) >= 0.0):
            bad = int(np.flatnonzero(np.diff(q) < 0.0)[0])
            raise ValueError(
                f"Knots decrease at index {bad}. A quantile function must be "
                "non-decreasing by definition."
            )
        object.__setattr__(self, "q", q)

        if not np.isfinite(self.zmax) or self.zmax <= 0.0:
            raise ValueError(f"zmax must be finite and positive; got {self.zmax!r}.")
        if not np.isfinite(self.source_std) or self.source_std <= 0.0:
            raise ValueError(
                f"source_std must be finite and positive; got {self.source_std!r}."
            )
        if not np.isfinite(self.source_mean):
            raise ValueError(f"source_mean must be finite; got {self.source_mean!r}.")
        if not self.discrete:
            distinct = int(np.unique(q).size)
            if distinct * _TIE_FRACTION < q.shape[0]:
                raise ValueError(
                    "Given distribution appears to be discrete, but `discrete=False` was passed. "
                    f"Only {distinct} of {q.shape[0]} knots are distinct, where a continuous "
                    "quantile function implies all knots to be distinct. Pass discrete=True, which "
                    "reads the nearest knot instead of interpolating between knots, or use a coninuous "
                    "quantile function."
                )

    @property
    def knots(self) -> int:
        """Number of knots, which is what fixes the grid's resolution."""
        return int(self.q.shape[0])

    @property
    def dz(self) -> float:
        """Probit-space spacing between knots."""
        return 2.0 * self.zmax / float(self.knots - 1)

    @property
    def z(self) -> NDF:
        """The grid the knots sit on, for inspection."""
        return np.linspace(-self.zmax, self.zmax, self.knots, dtype=float64)

    @property
    def inv_dz(self) -> float:
        """Knots per unit of ``z``, the factor a lookup multiplies by.

        Mirrors the native kernel's pre-computed ``inv_dx`` to eliminate
        floating-point discrepancies between the two.
        """
        return float(self.knots - 1) / (2.0 * self.zmax)

    def evaluate(self, z: NDF | float) -> NDF:
        """The standardized variate a standard normal draw ``z`` selects.

        The reference for what the kernel computes: clamp to the grid, locate the knot,
        then interpolate between the bracketing knots (or take the nearest one when the
        table is a staircase).

        Parameters
        ----------
        z : NDF | float
            Standard normal draws, in any shape.

        Returns
        -------
        NDF
            Standardized variates, in the shape of ``z``.

        """
        zz = np.clip(asarray(z, dtype=float64), -self.zmax, self.zmax)
        pos = (zz + self.zmax) * self.inv_dz
        if self.discrete:
            idx = np.clip(np.floor(pos + 0.5).astype(np.int64), 0, self.knots - 1)
            return self.q[idx]
        # One short of the end, since the interpolation reads the next knot.
        idx = np.clip(np.floor(pos).astype(np.int64), 0, self.knots - 2)
        frac = pos - idx
        lo = self.q[idx]
        return lo + frac * (self.q[idx + 1] - lo)

    @classmethod
    def from_callable(
        cls,
        ppf: Callable[[NDF], ArrayLike],
        *,
        discrete: bool = False,
        provenance: str = "",
    ) -> DistributionTable:
        """Tabulate a quantile function and standardize it.

        Parameters
        ----------
        ppf : Callable[[NDF], ArrayLike]
            Quantile function over ``(0, 1)``, taking an array and returning one.
        discrete : bool
            Whether the function is a staircase. Nothing about a bare callable says
            so, which is why it is declared rather than detected; the knots are
            checked against the declaration.
        provenance : str
            What this was built from, for a reader.

        Returns
        -------
        DistributionTable
            A table with mean zero and unit variance under the sampling measure.

        Raises
        ------
        ValueError
            When ``ppf`` does not return one value per probability, or when the knots
            carry no spread to standardize by.

        Notes
        -----
        The grid is bounded, so the knots have a finite variance whatever the
        distribution behind them does: a Cauchy quantile function tabulates here
        without complaint, standardized by the spread its outermost knots happen to
        have. A caller on this route owns that question.
        :meth:`from_scipy` does not, since it can ask.

        """
        z, u = _knot_probabilities(ZMAX, KNOTS)
        raw = asarray(ppf(u), dtype=float64)
        if raw.shape != u.shape:
            raise ValueError(
                f"The quantile function returned shape {tuple(raw.shape)} for "
                f"{KNOTS} probabilities. It must take an array and return one value "
                "per probability."
            )

        mean, var = _table_moments(z, u, raw, discrete)
        if not np.isfinite(var) or var <= 0.0:
            raise ValueError(
                f"The tabulated quantile function has variance {var!r}, so it cannot "
                "be standardized. A shock distribution needs a finite, positive "
                "variance for the calibrated covariance to mean anything."
            )
        std = float(np.sqrt(var))
        return cls(
            q=(raw - mean) / std,
            zmax=ZMAX,
            discrete=discrete,
            source_mean=mean,
            source_std=std,
            provenance=provenance,
        )

    @classmethod
    def from_scipy(cls, dist: QuantileDist, /, **kwargs: Any) -> DistributionTable:
        """Tabulate a scipy distribution, frozen or parameterized by ``kwargs``.

        The one route that reads a distribution object. Discreteness is read off
        the object.

        Parameters
        ----------
        dist : Any
            A scipy distribution, or anything exposing a compatible ``ppf``.
        **kwargs : Any
            Shape parameters, baked into the knots. They do not survive as live
            parameters: a table is not re-parameterizable.

        Returns
        -------
        DistributionTable
            A table with mean zero and unit variance under the sampling measure.

        Raises
        ------
        TypeError
            When ``dist`` exposes no quantile function.
        ValueError
            When the family has no finite variance, which no shock covariance can be
            satisfied with.

        """
        ppf: Callable[[NDF], ArrayLike] | None = getattr(dist, "ppf", None)
        if not callable(ppf):
            raise TypeError(
                f"{type(dist).__name__} has no callable 'ppf'. A shock distribution "
                "is tabulated through its quantile function, which is the only "
                "method it needs to provide."
            )

        # The bounded grid cannot see an infinite variance, so take the family's own
        # word for it while there is an object to ask.
        stats_fn = getattr(dist, "stats", None)
        if callable(stats_fn):
            declared = float(np.asarray(stats_fn(moments="v", **kwargs), dtype=float64))
            if not np.isfinite(declared):
                raise ValueError(
                    f"{_scipy_provenance(dist, kwargs)} has variance {declared}. The "
                    "shock covariance specification cannot be satisfied by a "
                    "distribution with no finite variance."
                )

        def _ppf(u: NDF) -> ArrayLike:
            return ppf(u, **kwargs)

        return cls.from_callable(
            _ppf,
            discrete=_is_discrete(dist),
            provenance=_scipy_provenance(dist, kwargs),
        )

    def to_dict(self) -> DistributionTableParameters:
        """Serialize the table to the stored layout."""
        return DistributionTableParameters(
            q=[float(x) for x in self.q],
            zmax=float(self.zmax),
            discrete=bool(self.discrete),
            source_mean=float(self.source_mean),
            source_std=float(self.source_std),
            provenance=str(self.provenance),
        )

    @classmethod
    def from_dict(cls, data: DistributionTableParameters) -> DistributionTable:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            q=asarray(data["q"], dtype=float64),
            zmax=float(data["zmax"]),
            discrete=bool(data["discrete"]),
            source_mean=float(data["source_mean"]),
            source_std=float(data["source_std"]),
            provenance=str(data.get("provenance", "")),
        )

    def __repr__(self) -> str:
        what = self.provenance or "callable"
        kind = "discrete" if self.discrete else "continuous"
        return f"DistributionTable({what}, {kind}, knots={self.knots})"
