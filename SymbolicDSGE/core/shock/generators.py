"""Shock generator and Array-shock handlers for DSGE simulations."""

from __future__ import annotations

from .table import DistributionTable, DistributionTableParameters, QuantileDist

import numpy as np
from numpy import asarray, ndarray, float64
from numpy.typing import ArrayLike, NDArray

from typing import Any, Callable, Literal, Mapping, TypedDict, Sequence, get_args
import copy

NDF = NDArray[float64]

#: A quantile function: probabilities in, quantiles out. ``table.py`` states
#: the same shape on :meth:`DistributionTable.from_callable`.
QuantileFn = Callable[[NDArray[float64]], ArrayLike]

#: The built-in families a spec may name.
ShockDistribution = Literal["norm", "t", "uni", "exp", "gamma", "beta"]


class Shock:
    """Shock generator specification for a simulation run.

    Parameters
    ----------
    dist : ShockDistribution | DistributionTable
        Distribution to draw shocks from. A family name ("norm", "t", "uni",
        "exp", "gamma", "beta") is drawn analytically by the kernel. Any callable
        `ppf`, or a class with `.ppf` (including a scipy distribution) can be tabulated
        with :meth:`from_ppf`, `DistributionTable.from_callable`, or
        `DistributionTable.from_scipy`.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_kwargs : dict | None
        Optional keyword arguments for the distribution.

    Attributes
    ----------
    dist : ShockDistribution | DistributionTable
        The resolved distribution, which is not always what was passed: a
        scipy object arrives here as the table it was tabulated into, and the
        object itself is not kept. A table cannot be re-parameterized, so
        changing ``dist_kwargs`` afterwards moves ``loc`` and nothing else.
    seed : int | None
        The configured random seed.
    dist_kwargs : dict | None
        The configured keyword arguments.

    Notes
    -----
    A ``Shock`` names no shock variable of its own: it is an unbound template
    until :meth:`joint` or :meth:`independent` binds a copy of it to one or more
    targets. The same template therefore serves any model, whatever that model
    names its shocks. A bound ``Shock`` cannot be rebound. A supplied path is
    not a ``Shock`` at all; it is a :class:`ShockPath`, which names its targets
    at construction.

    The bind also decides where the scale comes from. :meth:`joint` draws one
    entry from the targets' covariance block, so the correlations declared in
    ``calibration.shock_corr`` apply. :meth:`independent` draws one entry per
    target from that target's standard deviation alone, so those correlations
    do not. Neither reads a covariance off the ``Shock``; both read it off the
    model's calibration.
    """

    def __init__(
        self,
        dist: ShockDistribution | DistributionTable,
        seed: int | None = 0,
        **dist_kwargs: ArrayLike,
    ) -> None:
        # A Shock is a horizon-independent distribution spec: the number of
        # periods ``T`` is supplied by the caller at generation time, not baked
        # in here. The simulation is the single authority on its own horizon.
        self.dist = dist
        self.seed = seed
        self.dist_kwargs = dict(dist_kwargs)
        _validate_dist(dist, self.dist_kwargs)

        # Binding Slot (post-construction)
        self._target: tuple[str, ...] | None = None

    @classmethod
    def from_ppf(
        cls,
        dist: QuantileFn | QuantileDist,
        n_knots: int = 1024,
        discrete: bool = False,
        seed: int | None = 0,
        **dist_kwargs: ArrayLike,
    ) -> Shock:
        """A spec over a distribution the kernel has no family for.

        Anything that can produce quantiles is tabulated here and the object is
        then dropped: the table is the whole representation, so keeping both
        would leave two, only one of which survives a round trip. Shape
        parameters are baked into the knots, which is why a tabulated spec
        cannot be re-parameterized afterwards.

        Parameters
        ----------
        dist : QuantileFn | QuantileDist
            A quantile function with array input and output,
            or a class exposing one as ``.ppf``.
        n_knots : int
            Number of knots to tabulate. 1024 saturates most continuous distributions.
        discrete : bool
            Whether the distribution is discrete.
        seed : int | None
            Random seed for reproducibility. If None, a random seed is used.
        **dist_kwargs : ArrayLike
            Shape parameters, passed to the quantile function. ``loc`` is kept
            rather than baked in: standardizing removes the shift the knots
            carry, and the resolved entry applies it once afterwards.

        Returns
        -------
        Shock
            An unbound spec over the tabulated distribution.

        """
        fn: Callable[[ndarray], ArrayLike] = (
            (lambda u: dist.ppf(u, **dist_kwargs))
            if isinstance(dist, QuantileDist)
            else dist
        )

        table = DistributionTable.from_callable(fn, n_knots=n_knots, discrete=discrete)
        return cls(table, seed, **dist_kwargs)

    @property
    def target(self) -> tuple[str, ...]:
        """The shock variables this spec drives, once it has been bound."""
        if self._target is None:
            raise ValueError(
                "This `Shock` instance has not been bound to any variables "
                "yet. Use `.joint(*keys) to draw them from their calibrated "
                "covariance, or `.independent(*keys)` to draw each from its "
                "own standard deviation."
            )
        return self._target

    @property
    def is_bound(self) -> bool:
        """Whether this spec has been bound to any shock variables."""
        return self._target is not None

    def _bind(self, keys: tuple[str, ...]) -> Shock:
        """A copy of this spec bound to ``keys``.

        Binding copies rather than mutating, which is what lets one template be
        bound many times: ``s.independent("e_g", "e_z")`` is two specs, not one
        object rebound twice. ``dist_kwargs`` is copied with it, since two binds
        of one template must not drift into each other.

        Raises :class:`ValueError` when this spec is already bound. The family's
        own requirements are checked here as well, since a bind is the first
        point a width exists and the last one a caller cannot avoid.
        """
        if self.is_bound:
            raise ValueError(
                f"{self!r} is already bound to {self.target!r} and cannot rebind to {keys!r}. "
                "Maybe you passed a bound `Shock` in a mapping-style shock specification? "
                "If so, use a sequence-style spec or pass the unbound templates as values."
            )
        _validate_shock_family(self.dist, len(keys), self.dist_kwargs)
        bound = copy.copy(self)
        bound.dist_kwargs = dict(self.dist_kwargs)
        bound._target = keys
        return bound

    def joint(self, *keys: str) -> Shock:
        """Bind this spec to several shock variables, drawn together.

        The resolved entry spans every named target and draws through the factor
        of their block of the calibrated covariance, so the correlations declared
        in ``calibration.shock_corr`` between them apply.

        Parameters
        ----------
        *keys : str
            The shock variables this spec drives, in any order. The resolver
            sorts them into column order before building the block.

        Returns
        -------
        Shock
            A copy of this spec, bound to ``keys`` as one entry.

        Raises
        ------
        ValueError
            When ``keys`` is empty, or when this spec is already bound. Bind
            from the unbound template instead.
        """
        if not keys:
            raise ValueError("``joint`` needs at least one shock variable.")
        return self._bind(tuple(keys))

    def independent(self, *keys: str, offset_seeds: bool = True) -> list[Shock]:
        """Bind this spec to several shock variables, drawn separately.

        One copy per target, each resolving to a width-1 entry that draws
        through that target's own standard deviation. Any correlation
        ``calibration.shock_corr`` declares between the targets is not applied,
        which is the difference from :meth:`joint` and the reason both exist.

        Parameters
        ----------
        *keys : str
            The shock variables this spec drives, one copy each.
        offset_seeds : bool
            Whether to advance the seed by one per copy, so the copies carry
            distinct seeds rather than the template's. ``False`` does not make
            them draw alike: copies target different shocks, which separates
            their draws on its own. An unseeded template has nothing to offset.

        Returns
        -------
        list[Shock]
            One bound copy per target, in the order given. Slicing the result is
            a valid spec, since every element stands alone.

        Raises
        ------
        ValueError
            When this spec is already bound. Bind from the unbound template
            instead.
        """
        out: list[Shock] = []
        for i, key in enumerate(keys):
            bound = self._bind((key,))
            if offset_seeds and self.seed is not None:
                bound.seed = self.seed + i
            out.append(bound)
        return out

    def __repr__(self) -> str:
        return f"Shock({','.join(self.target)!r}, dist={self.dist!r})"


class ShockPath:
    """A pre-generated shock array for a simulation run.

    Parameters
    ----------
    path : NDArray[float64]
        Pre-generated shock array to use instead of drawing from a distribution.
        The shape must be (T, width), where T is the number of time periods and
        width is the number of shock variables.

    Attributes
    ----------
    path : NDArray[float64]
        The pre-generated shock array.
    *keys : str
        The shock variables this spec drives.
        Key order must match the order of columns in
        ``path``.
    """

    def __init__(
        self,
        path: NDArray[float64] | Sequence[float] | Sequence[Sequence[float]],
        *keys: str,
    ) -> None:
        self.path = asarray(path, dtype=float64)
        self.target = tuple(keys)

        try:
            self.path = self.path.reshape(self.path.shape[0], len(self.target))
        except ValueError:
            raise ValueError(
                f"Shock paths do not match the number of variables. "
                f"Expected {len(keys)} columns, but got {self.path.shape[1]}."
            )

    def __repr__(self) -> str:
        return f"ShockPath({','.join(self.target)!r}, T={self.path.shape[0]})"


class ShockParameters(TypedDict):
    """Parameters for a shock generator specification.

    Attributes
    ----------
    target : tuple[str, ...]
        Names of the shock variables this spec drives.
    dist : ShockDistribution | DistributionTable
        Name of the family the shocks are drawn from, or a tabulated distribution.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_kwargs : dict[str, Any]
        Keyword arguments for the distribution.

    """

    target: tuple[str, ...]
    dist: ShockDistribution | None
    table: DistributionTableParameters | None
    seed: int | None
    dist_kwargs: dict[str, Any]


class ShockPathParameters(TypedDict):
    """Parameters for a pre-generated shock array.

    Attributes
    ----------
    target : tuple[str, ...]
        Names of the shock variables this spec drives.
    path_id : str
        Identifier for the shock path. Hash-based over the data,
        Shocks with paths resolving to the same path_id restore their
        draws from the same array in a bundle.

    """

    target: tuple[str, ...]
    path_id: str


def _validate_dist(dist: object, dist_kwargs: Mapping[str, Any]) -> None:
    """Validate distribution and parameters before binding."""
    if dist is None:
        raise ValueError(
            "A distribution must be specified to draw shocks. "
            "Use `ShockPath` if you intend to supply a pre-generated "
            "shock array instead of drawing from a distribution."
        )
    if isinstance(dist, str):
        if dist not in get_args(ShockDistribution):
            raise ValueError(
                f"Unknown shock distribution family: {dist!r}. "
                f"Valid families: {list(get_args(ShockDistribution))}."
            )
    elif not isinstance(dist, DistributionTable):
        raise TypeError(
            f"dist must be one of {list(get_args(ShockDistribution))}; got "
            f"{type(dist).__name__}. A distribution object, or any quantile "
            "function, is tabulated through `Shock.from_ppf`, which is also "
            "where a discrete family is declared."
        )
    if "scale" in dist_kwargs:
        raise ValueError(
            "Shock scale comes from the model, not from dist_kwargs: adjust the "
            "shock std/corr variables in the config, or pass shock_scale to the "
            "simulation and irf functions, which multiply the drawn shocks directly."
        )


def _require_univariate(dist: str, width: int) -> None:
    """Raise if a family is not univariate at the width it drives."""
    if width > 1:
        raise NotImplementedError(f"Multivariate {dist!r} shocks are not implemented.")


def _validate_shock_family(
    dist: object,
    width: int,
    dist_kwargs: Mapping[str, Any],
) -> None:
    """Check a spec against what its family requires at the width it drives.

    Runs at bind, and only the named families have anything to check here: a
    tabulated distribution settled its parameters when its knots were built, and
    carries no width requirement, since one marginal serves every column of a
    grouped entry.
    """
    _validate_dist(dist, dist_kwargs)
    match dist:
        case "t":
            if "df" not in dist_kwargs:
                raise ValueError("Student-t shocks require 'df' in dist_kwargs.")

            if dist_kwargs["df"] <= 2.0 or not np.isfinite(dist_kwargs["df"]):
                raise ValueError(
                    "Student-t requires df > 2 for finite variance. "
                    "The shock covariance specification cannot be satisfied with "
                    f"df={dist_kwargs['df']}."
                )
        case "uni":
            _require_univariate(dist, width)  # pyright: ignore
        case "exp":
            _require_univariate(dist, width)  # pyright: ignore
        case "gamma":
            _require_univariate(dist, width)  # pyright: ignore
            if "a" not in dist_kwargs:
                raise ValueError(
                    "Gamma shocks require the shape parameter ('a') in dist_kwargs."
                )
            if dist_kwargs["a"] <= 0 or not np.isfinite(dist_kwargs["a"]):
                raise ValueError(
                    "Gamma shocks require finite and positive `a`. The shock covariance "
                    f"specification cannot be satisfied with `a={dist_kwargs['a']}`."
                )
        case "beta":
            _require_univariate(dist, width)  # pyright: ignore
            if "a" not in dist_kwargs or "b" not in dist_kwargs:
                raise ValueError(
                    "Beta shocks require the shape parameters ('a' and 'b') in dist_kwargs."
                )
            if (
                dist_kwargs["a"] <= 0
                or dist_kwargs["b"] <= 0
                or not np.isfinite(dist_kwargs["a"])
                or not np.isfinite(dist_kwargs["b"])
            ):
                raise ValueError(
                    "Beta shocks require finite and positive `a` and `b`. The shock "
                    f"covariance specification cannot be satisfied with `a={dist_kwargs['a']}` "
                    f"and `b={dist_kwargs['b']}`."
                )
