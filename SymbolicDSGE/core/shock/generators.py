"""Shock generator and Array-shock handlers for DSGE simulations."""

from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._multivariate import multi_rv_generic
import numpy as np
from numpy import asarray, ndarray, float64, random, generic
from numpy.linalg import cholesky, eigh, LinAlgError
from numpy.typing import NDArray
from typing import Any, Callable, Literal, Mapping, TypedDict, Sequence, cast, get_args
import copy

#: The built-in families a spec may name.
ShockDistribution = Literal["norm", "t", "uni", "exp", "gamma", "beta"]


# A family-resolved draw: ``(loc, factor, seed) -> (T, width)``, computing
# ``loc + factor @ v`` over the family's standardized variate. Both spec numbers
# are arguments rather than closed-over state, so the resolver is the only place
# that reads a spec and the two draw routes cannot disagree about it. See
# :meth:`Shock.draw_fn` for the contract each argument carries.
ShockDrawFn = Callable[
    [NDArray[float64], NDArray[float64], int | None],
    NDArray[float64],
]


class ShockParameters(TypedDict):
    """Parameters for a shock generator specification.

    Attributes
    ----------
    target : tuple[str, ...]
        Names of the shock variables this spec drives.
    dist : ShockDistribution
        Name of the family the shocks are drawn from.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_kwargs : dict[str, Any]
        Keyword arguments for the distribution.

    """

    target: tuple[str, ...]
    dist: ShockDistribution
    seed: int | None
    dist_kwargs: dict[str, Any]


class ShockPathParameters(TypedDict):
    """Parameters for a pre-generated shock array.

    Attributes
    ----------
    target : tuple[str, ...]
        Names of the shock variables this spec drives.
    path : NDArray[float64]
        Pre-generated shock array to use instead of drawing from a distribution.
        The shape must be (T, width), where T is the number of time periods and
        width is the number of shock variables.

    """

    target: tuple[str, ...]
    path: NDArray[float64]


def abstract_shock_array(
    T: int,
    seed: int | None,
    dist: rv_generic | multi_rv_generic,
    **dist_kwargs: object,
) -> ndarray:
    """
    Generate an array of shocks based on a specified distribution.

    Parameters
    ----------
    T (int): The number of time periods.
    dist: A scipy.stats distribution object (e.g., norm, t, uniform).
    **dist_kwargs: Keyword arguments for the distribution.

    Returns
    -------
    np.ndarray: An array of shocks of length T.
    """
    # `RandomState` seeds from 32-bit words.
    # Both deterministic bitops fix the low and high 32 bits
    # of a 64-bit seed, producung 2 words for the 32-bit engine.
    state = random.RandomState(
        None
        if seed is None
        else np.array([seed & 0xFFFFFFFF, seed >> 32], dtype=np.uint32)
    )
    shocks = dist.rvs(size=T, random_state=state, **dist_kwargs)  # type: ignore
    return asarray(shocks, dtype=float64)


# --- numpy Generator fast paths --------------------------------------------
# scipy's generic ``.rvs`` is ~9x slower than ``np.random.Generator`` here (and
# ``multivariate_normal.rvs`` re-factorizes the covariance every call). These
# draw the known shock families directly off a Generator; ``abstract_shock_array``
# above stays as the fallback for arbitrary scipy distribution objects.


def _gaussian_factor(cov: ndarray) -> ndarray:
    """A factor ``F`` with ``F @ F.T == cov``.

    Cholesky on the common positive-definite path, with an eigh-based fallback so
    positive-semidefinite (but not strictly PD) covariances still sample -- the
    robustness scipy's ``_PSD`` gives us, without scipy.
    """
    try:
        return cholesky(cov)
    except LinAlgError:
        w, V = eigh(cov)
        return cast(ndarray, V * np.sqrt(np.clip(w, 0.0, None)))


def resolve_loc(kwargs: Mapping[str, Any], width: int) -> ndarray:
    """The location one spec declares, as a ``width``-long vector.

    The split draw paths read two names for one parameter, ``mean`` on the
    multivariate families and ``loc`` on the univariate ones, because each
    mirrored the scipy call it wrapped. One expression now covers both widths, so
    both spellings resolve here and ``mean`` wins when a spec carries both.

    The resolver calls this once per entry and the result travels on the entry,
    which is what keeps the Python draw and the native lowering from reading the
    same keyword arguments under two different rules.

    Any shape holding one value per variable is taken, since a location read out
    of a matrix calculation arrives as a row or a column as readily as flat.
    """
    name = "mean" if "mean" in kwargs else "loc"
    loc = asarray(kwargs.get(name, np.zeros(width)), dtype=float64)
    if loc.size != width:
        raise ValueError(
            f"`{name}` needs one value per variable in the entry. "
            f"Expected {width}, got {loc.size}. "
            "Omit the keyword for a zero default."
        )
    return loc.reshape(width)


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
    elif not isinstance(dist, rv_generic | multi_rv_generic):
        raise TypeError(
            f"dist must be one of {list(get_args(ShockDistribution))} or a "
            f"scipy.stats distribution object; got {type(dist).__name__}."
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


def validate_shock_family(
    dist: object,
    width: int,
    dist_kwargs: Mapping[str, Any],
) -> None:
    """Check a spec against what its family requires at the width it drives.

    Runs at bind. Scipy distributions are not checked since their parameterization
    requirements are not known to the library.
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


class Shock:
    """Shock generator specification for a simulation run.

    Parameters
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        Distribution to draw shocks from. Can be a family name ("norm", "t",
        "uni", "exp", "gamma", "beta") or a scipy.stats distribution object.
        Alternatively, a custom class implementing ``rvs`` can be passed in.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_kwargs : dict | None
        Optional keyword arguments for the distribution.

    Attributes
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        The configured distribution.
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
        dist: ShockDistribution | rv_generic | multi_rv_generic,
        seed: int | None = 0,
        dist_kwargs: dict | None = None,
    ) -> None:
        # A Shock is a horizon-independent distribution spec: the number of
        # periods ``T`` is supplied by the caller at generation time, not baked
        # in here. The simulation is the single authority on its own horizon.
        self.dist = dist
        self.seed = seed
        self.dist_kwargs = dict(dist_kwargs) if dist_kwargs is not None else {}
        _validate_dist(self.dist, self.dist_kwargs)

        # Binding Slot (post-construction)
        self._target: tuple[str, ...] | None = None

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

    def _bind(self, keys: tuple[str, ...]) -> "Shock":
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
        validate_shock_family(self.dist, len(keys), self.dist_kwargs)
        bound = copy.copy(self)
        bound.dist_kwargs = dict(self.dist_kwargs)
        bound._target = keys
        return bound

    def joint(self, *keys: str) -> "Shock":
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

    def independent(self, *keys: str, offset_seeds: bool = True) -> "list[Shock]":
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

    def draw_fn(self, T: int, multivar: bool) -> ShockDrawFn:
        """Resolve the distribution family once for a ``T``-period horizon.

        ``multivar`` is the arity of the entry being resolved. It's required
        for the scipy route while native handles both arities uniformly.

        The returned callable is ``f(loc, factor, seed)``, returning ``(T, width)``.
        ``factor`` is the entry's scale with standard deviations represented as a
        1x1.
        """
        kwargs = self.dist_kwargs.copy()

        # A grouped object is handed its covariance under the keyword
        # ``multivariate_normal`` declares. An object that names it otherwise
        # (``multivariate_t`` calls it ``shape``) is not supported here and says
        # so through scipy rather than silently drawing something else.
        scale_key = "cov" if multivar else "scale"
        dist = cast(rv_generic | multi_rv_generic, self.dist)

        def _scipy_draw(
            loc: NDArray[float64],
            factor: NDArray[float64],
            seed: int | None,
        ) -> NDArray[float64]:
            # scipy owns its own factorization and is parameterized by the
            # second moment, so hand back what the factor came from. The location
            # rides in ``kwargs`` on this route, which is where scipy wants it.
            del loc
            s: float | NDArray[float64] = (
                factor @ factor.T if multivar else float(factor[0, 0])
            )
            drawn = abstract_shock_array(
                T,
                seed,
                dist,
                **{**kwargs, scale_key: s},
            )
            return drawn.reshape(T, -1)

        return _scipy_draw

    def to_dict(self) -> ShockParameters:
        """Serialize a generator-style Shock to a JSON-able dict.

        Only the generator form is representable: a string ``dist`` identifier
        and no materialized ``path``. A live scipy distribution object cannot be
        faithfully reproduced from JSON, and a placed shock array is bulk data
        that belongs with the parquet members, not the pipeline spec.
        """
        if not isinstance(self.dist, str):
            raise TypeError(
                "Only string-identified distributions "
                f"({list(get_args(ShockDistribution))}) are serializable; got a "
                "live scipy distribution object."
            )
        return ShockParameters(
            target=self.target,
            dist=self.dist,  # pyright: ignore
            seed=None if self.seed is None else int(self.seed),
            dist_kwargs={k: _jsonable(v) for k, v in self.dist_kwargs.items()},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Shock":
        """Rebuild a generator-style Shock from :meth:`to_dict` output.

        A ``multivar`` or ``dist_args`` written by an older release is ignored.
        Arity is fixed by the entry's own ``target``, and every scipy family
        takes its parameters by keyword; a positional list carries nothing a
        keyword does not.
        """
        dist = data["dist"]
        if dist not in get_args(ShockDistribution):
            raise ValueError(
                f"Shock.from_dict expects one of "
                f"{list(get_args(ShockDistribution))}, got {dist!r}."
            )
        seed = data.get("seed")
        return cls(
            dist=cast(ShockDistribution, dist),
            seed=None if seed is None else int(seed),
            dist_kwargs=dict(data.get("dist_kwargs") or {}),
        ).joint(*data["target"])

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

    def to_dict(self) -> ShockPathParameters:
        """Serialize a ShockPath to a JSON-able dict."""
        return ShockPathParameters(
            target=self.target,
            path=_jsonable(self.path),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShockPath":
        """Rebuild a ShockPath from :meth:`to_dict` output."""
        path = asarray(data["path"], dtype=float64)
        keys = tuple(data["target"])
        return cls(path, *keys)

    def __repr__(self) -> str:
        return f"ShockPath({','.join(self.target)!r}, T={self.path.shape[0]})"


def _jsonable(value: Any) -> Any:
    """Coerce shock arg/kwarg values into JSON-serializable form."""
    if isinstance(value, ndarray):
        return value.tolist()
    if isinstance(value, generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value
