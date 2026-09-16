"""Shock generator and Array-shock handlers for DSGE simulations."""

from scipy.stats import (
    norm,
    multivariate_normal as mnorm,
    t,
    multivariate_t as mt,
    uniform,
)
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._multivariate import multi_rv_generic
import numpy as np
from numpy import asarray, ndarray, float64, random, zeros, generic
from numpy.linalg import cholesky, eigh, LinAlgError
from numpy.typing import NDArray
from typing import Any, Callable, Literal, Mapping, TypedDict, cast, get_args
import copy
import warnings

#: The built-in families a spec may name.
ShockDistribution = Literal["norm", "t", "uni"]


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
    state = random.RandomState(seed)
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
    """
    loc = (
        kwargs["mean"]
        if "mean" in kwargs
        else kwargs.get("loc", np.zeros(width, dtype=float64))
    )
    return asarray(loc, dtype=float64).reshape(width)


def _draw_normal(
    T: int, seed: int | None, loc: float | ndarray, factor: ndarray
) -> ndarray:
    """``(T, width)`` normals with mean ``loc`` and covariance ``factor @ factor.T``.

    One expression covers every width: a scalar standard deviation is the 1x1
    factor, which is how the native draw already reads both cases.
    """
    z = random.default_rng(seed).standard_normal((T, factor.shape[0]))
    return cast(ndarray, (loc + z @ factor.T).astype(float64))


def _draw_t(
    T: int, seed: int | None, df: float, loc: float | ndarray, factor: ndarray
) -> ndarray:
    """``(T, width)`` Student-t draws, as a normal scaled by a chi-square.

    ``factor`` scales the Gaussian core, so the drawn covariance is
    ``factor @ factor.T`` only up to the t's own ``df/(df - 2)`` inflation.
    """
    k = factor.shape[0]
    rng = random.default_rng(seed)
    z = rng.standard_normal((T, k)) @ factor.T
    g = rng.chisquare(df, size=T) / df
    return cast(ndarray, (loc + z / np.sqrt(g)[:, None]).astype(float64))


def _draw_uniform(
    T: int, seed: int | None, loc: float | ndarray, factor: ndarray
) -> ndarray:
    """``(T, width)`` uniforms centred on ``loc``.

    The standardized variate is ``U(-sqrt(3), sqrt(3))``, so a 1x1 ``factor`` of ``sig``
    results in a standard deviation ``sig``.
    """
    u = random.default_rng(seed).random((T, factor.shape[0])) - np.sqrt(3.0)
    return cast(ndarray, (loc + u @ factor.T).astype(float64))


class Shock:
    """Shock generator specification for a simulation run.

    Parameters
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        Distribution to draw shocks from. Can be a family name ("norm", "t",
        "uni") or a scipy.stats distribution object. Alternatively, a custom class
        implementing ``rvs`` can be passed in. Mutually exclusive with ``path``.
    path : NDArray[float64] | None
        Pre-generated shock array to use instead of drawing from a distribution.
        Mutually exclusive with ``dist``.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_kwargs : dict | None
        Optional keyword arguments for the distribution.

    Attributes
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        The configured distribution.
    path : NDArray[float64] | None
        The pre-generated shock array.
    seed : int | None
        The configured random seed.
    dist_kwargs : dict | None
        The configured keyword arguments.

    Notes
    -----
    A ``Shock`` names no shock variable of its own: it is an unbound template
    until one of :meth:`at`, :meth:`joint`, or :meth:`independent` binds a copy
    of it to one or more targets. The same template therefore serves any model,
    whatever that model names its shocks.

    The bind also decides where the scale comes from. :meth:`joint` draws one
    entry from the targets' covariance block, so the correlations declared in
    ``calibration.shock_corr`` apply. :meth:`independent` draws one entry per
    target from that target's standard deviation alone, so those correlations
    do not. Neither reads a covariance off the ``Shock``; both read it off the
    model's calibration.
    """

    def __init__(
        self,
        dist: ShockDistribution | rv_generic | multi_rv_generic | None = None,
        seed: int | None = 0,
        dist_kwargs: dict | None = None,
    ) -> None:
        # A Shock is a horizon-independent distribution spec: the number of
        # periods ``T`` is supplied by the caller at generation time, not baked
        # in here. The simulation is the single authority on its own horizon.
        self.dist = dist
        self.seed = seed
        self.dist_kwargs = dict(dist_kwargs) if dist_kwargs is not None else {}

        # Binding Slot (post-construction)
        self._target: tuple[str, ...] | None = None

    @property
    def target(self) -> tuple[str, ...]:
        """The shock variables this spec drives, once it has been bound."""
        if self._target is None:
            raise ValueError(
                "This ``Shock`` instance has not been bound to any variables "
                "yet. Use ``.at(key)`` for a single shock, ``.joint(*keys)`` to "
                "draw several from their calibrated covariance, or "
                "``.independent(*keys)`` to draw each from its own standard "
                "deviation."
            )
        return self._target

    @property
    def is_bound(self) -> bool:
        """Whether this spec has been bound to any shock variables."""
        return self._target is not None

    def _bind(self, keys: tuple[str, ...]) -> "Shock":
        """A copy of this spec bound to ``keys``.

        Binding copies rather than mutating, which is what lets one template be
        bound many times: ``[s.at("e_g"), s.at("e_z")]`` is two specs, not one
        object rebound twice. ``path`` is shared by reference, since it is
        read-only bulk data, while ``dist_kwargs`` is copied so two binds of one
        template cannot drift into each other.

        The copy skips ``__init__``, so a ``path`` spec carrying an ignored seed
        warns once where the user wrote it rather than again at every bind.
        """
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
            Whether to advance the seed by one per copy. Copies sharing a seed
            draw the same path on the Python route, so the default keeps them
            apart. An unseeded template has nothing to offset.

        Returns
        -------
        list[Shock]
            One bound copy per target, in the order given. Slicing the result is
            a valid spec, since every element stands alone.
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

        ``multivar`` is the arity of the entry being resolved, which the spec key
        fixes. The native families no longer branch on it, since one factor
        expression covers both widths; it survives for the scipy route, which
        hands scipy a covariance and picks a different distribution object and a
        different keyword per arity.

        The returned callable is ``f(factor, seed)``, returning ``(T, width)``.
        ``factor`` is the entry's scale in every family and at every width: the
        1x1 holding a standard deviation, or the covariance block's factor.
        Resolving the family, its keyword arguments, and (on the scipy route)
        the distribution object costs the same whether one path or a hundred
        thousand are drawn, so callers that redraw under varying seeds resolve
        once and call many times.

        Family validation is eager: an unknown family, a Student-t without
        ``df``, or a multivariate uniform raises here rather than at draw time.
        """
        if self.dist is None:
            raise ValueError("Distribution must be specified to draw shocks.")
        if "scale" in self.dist_kwargs:
            raise ValueError(
                "The generator function returns a callable that takes scale as an argument."
                " Please adjust `sig_` variables in the config to change the distribution scale."
                " Alternatively, the scale parameter in simulation and irf functions are multiplied directly with the shocks generated."
            )

        kwargs = self.dist_kwargs.copy()

        # A linear map of independent uniforms is not uniform in its margins,
        # so the factor form cannot express a grouped uniform. Refused here,
        # where the arity is known, rather than at the first draw.
        if self.dist == "uni" and multivar:
            raise NotImplementedError(
                "Multivariate uniform shocks are not implemented."
            )

        # A named family is drawn by the numpy fast paths, a live distribution
        # object by its own ``.rvs``. The route follows what ``dist`` is and
        # nothing else: every scipy family takes its parameters by keyword, so no
        # parameter a caller supplies can change which implementation draws it.
        if isinstance(self.dist, str):
            return self._numpy_draw_fn(T, kwargs)

        # A grouped object is handed its covariance under the keyword
        # ``multivariate_normal`` declares. An object that names it otherwise
        # (``multivariate_t`` calls it ``shape``) is not supported here and says
        # so through scipy rather than silently drawing something else.
        scale_key = "cov" if multivar else "scale"
        dist = self._get_dist(multivar)

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

    def _numpy_draw_fn(self, T: int, kwargs: dict) -> ShockDrawFn:
        """Resolve a named family onto the numpy Generator fast paths.

        One closure per family: the factor carries the arity, so width 1 and a
        grouped block take the same expression, the way the native draw does.
        """
        if self.dist == "norm":
            return lambda loc, factor, seed: _draw_normal(T, seed, loc, factor)

        if self.dist == "t":
            if "df" not in kwargs:
                raise ValueError("Student-t shocks require 'df' in dist_kwargs.")
            df = kwargs["df"]
            return lambda loc, factor, seed: _draw_t(T, seed, df, loc, factor)

        if self.dist == "uni":
            return lambda loc, factor, seed: _draw_uniform(T, seed, loc, factor)

        raise ValueError(f"Unknown shock distribution family: {self.dist!r}")

    def _get_dist(self, multivar: bool) -> rv_generic | multi_rv_generic:
        """The distribution object the scipy route draws through.

        Only a live object reaches here: a named family is drawn by the numpy
        fast paths, so the built-in names never take this route.
        """
        del multivar
        dist = self.dist
        if not isinstance(dist, rv_generic | multi_rv_generic):
            raise TypeError(
                f"dist must be one of {list(get_args(ShockDistribution))} or a "
                f"scipy.stats distribution object; got {type(dist).__name__}."
            )
        return dist

    def to_dict(self) -> ShockParameters:
        """Serialize a generator-style Shock to a JSON-able dict.

        Only the generator form is representable: a string ``dist`` identifier
        (``"norm"``/``"t"``/``"uni"``) and no materialized ``shock_arr``. A live
        scipy distribution object cannot be faithfully reproduced from JSON, and
        a placed shock array is bulk data that belongs with the parquet members,
        not the pipeline spec.
        """
        if not isinstance(self.dist, str):
            raise TypeError(
                "Only string-identified distributions ('norm'/'t'/'uni') are "
                "serializable; got a live scipy distribution object."
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

        A ``multivar`` or ``dist_args`` written by an older release is ignored:
        the spec key the shock is filed under fixes its arity, and every scipy
        family takes its parameters by keyword, so a positional list carries
        nothing a keyword does not.
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

    def __init__(self, path: NDArray[float64], *keys: str) -> None:
        self.path = path
        self.target = tuple(keys)

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
