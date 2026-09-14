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
from typing import Any, Callable, Literal, Mapping, TypedDict, cast

ShockDistribution = Literal["norm", "t", "uni"]

# A family-resolved draw: ``(scale, seed, factor) -> (T,) | (T, k)``. See
# :meth:`Shock.draw_fn` for the contract each argument carries.
ShockDrawFn = Callable[
    [Any, int | None, NDArray[float64] | None],
    NDArray[float64],
]


class ShockParameters(TypedDict):
    """Parameters for a shock generator specification.

    Attributes
    ----------
    dist : ShockDistribution
        Distribution to draw shocks from. Can be a string identifier ("norm", "t", "uni").
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_args : list[Any]
        Positional arguments for the distribution.
    dist_kwargs : dict[str, Any]
        Keyword arguments for the distribution.

    """

    dist: ShockDistribution
    seed: int | None
    dist_args: list[Any]
    dist_kwargs: dict[str, Any]


def abstract_shock_array(
    T: int,
    seed: int | None,
    dist: rv_generic | multi_rv_generic,
    *dist_args: object,
    **dist_kwargs: object,
) -> ndarray:
    """
    Generate an array of shocks based on a specified distribution.

    Parameters
    ----------
    T (int): The number of time periods.
    dist: A scipy.stats distribution object (e.g., norm, t, uniform).
    *dist_args: Positional arguments for the distribution.
    **dist_kwargs: Keyword arguments for the distribution.

    Returns
    -------
    np.ndarray: An array of shocks of length T.
    """
    state = random.RandomState(seed)
    shocks = dist.rvs(size=T, random_state=state, *dist_args, **dist_kwargs)  # type: ignore
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


def _draw_normal(
    T: int, seed: int | None, mu: float | float64, sigma: float | float64
) -> ndarray:
    return random.default_rng(seed).normal(loc=mu, scale=sigma, size=T).astype(float64)


def _draw_normal_mv(
    T: int,
    seed: int | None,
    mean: ndarray | None,
    cov: ndarray,
    factor: ndarray | None = None,
) -> ndarray:
    cov = asarray(cov, dtype=float64)
    k = cov.shape[0]
    mean_vec = zeros(k, dtype=float64) if mean is None else asarray(mean, dtype=float64)
    F = _gaussian_factor(cov) if factor is None else factor
    z = random.default_rng(seed).standard_normal((T, k))
    return cast(ndarray, (mean_vec + z @ F.T).astype(float64))


def _draw_t(
    T: int,
    seed: int | None,
    df: float,
    loc: float | float64,
    scale: float | float64,
) -> ndarray:
    draws = random.default_rng(seed).standard_t(df, size=T)
    return (loc + scale * draws).astype(float64)


def _draw_t_mv(
    T: int,
    seed: int | None,
    df: float,
    loc: ndarray | None,
    shape: ndarray,
    factor: ndarray | None = None,
) -> ndarray:
    shape = asarray(shape, dtype=float64)
    k = shape.shape[0]
    loc_vec = zeros(k, dtype=float64) if loc is None else asarray(loc, dtype=float64)
    F = _gaussian_factor(shape) if factor is None else factor
    rng = random.default_rng(seed)
    z = rng.standard_normal((T, k)) @ F.T
    g = rng.chisquare(df, size=T) / df
    return cast(ndarray, (loc_vec + z / np.sqrt(g)[:, None]).astype(float64))


def _draw_uniform(
    T: int, seed: int | None, loc: float | float64, scale: float | float64
) -> ndarray:
    return (
        random.default_rng(seed)
        .uniform(low=loc, high=loc + scale, size=T)
        .astype(float64)
    )


class Shock:
    """Shock generator specification for a simulation run.

    Parameters
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        Distribution to draw shocks from. Can be a string identifier ("norm", "t",
        "uni") or a scipy.stats distribution object. Alternatively, a custom class
        implementing ``rvs`` can be passed in. If None, no distribution is specified.
    seed : int | None
        Random seed for reproducibility. If None, a random seed is used.
    dist_args : tuple
        Positional arguments for the distribution.
    dist_kwargs : dict | None
        Optional keyword arguments for the distribution.

    Attributes
    ----------
    dist : ShockDistribution | rv_generic | multi_rv_generic | None
        The configured distribution.
    seed : int | None
        The configured random seed.
    dist_args : tuple
        The configured positional arguments.
    dist_kwargs : dict | None
        The configured keyword arguments.
    """

    def __init__(
        self,
        dist: ShockDistribution | rv_generic | multi_rv_generic | None = None,
        seed: int | None = 0,
        dist_args: tuple = (),
        dist_kwargs: dict | None = None,
    ) -> None:
        # A Shock is a horizon-independent distribution spec: the number of
        # periods ``T`` is supplied by the caller at generation time, not baked
        # in here. The simulation is the single authority on its own horizon.
        self.dist = dist
        self.seed = seed
        self.dist_args = dist_args
        self.dist_kwargs = dist_kwargs if dist_kwargs is not None else {}

    # TODO: Pass through array if provided else generate based on dist

    def draw_fn(self, T: int, multivar: bool) -> ShockDrawFn:
        """Resolve the distribution family once for a ``T``-period horizon.

        ``multivar`` is the arity of the entry being resolved, which the spec key
        fixes: a key naming one shock draws a scalar standard deviation, a key
        naming several draws their covariance block. It is a required parameter
        rather than stored state because the key is the only authority on it.

        The returned callable is ``f(scale, seed, factor)``. Resolving the
        family, its keyword arguments, and (on the scipy route) the distribution
        object costs the same whether one path or a hundred thousand are drawn,
        so callers that redraw under varying seeds resolve once and call many
        times. ``factor`` is an optional precomputed matrix ``F`` with
        ``F @ F.T == scale`` for the multivariate families; passing it skips the
        per-call factorization of an unchanged covariance. The univariate
        families and the scipy route ignore it.

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

        # Known string families go through the numpy Generator fast paths. A raw
        # scipy distribution object (or a string with positional dist_args, which
        # the fast path doesn't model) keeps the scipy ``.rvs`` route.
        if isinstance(self.dist, str) and not self.dist_args:
            return self._numpy_draw_fn(T, kwargs, multivar)

        scale_key = "scale"
        if multivar:
            scale_key = "shape" if self.dist == "t" else "cov"
        dist = self._get_dist(multivar)
        dist_args = self.dist_args

        def _scipy_draw(
            s: float | NDArray[float64],
            seed: int | None,
            factor: NDArray[float64] | None = None,
        ) -> NDArray[float64]:
            del factor  # scipy's own rvs owns the factorization
            return abstract_shock_array(
                T,
                seed,
                dist,
                *dist_args,
                **{**kwargs, scale_key: s},
            )

        return _scipy_draw

    def _numpy_draw_fn(self, T: int, kwargs: dict, multivar: bool) -> ShockDrawFn:
        """Resolve a string family onto the numpy Generator fast paths.

        The returned callable takes the scale argument ``s`` (a scalar std for
        univariate families, a covariance/shape matrix for multivariate ones),
        mirroring the scipy-route contract.
        """
        family = self.dist

        if family == "norm":
            if multivar:
                mean = kwargs.get("mean")
                return lambda s, seed, factor: _draw_normal_mv(
                    T, seed, mean, cast(ndarray, s), factor
                )
            loc = kwargs.get("loc", 0.0)
            return lambda s, seed, factor: _draw_normal(T, seed, loc, cast(float, s))

        if family == "t":
            if "df" not in kwargs:
                raise ValueError("Student-t shocks require 'df' in dist_kwargs.")
            df = kwargs["df"]
            if multivar:
                loc_mv = kwargs.get("loc")
                return lambda s, seed, factor: _draw_t_mv(
                    T, seed, df, loc_mv, cast(ndarray, s), factor
                )
            loc = kwargs.get("loc", 0.0)
            return lambda s, seed, factor: _draw_t(T, seed, df, loc, cast(float, s))

        if family == "uni":
            if multivar:
                raise NotImplementedError(
                    "Multivariate uniform shocks are not implemented."
                )
            loc = kwargs.get("loc", 0.0)
            return lambda s, seed, factor: _draw_uniform(T, seed, loc, cast(float, s))

        raise ValueError(f"Unknown shock distribution family: {family!r}")

    def _get_dist(self, multivar: bool) -> rv_generic | multi_rv_generic:
        dist = self.dist

        if dist == "norm" and not multivar:
            return norm
        elif dist == "norm" and multivar:
            return mnorm
        elif dist == "t" and not multivar:
            return t
        elif dist == "t" and multivar:
            return mt
        elif dist == "uni" and not multivar:
            return uniform
        elif dist == "uni" and multivar:
            raise NotImplementedError(
                "Multivariate uniform distribution is not implemented."
            )
        else:
            assert isinstance(
                dist, rv_generic | multi_rv_generic
            ), "dist must be a valid scipy.stats distribution or a string identifier."
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
            dist=self.dist,
            seed=None if self.seed is None else int(self.seed),
            dist_args=[_jsonable(arg) for arg in self.dist_args],
            dist_kwargs={k: _jsonable(v) for k, v in self.dist_kwargs.items()},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Shock":
        """Rebuild a generator-style Shock from :meth:`to_dict` output.

        A ``multivar`` written by an older release is ignored: the spec key the
        shock is filed under fixes its arity, and a stored flag could contradict
        it.
        """
        dist = data["dist"]
        if dist not in {"norm", "t", "uni"}:
            raise ValueError(
                f"Shock.from_dict expects a 'norm'/'t'/'uni' dist, got {dist!r}."
            )
        seed = data.get("seed")
        return cls(
            dist=cast(ShockDistribution, dist),
            seed=None if seed is None else int(seed),
            dist_args=tuple(data.get("dist_args") or ()),
            dist_kwargs=dict(data.get("dist_kwargs") or {}),
        )


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
