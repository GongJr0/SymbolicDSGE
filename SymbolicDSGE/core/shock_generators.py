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
from typing import Any, Literal, Callable, Mapping, cast, overload


def abstract_shock_array(
    T: int,
    seed: int | None,
    dist: rv_generic | multi_rv_generic,
    *dist_args: object,
    **dist_kwargs: object,
) -> ndarray:
    """
    Generate an array of shocks based on a specified distribution.

    Parameters:
    T (int): The number of time periods.
    dist: A scipy.stats distribution object (e.g., norm, t, uniform).
    *dist_args: Positional arguments for the distribution.
    **dist_kwargs: Keyword arguments for the distribution.

    Returns:
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
    T: int, seed: int | None, mean: ndarray | None, cov: ndarray
) -> ndarray:
    cov = asarray(cov, dtype=float64)
    k = cov.shape[0]
    mean_vec = zeros(k, dtype=float64) if mean is None else asarray(mean, dtype=float64)
    z = random.default_rng(seed).standard_normal((T, k))
    return cast(ndarray, (mean_vec + z @ _gaussian_factor(cov).T).astype(float64))


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
    T: int, seed: int | None, df: float, loc: ndarray | None, shape: ndarray
) -> ndarray:
    shape = asarray(shape, dtype=float64)
    k = shape.shape[0]
    loc_vec = zeros(k, dtype=float64) if loc is None else asarray(loc, dtype=float64)
    rng = random.default_rng(seed)
    z = rng.standard_normal((T, k)) @ _gaussian_factor(shape).T
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


def normal_shock_array(
    T: int,
    seed: int,
    mu: float | float64 = 0.0,
    sigma: float | float64 = 1.0,
) -> ndarray:
    """
    Generate an array of normally distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    mu (float | float64): Mean of the normal distribution.
    sigma (float | float64): Standard deviation of the normal distribution.
    Returns:
    np.ndarray: An array of normally distributed shocks of length T.
    """
    return _draw_normal(T, seed, mu, sigma)


def normal_multivariate_shock_array(
    T: int,
    seed: int,
    mus: list[float | float64],
    cov_mat: list[list[float | float64]],
) -> ndarray:
    """
    Generate an array of multivariate normally distributed shocks.

    Parameters:
    T (int): The number of time periods.
    k (int): The number of variables (dimensions).
    seed (int): Seed for the random number generator.
    mu (float | float64): Mean of the normal distribution.
    sigma (float | float64): Standard deviation of the normal distribution.

    Returns:
    np.ndarray: An array of shape (T, k) of multivariate normally distributed shocks.
    """

    return _draw_normal_mv(T, seed, asarray(mus, dtype=float64), asarray(cov_mat))


def t_shock_array(
    T: int,
    seed: int | None,
    df: float,
    loc: float | float64 = 0.0,
    scale: float | float64 = 1.0,
) -> ndarray:
    """
    Generate an array of t-distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    df (float): Degrees of freedom for the t-distribution.
    loc (float | float64): Location parameter of the t-distribution.
    scale (float | float64): Scale parameter of the t-distribution.

    Returns:
    np.ndarray: An array of t-distributed shocks of length T.
    """
    return _draw_t(T, seed, df, loc, scale)


def t_multivariate_shock_array(
    T: int,
    seed: int | None,
    df: float,
    locs: list[float | float64],
    cov_mat: list[list[float | float64]],
) -> ndarray:
    """
    Generate an array of multivariate t-distributed shocks.

    Parameters:
    T (int): The number of time periods.
    k (int): The number of variables (dimensions).
    seed (int): Seed for the random number generator.
    df (float): Degrees of freedom for the t-distribution.
    loc (float | float64): Location parameter of the t-distribution.
    scale (float | float64): Scale parameter of the t-distribution.

    Returns:
    np.ndarray: An array of shape (T, k) of multivariate t-distributed shocks.
    """
    return _draw_t_mv(T, seed, df, asarray(locs, dtype=float64), asarray(cov_mat))


def uniform_shock_array(
    T: int, seed: int | None, loc: float | float64 = 0.0, scale: float | float64 = 1.0
) -> ndarray:
    """
    Generate an array of uniformly distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    low (float): Lower bound of the uniform distribution.
    high (float): Upper bound of the uniform distribution.

    Returns:
    np.ndarray: An array of uniformly distributed shocks of length T.
    """
    return _draw_uniform(T, seed, loc, scale)


def uniform_multivariate_shock_array(
    T: int,
    k: int,
    seed: int | None,
    locs: list[float | float64],
    cov_mat: list[list[float | float64]],
) -> ndarray:
    """
    [NOT IMPLEMENTED]

    Generate an array of multivariate uniformly distributed shocks.
    Rectangular uniform distributions implicitly indicate cov_ij = 0 for i != j.


    Parameters:
    T (int): The number of time periods.
    k (int): The number of variables (dimensions).
    seed (int): Seed for the random number generator.
    locs (list[float | float64]): List of means for each dimension.
    cov_mat (list[list[float | float64]]): Covariance matrix for the distribution.
    Returns:
    np.ndarray: An array of shape (T, k) of multivariate uniformly distributed shocks.
    """
    raise NotImplementedError(
        "Multivariate uniforms can get complex and computationally expensive."
        " The function will remain in the namespace but will not be implemented unless explicitly needed."
    )


def shock_placement(
    T: int, shock_spec: dict[int, float], shock_arr: ndarray = None
) -> ndarray:
    """
    Place shocks in a time series array based on a shock specification.

    Parameters:
    T (int): The number of time periods.
    shock_spec (dict): A dictionary where keys are time indices (0-based) and
                       values are shock scales (shock = scale * var_sigma at simulation time).

    Returns:
    np.ndarray: An array of shocks of length T with specified shocks placed.
    """

    if shock_arr is not None:
        shocks = shock_arr
    else:
        rdim = T
        shocks = zeros((rdim,), dtype=float64)

    for i, shock in shock_spec.items():
        shocks[i] = shock

    return shocks


ShockSpecUni = dict[int, float]
ShockSpecMulti = dict[tuple[int, int], float]


class Shock:
    def __init__(
        self,
        dist: Literal["norm", "t", "uni"] | rv_generic | multi_rv_generic | None = None,
        multivar: bool = False,
        seed: int | None = 0,
        dist_args: tuple = (),
        dist_kwargs: dict | None = None,
        shock_arr: ndarray | None = None,
    ) -> None:
        # A Shock is a horizon-independent distribution spec: the number of
        # periods ``T`` is supplied by the caller at generation time, not baked
        # in here. The simulation is the single authority on its own horizon.
        self.dist = dist
        self.multivar = multivar
        self.seed = seed
        self.dist_args = dist_args
        self.dist_kwargs = dist_kwargs if dist_kwargs is not None else {}
        self.shock_arr = shock_arr

    # TODO: Pass through array if provided else generate based on dist

    def _assert_generator(self) -> None:
        assert self.dist is not None, "Distribution must be specified."
        assert (
            self.shock_arr is None
        ), "shock_arr is already provided. Please use place_shocks to mutate it."
        assert "scale" not in self.dist_kwargs, (
            "The generator function returns a callable that takes scale as an argument."
            " Please adjust `sig_` variables in the config to change the distribution scale."
            " Alternatively, the scale parameter in simulation and irf functions are multiplied directly with the shocks generated."
        )

    def shock_generator(
        self, T: int
    ) -> Callable[[float | NDArray[float64]], NDArray[float64]]:
        """Build the per-scale draw closure for a ``T``-period horizon.

        ``T`` is supplied by the caller (the simulation), not stored on the
        Shock; the returned callable takes only the scale argument ``s``.
        """
        self._assert_generator()
        kwargs = self.dist_kwargs.copy()

        # Known string families go through the numpy Generator fast paths. A raw
        # scipy distribution object (or a string with positional dist_args, which
        # the fast path doesn't model) keeps the scipy ``.rvs`` route.
        if isinstance(self.dist, str) and not self.dist_args:
            return self._numpy_shock_generator(T, kwargs)

        scale_key = "scale"
        if self.multivar:
            scale_key = "shape" if self.dist == "t" else "cov"
        fun = lambda s: abstract_shock_array(
            T,
            self.seed,
            self._get_dist(),
            *self.dist_args,
            **{**kwargs, scale_key: s},
        )

        return fun

    def _numpy_shock_generator(
        self, T: int, kwargs: dict
    ) -> Callable[[float | NDArray[float64]], NDArray[float64]]:
        """Build the per-scale draw closure for a string family on numpy.

        The returned callable takes the scale argument ``s`` (a scalar std for
        univariate families, a covariance/shape matrix for multivariate ones),
        mirroring the scipy-path contract.
        """
        family = self.dist
        seed, multivar = self.seed, self.multivar

        if family == "norm":
            if multivar:
                mean = kwargs.get("mean")
                return lambda s: _draw_normal_mv(T, seed, mean, cast(ndarray, s))
            loc = kwargs.get("loc", 0.0)
            return lambda s: _draw_normal(T, seed, loc, cast(float, s))

        if family == "t":
            if "df" not in kwargs:
                raise ValueError("Student-t shocks require 'df' in dist_kwargs.")
            df = kwargs["df"]
            if multivar:
                loc = kwargs.get("loc")
                return lambda s: _draw_t_mv(T, seed, df, loc, cast(ndarray, s))
            loc = kwargs.get("loc", 0.0)
            return lambda s: _draw_t(T, seed, df, loc, cast(float, s))

        if family == "uni":
            if multivar:
                raise NotImplementedError(
                    "Multivariate uniform shocks are not implemented."
                )
            loc = kwargs.get("loc", 0.0)
            return lambda s: _draw_uniform(T, seed, loc, cast(float, s))

        raise ValueError(f"Unknown shock distribution family: {family!r}")

    @overload
    def place_shocks(self, shock_spec: ShockSpecUni, T: int) -> ndarray: ...
    @overload
    def place_shocks(self, shock_spec: ShockSpecMulti, T: int) -> ndarray: ...

    def place_shocks(
        self,
        shock_spec: ShockSpecUni | ShockSpecMulti,
        T: int,
    ) -> ndarray:
        if self.shock_arr is not None:
            assert self.shock_arr.shape[0] == T, "shock_arr length must match T."

        if not shock_spec:
            if self.shock_arr is not None:
                return self.shock_arr
            return (
                zeros((T,), dtype=float64)
                if not self.multivar
                else zeros((T, 0), dtype=float64)
            )

        if not self.multivar:
            # Narrow for mypy
            shock_spec_u = cast(ShockSpecUni, shock_spec)

            for k in shock_spec_u.keys():
                if k < 0 or k >= T:
                    raise IndexError(f"Time index {k} out of bounds for T={T}.")
            return shock_placement(T, shock_spec_u, self.shock_arr)

        # multivar
        shock_spec_m = cast(ShockSpecMulti, shock_spec)

        for t, k in shock_spec_m.keys():
            if t < 0 or t >= T:
                raise IndexError(f"Time index {t} out of bounds for T={T}.")
            if k < 0:
                raise IndexError(f"Shock dimension index {k} must be non-negative.")

        if self.shock_arr is not None:
            shocks = self.shock_arr
        else:
            K = max(k for (_, k) in shock_spec_m.keys()) + 1
            shocks = zeros((T, K), dtype=float64)

        for (t, k), val in shock_spec_m.items():
            if k >= shocks.shape[1]:
                raise IndexError(
                    f"Shock dimension index {k} out of bounds for K={shocks.shape[1]}."
                )
            shocks[t, k] = float64(val)

        return shocks

    def _get_dist(self) -> rv_generic | multi_rv_generic:
        dist = self.dist

        if dist == "norm" and not self.multivar:
            return norm
        elif dist == "norm" and self.multivar:
            return mnorm
        elif dist == "t" and not self.multivar:
            return t
        elif dist == "t" and self.multivar:
            return mt
        elif dist == "uni" and not self.multivar:
            return uniform
        elif dist == "uni" and self.multivar:
            raise NotImplementedError(
                "Multivariate uniform distribution is not implemented."
            )
        else:
            assert isinstance(
                dist, rv_generic | multi_rv_generic
            ), "dist must be a valid scipy.stats distribution or a string identifier."
            return dist

    def to_dict(self) -> dict[str, Any]:
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
        if self.shock_arr is not None:
            raise ValueError(
                "Cannot serialize a Shock carrying a materialized shock_arr; "
                "array-backed shocks must be shipped as bulk (parquet) data."
            )
        return {
            "dist": self.dist,
            "multivar": bool(self.multivar),
            "seed": None if self.seed is None else int(self.seed),
            "dist_args": [_jsonable(arg) for arg in self.dist_args],
            "dist_kwargs": {k: _jsonable(v) for k, v in self.dist_kwargs.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Shock":
        """Rebuild a generator-style Shock from :meth:`to_dict` output.

        A legacy ``"T"`` key (from bundles authored before the horizon moved to
        generation time) is ignored; the horizon is supplied at ``.generate``.
        """
        dist = data["dist"]
        if dist not in {"norm", "t", "uni"}:
            raise ValueError(
                f"Shock.from_dict expects a 'norm'/'t'/'uni' dist, got {dist!r}."
            )
        seed = data.get("seed")
        return cls(
            dist=cast(Literal["norm", "t", "uni"], dist),
            multivar=bool(data.get("multivar", False)),
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
