from scipy.stats import (
    norm,
    multivariate_normal as mnorm,
    t,
    multivariate_t as mt,
    uniform,
)
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._multivariate import multi_rv_generic
from numpy import asarray, ndarray, float64, random, zeros
from numpy.typing import NDArray
from typing import Literal, Callable, cast, overload


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
    return abstract_shock_array(T, seed, norm, loc=mu, scale=sigma)


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

    shocks = abstract_shock_array(T, seed, mnorm, mean=asarray(mus), cov=cov_mat)
    return asarray(shocks, dtype=float64)


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
    return abstract_shock_array(T, seed, t, df=df, loc=loc, scale=scale)


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
    return abstract_shock_array(T, seed, mt, df=df, loc=asarray(locs), shape=cov_mat)


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
    return abstract_shock_array(T, seed, uniform, loc=loc, scale=scale)


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
        T: int,
        dist: Literal["norm", "t", "uni"] | rv_generic | multi_rv_generic | None = None,
        multivar: bool = False,
        seed: int | None = 0,
        dist_args: tuple = (),
        dist_kwargs: dict | None = None,
        shock_arr: ndarray | None = None,
    ) -> None:

        self.T = T
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

    def shock_generator(self) -> Callable[[float | NDArray[float64]], NDArray[float64]]:
        self._assert_generator()
        kwargs = self.dist_kwargs.copy()
        fun = lambda s: abstract_shock_array(
            self.T,
            self.seed,
            self._get_dist(),
            *self.dist_args,
            **{**kwargs, "scale" if not self.multivar else "cov": s},
        )

        return fun

    @overload
    def place_shocks(self, shock_spec: ShockSpecUni) -> ndarray: ...
    @overload
    def place_shocks(self, shock_spec: ShockSpecMulti) -> ndarray: ...

    def place_shocks(
        self,
        shock_spec: ShockSpecUni | ShockSpecMulti,
    ) -> ndarray:
        if self.shock_arr is not None:
            assert self.shock_arr.shape[0] == self.T, "shock_arr length must match T."

        if not shock_spec:
            if self.shock_arr is not None:
                return self.shock_arr
            return (
                zeros((self.T,), dtype=float64)
                if not self.multivar
                else zeros((self.T, 0), dtype=float64)
            )

        if not self.multivar:
            # Narrow for mypy
            shock_spec_u = cast(ShockSpecUni, shock_spec)

            for k in shock_spec_u.keys():
                if k < 0 or k >= self.T:
                    raise IndexError(f"Time index {k} out of bounds for T={self.T}.")
            return shock_placement(self.T, shock_spec_u, self.shock_arr)

        # multivar
        shock_spec_m = cast(ShockSpecMulti, shock_spec)

        for t, k in shock_spec_m.keys():
            if t < 0 or t >= self.T:
                raise IndexError(f"Time index {t} out of bounds for T={self.T}.")
            if k < 0:
                raise IndexError(f"Shock dimension index {k} must be non-negative.")

        if self.shock_arr is not None:
            shocks = self.shock_arr
        else:
            K = max(k for (_, k) in shock_spec_m.keys()) + 1
            shocks = zeros((self.T, K), dtype=float64)

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
