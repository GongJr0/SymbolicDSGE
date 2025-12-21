from scipy.stats import norm, t, uniform
from scipy.stats._distn_infrastructure import rv_generic
from numpy import asarray, ndarray, float64, random, zeros


def abstract_shock_array(
    T: int, seed: int, dist: rv_generic, *dist_args: object, **dist_kwargs: object
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
    T: int, seed: int, mu: float = 0.0, sigma: float = 1.0
) -> ndarray:
    """
    Generate an array of normally distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    mu (float): Mean of the normal distribution.
    sigma (float): Standard deviation of the normal distribution.

    Returns:
    np.ndarray: An array of normally distributed shocks of length T.
    """
    return abstract_shock_array(T, seed, norm, loc=mu, scale=sigma)


def t_shock_array(
    T: int, seed: int, df: float, loc: float = 0.0, scale: float = 1.0
) -> ndarray:
    """
    Generate an array of t-distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    df (float): Degrees of freedom for the t-distribution.
    loc (float): Location parameter of the t-distribution.
    scale (float): Scale parameter of the t-distribution.

    Returns:
    np.ndarray: An array of t-distributed shocks of length T.
    """
    return abstract_shock_array(T, seed, t, df=df, loc=loc, scale=scale)


def uniform_shock_array(
    T: int, seed: int, low: float = 0.0, high: float = 1.0
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
    return abstract_shock_array(T, seed, uniform, loc=low, scale=high - low)


def shock_placement(
    T: int, shock_spec: dict[int, list[float]], shock_arr: ndarray = None
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
        cdim = len(list(shock_spec.values())[0])
        shocks = zeros((rdim, cdim), dtype=float64)

    for i, shock in shock_spec.items():
        shocks[i, :] = asarray(shock, dtype=float64)

    return shocks
