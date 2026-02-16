import numpy as np
import pandas as pd


def HP_two_sided(
    s: pd.Series | np.ndarray, lamb: float = 1600
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """
    Apply the Hodrick-Prescott filter to a time series to separate the trend and cyclical components.

    Parameters:
    s (pd.Series | np.ndarray): The input time series data.
    lamb (float): The smoothing parameter. Default is 1600, commonly used for quarterly data.

    Returns:
    trend (pd.Series | np.ndarray): The trend component of the series.
    cycle (pd.Series | np.ndarray): The cyclical component of the series.
    """
    if isinstance(s, pd.Series):
        series = s.values
    else:
        series = s

    n = len(series)
    I = np.eye(n)
    D = np.zeros((n - 2, n))

    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1

    DTD = D.T @ D
    trend = np.linalg.inv(I + lamb * DTD) @ series  # type: ignore
    cycle = series - trend

    if isinstance(s, pd.Series):
        trend = pd.Series(trend, index=s.index)
        cycle = pd.Series(cycle, index=s.index)

    return trend, cycle


def HP_one_sided(
    s: pd.Series | np.ndarray, lamb: float = 1600
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """
    Apply the one-sided Hodrick-Prescott filter to a time series to separate the trend and cyclical components.

    Parameters:
    series (pd.Series | np.ndarray): The input time series data.
    lamb (float): The smoothing parameter. Default is 1600, commonly used for quarterly data.

    Returns:
    trend (pd.Series | np.ndarray): The trend component of the series.
    cycle (pd.Series | np.ndarray): The cyclical component of the series.
    """
    if isinstance(s, pd.Series):
        series = s.values
    else:
        series = s

    n = len(series)
    trend = np.zeros(n)

    for t in range(n):
        if t < 2:
            trend[t] = series[t]
        else:
            A = np.array([[1 + lamb, -2 * lamb], [-2 * lamb, 1 + 5 * lamb]])
            b = np.array(
                [
                    series[t] + 4 * lamb * trend[t - 1] - lamb * trend[t - 2],
                    (
                        series[t - 1] + 4 * lamb * trend[t - 2] - lamb * trend[t - 3]
                        if t > 2
                        else series[t - 1]
                    ),
                ]
            )
            trend_segment = np.linalg.solve(A, b)
            trend[t] = trend_segment[0]

    cycle = series - trend  # type: ignore

    if isinstance(s, pd.Series):
        trend = pd.Series(trend, index=s.index)  # type: ignore
        cycle = pd.Series(cycle, index=s.index)

    return trend, cycle


def annualized_log_percent(
    s: pd.Series | np.ndarray, periods_per_year: int = 4
) -> pd.Series | np.ndarray:
    """
    Calculate the annualized log percent change of a time series.

    Parameters:
    series (pd.Series | np.ndarray): The input time series data.
    periods_per_year (int): Number of periods in a year. Default is 4 for quarterly data.

    Returns:
    pd.Series | np.ndarray: The annualized log percent change of the series.
    """
    if isinstance(s, pd.Series):
        series = s.values
    else:
        series = s

    log_diff = np.diff(np.log(series))
    annualized_log_percent_change: np.ndarray | pd.Series = (
        log_diff * periods_per_year * 100
    )

    if isinstance(s, pd.Series):
        annualized_log_percent_change = pd.Series(
            annualized_log_percent_change, index=s.index[1:]
        )

    return annualized_log_percent_change


def demean(s: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Demean a time series by subtracting its mean.

    Parameters:
    series (pd.Series | np.ndarray): The input time series data.

    Returns:
    pd.Series | np.ndarray: The demeaned series.
    """
    if isinstance(s, pd.Series):
        mean_value = s.mean()
        demeaned_series = s - mean_value
    else:
        mean_value = np.mean(s)
        demeaned_series = s - mean_value

    return demeaned_series


def detrend(s: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Detrend a time series by removing its linear trend.

    Parameters:
    s (pd.Series | np.ndarray): The input time series data.

    Returns:
    pd.Series | np.ndarray: The detrended series.
    """
    if isinstance(s, pd.Series):
        series = s.values

    x = np.arange(len(series))
    coefs = np.polyfit(x, series, 1)  # type: ignore
    trend: np.ndarray = np.polyval(coefs, x)
    detrended_series: np.ndarray | pd.Series = series - trend

    if isinstance(s, pd.Series):
        detrended_series = pd.Series(detrended_series, index=s.index)

    return detrended_series
