---
tags:
    - doc
---
# math_utils

```python
def HP_two_sided(
    s: pd.Series | np.ndarray,
    lamb: float = 1600,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]
```
Apply the **Two-Sided Hodrick-Prescott** filter to a time series to separate the trend and cyclical components.

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| s | `#!python pd.Series | np.ndarray` | The input time series data.|
| lamb | `#!python float` | The smoothing parameter. Default is 1600, commonly used for quarterly data. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python pd.Series | np.ndarray` | Tuple index 0. The trend component of the series.|
| `#!python pd.Series | np.ndarray` | Tuple index 1. The cyclical component of the series.|

&nbsp;

```python
def HP_one_sided(
    s: pd.Series | np.ndarray,
    lamb: float = 1600,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]
```
Apply the **One-Sided Hodrick-Prescott** filter to a time series to separate the trend and cyclical components.

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| s | `#!python pd.Series | np.ndarray` | The input time series data.|
| lamb | `#!python float` | The smoothing parameter. Default is 1600, commonly used for quarterly data. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python pd.Series | np.ndarray` | Tuple index 0. The trend component of the series.|
| `#!python pd.Series | np.ndarray` | Tuple index 1. The cyclical component of the series.|


&nbsp;

```python
def annualized_log_percent(
    s: pd.Series | np.ndarray, periods_per_year: int = 4
) -> pd.Series | np.ndarray
```
Calculate the annualized log percent change of a time series.

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| s | `#!python pd.Series | np.ndarray` | The input time series data.|
| periods_per_year | `#!python int` | Number of periods in a year. Default is 4 for quarterly data. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python pd.Series | np.ndarray` | The annualized log percent change of the series.|

&nbsp;

```python
def demean(
    s: pd.Series | np.ndarray
) -> pd.Series | np.ndarray
```
__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| s | `#!python pd.Series | np.ndarray` | The input series.|


__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python pd.Series | np.ndarray` | The demeaned series. |

&nbsp;

```python
def detrend(
    s: pd.Series | np.ndarray
) -> pd.Series | np.ndarray
```
Detrend a time series by removing its linear trend.

???+ note "Detrending"
    This process subtracts the "Line of Best Fit" (LoBF) from the data.
    LoBF is computed as the slope and intercept that satisfy:
    $$
    \underset{m,b}{\operatorname{argmin}}\,\ \sum_{i=0}^n  (y_i - (mx_i+b))^2
    $$

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| s | `#!python pd.Series | np.ndarray` | The input series.|


__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python pd.Series | np.ndarray` | The detrended series. |