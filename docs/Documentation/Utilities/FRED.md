---
tags:
    - doc
---
# FRED

???+ info "FRED Dependencies"
    `#!python FRED` relies on the optional dependency `#!python fredapi`. If you did not opt-in for optional dependencies, you can run:

    ```bash
    pip install SymbolicDSGE[fred]
    ```
    
    to get the required packages.

???+ warning "API Key Required"
    An API key from the "St. Louis FED" is required for the `#!python FRED` class to operate. If you don't have an API key, you can generate one for free by making an account on FRED.

    [FRED](//fred.stlouisfed.org){ .md-button }

```python
class FRED(key_name: str, key_env: str | pathlib.Path | None)
```

`#!python FRED` allows easy retrieval of time series stored in the FRED database. The class can identify the given `#!python key_name` if placed in the `.env` file anywhere in the discoverable file tree of the project. Alternatively, you can specify an `env` file that contains `key_name` in it.

??? info ".env File Discovery"
    `find_dotenv` is used for discovery. It will walk up the directory tree towards the project root until it encounters a file named `".env"`.

__Methods:__

```python
FRED.get_series(
    series_id: str,
    date_range: tuple[str, str] | pd.DatetimeIndex | Literal['max', 'ytd'] | None
) -> pd.Series
```
Returns the requested FRED series.

| __Argument__ | __Description__ |
|:-------------|----------------:|
| series_id | The FRED ID of the requested series. |
| date_range | Date range in either a `YYYY-MM-DD` format string pair, a `DatetimeIndex`, or a string literal `#!python 'max', 'ytd'`. `#!python None` defaults to `#!python 'max'`. |

__Returns:__

| __Type:__ | __Description__ |
|:---------:|----------------:|
| `#!python pd.Series` | The requested time series in a formatted `pandas` object. |

&nbsp;

```python
FRED.get_frame(
    series_ids: list[str],
    date_range: tuple[str, str] | pd.DatetimeIndex | Literal['max', 'ytd'] | None
) -> pd.DataFrame
```

Returns multiple requested series in a `DataFrame`.

| __Argument__ | __Description__ |
|:-------------|----------------:|
| series_ids | The FRED IDs of the requested series. |
| date_range | Date range in either a `YYYY-MM-DD` format string pair, a `DatetimeIndex`, or a string literal `#!python 'max', 'ytd'`. `#!python None` defaults to `#!python 'max'`. |

__Returns:__

| __Type:__ | __Description__ |
|:----------|----------------:|
| `#!python pd.DataFrame` | The requested time series in a formatted `pandas` object. |