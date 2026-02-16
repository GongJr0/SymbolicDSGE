import pandas as pd
from pathlib import Path
from os import getenv

from typing import Union, Literal


class FRED:
    def __init__(self, key_name: str, key_env: str | Path | None = None) -> None:
        try:
            import fredapi
            from dotenv import load_dotenv, find_dotenv

        except ImportError:
            raise ImportError(
                "FRED requires the 'fred' optional dependency package. Please install all extras or run 'pip install symbolicdsge[fred]' to use this feature."
            )

        def find_key(name: str, path: Path | str | None = None) -> str:
            key_path = Path(path) if path else Path(find_dotenv())

            if not key_path.exists() and path is not None:
                raise FileNotFoundError(f"Could not find .env file at {key_path}")
            elif not key_path.exists():
                raise ValueError(".env file not found in current directory or parents")

            load_dotenv(key_path)
            key = getenv(name)
            if key:
                return key
            else:
                raise ValueError(
                    f"FRED API key '{name}' not found in environment variables."
                )

        self.db = fredapi.Fred(api_key=find_key(key_name, key_env))

    def get_series(
        self,
        series_id: str,
        date_range: Union[
            tuple[str, str], pd.DatetimeIndex, Literal["max", "ytd"], None
        ],
    ) -> pd.Series:
        """
        Fetches a time series from the FRED database.

        Parameters
        ----------
        series_id : str
            The FRED series ID to fetch.
        date_range : Union[tuple[str, str], pd.DatetimeIndex, Literal["max", "ytd"], None]
            The date range for the data. Can be a tuple of start and end dates (YYYY-MM-DD),
            a Pandas DatetimeIndex, "max" or None for the maximum available range and "ytd" for year-to-date.

        Returns
        -------
        pd.Series
            A Pandas Series containing the requested time series data.
        """
        if isinstance(date_range, tuple):
            start, end = date_range

        elif isinstance(date_range, pd.DatetimeIndex):
            start = date_range.min().strftime("%Y-%m-%d")
            end = date_range.max().strftime("%Y-%m-%d")

        elif date_range == "max" or date_range is None:
            start, end = None, None

        elif date_range == "ytd":
            from datetime import datetime

            current_year = datetime.now().year
            start = f"{current_year}-01-01"
            end = None
        else:
            raise ValueError("Invalid date_range parameter.")

        series: pd.Series = self.db.get_series(
            series_id, observation_start=start, observation_end=end
        )
        info = self.db.get_series_info(series_id)
        series.attrs = info.to_dict()
        return series

    def get_frame(
        self,
        series_ids: list[str],
        date_range: Union[
            tuple[str, str], pd.DatetimeIndex, Literal["max", "ytd"], None
        ],
    ) -> pd.DataFrame:
        """
        Fetches multiple time series from the FRED database and returns them as a DataFrame.

        Parameters
        ----------
        series_ids : list[str]
            A list of FRED series IDs to fetch.
        date_range : Union[tuple[str, str], pd.DatetimeIndex, Literal["max", "ytd"], None]
            The date range for the data. Can be a tuple of start and end dates (YYYY-MM-DD),
            a Pandas DatetimeIndex, "max" or None for the maximum available range and "ytd" for year-to-date.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame containing the requested time series data.
        """
        data = {}
        infos = {}
        for series_id in series_ids:
            data[series_id] = self.get_series(series_id, date_range)
            infos[series_id] = data[series_id].attrs

        df = pd.DataFrame(data)
        df.attrs = infos  # type: ignore  # pandas types `attrs` as Mapping but uses dict in examples
        return df
