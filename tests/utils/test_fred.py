# type: ignore
from datetime import datetime

import pandas as pd
import pytest

from SymbolicDSGE.utils.fred import FRED


class _FakeDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:
        self.calls.append((series_id, observation_start, observation_end))
        idx = pd.date_range("2020-01-01", periods=3, freq="MS")
        return pd.Series([1.0, 2.0, 3.0], index=idx, name=series_id)

    def get_series_info(self, series_id: str) -> pd.Series:
        return pd.Series({"id": series_id, "title": f"title-{series_id}"})


def _fred_with_fake_db() -> FRED:
    fred = FRED.__new__(FRED)
    fred.db = _FakeDB()
    return fred


def test_get_series_with_tuple_range():
    fred = _fred_with_fake_db()
    out = fred.get_series("GDP", ("2020-01-01", "2020-12-31"))

    assert isinstance(out, pd.Series)
    assert out.attrs["id"] == "GDP"
    assert fred.db.calls[-1] == ("GDP", "2020-01-01", "2020-12-31")


def test_get_series_with_datetime_index_range():
    fred = _fred_with_fake_db()
    idx = pd.date_range("2018-01-01", periods=4, freq="QS")
    _ = fred.get_series("CPIAUCSL", idx)

    assert fred.db.calls[-1] == ("CPIAUCSL", "2018-01-01", "2018-10-01")


@pytest.mark.parametrize("date_range", ["max", None])
def test_get_series_with_max_or_none_range(date_range):
    fred = _fred_with_fake_db()
    _ = fred.get_series("UNRATE", date_range)

    assert fred.db.calls[-1] == ("UNRATE", None, None)


def test_get_series_with_ytd_range():
    fred = _fred_with_fake_db()
    _ = fred.get_series("FEDFUNDS", "ytd")

    start = f"{datetime.now().year}-01-01"
    assert fred.db.calls[-1] == ("FEDFUNDS", start, None)


def test_get_series_rejects_invalid_date_range():
    fred = _fred_with_fake_db()
    with pytest.raises(ValueError):
        fred.get_series("GDP", "bad-range")


def test_get_frame_combines_series_and_collects_attrs():
    fred = _fred_with_fake_db()
    out = fred.get_frame(["GDP", "CPIAUCSL"], ("2020-01-01", "2020-12-31"))

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["GDP", "CPIAUCSL"]
    assert out.attrs["GDP"]["id"] == "GDP"
    assert out.attrs["CPIAUCSL"]["title"] == "title-CPIAUCSL"
