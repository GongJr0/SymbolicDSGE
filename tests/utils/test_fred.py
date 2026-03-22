# type: ignore
import builtins
from datetime import datetime
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest

import SymbolicDSGE.utils.fred as fred_module
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


class _FakeFredClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


def _install_fake_fred_modules(monkeypatch, env_path: Path):
    load_calls: list[Path] = []

    fredapi_module = ModuleType("fredapi")
    fredapi_module.Fred = _FakeFredClient

    dotenv_module = ModuleType("dotenv")
    dotenv_module.find_dotenv = lambda: str(env_path)
    dotenv_module.load_dotenv = lambda path: load_calls.append(Path(path))

    monkeypatch.setitem(sys.modules, "fredapi", fredapi_module)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    return load_calls


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


def test_init_raises_when_optional_dependency_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"fredapi", "dotenv"}:
            raise ImportError("missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires the 'fred' optional dependency"):
        FRED("FRED_API_KEY")


def test_init_raises_for_missing_explicit_env_file(monkeypatch, tmp_path):
    missing_env = tmp_path / "missing.env"
    _install_fake_fred_modules(monkeypatch, missing_env)

    with pytest.raises(FileNotFoundError, match="Could not find .env file"):
        FRED("FRED_API_KEY", key_env=missing_env)


def test_init_raises_when_env_file_cannot_be_found_implicitly(monkeypatch, tmp_path):
    missing_env = tmp_path / "implicit.env"
    _install_fake_fred_modules(monkeypatch, missing_env)

    with pytest.raises(ValueError, match=".env file not found"):
        FRED("FRED_API_KEY")


def test_init_raises_when_key_is_missing(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_KEY=value\n", encoding="utf-8")
    _install_fake_fred_modules(monkeypatch, env_path)
    monkeypatch.setattr(fred_module, "getenv", lambda name: None)

    with pytest.raises(ValueError, match="not found in environment variables"):
        FRED("FRED_API_KEY", key_env=env_path)


def test_init_loads_api_key_from_env(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("FRED_API_KEY=secret\n", encoding="utf-8")
    load_calls = _install_fake_fred_modules(monkeypatch, env_path)
    monkeypatch.setattr(
        fred_module,
        "getenv",
        lambda name: "secret" if name == "FRED_API_KEY" else None,
    )

    fred = FRED("FRED_API_KEY", key_env=env_path)

    assert isinstance(fred.db, _FakeFredClient)
    assert fred.db.api_key == "secret"
    assert load_calls == [env_path]
