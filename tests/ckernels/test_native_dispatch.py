"""Tests for the native/numba selection env-var overrides (_native_dispatch)."""

import importlib

import pytest

import SymbolicDSGE._native_dispatch as dispatch


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("anything", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("off", False),
        ("", False),
        ("  ", False),
    ],
)
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("SDSGE_TEST_FLAG", value)
    assert dispatch._flag("SDSGE_TEST_FLAG") is expected


def test_flag_unset_is_false(monkeypatch):
    monkeypatch.delenv("SDSGE_TEST_FLAG", raising=False)
    assert dispatch._flag("SDSGE_TEST_FLAG") is False


def _reload(monkeypatch, *, always=None, never=None):
    for name, val in (("ALWAYS_USE_NUMBA", always), ("NEVER_USE_NUMBA", never)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)
    return importlib.reload(dispatch)


def test_default_is_prefer_native(monkeypatch):
    mod = _reload(monkeypatch)
    assert mod.FORCE_NUMBA is False
    assert mod.REQUIRE_NATIVE is False


def test_always_use_numba_sets_force(monkeypatch):
    mod = _reload(monkeypatch, always="1")
    assert mod.FORCE_NUMBA is True
    assert mod.REQUIRE_NATIVE is False


def test_never_use_numba_sets_require(monkeypatch):
    mod = _reload(monkeypatch, never="1")
    assert mod.FORCE_NUMBA is False
    assert mod.REQUIRE_NATIVE is True


def test_both_set_raises(monkeypatch):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _reload(monkeypatch, always="1", never="1")


@pytest.fixture(autouse=True)
def _restore_dispatch():
    """Reload the module clean after env-mutating tests so other tests see the
    process default (neither override set)."""
    yield
    importlib.reload(dispatch)
