# type: ignore
from __future__ import annotations

import pickle

import sympy as sp

from SymbolicDSGE.core.config import Base, PairGetterDict, SymbolGetterDict
from SymbolicDSGE.core.model_parser import ParsedConfig


def test_symbol_getter_dict_accepts_str_and_symbol_keys():
    alpha = sp.Symbol("alpha")
    d = SymbolGetterDict({alpha: 1.25})

    assert d[alpha] == 1.25
    assert d["alpha"] == 1.25


def test_pair_getter_dict_accepts_string_and_symbol_tuples():
    a, b = sp.Symbol("a"), sp.Symbol("b")
    d = PairGetterDict({frozenset((a, b)): 0.33})

    assert d[(a, b)] == 0.33
    assert d[("a", "b")] == 0.33
    assert d[("b", "a")] == 0.33


def test_base_to_dict_and_serialize_roundtrip(tmp_path):
    base = Base()
    assert base.to_dict() == {}

    out = tmp_path / "base.pkl"
    base.serialize(str(out))
    with out.open("rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, Base)


def test_model_config_is_dict_indexable_and_nested_objects_are_base(
    parsed_test: ParsedConfig,
):
    conf = parsed_test.model

    assert conf["name"] == "TEST"
    assert conf["equations"] is conf.equations
    assert list(map(str, conf.equations.to_dict()["model"])) == list(
        map(str, conf.equations.model)
    )
    assert (
        conf.calibration.to_dict()["parameters"].keys()
        == conf.calibration.parameters.keys()
    )
