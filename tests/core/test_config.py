# type: ignore
from __future__ import annotations

import sympy as sp

from SymbolicDSGE.core.config import PairGetterDict, SymbolGetterDict


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
