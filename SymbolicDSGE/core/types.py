from __future__ import annotations

from collections import UserDict
from typing import Any, AbstractSet, Sequence, TypeVar, TypeAlias, Self
from sympy import Eq, Symbol, Function

KT = TypeVar("KT")
V = TypeVar("V")

_IMMUTABILITY_MSG = (
    "This object is currently immutable. "
    "If you tried to mutate the config a SolvedModel holds, "
    "it is recommended to re-solve the model on the new config. "
    "If you know the edit you're going to make will not change the "
    "functional form or compiled structure of the model, call `.unlock()` "
    "on the config before mutating it."
)


class _Lockable:
    _locked: bool = False

    @property
    def is_locked(self) -> bool:
        return self._locked

    def lock(self) -> Self:
        if self.is_locked:
            return self
        self._freeze()
        object.__setattr__(self, "_locked", True)
        return self

    def unlock(self) -> Self:
        if not self.is_locked:
            return self
        object.__setattr__(self, "_locked", False)
        self._thaw()
        return self

    def _freeze(self) -> None: ...

    def _thaw(self) -> None: ...

    def __setattr__(self, name: str, val: Any) -> None:
        if self.is_locked:
            raise AttributeError(_IMMUTABILITY_MSG)
        object.__setattr__(self, name, val)


class _NormalizedKeyDict(UserDict[KT, V], _Lockable):
    """A mapping whose key accepts more than one spelling.

    A subclass supplies ``_key``, resolving any accepted spelling to the one the
    underlying dict is keyed by. The three primitives below route through it and
    ``UserDict`` builds ``get``, ``pop``, ``setdefault``, ``update`` and the rest
    on top of them, so the coercion reaches the whole mapping API. Subclassing
    ``dict`` instead would leave every method not overridden by hand reading the
    hash table directly, which is silently wrong rather than loud.
    """

    def __init__(self, inp: Any) -> None:
        super().__init__(inp)

    @staticmethod
    def _key(key: Any) -> Any:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> V:
        return self.data[self._key(key)]

    def __setitem__(self, key: Any, value: V) -> None:
        if self.is_locked:
            raise AttributeError(_IMMUTABILITY_MSG)
        self.data[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        if self.is_locked:
            raise AttributeError(_IMMUTABILITY_MSG)

        del self.data[self._key(key)]

    def __contains__(self, key: Any) -> bool:
        return self._key(key) in self.data


class LockableDict(_NormalizedKeyDict[KT, V]):
    @staticmethod
    def _key(key: Any) -> Any:
        return key


class SymbolGetterDict(_NormalizedKeyDict[Symbol, V]):
    @staticmethod
    def _key(key: Any) -> Any:
        return Symbol(key) if isinstance(key, str) else key


class PairGetterDict(_NormalizedKeyDict[frozenset[Symbol], V]):
    @staticmethod
    def _key(key: Any) -> Any:
        if isinstance(key, (Sequence, AbstractSet)) and not isinstance(key, str):
            return frozenset(Symbol(k) if isinstance(k, str) else k for k in key)
        return key


class FunctionGetterDict(_NormalizedKeyDict[Function, V]):
    @staticmethod
    def _key(key: Any) -> Any:
        return Function(key) if isinstance(key, str) else key


Regime: TypeAlias = LockableDict[str, Eq]  # {model_equation_name: replacement}


class RegimeGetterDict(_NormalizedKeyDict[frozenset[str], Regime]):
    """Key normalizing dictionary for regimes. Accessors accept a single regime name or a set of names, and normalize to the frozenset of names the underlying dict is keyed by."""

    @staticmethod
    def _key(key: Any) -> Any:
        if isinstance(key, str):
            return frozenset(s.strip() for s in key.split(","))
        if isinstance(key, (Sequence, AbstractSet)):
            return frozenset(str(k) for k in key)
        return key
