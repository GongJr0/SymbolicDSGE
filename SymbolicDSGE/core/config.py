from dataclasses import dataclass, asdict
from typing import AbstractSet, Any, Mapping, TypeAlias, TypeVar, Dict, Sequence
from collections import UserDict
from sympy import Symbol, Function, Eq, Expr, And, Or, Not
from sympy.core.relational import Relational
from numpy import array, asarray, eye, float64, ix_, outer
from numpy.typing import NDArray
import pickle

from .linearization import LinearizationMethod

KT = TypeVar("KT")
V = TypeVar("V")

Regime: TypeAlias = Dict[str, Eq]  # {model_equation_name: replacement}


class _NormalizedKeyDict(UserDict[KT, V]):
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

    def __getitem__(self, key: Any) -> Any:
        return self.data[self._key(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[self._key(key)]

    def __contains__(self, key: Any) -> bool:
        return self._key(key) in self.data


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


class RegimeGetterDict(_NormalizedKeyDict[frozenset[str], Regime]):
    @staticmethod
    def _key(key: Any) -> Any:
        if isinstance(key, str):
            return frozenset(s.strip() for s in key.split(","))
        if isinstance(key, (Sequence, AbstractSet)):
            return frozenset(str(k) for k in key)
        return key


@dataclass
class Base:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def serialize(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


@dataclass
class Constraint(Base):
    bind: Relational | And | Or | Not
    relax: Relational | And | Or | Not


@dataclass
class Equations(Base):
    model: Dict[str, Eq]
    constraint: Dict[str, Constraint] | None  # {constraint_name: Constraint}
    regime: RegimeGetterDict | None  # {binding_set: Regime}
    observable: SymbolGetterDict[Expr]
    obs_is_affine: SymbolGetterDict[bool]


@dataclass
class Calib(Base):
    parameters: SymbolGetterDict[float64]
    shock_std: SymbolGetterDict[str]
    shock_corr: PairGetterDict[str | None]

    def fingerprint(self) -> int:
        """Hashable snapshot of the parameter values, for keying caches."""
        return hash(
            (
                tuple(self.parameters.keys()),
                tuple(float(v) for v in self.parameters.values()),
            )
        )


def make_Q(
    shock_order: Sequence[Symbol | str],
    std_param_map: Mapping[Any, str],
    corr_param_map: Mapping[Any, str | None] | None,
    params: Mapping[Any, float64],
    *,
    shocks: Sequence[str] | None = None,
    corr: NDArray[float64] | None = None,
) -> NDArray[float64]:
    """Assemble a shock covariance ``Q = outer(sig, sig) * rho``.

    The structural-shock counterpart of
    :func:`SymbolicDSGE.kalman.config.make_R`, and built the same way: the two
    maps carry parameter *names* and ``params`` resolves each to a value, so one
    call serves a calibration and an estimation draw alike. ``shock_order`` fixes
    the row and column order. A pair mapped to ``None`` stays at zero
    correlation, and the diagonal is one.

    ``corr`` supplies an already-materialized correlation matrix over
    ``shock_order``, skipping the name gather; an estimated Cholesky block hands
    its correlation in this way.

    ``shocks`` subsets the result. Q is assembled over the whole ``shock_order``
    and sliced afterwards, since a correlation pair may name a shock the subset
    leaves out.
    """
    n = len(shock_order)
    pos = {str(s): i for i, s in enumerate(shock_order)}

    def _param(name: str) -> float64:
        if name not in params:
            raise KeyError(
                f"Missing shock parameter '{name}' in the supplied parameters."
            )
        return float64(params[name])

    sig_vec = array([_param(std_param_map[s]) for s in shock_order], dtype=float64)

    if corr is None:
        rho = eye(n, dtype=float64)
        for pair, param_name in (corr_param_map or {}).items():
            if param_name is None:
                continue
            i, j = (pos[str(member)] for member in pair)
            rho_ij = _param(param_name)
            rho[i, j] = rho_ij
            rho[j, i] = rho_ij
    else:
        rho = asarray(corr, dtype=float64)

    Q = outer(sig_vec, sig_vec) * rho
    if shocks is None:
        return Q

    idx = [pos[name] for name in shocks]
    return Q[ix_(idx, idx)]


@dataclass
class Variables(Base):
    variables: list[Function]
    # None == 0 seed newton.
    ss_seed: FunctionGetterDict[Expr | None]
    linearization: FunctionGetterDict[LinearizationMethod]


@dataclass(repr=False)
class ModelConfig(Base):
    name: str
    variables: Variables
    parameters: list[Symbol]
    shocks: list[Symbol]
    observables: list[Symbol]
    equations: Equations
    calibration: Calib
    symbolically_linearized: bool = False

    #: Source YAML text the config was parsed from, retained so a model can be
    #: round-tripped into a ``.sdsge`` bundle without re-reading from disk
    #: (avoiding the staleness window between solve and save). ``None`` for
    #: programmatic construction.
    source_yaml: str | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
