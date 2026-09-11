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
    """Key normalizing dictionary for regimes. Accessors accept a single regime name or a set of names, and normalize to the frozenset of names the underlying dict is keyed by."""

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
    """Constrants that can bind and relax on separate conditions, allowing for regime switching in the model.

    Attributes
    ----------
    bind : Relational | And | Or | Not
        Binding condition for the constraint. A regime is entered when this is
        satisfied.
    relax : Relational | And | Or | Not
        Relaxing condition for the constraint. A bound regime is exited when this is
        satisfied.
    """

    bind: Relational | And | Or | Not
    relax: Relational | And | Or | Not


@dataclass
class Equations(Base):
    """Equations defining the transitions, policy, and observables of the model.

    Attributes
    ----------
    model : Dict[str, Eq]
        The reference state-space model equations. This is the base (and only) set
        of equations for perturbation models. OccBin takes it as the reference state
        and any binding constraint replaces the relevant equations.
    constraint : Dict[str, Constraint] | None
        Constraints that can bind the model equations. Each constraint can declare a
        binding and relaxing condition separately.
    regime : RegimeGetterDict | None
        Regime to apply when the relevant constraint is binding. The regime is a
        mapping of model equation names to replacement equations.
    observable : SymbolGetterDict[Expr]
        Mapping of observable names to their symbolic expressions in terms of the
        model's variables.
    obs_is_affine : SymbolGetterDict[bool]
        Mapping of observable names to boolean values indicating whether the
        observable is affine (linear) in the model's variables. This information can
        be used to optimize computations and simplify analysis of the model's
        observables.
    """

    model: Dict[str, Eq]
    constraint: Dict[str, Constraint] | None  # {constraint_name: Constraint}
    regime: RegimeGetterDict | None  # {binding_set: Regime}
    observable: SymbolGetterDict[Expr]
    obs_is_affine: SymbolGetterDict[bool]


@dataclass
class Calib(Base):
    """Calibration of the model configuration, including parameter values, shock standard deviations, and shock correlations.

    Attributes
    ----------
    parameters : SymbolGetterDict[float64]
        Mapping of parameter names to their calibrated values.
    shock_std : SymbolGetterDict[str]
        Mapping of shock names to their standard deviation parameter names.
    shock_corr : PairGetterDict[str | None]
        Mapping of shock name pairs to their correlation parameter names. A pair mapped to ``None`` indicates zero correlation.

    """

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
    """Variable specification for a model, including the list of variables, the steady-state seed, and the linearization method.

    Attributes
    ----------
    variables : list[Function]
        List of model variables as :class:`sympy.Function`s of time.
    ss_seed : FunctionGetterDict[Expr | None]
        Initial guess for the steady-state Newton solve.
    linearization : FunctionGetterDict[LinearizationMethod]
        Linearization method for each variable, as a mapping from variable to
        :class:`LinearizationMethod`.
    """

    variables: list[Function]
    # None == 0 seed newton.
    ss_seed: FunctionGetterDict[Expr | None]
    linearization: FunctionGetterDict[LinearizationMethod]


@dataclass(repr=False)
class ModelConfig(Base):
    """A model configuration parsed into symbolic representations of its variables, parameters, shocks, observables, and equations.

    Attributes
    ----------
    name : str
        Name of the model.
    variables : Variables
        :class:`Variables` object containing the model's variables, steady-state
        seed, and linearization method.
    parameters : list[Symbol]
        List of model parameters as :class:`sympy.Symbol`s.
    shocks : list[Symbol]
        List of model shocks as :class:`sympy.Symbol`s.
    observables : list[Symbol]
        List of model observables as :class:`sympy.Symbol`s.
    equations : Equations
        :class:`Equations` object containing the model's equations, constraints,
        regimes, and observables.
    calibration : Calib
        :class:`Calib` object containing the model's calibration parameters, shock
        standard deviations, and shock correlations.
    symbolically_linearized : bool
        Boolean indicating whether the model has been symbolically linearized.
    source_yaml : str | None
        Optional string containing the source YAML text from which the model
        configuration was parsed.
    """

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
