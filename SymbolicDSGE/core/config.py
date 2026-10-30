"""Symbolic representation of a parsed DSGE model configuration."""

from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Mapping,
    TypeAlias,
    TypeVar,
    Sequence,
)
from types import MappingProxyType
from collections import UserDict
from sympy import Symbol, Function, Eq, Expr, And, Or, Not
from sympy.core.relational import Relational
from numpy import array, eye, float64, ix_, outer
from numpy.typing import NDArray

from pathlib import Path
from .config_emitter import write_config_yaml

KT = TypeVar("KT")
V = TypeVar("V")

Regime: TypeAlias = Mapping[str, Eq]  # {model_equation_name: replacement}

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

    def lock(self) -> None:
        object.__setattr__(self, "_locked", True)

    def unlock(self) -> None:
        object.__setattr__(self, "_locked", False)

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
class CovarianceSpec(_Lockable):
    """Specification of a covariance matrix in terms of standard deviations and correlations."""

    std: SymbolGetterDict[str]
    corr: PairGetterDict[str | None]

    @property
    def param_names(self) -> set[str]:
        """Parameter names used in the covariance specification."""
        std_params = list(self.std.values())
        corr_params = [v for v in self.corr.values() if v is not None]
        return set(std_params + corr_params)

    def lock(self) -> None:
        self.std.lock()
        self.corr.lock()
        super().lock()

    def unlock(self) -> None:
        super().unlock()
        self.std.unlock()
        self.corr.unlock()


@dataclass
class Constraint(_Lockable):
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

    # SymPy expressions are unmutable, super's lock/unlock is enough
    bind: Relational | And | Or | Not
    relax: Relational | And | Or | Not


@dataclass
class Equations(_Lockable):
    """Equations defining the transitions, policy, and observables of the model.

    Attributes
    ----------
    model : Dict[str, Eq]
        The reference state-space model equations. This is the base (and only) set
        of equations for perturbation models. OccBin takes it as the reference state
        and any binding constraint replaces the relevant equations.
    constraints : Dict[str, Constraint] | None
        Constraints that can bind the model equations. Each constraint can declare a
        binding and relaxing condition separately.
    regimes : RegimeGetterDict | None
        Regime to apply when the relevant constraint is binding. The regime is a
        mapping of model equation names to replacement equations.
    observables : SymbolGetterDict[Expr]
        Mapping of observable names to their symbolic expressions in terms of the
        model's variables.
    obs_is_affine : SymbolGetterDict[bool]
        Mapping of observable names to boolean values indicating whether the
        observable is affine (linear) in the model's variables. This information can
        be used to optimize computations and simplify analysis of the model's
        observables.
    """

    model: Mapping[str, Eq]
    constraints: Mapping[str, Constraint] | None  # {constraint_name: Constraint}
    regimes: RegimeGetterDict | None  # {binding_set: Regime}
    observables: SymbolGetterDict[Expr]
    obs_is_affine: SymbolGetterDict[bool]

    def lock(self) -> None:
        self.model = MappingProxyType(self.model)
        if self.constraints is not None:
            for constraint in self.constraints.values():
                constraint.lock()
            self.constraints = MappingProxyType(self.constraints)
        if self.regimes is not None:
            for k, v in self.regimes.items():
                self.regimes[k] = MappingProxyType(v)
            self.regimes.lock()
        self.observables.lock()
        self.obs_is_affine.lock()

        super().lock()

    def unlock(self) -> None:
        super().unlock()

        self.model = dict(self.model)
        if self.constraints is not None:
            for constraint in self.constraints.values():
                constraint.unlock()
            self.constraints = dict(self.constraints)
        if self.regimes is not None:
            for k, v in self.regimes.items():
                self.regimes[k] = dict(v)
            self.regimes.unlock()
        self.observables.unlock()
        self.obs_is_affine.unlock()


@dataclass
class Calib(_Lockable):
    """Calibration of the model configuration, including parameter values, shock standard deviations, and shock correlations.

    Attributes
    ----------
    parameters : SymbolGetterDict[float64]
        Mapping of parameter names to their calibrated values.
    shocks : CovarianceSpec
        Specification of the shock covariance matrix in terms of standard deviations and correlations.
    measurement_noise : CovarianceSpec
        Specification of the measurement noise covariance matrix in terms of standard deviations and correlations.

    """

    parameters: SymbolGetterDict[float]
    shocks: CovarianceSpec
    measurement_noise: CovarianceSpec | None

    def fingerprint(self) -> int:
        """Hashable snapshot of the parameter values, for keying caches."""
        return hash(
            (
                tuple(self.parameters.keys()),
                tuple(float(v) for v in self.parameters.values()),
            )
        )

    def lock(self) -> None:
        self.parameters.lock()
        self.shocks.lock()
        if self.measurement_noise is not None:
            self.measurement_noise.lock()
        super().lock()

    def unlock(self) -> None:
        super().unlock()

        self.parameters.unlock()
        self.shocks.unlock()
        if self.measurement_noise is not None:
            self.measurement_noise.unlock()


def cov_from_spec(
    order: Sequence[Symbol],
    spec: CovarianceSpec,
    params: Mapping[str, float] | SymbolGetterDict[float],
    *,
    subset: Sequence[str] | None = None,
) -> NDArray[float64]:
    """Assemble a covariance ``outer(sig, sig) * rho`` from a named spec.

    Serves Q and R alike: the spec carries parameter *names* and ``params``
    resolves each to a value, so one call covers a calibration and an estimation
    draw. ``order`` fixes the row and column order. A pair mapped to ``None``
    stays at zero correlation, and the diagonal is one.

    ``subset`` subsets the result. The matrix is assembled over the whole
    ``order`` and sliced afterwards, since a correlation pair may name a member
    the subset leaves out.

    Every member of ``order`` needs a standard deviation. Both parsers enforce
    that, so this repeats it for a spec that was built or edited in code and
    never passed through one.
    """
    n = len(order)
    pos = {str(m): i for i, m in enumerate(order)}

    if missing := pos.keys() - {str(name) for name in spec.std}:
        raise ValueError(
            "Covariance spec names no standard deviation parameter for: "
            f"{sorted(missing)!r}"
        )

    def _param(name: str) -> float64:
        if name not in params:
            raise KeyError(
                f"Missing covariance parameter '{name}' in the supplied parameters."
            )
        return float64(params[name])

    sig_vec = array([_param(spec.std[m]) for m in order], dtype=float64)

    rho = eye(n, dtype=float64)
    for pair, param_name in spec.corr.items():
        if param_name is None:
            continue
        i, j = (pos[str(member)] for member in pair)
        rho_ij = _param(param_name)
        rho[i, j] = rho_ij
        rho[j, i] = rho_ij

    cov = outer(sig_vec, sig_vec) * rho
    if subset is None:
        return cov

    idx = [pos[name] for name in subset]
    return cov[ix_(idx, idx)]


@dataclass(repr=False)
class ModelConfig(_Lockable):
    """A model configuration parsed into symbolic representations of its variables, parameters, shocks, observables, and equations.

    Attributes
    ----------
    name : str
        Name of the model.
    variables : list[Function]
        List of model variables as :class:`sympy.Function`s of time.
    ss_seed : FunctionGetterDict[Expr | None]
        Initial guess for the steady-state Newton solve.
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
    """

    name: str
    variables: Sequence[Function]
    ss_seed: FunctionGetterDict[Expr | None]
    shocks: Sequence[Symbol]
    observables: Sequence[Symbol]
    equations: Equations
    calibration: Calib

    def lock(self) -> None:
        # lockables
        self.ss_seed.lock()
        self.equations.lock()
        self.calibration.lock()

        # list->tuple
        self.variables = tuple(self.variables)
        self.shocks = tuple(self.shocks)
        self.observables = tuple(self.observables)

        super().lock()

    def unlock(self) -> None:
        super().unlock()

        # unlockables
        self.ss_seed.unlock()
        self.equations.unlock()
        self.calibration.unlock()

        # tuple->list
        self.variables = list(self.variables)
        self.shocks = list(self.shocks)
        self.observables = list(self.observables)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def update_parameters(
        self, new_params: Mapping[str, float] | Mapping[Symbol, float]
    ) -> None:
        """Update the model's calibration parameters with the given overrides.

        Parameters
        ----------
        new_params : Mapping[str, float] | Mapping[Symbol, float]
            Values to override existing calibration parameters with.
            All keys must already exist in the model's calibration parameters.

        """
        if missing_keys := [
            str(k) for k in new_params if k not in self.calibration.parameters
        ]:
            raise KeyError(
                "Cannot update parameters. "
                f"The following keys are missing from the model's calibration parameters: {missing_keys}"
            )

        was_locked = self.is_locked
        self.unlock()
        self.calibration.parameters.update(new_params)
        if was_locked:
            self.lock()

    def to_yaml(self, path: str | Path, *, digits: int | None = None) -> None:
        """Write the model configuration to a YAML file.

        Parameters
        ----------
        path : str | Path
            Path for the output YAML file.
        digits : int | None
            Number of digits to use for plain floats.
            If ``None``, the default formatting is used.

        """
        write_config_yaml(self, path, digits=digits)

    @staticmethod
    def parse_yaml(path: str | Path) -> "ModelConfig":
        """Parse a model configuration from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the input YAML file.

        Returns
        -------
        ModelConfig
            Parsed model configuration.

        """
        from .model_parser import ModelParser

        return ModelParser(path).get()
