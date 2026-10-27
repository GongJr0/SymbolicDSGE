from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from numpy import float64

from sympy import Symbol, Expr

from .config import FunctionGetterDict, ModelConfig, SymbolGetterDict, cov_from_spec

NDF = NDArray[np.float64]


@dataclass(frozen=True)
class VariableLayout:
    """Where every compiled variable sits, and which ones the compiler minted.

    The counts are the model's dimensions, and a consumer sizing an array takes
    them from here rather than measuring whichever list is nearest. ``n_var`` is
    the compiled set, ``n_declared`` the model's own variables and
    ``n_generated`` the minted ones, so ``n_var == n_declared + n_generated``.
    The first two coincide until the compiler mints something, which is why they
    are separate numbers and not one: a dense ``ss_seed`` or a parse-time ``P0``
    spans ``n_declared``, a dense ``x0`` spans ``n_var``. ``n_state`` and
    ``n_ctrl`` split the canonical order and sum to ``n_var``.

    ``declared_names`` is the model's own declaration order, and nothing else.
    ``generated_names`` is the compiler's minted names in mint order, which is
    where they sit in declaration order. Neither carries a position: a name's
    canonical slot is ``idx``'s answer, and asking one map for two orderings is
    what lets them drift.

    ``aux_origin`` maps each minted lag or lead to the declared variable it
    tracks, at every depth, which is what widens a declared-order input over the
    generated block. An aux sits at its origin some periods away, so it shares
    the origin's steady state and takes its variance.

    ``state_names`` and ``control_names`` split the canonical order at
    ``n_state``. A variable is a state when it occurs at ``t-1``, which is what
    the pencil partition needs.

    ``n_exog`` counts the model's innovations, which is the width of the shock
    matrix and of ``B``, not a count of variables.

    ``shock_names`` is ordered names of the shock columns.

    ``shock_idx`` is that order as a lookup.
    """

    n_var: int
    n_declared: int
    n_generated: int

    n_exog: int
    n_state: int
    n_ctrl: int

    idx: dict[str, int]

    declared_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    generated_names: tuple[str, ...] = ()

    aux_origin: dict[str, str] = field(default_factory=dict)
    shock_names: tuple[str, ...] = ()
    shock_idx: dict[str, int] = field(default_factory=dict)


def newton_seed(
    seeds: FunctionGetterDict[Expr | None],
    params: Mapping[Symbol, float],
    layout: VariableLayout,
    overrides: Mapping[str, float],
) -> NDF:
    """The model's own ``ss_seed``, resolved at ``params``, in canonical order.

    Omitted variables are zeroed. Raises if any seed expression does not resolve to a number.
    """
    seed = np.zeros(layout.n_var, dtype=float64)
    for i, name in enumerate(layout.canonical_names):
        origin = layout.aux_origin.get(name, name)
        if origin in overrides:
            seed[i] = overrides[origin]
            continue

        expr = seeds[origin]
        if expr is None:
            continue
        if expr.is_number:
            seed[i] = float(expr)
            continue
        resolved = expr.subs(
            {s: params[s] for s in expr.free_symbols}  # pyright: ignore
        )
        try:
            seed[i] = float(resolved)
        except TypeError as exc:
            raise ValueError(
                f"ss_seed for '{name}' did not resolve to a number at these "
                f"parameters: {resolved}"
            ) from exc
    return seed


def resolve_ss_seed(
    config: ModelConfig,
    layout: VariableLayout,
    ss_seed: list[float] | NDF | Mapping[str, float] | None = None,
    params: Mapping[Symbol, float] | Mapping[str, float] | None = None,
) -> NDF:
    """Newton seed for the steady state, in canonical variable order.

    Resolved per variable: an explicit ``ss_seed`` if it names the variable,
    else the config's ``ss_seed`` expression, else zero. Newton resolves
    ``F(ss, ss) = 0`` from here, so a gap model seeds at 0 and converges in
    one step while a level model that declares its seed seeds itself.

    An explicit seed names the model's own variables, as a mapping or as a
    dense sequence in declaration order. Compiler-minted variables are not
    addressable: at a steady state every date coincides, so an aux equals
    its origin identically and has no seed of its own to give.
    """
    overrides = _resolve_seed_override(ss_seed, layout)
    params = (
        config.calibration.parameters if params is None else SymbolGetterDict(params)
    )
    return newton_seed(config.ss_seed, params, layout, overrides)


def shock_covariance(
    config: ModelConfig,
    params: Mapping[str, float] | SymbolGetterDict[float] | None = None,
    *,
    shocks: Sequence[str] | None = None,
) -> NDF:
    """``Q`` for a model, at its own calibration unless ``params`` says otherwise.

    ``shocks`` slices the matrix to a subset when specified.
    """
    calib = config.calibration
    return cov_from_spec(
        config.shocks,
        calib.shocks,
        calib.parameters if params is None else params,
        subset=shocks,
    )


def measurement_covariance(
    config: ModelConfig,
    params: Mapping[str, float] | SymbolGetterDict[float] | None = None,
    *,
    observables: Sequence[str] | None = None,
) -> NDF:
    """``R`` from the named std/correlation maps, over ``observables``.

    Reads the model's own calibration unless ``params`` says otherwise. A
    config carrying a fixed ``R`` matrix rather than named parameters is the
    caller's to slice.
    """
    calib = config.calibration
    meas_cov = calib.measurement_noise
    if meas_cov is None:
        raise ValueError(
            "Building R from named parameters requires a `measurement_noise` block in the model's calibration."
        )
    if not meas_cov.std:
        raise ValueError(
            "`measurement_noise` does not expose named parameter metadata."
        )

    if params is None:
        params = calib.parameters
    return cov_from_spec(
        config.observables,
        meas_cov,
        params,
        subset=observables,
    )


def _resolve_seed_override(
    ss_seed: list[float] | NDF | Mapping[str, float] | None,
    layout: VariableLayout,
) -> dict[str, float]:
    """Validate and resolve seed overrides."""
    if ss_seed is None:
        return {}

    declared = layout.declared_names
    if isinstance(ss_seed, Mapping):
        by_name = {str(name): float(value) for name, value in ss_seed.items()}
        minted = sorted(set(by_name) & set(layout.generated_names))
        if minted:
            raise ValueError(
                f"ss_seed names compiler-minted variable(s) {minted}, which "
                f"take their origin's seed. Seed the origin instead."
            )
        unknown = sorted(set(by_name) - set(declared))
        if unknown:
            raise ValueError(
                f"ss_seed names variables the model does not have: {unknown}. "
                f"Model variables: {list(declared)}."
            )
    else:
        values = np.asarray(ss_seed, dtype=float64)
        if values.shape != (len(declared),):
            raise ValueError(
                f"ss_seed has shape {values.shape}, expected "
                f"({len(declared)},) for the model's variables in declaration "
                f"order: {list(declared)}. Name the variables in a dict to "
                f"seed only some of them."
            )
        by_name = dict(zip(declared, (float(v) for v in values)))
    return by_name
