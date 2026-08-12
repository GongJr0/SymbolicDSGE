"""Resolving a shock spec against a compiled model.

Nothing here reads the policy: target columns, the canonical order of a grouped
entry, the standard deviations and correlations off the calibration, the
assembled covariance and its factor all depend on the model and the spec alone.
"""

from __future__ import annotations

from typing import Callable, Mapping, Tuple, Union

import numpy as np
from numpy import asarray, float64, ndarray
from numpy.typing import NDArray
from sympy import Symbol

from ..compiled_model import CompiledModel
from ..shock_generators import Shock, _gaussian_factor
from ..shock_plan import ShockPlan, ShockPlanEntry, validate_shock_targets

NDF = NDArray[float64]

ShockSpec = Mapping[str, Union[Shock, Callable[[Union[float, NDF]], NDF], NDF]]


def _require_horizon(T: int | None, name: str) -> int:
    """A live ``Shock`` resolves its family against a horizon; demand one."""
    if T is None:
        raise ValueError(
            f"Shock spec {name!r} is a live Shock, so resolving it needs a "
            "horizon T. Pass T, or materialize the spec into a draw closure first."
        )
    return T


def materialize_shocks(
    shocks: ShockSpec,
    T: int,
) -> dict[str, Callable[[float | NDF], NDF] | NDF]:
    """Resolve any ``Shock`` specs into their ``T``-horizon draw closures.

    Live callables and raw arrays pass through untouched. Callers that hold the
    specs themselves can hand them to :func:`resolve_shock_plan` directly, which
    resolves the family without going through a closure.
    """
    return {
        name: shock.shock_generator(T) if isinstance(shock, Shock) else shock
        for name, shock in shocks.items()
    }


def resolve_shock_plan(
    compiled: CompiledModel,
    shocks: ShockSpec,
    T: int | None = None,
) -> ShockPlan:
    """Resolve a shock spec against a model, once.

    None of it depends on the seed, so a caller drawing many paths from one spec
    resolves a plan and redraws from it.

    ``T`` is required only when the mapping carries live :class:`Shock` specs,
    which resolve their distribution family against a horizon.
    """
    calib = compiled.config.calibration
    shock_stds = calib.shock_std

    shock_col = compiled.shock_idx
    validate_shock_targets(list(shocks), list(compiled.shock_names))

    entries: list[ShockPlanEntry] = []
    seeded_count = 0

    for name, shock in shocks.items():
        if isinstance(shock, Shock) and shock.seed is not None:
            seeded_count += 1

        if "," in name:
            multi_names = [n.strip() for n in name.split(",")]
            indices = [shock_col[n] for n in multi_names]
            perm = np.argsort(indices)
            multi_names_sorted = [multi_names[i] for i in perm]
            indices_sorted = tuple(indices[i] for i in perm)

            if isinstance(shock, ndarray):
                assert shock.shape[1] == len(
                    multi_names
                ), f"Shock array for {name} must have shape (T, {len(multi_names)})"
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=indices_sorted,
                        multivar=True,
                        values=asarray(shock[:, perm], dtype=float64),
                    )
                )
                continue

            if not isinstance(shock, Shock) and not callable(shock):
                raise TypeError(
                    f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
                )

            shock_syms = [Symbol(n) for n in multi_names_sorted]
            sig_params = [shock_stds[sym] for sym in shock_syms]
            sigs = [calib.get_param(sig, 1.0) for sig in sig_params]
            rhos = [
                calib.get_rho(n1, n2, 0.0) for n1 in shock_syms for n2 in shock_syms
            ]
            corr = np.array(rhos).reshape(
                (len(multi_names_sorted), len(multi_names_sorted))
            )
            cov = corr * np.outer(sigs, sigs)

            if isinstance(shock, Shock):
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=indices_sorted,
                        multivar=True,
                        scale=cov,
                        factor=_gaussian_factor(cov),
                        draw=shock.draw_fn(_require_horizon(T, name)),
                        base_seed=(None if shock.seed is None else int(shock.seed)),
                        spec=shock,
                    )
                )
            else:
                entries.append(
                    ShockPlanEntry(
                        key=name,
                        indices=indices_sorted,
                        multivar=True,
                        scale=cov,
                        func=shock,
                    )
                )
            continue

        # Uni-Var (target validity already checked by validate_shock_targets)
        idx = (shock_col[name],)
        if isinstance(shock, ndarray):
            entries.append(
                ShockPlanEntry(
                    key=name,
                    indices=idx,
                    multivar=False,
                    values=asarray(shock, dtype=float64),
                )
            )
            continue

        if not isinstance(shock, Shock) and not callable(shock):
            raise TypeError(
                f"Shock for {name} must be a callable or ndarray, got {type(shock)}."
            )

        sig = calib.get_param(shock_stds[Symbol(name)], 1.0)

        if isinstance(shock, Shock):
            entries.append(
                ShockPlanEntry(
                    key=name,
                    indices=idx,
                    multivar=False,
                    scale=sig,
                    draw=shock.draw_fn(_require_horizon(T, name)),
                    base_seed=None if shock.seed is None else int(shock.seed),
                    spec=shock,
                )
            )
        else:
            entries.append(
                ShockPlanEntry(
                    key=name,
                    indices=idx,
                    multivar=False,
                    scale=sig,
                    func=shock,
                )
            )

    return ShockPlan(
        entries=tuple(entries),
        n_exog=compiled.n_exog,
        seeded_count=seeded_count,
    )


def shock_unpack(
    compiled: CompiledModel,
    shocks: Mapping[str, NDF | Callable[[float | NDF], NDF]],
) -> list[Tuple[int, NDF]]:
    """Resolve a spec and draw it once, as ``(exogenous index, column)``."""
    return resolve_shock_plan(compiled, shocks).unpack()


def simulation_shock_matrix(
    compiled: CompiledModel,
    T: int,
    shocks: ShockSpec | None = None,
    shock_scale: float = 1.0,
) -> NDF:
    """``(T, n_exog)`` innovations for a spec, or zeros when there is none."""
    if shocks is None:
        return np.zeros((T, compiled.n_exog), dtype=float64)
    return resolve_shock_plan(compiled, shocks, T).matrix(T, shock_scale)
