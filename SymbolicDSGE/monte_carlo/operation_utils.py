from __future__ import annotations

from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from ..core.shock_generators import Shock
from ..kalman.filter import FilterResult
from .mc_constructs import MCContext, NDF, SeedIncrement, ShockMapping

InpSources = Literal[
    "states",
    "observables",
    "x_pred",
    "x_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
    "payload",
]


def _clone_or_pass_shocks(
    shocks: ShockMapping | None,
    *,
    T: int,
    rep_idx: int,
    seed_increment: SeedIncrement,
) -> Mapping[str, Callable[[float | NDF], NDF] | NDF] | None:
    if shocks is None:
        return None
    out: dict[str, Callable[[float | NDF], NDF] | NDF] = {}
    seed_offset = rep_idx * _resolve_seed_increment(shocks, seed_increment)
    for name, shock in shocks.items():
        if isinstance(shock, Shock):
            if shock.T != T:
                raise ValueError(f"Shock '{name}' has T={shock.T}, expected {T}.")
            if shock.shock_arr is not None:
                raise ValueError(
                    "MC simulation requires generator-style Shock instances."
                )
            if ("," in name) != shock.multivar:
                raise ValueError(
                    f"Shock '{name}' must set multivar={',' in name} to match its specification."
                )
            seed = None if shock.seed is None else int(shock.seed) + seed_offset
            out[name] = Shock(
                T=shock.T,
                dist=shock.dist,  # pyright: ignore
                multivar=shock.multivar,
                seed=seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            ).shock_generator()
        else:
            out[name] = shock
    return out


def _resolve_seed_increment(
    shocks: ShockMapping,
    seed_increment: SeedIncrement,
) -> int:
    if seed_increment == "auto":
        return sum(
            1
            for shock in shocks.values()
            if isinstance(shock, Shock) and shock.seed is not None
        )
    increment = int(seed_increment)
    if increment < 0:
        raise ValueError("seed_increment must be non-negative or 'auto'.")
    return increment


def _resolve_context_array(
    context: MCContext,
    *,
    source: str,
    filter_key: str,
    payload_key: str | None,
    columns: Sequence[int] | slice | None,
    burn_in: int,
    drop_initial: bool,
) -> NDF:
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if source == "states":
        arr = context.require_data().states
        if arr is None:
            raise ValueError("MC context has no generated states.")
        if drop_initial:
            arr = arr[1:]
    elif source == "observables":
        obs = context.require_data().observables
        if obs is None:
            raise ValueError("MC context has no generated observables.")
        arr = obs
    elif source == "payload":
        if payload_key is None:
            raise ValueError("payload_key is required when source='payload'.")
        arr = context.require_payload(payload_key)
    else:
        filter_result = context.require_payload(filter_key)
        if not isinstance(filter_result, FilterResult):
            raise TypeError(f"Payload '{filter_key}' is not a FilterResult.")
        if not hasattr(filter_result, source):
            raise ValueError(f"FilterResult has no array source '{source}'.")
        arr = getattr(filter_result, source)

    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"Selected MC array must be 1D or 2D, got shape {out.shape}.")
    if columns is not None:
        out = out[:, columns]
        if out.ndim == 1:
            out = out.reshape(-1, 1)
    if burn_in:
        out = out[burn_in:]
    return np.ascontiguousarray(out, dtype=np.float64)


def _select_raw_rep_array(
    value: NDF,
    *,
    rep_idx: int,
    name: str,
    allow_vector: bool,
) -> NDF:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3:
        if rep_idx >= arr.shape[0]:
            raise IndexError(
                f"{name} has {arr.shape[0]} replications, cannot select rep_idx={rep_idx}."
            )
        out = arr[rep_idx]
    elif arr.ndim == 2:
        out = arr
    elif allow_vector and arr.ndim == 1:
        out = arr.reshape(-1, 1)
    else:
        expected = "1D, 2D, or 3D" if allow_vector else "2D or 3D"
        raise ValueError(f"{name} must be {expected}, got shape {arr.shape}.")

    if out.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2D array, got shape {out.shape}.")
    return np.ascontiguousarray(out, dtype=np.float64)
