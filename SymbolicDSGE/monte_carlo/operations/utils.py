from __future__ import annotations

from typing import Any, Callable, Mapping
from .types import NDF

import numpy as np

from ...core.shock_generators import Shock
from ..mc_constructs import (
    MCContext,
    SeedIncrement,
    ShockMapping,
    SourceArgs,
)


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
                dist=shock.dist,  # pyright: ignore
                multivar=shock.multivar,
                seed=seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            ).shock_generator(T)
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


def _resolve_source_array(context: MCContext, selector: SourceArgs) -> NDF:
    out: NDF = context.payload_slots[selector.source_idx][selector.field_idx][
        selector.row_start :, selector.column_selector
    ]
    return out


def _select_raw_rep_array(
    name: str,
    value: NDF,
    rep_idx: int,
) -> NDF:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 3:
        out: NDF = arr[rep_idx]
        return out
    else:
        raise ValueError(f"Raw data for '{name}' must be a 3D, 2D, or 1D array).")
