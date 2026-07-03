"""Compile a live :class:`MCPipeline` back into a :class:`PipelineSpec`.

The inverse of :func:`SymbolicDSGE.monte_carlo.builder.build_pipeline`: it lets a
pipeline authored with plain library objects be serialized to the bundle's graph
language without the user ever touching the spec DTOs. Structure (nodes + edges)
is read from the pipeline's owned :class:`~SymbolicDSGE.monte_carlo.graph.PipelineGraph`;
per-node parameters are recovered into the form ``build_pipeline``'s catalogue
compile hooks expect, so ``to_spec`` is a fixed point under a rebuild.

Recovery is mostly pass-through. The cases that need inverting a compile hook:

- **simulation** — live :class:`Shock` objects are serialized via
  :meth:`Shock.to_dict`; the dual-form ``_compile_simulation`` rebuilds them.
- **wald** — the materialized ``target`` ndarray is inverted to the
  ``target_vector`` / ``target_matrix`` field the GUI/spec form carries.
- **raw_data** — bulk arrays cannot ride the JSON spec, so the node records a
  ``data_ref`` (the bundle member key), the array ``data_shapes``, and the scalar
  metadata; the bundle builder writes the parquet member from
  :func:`raw_data_arrays`.
- **custom** — the user callable cannot ride the JSON spec either, so the node
  records a ``func_ref`` (the bundle member key) alongside its plain kwargs; the
  bundle builder writes the cloudpickle member and ``build_pipeline`` reattaches
  the callable from the loaded resources.

Binder-derived dependency keys (``filter_key`` / ``*_payload_key``) are dropped:
``validate_pipeline_spec`` re-derives them from the edges, so emitting them would
break idempotency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..core.shock_generators import Shock
from .spec import EdgeSpec, NodeSpec, PipelineSpec, PostprocSpec

if TYPE_CHECKING:
    from .core import MCPipeline
    from .mc_constructs import MCStep

#: Dependency keys re-derived from edges on rebuild, so dropped from the spec.
#: ``filter_key`` is re-bound from the filter edge; the ``*_payload_key`` keys are
#: kept because payloads are referenced by key (no edge to re-derive them from).
_BINDER_KEYS = ("filter_key",)


def pipeline_to_spec(pipeline: "MCPipeline") -> PipelineSpec:
    """Serialize a live pipeline into its graph-form :class:`PipelineSpec`."""
    graph = pipeline.graph
    nodes = [
        NodeSpec(
            id=step.name,
            step_type=_step_type(step),
            name=step.name,
            params=_recover_params(step),
        )
        for step in pipeline.per_rep_steps
    ]
    edges = [EdgeSpec(source=src, target=dst) for src, dst in graph.edges()]
    postprocs = [
        PostprocSpec(
            name=step.name,
            step_type=_step_type(step),
            params=_recover_params(step),
        )
        for step in pipeline.postproc_steps
    ]
    return PipelineSpec(nodes=nodes, edges=edges, postprocs=postprocs)


def raw_data_arrays(kwargs: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """The named bulk arrays a ``raw_data`` datagen ships.

    ``states`` / ``observables`` keep their names; entries of ``raw`` are
    namespaced ``raw:<key>``. Shared with the bundle builder, which feeds them to
    :func:`SymbolicDSGE.bundle.parquet.arrays_to_parquet`.
    """
    out: dict[str, NDArray[Any]] = {}
    for key in ("states", "observables"):
        value = kwargs.get(key)
        if value is not None:
            out[key] = np.asarray(value, dtype=np.float64)
    raw = kwargs.get("raw")
    if raw:
        for name, value in raw.items():
            out[f"raw:{name}"] = np.asarray(value, dtype=np.float64)
    return out


def _step_type(step: "MCStep") -> str:
    step_type = step.step_type
    if step_type is None:
        raise ValueError(
            f"Step {step.name!r} has no step_type and cannot be serialized."
        )
    return step_type


def _recover_params(step: "MCStep") -> dict[str, Any]:
    step_type = step.step_type
    if step_type == "raw_data":
        return _recover_raw_data(step)
    if step_type == "simulation":
        params = _recover_simulation(step.kwargs)
    elif step_type == "wald":
        params = _recover_wald(step.kwargs)
    elif step_type in ("transform:custom", "postproc:custom"):
        params = _jsonable_params(dict(step.kwargs))
        params["func_ref"] = step.name
    else:
        params = _jsonable_params(dict(step.kwargs))
    for key in _BINDER_KEYS:
        params.pop(key, None)
    return params


def _recover_simulation(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(kwargs)
    shocks = params.get("shocks")
    if shocks is not None:
        params["shocks"] = {key: _shock_dict(value) for key, value in shocks.items()}
    return _jsonable_params(params)


def _shock_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Shock):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    hint = ""
    if isinstance(value, np.ndarray):
        hint = (
            " Got a raw shock array, which is not bundleable; author the "
            "simulation with a `Shock` generator spec instead."
        )
    elif callable(value):
        hint = (
            " Got a shock generator (a callable). Pass the `Shock` instance "
            "itself, e.g. `Shock(...)` rather than `Shock(...).shock_generator()`, "
            "so it can be serialized and replayed deterministically."
        )
    raise TypeError(
        "simulation shocks must be Shock instances (or serialized shock dicts)." + hint
    )


def _recover_wald(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(kwargs)
    if "target" in params:
        kind = str(params.get("kind", "mean"))
        target = np.asarray(params.pop("target"), dtype=np.float64)
        key = "target_vector" if kind == "mean" else "target_matrix"
        params[key] = target.tolist()
    return _jsonable_params(params)


def _recover_raw_data(step: "MCStep") -> dict[str, Any]:
    kwargs = step.kwargs
    shapes = {name: list(arr.shape) for name, arr in raw_data_arrays(kwargs).items()}
    return {
        "n_exog": int(kwargs.get("n_exog", -1)),
        "observable_names": [str(n) for n in kwargs.get("observable_names") or ()],
        "data_ref": step.name,
        "data_shapes": shapes,
    }


def _jsonable_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in params.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
