"""Compile a live :class:`MCPipeline` back into a :class:`PipelineSpec`.

The inverse of :func:`SymbolicDSGE.monte_carlo.builder.build_pipeline`: it lets a
pipeline authored with plain library objects be serialized to the bundle's graph
language without the user ever touching the spec DTOs. Structure (nodes + edges)
is read from the pipeline's owned :class:`~SymbolicDSGE.monte_carlo.graph.PipelineGraph`;
a node's kwargs are written as the step holds them and its source bindings are
written as their own objects, so ``to_spec`` is a fixed point under a rebuild.

Recovery is mostly pass-through. The one value that cannot travel as data:

- **simulation**: live :class:`Shock` objects are serialized via
  :meth:`Shock.to_dict`; :func:`restore_kwargs` rebuilds them on the way back.
- **raw_model_data**: bulk arrays cannot ride the JSON spec, so the node records a
  ``data_ref`` (the bundle member key), the array ``data_shapes``, and the scalar
  metadata; the bundle builder writes the parquet member from
  :func:`raw_model_data_arrays`.
- **custom**: the user callable cannot ride the JSON spec either, so the node
  records a ``func_ref`` (the bundle member key) alongside its plain kwargs; the
  bundle builder writes the cloudpickle member and ``build_pipeline`` reattaches
  the callable from the loaded resources.

Source dependencies are emitted as ``sources``, one object per binding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..core.shock_generators import Shock, ShockParameters
from .mc_constructs import ColumnSelector, OpType, SourceArgs
from .spec import EdgeSpec, NodeSpec, PipelineSpec, PostprocSpec, SourceSpec

if TYPE_CHECKING:
    from .core import MCPipeline
    from .mc_constructs import MCStep


def pipeline_to_spec(pipeline: "MCPipeline") -> PipelineSpec:
    """Serialize a live pipeline into its graph-form :class:`PipelineSpec`."""
    graph = pipeline.graph
    nodes = [
        NodeSpec(
            id=step.name,
            op_type=_op_type(step),
            step_type=_step_type(step),
            name=step.name,
            params=_recover_params(step),
            sources=_recover_sources(step.source_args),
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


def raw_model_data_arrays(kwargs: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """The named bulk arrays a ``raw_model_data`` datagen ships.

    ``states`` and ``observables`` keep their names. Shared with the bundle
    builder, which feeds them to :func:`SymbolicDSGE.bundle.parquet.arrays_to_parquet`.
    """
    out: dict[str, NDArray[Any]] = {}
    for key in ("states", "observables"):
        value = kwargs[key]
        if value is not None:
            out[key] = np.asarray(value, dtype=np.float64)
    return out


def _op_type(step: "MCStep") -> str:
    op_type = step.op_type
    if op_type is None:
        raise ValueError(f"Step {step.name!r} has no op_type and cannot be serialized.")
    return op_type.value


def _step_type(step: "MCStep") -> str:
    step_type = step.step_type
    if step_type is None:
        raise ValueError(
            f"Step {step.name!r} has no step_type and cannot be serialized."
        )
    return step_type


def _recover_params(step: "MCStep") -> dict[str, Any]:
    step_type = step.step_type
    if step_type == "raw_model_data":
        params = _recover_raw_model_data(step)
    elif step_type == "simulation":
        params = _recover_simulation(step.kwargs)
    elif step_type in ("transform:custom", "postproc:custom"):
        params = _jsonable_params(dict(step.kwargs))
        params["func_ref"] = step.name
    else:
        params = _jsonable_params(dict(step.kwargs))
    if step.op_type is not OpType.POSTPROC and step.n_retain != -1:
        params["n_retain"] = step.n_retain
    return params


def _columns_spec(columns: ColumnSelector) -> list[int] | None:
    """A selector's chosen columns as plain indices, or ``None`` for all of them.

    ``SourceArgs`` normalizes a scalar and an array into a tuple, so those arms
    are unreachable once one is built; a slice is passed through untouched and
    has no width to resolve against here.
    """
    if columns is None:
        return None
    if isinstance(columns, slice):
        raise TypeError(
            "A slice column selector cannot be serialized; give explicit indices."
        )
    if isinstance(columns, (int, np.integer)):
        return [int(columns)]
    return [int(column) for column in columns]


def _recover_sources(source_args: tuple[SourceArgs, ...]) -> list[SourceSpec]:
    """Emit a step's source bindings as their own objects, not flattened params."""
    return [
        SourceSpec(
            arg=selector.arg,
            source_step=selector.source_step,
            field=selector.field,
            columns=_columns_spec(selector.columns),
            burn_in=selector.burn_in,
            drop_initial=selector.drop_initial,
        )
        for selector in source_args
    ]


def restore_sources(sources: list[SourceSpec]) -> tuple[SourceArgs, ...]:
    """Rebuild the source bindings a node recorded."""
    return tuple(
        SourceArgs(
            arg=source["arg"],
            source_step=source["source_step"],
            field=source["field"],
            columns=source["columns"],
            burn_in=source["burn_in"],
            drop_initial=source["drop_initial"],
        )
        for source in sources
    )


def _recover_simulation(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(kwargs)
    shocks = params.get("shocks")
    if shocks is not None:
        params["shocks"] = {key: _shock_dict(value) for key, value in shocks.items()}
    return _jsonable_params(params)


def restore_kwargs(step_type: str | None, params: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a step's live kwargs from the plain values a node recorded.

    Only ``Shock`` needs restoring: arrays travel as nested lists and every
    consumer coerces them on the way in.
    """
    kwargs = dict(params)
    if step_type == "simulation" and kwargs.get("shocks") is not None:
        kwargs["shocks"] = coerce_shock_mapping(kwargs["shocks"])
    return kwargs


def coerce_shock_mapping(value: Any) -> dict[str, Shock]:
    """Normalize a ``shocks`` mapping of live :class:`Shock` / serialized dicts.

    Library-authored pipelines carry explicit :class:`Shock` instances; a spec
    loaded from a bundle carries their :meth:`Shock.to_dict` form.
    """
    if not isinstance(value, Mapping):
        raise TypeError("simulation 'shocks' must be a mapping of name -> Shock.")
    out: dict[str, Shock] = {}
    for key, shock in value.items():
        if isinstance(shock, Shock):
            out[str(key)] = shock
        elif isinstance(shock, Mapping):
            out[str(key)] = Shock.from_dict(shock)
        else:
            raise TypeError(
                f"shocks[{key!r}] must be a Shock or a serialized shock dict."
            )
    return out


def _shock_dict(value: Any) -> ShockParameters | dict[str, Any]:
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


def _recover_raw_model_data(step: "MCStep") -> dict[str, Any]:
    kwargs = step.kwargs
    shapes = {
        name: list(arr.shape) for name, arr in raw_model_data_arrays(kwargs).items()
    }
    return {
        "observable_names": [str(n) for n in kwargs["observable_names"] or ()],
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
