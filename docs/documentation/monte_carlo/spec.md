---
tags:
    - doc
---

# Pipeline Specification

The graph specification is the portable Monte Carlo representation stored in `.sdsge` bundles and accepted by the core pipeline compiler. It uses plain dataclasses rather than pydantic models.

Most in-code workflows should use `MCPipeline` directly. Bundle loading rebuilds a live `LoadedMC.pipeline`; `PipelineSpec` remains available for archive inspection, UI rendering, and explicit compile workflows.

## `NodeSpec`

```python
@dataclass
class NodeSpec()
```

One operation node in a serializable Monte Carlo graph.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| id | `#!python str` | Stable graph node id. Edges refer to this value. |
| step_type | `#!python str` | Built-in or supported custom step kind. Must be in `STEP_KINDS`. |
| name | `#!python str` | Runtime step name used for summaries and payload references. |
| params | `#!python dict[str, Any]` | JSON-like parameter payload passed to the step compiler. |

__Methods:__

```python
NodeSpec.to_dict() -> dict[str, Any]
NodeSpec.from_dict(data: Mapping[str, Any]) -> NodeSpec  # @classmethod
```

`from_dict` validates `step_type` against `STEP_KINDS`.

## `EdgeSpec`

```python
@dataclass
class EdgeSpec()
```

One structural edge in the portable Monte Carlo graph.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| source | `#!python str` | Producer node id. |
| target | `#!python str` | Consumer node id. |

__Methods:__

```python
EdgeSpec.to_dict() -> dict[str, str]
EdgeSpec.from_dict(data: Mapping[str, Any]) -> EdgeSpec  # @classmethod
```

## `PostprocSpec`

```python
@dataclass
class PostprocSpec()
```

One post-loop op. A postproc is a **terminal reduction** over the assembled
across-replication traces. It is not a graph node, so it has no `id` and no
edges; its inputs are trace keys carried in `params`. Postprocs live in
`PipelineSpec.postprocs`, never in `nodes`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Runtime step name (also the artifact key). |
| step_type | `#!python str` | A post-processing kind. Must be in `POSTPROC_KINDS` (`"kde"`, `"postproc:custom"`). |
| params | `#!python dict[str, Any]` | JSON-like parameter payload (e.g. the `trace` key a `kde` reads). |

__Methods:__

```python
PostprocSpec.to_dict() -> dict[str, Any]
PostprocSpec.from_dict(data: Mapping[str, Any]) -> PostprocSpec  # @classmethod
```

## `PipelineSpec`

```python
@dataclass
class PipelineSpec()
```

Serializable pipeline. `nodes` and `edges` are the **per-replication**
dependency DAG; `postprocs` is the post-loop phase. The two are kept separate
because postprocs are not graph participants.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| nodes | `#!python list[NodeSpec]` | Per-rep graph nodes (never postprocs). |
| edges | `#!python list[EdgeSpec]` | Structural producer-consumer edges. |
| postprocs | `#!python list[PostprocSpec]` | Post-loop ops, run once over the assembled traces. |

`from_dict` rejects a postproc-kind `step_type` appearing in `nodes` (they belong in `postprocs`).

__Methods:__

```python
PipelineSpec.to_dict() -> dict[str, Any]
PipelineSpec.from_dict(data: Mapping[str, Any]) -> PipelineSpec  # @classmethod
PipelineSpec.to_json(*, indent: int | None = None) -> str
PipelineSpec.from_json(text: str) -> PipelineSpec  # @classmethod
```

???+ info "Resources"
    Large arrays and custom callables are not embedded directly in `PipelineSpec`. `raw_model_data` nodes reference restored arrays through `data_ref`, and `custom` nodes reference restored callables through `func_ref`. Bundle loading exposes those objects through `LoadedMC.resources`.

## `MCStepKind` and `STEP_KINDS`

```python
MCStepKind = Literal[
    "simulation",
    "raw_model_data",
    "filter",
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "cusum",
    "cusumsq",
    "chow",
    "regression",
    "standardize",
    "log",
    "log_diff",
    "diff",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "kde",
    "transform:custom",
    "postproc:custom",
]

STEP_KINDS: frozenset[str]

PostprocStepKind = Literal["kde", "postproc:custom"]
POSTPROC_KINDS: frozenset[str]
PER_REP_KINDS: frozenset[str]
```

`MCStepKind` is the string-level schema used by portable specs; `STEP_KINDS` is the runtime set used for validation. `POSTPROC_KINDS` contains the post-loop kinds. `PER_REP_KINDS` contains the graph node kinds. Together they drive the `nodes` vs `postprocs` split.
