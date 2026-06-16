---
tags:
    - doc
---

# Pipeline Specification

The graph specification is the portable Monte Carlo representation stored in `.sdsge` bundles and accepted by the core pipeline compiler. It is intentionally pydantic-free and uses plain dataclasses.

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

## `PipelineSpec`

```python
@dataclass
class PipelineSpec()
```

Serializable graph-form pipeline.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| nodes | `#!python list[NodeSpec]` | All graph nodes. |
| edges | `#!python list[EdgeSpec]` | Structural producer-consumer edges. |

__Methods:__

```python
PipelineSpec.to_dict() -> dict[str, Any]
PipelineSpec.from_dict(data: Mapping[str, Any]) -> PipelineSpec  # @classmethod
PipelineSpec.to_json(*, indent: int | None = None) -> str
PipelineSpec.from_json(text: str) -> PipelineSpec  # @classmethod
```

???+ info "Resources"
    Large arrays and custom callables are not embedded directly in `PipelineSpec`. `raw_data` nodes reference restored arrays through `data_ref`, and `custom` nodes reference restored callables through `func_ref`. Bundle loading exposes those objects through `LoadedMC.resources`.

## `MCStepKind` and `STEP_KINDS`

```python
MCStepKind = Literal[
    "simulation",
    "raw_data",
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
    "custom",
]

STEP_KINDS: frozenset[str]
```

`MCStepKind` is the string-level schema used by portable specs. `STEP_KINDS` is the runtime set used for validation.

