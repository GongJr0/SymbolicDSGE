---
tags:
    - doc
---

# Step Catalog

The Monte Carlo step catalog is the single core-side registry for built-in operation metadata. It drives spec compilation, graph validation, and the GUI step palette without depending on the `[ui]` extra.

## `FieldSpec`

```python
@dataclass(frozen=True)
class FieldSpec()
```

One configurable field exposed by a step.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| key | `#!python str` | Runtime kwarg name. |
| label | `#!python str` | Human-readable label. |
| type | `#!python str` | UI/editor field type. |
| default | `#!python Any` | Default value. |
| required | `#!python bool` | Whether the field is required. |
| options | `#!python tuple[str, ...]` | Allowed options for select-like fields. |
| minimum | `#!python float \| None` | Optional numeric lower bound. |
| when | `#!python tuple[str, ...]` | Conditional display metadata. |

__Methods:__

```python
FieldSpec.to_dict() -> dict[str, Any]
```

## `StepDefinition`

```python
@dataclass(frozen=True)
class StepDefinition()
```

Everything the library knows about one built-in Monte Carlo step kind.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| step_type | `#!python str` | Serializable step kind. |
| title | `#!python str` | Display title. |
| default_name | `#!python str` | Default runtime step name. |
| description | `#!python str` | Short user-facing description. |
| op_role | `#!python Literal["datagen", "filter", "transform", "terminal"]` | Graph role. |
| factory | `#!python Callable[..., MCStep]` | Step factory used by `build(...)`. |
| fields | `#!python tuple[FieldSpec, ...]` | Declared configurable fields. |
| compile_params | `#!python Callable \| None` | Optional parameter normalization hook. |

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| is_terminal | `#!python bool` | `True` for test and regression terminal steps. |
| is_transform | `#!python bool` | `True` for transform steps. |
| category | `#!python str` | GUI grouping: `"core"`, `"tests"`, `"regressions"`, or `"transforms"`. |

__Methods:__

```python
StepDefinition.catalog_entry() -> dict[str, Any]
StepDefinition.build(name: str, params: dict[str, Any], dgp: SolvedModel | None) -> MCStep
```

## Registry Objects

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `STEP_CATALOG` | `#!python dict[str, StepDefinition]` | Built-in step definitions keyed by `step_type`. |
| `DATAGEN_STEP_TYPES` | `#!python frozenset[str]` | Datagen kinds: simulation and raw data. |
| `TRANSFORM_STEP_TYPES` | `#!python frozenset[str]` | Built-in transform kinds. |
| `TERMINAL_STEP_TYPES` | `#!python frozenset[str]` | Test and regression terminal kinds. |
| `INPUT_SOURCES` | `#!python list[str]` | Source channels diagnostic/regression steps may read. |
| `FILTER_SOURCES` | `#!python set[str]` | Source channels produced by the filter step. |

```python
catalog_payload() -> dict[str, Any]
```

Return a JSON-like payload of catalog entries for UI consumers.

???+ note "Custom steps"
    `STEP_CATALOG` only describes built-in operation kinds. Bundle-safe custom operations use the reserved `custom` step kind and are restored through bundle resources rather than through catalog dispatch.

