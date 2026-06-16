---
tags:
    - doc
---

# Pipeline Graph

`PipelineGraph` is the resolved dependency view of a live `MCPipeline`. A pipeline is authored as an ordered sequence of `MCStep` objects, but each step also names the upstream data it consumes. The graph resolves those references once so serializers and graph consumers do not re-infer wiring from kwargs.

## `InputEdge`

```python
@dataclass(frozen=True)
class InputEdge()
```

One resolved input dependency of a step.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| role | `#!python str` | Consumer leg such as `"source"`, `"residual_source"`, `"y_source"`, or `"X_source"`. |
| producer | `#!python str` | Upstream step name. |
| channel | `#!python str` | Channel read from the producer, such as `"observables"`, `"std_innov"`, or `"payload"`. |

## `PipelineNode`

```python
@dataclass(frozen=True)
class PipelineNode()
```

A live `MCStep` plus its resolved graph location.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| step | `#!python MCStep` | Underlying runtime step. |
| inputs | `#!python tuple[InputEdge, ...]` | Resolved upstream dependencies. |
| children | `#!python tuple[str, ...]` | Downstream step names. |

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Step name. |
| step_type | `#!python str \| None` | Serializable step kind when known. |
| is_root | `#!python bool` | `True` when the node has no inputs. |
| is_leaf | `#!python bool` | `True` when no other node consumes this node. |
| parents | `#!python tuple[str, ...]` | De-duplicated producer names in leg order. |
| primary_parent | `#!python str \| None` | Structural parent used when serializing the graph spec. |

## `PipelineGraph`

```python
class PipelineGraph()
```

The dependency DAG of a pipeline, keyed by step name.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| nodes | `#!python dict[str, PipelineNode]` | Nodes keyed by step name. |
| root | `#!python str` | Root datagen step name. |
| order | `#!python tuple[str, ...]` | Authored execution order, validated as a topological order. |

__Methods:__

```python
PipelineGraph.from_steps(steps: tuple[MCStep, ...]) -> PipelineGraph  # @classmethod
PipelineGraph.edges() -> list[tuple[str, str]]
```

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| leaves | `#!python tuple[PipelineNode, ...]` | Nodes with no downstream consumers. |

???+ warning "Custom source channels"
    Graph resolution understands the built-in source conventions. Custom operations can still run in-process, but bundle-safe graph serialization requires the step to declare a supported `step_type` and source payload convention.

