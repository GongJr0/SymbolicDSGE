---
tags:
    - doc
---
# Custom Transform Steps

```python
transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep
```

`transform_step` creates a custom per-replication transform operation. It lives under `SymbolicDSGE.monte_carlo.operations.transforms`.

The callable receives the normalized operation arguments:

```python
func(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    **kwargs,
) -> Any
```

If the callable returns an `MCData` object, the pipeline replaces `context.data` with that return value. The return value is stored in `context.payloads[store_key or name]`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Runtime step name and default payload key. |
| func | Callable to execute for each replication. |
| store_key | Optional override for the payload key. |
| kwargs | Extra keyword arguments forwarded to `func`. |

???+ note "Bundled custom operations"
    In-process pipelines may use any callable. A pipeline written to an `.sdsge` bundle requires the callable to be a [`NumpyCustomFunc`](../../custom_ops.md#numpycustomfunc), usually via the [`numpy_operation`](../../custom_ops.md#numpy_operation) decorator. The bundle stores the callable as a side-channel resource and the portable spec references it by key.
