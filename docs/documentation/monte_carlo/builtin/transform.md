---
tags:
    - doc
---
# Transform Steps

```python
transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep
```

`transform_step` creates a custom per-replication operation. The callable receives the normalized operation arguments:

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
