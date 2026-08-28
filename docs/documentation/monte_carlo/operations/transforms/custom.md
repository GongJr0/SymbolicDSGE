---
tags:
    - doc
---
# Custom Transform Steps

```python
transform_step(
    name: str,
    n_retain: int = -1,
    func: Callable[..., Any] | NumbaCustomFunc,
    *,
    source: str,
    field: str,
    output_shape: tuple[int, int],
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> MCStep
```

`transform_step` creates a custom per-replication transform operation. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

`func` is wrapped in a [`NumbaCustomFunc`](../../custom_ops.md#numbacustomfunc) if it is not one already, compiled, and called from the native replication loop through a pointer ABI. It must accept exactly two positional arguments and return an integer status code:

```python
func(sample, output) -> int
```

`sample` is the C-contiguous 2D `float64` slice selected by `source`, `field`, `columns`, and the row options. `output` is the C-contiguous 2D `float64` buffer of shape `output_shape` the function fills. Return `0` for success; any other value marks the replication as failed at this step.

The written buffer is the step's payload. Downstream steps read it with `source` set to the step name and `field="payload"`, and it is stacked across replications under the trace key `payload.<name>`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Runtime step name, payload source name, and trace key suffix. |
| n_retain | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |
| func | The transform. A `NumbaCustomFunc`, or a plain function wrapped into one. |
| source | Producer step name supplying the input array. |
| field | Field read from the producer, such as `"observables"`, `"std_innov"`, or `"payload"`. |
| output_shape | `(rows, columns)` of the output buffer. Both dimensions must be non-negative. It fixes the arena size, so it cannot depend on the replication. |
| columns | Column selector applied to the input: an int, a sequence of ints, a slice, or `None` for all columns. |
| burn_in | Number of leading input rows to drop. |
| drop_initial | If `True` and `burn_in` is zero, start at input row `1`. |

???+ note "Bundled custom operations"
    A pipeline written to an `.sdsge` bundle stores the `NumbaCustomFunc` as a side-channel resource, and the portable spec references it by key. Compiled Numba artifacts are never serialized; a loaded transform recompiles on first use.
