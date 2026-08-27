---
tags:
    - doc
---

# Difference Transform

```python
diff_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    order: int = 1,
) -> MCStep
```

`diff_step` applies `np.diff(..., n=order, axis=0)` to a selected input array.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| order | `1` | Number of differences. Must be at least `1`. |
| n_retain | `-1` | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |

The output has `order` fewer rows than the selected input and is stored as a payload.
