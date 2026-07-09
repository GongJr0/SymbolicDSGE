---
tags:
    - doc
---

# Log Difference Transform

```python
log_diff_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
) -> MCStep
```

`log_diff_step` applies first differences to the logged input:

```python
np.diff(np.log(x + offset), axis=0)
```

The output has one fewer row than the selected input.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| offset | `0.0` | Constant added before taking logs. |

The output is stored in the step payload and can be consumed downstream with `source` set to this step name and `field="payload"`.
