---
tags:
    - doc
---

# Rolling Standard Deviation Transform

```python
rolling_std_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
) -> MCStep
```

`rolling_std_step` computes a trailing rolling standard deviation over the time axis. The output shape is `(n - window + 1, k)`.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| window | `10` | Rolling window length. Must be at least `1` and no longer than the selected input. |
| ddof | `0` | Standard-deviation degrees-of-freedom correction. |

The output is stored in the step payload and can be consumed downstream with `source` set to this step name and `field="payload"`.
