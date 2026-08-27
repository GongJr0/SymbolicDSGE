---
tags:
    - doc
---

# Standardize Transform

```python
standardize_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    ddof: int = 0,
) -> MCStep
```

`standardize_step` applies per-column z-scores to a selected producer field:

```python
(x - x.mean(axis=0)) / x.std(axis=0, ddof=ddof)
```

Columns with zero standard deviation are returned as zeros.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| ddof | `0` | Standard-deviation degrees-of-freedom correction. |
| n_retain | `-1` | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |

The output is stored in the step payload and can be consumed downstream with `source` set to this step name and `field="payload"`.
