---
tags:
    - doc
---

# Log Difference Transform

```python
log_diff_step(
    name: str,
    **kwargs: Any,
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
| source | required | Input source channel. |
| filter_key | `"filter"` | Filter result key when reading filter output. |
| payload_key | `None` | Producer payload key when `source="payload"`. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Drop the initial state row for state inputs. |
| offset | `0.0` | Constant added before taking logs. |

The output is stored in the step payload and can be consumed downstream with `source="payload"`.

