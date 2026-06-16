---
tags:
    - doc
---

# Log Transform

```python
log_step(
    name: str,
    **kwargs: Any,
) -> MCStep
```

`log_step` applies `log(x + offset)` elementwise to a selected input array.

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

