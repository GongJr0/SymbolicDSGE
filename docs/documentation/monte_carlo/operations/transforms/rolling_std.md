---
tags:
    - doc
---

# Rolling Standard Deviation Transform

```python
rolling_std_step(
    name: str,
    **kwargs: Any,
) -> MCStep
```

`rolling_std_step` computes a trailing rolling standard deviation over the time axis. The output shape is `(n - window + 1, k)`.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Input source channel. |
| filter_key | `"filter"` | Filter result key when reading filter output. |
| payload_key | `None` | Producer payload key when `source="payload"`. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Drop the initial state row for state inputs. |
| window | `10` | Rolling window length. Must be at least `1` and no longer than the selected input. |
| ddof | `0` | Standard-deviation degrees-of-freedom correction. |

The output is stored in the step payload and can be consumed downstream with `source="payload"`.

