---
tags:
    - doc
---

# Difference Transform

```python
diff_step(
    name: str,
    **kwargs: Any,
) -> MCStep
```

`diff_step` applies `np.diff(..., n=order, axis=0)` to a selected input array.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Input source channel. |
| filter_key | `"filter"` | Filter result key when reading filter output. |
| payload_key | `None` | Producer payload key when `source="payload"`. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Drop the initial state row for state inputs. |
| order | `1` | Number of differences. Must be at least `1`. |

The output has `order` fewer rows than the selected input and is stored as a payload.

