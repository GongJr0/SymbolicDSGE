---
tags:
    - doc
---

# Standardize Transform

```python
standardize_step(
    name: str,
    **kwargs: Any,
) -> MCStep
```

`standardize_step` applies per-column z-scores to a selected input array:

```python
(x - x.mean(axis=0)) / x.std(axis=0, ddof=ddof)
```

Columns with zero standard deviation are returned as zeros.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Input source channel. |
| filter_key | `"filter"` | Filter result key when reading filter output. |
| payload_key | `None` | Producer payload key when `source="payload"`. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Drop the initial state row for state inputs. |
| ddof | `0` | Standard-deviation degrees-of-freedom correction. |

The output is stored in the step payload and can be consumed downstream with `source="payload"`.

