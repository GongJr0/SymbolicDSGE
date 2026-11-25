---
tags:
    - doc
---
# SmootherResult

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class SmootherResult[FilterT]
```
Contains the results of a smoother run produced by `Filter.smooth` of any filter. `FilterT` can be any of the `FilterResult` subclasses.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| filter_result | `FilterT` (`FilterResult`) | Results of the forward pass. Equivalent to the return of `Filter.filter` on the same object. |
| x_smooth | `ndarray` | Smoothed state estimates. |
| P_smooth | `ndarray` | Smoothed state covariance. |
| eps_smooth | `ndarray` | Smoothed structural shock estimates. |

???+ info "Filter Results"
    See [`FilterResult`](./FilterResult.md) for the fields of the forward pass results for a given filter implementation.
