---
tags:
    - doc
---
# Raw Data

```python
raw_data_step(
    name: str = "datagen",
    *,
    states: ndarray | None = None,
    observables: ndarray | None = None,
    n_exog: int = -1,
    raw: Mapping[str, ndarray] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep
```

`raw_data_step` wraps `raw_data_datagen(...)`. It does not require a DGP model.

__Accepted Shapes:__

| __Field__ | __Accepted Shapes__ | __Description__ |
|:----------|:--------------------:|----------------:|
| states | `(T, n_state)` or `(n_rep, T, n_state)` | A 2D array is reused for every replication. A 3D array selects `arr[rep_idx]`. |
| observables | `(T,)`, `(T, n_obs)`, or `(n_rep, T, n_obs)` | A 1D observable path is reshaped to `(T, 1)`. A 2D array is reused for every replication. |
| raw values | `(T,)`, `(T, n)`, or `(n_rep, T, n)` | Extra raw arrays are selected with the same convention as observables. |

???+ note "State-Only and Observable-Only Runs"
    `raw_data_step` accepts state-only and observable-only payloads. A reference filter step requires observables, while tests using `source="states"` require states.
