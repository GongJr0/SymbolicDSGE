---
tags:
    - doc
---
# Raw Model Data

```python
raw_model_data_step(
    name: str = "datagen",
    *,
    states: ndarray | None = None,
    observables: ndarray | None = None,
    raw: Mapping[str, ndarray] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep
```

`raw_model_data_step` wraps `raw_model_data_datagen(...)`. It does not require a DGP model.

__Accepted Shapes:__

| __Field__ | __Accepted Shapes__ | __Description__ |
|:----------|:--------------------:|----------------:|
| states | `(T, n_state)` or `(n_rep, T, n_state)` | A 2D array feeds every replication. A 3D array selects `arr[rep_idx]`, so each replication gets its own slice. |
| observables | `(T,)`, `(T, n_obs)`, or `(n_rep, T, n_obs)` | A 1D observable path is reshaped to `(T, 1)`. A 2D array feeds every replication; a 3D array selects `arr[rep_idx]`. |
| raw values | `(T,)`, `(T, n)`, or `(n_rep, T, n)` | Extra raw arrays follow the same convention as observables. |

???+ note "State and observable payloads"
    `raw_model_data_step` accepts state data, observable data, or both. A reference filter step requires observables, while downstream steps use the raw model data step name as `source` and `field="states"` or `field="observables"`.
