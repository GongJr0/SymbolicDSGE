---
tags:
    - doc
---
# Raw Model Data

```python
raw_model_data_step(
    name: str = "datagen",
    *,
    states: ndarray | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observables: ndarray | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep
```

`raw_model_data_step` feeds pre-computed arrays into the pipeline instead of simulating them. It does not require a DGP model. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

__Accepted Shapes:__

| __Field__ | __Accepted Shapes__ | __Description__ |
|:----------|:--------------------:|----------------:|
| states | `(T,)`, `(T, n_state)`, or `(n_rep, T, n_state)` | A 1D path is treated as `(T, 1)`. A 2D array feeds every replication. A 3D array selects `arr[rep_idx]`, so each replication gets its own slice. |
| observables | `(T,)`, `(T, n_obs)`, or `(n_rep, T, n_obs)` | Same convention as `states`. |

The array is bound to the native run once, before the loop, and the per-replication slice is read directly out of it.

???+ note "State and observable data"
    `raw_model_data_step` accepts state data, observable data, or both. A reference filter step requires observables, while downstream steps use the raw model data step name as `source` and `field="states"` or `field="observables"`.
