---
tags:
    - doc
---
# Payload Steps

```python
add_payload_step(
    name: str,
    n_retain: int = -1,
    payload: ndarray | Sequence[float] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
) -> MCStep
```

`add_payload_step` injects a pre-computed array into the pipeline as a transform payload, without reading from an upstream producer. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

Use it to supply an exogenous series that tests or regressions need alongside the generated data, for example a regressor the model does not produce.

__Accepted Shapes:__

| __Shape__ | __Behavior__ |
|:----------|-------------:|
| `(T,)` | Treated as `(T, 1)` and fed to every replication. |
| `(T, p)` | Fed to every replication. |
| `(n_rep, T, p)` | Replication `rep_idx` receives `payload[rep_idx]`. |

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Runtime step name. Downstream steps use it as `source` with `field="payload"`. |
| payload | The array to inject, cast to `float64`. |
| n_retain | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |

The step is a `TRANSFORM`, so its output is stacked across replications under the trace key `payload.<name>` like any other transform.
