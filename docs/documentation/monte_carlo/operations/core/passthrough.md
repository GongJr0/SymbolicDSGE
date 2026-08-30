---
tags:
    - doc
---

# Passthrough Steps

```python
passthrough_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: int | Sequence[str] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> MCStep
```

`passthrough_step` selects a specific source/field/column combination and registers it as a transform payload. It allows the retention of specific subsets of an operation.

As with any other transform, the result will be available in `MCPipelineResult.transform_outputs` and postprocessing steps can refer to it with the key `payload.<name>`. (see [MC Results](../../results.md) for further detail on acess patterns)

???+ note "When to use `passthrough_step`"
    When retaining filter and datagen outputs, the `n_retain` parameter will only choose how many total replication outputs to retain. If you're only interested in a subset of outputs from these steps, `passthrough_step` can let you hold onto significantly more samples before your run is blocked by the memory checks.

    Test and Regression steps build their summary objects and defer computations such as p-values and r2. You can still choose to retain any of the live outputs with a passthrough and compute the required statistics, but it is generally less convenient and the step is indended to reduce the strain of a filter step on a run's memory usage.

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| n_retain | `-1` | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before transformation. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
