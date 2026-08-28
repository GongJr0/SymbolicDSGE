---
tags:
    - doc
---

# Ljung-Box Tests

```python
ljung_box_test_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    column: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> MCStep
```

`ljung_box_test_step` runs as a native per-replication kernel. It selects one series from a producer step and runs the Ljung-Box test for autocorrelation up to the specified number of lags.

???+ warning "Lag Handling"
    If `lags` is greater than the selected sample length, the test runs up to the maximum possible lag. There is no automatic lag selection heuristic.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| column | `None` | Column selector. It must resolve to one series. |
| burn_in | `0` | Rows dropped before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| lags | `10` | Maximum autocorrelation lag. |
| alpha | `0.05` | Test size. |
| n_retain | `-1` | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |
