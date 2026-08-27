---
tags:
    - doc
---

# Jarque-Bera Tests

```python
jarque_bera_test_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    column: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`jarque_bera_test_step` runs as a native per-replication kernel. It selects one series from a producer step and runs the Jarque-Bera test for normality from the sample skewness and excess kurtosis.

???+ info "Reference Distribution"
    The statistic is compared against a $\chi^2(2)$ distribution asymptotically. For small samples the test uses a finite-sample critical-value lookup keyed on the sample size, so the reported degrees of freedom carry the sample size rather than a fixed value.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| column | `None` | Column selector. It must resolve to one series. |
| burn_in | `0` | Rows dropped before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| alpha | `0.05` | Test size. |
| n_retain | `-1` | Number of replications whose output is retained for this step. `-1` retains all replications. It may not exceed `n_rep`. |
