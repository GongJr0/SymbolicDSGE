---
tags:
    - doc
---

# Jarque-Bera Tests

```python
jarque_bera_test_step(
    name: str,
    *,
    source: Literal[
        "states",
        "observables",
        "x_pred",
        "x_filt",
        "y_pred",
        "y_filt",
        "innov",
        "std_innov",
        "payload",
    ],
    filter_key: str = "filter",
    payload_key: str | None = None,
    column: Sequence[int] | int | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`jarque_bera_test_step` wraps `run_jarque_bera_test(...)`. It selects a 1D array from generated data, a `FilterResult`, or a named payload, then runs the Jarque-Bera test for normality from the sample skewness and excess kurtosis.

???+ info "Reference Distribution"
    The statistic is compared against a $\chi^2(2)$ distribution asymptotically. For small samples the test uses a finite-sample critical-value lookup keyed on the sample size, so the reported degrees of freedom carry the sample size rather than a fixed value.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
